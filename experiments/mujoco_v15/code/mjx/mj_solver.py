"""MuJoCo 原生机械臂求解器 (替代 ik_solver.RobotIKSolver, 无 PyBullet 依赖)

提供与 RobotIKSolver 相同的对外 API, 供 app.py 可视化与 ik_3d/ik_6d
RL 求解器使用:

- forward_kinematics(q)        末端位姿 (ee site, 与 PyBullet EE 帧 1e-7 对齐)
- get_all_link_transforms(q)   各 link 世界变换 × URDF visual origin (前端渲染)
- solve_ik(...)                4 种位置级 IK (dls/jacobian/transpose/pseudoinverse)
                               雅可比来自 mujoco.mj_jac (替代 calculateJacobian)
- solve_path(...)              插值 + 逐点 IK
- get_robot_info()             前端元信息

注: 'dls' 方法原为 PyBullet 内置求解器, 现用与 'jacobian' 相同的
自实现 DLS 公式 (Δq = Jᵀ(JJᵀ+λ²I)⁻¹e), 数值行为一致。
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from robot_configs import get_robot_config

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED_XML = os.path.join(_BASE, 'mjx', 'assets', 'panda_fixed.xml')

# hand 帧 = link7 ∘ (z+0.107, 绕z -45°), 与 MJCF ee site / PyBullet
# getLinkState(panda_hand) 一致
HAND_POS = np.array([0.0, 0.0, 0.107])
HAND_QUAT = np.array([0.0, 0.0, -0.3826834323650898, 0.9238795325112867])  # xyzw, rz(-45°)


def mat_to_quat_xyzw(R):
    """3x3 旋转矩阵 -> 四元数 [x,y,z,w]"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-12)


def quat_multiply_xyzw(q1, q2):
    """四元数乘法 q1 ⊗ q2, [x,y,z,w]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_rotate_xyzw(q, v):
    """四元数旋转向量, q=[x,y,z,w]"""
    xyz = np.asarray(q[:3])
    w = q[3]
    t = 2.0 * np.cross(xyz, v)
    return np.asarray(v) + w * t + np.cross(xyz, t)


class MJRobotSolver:
    """通用机械臂IK求解器 (MuJoCo 原生, 接口对齐 RobotIKSolver)"""

    def __init__(self, robot_id='panda', robot_config=None):
        if robot_config is None:
            robot_config = get_robot_config(robot_id)
        self.config = robot_config
        self.robot_id_str = robot_id

        self.model = mujoco.MjModel.from_xml_path(FIXED_XML)
        self.data = mujoco.MjData(self.model)

        self.n_ctrl = 7
        self.controlled_joints = list(range(self.n_ctrl))
        self.joint_limits = [(float(self.model.jnt_range[i][0]),
                              float(self.model.jnt_range[i][1]))
                             for i in range(self.n_ctrl)]
        self.ndof = self.model.nv            # 含 2 个 finger 棱柱关节

        self.ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
        self.link_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                              f'panda_link{i}') for i in range(1, 8)]
        self.link7_body = self.link_body_ids[-1]

        # URDF visual origins (mesh 偏移, 与 PyBullet 版同一解析逻辑)
        self.visual_origins = self._parse_visual_origins(
            robot_config['urdf_path'])

        home = robot_config.get('home_position', [0.0] * self.n_ctrl)
        self.joint_angles = list(home[:self.n_ctrl])
        self._apply_joint_angles()

        # IK求解参数 (与 RobotIKSolver 一致)
        self.max_iterations = 200
        self.tolerance = 1e-4
        self.damping = 0.05

    # ==================== URDF 解析 (visual origins) ====================

    def _parse_visual_origins(self, urdf_path):
        """从URDF XML解析每个link的visual origin变换

        返回 dict: link_name -> {'position': [x,y,z], 'orientation': [x,y,z,w]}
        """
        origins = {}
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for link in root.findall('link'):
                link_name = link.get('name')
                visual = link.find('visual')
                if visual is not None:
                    origin = visual.find('origin')
                    if origin is not None:
                        xyz = list(map(float, origin.get('xyz', '0 0 0').split()))
                        rpy = list(map(float, origin.get('rpy', '0 0 0').split()))
                        orn = self._rpy_to_quaternion(rpy)
                        origins[link_name] = {'position': xyz, 'orientation': orn}
                    else:
                        origins[link_name] = {
                            'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]}
        except Exception as e:
            print(f"Warning: 解析URDF visual origins失败: {e}")
        return origins

    @staticmethod
    def _rpy_to_quaternion(rpy):
        """RPY (roll, pitch, yaw) -> 四元数 [x,y,z,w], ZYX 顺序 (URDF默认)"""
        r, pitch, y = rpy
        cr, sr = np.cos(r/2), np.sin(r/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(y/2), np.sin(y/2)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y_val = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return [float(x), float(y_val), float(z), float(w)]

    @staticmethod
    def _quat_to_rotation_matrix(q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    @staticmethod
    def _multiply_transforms(pos1, orn1, pos2, orn2):
        """组合两个变换: T1 * T2 (pos [x,y,z], orn [x,y,z,w])"""
        q_combined = quat_multiply_xyzw(np.asarray(orn1), np.asarray(orn2))
        R = MJRobotSolver._quat_to_rotation_matrix(np.asarray(orn1))
        combined_pos = (np.asarray(pos1) + R @ np.asarray(pos2)).tolist()
        return combined_pos, q_combined.tolist()

    # ==================== 关节状态 ====================

    def _apply_joint_angles(self):
        self._set_q(self.joint_angles)

    def _apply_q(self, q):
        self._set_q(q)

    def _set_q(self, q):
        """将关节角写入 MuJoCo 并前向计算"""
        self.data.qpos[:self.n_ctrl] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _get_ee_pos(self):
        return np.array(self.data.site_xpos[self.ee_site], dtype=np.float32)

    def _get_ee_pose(self):
        """当前构型末端位姿 -> (pos float32, quat_xyzw float32)"""
        pos = np.array(self.data.site_xpos[self.ee_site], dtype=np.float32)
        R = self.data.site_xmat[self.ee_site].reshape(3, 3)
        return pos, mat_to_quat_xyzw(R).astype(np.float32)

    def _clip_joints(self, q):
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            q[i] = np.clip(q[i], lower, upper)

    # ==================== 雅可比矩阵 (mj_jac) ====================

    def _get_jacobian(self, q, mode='pybullet'):
        """末端位置 3x7 雅可比 (mode 参数仅为 API 兼容, 均用 mj_jac)"""
        self._apply_q(q)
        jacp = np.zeros((3, self.ndof))
        jacr = np.zeros((3, self.ndof))
        point = self.data.site_xpos[self.ee_site].copy()
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point,
                      self.link7_body)
        return jacp[:, :self.n_ctrl]

    # ==================== 正运动学 ====================

    def forward_kinematics(self, joint_angles=None):
        if joint_angles is not None:
            self._set_q(joint_angles)
        else:
            self._apply_q(self.joint_angles)
        pos, quat = self._get_ee_pose()
        return {'position': pos.tolist(), 'orientation': quat.tolist()}

    def get_all_link_transforms(self, joint_angles=None):
        """各 link 世界变换 × visual origin (与 PyBullet 版同构)

        返回 panda_link0..7 + panda_hand 共 9 项。
        """
        if joint_angles is not None:
            self._set_q(joint_angles)
        else:
            self._apply_q(self.joint_angles)
        d = self.data

        # base_link (root)
        base_name = 'panda_link0' if self.robot_id_str == 'panda' else 'base_link'
        base_vo = self.visual_origins.get(base_name)
        if base_vo:
            base_pos, base_orn = self._multiply_transforms(
                [0, 0, 0], [0, 0, 0, 1], base_vo['position'], base_vo['orientation'])
        else:
            base_pos, base_orn = [0, 0, 0], [0, 0, 0, 1]
        transforms = [{'name': base_name, 'position': base_pos,
                       'orientation': base_orn}]

        link7_pose = None
        for i, bid in enumerate(self.link_body_ids):
            link_name = f'panda_link{i + 1}'
            link_pos = np.array(d.xpos[bid])
            link_orn = mat_to_quat_xyzw(d.xmat[bid].reshape(3, 3))
            if i == 6:
                link7_pose = (link_pos, link_orn)
            vo = self.visual_origins.get(link_name)
            if vo:
                cpos, corn = self._multiply_transforms(
                    link_pos, link_orn, vo['position'], vo['orientation'])
                transforms.append(
                    {'name': link_name, 'position': cpos, 'orientation': corn})
            else:
                transforms.append({'name': link_name,
                                   'position': link_pos.tolist(),
                                   'orientation': link_orn.tolist()})

        # panda_hand = link7 ∘ (z+0.107, rot_z(-45°)) (即 MJCF ee site 帧,
        # 与 PyBullet getLinkState(panda_hand) 同构)
        l7pos, l7orn = link7_pose
        hand_pos, hand_orn = self._multiply_transforms(
            l7pos, l7orn, HAND_POS, HAND_QUAT)
        vo = self.visual_origins.get('panda_hand')
        if vo:
            hand_pos, hand_orn = self._multiply_transforms(
                hand_pos, hand_orn, vo['position'], vo['orientation'])
        transforms.append({'name': 'panda_hand', 'position': hand_pos,
                           'orientation': hand_orn})
        return transforms

    def get_robot_info(self):
        """返回机械臂元信息(供前端使用)"""
        joints = []
        for i in range(self.n_ctrl):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            lo, hi = self.joint_limits[i]
            joints.append({
                'index': i,
                'name': name,
                'type': 'revolute',
                'lower_limit': lo,
                'upper_limit': hi,
            })
        return {
            'robot_id': self.robot_id_str,
            'name': self.config['name'],
            'num_axes': self.n_ctrl,
            'joints': joints,
            'home_position': self.joint_angles,
            'joint_limits': self.joint_limits,
            'link_meshes': self.config.get('link_meshes'),
            'reach_bounds': self.config.get('reach_bounds', {}),
            'presets': self.config.get('presets', []),
        }

    # ==================== 公共工具 ====================

    def _verify_result(self, q, target_pos):
        self._apply_q(q)
        actual_pos = self._get_ee_pos()
        final_error = float(np.linalg.norm(actual_pos - np.asarray(target_pos)))
        success = bool(final_error < 0.01)

        result = [float(x) for x in np.asarray(q).tolist()]
        if success:
            self.joint_angles = result

        return {
            'success': success,
            'joint_angles': result,
            'actual_position': actual_pos.tolist(),
            'target_position': list(target_pos),
            'position_error': float(final_error),
        }

    # ==================== IK算法 (位置级, 与 RobotIKSolver 同公式) ====================

    def solve_ik_dls(self, target_pos, target_orn=None, jac_mode='pybullet'):
        """DLS (原 PyBullet 内置, 现用自实现 DLS, 公式同 jacobian 方法)"""
        return self.solve_ik_jacobian(target_pos, target_orn, jac_mode=jac_mode)

    def solve_ik_jacobian(self, target_pos, target_orn=None, jac_mode='pybullet'):
        """阻尼最小二乘法 (DLS): Δq = J^T (J J^T + λ²I)^{-1} Δx"""
        q = np.array(self.joint_angles, dtype=float)
        target = np.array(target_pos, dtype=float)

        for iteration in range(self.max_iterations):
            self._apply_q(q)
            current_pos = self._get_ee_pos()
            error = target - current_pos

            if np.linalg.norm(error) < self.tolerance:
                break

            J = self._get_jacobian(q, mode=jac_mode)
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(damped, error)

            q = q + dq
            self._clip_joints(q)

        result = self._verify_result(q, target_pos)
        result['iterations'] = iteration + 1
        return result

    def solve_ik_transpose(self, target_pos, target_orn=None, jac_mode='pybullet'):
        """雅可比转置法: Δq = α · J^T · Δx"""
        q = np.array(self.joint_angles, dtype=float)
        target = np.array(target_pos, dtype=float)

        for iteration in range(self.max_iterations):
            self._apply_q(q)
            current_pos = self._get_ee_pos()
            error = target - current_pos

            if np.linalg.norm(error) < self.tolerance:
                break

            J = self._get_jacobian(q, mode=jac_mode)
            JT_error = J.T @ error
            JJT_error = J @ JT_error

            numerator = float(JJT_error @ JJT_error)
            denominator = float(JJT_error @ (J @ (J.T @ JJT_error)))
            alpha = numerator / denominator if denominator > 1e-12 else 0.5
            alpha = min(alpha, 2.0)

            dq = alpha * JT_error
            q = q + dq
            self._clip_joints(q)

        result = self._verify_result(q, target_pos)
        result['iterations'] = iteration + 1
        return result

    def solve_ik_pseudoinverse(self, target_pos, target_orn=None, jac_mode='pybullet'):
        """雅可比伪逆法: Δq = J^T (J J^T)^{-1} · Δx"""
        q = np.array(self.joint_angles, dtype=float)
        target = np.array(target_pos, dtype=float)

        for iteration in range(self.max_iterations):
            self._apply_q(q)
            current_pos = self._get_ee_pos()
            error = target - current_pos

            if np.linalg.norm(error) < self.tolerance:
                break

            J = self._get_jacobian(q, mode=jac_mode)
            JJT = J @ J.T
            reg = 1e-6
            JJT += reg * np.eye(3)

            try:
                dq = J.T @ np.linalg.solve(JJT, error)
            except np.linalg.LinAlgError:
                dq = 0.1 * J.T @ error

            dq_norm = np.linalg.norm(dq)
            if dq_norm > 0.5:
                dq = dq * 0.5 / dq_norm

            q = q + dq
            self._clip_joints(q)

        result = self._verify_result(q, target_pos)
        result['iterations'] = iteration + 1
        return result

    # ==================== 统一接口 ====================

    def solve_ik(self, target_pos, target_orn=None, method='jacobian', jac_mode='pybullet'):
        dispatch = {
            'dls':           self.solve_ik_dls,
            'jacobian':      self.solve_ik_jacobian,
            'transpose':     self.solve_ik_transpose,
            'pseudoinverse': self.solve_ik_pseudoinverse,
        }
        solver_fn = dispatch.get(method, self.solve_ik_jacobian)
        return solver_fn(target_pos, target_orn, jac_mode=jac_mode)

    # ==================== 路径规划 ====================

    def interpolate_path(self, start_pos, end_pos, steps=30):
        path = []
        for i in range(steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)
            point = [start_pos[j] + t * (end_pos[j] - start_pos[j]) for j in range(3)]
            path.append(point)
        return path

    def solve_path(self, start_angles, end_pos, steps=30, method='jacobian', jac_mode='pybullet'):
        start_fk = self.forward_kinematics(start_angles)
        path_points = self.interpolate_path(start_fk['position'], end_pos, steps)

        path_angles = [list(start_angles)]
        current_angles = list(start_angles)

        for point in path_points[1:]:
            self.joint_angles = list(current_angles)
            result = self.solve_ik(point, method=method, jac_mode=jac_mode)
            if result['success']:
                current_angles = result['joint_angles']
            path_angles.append(list(current_angles))

        return path_angles

    def close(self):
        pass
