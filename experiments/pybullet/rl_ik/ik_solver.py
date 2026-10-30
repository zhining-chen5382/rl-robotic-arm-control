"""
7轴/6轴机械臂IK求解器 (通用版)
支持多种机械臂，通过配置驱动。

包含多种IK算法用于学习对比:
  1. dls           - pybullet内置阻尼最小二乘法
  2. jacobian      - 手动实现阻尼最小二乘法 (DLS)
  3. transpose     - 雅可比转置法
  4. pseudoinverse - 雅可比伪逆法

雅可比矩阵可选来源:
  - pybullet  - 调用pybullet的calculateJacobian (精确，但黑盒)
  - manual    - 自己用几何公式计算: J_i = z_i × (p_ee - p_i)
"""
import numpy as np
import pybullet as p
import pybullet_data
import os
import xml.etree.ElementTree as ET
from robot_configs import get_robot_config


class RobotIKSolver:
    """通用机械臂IK求解器"""

    def __init__(self, robot_id='panda', robot_config=None):
        """初始化IK求解器
        
        Args:
            robot_id: 机械臂ID (如 'panda', 'ur5')
            robot_config: 自定义配置dict, 如果提供则忽略robot_id
        """
        if robot_config is None:
            robot_config = get_robot_config(robot_id)
        
        self.config = robot_config
        self.robot_id_str = robot_id

        # 创建pybullet DIRECT模式
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 加载机器人模型
        urdf_path = robot_config['urdf_path']
        self.robot = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        # 解析URDF获取visual origins (mesh偏移量)
        self.visual_origins = self._parse_visual_origins(urdf_path)

        # 自动检测关节信息
        self.num_joints = p.getNumJoints(self.robot)
        self.controlled_joints = []
        self.joint_limits = []

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.controlled_joints.append(i)
                self.joint_limits.append((info[8], info[9]))  # (lower, upper)

        self.n_ctrl = len(self.controlled_joints)

        # 自动检测末端link: 最后一个可控joint的child link
        last_ctrl_joint = self.controlled_joints[-1]
        # 在pybullet中, joint i 的child link index就是 i+1 (如果后面有fixed joint继续往后找)
        # 但更通用的方法: 找最后一个可控joint之后, 沿fixed joint链走到末端
        self.ee_link_index = self._find_ee_link(last_ctrl_joint)

        # 计算总DoF
        self.ndof = sum(1 for i in range(self.num_joints)
                        if p.getJointInfo(self.robot, i)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC))

        # 设置初始关节角度
        home = robot_config.get('home_position', [0.0] * self.n_ctrl)
        self.joint_angles = list(home[:self.n_ctrl])
        self._apply_joint_angles()

        # IK求解参数
        self.max_iterations = 200
        self.tolerance = 1e-4
        self.damping = 0.05

    def _find_ee_link(self, last_controlled_joint):
        """找到最后一个可控关节之后、沿fixed joint链到末端的link索引
        
        在pybullet中, getLinkState(robot, i) 返回joint i的child link状态,
        所以需要返回一个有效的joint索引 (0 ~ num_joints-1).
        """
        ee_idx = last_controlled_joint
        
        # 沿后续fixed joint链走到末端
        for i in range(last_controlled_joint + 1, self.num_joints):
            info = p.getJointInfo(self.robot, i)
            if info[2] == p.JOINT_FIXED:
                ee_idx = i  # 更新到当前fixed joint的索引
            else:
                break
        
        return ee_idx

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
                        rpy_str = origin.get('rpy', '0 0 0')
                        rpy = list(map(float, rpy_str.split()))
                        # rpy -> quaternion
                        orn = self._rpy_to_quaternion(rpy)
                        origins[link_name] = {
                            'position': xyz,
                            'orientation': orn,
                        }
                    else:
                        origins[link_name] = {
                            'position': [0, 0, 0],
                            'orientation': [0, 0, 0, 1],
                        }
        except Exception as e:
            print(f"Warning: 解析URDF visual origins失败: {e}")
        return origins

    @staticmethod
    def _rpy_to_quaternion(rpy):
        """将RPY (roll, pitch, yaw) 转换为四元数 [x,y,z,w]"""
        r, pitch, y = rpy
        cr, sr = np.cos(r/2), np.sin(r/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(y/2), np.sin(y/2)
        # ZYX 顺序 (URDF默认)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y_val = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return [float(x), float(y_val), float(z), float(w)]

    @staticmethod
    def _multiply_transforms(pos1, orn1, pos2, orn2):
        """组合两个变换: T1 * T2
        
        pos, orn 均为 [x,y,z] 和 [x,y,z,w] 格式
        返回 (combined_pos, combined_orn)
        """
        q1 = np.array(orn1)  # [x,y,z,w]
        q2 = np.array(orn2)
        
        # 四元数乘法 q1 * q2
        x1,y1,z1,w1 = q1
        x2,y2,z2,w2 = q2
        q_combined = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])
        
        # 旋转 pos2 到 q1 的坐标系
        R = RobotIKSolver._quat_to_rotation_matrix_static(q1)
        pos2_rotated = R @ np.array(pos2)
        combined_pos = (np.array(pos1) + pos2_rotated).tolist()
        
        return combined_pos, q_combined.tolist()

    @staticmethod
    def _quat_to_rotation_matrix_static(q):
        """四元数 [x,y,z,w] -> 3x3 旋转矩阵"""
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    def _apply_joint_angles(self):
        for i, joint_idx in enumerate(self.controlled_joints):
            p.resetJointState(self.robot, joint_idx, self.joint_angles[i])

    def _apply_q(self, q):
        """将关节角度数组应用到pybullet并更新碰撞检测"""
        for i, joint_idx in enumerate(self.controlled_joints):
            p.resetJointState(self.robot, joint_idx, q[i])
        p.performCollisionDetection()

    def _get_ee_pos(self):
        state = p.getLinkState(self.robot, self.ee_link_index)
        return np.array(state[4])

    # ==================== 雅可比矩阵计算 ====================

    def _get_jacobian_pybullet(self, q):
        """使用pybullet内置的calculateJacobian"""
        all_pos = [0.0] * self.ndof
        all_vel = [0.0] * self.ndof
        all_acc = [0.0] * self.ndof
        for i in range(self.n_ctrl):
            all_pos[i] = q[i]

        jac_linear, _ = p.calculateJacobian(
            self.robot, self.ee_link_index,
            localPosition=[0, 0, 0],
            objPositions=all_pos,
            objVelocities=all_vel,
            objAccelerations=all_acc
        )
        return np.array(jac_linear)[:, :self.n_ctrl]

    def _get_jacobian_manual(self, q):
        """自己计算几何雅可比矩阵
        
        对于旋转关节: J_i = z_i × (p_ee - p_i)
        z_i: 关节i旋转轴(世界坐标系), p_i: 关节i位置, p_ee: 末端位置
        """
        self._apply_q(q)
        ee_pos = self._get_ee_pos()

        J = np.zeros((3, self.n_ctrl))

        for i, joint_idx in enumerate(self.controlled_joints):
            info = p.getJointInfo(self.robot, joint_idx)
            joint_axis_local = np.array(info[13][:3])

            state = p.getLinkState(self.robot, joint_idx)
            link_pos = np.array(state[0])
            link_orn = np.array(state[1])

            R = self._quat_to_rotation_matrix(link_orn)
            z_i = R @ joint_axis_local
            z_i = z_i / (np.linalg.norm(z_i) + 1e-12)

            r = ee_pos - link_pos
            J[:, i] = np.cross(z_i, r)

        return J

    def _get_jacobian(self, q, mode='pybullet'):
        if mode == 'manual':
            return self._get_jacobian_manual(q)
        else:
            return self._get_jacobian_pybullet(q)

    def _quat_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    # ==================== 公共工具 ====================

    def _verify_result(self, q, target_pos):
        self._apply_q(q)
        actual_pos = self._get_ee_pos()
        final_error = float(np.linalg.norm(actual_pos - np.array(target_pos)))
        success = bool(final_error < 0.01)

        result = [float(x) for x in q.tolist()]
        if success:
            self.joint_angles = result

        return {
            'success': success,
            'joint_angles': result,
            'actual_position': actual_pos.tolist(),
            'target_position': list(target_pos),
            'position_error': float(final_error),
        }

    def _clip_joints(self, q):
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            q[i] = np.clip(q[i], lower, upper)

    # ==================== 正运动学 ====================

    def forward_kinematics(self, joint_angles=None):
        if joint_angles is not None:
            for i, joint_idx in enumerate(self.controlled_joints):
                p.resetJointState(self.robot, joint_idx, joint_angles[i])
        p.performCollisionDetection()

        ee_state = p.getLinkState(self.robot, self.ee_link_index)
        return {
            'position': list(ee_state[4]),
            'orientation': list(ee_state[5])
        }

    def get_all_link_transforms(self, joint_angles=None):
        if joint_angles is not None:
            for i, joint_idx in enumerate(self.controlled_joints):
                p.resetJointState(self.robot, joint_idx, joint_angles[i])
        p.performCollisionDetection()

        # base_link (root, no joint parent)
        base_name = 'panda_link0' if self.robot_id_str == 'panda' else 'base_link'
        base_vo = self.visual_origins.get(base_name)
        if base_vo:
            base_pos, base_orn = self._multiply_transforms(
                [0,0,0], [0,0,0,1], base_vo['position'], base_vo['orientation'])
        else:
            base_pos, base_orn = [0,0,0], [0,0,0,1]
        transforms = [{'name': base_name, 'position': base_pos, 'orientation': base_orn}]

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot, i)
            link_name = info[12].decode('utf-8') if isinstance(info[12], bytes) else info[12]
            state = p.getLinkState(self.robot, i)
            link_pos = list(state[4])
            link_orn = list(state[5])
            
            # 如果有visual origin, 组合变换: T_world = T_link * T_visual
            vo = self.visual_origins.get(link_name)
            if vo:
                combined_pos, combined_orn = self._multiply_transforms(
                    link_pos, link_orn, vo['position'], vo['orientation'])
                transforms.append({
                    'name': link_name,
                    'position': combined_pos,
                    'orientation': combined_orn,
                })
            else:
                transforms.append({
                    'name': link_name,
                    'position': link_pos,
                    'orientation': link_orn,
                })
        return transforms

    def get_robot_info(self):
        """返回机械臂元信息(供前端使用)"""
        joints = []
        for i in self.controlled_joints:
            info = p.getJointInfo(self.robot, i)
            joints.append({
                'index': i,
                'name': info[1].decode('utf-8'),
                'type': 'revolute',
                'lower_limit': info[8],
                'upper_limit': info[9],
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

    # ==================== IK算法 ====================

    def solve_ik_pybullet(self, target_pos, target_orn=None, jac_mode='pybullet'):
        """pybullet内置DLS求解器"""
        kwargs = dict(
            maxNumIterations=self.max_iterations,
            residualThreshold=self.tolerance,
            solver=p.IK_DLS
        )
        if target_orn is not None:
            kwargs['targetOrientation'] = target_orn

        joint_angles = p.calculateInverseKinematics(
            self.robot, self.ee_link_index, target_pos, **kwargs
        )

        result_angles = [float(x) for x in joint_angles[:self.n_ctrl]]
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            result_angles[i] = float(np.clip(result_angles[i], lower, upper))

        return self._verify_result(np.array(result_angles), target_pos)

    def solve_ik_jacobian(self, target_pos, target_orn=None, jac_mode='pybullet'):
        """阻尼最小二乘法 (DLS / Levenberg-Marquardt)
        
        公式: Δq = J^T (J J^T + λ²I)^{-1} Δx
        """
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
            'dls':           self.solve_ik_pybullet,
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
        p.disconnect(self.physics_client)
