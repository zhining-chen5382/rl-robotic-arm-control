"""
6D 位姿 RL IK 求解器 (MAPPO-IK 模型)

复用主 solver 的运动学后端 (MJRobotSolver/RobotIKSolver), 避免重复建场景。
用训练好的 PPO 策略 (map 阶段) 迭代控制求解 6D 位姿 IK。

用法:
    from rl_ik_solver_6d import RLIK6DSolver
    solver_6d = RLIK6DSolver(main_solver)          # 传入主 solver 实例
    result = solver_6d.solve(target_pos, target_quat)
"""
import os
import time
import numpy as np

from ik_avoid.policy_io import load_policy
from .quat_utils import (
    quat_multiply, quat_conjugate, quat_normalize,
    quat_to_axis_angle, quat_angle,
)


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class RLIK6DSolver:
    """6D 位姿 RL IK 求解器 (复用主 solver 的运动学后端)"""

    def __init__(self, robot_solver, model_path=None):
        """
        Args:
            robot_solver: 主 solver 实例 (需提供 _set_q/_get_ee_pose 接口)
            model_path: PPO 模型路径 (不含 .zip)
        """
        self.solver = robot_solver
        self.n_ctrl = robot_solver.n_ctrl
        self.joint_limits = robot_solver.joint_limits

        self.home_position = np.array(
            robot_solver.config.get('home_position', [0.0] * self.n_ctrl),
            dtype=np.float32,
        )

        # 加载 PPO 模型
        if model_path is None:
            model_path = os.path.join(_BASE_DIR, '..', 'models', 'panda_ppo_ik_6d_map')
        print(f"  [6D] 加载 RL 模型: {model_path}.zip")
        self.model = load_policy(model_path)

        self.max_steps = 100

    def _apply_joints(self, q):
        self.solver._set_q(q)

    def _get_ee_pose(self):
        return self.solver._get_ee_pose()

    def _get_obs(self, current_joints, target_pos, target_quat,
                 step_count, max_steps, prev_pos_dist):
        """构建 map 阶段观测向量 (与训练环境完全一致)"""
        ee_pos, ee_quat = self._get_ee_pose()
        pos_diff = target_pos - ee_pos
        pos_dist = float(np.linalg.norm(pos_diff))
        if pos_dist > 1e-8:
            pos_unit_dir = (pos_diff / pos_dist).astype(np.float32)
        else:
            pos_unit_dir = np.zeros(3, dtype=np.float32)

        # 姿态误差: 等效轴角
        q_err = quat_multiply(target_quat, quat_conjugate(ee_quat))
        orient_error_aa = quat_to_axis_angle(q_err)
        orient_dist = float(np.linalg.norm(orient_error_aa))

        # MAPPO-IK 耦合特征
        cosine_dist = 2.0 - np.cos(orient_dist)
        final_dist = pos_dist * cosine_dist + (cosine_dist - 1.0)
        step_norm = step_count / max_steps

        obs = np.concatenate([
            current_joints, pos_unit_dir, [pos_dist],
            orient_error_aa, [cosine_dist], [final_dist], [step_norm],
        ]).astype(np.float32)
        return obs, pos_dist, orient_dist, ee_pos

    @staticmethod
    def _get_action_scale(distance):
        """自适应动作尺度 (与训练环境一致)
        近端 min_scale=0.003 保证 5mm 精度不 overshoot
        """
        return 0.003 + 0.49 * min(distance / 0.5, 1.0)

    def _clip_joints(self, q):
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            q[i] = np.clip(q[i], lower, upper)
        return q

    def solve(self, target_pos, target_quat, max_steps=None):
        """
        求解 6D 位姿 IK

        Args:
            target_pos:  [x, y, z]
            target_quat: [x, y, z, w]
            max_steps:   最大迭代步数 (默认 100)

        Returns:
            dict: joint_angles, position_error, orientation_error, solve_time_ms, ...
        """
        if max_steps is None:
            max_steps = self.max_steps

        target_pos = np.array(target_pos, dtype=np.float32)
        target_quat = quat_normalize(np.array(target_quat, dtype=np.float32))

        # 从 home 位置出发
        current_joints = self.home_position.copy()
        self._apply_joints(current_joints)

        ee_pos, _ = self._get_ee_pose()
        prev_pos_dist = float(np.linalg.norm(ee_pos - target_pos))

        best_joints = current_joints.copy()
        best_pos_dist = prev_pos_dist
        best_orient_dist = quat_angle(self._get_ee_pose()[1], target_quat)

        t0 = time.perf_counter()
        steps = 0

        for step in range(max_steps):
            obs, pos_dist, orient_dist, ee_pos = self._get_obs(
                current_joints, target_pos, target_quat,
                step, max_steps, prev_pos_dist,
            )

            # 记录历史最优 (以耦合距离为准)
            cosine_d = 2.0 - np.cos(orient_dist)
            coupled = pos_dist * cosine_d + (cosine_d - 1.0)
            best_cosine_d = 2.0 - np.cos(best_orient_dist)
            best_coupled = best_pos_dist * best_cosine_d + (best_cosine_d - 1.0)
            if coupled < best_coupled:
                best_pos_dist = pos_dist
                best_orient_dist = orient_dist
                best_joints = current_joints.copy()

            # 收敛判断 (内部精度目标 5mm+10°, 保持高精度求解)
            if pos_dist < 0.005 and orient_dist < 0.175:
                break

            # RL 推理
            action, _ = self.model.predict(obs, deterministic=True)
            action_clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
            scale = self._get_action_scale(prev_pos_dist)
            delta_q = action_clipped * scale

            current_joints = current_joints + delta_q
            current_joints = self._clip_joints(current_joints)
            self._apply_joints(current_joints)

            ee_pos, _ = self._get_ee_pose()
            prev_pos_dist = float(np.linalg.norm(ee_pos - target_pos))
            steps += 1

        t_solve = (time.perf_counter() - t0) * 1000

        # 最终结果 (用历史最优)
        self._apply_joints(best_joints)
        final_pos, final_quat = self._get_ee_pose()
        pos_error = float(np.linalg.norm(final_pos - target_pos))
        orient_error = quat_angle(final_quat, target_quat)

        return {
            'success': bool(pos_error < 0.020 and orient_error < 0.175),  # 20mm + 10°
            'joint_angles': [float(x) for x in best_joints.tolist()],
            'actual_position': final_pos.tolist(),
            'actual_orientation': final_quat.tolist(),
            'target_position': target_pos.tolist(),
            'target_orientation': target_quat.tolist(),
            'position_error': pos_error,
            'orientation_error': float(orient_error),
            'orientation_error_deg': float(np.degrees(orient_error)),
            'iterations': steps,
            'solve_time_ms': round(t_solve, 2),
            'method': 'rl_ppo_6d_map',
        }

    def random_reachable_pose(self, margin=0.15):
        """生成随机可达 6D 位姿: 随机关节角 → FK"""
        q = np.zeros(self.n_ctrl, dtype=np.float32)
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            m = (upper - lower) * margin
            q[i] = np.random.uniform(lower + m, upper - m)
        self._apply_joints(q)
        pos, quat = self._get_ee_pose()
        return pos.tolist(), quat.tolist(), q.tolist()
