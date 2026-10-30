"""
RL IK 求解器 - 渐进式多轮迭代推理

核心思路:
  同一个 PPO 策略, 在不同动作尺度下多轮迭代:
    Round 1: scale=0.5 → 粗定位 (大步靠近)
    Round 2: scale=0.15 → 中定位
    Round 3: scale=0.05 → 精定位
    Round 4: scale=0.01 → 微调
  每轮从上一轮的最优解继续, 直到收敛或达到最大轮数。
"""
import os
import numpy as np

from ik_avoid.policy_io import load_policy
from robot_configs import get_robot_config


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_path(robot_id):
    """获取指定机器人的模型路径"""
    best_path = os.path.join(_BASE_DIR, '..', 'models', f'{robot_id}_ppo_ik_best', 'best_model')
    fallback_path = os.path.join(_BASE_DIR, '..', 'models', f'{robot_id}_ppo_ik')
    return best_path, fallback_path


class RLIKSolver:
    """基于 PPO 的渐进式 RL IK 求解器"""

    def __init__(self, model_path=None, robot_id='panda', robot_solver=None):
        self.config = get_robot_config(robot_id)
        self.robot_id = robot_id
        self.n_ctrl = self.config.get('num_axes', 6)

        if model_path is None:
            best_path, fallback_path = get_model_path(robot_id)
            if os.path.exists(best_path + '.zip'):
                model_path = best_path
            elif os.path.exists(fallback_path + '.zip'):
                model_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"未找到 {robot_id} 的训练好的 RL 模型。请先运行: ./train.sh"
                )

        print(f"  加载 RL 模型: {model_path}.zip")
        self.model = load_policy(model_path)

        # 运动学后端: 优先复用传入的 solver, 否则自建 MuJoCo 版
        if robot_solver is None:
            from mjx.mj_solver import MJRobotSolver
            robot_solver = MJRobotSolver(robot_id)
        self.robot_solver = robot_solver
        self.n_ctrl = robot_solver.n_ctrl
        self.joint_limits = robot_solver.joint_limits

        self.home_position = np.array(
            self.config.get('home_position', [0.0] * self.n_ctrl), dtype=np.float32
        )

    def _apply_joints(self, q):
        self.robot_solver._set_q(q)

    def _get_ee_pos(self):
        return self.robot_solver._get_ee_pos()

    def _clip_joints(self, q):
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            q[i] = np.clip(q[i], lower, upper)
        return q

    def _build_obs(self, current_joints, target):
        """构建观测向量 (与环境一致)"""
        ee_pos = self._get_ee_pos()
        diff = target - ee_pos
        distance = float(np.linalg.norm(diff))
        if distance > 1e-8:
            unit_dir = diff / distance
        else:
            unit_dir = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            current_joints,
            unit_dir,
            [distance],
            target,
        ]).astype(np.float32)
        return obs, distance

    def _get_action_scale(self, distance):
        """自适应动作尺度 (与环境一致)"""
        scale = 0.01 + 0.49 * min(distance / 0.5, 1.0)
        return scale

    def solve(self, target_pos, tolerance=0.005, max_rounds=6, steps_per_round=40, start_joints=None):
        """
        渐进式多轮迭代求解 IK

        每轮从上一轮最优解继续, 动作尺度随距离自适应缩小:
          Round 1: scale ≈ 0.5 (大步粗定位)
          Round 2: scale ≈ 0.15 (中步定位)
          Round 3: scale ≈ 0.05 (小步精定位)
          ...
          直到距离 < tolerance 或达到 max_rounds

        Args:
            target_pos: 目标位置 [x, y, z]
            tolerance: 收敛阈值 (m)
            max_rounds: 最大迭代轮数
            steps_per_round: 每轮最大步数
            start_joints: 起始关节角度 (可选，默认为 home position)

        Returns:
            dict: 与 RobotIKSolver._verify_result 相同格式
        """
        target = np.array(target_pos, dtype=np.float32)

        # 从指定位置或 home position 出发
        if start_joints is not None:
            current_joints = np.array(start_joints, dtype=np.float32)
        else:
            current_joints = self.home_position.copy()
        self._apply_joints(current_joints)

        best_joints = current_joints.copy()
        best_distance = float('inf')
        total_steps = 0

        for round_idx in range(max_rounds):
            round_best_distance = float('inf')
            improved_in_round = False

            for step in range(steps_per_round):
                obs, distance = self._build_obs(current_joints, target)

                # 记录历史最优
                if distance < best_distance:
                    best_distance = distance
                    best_joints = current_joints.copy()

                if distance < tolerance:
                    break

                # RL 推理
                action, _ = self.model.predict(obs, deterministic=True)

                # 自适应尺度
                action_clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
                scale = self._get_action_scale(distance)
                delta_q = action_clipped * scale

                current_joints = current_joints + delta_q
                current_joints = self._clip_joints(current_joints)
                self._apply_joints(current_joints)
                total_steps += 1

                new_distance = float(np.linalg.norm(self._get_ee_pos() - target))
                if new_distance < round_best_distance:
                    round_best_distance = new_distance
                    improved_in_round = True

            # 如果本轮没有改善, 从历史最优继续
            if not improved_in_round and best_distance < round_best_distance:
                current_joints = best_joints.copy()
                self._apply_joints(current_joints)

            # 收敛判断
            current_dist = float(np.linalg.norm(self._get_ee_pos() - target))
            if current_dist < tolerance:
                best_joints = current_joints.copy()
                best_distance = current_dist
                break

            # 如果已经很近 (< 1cm) 但还没到 tolerance, 继续下一轮
            if current_dist < 0.01 and round_idx >= 3:
                best_joints = current_joints.copy()
                best_distance = current_dist
                break

        # 最终结果
        self._apply_joints(best_joints)
        actual_pos = self._get_ee_pos()
        final_error = float(np.linalg.norm(actual_pos - target))

        return {
            'success': final_error < tolerance,
            'joint_angles': [float(x) for x in best_joints.tolist()],
            'actual_position': actual_pos.tolist(),
            'target_position': list(target_pos),
            'position_error': final_error,
            'iterations': total_steps,
            'rounds': round_idx + 1,
            'method': 'rl_ppo_progressive',
        }

    def close(self):
        pass

    def solve_multi_start(self, target_pos, tolerance=0.005, max_rounds=6, steps_per_round=40, n_starts=3):
        """
        多起始构型求解 IK
        
        尝试多个不同的起始关节配置，返回最优解。
        这对边缘区域或困难目标特别有效。
        
        Args:
            target_pos: 目标位置 [x, y, z]
            tolerance: 收敛阈值 (m)
            max_rounds: 最大迭代轮数
            steps_per_round: 每轮最大步数
            n_starts: 尝试的起始构型数量
        
        Returns:
            dict: 最优解的结果
        """
        # 生成多个起始构型
        starts = [self.home_position.copy()]  # 始终包含 home
        
        # 添加随机起始构型
        for _ in range(n_starts - 1):
            random_joints = np.zeros(self.n_ctrl, dtype=np.float32)
            for i in range(self.n_ctrl):
                lower, upper = self.joint_limits[i]
                margin = (upper - lower) * 0.20
                random_joints[i] = np.random.uniform(lower + margin, upper - margin)
            starts.append(random_joints)
        
        # 尝试每个起始构型
        results = []
        for start in starts:
            result = self.solve(
                target_pos,
                tolerance=tolerance,
                max_rounds=max_rounds,
                steps_per_round=steps_per_round,
                start_joints=start
            )
            results.append(result)
            
            # 如果已经成功，可以提前返回
            if result['success']:
                return result
        
        # 返回误差最小的结果
        best_result = min(results, key=lambda r: r['position_error'])
        best_result['method'] = 'rl_ppo_multi_start'
        best_result['n_attempts'] = len(results)
        return best_result


def is_model_available(robot_id='panda'):
    """检查是否有训练好的模型可用"""
    best_path, fallback_path = get_model_path(robot_id)
    return (
        os.path.exists(best_path + '.zip') or
        os.path.exists(fallback_path + '.zip')
    )
