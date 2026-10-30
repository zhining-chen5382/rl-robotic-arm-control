"""
机械臂逆运动学 RL 训练脚本

支持: UR5 (6轴), Panda (7轴)
模式: 3D 位置 IK / 6D 位姿 IK (MAPPO-IK)

用法:
    # 3D 位置 IK
    python train_rl_ik.py --robot panda --timesteps 3000000
    
    # 6D 位姿 IK (MAPPO-IK 一步联合)
    python train_rl_ik.py --robot panda --mode 6d --timesteps 30000000
    
    python train_rl_ik.py --eval                   # 仅评估 3D
    python train_rl_ik.py --eval --mode 6d         # 仅评估 6D
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
import multiprocessing

from ik_3d.rl_env import RobotIKEnv
from ik_6d.rl_env_6d import RobotIK6DEnv


def linear_schedule(initial_value, final_value):
    """线性学习率调度: initial_value → final_value"""
    def f(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return f


class ProgressCallback(BaseCallback):
    """显示训练进度百分比的回调"""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_timesteps = 0
        self.last_print = 0

    def _on_training_start(self):
        # 记录起始时间步 (续训时 num_timesteps > 0)
        self.initial_timesteps = self.model.num_timesteps

    def _on_step(self):
        # 每 1% 打印一次进度 (基于本轮新增步数)
        new_steps = self.model.num_timesteps - self.initial_timesteps
        progress = (new_steps / self.total_timesteps) * 100
        if progress - self.last_print >= 1.0:
            self.last_print = progress
            print(f"\n  [进度] {progress:.1f}% ({new_steps:,}/{self.total_timesteps:,} new steps)")
        return True


class CurriculumCallback(BaseCallback):
    """课程学习回调: 随训练进度扩大目标采样范围"""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_stage = -1
    
    def _on_step(self):
        progress = self.model.num_timesteps / self.total_timesteps
        
        # 课程阶段设计
        if progress < 0.25:
            margin = 0.35  # 前 25%: 只采样核心区域 (容易)
            stage = 0
        elif progress < 0.50:
            margin = 0.25  # 25-50%: 扩大范围
            stage = 1
        elif progress < 0.75:
            margin = 0.18  # 50-75%: 接近全空间
            stage = 2
        else:
            margin = 0.12  # 后 25%: 全空间探索
            stage = 3
        
        # 只在阶段变化时打印
        if stage != self.last_stage:
            self.last_stage = stage
            stage_names = ['核心区域', '扩大范围', '接近全空间', '全空间']
            print(f"\n  [课程学习] 进入阶段 {stage+1}: {stage_names[stage]} (margin={margin})")
        
        # 通过 env_method 设置所有环境的 margin
        try:
            self.training_env.env_method('set_curriculum_margin', margin)
        except Exception:
            # 如果 env_method 不可用，尝试其他方式
            pass
        
        return True


class CurriculumCallback6D(BaseCallback):
    """6D 课程学习: MAPPO-IK 全空间采样"""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_stage = -1

    def _on_step(self):
        # MAPPO-IK: 无课程, 全空间采样 (只设置一次)
        if self.last_stage != 99:
            self.last_stage = 99
            try:
                self.training_env.env_method('set_curriculum_margin', 0.15)
            except Exception:
                pass
        return True


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# 当前机器人配置 (默认为 Panda)
CURRENT_ROBOT = os.environ.get('ROBOT_ID', 'panda')
MODEL_PATH = os.path.join(MODELS_DIR, f'{CURRENT_ROBOT}_ppo_ik')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, f'{CURRENT_ROBOT}_ppo_ik_best')


def make_env(robot_id, rank, seed=0, mode='3d'):
    """工厂函数: 创建独立环境实例"""
    def _init():
        if mode == '6d':
            env = RobotIK6DEnv(
                robot_id=robot_id, max_steps=100,
                pos_tolerance=0.005, orient_tolerance=0.175,
            )
        else:
            env = RobotIKEnv(robot_id=robot_id, max_steps=40, tolerance=0.005)
        env.reset(seed=seed + rank)
        return env
    return _init


def _get_model_paths(robot_id, mode='3d'):
    """根据模式返回模型路径"""
    if mode == '6d':
        name = f'{robot_id}_ppo_ik_6d_map'
    else:
        name = f'{robot_id}_ppo_ik'
    model_path = os.path.join(MODELS_DIR, name)
    best_path = os.path.join(MODELS_DIR, f'{name}_best')
    return model_path, best_path


def train(timesteps=500_000, n_envs=8, robot_id='panda', mode='3d', device='cpu'):
    """训练 PPO 模型"""
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global MODEL_PATH, BEST_MODEL_PATH, CURRENT_ROBOT
    CURRENT_ROBOT = robot_id
    MODEL_PATH, BEST_MODEL_PATH = _get_model_paths(robot_id, mode)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    mode_str = '6D 位姿 (MAPPO-IK)' if mode == '6d' else '3D 位置'
    print("=" * 60)
    print(f"  {robot_id.upper()} RL-IK 训练 [{mode_str}]")
    print(f"  总步数: {timesteps:,}  |  并行环境: {n_envs}  |  设备: {device}")
    print(f"  模型保存: {MODEL_PATH}.zip")
    print("=" * 60)

    # 向量化环境
    env_fns = [make_env(robot_id, i, seed=i * 100, mode=mode) for i in range(n_envs)]
    
    try:
        env = SubprocVecEnv(env_fns, start_method='spawn')
        print(f"  使用 SubprocVecEnv ({n_envs} 进程, spawn)")
    except Exception as e:
        print(f"  SubprocVecEnv 失败: {e}")
        print("  回退到 DummyVecEnv")
        env = DummyVecEnv(env_fns)
        print(f"  使用 DummyVecEnv ({n_envs} 环境)")

    # episode 级别监控 (TensorBoard 记录 episode reward / length)
    env = VecMonitor(env)

    # 6D 模式: 奖励归一化 (对应 ML-Agents 默认机制)
    vec_normalize = None
    if mode == '6d':
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=20.0, gamma=0.99)
        vec_normalize = env
        print("  已启用 VecNormalize 奖励归一化")

    eval_env = DummyVecEnv([make_env(robot_id, 99, seed=9999, mode=mode)])
    eval_env = VecMonitor(eval_env)  # 与训练环境包装结构一致
    if vec_normalize is not None:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_PATH,
        log_path=LOG_DIR,
        eval_freq=500_000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
        verbose=1,
    )

    checkpoint_prefix = f'ppo_{robot_id}_ik_6d_map' if mode == '6d' else f'ppo_{robot_id}_ik'
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path=os.path.join(MODELS_DIR, 'checkpoints'),
        name_prefix=checkpoint_prefix,
        verbose=1,
    )

    resume = False
    if os.path.exists(MODEL_PATH + '.zip'):
        print(f"\n  发现已有模型, 从 {MODEL_PATH}.zip 继续训练...")
        model = PPO.load(MODEL_PATH, env=env, verbose=1, device=device, tensorboard_log=LOG_DIR)
        # 续训: 保守微调 (低 LR + 低 ent_coef 促使策略收敛)
        model.learning_rate = linear_schedule(5e-5, 1e-6)
        model.ent_coef = 0.005
        model._setup_lr_schedule()
        print(f"  续训学习率: {model.lr_schedule(1.0):.2e}")
        print(f"  续训 ent_coef: {model.ent_coef}")
        norm_path = MODEL_PATH + '_norm.pkl'
        if vec_normalize is not None and os.path.exists(norm_path):
            loaded_norm = VecNormalize.load(norm_path, env.venv)
            vec_normalize.ret_rms = loaded_norm.ret_rms
            print(f"  已恢复归一化统计: {norm_path}")
        resume = True
    else:
        if mode == '6d':
            # MAPPO-IK 论文超参: batch=1024, lr=2e-4→1e-5, 512x3 层
            net_arch = dict(pi=[512, 512, 512], vf=[512, 512, 512])
            model = PPO(
                policy='MlpPolicy',
                env=env,
                verbose=1,
                learning_rate=linear_schedule(2e-4, 1e-5),
                n_steps=2048,
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                device=device,
                policy_kwargs=dict(net_arch=net_arch, ortho_init=True),
                tensorboard_log=LOG_DIR,
            )
        else:
            net_arch = dict(pi=[512, 256], vf=[512, 256])
            model = PPO(
                policy='MlpPolicy',
                env=env,
                verbose=1,
                learning_rate=linear_schedule(3e-4, 1e-5),
                n_steps=2048,
                batch_size=256,
                n_epochs=15,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                device=device,
                policy_kwargs=dict(net_arch=net_arch, ortho_init=True),
                tensorboard_log=LOG_DIR,
            )

    print("\n  开始训练...\n")
    
    progress_callback = ProgressCallback(timesteps)
    if mode == '6d':
        curriculum_callback = CurriculumCallback6D(timesteps)
    else:
        curriculum_callback = CurriculumCallback(timesteps)
    
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback, curriculum_callback],
        progress_bar=False,
        reset_num_timesteps=True,  # 每次训练从 0 开始, 保证 LR 和进度正确
    )

    model.save(MODEL_PATH)
    print(f"\n  最终模型已保存: {MODEL_PATH}.zip")
    if vec_normalize is not None:
        norm_path = MODEL_PATH + '_norm.pkl'
        vec_normalize.save(norm_path)
        print(f"  归一化统计已保存: {norm_path}")

    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("  训练完成, 执行评估...")
    print("=" * 60)
    if mode == '6d':
        evaluate_6d_single_episode(n_episodes=50, robot_id=robot_id)
    else:
        evaluate_single_episode(n_episodes=50, robot_id=robot_id)
        evaluate_progressive(n_targets=30, robot_id=robot_id)


def evaluate_single_episode(n_episodes=50, robot_id='panda'):
    """评估单轮 (非渐进) 性能"""
    model_path = os.path.join(MODELS_DIR, f'{robot_id}_ppo_ik')
    if not os.path.exists(model_path + '.zip'):
        return

    model = PPO.load(model_path, device='cpu')
    env = RobotIKEnv(robot_id=robot_id, max_steps=40, tolerance=0.005)

    successes = 0
    distances = []
    unreachable = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        
        # 检查目标是否可达
        target_pos = info.get('target_position', None)
        if target_pos is not None:
            dist_from_base = np.linalg.norm(target_pos)
            # UR5 最大臂展约 0.85m，留 10% 余量
            if dist_from_base > 0.85 * 0.9:
                unreachable += 1
                continue
        
        done = False
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                done = True
                break
            if truncated:
                break

        dist = info.get('distance', float('inf'))
        distances.append(dist)
        if done or dist < env.tolerance:
            successes += 1

    env.close()
    
    valid_count = len(distances)
    print(f"\n  单轮评估 ({n_episodes} 目标点):")
    print(f"    不可达跳过: {unreachable}")
    print(f"    有效测试: {valid_count}")
    print(f"    成功率:   {successes}/{valid_count} ({successes/valid_count*100:.1f}%)")
    if distances:
        print(f"    平均误差: {np.mean(distances):.6f} m ({np.mean(distances)*1000:.2f} mm)")
        print(f"    中位误差: {np.median(distances):.6f} m ({np.median(distances)*1000:.2f} mm)")
        print(f"    最小误差: {np.min(distances):.6f} m ({np.min(distances)*1000:.2f} mm)")
        print(f"    最大误差: {np.max(distances):.6f} m ({np.max(distances)*1000:.2f} mm)")
        # 成功率细分
        within_5mm = sum(1 for d in distances if d < 0.005)
        within_10mm = sum(1 for d in distances if d < 0.01)
        within_20mm = sum(1 for d in distances if d < 0.02)
        print(f"    <5mm: {within_5mm}/{valid_count} ({within_5mm/valid_count*100:.1f}%)")
        print(f"    <10mm: {within_10mm}/{valid_count} ({within_10mm/valid_count*100:.1f}%)")
        print(f"    <20mm: {within_20mm}/{valid_count} ({within_20mm/valid_count*100:.1f}%)")


def evaluate_progressive(n_targets=30, robot_id='panda'):
    """评估渐进式多轮迭代推理性能"""
    try:
        from ik_3d.rl_ik_solver import RLIKSolver
    except Exception:
        print("  无法导入 RLIKSolver, 跳过渐进式评估")
        return

    # 找模型路径
    best_path = os.path.join(MODELS_DIR, f'{robot_id}_ppo_ik_best', 'best_model')
    fallback_path = os.path.join(MODELS_DIR, f'{robot_id}_ppo_ik')
    
    if os.path.exists(best_path + '.zip'):
        model_path = best_path
    elif os.path.exists(fallback_path + '.zip'):
        model_path = fallback_path
    else:
        return

    solver = RLIKSolver(model_path=model_path, robot_id=robot_id)

    successes = 0
    distances = []
    rounds_list = []
    steps_list = []
    unreachable = 0
    
    # 可达性参数
    if robot_id == 'ur5':
        MAX_REACH = 0.80  # 800mm
        MIN_Z = -0.05
    else:  # panda
        MAX_REACH = 0.80  # 800mm
        MIN_Z = 0.05      # Panda 基座较高

    attempts = 0
    while len(distances) < n_targets and attempts < n_targets * 3:
        attempts += 1
        
        # 随机采样目标（增加 margin 到 25%）
        target_q = np.zeros(solver.n_ctrl, dtype=np.float32)
        for i in range(solver.n_ctrl):
            lower, upper = solver.joint_limits[i]
            margin = (upper - lower) * 0.25  # 增加到 25%
            target_q[i] = np.random.uniform(lower + margin, upper - margin)

        solver._apply_joints(target_q)
        target_pos = solver._get_ee_pos().tolist()
        
        # 可达性检查
        dist_from_base = np.linalg.norm(target_pos)
        if dist_from_base > MAX_REACH:
            unreachable += 1
            continue
        if target_pos[2] < MIN_Z:
            unreachable += 1
            continue

        result = solver.solve(target_pos, tolerance=0.005, max_rounds=6, steps_per_round=40)
        dist = result['position_error']
        distances.append(dist)
        rounds_list.append(result.get('rounds', 0))
        steps_list.append(result['iterations'])

        if result['success']:
            successes += 1

    solver.close()

    valid_count = len(distances)
    print(f"\n  渐进式推理评估 (尝试 {attempts} 次，有效 {valid_count} 点):")
    print(f"    不可达跳过: {unreachable}")
    print(f"    成功率:   {successes}/{valid_count} ({successes/valid_count*100:.1f}%)")
    if distances:
        print(f"    平均误差: {np.mean(distances):.6f} m ({np.mean(distances)*1000:.2f} mm)")
        print(f"    中位误差: {np.median(distances):.6f} m ({np.median(distances)*1000:.2f} mm)")
        print(f"    最小误差: {np.min(distances):.6f} m ({np.min(distances)*1000:.2f} mm)")
        print(f"    最大误差: {np.max(distances):.6f} m ({np.max(distances)*1000:.2f} mm)")
        # 成功率细分
        within_5mm = sum(1 for d in distances if d < 0.005)
        within_10mm = sum(1 for d in distances if d < 0.01)
        within_20mm = sum(1 for d in distances if d < 0.02)
        print(f"    <5mm: {within_5mm}/{valid_count} ({within_5mm/valid_count*100:.1f}%)")
        print(f"    <10mm: {within_10mm}/{valid_count} ({within_10mm/valid_count*100:.1f}%)")
        print(f"    <20mm: {within_20mm}/{valid_count} ({within_20mm/valid_count*100:.1f}%)")
        # 失败分析
        failed = [d for d in distances if d >= 0.005]
        if failed:
            print(f"    失败用例: {len(failed)} 个")
            print(f"      失败平均误差: {np.mean(failed):.6f} m ({np.mean(failed)*1000:.2f} mm)")
            print(f"      失败中位误差: {np.median(failed):.6f} m ({np.median(failed)*1000:.2f} mm)")
    print(f"    平均轮数: {np.mean(rounds_list):.1f}")
    print(f"    平均步数: {np.mean(steps_list):.1f}")
    print("=" * 60)


def evaluate_6d_single_episode(n_episodes=50, robot_id='panda'):
    """6D 单轮评估"""
    _, best_path = _get_model_paths(robot_id, '6d')
    model_path = best_path if os.path.exists(best_path + '.zip') else None
    if model_path is None:
        model_path, _ = _get_model_paths(robot_id, '6d')
    if not os.path.exists(model_path + '.zip'):
        print(f"  未找到 6D 模型, 跳过评估")
        return

    model = PPO.load(model_path, device='cpu')
    env = RobotIK6DEnv(robot_id=robot_id, max_steps=100,
                       pos_tolerance=0.005, orient_tolerance=0.175)

    pos_successes = 0
    orient_successes = 0
    joint_successes = 0
    pos_dists = []
    orient_dists = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                done = True
                break
            if truncated:
                break

        pd = info.get('pos_distance', float('inf'))
        od = info.get('orient_distance', float('inf'))
        pos_dists.append(pd)
        orient_dists.append(od)

        if pd < env.pos_tolerance:
            pos_successes += 1
        if od < env.orient_tolerance:
            orient_successes += 1
        if pd < env.pos_tolerance and od < env.orient_tolerance:
            joint_successes += 1

    env.close()
    n = len(pos_dists)
    print(f"\n  6D 单轮评估 ({n_episodes} 目标点):")
    print(f"    有效测试: {n}")
    print(f"    位置成功率:  {pos_successes}/{n} ({pos_successes/n*100:.1f}%)")
    print(f"    姿态成功率:  {orient_successes}/{n} ({orient_successes/n*100:.1f}%)")
    print(f"    联合成功率:  {joint_successes}/{n} ({joint_successes/n*100:.1f}%)")
    if pos_dists:
        print(f"    位置中位误差: {np.median(pos_dists)*1000:.2f} mm")
        print(f"    姿态中位误差: {np.degrees(np.median(orient_dists)):.2f} deg")
        for thresh, name in [(0.005, '<5mm'), (0.01, '<10mm'), (0.02, '<20mm')]:
            cnt = sum(1 for d in pos_dists if d < thresh)
            print(f"    位置 {name}: {cnt}/{n} ({cnt/n*100:.1f}%)")
        for thresh_deg, name in [(5, '<5deg'), (10, '<10deg'), (15, '<15deg')]:
            thresh = np.radians(thresh_deg)
            cnt = sum(1 for d in orient_dists if d < thresh)
            print(f"    姿态 {name}: {cnt}/{n} ({cnt/n*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL-IK 训练')
    parser.add_argument('--robot', type=str, default='panda',
                        choices=['ur5', 'panda'],
                        help='机器人类型 (默认: panda)')
    parser.add_argument('--mode', type=str, default='3d',
                        choices=['3d', '6d'],
                        help='IK 模式 (默认: 3d)')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='训练总步数 (默认: 500000)')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='并行环境数 (默认: 8)')
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备: cpu/cuda/auto (默认: auto)')
    parser.add_argument('--eval', action='store_true',
                        help='仅评估已训练模型')
    args = parser.parse_args()

    if args.eval:
        if args.mode == '6d':
            evaluate_6d_single_episode(50, robot_id=args.robot)
        else:
            evaluate_single_episode(50, robot_id=args.robot)
            evaluate_progressive(30, robot_id=args.robot)
    else:
        train(timesteps=args.timesteps, n_envs=args.n_envs,
              robot_id=args.robot, mode=args.mode, device=args.device)
