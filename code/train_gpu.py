"""GPU 驻留 PPO 训练入口 (唯一学习器; SB3 路径已于 2026-08 移除)

环境/奖励/课程/评测口径沿用历史 SB3 版 (train_mjx.py) 的定义,
学习器为 ik_avoid.ppo_gpu.GpuPPO (rollout 数据常驻 GPU, 梯度步零搬运):
    - 环境直接用 MJXAvoidVecEnv (奖励归一化由 RewardNormalizer 逐字段复刻)
    - 课程推进/周期评测/大样本抽查复刻原 callbacks 语义
      (复用 eval_stats 的 inject_scenes/rollout_eval/summarize)
    - best 选择口径不变: 障碍/无障碍严格成功率均值
    - 保存同时产出 zip (推理读, policy_io.load_policy) 与自研 .pt (续训)

用法 (与 train_mjx.py 同名参数):
    python train_gpu.py --config configs/avoid_mjx_v15gpu.yaml --n_worlds 4096
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ik_avoid.config import load_config, save_config
from ik_avoid.eval_stats import inject_scenes, rollout_eval, summarize
from ik_avoid.ppo_gpu import GpuPPO, RewardNormalizer
from mjx.vec_env import MJXAvoidVecEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')


class _EvalSet:
    """固定评测场景集 (对齐 MJXBatchEvalCallback / MJXSpotEvalCallback)"""

    def __init__(self, cfg, n_total, seed_offset, seed):
        from mjx.env_mjx import RobotIKAvoidEnvMJX
        n_half = max(n_total // 2, 1)
        self.env = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=n_half,
                                      max_points=768)
        self.scenes_obs = self.env.generate_scenes(
            n_half, seed + seed_offset, with_obstacles=True)
        self.scenes_free = self.env.generate_scenes(
            n_half, seed + seed_offset + 1000, with_obstacles=False)

    def run(self, learner):
        inject_scenes(self.env, self.scenes_obs)
        r_obs = rollout_eval(self.env, learner, self.scenes_obs,
                             self.env.max_steps)
        inject_scenes(self.env, self.scenes_free)
        r_free = rollout_eval(self.env, learner, self.scenes_free,
                              self.env.max_steps)
        return r_obs, r_free


def _curriculum_stage(stages, progress, start_stage):
    """复刻 ObstacleCurriculumCallback 的阶段判定"""
    current = start_stage
    for i, s in enumerate(stages):
        if progress >= s.threshold:
            current = i
    return max(current, start_stage)


def train(args):
    import torch
    _n_threads = int(os.environ.get('TORCH_NUM_THREADS', '0'))
    if _n_threads > 0:
        torch.set_num_threads(_n_threads)
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'Training configuration does not exist: {args.config}')
    cfg = load_config(args.config)
    tc = cfg.train

    if args.timesteps:
        tc.timesteps = args.timesteps
    if args.model_name:
        tc.model_name = args.model_name
    n_worlds = args.n_worlds
    n_steps = args.n_steps
    batch_size = args.batch_size or tc.batch_size
    if (n_steps * n_worlds) % batch_size != 0:
        batch_size = n_steps * n_worlds // max(
            1, (n_steps * n_worlds) // batch_size)
        print(f"  [提示] batch_size 调整为 {batch_size} "
              f"(需整除 n_steps×n_worlds={n_steps*n_worlds})")
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    model_name = f'{tc.model_name}_mjx'
    model_path = os.path.join(MODELS_DIR, model_name)
    best_dir = os.path.join(MODELS_DIR, f'{model_name}_best')
    ckpt_dir = os.path.join(MODELS_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 62)
    print("  无碰撞 IK 训练 (GPU驻留 PPO + MJX GPU 向量化环境)")
    print(f"  总步数: {tc.timesteps:,}  |  并行世界: {n_worlds}  |  设备: {device}")
    print(f"  n_steps={n_steps}  batch_size={batch_size}  "
          f"n_epochs={tc.n_epochs}  (rollout={n_steps*n_worlds:,})")
    print(f"  课程阶段: {' -> '.join(s.name for s in cfg.obstacle.stages)}")
    print(f"  模型保存: {model_path}.zip (推理) + {model_path}_gpu.pt")
    print("=" * 62)

    save_config(cfg, os.path.join(LOG_DIR, f'{model_name}_config.yaml'))

    env = MJXAvoidVecEnv(cfg, n_worlds=n_worlds, stage=args.start_stage,
                         seed=args.seed)
    print(f"  训练环境: MJXAvoidVecEnv ({n_worlds} 世界, "
          f"起始阶段 {args.start_stage})")

    normalizer = RewardNormalizer(n_worlds, clip_reward=tc.clip_reward,
                                  gamma=tc.gamma)
    learner = GpuPPO(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        arch=tuple(tc.net_arch),
        device=device,
        n_steps=n_steps, n_envs=n_worlds, batch_size=batch_size,
        n_epochs=tc.n_epochs, gamma=tc.gamma, gae_lambda=tc.gae_lambda,
        clip_range=tc.clip_range, ent_coef=tc.ent_coef,
        lr_init=tc.learning_rate, lr_final=tc.lr_final, seed=args.seed,
        constrained=tc.constrained_ppo,
        cost_gamma=tc.cost_gamma, cost_gae_lambda=tc.cost_gae_lambda,
        cost_vf_coef=tc.cost_vf_coef,
        cost_limit_start=tc.cost_limit_start,
        cost_limit_final=tc.cost_limit_final,
        lagrange_init=tc.lagrange_init, lagrange_lr=tc.lagrange_lr,
        lagrange_max=tc.lagrange_max,
        collision_backtrace_steps=tc.collision_backtrace_steps,
        collision_backtrace_penalty=tc.collision_backtrace_penalty,
        collision_backtrace_decay=tc.collision_backtrace_decay,
        actor_update_mode=tc.actor_update_mode,
        step_action_index=tc.step_action_index,
        step_risk_head=tc.step_risk_head,
        step_risk_hidden=tc.step_risk_hidden,
        direction_risk_head=tc.direction_risk_head,
        direction_risk_hidden=tc.direction_risk_hidden,
        direction_risk_max_residual=tc.direction_risk_max_residual,
        direction_risk_margin_norm=tc.direction_risk_margin_norm,
    )
    actor_summary = learner.trainable_parameter_summary()
    print(f"  Actor更新模式: {actor_summary['actor_update_mode']} | "
          f"动作索引: {actor_summary['step_action_index']} | "
          f"可训练参数: {actor_summary['trainable_parameters']:,}/"
          f"{actor_summary['total_parameters']:,}")

    # ---------- 续训 / 从已有 champion 初始化新实验 ----------
    # 优先 final 产物; 否则回退最新检查点 (中断恢复场景)
    import glob
    import re
    num_timesteps = 0
    resume_src = None
    if (os.path.exists(model_path + '_gpu.pt')
            and os.path.exists(model_path + '_norm.pkl')
            and not args.no_resume):
        resume_src = (model_path + '_gpu.pt', model_path + '_norm.pkl')
    elif not args.no_resume:
        cands = [c for c in glob.glob(
            os.path.join(ckpt_dir, f'gpu_{model_name}_*_steps.pt'))
            if os.path.exists(c + '.meta.pkl')]
        if cands:
            latest = max(cands, key=lambda c: int(
                re.search(r'_(\d+)_steps\.pt$', c).group(1)))
            resume_src = (latest, latest + '.meta.pkl')
    if resume_src:
        print(f"\n  发现检查点, 从 {resume_src[0]} 继续训练...")
        learner.load(resume_src[0])
        with open(resume_src[1], 'rb') as f:
            norm_state = pickle.load(f)
        normalizer.load_state_dict(norm_state['ret'])
        num_timesteps = int(norm_state.get('num_timesteps', 0))
        print(f"  已恢复权重/归一化统计/步数 ({num_timesteps:,})")
    elif args.init_from:
        # init-from 与断点续训不同：只继承策略/价值权重和奖励统计，
        # 新实验的步数、优化器和学习率周期均从 0 开始。
        src = os.path.normpath(args.init_from)
        if src.lower().endswith('.zip'):
            model_src = src
            stem = src[:-4]
        elif src.lower().endswith('.pt'):
            model_src = src
            stem = src[:-7] if src.lower().endswith('_gpu.pt') else src[:-3]
        else:
            zip_src = src + '.zip'
            pt_src = src + '_gpu.pt'
            model_src = zip_src if os.path.exists(zip_src) else pt_src
            stem = src
        if not os.path.exists(model_src):
            raise FileNotFoundError(f'初始化模型不存在: {model_src}')
        print(f"\n  从已有模型初始化新实验: {model_src}")
        if model_src.lower().endswith('.zip'):
            learner.load_policy_zip(
                model_src,
                extra_action_init=tc.extra_action_init,
                extra_action_log_std=tc.extra_action_log_std,
                widen_noise_std=tc.widen_noise_std)
        else:
            learner.load(model_src)

        norm_candidates = [stem + '_norm.pkl']
        parent = os.path.dirname(stem)
        base = os.path.basename(stem)
        if base in ('champion_model', 'best_model'):
            norm_candidates.insert(0, os.path.join(
                parent, base.replace('_model', '_norm') + '.pkl'))
            # best/champion 权重位于 ``<run>_best|_champion/`` 子目录，
            # 奖励归一化统计则随 final 权重保存在相邻的 ``<run>_norm.pkl``。
            # 微调初始化时奖励统计与具体 actor 检查点无关，可安全复用。
            for suffix in ('_best', '_champion'):
                if parent.endswith(suffix):
                    norm_candidates.insert(0, parent[:-len(suffix)]
                                           + '_norm.pkl')
        norm_src = next((p for p in norm_candidates if os.path.exists(p)), None)
        if norm_src:
            with open(norm_src, 'rb') as f:
                norm_state = pickle.load(f)
            ret_state = dict(norm_state['ret'])
            ret_state['returns'] = np.zeros(n_worlds, dtype=np.float64)
            normalizer.load_state_dict(ret_state)
            print(f"  已继承奖励归一化统计: {norm_src} (回合累计值已清零)")
        else:
            print("  [提示] 未找到归一化统计，将使用新统计")
        num_timesteps = 0
        print("  新实验步数/优化器/学习率周期从 0 开始")

    elif tc.actor_update_mode != 'full':
        raise ValueError(
            'A frozen-backbone training mode requires --init-from or a valid '
            'resume checkpoint; see code/README.md.')

    # ---------- 评测场景集 (固定, 跨评测可比) ----------
    eval_set = _EvalSet(cfg, tc.n_eval_episodes, seed_offset=0, seed=9999)
    spot_set = None
    if tc.spot_eval_n > 0:
        spot_set = _EvalSet(cfg, tc.spot_eval_n, seed_offset=1000, seed=7777)
        print(f"  抽查评测: 每 {tc.spot_eval_freq_steps:,} 步, "
              f"n={tc.spot_eval_n} (障碍/无障碍各半)")

    total = tc.timesteps
    rollout_size = n_steps * n_worlds
    eval_freq = max(tc.eval_freq_steps // rollout_size, 1)      # 迭代粒度
    spot_freq = max(tc.spot_eval_freq_steps // rollout_size, 1)
    ckpt_freq = max(tc.checkpoint_freq_steps // rollout_size, 1)
    start_stage = int(np.clip(args.start_stage, 0,
                              len(cfg.obstacle.stages) - 1))

    best_metric = -np.inf
    obs = env.reset()
    ep_starts = np.zeros(n_worlds, dtype=bool)
    cur_stage = start_stage - 1
    last_print_pct = 0.0
    t_start = time.perf_counter()
    print("\n  开始训练...\n")

    iteration = 0
    while num_timesteps < total:
        iteration += 1
        t_it = time.perf_counter()
        obs, ep_starts, roll_m = learner.collect(env, obs, normalizer,
                                                 ep_starts)
        t_collect = time.perf_counter()
        progress_remaining = max(0.0, 1.0 - (num_timesteps + rollout_size)
                                 / total)
        upd = learner.update(progress_remaining)
        t_update = time.perf_counter()
        num_timesteps += rollout_size
        fps = rollout_size / max(t_update - t_it, 1e-8)

        # 进度 / 指标打印 (每 1%)
        pct = num_timesteps / total * 100
        if pct - last_print_pct >= 1.0 or iteration <= 3:
            last_print_pct = pct
            print(f"  [进度] {pct:.1f}% ({num_timesteps:,}/{total:,}) | "
                  f"fps={fps:,.0f} (collect {t_collect-t_it:.1f}s / "
                  f"update {t_update-t_collect:.1f}s) | "
                  f"ep_rew={roll_m['ep_rew_mean']:.2f} | "
                  f"pg={upd['pg_loss']:.4f} vf={upd['vf_loss']:.4f} "
                  f"std={upd['std']:.3f}")
            if roll_m.get('failure_resets', 0):
                print(
                    f"           failure curriculum: "
                    f"{roll_m['failure_resets']:,} resets in this rollout")
            if roll_m.get('backtrace_collision_events', 0):
                print(
                    f"           collision backtrace: "
                    f"events={roll_m['backtrace_collision_events']:,} "
                    f"penalty_sum={roll_m['backtrace_penalty_sum']:,.1f}")
            if tc.constrained_ppo:
                print(
                    f"           safety: obstacle episodes="
                    f"{roll_m['obstacle_episodes']} collisions="
                    f"{roll_m['collision_count']} rate="
                    f"{roll_m['obstacle_collision_rate']*100:.2f}% | "
                    f"limit={upd['cost_limit']*100:.2f}% "
                    f"lambda={upd['lagrange']:.3f} "
                    f"dense_cost={roll_m['dense_safety_cost_sum']:.1f} "
                    f"cost_vf={upd['cost_vf_loss']:.4f}")

        # 课程推进 (复刻 ObstacleCurriculumCallback)
        new_stage = _curriculum_stage(cfg.obstacle.stages,
                                      num_timesteps / total, start_stage)
        if new_stage != cur_stage:
            cur_stage = new_stage
            s = cfg.obstacle.stages[cur_stage]
            env.env_method('set_obstacle_stage', cur_stage)
            print(f"\n  [障碍物课程] 阶段 {cur_stage+1}/"
                  f"{len(cfg.obstacle.stages)}: {s.name} "
                  f"(障碍数 {s.n_min}~{s.n_max}, "
                  f"侧向距离 {s.dist_min:.2f}~{s.dist_max:.2f}m, "
                  f"间隙 {s.clearance:.2f}m)")

        # 周期评测 (整条轨迹安全成功率均值选 best)
        if iteration % eval_freq == 0:
            r_obs, r_free = eval_set.run(learner)
            s_obs = summarize(r_obs, 'eval_obstacles')['strict_success_rate']
            s_free = summarize(r_free, 'eval_free')['strict_success_rate']
            # 目标是两组同时达标；取较小值可避免用一组的高分掩盖另一组遗忘。
            metric = min(s_obs, s_free)
            print(f"\n  [MJX评测] @{num_timesteps:,} 环境步: "
                  f"障碍轨迹安全 {s_obs*100:.1f}% | "
                  f"无障碍轨迹安全 {s_free*100:.1f}%")
            if metric > best_metric:
                best_metric = metric
                os.makedirs(best_dir, exist_ok=True)
                learner.export_zip(os.path.join(best_dir, 'best_model'))
                print(f"  [MJX评测] 新 best 模型 (双组较小值 {metric*100:.1f}%) "
                      f"-> {best_dir}")

        # 大样本抽查 (不参与 best 选择)
        if spot_set is not None and iteration % spot_freq == 0:
            r_obs, r_free = spot_set.run(learner)
            s_obs = summarize(r_obs, 'spot_obstacles')
            s_free = summarize(r_free, 'spot_free')
            print(f"\n  [抽查 n={tc.spot_eval_n}] @{num_timesteps:,} 步: "
                  f"有障碍轨迹安全 {s_obs['strict_success_rate']*100:.1f}% "
                  f"(轨迹碰撞 {s_obs['trajectory_collision_rate']*100:.1f}%) | "
                  f"无障碍轨迹安全 {s_free['strict_success_rate']*100:.1f}%")

        # 检查点 (每 checkpoint_freq_steps; .meta.pkl 存归一化+步数供中断续训)
        if iteration % ckpt_freq == 0:
            ckpt = os.path.join(
                ckpt_dir, f'gpu_{model_name}_{num_timesteps}_steps.pt')
            learner.save(ckpt)
            # Every checkpoint must be directly evaluable/deployable.  The
            # zip contains the actor/value policy only; training-only safety
            # critic weights (when present) stay in the .pt checkpoint.
            learner.export_zip(os.path.splitext(ckpt)[0])
            with open(ckpt + '.meta.pkl', 'wb') as f:
                pickle.dump({'ret': normalizer.state_dict(),
                             'num_timesteps': num_timesteps}, f)

    # ---------- 最终保存: zip (推理) + 自研 .pt + 归一化统计 ----------
    learner.export_zip(model_path)
    learner.save(model_path + '_gpu.pt')
    with open(model_path + '_norm.pkl', 'wb') as f:
        pickle.dump({'ret': normalizer.state_dict(),
                     'num_timesteps': num_timesteps}, f)
    elapsed = time.perf_counter() - t_start
    print(f"\n  最终模型已保存: {model_path}.zip (推理) / "
          f"{model_path}_gpu.pt (自研)")
    print(f"  归一化统计已保存: {model_path}_norm.pkl")
    print(f"  训练完成: {num_timesteps:,} 步, 耗时 {elapsed/60:.1f} 分钟, "
          f"平均 {num_timesteps/elapsed:,.0f} FPS")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU驻留 PPO 无碰撞 IK 训练')
    parser.add_argument('--config', type=str, required=True,
                        help='Explicit training configuration; see code/README.md')
    parser.add_argument('--timesteps', type=int, default=0,
                        help='覆盖训练总步数')
    parser.add_argument('--model-name', type=str, default='',
                        help='覆盖输出模型名（烟雾/消融实验使用）')
    parser.add_argument('--n_worlds', type=int, default=2048,
                        help='MJX 并行世界数')
    parser.add_argument('--n_steps', type=int, default=64,
                        help='PPO rollout 长度 (步)')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='PPO minibatch (默认取配置)')
    parser.add_argument('--device', type=str, default='auto',
                        help='cpu / cuda / auto')
    parser.add_argument('--start-stage', type=int, default=0,
                        help='起始障碍物课程阶段索引')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--no-resume', action='store_true',
                        help='忽略已有模型, 从头训练')
    parser.add_argument(
        '--init-from', type=str, default='',
        help='从已有 .zip/.pt 权重初始化新模型（不继承旧训练步数）')
    args = parser.parse_args()
    train(args)
