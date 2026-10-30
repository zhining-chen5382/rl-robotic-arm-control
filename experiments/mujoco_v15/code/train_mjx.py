"""MJX (GPU 向量化) 无碰撞 IK PPO 训练脚本

与 train_avoid.py 同构 (课程学习 + VecNormalize + 周期评测),
训练环境替换为 MJX 批量向量化实现 (单进程 GPU, n_worlds 并行):
    - step_batch 一次推进全部世界 (~70k FPS 裸吞吐)
    - done 世界自动批量 reset (GPU 采样, 分桶定形)
评测用 MJX 批量评测回调 (MJXBatchEvalCallback, 无 PyBullet 依赖),
按严格成功率选 best 模型。

用法:
    python train_mjx.py                                   # 默认 2048 世界
    python train_mjx.py --n_worlds 4096 --timesteps 30000000
    python train_mjx.py --config configs/avoid_remote30m.yaml --start-stage 0
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from ik_avoid.config import load_config, save_config
from ik_avoid.callbacks import (linear_schedule, ProgressCallback,
                                ObstacleCurriculumCallback,
                                MJXBatchEvalCallback, MJXSpotEvalCallback)
from mjx.vec_env import MJXAvoidVecEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')


def train(args):
    import torch
    # 云平台 cgroup 限核时, 默认线程数(=物理核)会超额竞争反而拖慢小网络 PPO 更新;
    # 可用 TORCH_NUM_THREADS 环境变量显式限制 (如配额 14 核 -> 14)
    _n_threads = int(os.environ.get('TORCH_NUM_THREADS', '0'))
    if _n_threads > 0:
        torch.set_num_threads(_n_threads)
    cfg = load_config(args.config)
    tc = cfg.train

    if args.timesteps:
        tc.timesteps = args.timesteps
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
    best_path = os.path.join(MODELS_DIR, f'{model_name}_best')

    print("=" * 62)
    print("  无碰撞 IK 训练 (PPO + MJX GPU 向量化环境)")
    print(f"  总步数: {tc.timesteps:,}  |  并行世界: {n_worlds}  |  设备: {device}")
    print(f"  n_steps={n_steps}  batch_size={batch_size}  "
          f"(rollout={n_steps*n_worlds:,})")
    print(f"  课程阶段: {' -> '.join(s.name for s in cfg.obstacle.stages)}")
    print(f"  模型保存: {model_path}.zip")
    print("=" * 62)

    save_config(cfg, os.path.join(LOG_DIR, f'{model_name}_config.yaml'))

    env = MJXAvoidVecEnv(cfg, n_worlds=n_worlds, stage=args.start_stage,
                         seed=args.seed)
    print(f"  训练环境: MJXAvoidVecEnv ({n_worlds} 世界, "
          f"起始阶段 {args.start_stage})")
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True,
                       clip_reward=tc.clip_reward, gamma=tc.gamma)
    vec_normalize = env

    # 评测: MJX 批量评测回调 (固定场景集, 严格成功率选 best)
    eval_callback = MJXBatchEvalCallback(
        cfg,
        eval_freq=max(tc.eval_freq_steps // n_worlds, 1),
        n_eval_episodes=tc.n_eval_episodes,
        best_model_save_path=best_path,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(tc.checkpoint_freq_steps // n_worlds, 1),
        save_path=os.path.join(MODELS_DIR, 'checkpoints'),
        name_prefix=f'ppo_{model_name}',
        verbose=1,
    )
    # 大样本抽查 (低频, 供 go/no-go 决策; 不参与 best 选择)
    spot_callback = None
    if tc.spot_eval_n > 0:
        spot_callback = MJXSpotEvalCallback(
            cfg, spot_n=tc.spot_eval_n,
            eval_freq=max(tc.spot_eval_freq_steps // n_worlds, 1),
            verbose=1,
        )
        print(f"  抽查评测: 每 {tc.spot_eval_freq_steps:,} 步, "
              f"n={tc.spot_eval_n} (障碍/无障碍各半)")

    resume = os.path.exists(model_path + '.zip') and not args.no_resume
    if resume:
        print(f"\n  发现已有模型, 从 {model_path}.zip 继续训练...")
        model = PPO.load(model_path, env=env, verbose=1, device=device,
                         tensorboard_log=LOG_DIR)
        norm_path = model_path + '_norm.pkl'
        if os.path.exists(norm_path):
            loaded = VecNormalize.load(norm_path, env.venv)
            vec_normalize.ret_rms = loaded.ret_rms
            print(f"  已恢复归一化统计: {norm_path}")
    else:
        net_arch = dict(pi=list(tc.net_arch), vf=list(tc.net_arch))
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            learning_rate=linear_schedule(tc.learning_rate, tc.lr_final),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=tc.n_epochs,
            gamma=tc.gamma,
            gae_lambda=tc.gae_lambda,
            clip_range=tc.clip_range,
            ent_coef=tc.ent_coef,
            device=device,
            policy_kwargs=dict(net_arch=net_arch, ortho_init=True),
            tensorboard_log=LOG_DIR,
        )

    print("\n  开始训练...\n")
    callbacks = [
        eval_callback, checkpoint_callback,
        ProgressCallback(tc.timesteps),
        ObstacleCurriculumCallback(tc.timesteps, cfg.obstacle.stages,
                                   start_stage=args.start_stage),
    ]
    if spot_callback is not None:
        callbacks.append(spot_callback)
    model.learn(
        total_timesteps=tc.timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=True,
    )

    model.save(model_path)
    print(f"\n  最终模型已保存: {model_path}.zip")
    vec_normalize.save(model_path + '_norm.pkl')
    print(f"  归一化统计已保存: {model_path}_norm.pkl")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MJX 无碰撞 IK PPO 训练')
    parser.add_argument('--config', type=str,
                        default='configs/avoid_remote30m.yaml',
                        help='配置文件路径')
    parser.add_argument('--timesteps', type=int, default=0,
                        help='覆盖训练总步数')
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
    args = parser.parse_args()
    train(args)
