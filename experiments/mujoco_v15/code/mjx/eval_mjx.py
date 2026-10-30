"""MJX GPU 批量评测 (与 ik_avoid.eval_stats 统一口径)

n 个场景一次并行 rollout (场景生成与汇总统计见 ik_avoid.eval_stats),
复刻 RLAvoidSolver 评测口径:
  - best-so-far: 非碰撞帧按耦合距离记录历史最优构型
  - 碰撞帧立即终止该场景; 成功帧终止
  - success = pos<tol & ori<tol & 无碰撞 & (无障碍 or min_dist>=margin)

场景由 MJX 环境原生生成 (env.generate_scenes, 无 PyBullet 依赖),
与 PyBullet 版 generate_scenes 同构 (同 seed 下场景分布一致)。

用法:
    python mjx/eval_mjx.py --n 1000 --model models/panda_ppo_ik_avoid_mjx_best/best_model
    python mjx/eval_mjx.py --n 100 --include-free --max_points 384
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ik_avoid.config import load_config
from ik_avoid.eval_stats import (inject_scenes, rollout_eval,
                                 summarize, print_summary)
from stable_baselines3 import PPO
from stable_baselines3.common.utils import FloatSchedule

from mjx.env_mjx import RobotIKAvoidEnvMJX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')


def main(args):
    config_path = args.config
    if config_path is None:
        # 默认模型为 v3 best (49 维), 必须配套 avoid_remote30m.yaml;
        # 直接走 load_config() 会落到 avoid.yaml (57 维含手指) 导致 shape 报错
        cand = os.path.join(BASE_DIR, 'configs', 'avoid_remote30m.yaml')
        config_path = cand if os.path.exists(cand) else None
    cfg = load_config(config_path)
    model_path = args.model or os.path.join(
        BASE_DIR, 'models', f"{cfg.train.model_name}_mjx_best", 'best_model')
    print(f"  [MJX-Eval] 加载模型: {model_path}.zip")
    # 评测只做推理，不恢复训练机序列化的学习率闭包，避免模型由不同
    # Python 小版本训练时在本机反序列化发生原生崩溃。
    inference_lr = float(cfg.train.learning_rate)
    model = PPO.load(
        model_path,
        device='cpu',
        custom_objects={
            'learning_rate': inference_lr,
            'lr_schedule': FloatSchedule(inference_lr),
        },
    )

    all_results = {}
    for scene_kind in (['obstacles'] + (['free'] if args.include_free else [])):
        with_obs = scene_kind == 'obstacles'
        print(f"\n{'='*60}\n  场景类型: {'随机障碍环境' if with_obs else '无障碍'}"
              f"  (n={args.n}, seed={args.seed})\n{'='*60}")
        t_env = time.perf_counter()
        env = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=args.n,
                                 max_points=args.max_points)
        print(f"  MJX 环境构建: {time.perf_counter()-t_env:.1f}s")

        t_gen = time.perf_counter()
        scenes = env.generate_scenes(args.n, args.seed,
                                     with_obstacles=with_obs)
        print(f"  场景生成 (MJX 原生): {time.perf_counter()-t_gen:.1f}s")

        t_inj = time.perf_counter()
        inject_scenes(env, scenes)
        print(f"  场景注入: {time.perf_counter()-t_inj:.1f}s")

        t_solve = time.perf_counter()
        results = rollout_eval(env, model, scenes, env.max_steps)
        dt = time.perf_counter() - t_solve
        print(f"  批量 rollout: {dt:.2f}s "
              f"({args.n*env.max_steps/dt:,.0f} 步/秒)")
        key = f'rl_{scene_kind}'
        all_results[key] = results
        print_summary(summarize(results, key))

    os.makedirs(LOG_DIR, exist_ok=True)
    t_stamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(LOG_DIR, f'eval_mjx_{t_stamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'results': all_results}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n  评测日志已保存: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MJX 批量评测')
    parser.add_argument('--n', type=int, default=100, help='场景数量')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model', type=str, default=None,
                        help='RL 模型路径 (不含 .zip)')
    parser.add_argument('--include-free', action='store_true',
                        help='同时评测无障碍场景')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--max_points', type=int, default=768,
                        help='每 link 点云上限 (评测建议与训练一致)')
    args = parser.parse_args()
    main(args)
