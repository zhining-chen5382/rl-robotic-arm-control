"""RL + DLS 混合推理评测: RL 粗调 (避障) + 碰撞安全 DLS 精调 (精度)

流程 (与纯 RL 评测同场景集、同 rollout, 单变量只加精调段):
    1. MJX 批量 rollout → RL best-so-far 结果 (与 eval_mjx.py 完全同口径)
    2. 逐案取 RL best 构型, 注入同一障碍场景, 跑 refine_safe_dls:
       - 每步候选构型过 SDF 碰撞复检, 穿入步长逐级缩小重试, 仍碰撞则弃步
       - 安全帧按任务误差 best-so-far, 保底不退化
    3. 采纳门控 (任一不满足则保留 RL 粗解):
       - 精调轨迹零碰撞
       - 有障碍时终态 min_dist >= success_margin
       - 耦合距离 (pos×(2−cosθ)+(1−cosθ)) 严格下降

用法:
    python scripts/eval_hybrid_dls.py --config configs/avoid_mjx_v13.yaml \
        --model models/panda_ppo_ik_avoid_v13_mjx_champion/champion_model \
        --n 1000 --seed 2026 --include-free
"""
import argparse
import copy
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

from mjx.env_mjx import RobotIKAvoidEnvMJX
from mjx.avoid_solvers import AvoidBaselineMJX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')


def coupled(pd, od_rad):
    """MAPPO-IK 耦合距离 (与 env._coupled_distance 同式)"""
    return pd * (2.0 - np.cos(od_rad)) + (1.0 - np.cos(od_rad))


def refine_results(rl_results, cfg, refine_iters):
    """对 RL 结果逐案叠加碰撞安全 DLS 精调, 返回混合结果列表与采纳统计"""
    bl = AvoidBaselineMJX(cfg)
    margin = cfg.env.success_margin
    hybrid = []
    n_accept = 0
    n_reject_coll = n_reject_margin = n_reject_noimp = 0
    t0 = time.perf_counter()
    for i, r in enumerate(rl_results):
        hr = copy.deepcopy(r)
        q0 = np.asarray(r['joint_angles'], dtype=np.float32)
        specs = r['obstacle_specs']
        has_obs = len(specs) > 0
        bl.set_scene(r['target_position'], r['target_orientation'],
                     obstacle_specs=specs, start_q=q0)
        res = bl.refine_safe_dls(start_q=q0, max_iterations=refine_iters)
        accept = False
        if res['trajectory_collisions'] > 0:
            n_reject_coll += 1
        elif has_obs and res['min_distance'] is not None \
                and res['min_distance'] < margin:
            n_reject_margin += 1
        else:
            c_old = coupled(r['position_error'], r['orientation_error'])
            c_new = coupled(res['position_error'], res['orientation_error'])
            if c_new < c_old:
                accept = True
            else:
                n_reject_noimp += 1
        if accept:
            n_accept += 1
            hr.update({
                'method': 'rl_dls_hybrid',
                'joint_angles': res['joint_angles'],
                'position_error': res['position_error'],
                'orientation_error': res['orientation_error'],
                'orientation_error_deg': res['orientation_error_deg'],
                'min_distance': res['min_distance'],
                'n_collisions': res['n_collisions'],
                'iterations': r['iterations'] + res['iterations'],
                'solve_time_ms': round(r['solve_time_ms']
                                       + res['solve_time_ms'], 2),
            })
        hybrid.append(hr)
        if (i + 1) % 100 == 0:
            dt = time.perf_counter() - t0
            print(f"    [精调] {i+1}/{len(rl_results)} "
                  f"({(i+1)/dt:.1f} 案/秒)")
    stats = {'n_accept': n_accept, 'n_reject_collision': n_reject_coll,
             'n_reject_margin': n_reject_margin,
             'n_reject_no_improve': n_reject_noimp}
    return hybrid, stats


def main(args):
    cfg = load_config(args.config)
    model_path = args.model or os.path.join(
        BASE_DIR, 'models', f"{cfg.train.model_name}_mjx_champion",
        'champion_model')
    print(f"  [Hybrid-Eval] 加载模型: {model_path}.zip")
    model = PPO.load(model_path, device='cpu')

    all_results = {}
    for scene_kind in (['obstacles'] + (['free'] if args.include_free else [])):
        with_obs = scene_kind == 'obstacles'
        print(f"\n{'='*60}\n  场景类型: {'随机障碍环境' if with_obs else '无障碍'}"
              f"  (n={args.n}, seed={args.seed})\n{'='*60}")
        env = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=args.n,
                                 max_points=args.max_points)
        scenes = env.generate_scenes(args.n, args.seed,
                                     with_obstacles=with_obs)
        inject_scenes(env, scenes)
        rl_results = rollout_eval(env, model, scenes, env.max_steps)
        print_summary(summarize(rl_results, f'rl_{scene_kind}'))

        print(f"\n  ---- RL + DLS 精调 (每案 ≤{args.refine_iters} 步) ----")
        hybrid, stats = refine_results(rl_results, cfg, args.refine_iters)
        print(f"  采纳 {stats['n_accept']}/{len(rl_results)} | 弃: "
              f"碰撞 {stats['n_reject_collision']} / "
              f"margin {stats['n_reject_margin']} / "
              f"无改善 {stats['n_reject_no_improve']}")
        print_summary(summarize(hybrid, f'rl_dls_hybrid_{scene_kind}'))
        all_results[f'rl_{scene_kind}'] = rl_results
        all_results[f'rl_dls_hybrid_{scene_kind}'] = hybrid
        all_results[f'hybrid_stats_{scene_kind}'] = stats

    os.makedirs(LOG_DIR, exist_ok=True)
    t_stamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(LOG_DIR, f'eval_hybrid_dls_{t_stamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'results': all_results}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n  评测日志已保存: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL + DLS 混合推理评测')
    parser.add_argument('--n', type=int, default=100, help='场景数量')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model', type=str, default=None,
                        help='RL 模型路径 (不含 .zip), 默认取 champion')
    parser.add_argument('--include-free', action='store_true',
                        help='同时评测无障碍场景')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--max_points', type=int, default=768)
    parser.add_argument('--refine-iters', type=int, default=100,
                        help='每案 DLS 精调最大迭代数')
    args = parser.parse_args()
    main(args)
