"""多个纯 PPO 模型在同一批场景上的轨迹安全成功率对比。

有障碍组严格为 1~3 个障碍，无障碍组严格为 0 个障碍。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from ik_avoid.config import load_config  # noqa: E402
from ik_avoid.eval_stats import inject_scenes, rollout_eval, summarize  # noqa: E402
from ik_avoid.policy_io import load_policy  # noqa: E402
from ik_avoid.ppo_gpu import GpuPPO  # noqa: E402
from mjx.env_mjx import RobotIKAvoidEnvMJX  # noqa: E402


def parse_model(spec):
    if '=' not in spec:
        raise argparse.ArgumentTypeError('--model 格式必须是 LABEL=PATH')
    label, path = spec.split('=', 1)
    return label, path


def breakdown(results):
    out = {}
    for n_obs in range(4):
        group = [r for r in results if r['n_obstacles'] == n_obs]
        if not group:
            continue
        safe = [
            r['position_error'] < 0.01
            and r['orientation_error'] < np.radians(10)
            and r.get('trajectory_collisions', r['n_collisions']) == 0
            for r in group]
        collision = [
            r.get('trajectory_collisions', r['n_collisions']) > 0
            for r in group]
        out[str(n_obs)] = {
            'n': len(group),
            'trajectory_safe_success_rate': float(np.mean(safe)),
            'trajectory_collision_rate': float(np.mean(collision)),
        }
    return out


def main(args):
    cfg = load_config(args.config)
    envs_scenes = {}
    for kind, with_obs in [('obstacles', True), ('free', False)]:
        if kind == 'free' and not args.include_free:
            continue
        env = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=args.n,
                                 max_points=args.max_points)
        scenes = env.generate_scenes(args.n, args.seed,
                                     with_obstacles=with_obs)
        envs_scenes[kind] = (env, scenes)

    pt_policy = None
    all_summaries = {}
    for label, raw_path in args.model:
        path = os.path.normpath(raw_path)
        if path.lower().endswith('.pt'):
            if pt_policy is None:
                env = envs_scenes['obstacles'][0]
                pt_policy = GpuPPO(
                    obs_dim=env.obs_dim,
                    act_dim=env.n_ctrl + int(cfg.env.learned_step_scale),
                    arch=tuple(cfg.train.net_arch), device=args.device,
                    n_steps=1, n_envs=1, batch_size=1, n_epochs=1)
            pt_policy.load(path)
            policy = pt_policy
        else:
            policy = load_policy(path)

        print(f'\n[{label}] {path}', flush=True)
        model_summary = {}
        for kind, (env, scenes) in envs_scenes.items():
            inject_scenes(env, scenes)
            results = rollout_eval(env, policy, scenes, env.max_steps)
            stat = summarize(results, kind)
            model_summary[kind] = stat
            if kind == 'obstacles':
                model_summary['obstacle_count_breakdown'] = breakdown(results)
            print(
                f"  {kind}: 轨迹安全={stat['trajectory_safe_success_rate']:.1%} "
                f"碰撞={stat['trajectory_collision_rate']:.1%} "
                f"迭代={stat['iterations_mean']:.1f}", flush=True)
        all_summaries[label] = model_summary

    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(BASE, 'logs', f'eval_pure_models_{stamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'args': {
                'config': args.config, 'n': args.n, 'seed': args.seed,
                'include_free': args.include_free,
                'models': dict(args.model),
            },
            'results': all_summaries,
        }, f, ensure_ascii=False, indent=1)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', action='append', type=parse_model,
                        required=True, help='LABEL=PATH，可重复')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--include-free', action='store_true')
    parser.add_argument('--max-points', type=int, default=768)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    main(parser.parse_args())
