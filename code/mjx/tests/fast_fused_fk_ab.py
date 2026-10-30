"""v3 NumPy 数组式 FK vs 分量化融合 FK 的完整求解 A/B。"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ik_avoid.config import load_config
from mjx.env_mjx import RobotIKAvoidEnvMJX
from mjx.fast_inference import FastRLAvoidSolverMJX
from mjx.tests.fast_nearest_direction_ab import (
    compare_exact, solve_one, summarize,
)


def run_group(name, solvers, scenes, warmup, rounds):
    print(f"\n{'=' * 76}\n{name}: n={len(scenes) - warmup}\n{'=' * 76}")
    for label, solver in solvers.items():
        for scene in scenes[:warmup]:
            solve_one(solver, scene)
        print(f'  warmup: {label}', flush=True)
    pool = scenes[warmup:]
    if len(pool) % rounds:
        raise ValueError('计时场景数必须能被 rounds 整除')
    batch_size = len(pool) // rounds
    labels = list(solvers)
    rows = {label: [] for label in labels}
    for round_index in range(rounds):
        batch = pool[round_index * batch_size:(round_index + 1) * batch_size]
        order = labels[round_index % 2:] + labels[:round_index % 2]
        for label in order:
            for scene in batch:
                rows[label].append(solve_one(solvers[label], scene))
        print(f"  round {round_index + 1}/{rounds}: {' -> '.join(order)}",
              flush=True)

    report = {label: summarize(value, pool) for label, value in rows.items()}
    report['exact_comparison'] = compare_exact(
        rows['array_fk'], rows['fused_fk'])
    for label in labels:
        overall = report[label]['all']
        print(f"  {label:8s} strict={100*overall['strict_success_rate']:.1f}% "
              f"collision={100*overall['trajectory_collision_rate']:.1f}% "
              f"iters={overall['iterations_mean']:.1f} "
              f"mean={overall['solve_ms_mean']:.1f}ms "
              f"median={overall['solve_ms_median']:.1f}ms "
              f"p95={overall['solve_ms_p95']:.1f}ms")
        split = ', '.join(
            f"{key[6:]}obs={value['solve_ms_mean']:.1f}ms"
            for key, value in report[label].items() if key.startswith('n_obs_'))
        print(f'    by obstacle count: {split}')
    print(f"  exact comparison: {report['exact_comparison']}")
    return report


def main(args):
    cfg = load_config(args.config)
    generator = RobotIKAvoidEnvMJX(cfg, n_worlds=1)
    obstacle_scenes = generator.generate_scenes(
        args.obstacles + args.warmup, args.seed, True)
    free_scenes = generator.generate_scenes(
        args.free + args.warmup, args.seed + 1, False)
    common = dict(
        cfg=cfg, model_path=args.model, policy_backend='numpy',
        infer_max_points=args.infer_max_points,
        skip_inactive_obstacle_slots=True,
        defer_nearest_direction=True,
        reuse_sdf_buffers=False,
        fast_quat_rotate=True)
    solvers = {
        'array_fk': FastRLAvoidSolverMJX(**common, fused_fk=False),
        'fused_fk': FastRLAvoidSolverMJX(**common, fused_fk=True),
    }
    report = {
        'args': vars(args),
        'obstacles': run_group(
            'OBSTACLES', solvers, obstacle_scenes, args.warmup, args.rounds),
        'free': run_group(
            'FREE', solvers, free_scenes, args.warmup, args.rounds),
    }
    os.makedirs('logs', exist_ok=True)
    output = os.path.join(
        'logs', f'fast_fused_fk_ab_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f'\n日志: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/avoid_mjx_v16.yaml')
    parser.add_argument('--model', default=(
        'models/panda_ppo_ik_avoid_v16_mjx_champion/champion_model'))
    parser.add_argument('--obstacles', type=int, default=90)
    parser.add_argument('--free', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--infer-max-points', type=int, default=256)
    main(parser.parse_args())
