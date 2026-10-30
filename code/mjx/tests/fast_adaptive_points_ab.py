"""v3 NumPy 快速环境的闭环点云预算 A/B/C 基准。

固定同一模型、配置、场景和 NumPy 策略，只比较：
  fixed_low: 每步每组使用 --infer-min-points（默认 128）
  fixed256: 当前默认候选基线，每步每组最多 256 点
  adaptive: 远区 128、近区 256、极近区全量，最终轨迹仍全量复核

各轮轮换执行顺序，降低热漂移对 wall time 的影响。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ik_avoid.config import load_config
from mjx.env_mjx import RobotIKAvoidEnvMJX
from mjx.fast_inference import FastRLAvoidSolverMJX


def solve_one(solver, scene):
    t0 = time.perf_counter()
    result = solver.solve(
        np.asarray(scene['target_pos'], np.float32),
        np.asarray(scene['target_quat'], np.float32),
        obstacle_specs=scene['obstacle_specs'],
        start_q=np.asarray(scene['start_q'], np.float32),
        record_trajectory=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms, dict(solver.fenv.infer_tier_counts)


def strict(result):
    return (result['position_error'] < 0.01
            and result['orientation_error_deg'] < 10.0
            and result['n_collisions'] == 0)


def summarize(rows):
    results = [r for r, _, _ in rows]
    times = np.asarray([t for _, t, _ in rows], np.float64)
    tier_counts = {}
    for _, _, counts in rows:
        for tier, count in counts.items():
            tier_counts[tier] = tier_counts.get(tier, 0) + count
    tier_total = sum(tier_counts.values())
    return {
        'n': len(rows),
        'strict_success_rate': float(np.mean([strict(r) for r in results])),
        'trajectory_collision_rate': float(np.mean([
            r['trajectory_collisions'] > 0 for r in results])),
        'iterations_mean': float(np.mean([r['iterations'] for r in results])),
        'solve_ms_mean': float(times.mean()),
        'solve_ms_median': float(np.median(times)),
        'solve_ms_p95': float(np.percentile(times, 95)),
        'infer_tier_calls': tier_counts,
        'infer_tier_fraction': {
            key: value / tier_total for key, value in tier_counts.items()
        } if tier_total else {},
    }


def compare_to_fixed256(rows):
    base = rows['fixed256']
    report = {}
    for label, values in rows.items():
        if label == 'fixed256':
            continue
        report[label] = {
            'strict_outcome_mismatches': int(sum(
                strict(a[0]) != strict(b[0]) for a, b in zip(base, values))),
            'trajectory_collision_mismatches': int(sum(
                (a[0]['trajectory_collisions'] > 0)
                != (b[0]['trajectory_collisions'] > 0)
                for a, b in zip(base, values))),
            'max_joint_abs_diff': float(max(
                np.max(np.abs(np.asarray(a[0]['joint_angles'])
                              - np.asarray(b[0]['joint_angles'])))
                for a, b in zip(base, values))),
        }
    return report


def run_group(name, solvers, scenes, warmup, rounds):
    print(f"\n{'=' * 76}\n{name}: n={len(scenes) - warmup}\n{'=' * 76}")
    for label, solver in solvers.items():
        for scene in scenes[:warmup]:
            solve_one(solver, scene)
        print(f'  warmup: {label}', flush=True)

    pool = scenes[warmup:]
    if len(pool) % rounds:
        raise ValueError('计时场景数必须能被 rounds 整除')
    size = len(pool) // rounds
    labels = list(solvers)
    rows = {label: [] for label in labels}
    for round_index in range(rounds):
        batch = pool[round_index * size:(round_index + 1) * size]
        shift = round_index % len(labels)
        order = labels[shift:] + labels[:shift]
        for label in order:
            for scene in batch:
                rows[label].append(solve_one(solvers[label], scene))
        print(f"  round {round_index + 1}/{rounds}: {' -> '.join(order)}",
              flush=True)

    report = {label: summarize(value) for label, value in rows.items()}
    report['comparison_to_fixed256'] = compare_to_fixed256(rows)
    for label in labels:
        stat = report[label]
        tiers = ', '.join(
            f'{key}={100*value:.1f}%'
            for key, value in stat['infer_tier_fraction'].items())
        print(f"  {label:10s} strict={100*stat['strict_success_rate']:.1f}% "
              f"collision={100*stat['trajectory_collision_rate']:.1f}% "
              f"iters={stat['iterations_mean']:.1f} "
              f"mean={stat['solve_ms_mean']:.1f}ms "
              f"median={stat['solve_ms_median']:.1f}ms "
              f"p95={stat['solve_ms_p95']:.1f}ms tiers[{tiers}]")
    print(f"  outcome comparison: {report['comparison_to_fixed256']}")
    return report


def main(args):
    cfg = load_config(args.config)
    generator = RobotIKAvoidEnvMJX(cfg, n_worlds=1)
    obstacles = generator.generate_scenes(
        args.obstacles + args.warmup, args.seed, True)
    free = generator.generate_scenes(
        args.free + args.warmup, args.seed + 1, False)

    common = dict(cfg=cfg, model_path=args.model, policy_backend='numpy')
    solvers = {
        'fixed_low': FastRLAvoidSolverMJX(
            **common, infer_max_points=args.infer_min_points),
        'fixed256': FastRLAvoidSolverMJX(
            **common, infer_max_points=args.infer_max_points),
        'adaptive': FastRLAvoidSolverMJX(
            **common, infer_max_points=args.infer_max_points,
            adaptive_infer_points=True,
            infer_min_points=args.infer_min_points,
            infer_near_distance=args.infer_near_distance,
            infer_full_distance=args.infer_full_distance),
    }
    report = {
        'args': vars(args),
        'obstacles': run_group(
            'OBSTACLES', solvers, obstacles, args.warmup, args.rounds),
        'free': run_group(
            'FREE', solvers, free, args.warmup, args.rounds),
    }
    os.makedirs('logs', exist_ok=True)
    out = os.path.join(
        'logs', f'fast_adaptive_points_ab_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(out, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f'\n日志: {out}')


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
    parser.add_argument('--infer-min-points', type=int, default=128)
    parser.add_argument('--infer-max-points', type=int, default=256)
    parser.add_argument('--infer-near-distance', type=float, default=0.12)
    parser.add_argument('--infer-full-distance', type=float, default=0.04)
    main(parser.parse_args())
