"""v3 NumPy SDF 固定三插槽 vs 仅有效障碍插槽的完整求解 A/B。

模型、配置、场景、点云和策略均相同；各轮交换执行顺序。除总体统计外，
按场景障碍物数量分别报告时耗，并检查逐案关节、误差、迭代和碰撞结果。
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
    start = time.perf_counter()
    result = solver.solve(
        np.asarray(scene['target_pos'], np.float32),
        np.asarray(scene['target_quat'], np.float32),
        obstacle_specs=scene['obstacle_specs'],
        start_q=np.asarray(scene['start_q'], np.float32),
        record_trajectory=True)
    return result, (time.perf_counter() - start) * 1000.0


def strict(result):
    return (result['position_error'] < 0.01
            and result['orientation_error_deg'] < 10.0
            and result['n_collisions'] == 0)


def stats(rows):
    results = [result for result, _ in rows]
    times = np.asarray([elapsed for _, elapsed in rows], np.float64)
    return {
        'n': len(rows),
        'strict_success_rate': float(np.mean([strict(r) for r in results])),
        'trajectory_collision_rate': float(np.mean([
            r['trajectory_collisions'] > 0 for r in results])),
        'iterations_mean': float(np.mean([r['iterations'] for r in results])),
        'solve_ms_mean': float(times.mean()),
        'solve_ms_median': float(np.median(times)),
        'solve_ms_p95': float(np.percentile(times, 95)),
    }


def summarize(rows, scenes):
    report = {'all': stats(rows)}
    counts = np.asarray([len(scene['obstacle_specs']) for scene in scenes])
    for n_obs in sorted(set(counts.tolist())):
        selected = [row for row, count in zip(rows, counts) if count == n_obs]
        report[f'n_obs_{n_obs}'] = stats(selected)
    return report


def compare_exact(baseline, active):
    joint_diff = []
    pos_diff = []
    orient_diff = []
    iterations_mismatch = 0
    collision_mismatch = 0
    trajectory_length_mismatch = 0
    for (left, _), (right, _) in zip(baseline, active):
        joint_diff.append(np.max(np.abs(
            np.asarray(left['joint_angles']) - np.asarray(right['joint_angles']))))
        pos_diff.append(abs(left['position_error'] - right['position_error']))
        orient_diff.append(abs(
            left['orientation_error'] - right['orientation_error']))
        iterations_mismatch += left['iterations'] != right['iterations']
        collision_mismatch += (
            left['n_collisions'], left['trajectory_collisions']) != (
                right['n_collisions'], right['trajectory_collisions'])
        trajectory_length_mismatch += len(left['trajectory']) != len(right['trajectory'])
    return {
        'max_joint_abs_diff': float(max(joint_diff, default=0.0)),
        'max_position_error_diff': float(max(pos_diff, default=0.0)),
        'max_orientation_error_diff': float(max(orient_diff, default=0.0)),
        'iterations_mismatches': int(iterations_mismatch),
        'collision_mismatches': int(collision_mismatch),
        'trajectory_length_mismatches': int(trajectory_length_mismatch),
    }


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

    report = {
        label: summarize(value, pool) for label, value in rows.items()
    }
    report['exact_comparison'] = compare_exact(
        rows['fixed_three_slots'], rows['active_slots'])
    for label in labels:
        overall = report[label]['all']
        print(f"  {label:17s} strict={100*overall['strict_success_rate']:.1f}% "
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
        infer_max_points=args.infer_max_points)
    solvers = {
        'fixed_three_slots': FastRLAvoidSolverMJX(
            **common, skip_inactive_obstacle_slots=False),
        'active_slots': FastRLAvoidSolverMJX(
            **common, skip_inactive_obstacle_slots=True),
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
        'logs', f'fast_active_slots_ab_{time.strftime("%Y%m%d_%H%M%S")}.json')
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
