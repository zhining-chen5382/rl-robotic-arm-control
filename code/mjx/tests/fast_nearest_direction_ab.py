"""v3 NumPy SDF 全点方向 vs 仅最近点方向的完整求解 A/B。"""

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


def trajectory_safe_strict(result):
    """最终位姿达标、终态无碰撞，且整条轨迹没有碰撞帧。"""
    return strict(result) and result['trajectory_collisions'] == 0


def stats(rows):
    results = [result for result, _ in rows]
    times = np.asarray([elapsed for _, elapsed in rows], np.float64)
    return {
        'n': len(rows),
        'strict_success_rate': float(np.mean([strict(r) for r in results])),
        'trajectory_safe_success_rate': float(np.mean([
            trajectory_safe_strict(r) for r in results])),
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
        report[f'n_obs_{n_obs}'] = stats([
            row for row, count in zip(rows, counts) if count == n_obs])
    return report


def compare_exact(baseline, optimized):
    maxima = {'joint': 0.0, 'trajectory': 0.0,
              'position_error': 0.0, 'orientation_error': 0.0}
    mismatches = {'iterations': 0, 'collision': 0,
                  'trajectory_length': 0, 'strict_outcome': 0}
    for (left, _), (right, _) in zip(baseline, optimized):
        maxima['joint'] = max(maxima['joint'], float(np.max(np.abs(
            np.asarray(left['joint_angles']) - np.asarray(right['joint_angles'])))))
        maxima['position_error'] = max(
            maxima['position_error'],
            abs(left['position_error'] - right['position_error']))
        maxima['orientation_error'] = max(
            maxima['orientation_error'],
            abs(left['orientation_error'] - right['orientation_error']))
        if len(left['trajectory']) == len(right['trajectory']):
            maxima['trajectory'] = max(maxima['trajectory'], float(np.max(np.abs(
                np.asarray(left['trajectory']) - np.asarray(right['trajectory'])))))
        else:
            mismatches['trajectory_length'] += 1
        mismatches['iterations'] += left['iterations'] != right['iterations']
        mismatches['collision'] += (
            left['n_collisions'], left['trajectory_collisions']) != (
                right['n_collisions'], right['trajectory_collisions'])
        mismatches['strict_outcome'] += strict(left) != strict(right)
    return {'max_abs_diff': maxima, 'mismatches': mismatches}


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
        rows['all_point_directions'], rows['nearest_directions'])
    for label in labels:
        overall = report[label]['all']
        print(f"  {label:21s} strict={100*overall['strict_success_rate']:.1f}% "
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
        skip_inactive_obstacle_slots=True)
    solvers = {
        'all_point_directions': FastRLAvoidSolverMJX(
            **common, defer_nearest_direction=False),
        'nearest_directions': FastRLAvoidSolverMJX(
            **common, defer_nearest_direction=True),
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
        'logs', f'fast_nearest_direction_ab_{time.strftime("%Y%m%d_%H%M%S")}.json')
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
