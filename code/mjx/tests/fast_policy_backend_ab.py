"""v3 NumPy 快速环境上的策略后端 A/B/C 完整求解基准。

固定同一个模型、配置、场景与几何后端，只切换策略前向:
numpy / ONNX FP32 / ONNX INT8。采用轮次交替顺序抵消热漂移。
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
    return result, (time.perf_counter() - t0) * 1000.0


def strict(result):
    return (result['position_error'] < 0.01
            and result['orientation_error_deg'] < 10.0
            and result['n_collisions'] == 0)


def summarize(rows):
    results = [r for r, _ in rows]
    times = np.asarray([t for _, t in rows], np.float64)
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


def run_group(name, solvers, scenes, warmup, rounds):
    print(f"\n{'=' * 72}\n{name}: n={len(scenes) - warmup}\n{'=' * 72}")
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
    for r in range(rounds):
        batch = pool[r * size:(r + 1) * size]
        shift = r % len(labels)
        order = labels[shift:] + labels[:shift]
        for label in order:
            for scene in batch:
                rows[label].append(solve_one(solvers[label], scene))
        print(f"  round {r + 1}/{rounds}: {' -> '.join(order)}", flush=True)
    report = {label: summarize(value) for label, value in rows.items()}
    for label, stat in report.items():
        print(f"  {label:10s} strict={100*stat['strict_success_rate']:.1f}% "
              f"collision={100*stat['trajectory_collision_rate']:.1f}% "
              f"iters={stat['iterations_mean']:.1f} "
              f"mean={stat['solve_ms_mean']:.1f}ms "
              f"median={stat['solve_ms_median']:.1f}ms "
              f"p95={stat['solve_ms_p95']:.1f}ms")
    return report


def main(args):
    cfg = load_config(args.config)
    gen = RobotIKAvoidEnvMJX(cfg, n_worlds=1)
    obs = gen.generate_scenes(args.obstacles + args.warmup, args.seed, True)
    free = gen.generate_scenes(args.free + args.warmup, args.seed + 1, False)
    solvers = {
        backend: FastRLAvoidSolverMJX(
            cfg, model_path=args.model,
            infer_max_points=args.infer_max_points,
            policy_backend=backend)
        for backend in ('numpy', 'onnx_fp32', 'onnx_int8')
    }
    report = {
        'args': vars(args),
        'obstacles': run_group('OBSTACLES', solvers, obs,
                               args.warmup, args.rounds),
        'free': run_group('FREE', solvers, free,
                          args.warmup, args.rounds),
    }
    out = os.path.join('logs', f'fast_policy_backend_ab_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n日志: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/avoid_mjx_v16.yaml')
    parser.add_argument('--model', default=(
        'models/panda_ppo_ik_avoid_v16_mjx_champion/champion_model'))
    parser.add_argument('--obstacles', type=int, default=90)
    parser.add_argument('--free', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=6,
                        help='建议取 3 的倍数，使三种后端轮转顺序完全平衡')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--infer-max-points', type=int, default=256)
    main(parser.parse_args())
