"""Diagnose extended pure-RL rollout horizons with a fixed time feature."""
import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from ik_avoid.config import load_config  # noqa: E402
from ik_avoid.eval_stats import summarize  # noqa: E402
from mjx.env_mjx import RobotIKAvoidEnvMJX  # noqa: E402
from mjx.fast_inference import FastRLAvoidSolverMJX  # noqa: E402


def parse_horizons(text):
    values = [int(x) for x in text.split(',')]
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError('horizons must be positive integers')
    return values


def solve_scene(solver, scene):
    return solver.solve(
        scene['target_pos'], scene['target_quat'],
        obstacle_specs=scene['obstacle_specs'],
        start_q=np.asarray(scene['start_q'], np.float32),
        record_trajectory=True)


def main(args):
    generator = RobotIKAvoidEnvMJX(
        load_config(args.config), n_worlds=1, max_points=args.max_points)
    scenes = {
        'obstacles': generator.generate_scenes(
            args.n, args.seed, with_obstacles=True),
        'free': generator.generate_scenes(
            args.n, args.seed, with_obstacles=False),
    }
    solvers = {}
    for horizon in args.horizons:
        cfg = load_config(args.config)
        cfg.env.max_action_span = args.action_span
        label = f'horizon_{horizon}'
        solvers[label] = FastRLAvoidSolverMJX(
            cfg, model_path=args.model, max_steps=horizon,
            observation_max_steps=args.observation_horizon,
            infer_max_points=args.infer_max_points,
            policy_backend='numpy', use_dls_refine=False)

    for solver in solvers.values():
        solve_scene(solver, scenes['obstacles'][0])
    labels = list(solvers)
    results = {label: {kind: [] for kind in scenes} for label in labels}
    total = len(labels) * sum(len(x) for x in scenes.values())
    completed = 0
    start = time.perf_counter()
    for kind, group in scenes.items():
        for i, scene in enumerate(group):
            order = labels[i % len(labels):] + labels[:i % len(labels)]
            for label in order:
                results[label][kind].append(
                    solve_scene(solvers[label], scene))
                completed += 1
            if (i + 1) % args.progress_every == 0 or i + 1 == len(group):
                print(f'[{kind}] {i + 1}/{len(group)}; '
                      f'{completed}/{total}; '
                      f'wall={time.perf_counter() - start:.1f}s', flush=True)

    summaries = {}
    for label in labels:
        summaries[label] = {}
        for kind in scenes:
            raw = results[label][kind]
            stat = summarize(raw, kind)
            stat['iterations_p95'] = float(np.percentile(
                [r['iterations'] for r in raw], 95))
            stat['time_ms_p95'] = float(np.percentile(
                [r['solve_time_ms'] for r in raw], 95))
            stat['time_ms_max'] = float(max(r['solve_time_ms'] for r in raw))
            summaries[label][kind] = stat
            print(
                f"{label} {kind}: "
                f"safe={stat['trajectory_safe_success_rate']:.1%}, "
                f"collision={stat['trajectory_collision_rate']:.1%}, "
                f"steps={stat['iterations_mean']:.1f}/p95={stat['iterations_p95']:.0f}, "
                f"time={stat['time_ms_mean']:.2f}/p95={stat['time_ms_p95']:.2f}ms",
                flush=True)

    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = args.out or os.path.join(
        BASE, 'logs', f'eval_fast_horizon_sweep_{stamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'args': {**vars(args), 'horizons': args.horizons},
                   'results': summaries},
                  f, ensure_ascii=False, indent=1)
    print(f'Written: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(
        BASE, 'configs', 'avoid_mjx_v17_hard_ft.yaml'))
    parser.add_argument('--model', default=os.path.join(
        BASE, 'models', 'panda_ppo_ik_avoid_v17_hard_ft_mjx.zip'))
    parser.add_argument('--horizons', type=parse_horizons,
                        default=parse_horizons('100,125,150,200'))
    parser.add_argument('--observation-horizon', type=int, default=100)
    parser.add_argument('--action-span', type=float, default=0.3675)
    parser.add_argument('--n', type=int, default=200,
                        help='Scenes per group.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-points', type=int, default=768)
    parser.add_argument('--infer-max-points', type=int, default=256)
    parser.add_argument('--progress-every', type=int, default=50)
    parser.add_argument('--out')
    main(parser.parse_args())
