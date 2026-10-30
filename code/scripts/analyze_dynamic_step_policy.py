"""Summarize the learned v21 step multiplier along closed-loop trajectories."""
import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from ik_avoid.config import load_config  # noqa: E402
from mjx.env_mjx import RobotIKAvoidEnvMJX  # noqa: E402
from mjx.fast_inference import FastRLAvoidSolverMJX  # noqa: E402


def summarize(values):
    x = np.asarray(values, dtype=np.float32)
    if not len(x):
        return {'n': 0}
    return {
        'n': int(len(x)),
        'mean': float(x.mean()),
        'p05': float(np.percentile(x, 5)),
        'p10': float(np.percentile(x, 10)),
        'p50': float(np.percentile(x, 50)),
        'p90': float(np.percentile(x, 90)),
        'min': float(x.min()),
        'fraction_below_0_95': float(np.mean(x < 0.95)),
        'fraction_below_0_75': float(np.mean(x < 0.75)),
    }


def main(args):
    cfg = load_config(args.config)
    generator = RobotIKAvoidEnvMJX(cfg, n_worlds=1)
    solver = FastRLAvoidSolverMJX(
        cfg, model_path=args.model, infer_max_points=args.infer_max_points,
        policy_backend='numpy', use_dls_refine=False,
        max_steps=args.max_steps,
        observation_max_steps=args.observation_max_steps)
    groups = {name: [] for name in (
        'all', 'obstacles', 'free', 'obstacle_near_100mm',
        'obstacle_near_30mm', 'target_near_50mm',
        'risk_approaching', 'risk_receding')}
    direction_max_abs = []
    for kind, with_obstacles in (('obstacles', True), ('free', False)):
        scenes = generator.generate_scenes(
            args.n, args.seed, with_obstacles=with_obstacles)
        for scene in scenes:
            fs = solver.fenv
            obs = fs.set_scene(
                scene['target_pos'], scene['target_quat'], scene['start_q'],
                scene['obstacle_specs'], specs_inflated=False)
            for _ in range(fs.max_steps):
                action, _ = solver.policy.predict(obs, deterministic=True)
                direction_max_abs.append(float(
                    np.max(np.abs(action[:fs.n_ctrl]))))
                if fs.learned_step_scale:
                    raw = float(action[fs.n_ctrl])
                    mult = (fs.learned_step_min
                            + (1.0 - fs.learned_step_min)
                            * (raw + 1.0) * 0.5)
                else:
                    mult = 1.0
                groups['all'].append(mult)
                groups[kind].append(mult)
                if fs.prev_pos_dist < 0.05:
                    groups['target_near_50mm'].append(mult)
                if with_obstacles and fs.prev_min_dist < 0.10:
                    groups['obstacle_near_100mm'].append(mult)
                if with_obstacles and fs.prev_min_dist < 0.03:
                    groups['obstacle_near_30mm'].append(mult)
                if (with_obstacles and fs.prev_min_dist < 0.10
                        and obs.shape[-1] >= 58):
                    if obs[-2] < -1e-6:
                        groups['risk_approaching'].append(mult)
                    elif obs[-2] > 1e-6:
                        groups['risk_receding'].append(mult)
                obs, md = fs.step(action)
                if md <= 0.0 or (fs.prev_pos_dist < fs.pos_tolerance
                                 and fs.prev_orient_dist < fs.orient_tolerance):
                    break
    result = {name: summarize(values) for name, values in groups.items()}
    result['direction_max_abs'] = summarize(direction_max_abs)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(
        BASE, 'configs', 'avoid_mjx_v21_learned_step_ft.yaml'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--infer-max-points', type=int, default=256)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--observation-max-steps', type=int)
    parser.add_argument('--out')
    main(parser.parse_args())
