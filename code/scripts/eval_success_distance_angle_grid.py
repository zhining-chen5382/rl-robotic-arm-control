"""Evaluate canonical safe success over initial distance/orientation bins."""
import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from ik_avoid.config import load_config  # noqa: E402
from mjx.env_mjx import RobotIKAvoidEnvMJX  # noqa: E402


def quaternion_angle_deg(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    dots = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def evaluate_chunk(worker_id, config_path, model_path, scenes,
                   infer_max_points, progress_every):
    # Imports stay inside workers so Windows spawn does not construct the
    # solver in the parent process.
    from ik_avoid.config import load_config
    from mjx.fast_inference import FastRLAvoidSolverMJX

    cfg = load_config(config_path)
    solver = FastRLAvoidSolverMJX(
        cfg, model_path=model_path, infer_max_points=infer_max_points,
        policy_backend='numpy', use_dls_refine=False)
    q0 = np.asarray([scene['start_q'] for scene in scenes], np.float32)
    fk = solver.fenv._fk(q0)
    initial_pos = np.asarray(fk[2], np.float32)
    initial_quat = np.asarray(fk[3], np.float32)
    target_pos = np.asarray([scene['target_pos'] for scene in scenes],
                            np.float32)
    target_quat = np.asarray([scene['target_quat'] for scene in scenes],
                             np.float32)
    distances = np.linalg.norm(target_pos - initial_pos, axis=-1)
    angles = quaternion_angle_deg(initial_quat, target_quat)

    rows = []
    start = time.perf_counter()
    for index, scene in enumerate(scenes):
        result = solver.solve(
            scene['target_pos'], scene['target_quat'],
            obstacle_specs=scene['obstacle_specs'],
            start_q=np.asarray(scene['start_q'], np.float32),
            record_trajectory=True)
        collision = result['trajectory_collisions'] > 0
        success = (
            result['position_error'] < 0.010
            and result['orientation_error_deg'] < 10.0
            and not collision)
        rows.append((
            float(distances[index]), float(angles[index]), bool(success),
            bool(collision), float(result['position_error'] * 1000.0),
            float(result['orientation_error_deg']),
            int(result['n_obstacles'])))
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f'worker {worker_id}: {index+1}/{len(scenes)} '
                f'wall={time.perf_counter()-start:.1f}s', flush=True)
    return rows


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return None, None
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def aggregate(records, distance_bin, angle_bin):
    distances = np.asarray([row[0] for row in records])
    angles = np.asarray([row[1] for row in records])
    successes = np.asarray([row[2] for row in records], dtype=bool)
    collisions = np.asarray([row[3] for row in records], dtype=bool)
    distance_max = max(distance_bin,
                       math.ceil(float(distances.max()) / distance_bin)
                       * distance_bin)
    distance_edges = np.arange(0.0, distance_max + distance_bin * 0.5,
                               distance_bin)
    angle_edges = np.arange(0.0, 180.0 + angle_bin * 0.5, angle_bin)
    d_index = np.clip(np.digitize(distances, distance_edges) - 1,
                      0, len(distance_edges) - 2)
    a_index = np.clip(np.digitize(angles, angle_edges) - 1,
                      0, len(angle_edges) - 2)
    shape = (len(angle_edges) - 1, len(distance_edges) - 1)
    count = np.zeros(shape, dtype=np.int32)
    success_count = np.zeros(shape, dtype=np.int32)
    collision_count = np.zeros(shape, dtype=np.int32)
    for ai, di, ok, coll in zip(a_index, d_index, successes, collisions):
        count[ai, di] += 1
        success_count[ai, di] += int(ok)
        collision_count[ai, di] += int(coll)
    success_rate = np.divide(
        success_count, count, out=np.full(shape, np.nan, dtype=float),
        where=count > 0)
    collision_rate = np.divide(
        collision_count, count, out=np.full(shape, np.nan, dtype=float),
        where=count > 0)
    return {
        'distance_edges': distance_edges,
        'angle_edges': angle_edges,
        'count': count,
        'success_count': success_count,
        'collision_count': collision_count,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'distances': distances,
        'angles': angles,
        'successes': successes,
        'collisions': collisions,
    }


def marginal(edges, indices, successes, collisions):
    rows = []
    for index in range(len(edges) - 1):
        mask = indices == index
        n = int(mask.sum())
        ok = int(successes[mask].sum())
        coll = int(collisions[mask].sum())
        lo, hi = wilson(ok, n)
        rows.append({
            'low': round(float(edges[index]), 6),
            'high': round(float(edges[index + 1]), 6),
            'n': n,
            'successes': ok,
            'success_rate': ok / n if n else None,
            'success_ci95': [lo, hi] if n else None,
            'collision_rate': coll / n if n else None,
        })
    return rows


def save_outputs(args, records, agg, elapsed):
    prefix = args.out_prefix
    if not os.path.isabs(prefix):
        prefix = os.path.join(BASE, prefix)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    de, ae = agg['distance_edges'], agg['angle_edges']
    csv_path = prefix + '.csv'
    cells = []
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'distance_low_m', 'distance_high_m', 'angle_low_deg',
            'angle_high_deg', 'n', 'successes', 'success_rate',
            'success_ci95_low', 'success_ci95_high', 'collision_rate'])
        writer.writeheader()
        for ai in range(len(ae) - 1):
            for di in range(len(de) - 1):
                n = int(agg['count'][ai, di])
                ok = int(agg['success_count'][ai, di])
                lo, hi = wilson(ok, n)
                cell = {
                    'distance_low_m': round(float(de[di]), 6),
                    'distance_high_m': round(float(de[di + 1]), 6),
                    'angle_low_deg': round(float(ae[ai]), 6),
                    'angle_high_deg': round(float(ae[ai + 1]), 6),
                    'n': n,
                    'successes': ok,
                    'success_rate': (float(agg['success_rate'][ai, di])
                                     if n else None),
                    'success_ci95_low': lo,
                    'success_ci95_high': hi,
                    'collision_rate': (float(agg['collision_rate'][ai, di])
                                       if n else None),
                }
                writer.writerow(cell)
                cells.append(cell)

    distances, angles = agg['distances'], agg['angles']
    d_index = np.clip(np.digitize(distances, de) - 1, 0, len(de) - 2)
    a_index = np.clip(np.digitize(angles, ae) - 1, 0, len(ae) - 2)
    overall_n = len(records)
    overall_success = int(agg['successes'].sum())
    overall_collision = int(agg['collisions'].sum())
    output = {
        'definition': {
            'x': 'Euclidean distance from initial end-effector position to target position',
            'y': 'shortest quaternion rotation angle from initial end-effector orientation to target orientation',
            'success': 'position <10mm AND orientation <10deg AND whole trajectory zero collision',
            'scene': 'random scenes with 1-3 obstacles',
        },
        'args': vars(args),
        'overall': {
            'n': overall_n,
            'successes': overall_success,
            'success_rate': overall_success / overall_n,
            'trajectory_collisions': overall_collision,
            'trajectory_collision_rate': overall_collision / overall_n,
            'distance_range_m': [float(distances.min()), float(distances.max())],
            'angle_range_deg': [float(angles.min()), float(angles.max())],
            'wall_seconds': elapsed,
        },
        'distance_marginal': marginal(
            de, d_index, agg['successes'], agg['collisions']),
        'angle_marginal': marginal(
            ae, a_index, agg['successes'], agg['collisions']),
        'cells': cells,
    }
    json_path = prefix + '.json'
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    return json_path, csv_path, output


def main(args):
    start = time.perf_counter()
    config_path = os.path.abspath(args.config)
    model_path = os.path.abspath(args.model)
    cfg = load_config(config_path)
    generator = RobotIKAvoidEnvMJX(cfg, n_worlds=1, max_points=768)
    scenes = generator.generate_scenes(
        args.n, args.seed, with_obstacles=True)
    workers = max(1, min(args.workers, args.n))
    chunks = [scenes[start_index:args.n:workers]
              for start_index in range(workers)]
    records = []
    if workers == 1:
        records = evaluate_chunk(
            0, config_path, model_path, chunks[0],
            args.infer_max_points, args.progress_every)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers) as executor:
            futures = [executor.submit(
                evaluate_chunk, worker_id, config_path, model_path, chunk,
                args.infer_max_points, args.progress_every)
                for worker_id, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                records.extend(future.result())
                print(f'completed {len(records)}/{args.n}', flush=True)
    agg = aggregate(records, args.distance_bin_m, args.angle_bin_deg)
    paths = save_outputs(args, records, agg, time.perf_counter() - start)
    output = paths[-1]
    print(json.dumps({
        'overall': output['overall'],
        'json': paths[0], 'csv': paths[1],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=606060)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--infer-max-points', type=int, default=256)
    parser.add_argument('--distance-bin-m', type=float, default=0.10)
    parser.add_argument('--angle-bin-deg', type=float, default=10.0)
    parser.add_argument('--progress-every', type=int, default=500)
    parser.add_argument('--out-prefix', default=(
        'logs/v21b_success_by_initial_distance_angle_seed606060_n12000'))
    main(parser.parse_args())
