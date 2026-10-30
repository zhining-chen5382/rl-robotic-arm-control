"""Run a small, current-deployment evaluation through the application's API.

Defaults: v30 best, Panda, NumPy, 150 PPO steps, a 100-step observation scale,
256-point inference and full retained-cloud trajectory checks. The same API
adds up to 100 DLS refinement iterations for the hybrid method. No HTTP server,
training, automatic large benchmark, or implicit result-file write is started.

Examples, from code/:
    python scripts/evaluate_deployment.py
    python scripts/evaluate_deployment.py --scene scene.json --method all
    python scripts/evaluate_deployment.py --method rl_avoid_dls --output run.json

These are newly executed functional evaluations, not reproductions of the
historical success-rate or latency benchmarks. First-use compilation may affect
the measured API solve time; no steady-state warmup is performed by this script.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


CODE_DIR = Path(__file__).resolve().parents[1]
METHOD_CHOICES = ('rl_avoid', 'rl_avoid_dls', 'dls_avoid', 'nullspace')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path,
                        help='One saved portable Panda scene JSON object.')
    parser.add_argument('--method', choices=(*METHOD_CHOICES, 'all'),
                        default='rl_avoid',
                        help='Default: rl_avoid. Use all for four methods per scene.')
    parser.add_argument('--output', type=Path,
                        help='Optional JSON output file; otherwise stdout only.')
    return parser.parse_args(argv)


def load_scenes(scene_path):
    if scene_path is not None:
        with scene_path.open(encoding='utf-8-sig') as handle:
            scene = json.load(handle)
        if not isinstance(scene, dict):
            raise ValueError('--scene must contain one portable scene object, not an array.')
        return [(scene_path.name, scene)]
    with (CODE_DIR / 'static/demo_avoid_cases.json').open(encoding='utf-8') as handle:
        examples = json.load(handle)
    if not isinstance(examples, list) or len(examples) < 2:
        raise ValueError('The release must contain at least two fixed demonstration cases.')
    return [(f'fixed_demo_case_{index}', scene)
            for index, scene in enumerate(examples[:2])]


def run(args):
    scenes = load_scenes(args.scene)
    methods = METHOD_CHOICES if args.method == 'all' else (args.method,)
    # Keep JSON stdout usable even when model construction logs progress.
    with redirect_stdout(sys.stderr):
        sys.path.insert(0, str(CODE_DIR))
        import app as web

        web.check_files()
        web._load_avoid_config()  # Includes the release's Panda-only guard.
        with web.app.test_client() as client:
            info_response = client.get('/api/software_info')
            if info_response.status_code != 200:
                raise RuntimeError('Unable to read deployment metadata.')
            deployment = info_response.get_json()
            records = []
            for scene_id, scene in scenes:
                for method in methods:
                    if not isinstance(scene, dict):
                        raise ValueError(f'{scene_id} is not a scene object.')
                    submitted = dict(scene, method=method)
                    response = client.post('/api/solve_ik_avoid', json=submitted)
                    records.append({
                        'scene_id': scene_id,
                        'scene': submitted,
                        'method': method,
                        'http_status': response.status_code,
                        'result': response.get_json(),
                    })
    return {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'New functional evaluation of the current deployment; not a historical benchmark reproduction.',
        'warmup_performed': False,
        'timing_note': 'API solve and sampled-trajectory checks; first-use compilation may contribute. No playback or rendering.',
        'deployment': deployment,
        'model_sha256': hashlib.sha256(web.MODEL.read_bytes()).hexdigest(),
        'records': records,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            with args.output.open('w', encoding='utf-8', newline='\n') as handle:
                handle.write(encoded + '\n')
        print(encoded)
        # A valid solve that misses the goal is experimental data, not a CLI error.
        return 0 if all(row['http_status'] == 200 for row in report['records']) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
