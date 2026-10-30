# Source selection, changes and functional validation

[Portfolio](../README.md) · [Main source](../code/README.md) · [Historical experiments](../experiments/README.md)

## What this candidate includes

The main engineering tree is selected from the later MuJoCo/JAX implementation. `deployment.py` fixes Panda, the v30 best policy, a 150-step solve horizon and the original 100-step observation scale. The approved inference ZIP is unchanged, with SHA-256 `d8f9db135291e3dde06d40b02c43f31eeba849c266896e890e794c441b0905b8`.

Selected PyBullet and v15 source snapshots preserve original bytes. v16 timing tools remain in their existing module hierarchy, with their historical protocol documented separately. v21b's retained aggregate heatmap JSON/CSV is included without changing counts. Other policy weights, raw internal archives, registration materials and local environments are excluded.

## Changes to the current application

- Reject invalid or out-of-range initial joint configurations consistently before solving.
- Pass the actual initial configuration into random-obstacle generation; do not claim a connecting path has thereby been proved feasible.
- Synchronize the automatic case's exact start and target quaternion with scene save/load and manual replay.
- Update canvas and table labels from the same four-method mapping; keep method selection stable while a solve is running.
- Limit the release to Panda and return a clear unavailable response for old RL endpoints whose checkpoints are not shipped.
- Add `scripts/evaluate_deployment.py` to evaluate saved scenes through the current API rather than historical batch defaults.
- Require an explicit training configuration and reject frozen-backbone training without initialization or a valid resume checkpoint. Training mathematics, reward coefficients, FK, SDF, policy weights and historical success thresholds were not changed for these application fixes.

Frozen historical source is separate from these application changes. Known legacy limitations are documented instead of silently rewriting the historical implementation.

## Functional checks on 2026-10-29

Checks used the existing **Windows / Python 3.11.15 / CPU** environment: NumPy 2.2.6, Torch 2.12.1+cpu, MuJoCo 3.11.0, JAX/JAXlib 0.10.2, Flask 3.1.3, Gymnasium 1.3.0, PyYAML 6.0.3, Trimesh 5.0.0 and SciPy 1.17.1. No packages were installed and no training was started. This was not a fresh-environment installation test.

| Check | Observed outcome |
|---|---|
| Four fixed scenes, including a no-obstacle variant, with all four methods | 16 API solves returned valid metrics; initial/final trajectory states and canonical success classification agreed |
| Invalid scenes and out-of-range starts | Rejected with HTTP 400 |
| Disabled training and unsupported robot | Training remained disabled; UR5 rejected |
| Shared-state concurrency | Two baseline requests agreed with their sequential references |
| Deployment evaluation CLI | Two default cases returned valid JSON and the expected checkpoint hash |
| Real Chrome with external page requests blocked | Panda meshes loaded locally; no external page requests, HTTP errors or browser script errors observed |
| Four-method labels | Table headings and canvas labels matched the selected methods |
| Import A, run automatic case B, save and manually replay B | Precise start and target quaternion were retained; automatic/manual inputs agreed |
| Scene and experiment export | Scene round trip and JSON/CSV exports completed |

Some numerical baseline solves did not meet the pose/collision success criterion. A valid failure result is part of the comparison; these checks do not claim all methods solved every case. No new population success rate or speedup is reported.

To repeat the small API checks in an existing suitable environment:

```bash
python -B code/scripts/check_release.py
```

`--output <file.json>` optionally saves the API report. The browser checks also captured real screenshots and downloads for private review; these are separate from the unchanged historical demo media.

## What was not revalidated

The original v15 84.4% to 88.1% comparison, v16 160.76 to 57.75 ms timing result, and v30 seed evaluations remain historical experiments with their documented protocols. The small checks above do not reproduce their cohorts or establish equivalent speed after the application fixes. They do not validate the reported Waypoint result, continuous collision safety, dynamics or physical-robot deployment.

[Third-party provenance](../THIRD_PARTY_NOTICES.md) · [Limitations](limitations.md)
