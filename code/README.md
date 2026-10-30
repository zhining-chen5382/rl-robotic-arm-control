# Main engineering implementation

[Portfolio](../README.md) · [Research experiments](../experiments/README.md) · [Methodology](../docs/methodology.md)

This is the integrated **Franka Panda / v30 best** implementation selected from the research workspace. It exposes PPO, PPO + collision-checked DLS refinement, plain DLS, and null-space avoidance. Joint configurations are updated kinematically; this is not torque-level control or hardware deployment.

## Run the interactive application

Use an isolated **Python 3.11** environment. From the repository root:

```bash
python -m pip install -r code/requirements-demo-tested.txt
python -B code/app.py
```

Open `http://127.0.0.1:5000`. Windows users can also run `code/start_demo.ps1 -Python <absolute-python-path>`; Linux users can run `bash code/visual_deploy.sh` with their environment active. These launchers do not start training. Three.js and the required loaders are bundled for local visualization.

The pinned file records the existing Windows CPU environment used for functional checks. It is not a claim of clean-machine installation testing, a GPU training lock, or universal operating-system compatibility. The broader optional research dependencies are in `requirements.txt`. The application does not install packages.

## Deployment contract

| Item | Selected setting |
|---|---|
| Robot | Franka Panda, seven arm joints |
| Checkpoint | `models/panda_ppo_ik_avoid_v30_risk_head_ft_mjx_best/best_model.zip` |
| Configuration | `configs/avoid_mjx_v30_risk_head_ft.yaml` |
| Observations / actions | 58 / 8 |
| Policy backend | NumPy |
| PPO solve horizon / time-feature denominator | 150 / 100 |
| Inference geometry | 256-point budget; retained full-cloud terminal/trajectory checks |
| Hybrid refinement | Up to 100 additional DLS iterations |
| Success criterion | Position <10 mm, orientation <10°, zero terminal and sampled-trajectory obstacle collisions |

`deployment.py` is the source of truth. The training configuration retains its original 100-step setting; deployment overrides are explicit. v30's independent risk head is not supported by the historical ONNX exporter.

## Evaluate a scene

Save a scene from the interface, then use the same API without starting an HTTP server:

```bash
python -B code/scripts/evaluate_deployment.py --scene scene.json --method all
```

Without `--scene`, the script uses two shipped fixed cases. Method names are `rl_avoid`, `rl_avoid_dls`, `dls_avoid`, and `nullspace`. JSON goes to stdout; `--output run.json` explicitly saves a result. Valid solver failures remain results rather than being discarded.

This is a small current-deployment evaluation, not a recreation of the historical v15/v16/v30 benchmark cohorts. First-use compilation can contribute to API solve time; animation and rendering are outside that timer.

## Reading the source

| Module | Responsibility |
|---|---|
| `app.py`, `templates/index.html` | Scene API, serialized shared state, comparison UI, scene/result export |
| `mjx/env_mjx.py` | JAX FK, observations, SDF geometry, reward and kinematic updates |
| `ik_avoid/ppo_gpu.py`, `train_gpu.py` | PPO collection/update, GAE, normalization, risk heads and checkpoints |
| `mjx/fast_inference.py`, `ik_avoid/policy_io.py` | NumPy policy/FK/SDF and complete solver pipeline |
| `mjx/avoid_solvers.py` | DLS, damped null-space avoidance, collision-checked refinement |
| `ik_avoid/eval_stats.py` | Canonical sampled-trajectory metrics |
| `mjx/tests/`, selected `scripts/` | Historical optimization and research tools |

## Training source and historical tools

Training is a separately invoked research workflow. `train_gpu.py` requires `--config`; it has no implicit default experiment. Set the world count, rollout length, seed, output model name, device and initialization deliberately. For risk-head fine-tuning, initialize from a compatible learned policy rather than freezing a randomly initialized reaching backbone.

Only the v30 inference ZIP is supplied. It does not contain an optimizer, complete training state, or reward-normalization history. Initializing a new fine-tuning run from it is not an exact resume of the original experiment. The original v28-to-v30 training schedule is not claimed reproducible from this checkpoint alone. No training was performed for this candidate.

`mjx/eval_mjx.py` and the optimization tools retain historical batch/default conventions. Use `scripts/evaluate_deployment.py` for the current application contract. Do not report a newly run historical tool as the original benchmark without matching its checkpoint and protocol.

The retained `ik_3d` and `ik_6d` code explains earlier implementations. Their old RL checkpoints are not shipped; unavailable legacy RL requests return a clear response. The supported application methods are the four above.

## Boundaries

Obstacle checks use sampled configurations and sampled link geometry. They do not certify continuous collision freedom, full runtime self-collision safety, or a physical robot. Random-scene generation protects the submitted start configuration where supported; it does not prove that a collision-free connecting path exists. [Source changes and validation](../docs/source_release.md) · [Resource provenance](../THIRD_PARTY_NOTICES.md)
