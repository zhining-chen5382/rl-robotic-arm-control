# PyBullet research archive

[Portfolio home](../../README.md) · [Current implementation](../../code/) · [Experimental results](../../docs/experimental_results.md)

This directory preserves the earlier Panda IK research behind the current geometric controller. It includes position and pose PPO, synthetic point-cloud observations, frozen-base residual learning, and controlled reward/representation experiments. It is an independent legacy source archive, not the current application's runtime or an automatically executed test suite.

The 45 Python files and the single Reward V2 protocol JSON were copied byte-for-byte from the research snapshot. Historical behavior, including limitations described below, is preserved. No legacy module was imported and no training, simulation, or evaluation was run while preparing this archive; checks were limited to syntax, local import dependencies, and file hashes.

## Reading the implementation

| Research question | Main source files |
|---|---|
| How does a policy learn position IK? | [`ik_3d/rl_env.py`](rl_ik/ik_3d/rl_env.py), [`ik_3d/rl_ik_solver.py`](rl_ik/ik_3d/rl_ik_solver.py) |
| How are position and orientation coupled? | [`ik_6d/rl_env_6d.py`](rl_ik/ik_6d/rl_env_6d.py), [`ik_6d/rl_ik_solver_6d.py`](rl_ik/ik_6d/rl_ik_solver_6d.py) |
| What are the numerical position-IK comparisons? | [`ik_solver.py`](rl_ik/ik_solver.py) |
| How is an obstacle represented with PointNet? | [`rl_env_6d_obstacle.py`](rl_ik/ik_6d/rl_env_6d_obstacle.py), [`pointnet_encoder.py`](rl_ik/ik_6d/pointnet_encoder.py), [`pointnet_features.py`](rl_ik/ik_6d/pointnet_features.py) |
| How can the original controller remain frozen? | [`frozen_base_policy.py`](rl_ik/ik_6d/frozen_base_policy.py), [`rl_env_6d_residual_obstacle.py`](rl_ik/ik_6d/rl_env_6d_residual_obstacle.py) |
| What happens when the base action saturates? | [`rl_env_6d_residual_headroom.py`](rl_ik/ik_6d/rl_env_6d_residual_headroom.py), [`diagnose_direct_vs_headroom.py`](rl_ik/diagnose_direct_vs_headroom.py) |
| Does explicit robot geometry help? | [`rl_env_6d_residual_geometry.py`](rl_ik/ik_6d/rl_env_6d_residual_geometry.py), [`rl_env_6d_residual_relative.py`](rl_ik/ik_6d/rl_env_6d_residual_relative.py), corresponding feature extractors |
| How do reward and episode termination affect learning? | [`reward_v2.py`](rl_ik/ik_6d/reward_v2.py), [`reward_v2_wrapper.py`](rl_ik/ik_6d/reward_v2_wrapper.py), [`isolated_episode_seed_wrapper.py`](rl_ik/ik_6d/isolated_episode_seed_wrapper.py) |

The base pose policy has 17 inputs and seven joint-increment actions. The first PointNet experiment adds a synthetic `512 × 3` surface cloud for one sphere or box. Residual variants keep that base policy frozen and learn seven action corrections; headroom composition accounts for the remaining action range. Later variants add six robot FK keypoints and a `6 × 512 × 4` robot-centered relation cloud.

These controllers are distinct from the current v30 model's 58 inputs and eight actions. The included current-model checkpoint cannot substitute for any legacy checkpoint.

## Entry points preserved for inspection

| Experiment family | Training entry | Evaluation or diagnosis |
|---|---|---|
| Base PPO | [`train_rl_ik.py`](rl_ik/train_rl_ik.py) | [`eval_1000.py`](rl_ik/eval_1000.py), [`eval_frozen_base_obstacle.py`](rl_ik/eval_frozen_base_obstacle.py) |
| PointNet-PPO | [`train_pointnet_obstacle.py`](rl_ik/train_pointnet_obstacle.py) | [`eval_pointnet_obstacle.py`](rl_ik/eval_pointnet_obstacle.py) |
| Direct residual | [`train_residual_obstacle.py`](rl_ik/train_residual_obstacle.py) | [`eval_residual_obstacle.py`](rl_ik/eval_residual_obstacle.py), [`diagnose_residual_vs_baseline.py`](rl_ik/diagnose_residual_vs_baseline.py) |
| Direct versus headroom | [`train_residual_composition_ablation.py`](rl_ik/train_residual_composition_ablation.py) | [`evaluate_residual_retraining_ablation.py`](rl_ik/evaluate_residual_retraining_ablation.py) |
| Robot keypoints | [`train_residual_geometry_ablation.py`](rl_ik/train_residual_geometry_ablation.py) | [`evaluate_residual_geometry_ablation.py`](rl_ik/evaluate_residual_geometry_ablation.py) |
| Relative geometry | [`train_residual_relative_ablation.py`](rl_ik/train_residual_relative_ablation.py) | [`evaluate_residual_relative_ablation.py`](rl_ik/evaluate_residual_relative_ablation.py) |
| Reward V2 | [`train_reward_v2_ablation.py`](rl_ik/train_reward_v2_ablation.py) | [`evaluate_reward_v2_ablation.py`](rl_ik/evaluate_reward_v2_ablation.py) |

Local helper imports are included. In particular, several evaluation scripts reuse statistics and scene-comparison helpers from `diagnose_direct_vs_headroom.py`, while Reward V2 training reuses functions from the relative-geometry training script.

Some historical scripts execute work at module import. Importing `train_pointnet_obstacle.py` starts training; importing `eval_pointnet_obstacle.py` starts evaluation. Read these as source files, and do not import the archive during automatic discovery or CI. Evaluation, diagnostic, initialization, and preflight commands can also create output files or simulate an environment.

## Manual preparation for a future experiment

This archive intentionally contains no historical weights, robot resources, raw benchmark results, or logs. Before deliberately running a legacy entry point:

1. Use a separate environment. [`requirements-legacy.txt`](requirements-legacy.txt) documents dependencies and versions recorded in historical model metadata; it is not a newly validated installation lock. The residual artifacts record Python 3.11.15, while the base pose checkpoint records Python 3.13.12. Serialized functions and library versions can affect cross-version loading.
2. Prepare a properly licensed Panda URDF and its referenced meshes at `rl_ik/urdf/panda.urdf` and the corresponding relative mesh locations. Asset provenance and licensing must be retained. Robot resources are not copied into this legacy package.
3. Supply the authorized checkpoint required by the selected experiment. The frozen base expects `rl_ik/models/panda_ppo_ik_6d_map.zip`, with a 17-dimensional observation and seven-dimensional action. Position-only PPO expects its own `panda_ppo_ik.zip`. PointNet and residual evaluators require their respective historical checkpoint directories. A normalization snapshot is also needed when resuming the original reward-normalized training workflow.
4. Preserve the package layout and use `rl_ik/` as the working directory for scripts with relative paths. The retained `robot_configs.py` also contains UR5 definitions, but no UR5 learned weights are supplied or claimed ready to run.
5. Inspect experiment prerequisites before opting into a run. The included [`Reward V2 protocol`](rl_ik/benchmarks/reward_v2/reward_v2_ablation_protocol.json) is configuration, not a result. Its training entry additionally requires the historical common initialization; other ablations expect their own initialization or generated preflight/performance artifacts. These are not included. Creating a new initialization or changing the protocol defines a new experiment and does not reproduce the archived run exactly.
6. Keep new outputs in a separate working copy. Training and diagnostics create `models/`, `logs/`, and `benchmarks/` artifacts and often refuse to overwrite existing outputs. Do not use the frozen copy as an automatic training target.

The older web application, shell launchers, interactive test scripts, duplicate backups, and intermediate checkpoints are excluded. In particular, historical shell evaluation used temporary checkpoint replacement; those shell scripts are not recommended entry points here.

## Known implementation and statistical boundaries

- **Simulation model:** actions update joint positions with `resetJointState`. Collision checks occur at discrete configurations against the single external obstacle. This is not torque control, continuous swept collision checking, or physical-robot safety verification.
- **Multiple clients:** legacy PyBullet calls do not consistently specify `physicsClientId`. Multiple environments in one process or a fallback to a shared-process vector environment can therefore use the wrong client. A future runnable derivative should explicitly bind clients; these historical files preserve the original behavior.
- **Randomness:** the early environments use global NumPy RNG despite accepting a reset seed. The older episode wrapper also resets Torch's global RNG. The isolated wrapper fixes environment-reset RNG coupling for later experiments, without changing earlier results.
- **Development-data reuse:** early Direct/Headroom, geometry, and relative runs have training seed ranges overlapping development seeds `2026081300–2026082299`. They are developmental comparisons, not independent unseen-scene tests. Reward V2 separates training at `3000000000 + episode_index` and designates another holdout, but DEV still supports reward calibration and repeated model development. The protocol records the independent holdout as `LOCKED_UNSEEN`.
- **Update budgets:** the initial 20,000-step residual callbacks stop during a 1,024-step PPO rollout and save from a step callback. Their final partial rollout does not receive a complete PPO update. Later 20,480-step experiments save after complete rollout/update segments. These historical budgets must not be silently treated as identical.
- **Success criteria:** the base training environment targets 5 mm / approximately 10°, while the legacy 6D solver returns success at 20 mm / approximately 10°. Later reports use other explicitly stated tolerances and collision conditions. Compare errors and collision history under a common stated criterion, not the returned Boolean alone.
- **Numerical baselines:** the hand-written Jacobian/DLS, transpose, and pseudoinverse methods optimize position; their accepted orientation argument does not make them full 6D pose baselines. `solve_path` interpolates a Cartesian line and does not implement obstacle-aware waypoint planning.
- **Research outcome:** the retained paired residual experiments did not establish a consistent safety improvement over the frozen base. The PointNet source is implemented, but a matching retained formal standalone PointNet performance record was not available. These limitations remain part of the [reported results](../../docs/experimental_results.md).

This source archive supports inspection of the research progression. The current application, its supplied checkpoint, and its separate validation path are maintained under [`code/`](../../code/).
