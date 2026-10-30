# Historical v15: geometric PPO and DLS refinement

This directory preserves the earlier MuJoCo/JAX research implementation that preceded the [current v30 application](../../code/app.py). It makes the training, curriculum, geometry, reward, and hybrid-evaluation logic inspectable. The Python files and v15 configuration in `code/` are unchanged byte-for-byte copies of the retained historical source snapshot; they have not been updated to behave like the current application.

**This is a source archive, not a runnable historical reproduction package.** No v15 checkpoint, normalization state, robot mesh, URDF, MJCF asset, or raw experiment log is included here. The current release's v30 checkpoint cannot substitute for v15: its observations, action outputs, and learned policy differ. No training or historical simulation benchmark was executed to prepare this archive.

## What the branch implemented

The environment performs batched JAX forward kinematics and evaluates sampled robot geometry against analytic box, sphere, and cylinder SDFs. It updates joint configurations directly rather than simulating torque-controlled contact dynamics.

The v15 configuration combines 68 observation features with seven joint-direction actions, task-dependent step scaling, and per-joint braking. PPO trains from scratch with a mixed curriculum: an initial obstacle-free stage, then stages that retain obstacle-free episodes while introducing one and multiple obstacles. The motivation was to preserve base IK ability while learning obstacle interaction.

Hybrid evaluation starts from the PPO result and allows up to **100 additional DLS iterations**. Candidate steps are checked against sampled geometry; the refinement code shrinks rejected steps and applies adoption gates involving collision, clearance, and error improvement. These checks do not establish continuous collision freedom between configurations.

## Recounted historical result

**Experiment `EXP-HYBRID-V15`; seed 2026.** Success below requires position error below 10 mm, orientation error below 10 degrees, and no terminal or recorded-trajectory obstacle collision.

| Actual scene group | Method | Trajectory-safe successes | Episodes with a trajectory collision |
| --- | --- | ---: | ---: |
| Obstacle-bearing subset, n = 764 | PPO | **645 / 764 (84.4%)** | 57 / 764 (7.5%) |
| Same subset | PPO + DLS | **673 / 764 (88.1%)** | 57 / 764 (7.5%) |

The hybrid adds 28 safe successes within this subset, with an extra computation budget. The original group labelled “obstacles” actually contained 1,000 cases, including 236 with no obstacles. In that mixed group, trajectory-safe success is 863/1,000 for PPO and 903/1,000 for hybrid. The older terminal-only rates are 864/1,000 and 912/1,000; reporting those as full-trajectory success on 1,000 obstructed scenes would change both the criterion and denominator.

The retained per-episode log supported this recount during the project review; it is not redistributed here. The [results discussion](../../docs/experimental_results.md) and [curated aggregate data](../../assets/data/verified_results.csv) preserve the distinction. The unchanged 57 collision episodes are an observation for this experiment, not a general safety guarantee for refinement.

## Read the implementation

| Source | Role |
| --- | --- |
| [train_mjx.py](code/train_mjx.py) | Historical SB3 PPO training, normalization, checkpointing, and evaluation setup |
| [v15 configuration](code/configs/avoid_mjx_v15.yaml) | Curriculum, reward, observation, action, and optimizer settings |
| [env_mjx.py](code/mjx/env_mjx.py) | Batched kinematics, sampled geometry, SDF features, joint updates, and reward |
| [vec_env.py](code/mjx/vec_env.py) | Historical SB3-compatible vector environment |
| [callbacks.py](code/ik_avoid/callbacks.py) | Curriculum transitions and periodic evaluations |
| [avoid_solvers.py](code/mjx/avoid_solvers.py) | PPO inference, DLS, null-space baseline, and checked refinement |
| [eval_stats.py](code/ik_avoid/eval_stats.py) | Historical statistics and batched rollout; its `strict_success_rate` is terminal-only |
| [eval_hybrid_dls.py](code/scripts/eval_hybrid_dls.py) | Same-scene PPO/hybrid comparison and adoption logic |
| [eval_dual.py](code/scripts/eval_dual.py) | Comparing best/final checkpoints and archiving the selected champion |

The supporting configuration, robot-parameter, quaternion, model-building, and ONNX-wrapper modules are also preserved. No older web application or launch scripts are copied into this branch.

## Reproduction boundary

Static AST inspection found the local Python import dependencies represented by these files. External dependencies include NumPy, JAX, MuJoCo, Torch, Stable-Baselines3, Gymnasium, PyYAML, and trimesh; the optional historical ONNX wrapper also imports ONNX Runtime. This check did not install or import them and does not establish runtime compatibility.

Execution still requires the matching historical checkpoints and robot assets in the locations expected by the source. Some unchanged script defaults refer to other historical configurations or models that are not shipped here; this directory intentionally does not offer an install-and-run recipe. Training, checkpoint selection, conversion, and evaluation scripts can write models, assets, or logs when executed, so reading this archive is separate from authorizing a run.

The retained source snapshot is not a complete versioned environment from the original measurement date. Later edits, dependency changes, and absent scene histories can prevent seed-only reconstruction of old experiments. Historical comments and theoretical claims remain part of the frozen source, not new guarantees endorsed by this release. For a small current evaluation, use the [v30 deployment evaluator](../../code/scripts/evaluate_deployment.py).

[Portfolio](../../README.md) · [Research journey](../../docs/research_journey.md) · [Limitations](../../docs/limitations.md)
