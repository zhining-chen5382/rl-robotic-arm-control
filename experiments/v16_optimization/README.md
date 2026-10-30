# v16: paired solver-optimization study

This experiment separates solver engineering from policy training: both sides used the same historical v16 policy and configuration, with NumPy inference, while changing the geometric computation inside a complete solve. It is distinct from both the older `v16_speed_base` ONNX experiment and the current v30 deployment.

## Retained measurements

**Experiment `EXP-FAST-V16`; seed parameter 2026.** These values come from the retained aggregate comparison, not a newly executed benchmark.

| Measured group | Cases | Original NumPy mean | Optimized NumPy mean |
| --- | ---: | ---: | ---: |
| Historical mixed group, 0–3 obstacles | 90 | **160.7639 ms** | **57.7480 ms** |
| Separate obstacle-free group | 42 | **101.5561 ms** | **23.8480 ms** |

The 90-case pool contained **21 zero-obstacle cases**, 23 with one obstacle, 25 with two, and 21 with three. It must not be relabelled as a purely obstacle-bearing sample. The separate free-space generator used seed 2027.

Six warmup scenes per solver and group were excluded from timing. The measured pool was split into six disjoint batches, alternating implementation order between batches. This is one paired solve per case, not six repeated measurements of every case. An outer timer enclosed scene setup, rollout, trajectory checks, and result construction; model construction and browser playback were outside that timer.

Both sides used 256-point closed-loop inference and retained full-cloud safety-related checks. The combined optimization enabled inactive-slot skipping, deferred nearest-direction calculations, direct quaternion-vector rotation, fused forward kinematics, and batched point rotation. Adaptive point budgets and SDF-buffer reuse were disabled, so their benefits are not included in this measured bundle.

The retained comparison reports zero final-joint, recorded-trajectory, position-error, and orientation-error differences across **132 case pairs**, plus zero mismatches in iterations, collision results, trajectory lengths, and terminal outcomes. This supports equivalence on those cases; it is not proof for every scene, platform, or numerical environment.

## Inspect the comparison code

- [Combined A/B harness](../../code/mjx/tests/fast_original_current_ab.py): constructs both implementations with fixed policy/configuration and alternates their timing order.
- [Timing, summary, and equivalence helpers](../../code/mjx/tests/fast_nearest_direction_ab.py): records complete-solve wall time and compares trajectories and outcomes.
- [Fast solver implementation](../../code/mjx/fast_inference.py): the geometric options and full-cloud verification path.
- [Historical v16 configuration](../../code/configs/avoid_mjx_v16.yaml): the matching observation and action settings.
- [Selected measurement data](../../assets/data/solver_optimization.csv): means, medians, p95, and recorded conditions.
- [Engineering discussion](../../docs/engineering_optimization.md): interpretation and the separate ONNX experiment.

## What is required for reproduction

The historical v16 champion checkpoint is **not included**. The released v30 model has a different observation/action representation and risk head; replacing the missing v16 model with v30 would make a different experiment.

The linked harness is retained implementation evidence, not an exact frozen package of every historical dependency. The present scene generator has stricter obstacle-group handling than the historical mixed pool. Running it today with the same seed and counts therefore must not be described as recovering those exact 90 scenes. Recovering the original explicit scene set and environment would be necessary for a faithful rerun.

The recorded CPU model and complete historical package environment are missing. No equivalence test or timing benchmark was rerun to assemble this research folder. Existing scripts can write result logs if executed; no execution is needed to inspect them.

The earlier 1068.9 ms value belongs to a different SB3/ONNX experiment. An exact matching source for the project-reported later 57.3 ms value has not been located. Those endpoints are not combined into a verified speedup here; the paired 160.76 → 57.75 ms study is the supported comparison.

[Portfolio](../../README.md) · [Historical v15](../mujoco_v15/README.md) · [Limitations](../../docs/limitations.md)
