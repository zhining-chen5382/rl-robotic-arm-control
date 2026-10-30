# Making the solver faster without changing the measured trajectories

The engineering objective was to reduce the cost of a complete inverse-kinematics solve while keeping its geometric checks visible. A faster policy call alone would not address repeated forward kinematics, point transformations, distance queries, or trajectory verification.

An earlier CPU profile made this distinction concrete. In its 120-case obstacle-labelled group, policy prediction accounted for about **5.4%** of the recorded loop components, environment stepping **79.6%**, and the additional minimum-distance query **14.8%**. This was a separate historical profile, not the paired optimization benchmark below. It redirected attention from the neural network alone to repeated geometry and environment work.

The strongest retained comparison, **EXP-FAST-V16**, is a paired experiment with the **same v16 policy and configuration**, using the NumPy inference backend for both implementations. The optimized implementation reduced mean solve time from **160.76 to 57.75 ms** on a 90-scene mixed pool and from **101.56 to 23.85 ms** on a separate 42-scene obstacle-free pool. These are historical measurements, with the scope below.

![Paired mean solver latency before and after NumPy geometry optimizations](../assets/figures/solver_optimization.svg)

## The work removed from each solve

The deployment path translated the JAX batch environment's kinematic operations into NumPy for single-scene inference. Training benefits from large batches; a single-scene solver is more sensitive to dispatch overhead and repeated small operations. The measured optimization bundle then targeted the geometry work inside that NumPy path:

| Change | Purpose |
| --- | --- |
| Skip inactive obstacle slots | Avoid distance calculations for unused scene slots. |
| Defer direction calculations until the nearest points are known | Compute the direction features needed by the policy without constructing them for every candidate point. |
| Use a direct quaternion-vector rotation expression | Reduce temporary operations during repeated rigid transforms. |
| Fuse forward-kinematics calculations | Reuse intermediate transforms along the robot chain. |
| Batch point rotations | Transform link point clouds together instead of repeating small rotation calls. |

The combined experiment enabled these five options together. It does not isolate their individual contributions. Adaptive point budgets and reusable SDF buffers were disabled in both arms of this comparison; their possible benefits are not included in this claim.

## What the comparison measured

| Measured pool | Scenes | Original NumPy mean | Optimized NumPy mean |
| --- | ---: | ---: | ---: |
| Mixed pool: 0–3 obstacles | 90 | 160.7639 ms | 57.7480 ms |
| Dedicated obstacle-free pool | 42 | 101.5561 ms | 23.8480 ms |

The summarized figure data are available as [CSV](../assets/data/solver_optimization.csv).

The mixed pool contained **21 scenes with no obstacles**, 23 with one, 25 with two, and 21 with three. Its mean is therefore not a mean over obstacle-bearing scenes alone. The dedicated obstacle-free pool is a separate set.

The retained protocol used:

- The v16 champion policy, its matching configuration, and the NumPy policy backend on both sides.
- A 256-point inference budget per link, with full-cloud geometric verification retained for the relevant safety checks.
- Six warmup scenes per solver in each pool, excluded from the reported sample counts.
- Six disjoint timing batches, alternating implementation order between batches. Each measured scene was solved once by each implementation; the six batches are not six repeated measurements of every scene.
- An outer wall-clock timer around `solve(..., record_trajectory=True)`, including scene setup, rollout, trajectory checking, and result construction. Model loading and solver construction occurred before this timer; rendering and browser interaction were outside it.
- Seed parameter 2026; the harness used 2027 for the dedicated obstacle-free generator.

The paired checks reported zero differences in final joint angles, recorded joint trajectories, position errors, and orientation errors. They also reported zero mismatches in iteration counts, collision results, trajectory lengths, and terminal success outcomes across these **132 scene pairs**. This supports behavioral equivalence on the measured cases; it is not a proof for every possible scene or floating-point environment.

## How safety checks survived the optimization

The policy's closed-loop observations could use a reduced point cloud. Safety-related candidate, terminal, and recorded-trajectory checks used the full retained point cloud. The implementation also used bounding-sphere lower bounds to reject clearly separated frames before running detailed SDF queries on the remaining frames.

This distinction matters because a sparse cloud can miss the nearest surface point. A sparse-cloud collision is sufficient to trigger a check or stop, while a sparse-cloud clearance alone does not certify safety. The geometry remains an approximation: full-cloud checking here means checking the complete sampled cloud at recorded configurations, not exact continuous collision detection on the robot mesh. See [limitations](limitations.md).

## ONNX Runtime was a separate deployment experiment

Earlier work exported policy networks to ONNX and compared FP32 and INT8 inference with an SB3 policy path. In **EXP-ONNX-LEGACY**, one retained run with 200 requested scenes per group, ten warmups, a 768-point setting, and the older `v16_speed_base` model paired with a v15 configuration, the mixed-group means were:

| Backend | Mean solve time |
| --- | ---: |
| SB3 | 1068.9468 ms |
| ONNX FP32 | 672.2543 ms |
| ONNX INT8 | 691.3359 ms |

That run also recorded different success outcomes across backends. It supports an inference-backend experiment, not a claim that quantization preserved all solve behavior. INT8 was not the fastest backend in that mixed group.

The final application's fast path defaults to **NumPy policy inference**. ONNX export and benchmarking are part of the engineering history, rather than the sole explanation for the final solver's speed.

An earlier project-reported summary described a change from 1068.9 to 57.3 ms. The 1068.9 ms endpoint is traceable to the older run above; an exact matching source for 57.3 ms has not been located. Those endpoints are not presented as a controlled pair, and no combined speedup is claimed. The paired NumPy experiment is the basis of the figure in this portfolio.

## Evidence and reproducibility boundary

The retained evidence identifiers are `fast_original_current_ab_20260821_110945.json`, the `fast_original_current_ab.py` harness and its comparison helper, and `bench_quant_e2e_20260818_154804.json`. The source candidate includes the [NumPy comparison harness](../code/mjx/tests/fast_original_current_ab.py), its helpers, and [protocol notes](../experiments/v16_optimization/README.md). The original v16 checkpoint and raw internal timing logs are not included; the supplied v30 policy is a different experiment.

The timing record does not identify the CPU model, and the exact historical package/runtime environment was not preserved. The experiment has not been rerun for this portfolio. These measurements describe the retained runs and should not be treated as a hardware-independent latency guarantee or a current v30 performance claim.

[Project overview](../README.md) · [Experimental results](experimental_results.md) · [Software demonstration](../demo/README.md) · [Limitations](limitations.md)
