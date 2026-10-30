# Selected research experiments

[Portfolio](../README.md) · [Main implementation](../code/README.md) · [Research journey](../docs/research_journey.md)

These selections preserve the research process rather than making every historical run depend on today's implementation.

| Selection | Research question | Included material |
|---|---|---|
| [PyBullet IK, PointNet and Residual](pybullet/README.md) | Why did richer representations or residual interventions fail to deliver reliable avoidance? | Frozen source, required protocol, dependency recipe and limitations |
| [v15 PPO + DLS](mujoco_v15/README.md) | Can refinement recover pose convergence while retaining collision history? | Frozen source/configuration; checkpoint and raw internal logs excluded |
| [v16 solver optimization](v16_optimization/README.md) | Can the same policy solve faster without changing recorded outcomes? | Protocol and source navigation for the paired NumPy comparison |
| [v21b heatmap](v21b_heatmap/README.md) | How did success vary with initial distance and orientation difference? | Evaluation source navigation and aggregate JSON/CSV |

The main v30 checkpoint is the only policy weight included. It cannot substitute for the different observation/action spaces of PyBullet or v15/v16 experiments. Frozen historical code retains its original limitations. Reproducing historical runs requires the corresponding artifacts.

Later v31-v33 configurations and records exist in the local research archive. The reviewed v33 record points to a missing checkpoint, so that branch is not made the default. The later waypoint-assisted research direction and its reported outcome are documented in [Research journey](../docs/research_journey.md#4-waypoint-assisted-improvement); the original implementation and evaluation logs for that iteration are unavailable.
