# Scope and limitations

This project demonstrates an engineering process for learning and evaluating collision-aware inverse kinematics for a seven-joint arm: defining a task, building representations and rewards, diagnosing failures, comparing alternatives, and integrating a simulation application. Its evidence supports that scope with several boundaries.

## Simulation and control scope

The retained implementation uses the Franka Panda model in offline simulation. The main batch environment updates joint configurations directly and evaluates forward kinematics and sampled geometry. It is a **kinematic IK and obstacle-avoidance environment**, rather than torque control with contact dynamics.

The portfolio does not establish deployment on a physical robot. The simulation viewer, exported joint trajectories, and browser verification are software demonstrations. They do not measure actuator tracking, calibration error, communication latency, physical contact, or transfer from simulation to hardware. Internship context does not change this evidence boundary.

## What “trajectory-safe success” means here

The later evaluation convention requires position error below 10 mm, orientation error below 10 degrees, no terminal obstacle collision, and no obstacle collision at any recorded trajectory configuration. This differs from both the tighter training termination settings and older terminal-only success summaries.

Robot clearance is approximated with sampled link geometry and analytic SDF queries for the supported obstacle shapes. Checking the full retained cloud is more complete than checking a reduced inference cloud, but it is still a sampled geometric test. The retained trajectory checker does not establish continuous collision freedom between consecutive configurations.

Joint limits are enforced in the kinematic updates. Self-collision filtering is used during scene or target sampling, but the main rollout obstacle metric should not be read as a complete runtime self-collision guarantee. Per-joint braking reduces steps near obstacles; its nonzero floor and remaining recorded collisions preclude a claim of guaranteed collision avoidance.

## Success rates need their original conditions

Model version, scene distribution, horizon, pose tolerance, and collision convention all affect the result. Some historical groups labelled “obstacles” also contain zero-obstacle scenes. A percentage from such a group cannot be relabelled as performance on obstacle-bearing scenes alone.

The later v30 records (**EXP-DEPLOY-V30**), for example, report trajectory-safe success of **87.3%** and **85.2%** on the obstacle groups for seeds 2026 and 30303, with **7.9%** trajectory collision in each group. Their separate obstacle-free groups report **93.9%** and **93.5%** success. These are per-group, 1,000-scene logged summaries at a 150-step solve horizon, not evidence of approximately 93% success in multi-obstacle planning.

The applicant also reports a historical waypoint-based method with approximately 93% success (**REPORT-WAYPOINT**). The matching implementation and original evaluation record are absent from the material reviewed for this portfolio. That result is identified as **applicant-reported historical work, currently unverified**. No waypoint algorithm, benchmark protocol, or reproduced result is invented to fill this gap.

## Evaluation and reproducibility gaps

- Several configurations were selected or revised after inspecting recurring evaluation seeds. A seed labelled “holdout” in a filename is useful additional evidence, but does not by itself establish a permanently untouched independent test set.
- The retained records do not establish consistent gains across multiple independent training seeds. Small differences between later variants should not be presented as statistically significant improvements.
- Some later files contain aggregate summaries without the underlying per-case trajectories. Their recorded metrics can be checked for consistency, but the complete numerator cannot be independently reconstructed from those files alone.
- Historical snapshots, exact package environments, hardware metadata, and some original experiment artifacts are incomplete. Later source code may not exactly reconstruct every historical run.
- The historical training runs, population success-rate evaluations, and timing benchmarks have not been rerun for this source candidate. Small application checks are reported separately in [Source changes and validation](source_release.md); they do not revalidate those historical results.

The [optimization study](engineering_optimization.md) therefore uses a specifically identified paired comparison and limits its equivalence claim to the 132 evaluated scene pairs. It does not combine measurements from different policies and protocols into one speedup.

## Public-release boundary

This is a curated source candidate. The related algorithm source has my supervisor's publication permission, and the v30 best inference checkpoint has been approved for inclusion. Other historical checkpoints, raw internal experiment archives, registration materials, and UR5 assets are excluded. My independent development of the integration application does not establish ownership of third-party components; see [Third-party provenance](../THIRD_PARTY_NOTICES.md).

The [main source guide](../code/README.md) identifies the included model, configuration, environment, and four supported methods. Historical snapshots are not all self-contained executable experiments. The bundled Panda resources have been matched to the PyBullet 3.2.5 distribution; no upstream Git revision is invented. The [archived screen recording](../demo/README.md) retains an unidentified checkpoint; adding current source does not retroactively establish the recording's model version.

[Project overview](../README.md) · [Methodology](methodology.md) · [Experimental results](experimental_results.md) · [Engineering optimization](engineering_optimization.md) · [Software demonstration](../demo/README.md)
