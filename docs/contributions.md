# Contributions and Project Context

[Project overview](../README.md) · [Research journey](research_journey.md) · [Methodology](methodology.md) · [Experimental results](experimental_results.md)

I carried out this work during a robotics algorithm internship at **SIASUN**, with guidance from the supervising team. The team supplied the task definition and initial software framework. My work developed the algorithms, experiments, and integrated application from that starting point.

## My engineering work

| Area | My contribution |
|---|---|
| **Algorithm development** | Investigated PPO-based pose IK, PointNet-PPO obstacle encoding, and residual corrections around an existing reaching policy. Used failed experiments to examine whether observations, action composition, and reward signals supported useful avoidance behavior. |
| **Reward and curriculum design** | Redesigned the task reward around progress, investigated incentives associated with early collision termination, and used mixed obstacle-free and obstacle-containing training to retain reaching capability. |
| **Geometry and refinement** | Developed and investigated joint-wise action scaling and adaptive step regulation, and integrated collision-checked DLS refinement with the learned solver. Examined the roles of SDF clearance and historical Jacobian-based safety-direction features. |
| **Experimental reasoning** | Analyzed failure trajectories, compared controllers on paired scenes, distinguished final pose attainment from recorded-motion safety, and retained unsuccessful residual and representation experiments in the research record. |
| **Computational optimization** | Explored ONNX Runtime inference, profiled the complete solver, and optimized NumPy forward kinematics and geometry processing. Used a fixed-policy paired benchmark to check measured speed improvements against trajectories and outcomes. |
| **Software development** | Independently developed the final interactive simulation and performance-evaluation application, integrating scene configuration, algorithm comparison, 3D trajectory inspection, numerical metrics, and scene/result export on top of the initial framework. |

“Independently developed” describes my responsibility for the final application. The broader project builds on the team's starting framework, existing solver work, and third-party libraries and robot resources. It does not mean that I independently authored every underlying component or created the Panda model.

## The supervising team's role

The team defined the initial robotics problem, provided the software starting point, and supervised the internship. That support is part of the project's provenance. The portfolio focuses on the decisions I investigated and the engineering work I performed within that context.

The experiments ran on a **Franka Panda simulation model**. The SIASUN internship context does not imply that the archived results were obtained on a physical SIASUN robot. The evidence presented here supports kinematic simulation and software integration; it does not establish physical robot deployment.

## What the evidence supports

The research branches have separate models and protocols. The [experimental results](experimental_results.md) preserve the hybrid refinement gain, later policy evaluation, and negative residual findings; the [optimization document](engineering_optimization.md) explains the fixed-policy timing comparison. These results demonstrate particular engineering decisions rather than universal improvement across one benchmark.

I also reported a later waypoint-assisted iteration with approximately 93% simulation success. Its code and original evaluation are absent from the reviewed archive, so that result remains unverified and is excluded from verified performance claims.

## External tools and assets

The project uses established tools for specialized tasks: **PyBullet** and **MuJoCo/JAX** for simulation and batched computation, **PyTorch** and PPO implementations for learning, **ONNX Runtime** for inference experiments, **Flask** for the application backend, and **Three.js** for browser visualization. The research also draws on the published PointNet approach and classical Jacobian and damped least-squares methods.

The **Panda URDF, mesh resources, and derived simulation models** are robot assets, rather than original robot designs. Historical UR5 resources belong to an earlier platform branch. Third-party software, robot resources, and their licenses remain separate from my algorithm and application contributions.

This portfolio draft presents the engineering narrative, methods, and selected evidence without distributing the original internal source archive, model weights, or vendor assets. Before any such material is released, its provenance, redistribution permissions, and required notices must be confirmed. Public code and asset clearance remains pending.
