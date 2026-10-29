# Methodology

[Portfolio home](../README.md) · [Experimental results](experimental_results.md)

The project studies a practical question: how can a learned inverse-kinematics policy reach a target pose while accounting for obstacles around an entire redundant arm? The work progresses from a seven-joint pose controller to geometric obstacle representations, residual learning, and a solver that can be compared with numerical IK.

The implemented task is **kinematic motion generation in simulation**. Actions change joint positions; the reported experiments do not establish torque control, actuator tracking, or deployment on a physical robot.

## 1. Task and algorithm families

The reference arm is a seven-degree-of-freedom Panda. A problem instance contains an initial joint configuration, a target end-effector position and orientation, and a simulated obstacle scene. A solver returns joint configurations together with pose errors, collision indicators, and timing information.

| Family | Policy input | Policy output | Purpose |
|---|---|---|---|
| Base pose PPO | 17-dimensional robot/task state | Seven joint increments | Learn position and orientation control without obstacle input |
| PointNet-PPO | Task state and a synthetic 512-point obstacle cloud | Seven joint increments | Learn perception and control jointly in a single-obstacle environment |
| Frozen-base Residual PPO | Task state, obstacle cloud, and the frozen base action; later variants add robot geometry | Seven residual action components | Preserve the base policy while learning obstacle-related corrections |
| JAX geometric PPO | Task state and link-level SDF features; optional geometric or temporal features depend on version | Seven components in earlier versions; eight in the later learned-step controller | Train and evaluate batched geometric IK policies |
| PPO + DLS | A PPO solution followed by numerical refinement | A refined joint trajectory | Improve final pose accuracy with additional collision-checked computation |

The PyBullet residual controller and the later risk-step head are different designs. The former corrects seven joint-action components using obstacle features. The latter adjusts an eighth action controlling step magnitude while holding the existing direction actor fixed.

## 2. Establishing a pose-control baseline

The base 6D environment exposes the following 17-dimensional state:

| Component | Dimensions |
|---|---:|
| Current joint angles | 7 |
| Unit direction toward the target position | 3 |
| Position error magnitude | 1 |
| Orientation error as an axis-angle vector | 3 |
| Orientation-derived cosine distance | 1 |
| Coupled position/orientation distance | 1 |
| Normalized episode step | 1 |

The policy predicts a normalized action in `[-1, 1]^7`. The environment scales the action, adds it to the current joint angles, clips the result to joint limits, and recomputes forward kinematics. For this baseline, the distance-dependent scale is:

```text
s = 0.003 + 0.49 × min(position_error / 0.5, 1)
q_next = clip_to_joint_limits(q + s × clipped_action)
```

The smaller scale near the target supports finer adjustments. It is an algorithmic step size, not a measured physical joint velocity.

The pose reward uses a coupled distance:

```text
c = 2 − cos(orientation_error)
D = position_error × c + (c − 1)
r_task = 2.006 − 1000 × D
```

Step, joint-limit, and success terms supplement this task reward. The implementation uses single-agent PPO; the historical “MAPPO-IK” label refers to the adopted reward formulation and does not indicate a multi-agent training implementation.

Training targets are generated through forward kinematics, with a small position perturbation in the base environment. Episodes start either near the home configuration or from a random configuration. Forward-kinematics sampling provides a useful task distribution, but a subsequently perturbed target is not a formal reachability guarantee.

## 3. Adding obstacle information with PointNet

The initial obstacle environment generates one sphere or box and samples 512 points from its surface. These are **synthetic geometric points**, not camera or depth-sensor measurements.

A shared pointwise network maps coordinates through `3 → 64 → 128 → feature_dimension`; max pooling produces a global obstacle feature. Concatenating that feature with the task state lets PPO learn an action conditioned on obstacle geometry.

This experiment changes both perception and the control-learning problem. Its existence demonstrates an implemented PointNet-PPO pipeline, while the available evaluation evidence does not support a verified standalone PointNet success-rate claim.

## 4. Separating base control from residual learning

The residual design freezes the original pose PPO and supplies its deterministic action to a second PPO policy. The initial composition is:

```text
a_final = clip(a_base + α × clip(a_residual), −1, 1)
```

With the base action frequently near its limits, some corrections disappear under clipping. A subsequent headroom formulation scales each correction by the available room in its requested direction:

```text
headroom_i = 1 − a_base_i    if a_residual_i ≥ 0
             1 + a_base_i    otherwise
a_final_i = clip(a_base_i + α × clip(a_residual_i) × headroom_i, −1, 1)
```

A zero residual preserves the base action. The original environment applies its adaptive joint step exactly once after composition. An additional quadratic residual penalty discourages unnecessary intervention.

Later representation experiments add six ordered robot FK keypoints. A robot-centered relation cloud describes every obstacle point relative to every keypoint using `[dx, dy, dz, distance]`, producing a `6 × 512 × 4` tensor. Controls with zero geometry or uncentered coordinates help test the contribution of geometric representation.

Reward development also changes the learning signal. The later Reward V2 design combines clipped IK progress, change in a squared danger measure within 150 mm, danger occupancy, joint-limit and residual terms, and explicit terminal rewards. It terminates at the first collision. Because episode handling changes with the reward, this is a reward-and-credit-assignment experiment, not a reward-number-only ablation.

The residual experiments produced useful diagnostics, but the archived development results do not establish a consistent obstacle-avoidance improvement over the frozen base. The [results document](experimental_results.md#4-residual-learning-and-negative-results) preserves that distinction.

## 5. Scaling geometric evaluation with JAX

The later backend is organized under a historical MJX name and uses MuJoCo-related models and numerical solvers. Its central training environment performs **batched analytic forward kinematics and signed-distance calculations in JAX**. A step updates joint positions algebraically and recomputes geometry; it is not an `mjx.step` torque-driven physics rollout.

The geometric representation samples robot link surfaces and evaluates their distance to analytic sphere, box, and cylinder obstacles. Ten monitored groups cover the arm, hand, and fingers. For each group, the observation includes a clipped distance and a direction associated with its nearest obstacle point.

Sampling resolution is part of the protocol. The archived later evaluation uses 768 points for the scene-generation environment and 256 for fast inference. Different resolutions can affect both observed clearance and collision detection; they are not interchangeable safety guarantees.

### Version-specific safety and control features

Several mechanisms were explored rather than being present in every model:

- **Jacobian safety direction:** historical variants map a nearest-point SDF direction into joint space using a geometric Jacobian. This feature is disabled in the v30 configuration.
- **Per-joint braking:** each joint's step is reduced according to the nearest obstacle distance among the monitored distal links it moves. Joint limits are enforced after the update.
- **Task-dependent scale:** the later controller uses position and weighted orientation error to determine a nominal step, allowing orientation refinement to retain control authority after position error becomes small.
- **Learned step magnitude:** the later policy separates joint direction from a scalar step multiplier.
- **Temporal risk features:** clearance change and a one-step extrapolation of clearance provide a short history of obstacle approach. This extrapolation is not a full future collision rollout.

The v30 policy has a **58-dimensional observation and an eight-dimensional action**. Its observation consists of seven joint angles, eight task features, forty link-level geometric features, one time feature, and two temporal risk features. The first seven action components describe joint direction; the eighth controls a multiplier between 0.10 and 1.00. Direction normalization, adaptive scale, learned magnitude, per-joint braking, and joint-limit clipping jointly determine the executed increment.

An independent, zero-initialized risk-step head reads the two temporal features and adds a correction to the eighth action. During this fine-tuning mode, the existing direction actor is frozen; the risk head and relevant learning components are updated. This differs from learning a seven-dimensional obstacle residual around the PyBullet base policy.

The archived v30 configuration uses a 100-step training horizon. The demonstration allows 150 PPO steps while preserving the time observation denominator of 100. These settings must be reported separately because extending the horizon also extends the state distribution beyond the training episode length.

### Rewards and curriculum

The geometric branch replaces an absolute task-distance cost with a potential-difference task term, then adds task progress, orientation progress, clearance shaping, collision, joint-limit, action-magnitude, and step terms. The risk-step variant adds a penalty for large learned steps while closing on an obstacle in a risk region.

The v15 curriculum trains on free-space IK before introducing mixed zero/one-obstacle and zero-to-three-obstacle stages. Retaining free-space episodes during the later stages is intended to reduce forgetting. Reward and curriculum changes are engineering hypotheses supported by the archived experiments; cross-version results alone do not isolate each change's causal contribution.

### Training and checkpoint selection

The initial pose and perception branches use Stable-Baselines3 PPO. The later custom learner implements generalized advantage estimation, a clipped policy objective, value fitting, and entropy regularization. It collects on-policy transitions from batched kinematic environments; deterministic policy actions are used for the archived solver evaluations.

One separate, logged v15gpu training run used 4,096 parallel worlds, 64 collected steps per world, a 1,024-sample update batch, and five PPO epochs. Its reported 50,069,504 environment steps are transition counts, not independent evaluation episodes. These settings describe that run, not every version or the v30 evaluation. Buffer storage and gradient updates on the GPU also do not imply a zero-copy JAX-to-PyTorch pipeline. Training-period spot checks and final checkpoint comparisons serve different purposes; a high spot-check percentage is not a substitute for the documented evaluation sets.

## 6. Numerical refinement and runtime engineering

DLS and null-space avoidance provide numerical comparisons. The interface's plain DLS baseline does not actively avoid obstacles; its returned motion is collision-checked. The null-space method adds a secondary repulsive objective. These baselines differ from the hybrid refinement procedure, which starts from a PPO candidate and may run up to 100 additional DLS iterations with acceptance checks for pose improvement, clearance, and sampled trajectory collision. Refinement does not erase a collision already recorded in the PPO rollout.

The demonstration therefore gives PPO and the numerical baselines up to 150 steps or iterations, while the hybrid may perform additional work. Its results should not be described as a comparison under equal total iteration budgets.

Runtime work includes exporting policies to ONNX Runtime, evaluating FP32 and INT8 inference, and reducing overhead in the complete solver. A separate fast NumPy path optimizes FK, point transformations, SDF calculation, memory reuse, and state updates. Policy inference time and end-to-end solve time are distinct metrics; improvements in one do not automatically establish improvements in the other.

## 7. Evaluation boundary

The main reported geometric criterion requires position error below 10 mm, orientation error below 10°, and no detected collision over the checked trajectory samples. Terminal-only pose success is retained separately so a best-so-far returned configuration cannot hide an earlier collision.

PyBullet contact checks and sampled link/SDF checks are different collision models. Both are discrete. Neither establishes continuous swept-path safety, complete self-collision handling, or physical-robot safety.

The browser presents two algorithms on the same scene; it is a comparison view, not coordinated dual-arm control. Its animation is a visualization of computed configurations. The evidence supporting the portfolio is archived simulation and software validation, with experiment-specific limitations documented in [Experimental results](experimental_results.md).
