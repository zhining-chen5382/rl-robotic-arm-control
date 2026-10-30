# Original simulation demonstration

**[Watch the original recording — 37 seconds](../assets/media/robot_arm_simulation.mp4)**

The original 37-second screen recording shows the historical simulation interface, obstacle scenes, an RL/DLS comparison, joint-angle displays, and pose/collision metrics. The H.264 MP4 (2560 × 1516, 30 fps), including its original audio, is preserved unchanged.

The checkpoint and refinement mode cannot be identified from the visible labels. The clip is presented as a software demonstration; its rolling on-screen statistics are interaction history rather than the benchmark results reported in [Experimental results](../docs/experimental_results.md).

The two views show different control methods on the same simulated task.

## What the later documented application supports

The integrated application supports PPO, PPO + DLS refinement, plain DLS without active avoidance, and null-space avoidance. Its retained deployment selects v30 best, with up to 150 PPO steps and a time-observation denominator of 100; hybrid mode may add up to 100 DLS refinement iterations. These describe the documented later software, not a confirmed configuration of this historical clip.

The software also provides scene save/load, trajectory inspection, pose errors, terminal and sampled-trajectory collision checking, and JSON/CSV export. Features not shown in the recording are explained in the technical documents rather than implied to have been demonstrated by the clip.

## Media status

The 37-second recording is the selected demonstration asset. The [recording guide](recording_guide.md) is retained as a reference for future updates.

The main README provides a GitHub video-attachment preview and a relative link to the unchanged original recording.

**Scope:** offline kinematic simulation and discrete geometric checks, without a continuous-motion or physical-robot safety guarantee.

[Portfolio](../README.md) · [Research journey](../docs/research_journey.md) · [Experimental results](../docs/experimental_results.md) · [Limitations](../docs/limitations.md)
