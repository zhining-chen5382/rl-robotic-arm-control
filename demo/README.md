# Original simulation demonstration

**[Watch the original recording — 37 seconds](../assets/media/robot_arm_simulation.mp4)**

The portfolio uses the applicant's existing screen recording **unchanged**. It shows the historical robotic-arm simulation interface, obstacle scenes, an RL/DLS comparison, joint-angle displays, and pose/collision metrics. The file is H.264 MP4, 2560 × 1516, at 30 frames per second. Its original audio is retained.

This is an archived software demonstration. The exact checkpoint and refinement mode cannot be established from the visible labels, so the clip is not identified as v30, a waypoint implementation, or a controlled evaluation. The rolling on-screen statistics are interaction history rather than the success rates reported in [Experimental results](../docs/experimental_results.md).

No new robot execution, training, or recording was performed to prepare this media. The two displayed arms compare methods for one simulated task; they do not demonstrate cooperative dual-arm control.

## What the later documented application supports

The integrated application supports PPO, PPO + DLS refinement, plain DLS without active avoidance, and null-space avoidance. Its retained deployment selects v30 best, with up to 150 PPO steps and a time-observation denominator of 100; hybrid mode may add up to 100 DLS refinement iterations. These describe the documented later software, not a confirmed configuration of this historical clip.

The software also provides scene save/load, trajectory inspection, pose errors, terminal and sampled-trajectory collision checking, and JSON/CSV export. Features not shown in the recording are explained in the technical documents rather than implied to have been demonstrated by the clip.

## Media status

The selected video is the current demo asset; additional screenshots and a longer recording are optional. The [recording guide](recording_guide.md) is retained for a possible future replacement, not as unfinished work required for this portfolio version.

The README uses an ordinary relative link to the repository file. This avoids assuming that every GitHub account or Markdown context will provide an inline video player. The actual playback behavior should be checked after the authorized upload.

**Scope:** offline kinematic simulation and discrete geometric checks, without a continuous-motion or physical-robot safety guarantee.

[Portfolio](../README.md) · [Research journey](../docs/research_journey.md) · [Experimental results](../docs/experimental_results.md) · [Limitations](../docs/limitations.md)
