# Recording the developed simulation application

**Optional future guide.** The portfolio now uses an existing 37-second recording unchanged: [watch the selected video](../assets/media/robot_arm_simulation.mp4). No new recording or editing is required for this version. The steps below remain available for a future replacement; their v30 setup is not an identification of the archived video's checkpoint.

This guide plans a 60–90 second capture of the existing application. It was prepared from source inspection and retained artifacts: **no application was started and no new run, screenshot, or video was produced**. The earlier 16 method/scene checks and browser exports are historical functional evidence, not a fresh verification.

Use an existing authorized local installation and available recorder. This portfolio does not contain runnable application source, models, robot assets, or scene inputs. No installation or training is needed for this recording plan.

## 1. Restore a reproducible scene

Use an existing portable scene saved with **`保存场景` — Save scene**. Keep its exact start, target, orientation, and obstacle dimensions in the private capture record. Import it with **`加载场景` — Load scene** and wait for `场景已加载；起点、目标与障碍物已恢复` (scene loaded; start, target, and obstacles restored). Loading also restores the saved method, so select the recording method afterward.

Import expects one object, not an array of demo cases or the entire experiments-export envelope:

| Field | Meaning |
| --- | --- |
| `schema_version`, `robot_id` | Version 1 and `panda`; omitted values receive these defaults. |
| `target_pos` | Required three finite values, in metres. |
| `target_rpy` | Roll, pitch, yaw in degrees; defaults to zeros. |
| `target_quat` | Optional finite nonzero quaternion `[x,y,z,w]`; normalized and used instead of RPY. |
| `start_q` | Optional seven finite joint angles, in radians; omission uses home. |
| `obstacles` | Zero to three `box`, `sphere`, or `cylinder` objects, with metre-based positions and dimensions. |
| `method` | `rl_avoid`, `rl_avoid_dls`, `dls_avoid`, or `nullspace`; default `rl_avoid`. |

Boxes use positive `size.halfExtents`—half the visible edge lengths. Spheres use positive `size.radius`; cylinders use positive `size.radius` and full `size.height`. Files must be no larger than 2 MB. Input validation does not certify reachability or safety.

Alternatively, prepare a new scene with the UI and save it before comparing methods. Identify it as a newly selected demonstration case, rather than a replay of a historical benchmark. No new scene file is created by this guide.

## 2. Know the actual controls

| Control | Behavior |
| --- | --- |
| `目标位置` | X/Y/Z in metres; displayed Panda bounds are X/Y −1 to 1, Z 0 to 1.2. These limits do not guarantee reachability. |
| `目标姿态 (RPY°)` | Degrees, with controls from −180 to 180. Editing an angle clears an imported quaternion override. |
| `障碍物` → `数量` → `随机生成` | Requests one to three random corridor obstacles. Check the actual count; there are no individual shape/size editors here. |
| `清除` | Removes obstacles. Zero is obtained here or by import, not through the count selector. |
| `显示运动轨迹动画` | Replays returned configurations after the solve. |
| `求解IK` | Restores the imported start, or home, then solves both views under the same conditions. |
| Large `重置`, beside Solve | Returns to home and clears the imported start and quaternion. Do not use it between methods; reimport if pressed. |
| Small `重置`, in `求解统计` | Clears interactive statistics only; exported experiments remain accumulated. |

Left drag rotates, the wheel zooms, and right drag pans. A canvas click also changes the target on the Z = 0.5 m plane with a reach projection. Frame first, then reimport the saved scene. Scroll the control panel using its scrollbar, rather than changing the camera with the wheel over the canvas.

## 3. Select the real comparison

Choose **`RL 避障 + DLS 精调 (混合推理)`** (`rl_avoid_dls`) in **`IK求解方法 (左臂)`**. The right view automatically uses **`DLS (不避障, 对照组)`**. Its API key is `dls_avoid`, but its iterations have no active obstacle-avoidance term; collision is checked afterward.

There is no right-method selector. Left-side PPO, hybrid, or null-space selection produces plain DLS on the right. Selecting plain DLS on the left instead produces null-space avoidance on the right.

The canvas labels are legacy fixed text: `左: RL 避障` and `右: DLS`. Hybrid may appear as `rl_avoid_dls` in the result table. Show the actual dropdown and add the caption **“Left: PPO + DLS refinement; right: plain DLS.”** Verify it against exported `scene.method` or `result.requested_method`; do not edit the application or its displayed results.

These are two spatially offset views of one arm's task. Backend requests share serialized solver state; this is not cooperative dual-arm control or a parallel-throughput test.

## 4. Prepare a clean take

1. Wait for startup and warmup. Confirm v30 best and the 150-step PPO limit. Hybrid may add 100 DLS iterations; the time-observation denominator remains 100.
2. Load the scene, select hybrid, uncheck trajectory animation, and run a preparation solve. Warm optional PPO/null-space paths too if they will be shown.
3. Frame both robots. The right panel scrolls: plan separate setup and results views, since its file controls and lower metrics may not fit together.
4. Refresh the page to clear the in-memory export list, wait for the viewer, and **reload the same scene**. Keep the server running so its initialized solvers remain available.
5. Select hybrid again and check animation. The small statistics reset can clear its panel, but cannot remove warmup rows from exports; this is why the refresh matters.
6. Start recording with personal paths, terminals, unrelated tabs, and notifications outside the frame. During the comparison, do not edit targets, regenerate obstacles, or press the large Reset.

Compilation or initialization occurring inside a cold solve can affect the displayed time. Record that condition honestly; a startup delay is not an algorithmic speed comparison.

## 5. An approximately 85-second shot plan

| Time | Action | Suggested caption |
| --- | --- | --- |
| 0–8 s | Hold application/model context and loaded viewer. | “Seven-DoF collision-aware IK · offline kinematic simulation” |
| 8–20 s | Show loaded confirmation, target, actual obstacle count, and hybrid selection. | “Same saved start, target pose, and obstacles” |
| 20–40 s | Click `求解IK` and let both returned trajectories play. | “Left: PPO + DLS refinement · Right: plain DLS” |
| 40–58 s | Scroll the panel to `状态信息 (对比)` and hold the table. | “Pose error and sampled-trajectory checks from this solve” |
| 58–75 s | Save scene; export experiments JSON and metrics CSV. Keep save paths off-screen. | “Scene restoration and inspectable experiment exports” |
| 75–85 s | Hold a clear final scene/results view. | “One demonstration case; aggregate evaluation reported separately” |

Adjust shot length to the actual trajectory. Disclose cuts or playback speed changes; preserve failed outcomes and collision information. Manual animation waits approximately 25 ms per paired configuration plus FK requests and rendering. Playback duration is not solve time or a physical control clock.

Aim for **1920 × 1080, 30 fps, H.264 MP4** where an existing recorder supports it, while keeping interface text readable. Use an available Windows recorder or existing OBS installation; no download is required. Retain the original recording and edit notes privately. This is a local capture/export plan, not upload instructions.

## 6. Read the metrics and preserve the take

| Result label | Interpretation |
| --- | --- |
| `求解状态` | Success/failure under the sampled-trajectory criterion. |
| `位置误差`, `姿态误差` | Final errors, in mm and degrees. |
| `最小距离` | Clearance at the returned configuration, not the minimum over the entire trajectory. |
| `终态碰撞` | Terminal collision indicator/count under the implementation's metric. |
| `轨迹碰撞` | Number of sampled configurations flagged as colliding, not separate collision events. |
| `耗时` | Backend solve and trajectory recheck time; whole ms on screen, more precision in exports. Subsequent playback/rendering are outside this measurement; compilation inside it can contribute. |

`求解统计` retains up to 100 interactive records per side. Its rates and combined error/time summaries describe that interaction history, not a validation set.

Save the scene, `experiments.json`, and `metrics.csv` together privately. One ordinary Solve action produces two method results; identify them by method and conditions, not row order. JSON preserves scene and deployment metadata. CSV records method, success, errors, terminal collisions, trajectory collisions, and time, but cannot alone reconstruct the scene/version.

## 7. Optional shots and final captions

**Pure PPO → hybrid:** restore the same scene, solve with `RL 避障 (PPO collision-aware)`, wait, then select hybrid and solve again without editing inputs or pressing the large Reset. Both calls restore the same imported start internally. These are sequential left-side comparisons; the right remains DLS. Claim improvement only when the actual exports show it. Equal results are valid.

**DLS → null-space:** choose `DLS (不避障, 对照组)` on the left for this separate comparison. Account for the fixed canvas labels when captioning.

**Automatic demo:** `开始自动演示` runs bundled cases in order; `停止演示` stops progression, but wait for current animation before manual interaction. There is no case picker. Playback uses approximately 80 ms frame delays and five/eight-second start/end holds. Saving manual controls afterward does not reliably preserve that case's start/quaternion, so reimport the portable scene for the main take.

For a results screenshot, retain both scene views and the readable result table, with the method selection or a verified caption:

> Offline simulation, v30 best. Left: PPO + DLS refinement; right: plain DLS. Same saved starting joints, target pose, and obstacles. Displayed values are from this take; collision checks use sampled geometry at recorded configurations.

If listing numbers, transcribe each method's actual errors, terminal collision result, trajectory-collision frame count, and time from the new export. Do not reuse historical functional-check values as fresh results or attach the approximately 93% historical claim to a one-case capture. Review visible information and asset rights before publishing media.

[Demo overview](README.md) · [Experimental results](../docs/experimental_results.md) · [Limitations](../docs/limitations.md)
