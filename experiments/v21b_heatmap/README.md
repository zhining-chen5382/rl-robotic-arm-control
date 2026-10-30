# v21b: initial-distance and orientation-error analysis

This folder preserves the **aggregated bin tables** behind the distance–angle heatmap. The CSV and JSON are unchanged byte-for-byte copies of the retained measurement artifacts. They contain counts, rates, confidence intervals, and experiment metadata—not the original per-episode trajectories or model weights.

## Recorded experiment

| Setting | Recorded value |
| --- | --- |
| Policy | v21b best, disentangled direction/step actions |
| Scene set | 12,000 sampled scenes with 1–3 obstacles |
| Seed | 606060 |
| Inference point budget | 256 |
| Bin width | 0.1 m × 10 degrees |
| Success criterion | Position <10 mm, orientation <10°, and whole recorded trajectory free of detected obstacle collision |
| Trajectory-safe successes | **9,933 / 12,000 (82.775%)** |
| Episodes with a trajectory collision | **1,113 / 12,000 (9.275%)** |

The horizontal variable is the initial end-effector-to-target Euclidean distance. The vertical variable is the shortest quaternion rotation angle between the initial and target orientations. Neither is joint-space distance, obstacle clearance, or the length of the executed trajectory.

## Data and implementation

- [CSV bin table](data/v21b_success_by_initial_distance_angle_seed606060_n12000.csv): one row per distance/angle cell, including empty cells.
- [JSON aggregate record](data/v21b_success_by_initial_distance_angle_seed606060_n12000.json): definitions, run settings, overall counts, marginal summaries, and two-dimensional bins.
- [Retained evaluation script](../../code/scripts/eval_success_distance_angle_grid.py): the binning/evaluation implementation for inspection; its historical v21b checkpoint is not supplied.

Empty CSV rate fields mean no observations in that cell, not a 0% success rate. Cells with fewer than 30 observations should be visually de-emphasized, as in the original heatmap. The stored 95% intervals describe binomial uncertainty within a bin; they do not remove scene-distribution bias or establish causal effects.

## What the heatmap suggests

For the same 120–130° orientation-error interval, three sufficiently populated cells show:

| Initial distance | Successes / cases | Rate |
| --- | ---: | ---: |
| 0.3–0.4 m | 61 / 68 | 89.7% |
| 0.7–0.8 m | 110 / 133 | 82.7% |
| 1.1–1.2 m | 36 / 48 | 75.0% |

These cells illustrate a descriptive decline with larger initial distance. They do not establish a failure threshold or a statistically isolated distance effect: starting joints, obstacle layout, and other difficulty factors vary with the sampled scenes. Orientation difficulty is not simply monotonic—for example, at 0.7–0.8 m, the 70–80° cell is 54/73, while the 170–180° cell is 184/220.

This diagnosis can motivate a hypothesis that intermediate targets might help with larger global pose changes. **It is not a waypoint evaluation.** The project's approximately 93% waypoint result remains applicant-reported historical work without a matching retained implementation/evaluation record. It must not be substituted for the measured 82.775% v21b result or presented as an improvement demonstrated by this heatmap.

No simulation or training was rerun for this folder. The aggregate files can support redrawing the heatmap and checking totals; without per-episode records and the original checkpoint, they cannot independently reconstruct all trajectories or establish an untouched test set.

[Portfolio](../../README.md) · [Research journey](../../docs/research_journey.md) · [Experimental results](../../docs/experimental_results.md) · [Limitations](../../docs/limitations.md)
