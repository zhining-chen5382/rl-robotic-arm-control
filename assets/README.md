# Visual assets and data

[Portfolio](../README.md) · [Demo overview](../demo/README.md) · [Recording guide](../demo/recording_guide.md)

The diagrams and charts here were created for this portfolio from the reviewed implementation and selected numerical records. They are explanatory graphics, not robot photographs or simulated experimental footage.

## Included

| Asset | Purpose | Evidence boundary |
|---|---|---|
| `figures/system_architecture.svg` / `.png` | Implemented scene-to-result solver flow | Geometry-regulated kinematic updates, optional DLS, sampled-state checking |
| `figures/research_evolution.svg` / `.png` | Six-stage research narrative | Conceptual sequence; overlapping branches; waypoint stage explicitly applicant-reported |
| `figures/verified_results.svg` / `.png` | v15 paired obstacle subset and separate v30 seed results | Distinct experiments in separate panels; no invented learning curve |
| `figures/solver_optimization.svg` / `.png` | Paired v16 complete-solve latency | 90 mixed and 42 separate free scenes; means, not per-case traces |
| `data/verified_results.csv` | Selected, normalized aggregate values used in the results chart | Includes evidence level and criterion; no original internal logs |
| `data/solver_optimization.csv` | Means, medians, p95, sample sizes, and fixed benchmark settings | No CPU model or package version has been inferred |
| `media/robot_arm_simulation.mp4` | Selected original, unedited 37-second screen recording | Historical simulation demonstration; exact checkpoint and refinement mode not established from the visible interface |

SVG provides scalable GitHub display; PNG is a viewing fallback. The charts use zero-based axes and do not invent confidence intervals. Values are rounded only for labels; CSVs retain measurement precision.

The presentation revision is designed for an approximately 850-pixel GitHub content column. Labels were shortened and enlarged, the optional DLS branch now has a separate path, and the reported waypoint extension is visually separated from implemented components. Version, sample count, scene type, and measurement units remain visible. The CSV values were not changed.

The original portfolio script `build_figures.py` regenerates all four diagrams/charts from the two CSVs using an existing Matplotlib installation. It imports no robot-project modules and loads no policies. This small visualization script was written for the portfolio; it is not a copy of the original robotics code.

## Media status

- **Hero video:** included as the applicant-selected original recording, unchanged. [Watch the recording](media/robot_arm_simulation.mp4). No new capture or editing is required for this version.
- **Algorithm comparison:** the same initial configuration, target, and obstacles for both views; disclose any extra DLS budget.
- **Trajectory inspection screenshot:** identify the active model and distinguish terminal from recorded-trajectory collision metrics.
- **Scene/export screenshot:** show scene reconstruction and exported metrics with personal paths and unrelated desktop content removed.

The additional screenshots listed above are optional future assets. The selected recording retains its original interface, audio, and on-screen interaction statistics; it is not a new benchmark. The [public source release](../code/README.md) separately includes the current application, one approved v30 inference checkpoint, and attributed resources; it does not establish the historical recording's exact checkpoint.

