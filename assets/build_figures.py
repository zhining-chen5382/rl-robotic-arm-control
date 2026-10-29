"""Build portfolio figures from curated, read-only aggregate CSV data.

Requires an existing matplotlib installation. No project modules, models, or
original logs are loaded. Run: python -B assets/build_figures.py
"""
from pathlib import Path
import csv
import os

BASE = Path(__file__).resolve().parent
# Keep matplotlib's configuration and font cache inside the staging package.
CACHE = BASE.parent / "private_review" / ".matplotlib_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)
NAVY, BLUE, TEAL = "#18334D", "#456A91", "#087F83"
MUTED, BG, LINE, AMBER = "#53687B", "#F5F8FB", "#D8E3EC", "#9C640C"
WIDTH, PNG_DPI = 12.5, 68  # PNG previews are exactly 850 px wide.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "text.color": NAVY,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": NAVY,
    "svg.fonttype": "none", "svg.hashsalt": "robot-arm-portfolio-v2",
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})


def save(fig, name):
    """Keep the designed canvas instead of expanding it for stray labels."""
    fig.savefig(OUT / f"{name}.svg", metadata={
        "Creator": "Portfolio figure builder", "Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=PNG_DPI,
                metadata={"Creator": "Portfolio figure builder"})
    plt.close(fig)


def header(fig, eyebrow, title, subtitle):
    fig.text(.045, .965, eyebrow, fontsize=13, weight="bold", color=TEAL, va="top")
    fig.text(.045, .913, title, fontsize=25, weight="bold", va="top")
    fig.text(.045, .851, subtitle, fontsize=13.5, color=MUTED,
             va="top", linespacing=1.4)


def card(ax, x, y, w, h, title, body, accent=TEAL, dashed=False,
         title_size=15, body_size=14):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.012",
        edgecolor=accent if dashed else LINE, facecolor="white", linewidth=1.4,
        linestyle=(0, (4, 3)) if dashed else "-"))
    ax.plot([x+.015, x+.015], [y+.02, y+h-.02], color=accent, lw=3)
    ax.text(x+.032, y+h-.026, title, va="top", fontsize=title_size,
            weight="bold", color=accent)
    ax.text(x+.032, y+h-.075, body, va="top", fontsize=body_size,
            color=NAVY, linespacing=1.45)


def arrow(ax, start, end, color=MUTED, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.55,
        color=color, linestyle=(0, (4, 3)) if dashed else "-"))


def diagram_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def build_architecture():
    fig = plt.figure(figsize=(WIDTH, 10.5))
    header(fig, "SYSTEM DESIGN", "A shared evaluation path for PPO and hybrid IK",
           "v30 runtime: 58 inputs, 8 actions. Joint-position updates use kinematic FK and SDF.")
    ax = diagram_axes(fig, [.04, .145, .92, .64])
    card(ax, .02, .77, .42, .20, "01  Scene and state",
         "Start joints + target pose\nPrimitive obstacles + link SDF\nPose, clearance, temporal features")
    card(ax, .56, .77, .42, .20, "02  PPO joint update",
         "Direction + learned step magnitude\nAdaptive scale + per-joint braking\nClip joints; recompute kinematic FK")
    card(ax, .56, .43, .42, .22, "03  Rollout and candidate",
         "Store states + collision history\nKeep the best pose candidate\nBudget: up to 150 PPO steps")
    card(ax, .02, .43, .42, .22, "04  Final checks and metrics",
         "Full-point sampled collision checks\nFinal pose + rollout + any DLS path\nCompute pose errors and outcome")
    card(ax, .56, .07, .42, .22, "Optional DLS refinement",
         "Collision-free PPO rollouts only\nUp to 100 extra local iterations\nAdopt checked pose improvements", dashed=True)
    card(ax, .02, .07, .42, .22, "05  Compare and export",
         "3D comparison + scene replay\nPose / collision / latency metrics\nExperiment JSON and metrics CSV")
    arrow(ax, (.45, .87), (.55, .87))
    arrow(ax, (.77, .76), (.77, .66))
    # Iteration returns to the observation; finalization follows rollout stopping.
    ax.plot([.61, .61, .23], [.66, .71, .71], color=TEAL, lw=1.5)
    arrow(ax, (.23, .71), (.23, .76), color=TEAL)
    ax.text(.23, .685, "Repeat with updated pose + clearance",
            ha="center", va="top", fontsize=13, color=TEAL)
    # The pure-PPO path bypasses DLS. Both paths merge before final checks.
    arrow(ax, (.55, .555), (.45, .555), color=BLUE)
    ax.text(.50, .595, "PPO", ha="center", fontsize=13, color=BLUE)
    arrow(ax, (.77, .42), (.77, .30))
    ax.text(.80, .35, "Hybrid only", va="center", fontsize=13, color=MUTED)
    ax.plot([.55, .50, .50], [.18, .18, .555], color=MUTED, lw=1.5)
    ax.plot(.50, .555, "o", ms=4, color=MUTED)
    arrow(ax, (.23, .42), (.23, .30))
    fig.text(.045, .105,
             "Separate research branches: PointNet-PPO and seven-action frozen-base Residual PPO.",
             fontsize=13, color=MUTED)
    fig.text(.045, .045,
             "Discrete geometric checks in simulation; no continuous-motion or physical-robot safety proof.",
             fontsize=13, color=MUTED)
    save(fig, "system_architecture")


def build_evolution():
    fig = plt.figure(figsize=(WIDTH, 9.3))
    header(fig, "RESEARCH PROGRESSION", "A sequence of engineering questions",
           "Conceptual progression through implemented work; not a monotonic performance curve.")
    ax = diagram_axes(fig, [.04, .21, .92, .57])
    stages = [
        (.015, .62, "01  Pose IK", "PPO + numerical IK\nReward and step size\nPosition + orientation"),
        (.355, .62, "02  Perception", "PointNet point clouds\nFrozen-base residuals\nDiagnose failure modes"),
        (.695, .62, "03  Geometric control", "SDF link features\nAdaptive joint braking\nLearned step magnitude"),
        (.015, .04, "04  Evaluate + refine", "Pose and path metrics\nControlled ablations\nOptional DLS refinement"),
        (.355, .04, "05  Optimize runtime", "Profile FK / SDF costs\nONNX and NumPy paths\nPaired timing checks"),
        (.695, .04, "06  Integrate + compare", "Interactive evaluation\nScene / trajectory replay\nMetrics and exports"),
    ]
    for x, y, title, body in stages:
        card(ax, x, y, .285, .34, title, body, title_size=14.5, body_size=13.5)
    arrow(ax, (.307, .79), (.343, .79))
    arrow(ax, (.647, .79), (.683, .79))
    # A visible connection carries the implemented sequence from row 1 to row 2.
    ax.plot([.837, .837, .157], [.61, .50, .50], color=MUTED, lw=1.5)
    arrow(ax, (.157, .50), (.157, .395))
    ax.text(.50, .53, "Failure analysis informs the next design choice",
            ha="center", fontsize=13, color=MUTED)
    arrow(ax, (.307, .21), (.343, .21))
    arrow(ax, (.647, .21), (.683, .21))
    # Unsupported later work is kept outside the implemented sequence.
    note = diagram_axes(fig, [.04, .075, .92, .095])
    note.add_patch(FancyBboxPatch(
        (.005, .04), .99, .91, boxstyle="round,pad=0.004,rounding_size=0.025",
        edgecolor=AMBER, facecolor="white", linewidth=1.5,
        linestyle=(0, (4, 3))))
    note.text(.025, .68, "REPORTED LATER EXTENSION: WAYPOINT GUIDANCE",
              fontsize=13.5, weight="bold", color=AMBER, va="center")
    note.text(.025, .27,
              "Unverified: matching implementation and evaluation artifacts were unavailable.",
              fontsize=13, color=MUTED, va="center")
    save(fig, "research_evolution")


def read_csv(name):
    with (BASE / "data" / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def find_result(rows, experiment, group, model, seed):
    return next(row for row in rows if row["experiment_id"] == experiment
                and row["scene_group"] == group and row["model"] == model
                and row["seed"] == str(seed))


def clean_axis(ax):
    ax.grid(axis="x", color=LINE, linewidth=.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=13.5)
    ax.tick_params(axis="x", labelsize=13)


def panel_background(fig, rect):
    fig.add_artist(FancyBboxPatch(
        (rect[0], rect[1]), rect[2], rect[3], transform=fig.transFigure,
        boxstyle="round,pad=0.008,rounding_size=0.012", facecolor="white",
        edgecolor=LINE, linewidth=1.1, zorder=-1))


def build_results():
    rows = read_csv("verified_results.csv")
    fig = plt.figure(figsize=(WIDTH, 7.4))
    header(fig, "SELECTED EXPERIMENTS", "Verified trajectory-safe success",
           "Position <10 mm + orientation <10° + no final or checked-trajectory collision.")
    panel_background(fig, (.045, .19, .435, .59))
    panel_background(fig, (.535, .19, .42, .59))
    panels = [
        {
            "left": .16, "width": .285, "title_x": .065,
            "title": "v15 · paired n=764 · seed 2026",
            "rows": [find_result(rows, "EXP-HYBRID-V15", "obstacle_subset", "PPO", 2026),
                     find_result(rows, "EXP-HYBRID-V15", "obstacle_subset", "PPO+DLS", 2026)],
            "labels": ["PPO", "PPO + DLS"],
            "note": "Collision episodes: 57/764 in both\nDLS adds up to 100 iterations",
            "evidence": "Episode fields independently recounted",
        },
        {
            "left": .655, "width": .265, "title_x": .555,
            "title": "v30 · 1,000 cases per seed",
            "rows": [find_result(rows, "EXP-DEPLOY-V30", "obstacles", "v30 best", 2026),
                     find_result(rows, "EXP-DEPLOY-V30", "obstacles", "v30 best", 30303)],
            "labels": ["Seed 2026", "Seed 30303"],
            "note": "Trajectory collisions: 7.9% per seed\nSeparate evaluations of one model",
            "evidence": "Archived aggregate measurements",
        },
    ]
    for panel in panels:
        fig.text(panel["title_x"], .738, panel["title"],
                 fontsize=14.5, weight="bold", va="top")
        ax = fig.add_axes([panel["left"], .39, panel["width"], .27], facecolor="white")
        values = [float(row["success_rate_percent"]) for row in panel["rows"]]
        ax.barh([1, 0], values, color=[BLUE, TEAL], height=.49)
        for y, value, row in zip([1, 0], values, panel["rows"]):
            label = f"{value:.1f}%"
            if row["success_count"]:
                label += f"  ·  {row['success_count']}/{row['n']}"
            ax.text(3, y, label, color="white", fontsize=14, weight="bold", va="center")
        ax.set_yticks([1, 0])
        ax.set_yticklabels(panel["labels"])
        # Success uses the full 0–100 scale, without an extra axis range.
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(-.48, 1.48)
        ax.set_xlabel("Trajectory-safe success (%)", fontsize=13)
        clean_axis(ax)
        fig.text(panel["title_x"], .295, panel["note"], fontsize=13,
                 color=MUTED, va="top", linespacing=1.4)
        fig.text(panel["title_x"], .211, panel["evidence"],
                 fontsize=13, color=TEAL, va="bottom")
    fig.text(.045, .09,
             "Obstacle-containing cases. Distinct experiments; no cross-version causal comparison.",
             fontsize=13, color=MUTED)
    fig.text(.045, .043, "Collision checks are discrete geometric samples in simulation.",
             fontsize=13, color=MUTED)
    save(fig, "verified_results")


def find_timing(rows, group, path):
    return next(row for row in rows
                if row["scene_group"] == group and row["path"] == path)


def build_optimization():
    rows = read_csv("solver_optimization.csv")
    fig = plt.figure(figsize=(WIDTH, 8.6))
    header(fig, "SOLVER OPTIMIZATION", "Same policy, faster complete solves",
           "Fixed v16 checkpoint · NumPy backend · 256 inference points · trajectory recording on")
    cohorts = [
        ("mixed", .515, "90 mixed scenes · 21 without obstacles · seed 2026"),
        ("free", .215, "42 separate free-space scenes · seed 2027"),
    ]
    for group, bottom, title in cohorts:
        reference = find_timing(rows, group, "reference")
        optimized = find_timing(rows, group, "optimized")
        values = [float(reference["mean_ms"]), float(optimized["mean_ms"])]
        fig.text(.055, bottom+.23, title, fontsize=14, weight="bold")
        fig.text(.94, bottom+.23, f"{values[0] / values[1]:.2f}×", fontsize=17,
                 weight="bold", color=TEAL, ha="right")
        ax = fig.add_axes([.185, bottom, .755, .192])
        ax.barh([1, 0], values, height=.50, color=[BLUE, TEAL])
        for y, value in zip([1, 0], values):
            ax.text(value - 2, y, f"{value:.3f} ms", ha="right", va="center",
                    color="white", fontsize=14, weight="bold")
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["Reference", "Optimized"])
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 40, 80, 120, 160, 180])
        ax.set_ylim(-.46, 1.46)
        clean_axis(ax)
        if group == "free":
            ax.set_xlabel("Mean complete solve time (ms)", fontsize=13.5)
    fig.text(.055, .105, "Checked trajectories and outcomes match across all 132 cases.",
             fontsize=14, weight="bold")
    fig.text(.055, .054,
             "6 warmups per group · 6 alternating partitions · CPU model not recorded",
             fontsize=13, color=MUTED)
    save(fig, "solver_optimization")


def verify_curated_inputs():
    """Check the chart's experiment identities, denominators, and key values."""
    results = read_csv("verified_results.csv")
    for model, count in (("PPO", 645), ("PPO+DLS", 673)):
        row = find_result(results, "EXP-HYBRID-V15", "obstacle_subset", model, 2026)
        assert int(row["n"]) == 764 and int(row["success_count"]) == count
        assert int(row["trajectory_collision_count"]) == 57
    for seed, success in ((2026, 87.3), (30303, 85.2)):
        row = find_result(results, "EXP-DEPLOY-V30", "obstacles", "v30 best", seed)
        assert int(row["n"]) == 1000
        assert float(row["success_rate_percent"]) == success
        assert float(row["trajectory_collision_percent"]) == 7.9
    timings = read_csv("solver_optimization.csv")
    for row in timings:
        assert row["experiment_id"] == "EXP-FAST-V16"
        assert row["model"] == "v16 champion" and row["backend"] == "numpy"
        assert int(row["infer_max_points"]) == 256
        assert int(row["warmup_per_group"]) == 6 and int(row["batches"]) == 6
        assert int(row["n"]) == (90 if row["scene_group"] == "mixed" else 42)
        assert int(row["no_obstacle_cases"]) == (21 if row["scene_group"] == "mixed" else 42)
    expected = {
        ("mixed", "reference"): 160.76390777727485,
        ("mixed", "optimized"): 57.74800666614384,
        ("free", "reference"): 101.5561071489217,
        ("free", "optimized"): 23.847961900866636,
    }
    for (group, path), mean in expected.items():
        assert abs(float(find_timing(timings, group, path)["mean_ms"]) - mean) < 1e-10


def main():
    verify_curated_inputs()
    build_architecture()
    build_evolution()
    build_results()
    build_optimization()
    print("Built 4 SVG and 4 PNG figures; PNG previews are 850 px wide.")


if __name__ == "__main__":
    main()
