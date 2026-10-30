"""严格配对评估 frozen baseline 与 Residual Final，并诊断动作 clipping。

本脚本只执行 deterministic inference，不调用 ``model.learn()``。为规避当前
PyBullet 多连接风险，Baseline 和 Residual 分两个阶段顺序运行；每个 episode
都在 reset 前重置 Python、NumPy global、Torch 和 Gymnasium seed。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from ik_6d.frozen_base_policy import FrozenBasePolicy
from ik_6d.residual_pointnet_features import ResidualPointNetFeatureExtractor
from ik_6d.rl_env_6d_obstacle import RobotIK6DEnv as ObstacleIKEnv
from ik_6d.rl_env_6d_residual_obstacle import ResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "models" / "panda_ppo_ik_6d_map.zip"
RESIDUAL_MODEL = ROOT / "models" / "residual_obstacle" / "residual_final.zip"
BENCHMARK_DIR = ROOT / "benchmarks" / "residual_obstacle"
CASES_PATH = BENCHMARK_DIR / "paired_diagnostic_cases.json"
SUMMARY_PATH = BENCHMARK_DIR / "paired_diagnostic_summary.json"

DEFAULT_EPISODES = 1_000
DEFAULT_BASE_SEED = 2_026_081_300
BOOTSTRAP_SAMPLES = 20_000
SATURATION_THRESHOLD = 0.999
EPSILON = 1e-12


@dataclass(frozen=True)
class SceneSignature:
    state: np.ndarray
    target_position: np.ndarray
    target_orientation: np.ndarray
    obstacle_type: str
    obstacle_position: np.ndarray
    obstacle_size_key: str
    obstacle_size: np.ndarray
    point_cloud: np.ndarray


@dataclass(frozen=True)
class EpisodeResult:
    index: int
    seed: int
    final_position_mm: float
    final_orientation_deg: float
    had_collision: bool
    episode_min_distance_m: float
    steps: int

    @property
    def success_5(self) -> bool:
        return self.final_position_mm < 5.0 and self.final_orientation_deg < 10.0

    @property
    def success_10(self) -> bool:
        return self.final_position_mm < 10.0 and self.final_orientation_deg < 10.0

    @property
    def collision_free_5(self) -> bool:
        return self.success_5 and not self.had_collision

    @property
    def collision_free_10(self) -> bool:
        return self.success_10 and not self.had_collision


class ActionAccumulator:
    def __init__(self) -> None:
        self.base_actions: list[np.ndarray] = []
        self.scaled_residuals: list[np.ndarray] = []
        self.preclip_actions: list[np.ndarray] = []
        self.final_actions: list[np.ndarray] = []

    def add_base(self, base_action: np.ndarray) -> None:
        self.base_actions.append(np.asarray(base_action, dtype=np.float32).copy())

    def add_residual(
        self,
        base_action: np.ndarray,
        scaled_residual: np.ndarray,
        final_action: np.ndarray,
    ) -> None:
        base = np.asarray(base_action, dtype=np.float32)
        scaled = np.asarray(scaled_residual, dtype=np.float32)
        final = np.asarray(final_action, dtype=np.float32)
        self.base_actions.append(base.copy())
        self.scaled_residuals.append(scaled.copy())
        self.preclip_actions.append((base + scaled).copy())
        self.final_actions.append(final.copy())

    def base_summary(self) -> dict:
        actions = np.asarray(self.base_actions, dtype=np.float64)
        saturated = np.abs(actions) >= SATURATION_THRESHOLD
        return {
            "steps": int(actions.shape[0]),
            "base_action_dimension_saturation_rate": float(np.mean(saturated)),
            "mean_abs_base_action": float(np.mean(np.abs(actions))),
            "per_joint": [
                {
                    "joint": joint,
                    "base_saturation_rate": float(np.mean(saturated[:, joint])),
                    "mean_abs_base_action": float(np.mean(np.abs(actions[:, joint]))),
                }
                for joint in range(actions.shape[1])
            ],
        }

    def residual_summary(self) -> dict:
        base = np.asarray(self.base_actions, dtype=np.float64)
        scaled = np.asarray(self.scaled_residuals, dtype=np.float64)
        preclip = np.asarray(self.preclip_actions, dtype=np.float64)
        final = np.asarray(self.final_actions, dtype=np.float64)
        clipped_amount = np.abs(preclip - final)
        clipped_mask = clipped_amount > 0.0
        step_any_clip = np.any(clipped_mask, axis=1)
        effective_residual = final - base
        desired_magnitude = float(np.sum(np.abs(scaled)))
        effective_magnitude = float(np.sum(np.abs(effective_residual)))
        effective_fraction = (
            effective_magnitude / desired_magnitude
            if desired_magnitude > EPSILON
            else float("nan")
        )
        saturated = np.abs(base) >= SATURATION_THRESHOLD

        return {
            "steps": int(base.shape[0]),
            "step_any_clip_rate": float(np.mean(step_any_clip)),
            "dimension_clip_rate": float(np.mean(clipped_mask)),
            "mean_clipped_amount": float(np.mean(clipped_amount)),
            "mean_clipped_amount_given_clip": (
                float(np.mean(clipped_amount[clipped_mask]))
                if np.any(clipped_mask)
                else 0.0
            ),
            "max_clipped_amount": float(np.max(clipped_amount)),
            "residual_effective_fraction": float(effective_fraction),
            "desired_scaled_residual_abs_sum": desired_magnitude,
            "effective_residual_abs_sum": effective_magnitude,
            "mean_scaled_residual_l2": float(np.mean(np.linalg.norm(scaled, axis=1))),
            "mean_abs_scaled_residual": float(np.mean(np.abs(scaled))),
            "base_action_dimension_saturation_rate": float(np.mean(saturated)),
            "mean_abs_base_action": float(np.mean(np.abs(base))),
            "per_joint": [
                {
                    "joint": joint,
                    "dimension_clip_rate": float(np.mean(clipped_mask[:, joint])),
                    "mean_clipped_amount": float(np.mean(clipped_amount[:, joint])),
                    "mean_clipped_amount_given_clip": (
                        float(np.mean(clipped_amount[clipped_mask[:, joint], joint]))
                        if np.any(clipped_mask[:, joint])
                        else 0.0
                    ),
                    "base_saturation_rate": float(np.mean(saturated[:, joint])),
                    "mean_abs_base_action": float(np.mean(np.abs(base[:, joint]))),
                    "mean_abs_scaled_residual": float(np.mean(np.abs(scaled[:, joint]))),
                    "residual_effective_fraction": (
                        float(np.sum(np.abs(effective_residual[:, joint])))
                        / float(np.sum(np.abs(scaled[:, joint])))
                        if float(np.sum(np.abs(scaled[:, joint]))) > EPSILON
                        else float("nan")
                    ),
                }
                for joint in range(base.shape[1])
            ],
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_scene(env: ObstacleIKEnv, observation: dict) -> SceneSignature:
    info = env.obstacle_info
    if info is None:
        raise RuntimeError("reset 后 obstacle_info 为空")
    obstacle_type = str(info["type"])
    if obstacle_type == "sphere":
        size_key = "radius"
        size = np.asarray([info["radius"]], dtype=np.float64)
    elif obstacle_type == "box":
        size_key = "half_extents"
        size = np.asarray(info["half_extents"], dtype=np.float64)
    else:
        raise RuntimeError(f"未知障碍物类型: {obstacle_type}")
    return SceneSignature(
        state=np.asarray(observation["state"], dtype=np.float32).copy(),
        target_position=np.asarray(env.target_pos, dtype=np.float32).copy(),
        target_orientation=np.asarray(env.target_quat, dtype=np.float32).copy(),
        obstacle_type=obstacle_type,
        obstacle_position=np.asarray(info["position"], dtype=np.float64).copy(),
        obstacle_size_key=size_key,
        obstacle_size=size.copy(),
        point_cloud=np.asarray(observation["point_cloud"], dtype=np.float32).copy(),
    )


def assert_same_scene(expected: SceneSignature, actual: SceneSignature, index: int, seed: int) -> None:
    scalar_fields = ("obstacle_type", "obstacle_size_key")
    for field in scalar_fields:
        if getattr(expected, field) != getattr(actual, field):
            raise RuntimeError(
                f"paired scene mismatch: episode={index}, seed={seed}, field={field}, "
                f"baseline={getattr(expected, field)!r}, residual={getattr(actual, field)!r}"
            )
    array_fields = (
        "state",
        "target_position",
        "target_orientation",
        "obstacle_position",
        "obstacle_size",
        "point_cloud",
    )
    for field in array_fields:
        left = getattr(expected, field)
        right = getattr(actual, field)
        if not np.array_equal(left, right):
            max_difference = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            raise RuntimeError(
                f"paired scene mismatch: episode={index}, seed={seed}, field={field}, "
                f"max_abs_difference={max_difference}"
            )


def run_baseline(episodes: int, base_seed: int):
    seed_everything(base_seed)
    env = ObstacleIKEnv()
    policy = FrozenBasePolicy(BASE_MODEL)
    scenes: list[SceneSignature] = []
    results: list[EpisodeResult] = []
    actions = ActionAccumulator()
    try:
        for index in range(episodes):
            episode_seed = base_seed + index
            seed_everything(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            scenes.append(capture_scene(env, observation))
            episode_min = float(env._get_min_obstacle_distance())
            final_info = None
            steps = 0
            for _ in range(env.max_steps):
                base_action = policy.predict(observation["state"])
                actions.add_base(base_action)
                observation, _, terminated, truncated, info = env.step(base_action)
                final_info = info
                steps += 1
                episode_min = min(episode_min, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError("Baseline episode 未执行 step")
            results.append(
                EpisodeResult(
                    index=index,
                    seed=episode_seed,
                    final_position_mm=float(final_info["pos_distance"]) * 1000.0,
                    final_orientation_deg=float(final_info["orient_distance_deg"]),
                    had_collision=bool(final_info.get("had_collision", False)),
                    episode_min_distance_m=episode_min,
                    steps=steps,
                )
            )
            if (index + 1) % 100 == 0:
                print(f"Baseline paired episodes: {index + 1}/{episodes}", flush=True)
    finally:
        env.close()
    return scenes, results, actions


def run_residual(expected_scenes: list[SceneSignature], episodes: int, base_seed: int):
    seed_everything(base_seed)
    env = ResidualObstacleEnv(residual_scale=0.05, residual_penalty_weight=1e-4)
    model = PPO.load(str(RESIDUAL_MODEL), device="cpu")
    results: list[EpisodeResult] = []
    actions = ActionAccumulator()
    try:
        for index in range(episodes):
            episode_seed = base_seed + index
            seed_everything(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            actual_scene = capture_scene(env.unwrapped, observation)
            assert_same_scene(expected_scenes[index], actual_scene, index, episode_seed)
            episode_min = float(env.unwrapped._get_min_obstacle_distance())
            final_info = None
            steps = 0
            for _ in range(env.unwrapped.max_steps):
                residual_action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(residual_action)
                actions.add_residual(
                    info["base_action"], info["scaled_residual"], info["final_action"]
                )
                final_info = info
                steps += 1
                episode_min = min(episode_min, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError("Residual episode 未执行 step")
            results.append(
                EpisodeResult(
                    index=index,
                    seed=episode_seed,
                    final_position_mm=float(final_info["pos_distance"]) * 1000.0,
                    final_orientation_deg=float(final_info["orient_distance_deg"]),
                    had_collision=bool(final_info.get("had_collision", False)),
                    episode_min_distance_m=episode_min,
                    steps=steps,
                )
            )
            if (index + 1) % 100 == 0:
                print(f"Residual paired episodes: {index + 1}/{episodes}", flush=True)
    finally:
        env.close()
    return results, actions


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - half, center + half]


def paired_bootstrap_ci(
    residual_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 73_001,
) -> list[float]:
    differences = residual_values.astype(np.float64) - baseline_values.astype(np.float64)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    chunk = 500
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        bootstrap_means[start : start + count] = np.mean(differences[indices], axis=1)
    return [float(x) for x in np.percentile(bootstrap_means, [2.5, 97.5])]


def exact_mcnemar(discordant_a: int, discordant_b: int) -> dict:
    total = discordant_a + discordant_b
    if total == 0:
        return {"discordant_total": 0, "two_sided_exact_p": 1.0}
    smaller = min(discordant_a, discordant_b)
    lower_tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2.0**total)
    return {
        "discordant_total": total,
        "two_sided_exact_p": float(min(1.0, 2.0 * lower_tail)),
    }


def summarize_policy(results: list[EpisodeResult]) -> dict:
    total = len(results)
    binary_names = (
        "success_5",
        "success_10",
        "had_collision",
        "collision_free_5",
        "collision_free_10",
    )
    summary: dict[str, object] = {"episodes": total}
    for name in binary_names:
        values = np.asarray([bool(getattr(item, name)) for item in results])
        count = int(np.sum(values))
        summary[name] = {
            "count": count,
            "rate": count / total,
            "wilson_95_ci": wilson_interval(count, total),
        }
    summary.update(
        {
            "mean_final_position_error_mm": float(
                np.mean([item.final_position_mm for item in results])
            ),
            "mean_final_orientation_error_deg": float(
                np.mean([item.final_orientation_deg for item in results])
            ),
            "mean_episode_minimum_obstacle_distance_m": float(
                np.mean([item.episode_min_distance_m for item in results])
            ),
            "mean_steps": float(np.mean([item.steps for item in results])),
        }
    )
    return summary


def transition_matrix(
    baseline_values: np.ndarray, residual_values: np.ndarray, labels: tuple[str, str]
) -> dict:
    false_label, true_label = labels
    return {
        f"baseline_{true_label}__residual_{true_label}": int(
            np.sum(baseline_values & residual_values)
        ),
        f"baseline_{true_label}__residual_{false_label}": int(
            np.sum(baseline_values & ~residual_values)
        ),
        f"baseline_{false_label}__residual_{true_label}": int(
            np.sum(~baseline_values & residual_values)
        ),
        f"baseline_{false_label}__residual_{false_label}": int(
            np.sum(~baseline_values & ~residual_values)
        ),
    }


def case_record(baseline: EpisodeResult, residual: EpisodeResult) -> dict:
    return {
        "episode_index": baseline.index,
        "episode_seed": baseline.seed,
        "baseline": asdict(baseline),
        "residual": asdict(residual),
    }


def build_cases(baseline: list[EpisodeResult], residual: list[EpisodeResult]) -> dict:
    categories = {
        "baseline_collision_residual_no_collision": [],
        "baseline_no_collision_residual_collision": [],
        "baseline_cf10_fail_residual_cf10_success": [],
        "baseline_cf10_success_residual_cf10_fail": [],
    }
    for base_item, residual_item in zip(baseline, residual):
        if base_item.had_collision and not residual_item.had_collision:
            categories["baseline_collision_residual_no_collision"].append(
                case_record(base_item, residual_item)
            )
        if not base_item.had_collision and residual_item.had_collision:
            categories["baseline_no_collision_residual_collision"].append(
                case_record(base_item, residual_item)
            )
        if not base_item.collision_free_10 and residual_item.collision_free_10:
            categories["baseline_cf10_fail_residual_cf10_success"].append(
                case_record(base_item, residual_item)
            )
        if base_item.collision_free_10 and not residual_item.collision_free_10:
            categories["baseline_cf10_success_residual_cf10_fail"].append(
                case_record(base_item, residual_item)
            )
    return {
        key: {"total_count": len(items), "saved_cases": items[:20]}
        for key, items in categories.items()
    }


def paired_analysis(baseline: list[EpisodeResult], residual: list[EpisodeResult]) -> dict:
    base_collision = np.asarray([item.had_collision for item in baseline], dtype=bool)
    residual_collision = np.asarray([item.had_collision for item in residual], dtype=bool)
    base_cf10 = np.asarray([item.collision_free_10 for item in baseline], dtype=bool)
    residual_cf10 = np.asarray([item.collision_free_10 for item in residual], dtype=bool)

    collision_matrix = transition_matrix(
        base_collision, residual_collision, ("no_collision", "collision")
    )
    cf10_matrix = transition_matrix(base_cf10, residual_cf10, ("fail", "success"))
    collision_rescued = int(np.sum(base_collision & ~residual_collision))
    collision_created = int(np.sum(~base_collision & residual_collision))
    cf10_gained = int(np.sum(~base_cf10 & residual_cf10))
    cf10_lost = int(np.sum(base_cf10 & ~residual_cf10))

    return {
        "collision_transition_matrix": collision_matrix,
        "collision_rate_difference_residual_minus_baseline": float(
            np.mean(residual_collision) - np.mean(base_collision)
        ),
        "collision_rate_difference_paired_bootstrap_95_ci": paired_bootstrap_ci(
            residual_collision, base_collision, seed=73_001
        ),
        "collision_exact_mcnemar": {
            **exact_mcnemar(collision_rescued, collision_created),
            "baseline_collision_to_residual_no_collision": collision_rescued,
            "baseline_no_collision_to_residual_collision": collision_created,
        },
        "collision_free_10mm_transition_matrix": cf10_matrix,
        "collision_free_10mm_difference_residual_minus_baseline": float(
            np.mean(residual_cf10) - np.mean(base_cf10)
        ),
        "collision_free_10mm_difference_paired_bootstrap_95_ci": paired_bootstrap_ci(
            residual_cf10, base_cf10, seed=73_002
        ),
        "collision_free_10mm_exact_mcnemar": {
            **exact_mcnemar(cf10_gained, cf10_lost),
            "baseline_fail_to_residual_success": cf10_gained,
            "baseline_success_to_residual_fail": cf10_lost,
        },
    }


def print_report(summary: dict) -> None:
    print("\n" + "=" * 80)
    print("STRICT PAIRED DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"episodes: {summary['episodes']}")
    print(f"all_reset_scenes_exactly_equal: {summary['all_reset_scenes_exactly_equal']}")
    for policy_name in ("baseline", "residual"):
        item = summary[policy_name]
        print(f"\n[{policy_name}]")
        print(f"5mm+10deg IK:       {item['success_5']['rate'] * 100:.2f}%")
        print(f"10mm+10deg IK:      {item['success_10']['rate'] * 100:.2f}%")
        print(f"collision:          {item['had_collision']['rate'] * 100:.2f}%")
        print(f"collision-free 5mm: {item['collision_free_5']['rate'] * 100:.2f}%")
        print(f"collision-free 10mm:{item['collision_free_10']['rate'] * 100:.2f}%")
        print(f"mean position:      {item['mean_final_position_error_mm']:.3f} mm")
        print(f"mean orientation:   {item['mean_final_orientation_error_deg']:.3f} deg")
        print(f"mean min distance:  {item['mean_episode_minimum_obstacle_distance_m']:.5f} m")
    paired = summary["paired_analysis"]
    print("\n[paired collision]")
    print(json.dumps(paired["collision_transition_matrix"], ensure_ascii=False, indent=2))
    print(
        "difference residual-baseline: "
        f"{paired['collision_rate_difference_residual_minus_baseline'] * 100:.3f} pp"
    )
    print(
        "paired bootstrap 95% CI: "
        f"{[x * 100 for x in paired['collision_rate_difference_paired_bootstrap_95_ci']]} pp"
    )
    print(json.dumps(paired["collision_exact_mcnemar"], ensure_ascii=False, indent=2))
    print("\n[paired collision-free 10mm]")
    print(json.dumps(paired["collision_free_10mm_transition_matrix"], ensure_ascii=False, indent=2))
    print(
        "difference residual-baseline: "
        f"{paired['collision_free_10mm_difference_residual_minus_baseline'] * 100:.3f} pp"
    )
    print(
        "paired bootstrap 95% CI: "
        f"{[x * 100 for x in paired['collision_free_10mm_difference_paired_bootstrap_95_ci']]} pp"
    )
    print(json.dumps(paired["collision_free_10mm_exact_mcnemar"], ensure_ascii=False, indent=2))
    print("\n[action diagnostics]")
    print(json.dumps(summary["baseline_action_diagnostics"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["residual_action_diagnostics"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes 必须大于 0")
    if not BASE_MODEL.is_file() or not RESIDUAL_MODEL.is_file():
        raise FileNotFoundError("baseline 或 residual_final 模型不存在")

    print("Phase 1/2: running frozen baseline sequentially", flush=True)
    scenes, baseline_results, baseline_actions = run_baseline(args.episodes, args.base_seed)
    print("Baseline environment closed", flush=True)
    print("Phase 2/2: running residual and checking every reset scene", flush=True)
    residual_results, residual_actions = run_residual(scenes, args.episodes, args.base_seed)
    print("Residual environment closed; every paired reset scene matched exactly", flush=True)

    cases = build_cases(baseline_results, residual_results)
    summary = {
        "episodes": args.episodes,
        "base_seed": args.base_seed,
        "all_reset_scenes_exactly_equal": True,
        "scene_fields_compared": [
            "initial_state",
            "target_position",
            "target_orientation",
            "obstacle_type",
            "obstacle_position",
            "obstacle_size",
            "full_512x3_point_cloud",
        ],
        "baseline": summarize_policy(baseline_results),
        "residual": summarize_policy(residual_results),
        "paired_analysis": paired_analysis(baseline_results, residual_results),
        "baseline_action_diagnostics": baseline_actions.base_summary(),
        "residual_action_diagnostics": residual_actions.residual_summary(),
        "case_category_counts": {key: value["total_count"] for key, value in cases.items()},
        "statistics": {
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "mcnemar": "two-sided exact binomial test on discordant pairs",
            "individual_binary_intervals": "Wilson 95% confidence intervals",
        },
        "reproducibility_limit": (
            "The existing environment partly uses global np.random. This script reseeds "
            "Python, global NumPy, Torch and env.reset before every episode and verifies "
            "all requested reset fields plus the full point cloud exactly. Results are "
            "strictly paired for this run, but cross-implementation reproducibility is "
            "not guaranteed."
        ),
    }

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary)
    print(f"\ncases saved: {CASES_PATH}")
    print(f"summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
