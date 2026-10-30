"""Strict Frozen/Uncentered/Robot-Centered 1000-pair benchmark."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict as dataclass_asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from diagnose_direct_vs_headroom import (
    ActionDiagnostics,
    EpisodeResult,
    SceneSignature,
    asdict,
    capture_scene,
    exact_mcnemar,
    paired_bootstrap_ci,
    summarize,
)
from ik_6d.frozen_base_policy import FrozenBasePolicy
from ik_6d.residual_relative_features import ResidualRelativeFeatureExtractor  # noqa: F401
from ik_6d.rl_env_6d_obstacle import RobotIK6DEnv as ObstacleIKEnv
from ik_6d.rl_env_6d_residual_relative import RelativeResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "models" / "panda_ppo_ik_6d_map.zip"
UNCENTERED_MODEL = (
    ROOT / "models" / "residual_relative_ablation" / "uncentered" / "final.zip"
)
ROBOT_MODEL = (
    ROOT / "models" / "residual_relative_ablation" / "robot_centered" / "final.zip"
)
OUTPUT_ROOT = ROOT / "benchmarks" / "residual_relative_ablation"
SUMMARY_PATH = OUTPUT_ROOT / "relative_three_model_summary.json"
CASES_PATH = OUTPUT_ROOT / "relative_three_model_cases.json"
EPISODES_PATH = OUTPUT_ROOT / "relative_three_model_episodes.json"
EPISODES = 1_000
BASE_SEED = 2_026_081_300
SENSITIVITY_EPISODES = 200
FROZEN_REFERENCE = {
    "ik_success_5mm_10deg": 0.819,
    "ik_success_10mm_10deg": 0.877,
    "collision_rate": 0.227,
    "collision_free_5mm_success": 0.676,
    "collision_free_10mm_success": 0.685,
    "mean_final_position_mm": 9.260,
    "mean_final_orientation_deg": 7.914,
    "mean_episode_min_distance_m": 0.084408,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_same_scene(
    expected: SceneSignature,
    actual: SceneSignature,
    index: int,
    seed: int,
) -> None:
    for field in ("obstacle_type", "obstacle_size_key"):
        if getattr(expected, field) != getattr(actual, field):
            raise RuntimeError(
                f"reset mismatch episode={index} seed={seed} field={field}"
            )
    for field in (
        "state",
        "target_position",
        "target_orientation",
        "obstacle_position",
        "obstacle_size",
        "point_cloud",
    ):
        left = getattr(expected, field)
        right = getattr(actual, field)
        if not np.array_equal(left, right):
            maximum = float(
                np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
            )
            raise RuntimeError(
                f"reset mismatch episode={index} seed={seed} "
                f"field={field} max_abs={maximum}"
            )


def result(
    index: int,
    seed: int,
    info: dict,
    minimum: float,
    steps: int,
) -> EpisodeResult:
    return EpisodeResult(
        index=index,
        seed=seed,
        final_position_mm=float(info["pos_distance"]) * 1000.0,
        final_orientation_deg=float(info["orient_distance_deg"]),
        had_collision=bool(info.get("had_collision", False)),
        episode_min_distance_m=minimum,
        steps=steps,
    )


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self) -> dict:
        if not self.count:
            return {"count": 0, "mean": None, "std": None}
        return {
            "count": self.count,
            "mean": self.mean,
            "std": float(np.sqrt(self.m2 / self.count)),
            "min": self.minimum,
            "max": self.maximum,
        }


def uncentered_counterfactual(observation: dict) -> dict:
    relation = np.asarray(observation["relation_cloud"], dtype=np.float32)
    keypoints = np.asarray(observation["robot_keypoints"], dtype=np.float32)
    points_base = relation[0, :, :3] + keypoints[0][None, :]
    distance = np.linalg.norm(points_base, axis=1, keepdims=True).astype(np.float32)
    slot = np.concatenate((points_base, distance), axis=1).astype(np.float32)
    counterfactual = dict(observation)
    counterfactual["relation_cloud"] = np.broadcast_to(
        slot[None, :, :], relation.shape
    ).copy()
    return counterfactual


def run_frozen():
    seed_everything(BASE_SEED)
    env = ObstacleIKEnv()
    policy = FrozenBasePolicy(BASE_MODEL)
    scenes, results = [], []
    try:
        for index in range(EPISODES):
            seed = BASE_SEED + index
            seed_everything(seed)
            observation, _ = env.reset(seed=seed)
            scenes.append(capture_scene(env, observation))
            minimum = float(env._get_min_obstacle_distance())
            steps = 0
            final_info = None
            for _ in range(env.max_steps):
                action = policy.predict(observation["state"])
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1
                final_info = info
                minimum = min(minimum, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError("Frozen episode produced no step")
            results.append(result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"frozen: {index + 1}/{EPISODES}", flush=True)
    finally:
        env.close()
    return scenes, results


def relation_feature_norm(model: PPO, observation: dict) -> float:
    with torch.inference_mode():
        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        feature = model.policy.features_extractor.relation_encoder(
            observation_tensor["relation_cloud"].float()
        )
        return float(torch.linalg.vector_norm(feature, dim=1).item())


def run_relative(mode: str, model_path: Path, scenes):
    seed_everything(BASE_SEED)
    env = RelativeResidualObstacleEnv(
        relation_mode=mode,
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    model = PPO.load(str(model_path), device="cpu")
    model.policy.set_training_mode(False)
    results = []
    diagnostics = ActionDiagnostics("headroom")
    feature_norms = RunningMoments()
    sensitivity_abs = RunningMoments()
    sensitivity_l2 = RunningMoments()
    sensitivity_max = RunningMoments()
    changed_dimensions = 0
    sensitivity_dimensions = 0
    try:
        for index in range(EPISODES):
            seed = BASE_SEED + index
            seed_everything(seed)
            observation, _ = env.reset(seed=seed)
            assert_same_scene(
                scenes[index], capture_scene(env.unwrapped, observation), index, seed
            )
            minimum = float(env.unwrapped._get_min_obstacle_distance())
            steps = 0
            final_info = None
            for _ in range(env.unwrapped.max_steps):
                feature_norms.add(relation_feature_norm(model, observation))
                with torch.inference_mode():
                    action, _ = model.predict(observation, deterministic=True)
                    if mode == "robot_centered" and index < SENSITIVITY_EPISODES:
                        counterfactual_action, _ = model.predict(
                            uncentered_counterfactual(observation),
                            deterministic=True,
                        )
                        delta = np.asarray(counterfactual_action) - np.asarray(action)
                        sensitivity_abs.add(float(np.mean(np.abs(delta))))
                        sensitivity_l2.add(float(np.linalg.norm(delta)))
                        sensitivity_max.add(float(np.max(np.abs(delta))))
                        changed_dimensions += int(np.sum(np.abs(delta) > 1e-7))
                        sensitivity_dimensions += int(delta.size)
                observation, _, terminated, truncated, info = env.step(action)
                diagnostics.add(info)
                steps += 1
                final_info = info
                minimum = min(minimum, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError(f"{mode} episode produced no step")
            results.append(result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"{mode}: {index + 1}/{EPISODES}", flush=True)
    finally:
        env.close()
    sensitivity = None
    if mode == "robot_centered":
        sensitivity = {
            "episodes": SENSITIVITY_EPISODES,
            "comparison": "same on-policy observation; relation_cloud only replaced by uncentered",
            "mean_abs_action_delta": sensitivity_abs.summary(),
            "action_l2_delta": sensitivity_l2.summary(),
            "max_abs_action_delta": sensitivity_max.summary(),
            "dimension_fraction_abs_delta_gt_1e-7": (
                changed_dimensions / sensitivity_dimensions
            ),
        }
    return results, diagnostics, feature_norms.summary(), sensitivity


def compare(
    first_name: str,
    first,
    second_name: str,
    second,
    seed_offset: int,
) -> dict:
    first_collision = np.asarray([item.had_collision for item in first], dtype=bool)
    second_collision = np.asarray([item.had_collision for item in second], dtype=bool)
    first_cf10 = np.asarray([item.collision_free_10 for item in first], dtype=bool)
    second_cf10 = np.asarray([item.collision_free_10 for item in second], dtype=bool)
    rescued = int(np.sum(first_collision & ~second_collision))
    created = int(np.sum(~first_collision & second_collision))
    gained = int(np.sum(~first_cf10 & second_cf10))
    lost = int(np.sum(first_cf10 & ~second_cf10))
    return {
        "comparison": f"{second_name}_minus_{first_name}",
        "collision": {
            f"{first_name}_collision_to_{second_name}_no_collision": rescued,
            f"{first_name}_no_collision_to_{second_name}_collision": created,
            "both_collision": int(np.sum(first_collision & second_collision)),
            "both_no_collision": int(np.sum(~first_collision & ~second_collision)),
            "paired_difference": float(
                np.mean(second_collision) - np.mean(first_collision)
            ),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                second_collision, first_collision, 76_000 + seed_offset
            ),
            "exact_mcnemar": exact_mcnemar(rescued, created),
        },
        "collision_free_10mm": {
            f"{first_name}_fail_to_{second_name}_success": gained,
            f"{first_name}_success_to_{second_name}_fail": lost,
            "both_success": int(np.sum(first_cf10 & second_cf10)),
            "both_fail": int(np.sum(~first_cf10 & ~second_cf10)),
            "paired_difference": float(np.mean(second_cf10) - np.mean(first_cf10)),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                second_cf10, first_cf10, 76_100 + seed_offset
            ),
            "exact_mcnemar": exact_mcnemar(gained, lost),
        },
    }


def cases(first_name, first, second_name, second) -> dict:
    categories: dict[str, list[dict]] = {}
    for left, right in zip(first, second):
        selected = {
            "collision_to_no_collision": left.had_collision and not right.had_collision,
            "no_collision_to_collision": not left.had_collision and right.had_collision,
            "cf10_fail_to_success": not left.collision_free_10 and right.collision_free_10,
            "cf10_success_to_fail": left.collision_free_10 and not right.collision_free_10,
        }
        for label, include in selected.items():
            if include:
                key = f"{first_name}_to_{second_name}_{label}"
                categories.setdefault(key, []).append(
                    {
                        "seed": left.seed,
                        first_name: asdict(left),
                        second_name: asdict(right),
                    }
                )
    return {
        key: {"count": len(items), "first_20": items[:20]}
        for key, items in categories.items()
    }


def frozen_reference_check(actual: dict) -> dict:
    actual_values = {
        "ik_success_5mm_10deg": actual["success_5"]["rate"],
        "ik_success_10mm_10deg": actual["success_10"]["rate"],
        "collision_rate": actual["had_collision"]["rate"],
        "collision_free_5mm_success": actual["collision_free_5"]["rate"],
        "collision_free_10mm_success": actual["collision_free_10"]["rate"],
        "mean_final_position_mm": actual["mean_final_position_error_mm"],
        "mean_final_orientation_deg": actual["mean_final_orientation_error_deg"],
        "mean_episode_min_distance_m": (
            actual["mean_episode_minimum_obstacle_distance_m"]
        ),
    }
    differences = {
        key: float(actual_values[key] - FROZEN_REFERENCE[key])
        for key in actual_values
    }
    exact_rates = all(
        differences[key] == 0.0
        for key in (
            "ik_success_5mm_10deg",
            "ik_success_10mm_10deg",
            "collision_rate",
            "collision_free_5mm_success",
            "collision_free_10mm_success",
        )
    )
    return {
        "reference": FROZEN_REFERENCE,
        "differences_actual_minus_reference": differences,
        "all_five_rates_exactly_reproduced": exact_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    if args.episodes != EPISODES or args.base_seed != BASE_SEED:
        raise ValueError("Preregistered benchmark requires exactly 1000 seeds from 2026081300")
    for path in (BASE_MODEL, UNCENTERED_MODEL, ROBOT_MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    for output in (SUMMARY_PATH, CASES_PATH, EPISODES_PATH):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark: {output}")

    print("Phase 1/3 Frozen", flush=True)
    scenes, frozen = run_frozen()
    print("Phase 2/3 Uncentered Relative", flush=True)
    uncentered, uncentered_actions, uncentered_features, _ = run_relative(
        "uncentered", UNCENTERED_MODEL, scenes
    )
    print("Phase 3/3 Robot-Centered Relative", flush=True)
    robot, robot_actions, robot_features, sensitivity = run_relative(
        "robot_centered", ROBOT_MODEL, scenes
    )
    robot_vs_uncentered = compare("uncentered", uncentered, "robot", robot, 1)
    robot_vs_frozen = compare("frozen", frozen, "robot", robot, 2)
    uncentered_vs_frozen = compare("frozen", frozen, "uncentered", uncentered, 3)
    frozen_summary = summarize(frozen)
    robot_summary = summarize(robot)
    summary = {
        "episodes": EPISODES,
        "base_seed": BASE_SEED,
        "all_three_reset_scenes_exactly_equal": True,
        "frozen": frozen_summary,
        "frozen_reference_check": frozen_reference_check(frozen_summary),
        "uncentered_relative_20480": summarize(uncentered),
        "robot_centered_relative_20480": robot_summary,
        "robot_vs_uncentered": robot_vs_uncentered,
        "robot_vs_frozen": robot_vs_frozen,
        "uncentered_vs_frozen": uncentered_vs_frozen,
        "uncentered_action_diagnostics": uncentered_actions.summary(),
        "robot_centered_action_diagnostics": robot_actions.summary(),
        "uncentered_relation_feature_norm": uncentered_features,
        "robot_centered_relation_feature_norm": robot_features,
        "robot_centered_relation_sensitivity": sensitivity,
        "robot_ik_10mm_protection_threshold": 0.857,
        "robot_ik_10mm_protection_passed": (
            robot_summary["success_10"]["rate"] >= 0.857
        ),
    }
    case_report = {
        **cases("uncentered", uncentered, "robot", robot),
        **cases("frozen", frozen, "robot", robot),
        **cases("frozen", frozen, "uncentered", uncentered),
    }
    episode_report = {
        "frozen": [dataclass_asdict(item) for item in frozen],
        "uncentered": [dataclass_asdict(item) for item in uncentered],
        "robot_centered": [dataclass_asdict(item) for item in robot],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    CASES_PATH.write_text(
        json.dumps(case_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    EPISODES_PATH.write_text(
        json.dumps(episode_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {SUMMARY_PATH}")
    print(f"cases: {CASES_PATH}")
    print(f"episodes: {EPISODES_PATH}")


if __name__ == "__main__":
    main()
