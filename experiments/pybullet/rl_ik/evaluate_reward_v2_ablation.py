"""Neutral 1,000-pair DEV benchmark for Frozen, Old, and V2-trained policies."""

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
OLD_MODEL = ROOT / "models" / "reward_v2_ablation" / "old_reward" / "final.zip"
V2_MODEL = ROOT / "models" / "reward_v2_ablation" / "v2_balanced" / "final.zip"
OUTPUT_ROOT = ROOT / "benchmarks" / "reward_v2"
SUMMARY_PATH = OUTPUT_ROOT / "reward_v2_ablation_dev_summary.json"
CASES_PATH = OUTPUT_ROOT / "reward_v2_ablation_dev_cases.json"
EPISODES_PATH = OUTPUT_ROOT / "reward_v2_ablation_dev_episodes.json"
EPISODES = 1_000
BASE_SEED = 2_026_081_300
FROZEN_REFERENCE = {
    "success_5": 0.819,
    "success_10": 0.877,
    "had_collision": 0.227,
    "collision_free_5": 0.676,
    "collision_free_10": 0.685,
    "mean_final_position_error_mm": 9.260,
    "mean_final_orientation_error_deg": 7.914,
    "mean_episode_minimum_obstacle_distance_m": 0.084408,
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
            raise RuntimeError(f"scene mismatch {index=} {seed=} {field=}")
    for field in (
        "state", "target_position", "target_orientation", "obstacle_position",
        "obstacle_size", "point_cloud",
    ):
        left = getattr(expected, field)
        right = getattr(actual, field)
        if not np.array_equal(left, right):
            raise RuntimeError(
                f"scene mismatch {index=} {seed=} {field=} "
                f"max_abs={np.max(np.abs(left.astype(float)-right.astype(float)))}"
            )


def episode_result(index: int, seed: int, info: dict, minimum: float, steps: int):
    return EpisodeResult(
        index=index,
        seed=seed,
        final_position_mm=float(info["pos_distance"]) * 1000.0,
        final_orientation_deg=float(info["orient_distance_deg"]),
        had_collision=bool(info.get("had_collision", False)),
        episode_min_distance_m=minimum,
        steps=steps,
    )


def run_frozen():
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
            final_info = None
            steps = 0
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
            results.append(episode_result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"frozen {index + 1}/{EPISODES}", flush=True)
    finally:
        env.close()
    return scenes, results


def run_policy(label: str, model_path: Path, scenes):
    # Neutral evaluation: deliberately no RewardV2Wrapper for either model.
    env = RelativeResidualObstacleEnv(
        relation_mode="robot_centered",
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    model = PPO.load(str(model_path), device="cpu")
    model.policy.set_training_mode(False)
    results = []
    diagnostics = ActionDiagnostics("headroom")
    try:
        for index in range(EPISODES):
            seed = BASE_SEED + index
            seed_everything(seed)
            observation, _ = env.reset(seed=seed)
            assert_same_scene(
                scenes[index], capture_scene(env.unwrapped, observation), index, seed
            )
            minimum = float(env.unwrapped._get_min_obstacle_distance())
            final_info = None
            steps = 0
            for _ in range(env.unwrapped.max_steps):
                with torch.inference_mode():
                    action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                diagnostics.add(info)
                steps += 1
                final_info = info
                minimum = min(minimum, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError(f"{label} episode produced no step")
            results.append(episode_result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"{label} {index + 1}/{EPISODES}", flush=True)
    finally:
        env.close()
    return results, diagnostics.summary()


def compare(first_name: str, first, second_name: str, second, seed_offset: int) -> dict:
    first_collision = np.asarray([x.had_collision for x in first], dtype=bool)
    second_collision = np.asarray([x.had_collision for x in second], dtype=bool)
    first_cf10 = np.asarray([x.collision_free_10 for x in first], dtype=bool)
    second_cf10 = np.asarray([x.collision_free_10 for x in second], dtype=bool)
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
            "paired_difference": float(np.mean(second_collision) - np.mean(first_collision)),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                second_collision, first_collision, 88_000 + seed_offset
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
                second_cf10, first_cf10, 88_100 + seed_offset
            ),
            "exact_mcnemar": exact_mcnemar(gained, lost),
        },
    }


def transition_cases(first_name: str, first, second_name: str, second) -> dict:
    categories: dict[str, list[dict]] = {}
    for left, right in zip(first, second):
        selections = {
            "collision_to_no_collision": left.had_collision and not right.had_collision,
            "no_collision_to_collision": not left.had_collision and right.had_collision,
            "cf10_fail_to_success": not left.collision_free_10 and right.collision_free_10,
            "cf10_success_to_fail": left.collision_free_10 and not right.collision_free_10,
        }
        for label, include in selections.items():
            if include:
                key = f"{first_name}_to_{second_name}_{label}"
                categories.setdefault(key, []).append({
                    "seed": left.seed,
                    first_name: asdict(left),
                    second_name: asdict(right),
                })
    return {key: {"count": len(items), "first_20": items[:20]} for key, items in categories.items()}


def frozen_reference_check(summary: dict) -> dict:
    actual = {
        "success_5": summary["success_5"]["rate"],
        "success_10": summary["success_10"]["rate"],
        "had_collision": summary["had_collision"]["rate"],
        "collision_free_5": summary["collision_free_5"]["rate"],
        "collision_free_10": summary["collision_free_10"]["rate"],
        "mean_final_position_error_mm": summary["mean_final_position_error_mm"],
        "mean_final_orientation_error_deg": summary["mean_final_orientation_error_deg"],
        "mean_episode_minimum_obstacle_distance_m": summary["mean_episode_minimum_obstacle_distance_m"],
    }
    differences = {key: float(actual[key] - FROZEN_REFERENCE[key]) for key in actual}
    return {
        "reference": FROZEN_REFERENCE,
        "actual_minus_reference": differences,
        "five_rates_exactly_reproduced": all(
            differences[key] == 0.0
            for key in ("success_5", "success_10", "had_collision", "collision_free_5", "collision_free_10")
        ),
    }


def gates(v2_summary: dict, v2_vs_old: dict) -> dict:
    ik10 = float(v2_summary["success_10"]["rate"])
    collision = v2_vs_old["collision"]
    cf10 = v2_vs_old["collision_free_10mm"]
    gate_a = (
        ik10 >= 0.857
        and cf10["paired_difference"] >= 0.02
        and cf10["paired_bootstrap_95_ci"][0] > 0.0
        and collision["paired_difference"] <= 0.005
    )
    gate_b = (
        ik10 >= 0.857
        and collision["paired_difference"] <= -0.02
        and collision["paired_bootstrap_95_ci"][1] < 0.0
        and cf10["paired_difference"] >= -0.005
    )
    return {
        "ik_10mm_protection_pass": ik10 >= 0.857,
        "gate_a_cf10_path": gate_a,
        "gate_b_collision_path": gate_b,
        "DEV_HOLDOUT_GATE": "PASS" if gate_a or gate_b else "FAIL",
        "holdout_action": "REMAIN_LOCKED_UNSEEN_IN_THIS_TASK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    if args.episodes != EPISODES or args.base_seed != BASE_SEED:
        raise ValueError("Preregistered DEV benchmark is exactly 1000 seeds from 2026081300")
    for path in (BASE_MODEL, OLD_MODEL, V2_MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (SUMMARY_PATH, CASES_PATH, EPISODES_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite DEV output: {path}")

    print("Phase 1/3 Frozen", flush=True)
    scenes, frozen = run_frozen()
    print("Phase 2/3 Old Reward/Credit final in neutral environment", flush=True)
    old, old_actions = run_policy("old", OLD_MODEL, scenes)
    print("Phase 3/3 V2-Balanced final in neutral environment", flush=True)
    v2, v2_actions = run_policy("v2", V2_MODEL, scenes)
    summaries = {
        "frozen": summarize(frozen),
        "old_reward_credit_20480": summarize(old),
        "v2_balanced_reward_credit_20480": summarize(v2),
    }
    v2_vs_old = compare("old", old, "v2", v2, 1)
    v2_vs_frozen = compare("frozen", frozen, "v2", v2, 2)
    old_vs_frozen = compare("frozen", frozen, "old", old, 3)
    gate_report = gates(summaries["v2_balanced_reward_credit_20480"], v2_vs_old)
    summary = {
        "experiment_name": "Old Reward/Credit vs V2-Balanced Reward/Credit",
        "episodes": EPISODES,
        "base_seed": BASE_SEED,
        "neutral_environment_without_reward_v2_wrapper": True,
        "all_three_reset_scenes_exactly_identical": True,
        **summaries,
        "frozen_reference_check": frozen_reference_check(summaries["frozen"]),
        "v2_vs_old_primary": v2_vs_old,
        "v2_vs_frozen_secondary": v2_vs_frozen,
        "old_vs_frozen_secondary": old_vs_frozen,
        "old_action_diagnostics": old_actions,
        "v2_action_diagnostics": v2_actions,
        "gates": gate_report,
        "holdout_status": "LOCKED_UNSEEN",
    }
    cases = {
        **transition_cases("old", old, "v2", v2),
        **transition_cases("frozen", frozen, "v2", v2),
        **transition_cases("frozen", frozen, "old", old),
    }
    episodes = {
        "frozen": [dataclass_asdict(item) for item in frozen],
        "old_reward_credit": [dataclass_asdict(item) for item in old],
        "v2_balanced_reward_credit": [dataclass_asdict(item) for item in v2],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    EPISODES_PATH.write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
