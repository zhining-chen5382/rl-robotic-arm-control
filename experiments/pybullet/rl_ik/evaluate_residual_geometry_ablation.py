"""Strict Frozen/Zero-Geometry/Actual-Geometry 1000-pair benchmark."""

from __future__ import annotations

import argparse
import json
import random
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
from ik_6d.residual_geometry_features import ResidualGeometryFeatureExtractor  # noqa: F401
from ik_6d.rl_env_6d_obstacle import RobotIK6DEnv as ObstacleIKEnv
from ik_6d.rl_env_6d_residual_geometry import GeometryResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "models" / "panda_ppo_ik_6d_map.zip"
ZERO_MODEL = ROOT / "models" / "residual_geometry_ablation" / "zero_geometry" / "final.zip"
ACTUAL_MODEL = ROOT / "models" / "residual_geometry_ablation" / "actual_geometry" / "final.zip"
OUTPUT_ROOT = ROOT / "benchmarks" / "residual_geometry_ablation"
SUMMARY_PATH = OUTPUT_ROOT / "geometry_three_model_summary.json"
CASES_PATH = OUTPUT_ROOT / "geometry_three_model_cases.json"
EPISODES = 1_000
BASE_SEED = 2_026_081_300


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_same_scene(expected: SceneSignature, actual: SceneSignature, index: int, seed: int) -> None:
    for field in ("obstacle_type", "obstacle_size_key"):
        if getattr(expected, field) != getattr(actual, field):
            raise RuntimeError(f"reset mismatch episode={index} seed={seed} field={field}")
    for field in (
        "state", "target_position", "target_orientation", "obstacle_position",
        "obstacle_size", "point_cloud",
    ):
        left = getattr(expected, field)
        right = getattr(actual, field)
        if not np.array_equal(left, right):
            maximum = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            raise RuntimeError(
                f"reset mismatch episode={index} seed={seed} field={field} max_abs={maximum}"
            )


def result(index: int, seed: int, info: dict, minimum: float, steps: int) -> EpisodeResult:
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
            results.append(result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"frozen: {index + 1}/{EPISODES}", flush=True)
    finally:
        env.close()
    return scenes, results


def run_geometry(mode: str, model_path: Path, scenes):
    seed_everything(BASE_SEED)
    env = GeometryResidualObstacleEnv(
        geometry_mode=mode,
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
            assert_same_scene(scenes[index], capture_scene(env.unwrapped, observation), index, seed)
            minimum = float(env.unwrapped._get_min_obstacle_distance())
            steps = 0
            final_info = None
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
            results.append(result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"{mode}: {index + 1}/{EPISODES}", flush=True)
    finally:
        env.close()
    return results, diagnostics


def compare(first_name: str, first, second_name: str, second, seed_offset: int) -> dict:
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
            "paired_difference": float(np.mean(second_collision) - np.mean(first_collision)),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                second_collision, first_collision, 75_000 + seed_offset
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
                second_cf10, first_cf10, 75_100 + seed_offset
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
                    {"seed": left.seed, first_name: asdict(left), second_name: asdict(right)}
                )
    return {key: {"count": len(items), "first_20": items[:20]} for key, items in categories.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    if args.episodes != EPISODES or args.base_seed != BASE_SEED:
        raise ValueError("Preregistered benchmark requires exactly 1000 seeds from 2026081300")
    for path in (BASE_MODEL, ZERO_MODEL, ACTUAL_MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Phase 1/3 Frozen", flush=True)
    scenes, frozen = run_frozen()
    print("Phase 2/3 Zero Geometry", flush=True)
    zero, zero_actions = run_geometry("zero", ZERO_MODEL, scenes)
    print("Phase 3/3 Actual Geometry", flush=True)
    actual, actual_actions = run_geometry("actual", ACTUAL_MODEL, scenes)
    actual_vs_zero = compare("zero", zero, "actual", actual, 1)
    actual_vs_frozen = compare("frozen", frozen, "actual", actual, 2)
    zero_vs_frozen = compare("frozen", frozen, "zero", zero, 3)
    summary = {
        "episodes": EPISODES,
        "base_seed": BASE_SEED,
        "all_three_reset_scenes_exactly_equal": True,
        "frozen": summarize(frozen),
        "zero_geometry_20480": summarize(zero),
        "actual_geometry_20480": summarize(actual),
        "actual_vs_zero": actual_vs_zero,
        "actual_vs_frozen": actual_vs_frozen,
        "zero_vs_frozen": zero_vs_frozen,
        "zero_action_diagnostics": zero_actions.summary(),
        "actual_action_diagnostics": actual_actions.summary(),
        "actual_ik_10mm_protection_threshold": 0.857,
    }
    case_report = {
        **cases("zero", zero, "actual", actual),
        **cases("frozen", frozen, "actual", actual),
        **cases("frozen", frozen, "zero", zero),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    CASES_PATH.write_text(json.dumps(case_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {SUMMARY_PATH}")
    print(f"cases: {CASES_PATH}")


if __name__ == "__main__":
    main()
