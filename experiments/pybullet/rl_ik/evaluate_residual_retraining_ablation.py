"""Strict 1000-seed paired Frozen/Direct-20k/Headroom-20k benchmark."""

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
from ik_6d.residual_pointnet_features import ResidualPointNetFeatureExtractor  # noqa: F401
from ik_6d.rl_env_6d_obstacle import RobotIK6DEnv as ObstacleIKEnv
from ik_6d.rl_env_6d_residual_headroom import HeadroomResidualObstacleEnv
from ik_6d.rl_env_6d_residual_obstacle import ResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "models" / "panda_ppo_ik_6d_map.zip"
DIRECT_MODEL = ROOT / "models" / "residual_ablation" / "direct" / "final.zip"
HEADROOM_MODEL = ROOT / "models" / "residual_ablation" / "headroom" / "final.zip"
OUTPUT = ROOT / "benchmarks" / "residual_obstacle" / "retrained_three_model_summary.json"
CASES = ROOT / "benchmarks" / "residual_obstacle" / "retrained_three_model_cases.json"
DEFAULT_EPISODES = 1_000
DEFAULT_BASE_SEED = 2_026_081_300


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


def make_result(index: int, seed: int, info: dict, episode_min: float, steps: int) -> EpisodeResult:
    return EpisodeResult(
        index=index,
        seed=seed,
        final_position_mm=float(info["pos_distance"]) * 1000.0,
        final_orientation_deg=float(info["orient_distance_deg"]),
        had_collision=bool(info.get("had_collision", False)),
        episode_min_distance_m=episode_min,
        steps=steps,
    )


def run_frozen(episodes: int, base_seed: int):
    seed_everything(base_seed)
    env = ObstacleIKEnv()
    policy = FrozenBasePolicy(BASE_MODEL)
    scenes: list[SceneSignature] = []
    results: list[EpisodeResult] = []
    try:
        for index in range(episodes):
            seed = base_seed + index
            seed_everything(seed)
            observation, _ = env.reset(seed=seed)
            scenes.append(capture_scene(env, observation))
            minimum = float(env._get_min_obstacle_distance())
            final_info = None
            steps = 0
            for _ in range(env.max_steps):
                action = policy.predict(observation["state"])
                observation, _, terminated, truncated, info = env.step(action)
                final_info = info
                steps += 1
                minimum = min(minimum, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError("Frozen episode executed no step")
            results.append(make_result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"frozen: {index + 1}/{episodes}", flush=True)
    finally:
        env.close()
    return scenes, results


def run_residual(mode: str, model_path: Path, expected_scenes, episodes: int, base_seed: int):
    seed_everything(base_seed)
    env = (
        ResidualObstacleEnv(residual_scale=0.05, residual_penalty_weight=1e-4)
        if mode == "direct"
        else HeadroomResidualObstacleEnv(residual_scale=0.05, residual_penalty_weight=1e-4)
    )
    model = PPO.load(str(model_path), device="cpu")
    model.policy.set_training_mode(False)
    results: list[EpisodeResult] = []
    diagnostics = ActionDiagnostics(mode)
    try:
        for index in range(episodes):
            seed = base_seed + index
            seed_everything(seed)
            observation, _ = env.reset(seed=seed)
            assert_same_scene(
                expected_scenes[index], capture_scene(env.unwrapped, observation), index, seed
            )
            minimum = float(env.unwrapped._get_min_obstacle_distance())
            final_info = None
            steps = 0
            for _ in range(env.unwrapped.max_steps):
                with torch.inference_mode():
                    action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                diagnostics.add(info)
                final_info = info
                steps += 1
                minimum = min(minimum, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError(f"{mode} episode executed no step")
            results.append(make_result(index, seed, final_info, minimum, steps))
            if (index + 1) % 100 == 0:
                print(f"{mode}: {index + 1}/{episodes}", flush=True)
    finally:
        env.close()
    return results, diagnostics


def compare(first_name: str, first: list[EpisodeResult], second_name: str, second: list[EpisodeResult]) -> dict:
    first_collision = np.asarray([x.had_collision for x in first], dtype=bool)
    second_collision = np.asarray([x.had_collision for x in second], dtype=bool)
    first_cf10 = np.asarray([x.collision_free_10 for x in first], dtype=bool)
    second_cf10 = np.asarray([x.collision_free_10 for x in second], dtype=bool)
    collision_rescued = int(np.sum(first_collision & ~second_collision))
    collision_created = int(np.sum(~first_collision & second_collision))
    cf10_gained = int(np.sum(~first_cf10 & second_cf10))
    cf10_lost = int(np.sum(first_cf10 & ~second_cf10))
    return {
        "comparison": f"{second_name}_minus_{first_name}",
        "collision": {
            f"{first_name}_collision_to_{second_name}_no_collision": collision_rescued,
            f"{first_name}_no_collision_to_{second_name}_collision": collision_created,
            "both_collision": int(np.sum(first_collision & second_collision)),
            "both_no_collision": int(np.sum(~first_collision & ~second_collision)),
            "paired_difference": float(np.mean(second_collision) - np.mean(first_collision)),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(second_collision, first_collision, 74_001),
            "exact_mcnemar": exact_mcnemar(collision_rescued, collision_created),
        },
        "collision_free_10mm": {
            f"{first_name}_fail_to_{second_name}_success": cf10_gained,
            f"{first_name}_success_to_{second_name}_fail": cf10_lost,
            "both_success": int(np.sum(first_cf10 & second_cf10)),
            "both_fail": int(np.sum(~first_cf10 & ~second_cf10)),
            "paired_difference": float(np.mean(second_cf10) - np.mean(first_cf10)),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(second_cf10, first_cf10, 74_002),
            "exact_mcnemar": exact_mcnemar(cf10_gained, cf10_lost),
        },
    }


def transition_cases(first_name, first, second_name, second) -> dict:
    categories: dict[str, list[dict]] = {}
    for left, right in zip(first, second):
        tests = {
            "collision_to_no_collision": left.had_collision and not right.had_collision,
            "no_collision_to_collision": not left.had_collision and right.had_collision,
            "cf10_fail_to_success": not left.collision_free_10 and right.collision_free_10,
            "cf10_success_to_fail": left.collision_free_10 and not right.collision_free_10,
        }
        for label, selected in tests.items():
            if selected:
                key = f"{first_name}_to_{second_name}_{label}"
                categories.setdefault(key, []).append(
                    {"seed": left.seed, first_name: asdict(left), second_name: asdict(right)}
                )
    return {key: {"count": len(items), "first_20": items[:20]} for key, items in categories.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()
    if args.episodes != 1_000:
        raise ValueError("This preregistered benchmark must use exactly 1000 episodes")
    for path in (BASE_MODEL, DIRECT_MODEL, HEADROOM_MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Phase 1/3: Frozen", flush=True)
    scenes, frozen = run_frozen(args.episodes, args.base_seed)
    print("Phase 2/3: retrained Direct", flush=True)
    direct, direct_actions = run_residual(
        "direct", DIRECT_MODEL, scenes, args.episodes, args.base_seed
    )
    print("Phase 3/3: retrained Headroom", flush=True)
    headroom, headroom_actions = run_residual(
        "headroom", HEADROOM_MODEL, scenes, args.episodes, args.base_seed
    )

    direct_vs_headroom = compare("direct", direct, "headroom", headroom)
    frozen_vs_headroom = compare("frozen", frozen, "headroom", headroom)
    summary = {
        "episodes": args.episodes,
        "base_seed": args.base_seed,
        "all_three_reset_scenes_exactly_equal": True,
        "frozen": summarize(frozen),
        "direct_20k": summarize(direct),
        "headroom_20k": summarize(headroom),
        "direct_vs_headroom": direct_vs_headroom,
        "frozen_vs_headroom": frozen_vs_headroom,
        "direct_action_diagnostics": direct_actions.summary(),
        "headroom_action_diagnostics": headroom_actions.summary(),
        "preregistered_thresholds": {
            "headroom_10mm_ik_minimum": 0.857,
            "headroom_collision_below": 0.227,
            "headroom_collision_free_10mm_above": 0.685,
        },
    }
    cases = {
        **transition_cases("direct", direct, "headroom", headroom),
        **transition_cases("frozen", frozen, "headroom", headroom),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    CASES.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {OUTPUT}")
    print(f"cases: {CASES}")


if __name__ == "__main__":
    main()
