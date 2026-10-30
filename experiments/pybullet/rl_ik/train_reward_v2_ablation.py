"""Preregistered 20,480-step Old-vs-V2 Reward/Credit ablation."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ik_6d.isolated_episode_seed_wrapper import IsolatedEpisodeSeedWrapper
from ik_6d.reward_credit_ablation_callback import RewardCreditAblationCallback
from ik_6d.reward_v2 import V2_BALANCED_CONFIG
from ik_6d.reward_v2_wrapper import RewardV2Wrapper
from ik_6d.residual_relative_features import ResidualRelativeFeatureExtractor  # noqa: F401
from ik_6d.rl_env_6d_residual_relative import RelativeResidualObstacleEnv
from train_residual_relative_ablation import (
    assert_model_configuration,
    component_fingerprints,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "benchmarks" / "reward_v2" / "reward_v2_ablation_protocol.json"
PREFLIGHT_PATH = ROOT / "benchmarks" / "reward_v2" / "reward_v2_ablation_preflight.json"
TRAINING_SUMMARY_PATH = (
    ROOT / "benchmarks" / "reward_v2" / "reward_v2_ablation_training_summary.json"
)
COMMON_INIT = (
    ROOT / "models" / "residual_relative_ablation" / "relative_residual_common_init.zip"
)
MODEL_ROOT = ROOT / "models" / "reward_v2_ablation"
LOG_ROOT = ROOT / "logs" / "reward_v2_ablation"
BRANCHES = ("old_reward", "v2_balanced")
TRAIN_BASE_SEED = 3_000_000_000
DEV_FIRST, DEV_LAST = 2_026_081_300, 2_026_082_299
HOLDOUT_FIRST, HOLDOUT_LAST = 2_026_090_100, 2_026_091_099
ALGORITHM_SEED = 20_260_817
SEGMENT_TIMESTEPS = 5_120
TOTAL_TIMESTEPS = 20_480
CHECKPOINT_STEPS = (5_120, 10_240, 15_360, 20_480)


def validate_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["train"]["base_seed"] != TRAIN_BASE_SEED:
        raise RuntimeError("unexpected TRAIN base seed")
    if protocol["dev"]["first_seed"] != DEV_FIRST or protocol["dev"]["last_seed"] != DEV_LAST:
        raise RuntimeError("unexpected DEV seed range")
    if (
        protocol["holdout"]["first_seed"] != HOLDOUT_FIRST
        or protocol["holdout"]["last_seed"] != HOLDOUT_LAST
        or protocol["holdout"]["status"] != "LOCKED_UNSEEN"
    ):
        raise RuntimeError("HOLDOUT protocol is not locked as preregistered")
    # A worst-case one-step episode schedule still cannot overlap either benchmark.
    train_last_possible = TRAIN_BASE_SEED + TOTAL_TIMESTEPS
    ranges = {
        "train": (TRAIN_BASE_SEED, train_last_possible),
        "dev": (DEV_FIRST, DEV_LAST),
        "holdout": (HOLDOUT_FIRST, HOLDOUT_LAST),
    }
    names = tuple(ranges)
    for left_index, left_name in enumerate(names):
        left = ranges[left_name]
        for right_name in names[left_index + 1:]:
            right = ranges[right_name]
            if max(left[0], right[0]) <= min(left[1], right[1]):
                raise RuntimeError(f"seed ranges overlap: {left_name}, {right_name}")
    return protocol


def make_env(branch: str):
    relative = RelativeResidualObstacleEnv(
        relation_mode="robot_centered",
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    treatment = (
        RewardV2Wrapper(relative, config=V2_BALANCED_CONFIG)
        if branch == "v2_balanced" else relative
    )
    isolated = IsolatedEpisodeSeedWrapper(treatment, base_seed=TRAIN_BASE_SEED)
    return Monitor(isolated), isolated


def load_common(env) -> PPO:
    model = PPO.load(str(COMMON_INIT), env=env, device="cpu", force_reset=True)
    assert_model_configuration(model)
    return model


def set_algorithm_rng_once(model: PPO) -> None:
    model.seed = ALGORITHM_SEED
    random.seed(ALGORITHM_SEED)
    np.random.seed(ALGORITHM_SEED)
    torch.manual_seed(ALGORITHM_SEED)


def reset_scene_records(branch: str, count: int = 5) -> tuple[list[dict], dict[str, str]]:
    env, schedule = make_env(branch)
    try:
        model = load_common(env)
        fingerprints = component_fingerprints(model)
        for _ in range(count):
            env.reset()
        return schedule.reset_records, fingerprints
    finally:
        env.close()


def preflight() -> dict:
    validate_protocol()
    if not COMMON_INIT.is_file():
        raise FileNotFoundError(COMMON_INIT)
    if PREFLIGHT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {PREFLIGHT_PATH}")
    if MODEL_ROOT.exists() or LOG_ROOT.exists():
        raise FileExistsError("Reward V2 ablation output directories already exist")
    old_scenes, old_fingerprints = reset_scene_records("old_reward")
    v2_scenes, v2_fingerprints = reset_scene_records("v2_balanced")
    if old_fingerprints != v2_fingerprints:
        raise RuntimeError("Old and V2 did not load identical policy tensors")
    if old_scenes != v2_scenes:
        raise RuntimeError("Old and V2 preflight reset scenes differ")
    report = {
        "passed": True,
        "experiment_name": "Old Reward/Credit vs V2-Balanced Reward/Credit",
        "seed_ranges_pairwise_disjoint": True,
        "holdout_status": "LOCKED_UNSEEN",
        "common_init_path": str(COMMON_INIT),
        "old_and_v2_component_fingerprints_identical": True,
        "component_fingerprints": old_fingerprints,
        "preflight_common_scene_prefix_count": len(old_scenes),
        "preflight_common_scene_prefix_identical": True,
        "old_scenes": old_scenes,
        "v2_scenes": v2_scenes,
        "learn_called": False,
    }
    PREFLIGHT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def refuse_training_outputs() -> None:
    if MODEL_ROOT.exists() or LOG_ROOT.exists() or TRAINING_SUMMARY_PATH.exists():
        raise FileExistsError("Refusing to overwrite Reward/Credit ablation outputs")


def train_one(branch: str, expected_common: dict[str, str]) -> dict:
    model_dir = MODEL_ROOT / branch
    log_dir = LOG_ROOT / branch
    model_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    env, schedule = make_env(branch)
    started = time.perf_counter()
    try:
        model = load_common(env)
        loaded = component_fingerprints(model)
        if loaded != expected_common:
            raise RuntimeError(f"{branch} common-init fingerprint mismatch")
        model.save(str(model_dir / "common_init_loaded"))
        saved_common = PPO.load(str(model_dir / "common_init_loaded.zip"), device="cpu")
        if component_fingerprints(saved_common) != expected_common:
            raise RuntimeError(f"{branch} common_init_loaded tensors changed")
        del saved_common

        # Exactly one explicit algorithm RNG initialization for this branch.
        set_algorithm_rng_once(model)
        callback = RewardCreditAblationCallback(
            log_dir / "training_metrics.csv",
            branch=branch,
            window_size=100,
        )
        for segment_index, checkpoint_step in enumerate(CHECKPOINT_STEPS):
            model.learn(
                total_timesteps=SEGMENT_TIMESTEPS,
                callback=callback,
                reset_num_timesteps=(segment_index == 0),
                progress_bar=False,
                tb_log_name="PPO",
            )
            callback.finalize_rollout()
            if model.num_timesteps != checkpoint_step:
                raise RuntimeError(
                    f"{branch}: expected {checkpoint_step}, got {model.num_timesteps}"
                )
            model.save(str(model_dir / f"{checkpoint_step}_steps"))
            print(
                f"[{branch}] complete rollout-boundary checkpoint {checkpoint_step}",
                flush=True,
            )

        model.save(str(model_dir / "final"))
        checkpoint = PPO.load(str(model_dir / "20480_steps.zip"), device="cpu")
        final = PPO.load(str(model_dir / "final.zip"), device="cpu")
        checkpoint_fp = component_fingerprints(checkpoint)
        final_fp = component_fingerprints(final)
        if checkpoint_fp != final_fp:
            raise RuntimeError(f"{branch}: 20480 checkpoint differs from final")
        schedule_path = log_dir / "episode_seed_schedule.json"
        schedule_path.write_text(
            json.dumps(schedule.reset_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics = callback.summary()
        if branch == "v2_balanced" and metrics["post_collision_transition_count"] != 0:
            raise RuntimeError("V2 post-collision transition count is not zero")
        result = {
            "branch": branch,
            "timesteps": int(model.num_timesteps),
            "complete_rollouts": 20,
            "segments": 4,
            "algorithm_seed": ALGORITHM_SEED,
            "algorithm_rng_explicitly_initialized_once": True,
            "episode_base_seed": TRAIN_BASE_SEED,
            "reset_count": len(schedule.reset_records),
            "unique_training_scenes": len({
                record["scene_fingerprint"] for record in schedule.reset_records
            }),
            "first_seed": schedule.reset_records[0]["seed"],
            "last_seed": schedule.reset_records[-1]["seed"],
            "common_init_fingerprints": loaded,
            "final_fingerprints": final_fp,
            "checkpoint_20480_and_final_identical": True,
            "training_seconds": time.perf_counter() - started,
            "metrics": metrics,
        }
        (log_dir / "training_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"STOP {branch}: exactly {TOTAL_TIMESTEPS} steps", flush=True)
        return result
    finally:
        env.close()


def verify_schedules() -> dict:
    schedules = {
        branch: json.loads(
            (LOG_ROOT / branch / "episode_seed_schedule.json").read_text(
                encoding="utf-8"
            )
        )
        for branch in BRANCHES
    }
    common_count = min(len(items) for items in schedules.values())
    if schedules["old_reward"][:common_count] != schedules["v2_balanced"][:common_count]:
        raise RuntimeError("Old/V2 common episode prefix scenes are not identical")
    return {
        "passed": True,
        "old_reset_count": len(schedules["old_reward"]),
        "v2_reset_count": len(schedules["v2_balanced"]),
        "common_prefix_count": common_count,
        "common_prefix_seed_target_joints_obstacle_pointcloud_identical": True,
    }


def train_all() -> None:
    validate_protocol()
    refuse_training_outputs()
    preflight_report = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if not preflight_report.get("passed"):
        raise RuntimeError("preflight did not pass")
    expected = preflight_report["component_fingerprints"]
    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    LOG_ROOT.mkdir(parents=True, exist_ok=False)
    print("Phase 1/2: Old Reward/Credit", flush=True)
    old = train_one("old_reward", expected)
    print("Old environment closed. Phase 2/2: V2-Balanced Reward/Credit", flush=True)
    v2 = train_one("v2_balanced", expected)
    schedules = verify_schedules()
    report = {
        "experiment_name": "Old Reward/Credit vs V2-Balanced Reward/Credit",
        "old_reward_credit": old,
        "v2_balanced_reward_credit": v2,
        "training_scene_schedules": schedules,
        "holdout_status": "LOCKED_UNSEEN",
    }
    TRAINING_SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Both branches completed exactly 20,480 steps and stopped", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train-all", action="store_true")
    group.add_argument("--verify-schedules", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.train_all:
        train_all()
    else:
        print(json.dumps(verify_schedules(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
