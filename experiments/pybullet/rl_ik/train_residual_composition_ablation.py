"""Controlled 20k Direct-add vs Headroom residual PPO retraining ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ik_6d.episode_seed_wrapper import EpisodeSeedScheduleWrapper, seed_everything
from ik_6d.residual_ablation_callback import ResidualAblationCallback
from ik_6d.residual_pointnet_features import ResidualPointNetFeatureExtractor
from ik_6d.rl_env_6d_residual_headroom import HeadroomResidualObstacleEnv
from ik_6d.rl_env_6d_residual_obstacle import ResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
INIT_MODEL = ROOT / "models" / "residual_obstacle" / "residual_init.zip"
MODEL_ROOT = ROOT / "models" / "residual_ablation"
LOG_ROOT = ROOT / "logs" / "residual_ablation"
TRAINING_SEED = 42
EPISODE_BASE_SEED = 2_026_081_400
TOTAL_TIMESTEPS = 20_000
CHECKPOINT_STEPS = (5_000, 10_000, 15_000, 20_000)
EXPECTED_INIT_FILE_SHA256 = "60fb0c16f1b43b46cbcfc8ab98bc7c756da4069e3a0926e801d51d6e481c8264"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_fingerprint(model: PPO) -> str:
    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def space_signature(space) -> dict:
    if hasattr(space, "spaces"):
        return {
            "type": type(space).__name__,
            "spaces": {key: space_signature(value) for key, value in space.spaces.items()},
        }
    return {
        "type": type(space).__name__,
        "shape": list(space.shape),
        "dtype": str(space.dtype),
        "low": np.asarray(space.low).tolist(),
        "high": np.asarray(space.high).tolist(),
    }


def hyperparameter_signature(model: PPO) -> dict:
    activation = model.policy.activation_fn
    net_arch = model.policy.net_arch
    extractor = model.policy.features_extractor
    point_dim = int(extractor.pointnet.output_dim)
    return {
        "learning_rate": float(model.learning_rate),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range_at_1": float(model.clip_range(1.0)),
        "device": str(model.device),
        "log_std_init_saved_value_mean": float(model.policy.log_std.detach().mean()),
        "point_feature_dim": point_dim,
        "net_arch": net_arch,
        "activation": activation.__name__,
    }


def assert_expected_config(config: dict) -> None:
    expected = {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range_at_1": 0.2,
        "device": "cpu",
        "point_feature_dim": 128,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation": "Tanh",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(f"Unexpected PPO configuration {key}: {config[key]!r} != {value!r}")
    if not np.isclose(config["log_std_init_saved_value_mean"], -2.0, rtol=0, atol=0):
        raise RuntimeError("residual_init log_std is not exactly -2.0")


def make_scheduled_env(mode: str):
    residual_env = (
        ResidualObstacleEnv(residual_scale=0.05, residual_penalty_weight=1e-4)
        if mode == "direct"
        else HeadroomResidualObstacleEnv(residual_scale=0.05, residual_penalty_weight=1e-4)
    )
    schedule = EpisodeSeedScheduleWrapper(
        residual_env,
        base_seed=EPISODE_BASE_SEED,
    )
    schedule.action_space.seed(TRAINING_SEED)
    return Monitor(schedule), schedule


def load_init(env, tensorboard_log: Path) -> PPO:
    model = PPO.load(
        str(INIT_MODEL),
        env=env,
        device="cpu",
        tensorboard_log=str(tensorboard_log),
        force_reset=True,
    )
    model.set_random_seed(TRAINING_SEED)
    config = hyperparameter_signature(model)
    assert_expected_config(config)
    return model


def sanity_evaluation(mode: str, episodes: int = 5) -> dict:
    seed_everything(TRAINING_SEED)
    env, schedule = make_scheduled_env(mode)
    model = load_init(env, LOG_ROOT / mode)
    rewards: list[float] = []
    try:
        observation, _ = env.reset()
        for _ in range(episodes):
            episode_reward = 0.0
            while True:
                with torch.inference_mode():
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    rewards.append(episode_reward)
                    if len(rewards) == episodes:
                        break
                    observation, _ = env.reset()
            if len(rewards) == episodes:
                break
        return {
            "mode": mode,
            "episodes": episodes,
            "policy_fingerprint": policy_fingerprint(model),
            "observation_space": space_signature(env.observation_space),
            "action_space": space_signature(env.action_space),
            "mean_reward": float(np.mean(rewards)),
            "reset_records": schedule.reset_records,
            "configuration": hyperparameter_signature(model),
        }
    finally:
        env.close()


def preflight() -> dict:
    if not INIT_MODEL.is_file():
        raise FileNotFoundError(INIT_MODEL)
    init_file_hash = file_sha256(INIT_MODEL)
    if init_file_hash != EXPECTED_INIT_FILE_SHA256:
        raise RuntimeError(
            f"residual_init file hash changed: {init_file_hash}"
        )
    direct = sanity_evaluation("direct")
    headroom = sanity_evaluation("headroom")
    if direct["policy_fingerprint"] != headroom["policy_fingerprint"]:
        raise RuntimeError("Direct and Headroom init policy tensors differ")
    if direct["observation_space"] != headroom["observation_space"]:
        raise RuntimeError("Direct and Headroom observation spaces differ")
    if direct["action_space"] != headroom["action_space"]:
        raise RuntimeError("Direct and Headroom action spaces differ")
    direct_scenes = [record["scene_fingerprint"] for record in direct["reset_records"]]
    headroom_scenes = [record["scene_fingerprint"] for record in headroom["reset_records"]]
    if direct_scenes != headroom_scenes:
        raise RuntimeError("Preflight scene seed schedules differ")
    report = {
        "passed": True,
        "init_file_sha256": init_file_hash,
        "common_policy_fingerprint": direct["policy_fingerprint"],
        "spaces_identical": True,
        "sanity_scenes_identical": True,
        "direct": direct,
        "headroom": headroom,
    }
    path = ROOT / "benchmarks" / "residual_obstacle" / "retraining_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS preflight: common policy fingerprint {direct['policy_fingerprint']}")
    print(f"report: {path}")
    return report


def refuse_existing_outputs() -> None:
    paths = []
    for mode in ("direct", "headroom"):
        model_dir = MODEL_ROOT / mode
        log_dir = LOG_ROOT / mode
        paths.extend(
            [model_dir / "init_loaded.zip", model_dir / "final.zip", log_dir / "training_metrics.csv"]
        )
        paths.extend(model_dir / f"{step}_steps.zip" for step in CHECKPOINT_STEPS)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite ablation outputs: {existing}")


def train_one(mode: str, expected_init_fingerprint: str) -> dict:
    seed_everything(TRAINING_SEED)
    model_dir = MODEL_ROOT / mode
    log_dir = LOG_ROOT / mode
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=False)
    log_dir.mkdir(exist_ok=False)
    env, schedule = make_scheduled_env(mode)
    try:
        model = load_init(env, log_dir)
        loaded_fingerprint = policy_fingerprint(model)
        if loaded_fingerprint != expected_init_fingerprint:
            raise RuntimeError(f"{mode} did not load the common residual_init tensors")
        model.save(str(model_dir / "init_loaded"))
        reloaded_init = PPO.load(str(model_dir / "init_loaded.zip"), device="cpu")
        if policy_fingerprint(reloaded_init) != expected_init_fingerprint:
            raise RuntimeError(f"{mode} init_loaded tensors changed during save")

        callback = ResidualAblationCallback(
            mode=mode,
            checkpoint_dir=model_dir,
            metrics_csv_path=log_dir / "training_metrics.csv",
            checkpoint_steps=CHECKPOINT_STEPS,
            window_size=100,
            max_timesteps=TOTAL_TIMESTEPS,
            log_interval_steps=1_000,
            verbose=1,
        )
        print(f"START {mode}: independent load from residual_init", flush=True)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=False,
            tb_log_name="PPO",
        )
        if model.num_timesteps != TOTAL_TIMESTEPS:
            raise RuntimeError(
                f"{mode} must stop at exactly {TOTAL_TIMESTEPS}, got {model.num_timesteps}"
            )
        model.save(str(model_dir / "final"))
        checkpoint = PPO.load(str(model_dir / "20000_steps.zip"), device="cpu")
        final = PPO.load(str(model_dir / "final.zip"), device="cpu")
        checkpoint_fingerprint = policy_fingerprint(checkpoint)
        final_fingerprint = policy_fingerprint(final)
        if checkpoint_fingerprint != final_fingerprint:
            raise RuntimeError(f"{mode} 20000_steps and final policy tensors differ")
        records_path = log_dir / "episode_seed_schedule.json"
        records_path.write_text(
            json.dumps(schedule.reset_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = {
            "mode": mode,
            "timesteps": model.num_timesteps,
            "init_file_sha256": file_sha256(INIT_MODEL),
            "init_policy_fingerprint": loaded_fingerprint,
            "final_policy_fingerprint": final_fingerprint,
            "checkpoint_20000_policy_fingerprint": checkpoint_fingerprint,
            "checkpoint_and_final_parameters_identical": True,
            "completed_episode_resets": len(schedule.reset_records),
            "first_episode_seed": schedule.reset_records[0]["seed"],
            "last_episode_seed": schedule.reset_records[-1]["seed"],
            "configuration": hyperparameter_signature(final),
        }
        (log_dir / "training_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"STOP {mode}: exactly {TOTAL_TIMESTEPS} timesteps", flush=True)
        return result
    finally:
        env.close()


def verify_training_schedules() -> dict:
    direct = json.loads((LOG_ROOT / "direct" / "episode_seed_schedule.json").read_text(encoding="utf-8"))
    headroom = json.loads((LOG_ROOT / "headroom" / "episode_seed_schedule.json").read_text(encoding="utf-8"))
    common = min(len(direct), len(headroom))
    for index in range(common):
        if direct[index] != headroom[index]:
            raise RuntimeError(f"Training episode seed/scene mismatch at index {index}")
    report = {
        "passed": True,
        "direct_reset_count": len(direct),
        "headroom_reset_count": len(headroom),
        "common_prefix_count": common,
        "common_prefix_seed_and_scene_fingerprints_identical": True,
        "note": (
            "Different compositions may finish different numbers of episodes within 20k; "
            "every common episode index used the same seed and exact reset scene."
        ),
    }
    path = ROOT / "benchmarks" / "residual_obstacle" / "training_seed_schedule_check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def train_all() -> None:
    refuse_existing_outputs()
    preflight_report = preflight()
    common = preflight_report["common_policy_fingerprint"]
    direct = train_one("direct", common)
    headroom = train_one("headroom", common)
    schedule = verify_training_schedules()
    combined = {"direct": direct, "headroom": headroom, "schedule": schedule}
    path = ROOT / "benchmarks" / "residual_obstacle" / "retraining_summary.json"
    path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Both independent 20k runs complete; summary: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight:
        preflight()
    else:
        train_all()


if __name__ == "__main__":
    main()
