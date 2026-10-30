"""Common initialization and controlled 20,480-step Geometry ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ik_6d.geometry_ablation_callback import GeometryAblationMetricsCallback
from ik_6d.isolated_episode_seed_wrapper import IsolatedEpisodeSeedWrapper
from ik_6d.residual_geometry_features import ResidualGeometryFeatureExtractor
from ik_6d.rl_env_6d_residual_geometry import GeometryResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models" / "residual_geometry_ablation"
LOG_ROOT = ROOT / "logs" / "residual_geometry_ablation"
BENCHMARK_ROOT = ROOT / "benchmarks" / "residual_geometry_ablation"
COMMON_INIT = MODEL_ROOT / "geometry_residual_common_init.zip"
INITIALIZATION_SEED = 20_260_815
ALGORITHM_SEED = 20_260_815
EPISODE_BASE_SEED = 2_026_081_500
SEGMENT_TIMESTEPS = 5_120
TOTAL_TIMESTEPS = 20_480
CHECKPOINT_STEPS = (5_120, 10_240, 15_360, 20_480)


def policy_fingerprint(model: PPO, prefixes: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    state = model.policy.state_dict()
    names = [
        name for name in sorted(state)
        if prefixes is None or any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not names:
        raise RuntimeError(f"No policy tensors matched prefixes {prefixes}")
    for name in names:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def component_fingerprints(model: PPO) -> dict[str, str]:
    return {
        "full_policy": policy_fingerprint(model),
        "pointnet": policy_fingerprint(
            model, ("features_extractor.pointnet.",)
        ),
        "geometry_mlp": policy_fingerprint(
            model, ("features_extractor.geometry_mlp.",)
        ),
        "actor": policy_fingerprint(
            model, ("mlp_extractor.policy_net.", "action_net.")
        ),
        "critic": policy_fingerprint(
            model, ("mlp_extractor.value_net.", "value_net.")
        ),
        "log_std": policy_fingerprint(model, ("log_std",)),
    }


def policy_kwargs() -> dict:
    return {
        "features_extractor_class": ResidualGeometryFeatureExtractor,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.Tanh,
        "log_std_init": -2.0,
    }


def make_env(mode: str):
    geometry_env = GeometryResidualObstacleEnv(
        geometry_mode=mode,
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    isolated = IsolatedEpisodeSeedWrapper(
        geometry_env,
        base_seed=EPISODE_BASE_SEED,
    )
    return Monitor(isolated), isolated


def create_model(env) -> PPO:
    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs(),
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
        seed=INITIALIZATION_SEED,
        verbose=1,
    )


def assert_model_configuration(model: PPO) -> None:
    expected = {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "device": "cpu",
        "features_dim": 216,
    }
    actual = {
        "learning_rate": float(model.learning_rate),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range": float(model.clip_range(1.0)),
        "device": str(model.device),
        "features_dim": int(model.policy.features_extractor.features_dim),
    }
    if actual != expected:
        raise RuntimeError(f"Unexpected model configuration: {actual}")
    if model.policy.net_arch != {"pi": [256, 256], "vf": [256, 256]}:
        raise RuntimeError(f"Unexpected net_arch: {model.policy.net_arch}")
    if model.policy.activation_fn is not torch.nn.Tanh:
        raise RuntimeError("Policy activation must be Tanh")


def prepare_common_init() -> dict:
    if COMMON_INIT.exists():
        raise FileExistsError(f"Refusing to overwrite common init: {COMMON_INIT}")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(INITIALIZATION_SEED)
    np.random.seed(INITIALIZATION_SEED)
    torch.manual_seed(INITIALIZATION_SEED)
    env, _ = make_env("zero")
    try:
        model = create_model(env)
        assert_model_configuration(model)
        if not np.allclose(
            model.policy.log_std.detach().cpu().numpy(), -2.0, rtol=0, atol=0
        ):
            raise RuntimeError("Initial log_std is not exactly -2.0")
        fingerprints = component_fingerprints(model)
        model.save(str(COMMON_INIT.with_suffix("")))
    finally:
        env.close()
    report = {
        "created_without_learn": True,
        "path": str(COMMON_INIT),
        "initialization_seed": INITIALIZATION_SEED,
        "fingerprints": fingerprints,
    }
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "common_init_creation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created common init without learn(): {COMMON_INIT}")
    return report


def load_common(env) -> PPO:
    model = PPO.load(str(COMMON_INIT), env=env, device="cpu", force_reset=True)
    assert_model_configuration(model)
    return model


def deterministic_sanity(mode: str, episodes: int = 3) -> dict:
    env, schedule = make_env(mode)
    model = load_common(env)
    rewards = []
    try:
        observation, _ = env.reset()
        while len(rewards) < episodes:
            total = 0.0
            while True:
                with torch.inference_mode():
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    rewards.append(total)
                    if len(rewards) < episodes:
                        observation, _ = env.reset()
                    break
        return {
            "mode": mode,
            "component_fingerprints": component_fingerprints(model),
            "mean_reward": float(np.mean(rewards)),
            "reset_records": schedule.reset_records,
        }
    finally:
        env.close()


def preflight() -> dict:
    if not COMMON_INIT.is_file():
        raise FileNotFoundError(COMMON_INIT)
    zero = deterministic_sanity("zero")
    actual = deterministic_sanity("actual")
    if zero["component_fingerprints"] != actual["component_fingerprints"]:
        raise RuntimeError("Zero and Actual did not load identical common init tensors")
    zero_scenes = [item["scene_fingerprint"] for item in zero["reset_records"]]
    actual_scenes = [item["scene_fingerprint"] for item in actual["reset_records"]]
    if zero_scenes != actual_scenes:
        raise RuntimeError("Zero/Actual sanity reset scenes differ")
    report = {
        "passed": True,
        "component_fingerprints_identical": True,
        "sanity_scenes_identical": True,
        "zero": zero,
        "actual": actual,
    }
    (BENCHMARK_ROOT / "common_init_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "PASS common init preflight: "
        f"{zero['component_fingerprints']['full_policy']}"
    )
    return report


def set_algorithm_rng_once() -> None:
    random.seed(ALGORITHM_SEED)
    np.random.seed(ALGORITHM_SEED)
    torch.manual_seed(ALGORITHM_SEED)


def refuse_training_outputs() -> None:
    expected = []
    for mode in ("zero_geometry", "actual_geometry"):
        model_dir = MODEL_ROOT / mode
        log_dir = LOG_ROOT / mode
        expected.extend(
            [model_dir / "common_init_loaded.zip", model_dir / "final.zip",
             log_dir / "training_metrics.csv", log_dir / "episode_seed_schedule.json"]
        )
        expected.extend(model_dir / f"{step}_steps.zip" for step in CHECKPOINT_STEPS)
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite Geometry outputs: {existing}")


def train_one(mode: str, expected_common: dict[str, str]) -> dict:
    directory_name = f"{mode}_geometry"
    model_dir = MODEL_ROOT / directory_name
    log_dir = LOG_ROOT / directory_name
    model_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    env, schedule = make_env(mode)
    try:
        model = load_common(env)
        loaded = component_fingerprints(model)
        if loaded != expected_common:
            raise RuntimeError(f"{mode} common init fingerprint mismatch")
        model.save(str(model_dir / "common_init_loaded"))
        reloaded = PPO.load(str(model_dir / "common_init_loaded.zip"), device="cpu")
        if component_fingerprints(reloaded) != expected_common:
            raise RuntimeError(f"{mode} common_init_loaded changed during save")

        # The only algorithm-side RNG initialization for this branch.
        set_algorithm_rng_once()
        env.action_space.seed(ALGORITHM_SEED)
        callback = GeometryAblationMetricsCallback(
            log_dir / "training_metrics.csv",
            window_size=100,
            log_interval_steps=1_024,
        )
        for segment_index, checkpoint_step in enumerate(CHECKPOINT_STEPS):
            model.learn(
                total_timesteps=SEGMENT_TIMESTEPS,
                callback=callback,
                reset_num_timesteps=(segment_index == 0),
                progress_bar=False,
                tb_log_name="PPO",
            )
            if model.num_timesteps != checkpoint_step:
                raise RuntimeError(
                    f"{mode} expected {checkpoint_step}, got {model.num_timesteps}"
                )
            model.save(str(model_dir / f"{checkpoint_step}_steps"))
            callback.write_snapshot(force=True)
            print(
                f"[{mode}] complete rollout update checkpoint {checkpoint_step}",
                flush=True,
            )

        model.save(str(model_dir / "final"))
        checkpoint = PPO.load(str(model_dir / "20480_steps.zip"), device="cpu")
        final = PPO.load(str(model_dir / "final.zip"), device="cpu")
        checkpoint_fp = component_fingerprints(checkpoint)
        final_fp = component_fingerprints(final)
        if checkpoint_fp != final_fp:
            raise RuntimeError(f"{mode} 20480 checkpoint and final tensors differ")
        (log_dir / "episode_seed_schedule.json").write_text(
            json.dumps(schedule.reset_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {
            "mode": mode,
            "timesteps": model.num_timesteps,
            "segments": 4,
            "segment_timesteps": SEGMENT_TIMESTEPS,
            "complete_rollouts": 20,
            "algorithm_seed": ALGORITHM_SEED,
            "algorithm_rng_initialized_once_before_first_learn": True,
            "common_init_fingerprints": loaded,
            "final_fingerprints": final_fp,
            "checkpoint_20480_and_final_identical": True,
            "reset_count": len(schedule.reset_records),
            "first_seed": schedule.reset_records[0]["seed"],
            "last_seed": schedule.reset_records[-1]["seed"],
            "recent_100": callback.snapshot(),
        }
        (log_dir / "training_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"STOP {mode}: exactly {TOTAL_TIMESTEPS} steps", flush=True)
        return result
    finally:
        env.close()


def verify_schedules() -> dict:
    zero = json.loads(
        (LOG_ROOT / "zero_geometry" / "episode_seed_schedule.json").read_text(encoding="utf-8")
    )
    actual = json.loads(
        (LOG_ROOT / "actual_geometry" / "episode_seed_schedule.json").read_text(encoding="utf-8")
    )
    common = min(len(zero), len(actual))
    if zero[:common] != actual[:common]:
        for index in range(common):
            if zero[index] != actual[index]:
                raise RuntimeError(f"Training scene mismatch at episode index {index}")
        raise RuntimeError("Training scene schedule mismatch")
    return {
        "passed": True,
        "zero_reset_count": len(zero),
        "actual_reset_count": len(actual),
        "common_prefix_count": common,
        "common_prefix_seed_and_scene_fingerprints_identical": True,
    }


def train_all() -> None:
    refuse_training_outputs()
    common_report = preflight()
    expected = common_report["zero"]["component_fingerprints"]
    zero = train_one("zero", expected)
    actual = train_one("actual", expected)
    schedules = verify_schedules()
    report = {"zero": zero, "actual": actual, "schedules": schedules}
    (BENCHMARK_ROOT / "geometry_training_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Both independent Geometry branches completed and stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-common-init", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_common_init:
        prepare_common_init()
    elif args.preflight:
        preflight()
    else:
        train_all()


if __name__ == "__main__":
    main()
