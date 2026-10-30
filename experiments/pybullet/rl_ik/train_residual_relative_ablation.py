"""Controlled 20,480-step Uncentered-vs-Robot-Centered ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import ik_6d.rl_env_6d_residual_relative as relative_module
from ik_6d.geometry_ablation_callback import GeometryAblationMetricsCallback
from ik_6d.isolated_episode_seed_wrapper import IsolatedEpisodeSeedWrapper
from ik_6d.residual_relative_features import ResidualRelativeFeatureExtractor
from ik_6d.rl_env_6d_residual_geometry import GeometryResidualObstacleEnv
from ik_6d.rl_env_6d_residual_relative import RelativeResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models" / "residual_relative_ablation"
LOG_ROOT = ROOT / "logs" / "residual_relative_ablation"
BENCHMARK_ROOT = ROOT / "benchmarks" / "residual_relative_ablation"
COMMON_INIT = MODEL_ROOT / "relative_residual_common_init.zip"
INITIALIZATION_SEED = 20_260_816
ALGORITHM_SEED = 20_260_816
EPISODE_BASE_SEED = 2_026_081_600
PERFORMANCE_BASE_SEED = 2_026_081_600
SEGMENT_TIMESTEPS = 5_120
TOTAL_TIMESTEPS = 20_480
CHECKPOINT_STEPS = (5_120, 10_240, 15_360, 20_480)
RELATION_MODES = ("uncentered", "robot_centered")


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
        "pointnet": policy_fingerprint(model, ("features_extractor.pointnet.",)),
        "geometry_mlp": policy_fingerprint(
            model, ("features_extractor.geometry_mlp.",)
        ),
        "relation_encoder": policy_fingerprint(
            model, ("features_extractor.relation_encoder.",)
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
        "features_extractor_class": ResidualRelativeFeatureExtractor,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.Tanh,
        "log_std_init": -2.0,
    }


def make_env(mode: str):
    relative_env = RelativeResidualObstacleEnv(
        relation_mode=mode,
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    isolated = IsolatedEpisodeSeedWrapper(
        relative_env,
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
        "features_dim": 344,
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


class _Timer:
    def __init__(self) -> None:
        self.enabled = False
        self.seconds = 0.0
        self.calls = 0

    def add(self, elapsed: float) -> None:
        if self.enabled:
            self.seconds += elapsed
            self.calls += 1

    def mean_ms(self) -> float:
        return 1000.0 * self.seconds / self.calls if self.calls else 0.0


def _run_geometry_reference(steps: int) -> dict:
    random.seed(PERFORMANCE_BASE_SEED)
    np.random.seed(PERFORMANCE_BASE_SEED)
    env = GeometryResidualObstacleEnv(
        geometry_mode="actual",
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    action = np.zeros(7, dtype=np.float32)
    elapsed = 0.0
    episode = 0
    try:
        env.reset(seed=PERFORMANCE_BASE_SEED)
        for _ in range(steps):
            started = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(action)
            elapsed += time.perf_counter() - started
            if terminated or truncated:
                episode += 1
                env.reset(seed=PERFORMANCE_BASE_SEED + episode)
    finally:
        env.close()
    return {
        "steps": steps,
        "total_step_seconds": elapsed,
        "mean_step_ms": 1000.0 * elapsed / steps,
        "steps_per_second": steps / elapsed,
    }


def _run_relative_performance(mode: str, steps: int) -> dict:
    random.seed(PERFORMANCE_BASE_SEED)
    np.random.seed(PERFORMANCE_BASE_SEED)
    transform_timer = _Timer()
    relation_math_timer = _Timer()
    observation_timer = _Timer()
    original_transform = relative_module.transform_obstacle_points_to_base_frame
    original_relation = relative_module.compute_relation_cloud

    def timed_transform(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_transform(*args, **kwargs)
        finally:
            transform_timer.add(time.perf_counter() - started)

    def timed_relation(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_relation(*args, **kwargs)
        finally:
            relation_math_timer.add(time.perf_counter() - started)

    relative_module.transform_obstacle_points_to_base_frame = timed_transform
    relative_module.compute_relation_cloud = timed_relation
    env = None
    action = np.zeros(7, dtype=np.float32)
    elapsed = 0.0
    episode = 0
    try:
        env = RelativeResidualObstacleEnv(
            relation_mode=mode,
            residual_scale=0.05,
            residual_penalty_weight=1e-4,
        )
        original_build = env._build_residual_observation

        def timed_build(observation):
            started = time.perf_counter()
            try:
                return original_build(observation)
            finally:
                observation_timer.add(time.perf_counter() - started)

        env._build_residual_observation = timed_build
        env.reset(seed=PERFORMANCE_BASE_SEED)
        transform_timer.enabled = True
        relation_math_timer.enabled = True
        observation_timer.enabled = True
        for _ in range(steps):
            started = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(action)
            elapsed += time.perf_counter() - started
            if terminated or truncated:
                episode += 1
                transform_timer.enabled = False
                relation_math_timer.enabled = False
                observation_timer.enabled = False
                env.reset(seed=PERFORMANCE_BASE_SEED + episode)
                transform_timer.enabled = True
                relation_math_timer.enabled = True
                observation_timer.enabled = True
    finally:
        if env is not None:
            env.close()
        relative_module.transform_obstacle_points_to_base_frame = original_transform
        relative_module.compute_relation_cloud = original_relation
    if not (
        transform_timer.calls == relation_math_timer.calls
        == observation_timer.calls == steps
    ):
        raise RuntimeError(
            "Performance instrumentation call counts differ from environment steps"
        )
    return {
        "mode": mode,
        "steps": steps,
        "total_step_seconds": elapsed,
        "mean_step_ms": 1000.0 * elapsed / steps,
        "steps_per_second": steps / elapsed,
        "world_to_base_transform_mean_ms": transform_timer.mean_ms(),
        "relation_vector_math_mean_ms": relation_math_timer.mean_ms(),
        "relation_cloud_construction_mean_ms": (
            transform_timer.mean_ms() + relation_math_timer.mean_ms()
        ),
        "whole_observation_construction_mean_ms": observation_timer.mean_ms(),
        "timed_observation_count": observation_timer.calls,
    }


def performance_benchmark(steps: int = 300) -> dict:
    if not 200 <= steps <= 500:
        raise ValueError("Performance benchmark must use 200 to 500 steps")
    output_path = BENCHMARK_ROOT / "relative_observation_performance.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output_path}")
    reference = _run_geometry_reference(steps)
    arms = {mode: _run_relative_performance(mode, steps) for mode in RELATION_MODES}
    for item in arms.values():
        item["step_slowdown_vs_actual_geometry"] = (
            item["mean_step_ms"] / reference["mean_step_ms"]
        )
    severe = any(
        item["step_slowdown_vs_actual_geometry"] >= 10.0
        for item in arms.values()
    )
    report = {
        "model_learn_called": False,
        "actual_geometry_reference": reference,
        "uncentered": arms["uncentered"],
        "robot_centered": arms["robot_centered"],
        "severe_order_of_magnitude_slowdown": severe,
        "threshold": "step slowdown >= 10x vs Actual Geometry Wrapper",
    }
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if severe:
        print("STOP: severe performance slowdown; do not create common init")
    return report


def prepare_common_init() -> dict:
    if COMMON_INIT.exists():
        raise FileExistsError(f"Refusing to overwrite common init: {COMMON_INIT}")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(INITIALIZATION_SEED)
    np.random.seed(INITIALIZATION_SEED)
    torch.manual_seed(INITIALIZATION_SEED)
    env, _ = make_env("uncentered")
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
    creation_path = BENCHMARK_ROOT / "common_init_creation.json"
    if creation_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {creation_path}")
    creation_path.write_text(
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
    output_path = BENCHMARK_ROOT / "common_init_preflight.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {output_path}")
    uncentered = deterministic_sanity("uncentered")
    robot = deterministic_sanity("robot_centered")
    if uncentered["component_fingerprints"] != robot["component_fingerprints"]:
        raise RuntimeError("The two modes did not load identical common init tensors")
    left = [item["scene_fingerprint"] for item in uncentered["reset_records"]]
    right = [item["scene_fingerprint"] for item in robot["reset_records"]]
    if left != right:
        raise RuntimeError("Uncentered/Robot-Centered sanity reset scenes differ")
    report = {
        "passed": True,
        "all_component_fingerprints_identical": True,
        "sanity_scenes_identical": True,
        "uncentered": uncentered,
        "robot_centered": robot,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "PASS common init preflight: "
        f"{uncentered['component_fingerprints']['full_policy']}"
    )
    return report


def set_algorithm_rng_once() -> None:
    random.seed(ALGORITHM_SEED)
    np.random.seed(ALGORITHM_SEED)
    torch.manual_seed(ALGORITHM_SEED)


def refuse_training_outputs() -> None:
    expected = []
    for mode in RELATION_MODES:
        model_dir = MODEL_ROOT / mode
        log_dir = LOG_ROOT / mode
        expected.extend(
            [
                model_dir / "common_init_loaded.zip",
                model_dir / "final.zip",
                log_dir / "training_metrics.csv",
                log_dir / "episode_seed_schedule.json",
            ]
        )
        expected.extend(model_dir / f"{step}_steps.zip" for step in CHECKPOINT_STEPS)
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite Relative outputs: {existing}")


def train_one(mode: str, expected_common: dict[str, str]) -> dict:
    model_dir = MODEL_ROOT / mode
    log_dir = LOG_ROOT / mode
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

        # The only manual algorithm RNG initialization for this branch.
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
            print(f"[{mode}] complete rollout checkpoint {checkpoint_step}", flush=True)

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
            "episode_base_seed": EPISODE_BASE_SEED,
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
    records = {
        mode: json.loads(
            (LOG_ROOT / mode / "episode_seed_schedule.json").read_text(
                encoding="utf-8"
            )
        )
        for mode in RELATION_MODES
    }
    common = min(len(items) for items in records.values())
    if records["uncentered"][:common] != records["robot_centered"][:common]:
        raise RuntimeError("Training seed or scene fingerprint schedule mismatch")
    return {
        "passed": True,
        "uncentered_reset_count": len(records["uncentered"]),
        "robot_centered_reset_count": len(records["robot_centered"]),
        "common_prefix_count": common,
        "common_prefix_seed_and_scene_fingerprints_identical": True,
    }


def train_all() -> None:
    performance_path = BENCHMARK_ROOT / "relative_observation_performance.json"
    if not performance_path.is_file():
        raise FileNotFoundError("Run --benchmark-performance before training")
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    if performance["severe_order_of_magnitude_slowdown"]:
        raise RuntimeError("Performance benchmark requires stopping before training")
    refuse_training_outputs()
    common_report = preflight()
    expected = common_report["uncentered"]["component_fingerprints"]
    uncentered = train_one("uncentered", expected)
    robot = train_one("robot_centered", expected)
    schedules = verify_schedules()
    report = {
        "uncentered": uncentered,
        "robot_centered": robot,
        "schedules": schedules,
    }
    output_path = BENCHMARK_ROOT / "relative_training_summary.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output_path}")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Both independent Relative branches completed and stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark-performance", action="store_true")
    group.add_argument("--prepare-common-init", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train-all", action="store_true")
    parser.add_argument("--performance-steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark_performance:
        performance_benchmark(args.performance_steps)
    elif args.prepare_common_init:
        prepare_common_init()
    elif args.preflight:
        preflight()
    else:
        train_all()


if __name__ == "__main__":
    main()
