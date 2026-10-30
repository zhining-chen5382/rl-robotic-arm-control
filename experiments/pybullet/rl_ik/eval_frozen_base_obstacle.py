"""评估冻结的原始 Panda 6D PPO IK 策略。

该脚本不会训练或修改模型。障碍环境评估时，策略只接收 Dict observation
中的 17 维 ``state``，不会接收点云或 PyBullet 计算的真实 clearance。
clearance 只用于最终评估指标。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO

from ik_6d.rl_env_6d import RobotIK6DEnv as BaseIKEnv
from ik_6d.rl_env_6d_obstacle import RobotIK6DEnv as ObstacleIKEnv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "panda_ppo_ik_6d_map.zip"
BEST_MODEL = (
    PROJECT_ROOT / "models" / "panda_ppo_ik_6d_map_best" / "best_model.zip"
)

STATE_DIM = 17
ACTION_DIM = 7
POSITION_5MM = 0.005
POSITION_10MM = 0.010
ORIENTATION_10DEG = np.deg2rad(10.0)


@dataclass(frozen=True)
class EvalResult:
    episodes: int
    success_5mm_10deg: float
    success_10mm_10deg: float
    mean_position_error_mm: float
    mean_orientation_error_deg: float
    collision_rate: float | None = None
    collision_free_success_5mm_10deg: float | None = None
    collision_free_success_10mm_10deg: float | None = None
    mean_episode_min_obstacle_distance_m: float | None = None


def seed_everything(seed: int) -> None:
    """设置脚本能够控制的随机数源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_frozen_model(model_path: Path) -> PPO:
    """加载只用于确定性推理的 PPO，并显式冻结其全部参数。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    model = PPO.load(
        str(model_path),
        device="cpu",
        custom_objects={"learning_rate": 1e-5},
    )
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)

    observation_shape = getattr(model.observation_space, "shape", None)
    action_shape = getattr(model.action_space, "shape", None)
    if observation_shape != (STATE_DIM,):
        raise ValueError(
            f"冻结模型观察空间应为 ({STATE_DIM},)，实际为 {observation_shape}"
        )
    if action_shape != (ACTION_DIM,):
        raise ValueError(
            f"冻结模型动作空间应为 ({ACTION_DIM},)，实际为 {action_shape}"
        )
    if any(parameter.requires_grad for parameter in model.policy.parameters()):
        raise RuntimeError("冻结失败：仍有 PPO policy 参数允许计算梯度")

    return model


def validate_state(state: np.ndarray, source: str) -> np.ndarray:
    """确认输入与原始 PPO 的 17 维 observation 严格兼容。"""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(
            f"{source} observation 应为 ({STATE_DIM},)，实际为 {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{source} observation 包含 NaN 或 Inf")
    return state


def summarize_errors(
    position_errors: list[float], orientation_errors: list[float]
) -> tuple[float, float, float, float]:
    positions = np.asarray(position_errors, dtype=np.float64)
    orientations = np.asarray(orientation_errors, dtype=np.float64)
    success_5 = (positions < POSITION_5MM) & (orientations < ORIENTATION_10DEG)
    success_10 = (positions < POSITION_10MM) & (orientations < ORIENTATION_10DEG)
    return (
        float(np.mean(success_5)),
        float(np.mean(success_10)),
        float(np.mean(positions) * 1000.0),
        float(np.degrees(np.mean(orientations))),
    )


def evaluate_clean(model: PPO, episodes: int, seed: int) -> EvalResult:
    """在原始无障碍环境中评估冻结基线。"""
    seed_everything(seed)
    env = BaseIKEnv(
        robot_id="panda",
        max_steps=100,
        pos_tolerance=POSITION_5MM,
        orient_tolerance=0.175,
    )
    env.action_space.seed(seed)
    position_errors: list[float] = []
    orientation_errors: list[float] = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            state = validate_state(observation, "无障碍环境")
            final_info = None

            for _ in range(env.max_steps):
                action, _ = model.predict(state, deterministic=True)
                observation, _, terminated, truncated, final_info = env.step(action)
                state = validate_state(observation, "无障碍环境")
                if terminated or truncated:
                    break

            if final_info is None:
                raise RuntimeError("无障碍 episode 未执行任何 step")
            position_errors.append(float(final_info["pos_distance"]))
            orientation_errors.append(float(final_info["orient_distance"]))
    finally:
        env.close()

    success_5, success_10, mean_pos_mm, mean_orient_deg = summarize_errors(
        position_errors, orientation_errors
    )
    return EvalResult(
        episodes=episodes,
        success_5mm_10deg=success_5,
        success_10mm_10deg=success_10,
        mean_position_error_mm=mean_pos_mm,
        mean_orientation_error_deg=mean_orient_deg,
    )


def evaluate_obstacle(model: PPO, episodes: int, seed: int) -> EvalResult:
    """在障碍环境中评估；策略只读取 observation['state']。"""
    seed_everything(seed)
    env = ObstacleIKEnv(
        robot_id="panda",
        max_steps=100,
        pos_tolerance=POSITION_5MM,
        orient_tolerance=0.175,
    )
    env.action_space.seed(seed)

    position_errors: list[float] = []
    orientation_errors: list[float] = []
    collisions: list[bool] = []
    episode_min_distances: list[float] = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            if set(observation) != {"state", "point_cloud"}:
                raise ValueError(
                    "障碍环境 Dict observation keys 与预期不一致: "
                    f"{tuple(observation.keys())}"
                )

            # 唯一传给冻结 PPO 的输入。point_cloud 和 clearance 均不进入策略。
            state = validate_state(observation["state"], "障碍环境 state")
            episode_min_distance = float(env._get_min_obstacle_distance())
            final_info = None

            for _ in range(env.max_steps):
                base_action, _ = model.predict(state, deterministic=True)
                observation, _, terminated, truncated, final_info = env.step(base_action)
                state = validate_state(observation["state"], "障碍环境 state")
                episode_min_distance = min(
                    episode_min_distance,
                    float(final_info["min_obstacle_distance"]),
                )
                if terminated or truncated:
                    break

            if final_info is None:
                raise RuntimeError("障碍 episode 未执行任何 step")
            position_errors.append(float(final_info["pos_distance"]))
            orientation_errors.append(float(final_info["orient_distance"]))
            collisions.append(bool(final_info.get("had_collision", False)))
            episode_min_distances.append(episode_min_distance)
    finally:
        env.close()

    positions = np.asarray(position_errors, dtype=np.float64)
    orientations = np.asarray(orientation_errors, dtype=np.float64)
    collision_array = np.asarray(collisions, dtype=bool)
    success_5_array = (
        (positions < POSITION_5MM) & (orientations < ORIENTATION_10DEG)
    )
    success_10_array = (
        (positions < POSITION_10MM) & (orientations < ORIENTATION_10DEG)
    )

    return EvalResult(
        episodes=episodes,
        success_5mm_10deg=float(np.mean(success_5_array)),
        success_10mm_10deg=float(np.mean(success_10_array)),
        collision_rate=float(np.mean(collision_array)),
        collision_free_success_5mm_10deg=float(
            np.mean(success_5_array & ~collision_array)
        ),
        collision_free_success_10mm_10deg=float(
            np.mean(success_10_array & ~collision_array)
        ),
        mean_position_error_mm=float(np.mean(positions) * 1000.0),
        mean_orientation_error_deg=float(np.degrees(np.mean(orientations))),
        mean_episode_min_obstacle_distance_m=float(
            np.mean(np.asarray(episode_min_distances, dtype=np.float64))
        ),
    )


def print_clean_result(result: EvalResult) -> None:
    print("\nA. 原始无障碍基线性能")
    print(f"  deterministic episodes: {result.episodes}")
    print(f"  位置 <5 mm 且姿态 <10°:  {result.success_5mm_10deg * 100:.2f}%")
    print(f"  位置 <10 mm 且姿态 <10°: {result.success_10mm_10deg * 100:.2f}%")
    print(f"  平均最终位置误差:          {result.mean_position_error_mm:.2f} mm")
    print(f"  平均最终姿态误差:          {result.mean_orientation_error_deg:.2f}°")


def print_obstacle_result(result: EvalResult) -> None:
    print("\nB. 冻结基线直接进入障碍环境后的性能")
    print(f"  deterministic episodes: {result.episodes}")
    print(f"  位置 <5 mm 且姿态 <10°:                {result.success_5mm_10deg * 100:.2f}%")
    print(f"  位置 <10 mm 且姿态 <10°:               {result.success_10mm_10deg * 100:.2f}%")
    print(f"  发生过碰撞的 episode 比例:             {result.collision_rate * 100:.2f}%")
    print(
        "  无碰撞 + 位置 <5 mm + 姿态 <10°:      "
        f"{result.collision_free_success_5mm_10deg * 100:.2f}%"
    )
    print(
        "  无碰撞 + 位置 <10 mm + 姿态 <10°:     "
        f"{result.collision_free_success_10mm_10deg * 100:.2f}%"
    )
    print(f"  平均最终位置误差:                       {result.mean_position_error_mm:.2f} mm")
    print(f"  平均最终姿态误差:                       {result.mean_orientation_error_deg:.2f}°")
    print(
        "  平均 episode 最小障碍物距离:           "
        f"{result.mean_episode_min_obstacle_distance_m:.4f} m "
        f"({result.mean_episode_min_obstacle_distance_m * 1000.0:.2f} mm)"
    )
    print("  注：距离查询上限为现有环境定义的 0.20 m，因此该均值在 0.20 m 处截断。")


def evaluate_model(
    model_path: Path, clean_episodes: int, obstacle_episodes: int, seed: int
) -> tuple[EvalResult, EvalResult]:
    print("=" * 72)
    print(f"冻结模型: {model_path}")
    print(f"随机种子: clean={seed}, obstacle={seed + 1_000_000}")
    print("策略输入: 仅 17 维原始 IK state；不输入 point_cloud/clearance")
    print("=" * 72)

    model = load_frozen_model(model_path)
    clean_result = evaluate_clean(model, clean_episodes, seed)
    obstacle_result = evaluate_obstacle(
        model, obstacle_episodes, seed + 1_000_000
    )
    print_clean_result(clean_result)
    print_obstacle_result(obstacle_result)
    return clean_result, obstacle_result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("episode 数必须大于 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--clean-episodes", type=positive_int, default=200)
    parser.add_argument("--obstacle-episodes", type=positive_int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare-best",
        action="store_true",
        help="使用相同设置额外评估 best_model.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        args.model.resolve(),
        args.clean_episodes,
        args.obstacle_episodes,
        args.seed,
    )
    if args.compare_best:
        evaluate_model(
            BEST_MODEL.resolve(),
            args.clean_episodes,
            args.obstacle_episodes,
            args.seed,
        )

    print("\n可复现性限制：现有环境内部使用全局 np.random，而非完全使用")
    print("Gymnasium self.np_random。本脚本固定全局 NumPy/Python/Torch 和 reset seed，")
    print("可复现当前单进程、固定调用顺序的运行，但不保证跨并行方式或环境改动后逐场景一致。")


if __name__ == "__main__":
    main()
