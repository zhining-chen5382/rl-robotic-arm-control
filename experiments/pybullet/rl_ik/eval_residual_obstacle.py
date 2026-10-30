"""顺序评估 Residual PPO checkpoints；同一时刻只保留一个 PyBullet 环境。"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from ik_6d.residual_pointnet_features import ResidualPointNetFeatureExtractor
from ik_6d.rl_env_6d_residual_obstacle import ResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "residual_obstacle"
DEFAULT_SEED = 1_000_042


@dataclass(frozen=True)
class Result:
    model: str
    episodes: int
    success_5: float
    success_10: float
    collision_rate: float
    collision_free_5: float
    collision_free_10: float
    mean_position_mm: float
    mean_orientation_deg: float
    mean_episode_min_distance: float
    mean_scaled_residual_l2: float
    mean_abs_scaled_residual: float
    final_action_clip_rate: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checkpoint_sequence() -> list[tuple[Path, int]]:
    return [
        (MODEL_DIR / "residual_init.zip", 200),
        (MODEL_DIR / "residual_5000_steps.zip", 200),
        (MODEL_DIR / "residual_10000_steps.zip", 200),
        (MODEL_DIR / "residual_15000_steps.zip", 200),
        (MODEL_DIR / "residual_20000_steps.zip", 200),
        (MODEL_DIR / "residual_final.zip", 500),
    ]


def evaluate(path: Path, episodes: int, seed: int) -> Result:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    # 每个模型重置相同随机源和调用顺序，尽量得到相同障碍场景序列。
    seed_everything(seed)
    env = ResidualObstacleEnv(
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    env.action_space.seed(seed)
    try:
        model = PPO.load(str(path), env=env, device="cpu")
        positions: list[float] = []
        orientations_deg: list[float] = []
        collisions: list[bool] = []
        episode_mins: list[float] = []
        scaled_l2: list[float] = []
        mean_abs_scaled: list[float] = []
        clip_events: list[bool] = []

        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_min = float(env.unwrapped._get_min_obstacle_distance())
            final_info = None
            for _ in range(env.unwrapped.max_steps):
                residual_action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(residual_action)
                final_info = info
                scaled = np.asarray(info["scaled_residual"], dtype=np.float32)
                base = np.asarray(info["base_action"], dtype=np.float32)
                proposed = base + scaled
                scaled_l2.append(float(np.linalg.norm(scaled, ord=2)))
                mean_abs_scaled.append(float(np.mean(np.abs(scaled))))
                clip_events.append(bool(np.any((proposed < -1.0) | (proposed > 1.0))))
                episode_min = min(episode_min, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break

            if final_info is None:
                raise RuntimeError("episode 未执行任何 step")
            positions.append(float(final_info["pos_distance"]))
            orientations_deg.append(float(final_info["orient_distance_deg"]))
            collisions.append(bool(final_info.get("had_collision", False)))
            episode_mins.append(episode_min)

        pos = np.asarray(positions)
        orient = np.asarray(orientations_deg)
        collision = np.asarray(collisions, dtype=bool)
        success_5 = (pos < 0.005) & (orient < 10.0)
        success_10 = (pos < 0.010) & (orient < 10.0)
        return Result(
            model=path.name,
            episodes=episodes,
            success_5=float(np.mean(success_5)),
            success_10=float(np.mean(success_10)),
            collision_rate=float(np.mean(collision)),
            collision_free_5=float(np.mean(success_5 & ~collision)),
            collision_free_10=float(np.mean(success_10 & ~collision)),
            mean_position_mm=float(np.mean(pos) * 1000.0),
            mean_orientation_deg=float(np.mean(orient)),
            mean_episode_min_distance=float(np.mean(episode_mins)),
            mean_scaled_residual_l2=float(np.mean(scaled_l2)),
            mean_abs_scaled_residual=float(np.mean(mean_abs_scaled)),
            final_action_clip_rate=float(np.mean(clip_events)),
        )
    finally:
        env.close()


def print_result(result: Result) -> None:
    print("\n" + "=" * 72)
    print(f"模型: {result.model} | episodes: {result.episodes}")
    print(f"<5 mm + <10° IK success:                  {result.success_5 * 100:.2f}%")
    print(f"<10 mm + <10° IK success:                 {result.success_10 * 100:.2f}%")
    print(f"collision episode rate:                    {result.collision_rate * 100:.2f}%")
    print(f"collision-free <5 mm + <10° success:       {result.collision_free_5 * 100:.2f}%")
    print(f"collision-free <10 mm + <10° success:      {result.collision_free_10 * 100:.2f}%")
    print(f"mean final position error:                  {result.mean_position_mm:.2f} mm")
    print(f"mean final orientation error:               {result.mean_orientation_deg:.2f}°")
    print(f"mean episode minimum obstacle distance:     {result.mean_episode_min_distance:.5f} m")
    print(f"mean scaled residual L2:                    {result.mean_scaled_residual_l2:.6f}")
    print(f"mean absolute scaled residual:              {result.mean_abs_scaled_residual:.6f}")
    print(f"final action clipping rate (per step):      {result.final_action_clip_rate * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model is not None:
        episodes = args.episodes if args.episodes is not None else 200
        if episodes <= 0:
            raise ValueError("episodes 必须大于 0")
        print_result(evaluate(args.model.resolve(), episodes, args.seed))
        return

    if args.episodes is not None:
        raise ValueError("--episodes 只能与 --model 一起使用")
    for path, episodes in checkpoint_sequence():
        print_result(evaluate(path, episodes, args.seed))

    print("\n可复现性限制：原障碍环境仍部分使用全局 np.random。")
    print("本脚本固定全局随机源、逐 episode seed 和调用顺序，但不修改原环境，")
    print("因此不能保证跨并行方式或未来环境实现变化后的逐场景严格一致。")


if __name__ == "__main__":
    main()
