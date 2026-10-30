"""Residual PPO 的近期窗口指标与精确步数 checkpoint callback。"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ResidualMetricsCallback(BaseCallback):
    """统计最近完整 episodes，并在指定步数保存 checkpoint。

    ``final_action_clip_rate`` 定义为：一个环境 step 中，只要
    ``base_action + scaled_residual`` 的任一维超出 [-1, 1]，该 step
    就计为发生 clipping。
    """

    FIELDNAMES = [
        "timesteps",
        "recent_episode_count",
        "recent_collision_rate",
        "recent_5mm_10deg_ik_success_rate",
        "recent_10mm_10deg_ik_success_rate",
        "recent_collision_free_5mm_success_rate",
        "recent_collision_free_10mm_success_rate",
        "recent_mean_episode_min_distance",
        "recent_mean_scaled_residual_l2",
        "recent_mean_abs_scaled_residual",
        "recent_final_action_clip_rate",
        "recent_mean_position_error_mm",
        "recent_mean_orientation_error_deg",
    ]

    def __init__(
        self,
        checkpoint_dir: str | Path,
        metrics_csv_path: str | Path,
        *,
        checkpoint_steps=(5_000, 10_000, 15_000, 20_000),
        window_size: int = 100,
        max_timesteps: int = 20_000,
        log_interval_steps: int = 1_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        if window_size <= 0:
            raise ValueError("window_size 必须大于 0")
        if max_timesteps <= 0:
            raise ValueError("max_timesteps 必须大于 0")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_csv_path = Path(metrics_csv_path)
        self.checkpoint_steps = tuple(sorted(set(int(x) for x in checkpoint_steps)))
        self.window_size = int(window_size)
        self.max_timesteps = int(max_timesteps)
        self.log_interval_steps = int(log_interval_steps)
        self._saved_steps: set[int] = set()
        self._last_csv_step = -1

        self._episodes: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._episode_min_distance = np.inf
        self._episode_scaled_l2: list[float] = []
        self._episode_mean_abs_scaled: list[float] = []
        self._episode_clip_events: list[float] = []

    @staticmethod
    def _as_action(info: dict, key: str) -> np.ndarray:
        value = np.asarray(info[key], dtype=np.float32)
        if value.shape != (7,):
            raise RuntimeError(f"info[{key!r}] shape 应为 (7,)，实际为 {value.shape}")
        return value

    def _record_step_metrics(self, info: dict) -> None:
        min_distance = float(info["min_obstacle_distance"])
        self._episode_min_distance = min(self._episode_min_distance, min_distance)

        scaled = self._as_action(info, "scaled_residual")
        base = self._as_action(info, "base_action")
        proposed_action = base + scaled
        clipped = bool(np.any((proposed_action < -1.0) | (proposed_action > 1.0)))

        self._episode_scaled_l2.append(float(np.linalg.norm(scaled, ord=2)))
        self._episode_mean_abs_scaled.append(float(np.mean(np.abs(scaled))))
        self._episode_clip_events.append(float(clipped))

    def _finish_episode(self, info: dict) -> None:
        pos_distance = float(info["pos_distance"])
        orient_deg = float(info["orient_distance_deg"])
        had_collision = bool(info.get("had_collision", False))
        success_5 = pos_distance < 0.005 and orient_deg < 10.0
        success_10 = pos_distance < 0.010 and orient_deg < 10.0

        self._episodes.append(
            {
                "collision": float(had_collision),
                "success_5": float(success_5),
                "success_10": float(success_10),
                "collision_free_5": float(success_5 and not had_collision),
                "collision_free_10": float(success_10 and not had_collision),
                "episode_min_distance": float(self._episode_min_distance),
                "mean_scaled_l2": float(np.mean(self._episode_scaled_l2)),
                "mean_abs_scaled": float(np.mean(self._episode_mean_abs_scaled)),
                "clip_rate": float(np.mean(self._episode_clip_events)),
                "position_error_mm": pos_distance * 1000.0,
                "orientation_error_deg": orient_deg,
            }
        )
        self._episode_min_distance = np.inf
        self._episode_scaled_l2.clear()
        self._episode_mean_abs_scaled.clear()
        self._episode_clip_events.clear()

    def _snapshot(self) -> dict[str, float] | None:
        if not self._episodes:
            return None
        items = list(self._episodes)

        def mean(key: str) -> float:
            return float(np.mean([item[key] for item in items]))

        return {
            "timesteps": int(self.num_timesteps),
            "recent_episode_count": len(items),
            "recent_collision_rate": mean("collision"),
            "recent_5mm_10deg_ik_success_rate": mean("success_5"),
            "recent_10mm_10deg_ik_success_rate": mean("success_10"),
            "recent_collision_free_5mm_success_rate": mean("collision_free_5"),
            "recent_collision_free_10mm_success_rate": mean("collision_free_10"),
            "recent_mean_episode_min_distance": mean("episode_min_distance"),
            "recent_mean_scaled_residual_l2": mean("mean_scaled_l2"),
            "recent_mean_abs_scaled_residual": mean("mean_abs_scaled"),
            "recent_final_action_clip_rate": mean("clip_rate"),
            "recent_mean_position_error_mm": mean("position_error_mm"),
            "recent_mean_orientation_error_deg": mean("orientation_error_deg"),
        }

    def _publish_snapshot(self, force_csv: bool = False) -> None:
        snapshot = self._snapshot()
        if snapshot is None:
            return
        for key, value in snapshot.items():
            if key not in ("timesteps", "recent_episode_count"):
                self.logger.record(f"residual/{key}", value)

        should_write = force_csv or (
            self.log_interval_steps > 0
            and self.num_timesteps % self.log_interval_steps == 0
        )
        if not should_write or self._last_csv_step == self.num_timesteps:
            return
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            if not exists:
                writer.writeheader()
            writer.writerow(snapshot)
        self._last_csv_step = self.num_timesteps

    def _save_due_checkpoints(self) -> None:
        for step in self.checkpoint_steps:
            if self.num_timesteps >= step and step not in self._saved_steps:
                path = self.checkpoint_dir / f"residual_{step}_steps"
                self.model.save(str(path))
                self._saved_steps.add(step)
                if self.verbose:
                    print(f"[checkpoint] saved: {path}.zip")

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if len(infos) != 1 or len(dones) != 1:
            raise RuntimeError("第一版 ResidualMetricsCallback 只允许单环境训练")

        info = infos[0]
        self._record_step_metrics(info)
        if bool(dones[0]):
            self._finish_episode(info)

        self._save_due_checkpoints()
        self._publish_snapshot(force_csv=self.num_timesteps >= self.max_timesteps)

        # PPO 的 rollout 长度为 1024；在 20k 中途停止，避免自动越过硬上限。
        return self.num_timesteps < self.max_timesteps


__all__ = ["ResidualMetricsCallback"]
