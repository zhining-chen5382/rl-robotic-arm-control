"""Shared recent-100 metrics for Zero-vs-Actual Geometry training."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GeometryAblationMetricsCallback(BaseCallback):
    FIELDNAMES = [
        "timesteps", "recent_episode_count", "recent_collision_rate",
        "recent_5mm_10deg_ik_success_rate", "recent_10mm_10deg_ik_success_rate",
        "recent_collision_free_5mm_success_rate",
        "recent_collision_free_10mm_success_rate",
        "recent_mean_position_error_mm", "recent_mean_orientation_error_deg",
        "recent_mean_episode_min_distance", "recent_mean_abs_nominal_residual",
        "recent_mean_abs_effective_residual", "recent_mean_effective_residual_l2",
        "recent_numerical_clipping_dimension_rate",
        "recent_numerical_clipping_step_rate", "recent_mean_inward_headroom",
        "recent_mean_outward_headroom",
    ]

    def __init__(
        self,
        metrics_csv_path: str | Path,
        *,
        window_size: int = 100,
        log_interval_steps: int = 1_024,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.metrics_csv_path = Path(metrics_csv_path)
        self.window_size = int(window_size)
        self.log_interval_steps = int(log_interval_steps)
        self._episodes: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._last_written_step = -1
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._minimum_distance = np.inf
        self._nominal_abs: list[float] = []
        self._effective_abs: list[float] = []
        self._effective_l2: list[float] = []
        self._dimension_clips: list[float] = []
        self._step_clips: list[float] = []
        self._inward_headroom: list[float] = []
        self._outward_headroom: list[float] = []

    @staticmethod
    def _array(info: dict, key: str) -> np.ndarray:
        value = np.asarray(info[key], dtype=np.float32)
        if value.shape != (7,):
            raise RuntimeError(f"info[{key!r}] must have shape (7,), got {value.shape}")
        return value

    def _record_step(self, info: dict) -> None:
        base = self._array(info, "base_action")
        clipped = self._array(info, "clipped_residual_action")
        nominal = self._array(info, "nominal_scaled_residual")
        effective = self._array(info, "effective_residual")
        headroom = self._array(info, "headroom")
        preclip = self._array(info, "preclip_final_action")
        final = self._array(info, "final_action")
        numerical_clip = preclip != final
        inward = base * clipped < 0.0
        outward = base * clipped > 0.0
        if np.any(inward):
            self._inward_headroom.extend(headroom[inward].astype(float).tolist())
        if np.any(outward):
            self._outward_headroom.extend(headroom[outward].astype(float).tolist())
        self._minimum_distance = min(
            self._minimum_distance, float(info["min_obstacle_distance"])
        )
        self._nominal_abs.append(float(np.mean(np.abs(nominal))))
        self._effective_abs.append(float(np.mean(np.abs(effective))))
        self._effective_l2.append(float(np.linalg.norm(effective)))
        self._dimension_clips.append(float(np.mean(numerical_clip)))
        self._step_clips.append(float(np.any(numerical_clip)))

    def _finish_episode(self, info: dict) -> None:
        position_m = float(info["pos_distance"])
        orientation_deg = float(info["orient_distance_deg"])
        collision = bool(info.get("had_collision", False))
        success_5 = position_m < 0.005 and orientation_deg < 10.0
        success_10 = position_m < 0.010 and orientation_deg < 10.0
        self._episodes.append(
            {
                "collision": float(collision),
                "success_5": float(success_5),
                "success_10": float(success_10),
                "collision_free_5": float(success_5 and not collision),
                "collision_free_10": float(success_10 and not collision),
                "position_error_mm": position_m * 1000.0,
                "orientation_error_deg": orientation_deg,
                "minimum_distance": float(self._minimum_distance),
                "nominal_abs": float(np.mean(self._nominal_abs)),
                "effective_abs": float(np.mean(self._effective_abs)),
                "effective_l2": float(np.mean(self._effective_l2)),
                "dimension_clip": float(np.mean(self._dimension_clips)),
                "step_clip": float(np.mean(self._step_clips)),
                "inward_headroom": float(np.mean(self._inward_headroom)),
                "outward_headroom": float(np.mean(self._outward_headroom)),
            }
        )
        self._reset_episode()

    def snapshot(self) -> dict[str, float] | None:
        if not self._episodes:
            return None
        items = list(self._episodes)

        def mean(key: str) -> float:
            return float(np.mean([item[key] for item in items]))

        return {
            "timesteps": int(self.model.num_timesteps),
            "recent_episode_count": len(items),
            "recent_collision_rate": mean("collision"),
            "recent_5mm_10deg_ik_success_rate": mean("success_5"),
            "recent_10mm_10deg_ik_success_rate": mean("success_10"),
            "recent_collision_free_5mm_success_rate": mean("collision_free_5"),
            "recent_collision_free_10mm_success_rate": mean("collision_free_10"),
            "recent_mean_position_error_mm": mean("position_error_mm"),
            "recent_mean_orientation_error_deg": mean("orientation_error_deg"),
            "recent_mean_episode_min_distance": mean("minimum_distance"),
            "recent_mean_abs_nominal_residual": mean("nominal_abs"),
            "recent_mean_abs_effective_residual": mean("effective_abs"),
            "recent_mean_effective_residual_l2": mean("effective_l2"),
            "recent_numerical_clipping_dimension_rate": mean("dimension_clip"),
            "recent_numerical_clipping_step_rate": mean("step_clip"),
            "recent_mean_inward_headroom": mean("inward_headroom"),
            "recent_mean_outward_headroom": mean("outward_headroom"),
        }

    def write_snapshot(self, *, force: bool = False) -> None:
        snapshot = self.snapshot()
        if snapshot is None:
            return
        step = int(snapshot["timesteps"])
        due = force or (
            self.log_interval_steps > 0 and step % self.log_interval_steps == 0
        )
        if not due or step == self._last_written_step:
            return
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            if not exists:
                writer.writeheader()
            writer.writerow(snapshot)
        self._last_written_step = step

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if len(infos) != 1 or len(dones) != 1:
            raise RuntimeError("Geometry ablation requires one environment")
        self._record_step(infos[0])
        if bool(dones[0]):
            self._finish_episode(infos[0])
        self.write_snapshot()
        return True


__all__ = ["GeometryAblationMetricsCallback"]
