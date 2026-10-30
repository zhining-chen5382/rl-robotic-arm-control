"""Shared metrics/checkpoint callback for Direct vs Headroom 20k ablation."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ResidualAblationCallback(BaseCallback):
    FIELDNAMES = [
        "timesteps", "recent_episode_count", "recent_collision_rate",
        "recent_5mm_10deg_ik_success_rate", "recent_10mm_10deg_ik_success_rate",
        "recent_collision_free_5mm_success_rate",
        "recent_collision_free_10mm_success_rate",
        "recent_mean_position_error_mm", "recent_mean_orientation_error_deg",
        "recent_mean_episode_min_distance", "recent_mean_abs_nominal_residual",
        "recent_mean_abs_actually_applied_residual",
        "recent_dimension_clipping_rate", "recent_step_any_clipping_rate",
        "recent_mean_inward_headroom", "recent_mean_outward_headroom",
    ]

    def __init__(
        self,
        *,
        mode: str,
        checkpoint_dir: str | Path,
        metrics_csv_path: str | Path,
        checkpoint_steps=(5_000, 10_000, 15_000, 20_000),
        window_size: int = 100,
        max_timesteps: int = 20_000,
        log_interval_steps: int = 1_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        if mode not in {"direct", "headroom"}:
            raise ValueError("mode must be 'direct' or 'headroom'")
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_csv_path = Path(metrics_csv_path)
        self.checkpoint_steps = tuple(sorted(set(int(x) for x in checkpoint_steps)))
        self.window_size = int(window_size)
        self.max_timesteps = int(max_timesteps)
        self.log_interval_steps = int(log_interval_steps)
        self._saved_steps: set[int] = set()
        self._last_csv_step = -1
        self._episodes: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._minimum_distance = np.inf
        self._nominal_abs: list[float] = []
        self._applied_abs: list[float] = []
        self._dimension_clip: list[float] = []
        self._step_clip: list[float] = []
        self._inward_headroom: list[float] = []
        self._outward_headroom: list[float] = []

    @staticmethod
    def _action(info: dict, key: str) -> np.ndarray:
        value = np.asarray(info[key], dtype=np.float32)
        if value.shape != (7,):
            raise RuntimeError(f"info[{key!r}] must have shape (7,), got {value.shape}")
        return value

    def _record_step(self, info: dict) -> None:
        base = self._action(info, "base_action")
        final = self._action(info, "final_action")
        clipped_residual = self._action(info, "clipped_residual_action")
        if self.mode == "direct":
            nominal = self._action(info, "scaled_residual")
            preclip = (base + nominal).astype(np.float32)
        else:
            nominal = self._action(info, "nominal_scaled_residual")
            preclip = self._action(info, "preclip_final_action")
            headroom = self._action(info, "headroom")
            inward = base * clipped_residual < 0.0
            outward = base * clipped_residual > 0.0
            if np.any(inward):
                self._inward_headroom.extend(headroom[inward].astype(float).tolist())
            if np.any(outward):
                self._outward_headroom.extend(headroom[outward].astype(float).tolist())
        clipping = preclip != final
        self._minimum_distance = min(
            self._minimum_distance, float(info["min_obstacle_distance"])
        )
        self._nominal_abs.append(float(np.mean(np.abs(nominal))))
        self._applied_abs.append(float(np.mean(np.abs(final - base))))
        self._dimension_clip.append(float(np.mean(clipping)))
        self._step_clip.append(float(np.any(clipping)))

    def _finish_episode(self, info: dict) -> None:
        pos_m = float(info["pos_distance"])
        orient_deg = float(info["orient_distance_deg"])
        collision = bool(info.get("had_collision", False))
        success_5 = pos_m < 0.005 and orient_deg < 10.0
        success_10 = pos_m < 0.010 and orient_deg < 10.0
        self._episodes.append(
            {
                "collision": float(collision),
                "success_5": float(success_5),
                "success_10": float(success_10),
                "collision_free_5": float(success_5 and not collision),
                "collision_free_10": float(success_10 and not collision),
                "position_error_mm": pos_m * 1000.0,
                "orientation_error_deg": orient_deg,
                "minimum_distance": float(self._minimum_distance),
                "nominal_abs": float(np.mean(self._nominal_abs)),
                "applied_abs": float(np.mean(self._applied_abs)),
                "dimension_clip": float(np.mean(self._dimension_clip)),
                "step_clip": float(np.mean(self._step_clip)),
                "inward_headroom": (
                    float(np.mean(self._inward_headroom))
                    if self._inward_headroom else float("nan")
                ),
                "outward_headroom": (
                    float(np.mean(self._outward_headroom))
                    if self._outward_headroom else float("nan")
                ),
            }
        )
        self._reset_episode()

    def _snapshot(self) -> dict[str, float] | None:
        if not self._episodes:
            return None
        items = list(self._episodes)

        def mean(key: str) -> float:
            values = np.asarray([item[key] for item in items], dtype=np.float64)
            return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")

        return {
            "timesteps": int(self.num_timesteps),
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
            "recent_mean_abs_actually_applied_residual": mean("applied_abs"),
            "recent_dimension_clipping_rate": mean("dimension_clip"),
            "recent_step_any_clipping_rate": mean("step_clip"),
            "recent_mean_inward_headroom": mean("inward_headroom"),
            "recent_mean_outward_headroom": mean("outward_headroom"),
        }

    def _publish(self, force: bool = False) -> None:
        snapshot = self._snapshot()
        if snapshot is None:
            return
        for key, value in snapshot.items():
            if key not in {"timesteps", "recent_episode_count"}:
                self.logger.record(f"ablation/{key}", value)
        due = force or (
            self.log_interval_steps > 0
            and self.num_timesteps % self.log_interval_steps == 0
        )
        if not due or self._last_csv_step == self.num_timesteps:
            return
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            if not exists:
                writer.writeheader()
            writer.writerow(snapshot)
        self._last_csv_step = self.num_timesteps

    def _save_checkpoints(self) -> None:
        for step in self.checkpoint_steps:
            if self.num_timesteps >= step and step not in self._saved_steps:
                path = self.checkpoint_dir / f"{step}_steps"
                self.model.save(str(path))
                self._saved_steps.add(step)
                if self.verbose:
                    print(f"[{self.mode}] checkpoint: {path}.zip", flush=True)

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if len(infos) != 1 or len(dones) != 1:
            raise RuntimeError("Residual ablation requires exactly one environment")
        self._record_step(infos[0])
        if bool(dones[0]):
            self._finish_episode(infos[0])
        self._save_checkpoints()
        self._publish(force=self.num_timesteps >= self.max_timesteps)
        return self.num_timesteps < self.max_timesteps


__all__ = ["ResidualAblationCallback"]
