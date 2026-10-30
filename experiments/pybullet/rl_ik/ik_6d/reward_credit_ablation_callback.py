"""Metrics callback shared by Old Reward/Credit and V2-Balanced training."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardCreditAblationCallback(BaseCallback):
    PPO_KEYS = {
        "ppo_approx_kl": "train/approx_kl",
        "ppo_clip_fraction": "train/clip_fraction",
        "ppo_entropy_loss": "train/entropy_loss",
        "ppo_policy_gradient_loss": "train/policy_gradient_loss",
        "ppo_value_loss": "train/value_loss",
        "ppo_explained_variance": "train/explained_variance",
    }
    V2_COMPONENT_KEYS = {
        "reward_v2_ik_progress_mean": "reward_v2_ik_progress",
        "reward_v2_safety_progress_mean": "reward_v2_safety_progress",
        "reward_v2_zone_mean": "reward_v2_zone",
        "reward_v2_joint_limit_mean": "reward_v2_joint_limit",
        "reward_v2_residual_mean": "reward_v2_residual",
        "reward_v2_terminal_mean": "reward_v2_terminal",
    }
    FIELDNAMES = [
        "timesteps", "branch", "rollout_step_count", "recent_episode_count",
        "recent_5mm_10deg_ik_success_rate", "recent_10mm_10deg_ik_success_rate",
        "recent_collision_rate", "recent_collision_free_5mm_success_rate",
        "recent_collision_free_10mm_success_rate",
        "recent_mean_position_error_mm", "recent_mean_orientation_error_deg",
        "recent_mean_episode_min_distance", "recent_mean_episode_length",
        "recent_mean_abs_nominal_residual", "recent_mean_abs_effective_residual",
        "recent_mean_effective_residual_l2",
        "recent_numerical_clipping_dimension_rate",
        "recent_numerical_clipping_step_rate", "recent_mean_inward_headroom",
        "recent_mean_outward_headroom", "reward_mean", "reward_std", "reward_p1",
        "reward_p99", "reward_max_abs", "mean_old_returned_reward",
        "old_post_collision_transition_count",
        "old_post_collision_transition_fraction",
        "old_mean_steps_after_first_collision", "v2_success_terminal_count",
        "v2_collision_terminal_count", "v2_timeout_count",
        "v2_post_collision_transition_count", *V2_COMPONENT_KEYS,
        *PPO_KEYS,
    ]

    def __init__(
        self,
        metrics_csv_path: str | Path,
        *,
        branch: str,
        window_size: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if branch not in {"old_reward", "v2_balanced"}:
            raise ValueError(f"unsupported branch: {branch}")
        self.metrics_csv_path = Path(metrics_csv_path)
        self.branch = branch
        self.window_size = int(window_size)
        self._episodes: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._all_episodes: list[dict[str, float]] = []
        self._all_rewards: list[float] = []
        self._all_v2_components = {
            key: [] for key in self.V2_COMPONENT_KEYS.values()
        }
        self._rollout_rewards: list[float] = []
        self._rollout_v2_components = {
            key: [] for key in self.V2_COMPONENT_KEYS.values()
        }
        self._last_written_step = -1
        self.total_post_collision_transitions = 0
        self.collision_episodes = 0
        self.v2_success_terminals = 0
        self.v2_collision_terminals = 0
        self.v2_timeouts = 0
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._episode_length = 0
        self._episode_minimum_distance = np.inf
        self._episode_had_collision = False
        self._episode_post_collision_steps = 0
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

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _reward_stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {key: 0.0 for key in (
                "reward_mean", "reward_std", "reward_p1", "reward_p99",
                "reward_max_abs",
            )}
        return {
            "reward_mean": float(np.mean(array)),
            "reward_std": float(np.std(array)),
            "reward_p1": float(np.percentile(array, 1)),
            "reward_p99": float(np.percentile(array, 99)),
            "reward_max_abs": float(np.max(np.abs(array))),
        }

    def _record_step(self, info: dict, reward: float) -> None:
        if self.branch == "v2_balanced" and self._episode_had_collision:
            raise RuntimeError("V2 produced a strict post-collision transition")
        if self._episode_had_collision:
            self._episode_post_collision_steps += 1
            self.total_post_collision_transitions += 1
        self._episode_length += 1
        self._episode_minimum_distance = min(
            self._episode_minimum_distance,
            float(info["min_obstacle_distance"]),
        )
        self._episode_had_collision = bool(
            self._episode_had_collision or info.get("collision", False)
        )

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
        self._nominal_abs.append(float(np.mean(np.abs(nominal))))
        self._effective_abs.append(float(np.mean(np.abs(effective))))
        self._effective_l2.append(float(np.linalg.norm(effective)))
        self._dimension_clips.append(float(np.mean(numerical_clip)))
        self._step_clips.append(float(np.any(numerical_clip)))

        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise RuntimeError("non-finite training reward")
        self._rollout_rewards.append(reward_value)
        self._all_rewards.append(reward_value)
        if self.branch == "v2_balanced":
            # DummyVecEnv stores PPO rewards as float32; info keeps float64.
            if abs(reward_value - float(info["reward_v2_total"])) > 1e-6:
                raise RuntimeError("V2 returned reward differs from info decomposition")
            for info_key in self._rollout_v2_components:
                component = float(info[info_key])
                if not np.isfinite(component):
                    raise RuntimeError(f"non-finite V2 component {info_key}")
                self._rollout_v2_components[info_key].append(component)
                self._all_v2_components[info_key].append(component)

    def _finish_episode(self, info: dict) -> None:
        position_m = float(info["pos_distance"])
        orientation_deg = float(info["orient_distance_deg"])
        collision = bool(info.get("had_collision", False))
        success_5 = position_m < 0.005 and orientation_deg < 10.0
        success_10 = position_m < 0.010 and orientation_deg < 10.0
        terminal_type = str(info.get("reward_v2_terminal_type", "old"))
        if collision:
            self.collision_episodes += 1
        if terminal_type == "success":
            self.v2_success_terminals += 1
        elif terminal_type == "collision":
            self.v2_collision_terminals += 1
        elif terminal_type == "timeout":
            self.v2_timeouts += 1
        item = {
            "collision": float(collision),
            "success_5": float(success_5),
            "success_10": float(success_10),
            "collision_free_5": float(success_5 and not collision),
            "collision_free_10": float(success_10 and not collision),
            "position_error_mm": position_m * 1000.0,
            "orientation_error_deg": orientation_deg,
            "minimum_distance": float(self._episode_minimum_distance),
            "episode_length": float(self._episode_length),
            "post_collision_steps": float(self._episode_post_collision_steps),
            "nominal_abs": self._mean(self._nominal_abs),
            "effective_abs": self._mean(self._effective_abs),
            "effective_l2": self._mean(self._effective_l2),
            "dimension_clip": self._mean(self._dimension_clips),
            "step_clip": self._mean(self._step_clips),
            "inward_headroom": self._mean(self._inward_headroom),
            "outward_headroom": self._mean(self._outward_headroom),
        }
        self._episodes.append(item)
        self._all_episodes.append(item)
        self._reset_episode()

    def _episode_snapshot(self) -> dict[str, float | int]:
        if not self._episodes:
            return {"recent_episode_count": 0}
        items = list(self._episodes)

        def mean(key: str) -> float:
            return float(np.mean([item[key] for item in items]))

        return {
            "recent_episode_count": len(items),
            "recent_5mm_10deg_ik_success_rate": mean("success_5"),
            "recent_10mm_10deg_ik_success_rate": mean("success_10"),
            "recent_collision_rate": mean("collision"),
            "recent_collision_free_5mm_success_rate": mean("collision_free_5"),
            "recent_collision_free_10mm_success_rate": mean("collision_free_10"),
            "recent_mean_position_error_mm": mean("position_error_mm"),
            "recent_mean_orientation_error_deg": mean("orientation_error_deg"),
            "recent_mean_episode_min_distance": mean("minimum_distance"),
            "recent_mean_episode_length": mean("episode_length"),
            "recent_mean_abs_nominal_residual": mean("nominal_abs"),
            "recent_mean_abs_effective_residual": mean("effective_abs"),
            "recent_mean_effective_residual_l2": mean("effective_l2"),
            "recent_numerical_clipping_dimension_rate": mean("dimension_clip"),
            "recent_numerical_clipping_step_rate": mean("step_clip"),
            "recent_mean_inward_headroom": mean("inward_headroom"),
            "recent_mean_outward_headroom": mean("outward_headroom"),
        }

    def _write_rollout(self) -> None:
        if not self._rollout_rewards:
            return
        step = int(self.model.num_timesteps)
        if step == self._last_written_step:
            return
        episodes = list(self._all_episodes)
        row: dict[str, object] = {
            name: "" for name in self.FIELDNAMES
        }
        row.update({
            "timesteps": step,
            "branch": self.branch,
            "rollout_step_count": len(self._rollout_rewards),
            **self._episode_snapshot(),
            **self._reward_stats(self._rollout_rewards),
            "mean_old_returned_reward": (
                self._mean(self._rollout_rewards)
                if self.branch == "old_reward" else ""
            ),
            "old_post_collision_transition_count": (
                self.total_post_collision_transitions
                if self.branch == "old_reward" else ""
            ),
            "old_post_collision_transition_fraction": (
                self.total_post_collision_transitions / step
                if self.branch == "old_reward" else ""
            ),
            "old_mean_steps_after_first_collision": (
                self._mean([item["post_collision_steps"] for item in episodes if item["collision"]])
                if self.branch == "old_reward" else ""
            ),
            "v2_success_terminal_count": (
                self.v2_success_terminals if self.branch == "v2_balanced" else ""
            ),
            "v2_collision_terminal_count": (
                self.v2_collision_terminals if self.branch == "v2_balanced" else ""
            ),
            "v2_timeout_count": (
                self.v2_timeouts if self.branch == "v2_balanced" else ""
            ),
            "v2_post_collision_transition_count": (
                self.total_post_collision_transitions
                if self.branch == "v2_balanced" else ""
            ),
        })
        if self.branch == "v2_balanced":
            for column, info_key in self.V2_COMPONENT_KEYS.items():
                row[column] = self._mean(self._rollout_v2_components[info_key])
        logger_values = self.model.logger.name_to_value
        for column, logger_key in self.PPO_KEYS.items():
            value = logger_values.get(logger_key, "")
            row[column] = float(value) if value != "" else ""
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self._last_written_step = step
        self._rollout_rewards.clear()
        for values in self._rollout_v2_components.values():
            values.clear()

    def _on_rollout_start(self) -> None:
        # The previous rollout's PPO update has completed at this point.
        self._write_rollout()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        if len(infos) != 1 or len(dones) != 1 or len(rewards) != 1:
            raise RuntimeError("Reward/Credit ablation requires one environment")
        self._record_step(infos[0], float(rewards[0]))
        if bool(dones[0]):
            self._finish_episode(infos[0])
        return True

    def finalize_rollout(self) -> None:
        """Write the final rollout after ``learn()`` returns."""
        self._write_rollout()

    def summary(self) -> dict:
        items = self._all_episodes

        def mean(key: str) -> float:
            return self._mean([item[key] for item in items])

        return {
            "completed_episode_count": len(items),
            "collision_episode_count": self.collision_episodes,
            "mean_episode_length": mean("episode_length"),
            "post_collision_transition_count": self.total_post_collision_transitions,
            "post_collision_transition_fraction": (
                self.total_post_collision_transitions / len(self._all_rewards)
                if self._all_rewards else 0.0
            ),
            "mean_steps_after_first_collision": self._mean([
                item["post_collision_steps"] for item in items if item["collision"]
            ]),
            "reward": self._reward_stats(self._all_rewards),
            "v2_terminal_counts": {
                "success": self.v2_success_terminals,
                "collision": self.v2_collision_terminals,
                "timeout": self.v2_timeouts,
            },
            "v2_component_means": {
                key: self._mean(values)
                for key, values in self._all_v2_components.items()
            },
            "recent_100": self._episode_snapshot(),
        }


__all__ = ["RewardCreditAblationCallback"]
