"""Outer reward-only wrapper for Robot-Centered Relative residual control."""

from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np

from .reward_v2 import (
    RewardV2Config,
    TerminalType,
    V2_BALANCED_CONFIG,
    compute_reward_v2,
    ik_distance,
    joint_limit_component,
)


class RewardV2Wrapper(gym.Wrapper):
    """Replace the old returned reward while preserving observations/actions.

    The wrapped RelativeResidualObstacleEnv still owns frozen-base inference,
    Headroom action composition, and observation construction.  This outer
    layer ignores its scalar reward, keeps exactly one residual penalty from
    ``info``, and terminates immediately on the first collision.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        config: RewardV2Config = V2_BALANCED_CONFIG,
    ) -> None:
        super().__init__(env)
        self.config = config
        self._previous_D: float | None = None
        self._previous_distance: float | None = None
        self._episode_done = True

    @staticmethod
    def _D_from_observation(observation: dict) -> float:
        if not isinstance(observation, dict) or "state" not in observation:
            raise TypeError("Reward V2 requires the unchanged Dict observation")
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (17,):
            raise ValueError(f"Reward V2 expected state shape (17,), got {state.shape}")
        value = float(state[15])
        if not np.isfinite(value):
            raise ValueError("state[15] IK distance is not finite")
        return value

    def _minimum_distance(self) -> float:
        root = self.env.unwrapped
        helper = getattr(root, "_get_min_obstacle_distance", None)
        if helper is None or not callable(helper):
            raise AttributeError("underlying environment has no clearance helper")
        value = float(helper())
        if not np.isfinite(value):
            raise ValueError("minimum obstacle distance is not finite")
        return value

    def _joint_limit_reward(self) -> float:
        root = self.env.unwrapped
        return joint_limit_component(
            np.asarray(root.current_joints),
            root.joint_limits,
            self.config,
        )

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._previous_D = self._D_from_observation(observation)
        self._previous_distance = self._minimum_distance()
        self._episode_done = False
        return observation, info

    @staticmethod
    def _terminal_type(
        info: dict,
        lower_terminated: bool,
        lower_truncated: bool,
    ) -> TerminalType:
        # Collision has strict precedence over success and timeout.
        if bool(info.get("collision", False)):
            return "collision"
        if bool(info.get("success", False)):
            return "success"
        if lower_truncated:
            return "timeout"
        if lower_terminated:
            raise RuntimeError("unexpected non-success termination from lower environment")
        return "none"

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("reset() is required before step() or after episode end")
        if self._previous_D is None or self._previous_distance is None:
            raise RuntimeError("Reward V2 previous state was not initialized")

        observation, old_reward, lower_terminated, lower_truncated, info = (
            self.env.step(action)
        )
        info = dict(info)
        terminal_type = self._terminal_type(
            info,
            bool(lower_terminated),
            bool(lower_truncated),
        )

        D_next = ik_distance(
            float(info["pos_distance"]),
            float(info["orient_distance"]),
        )
        distance_next = float(info["min_obstacle_distance"])
        residual_reward = float(info.get("residual_penalty", 0.0))
        breakdown = compute_reward_v2(
            D_current=self._previous_D,
            D_next=D_next,
            distance_current=self._previous_distance,
            distance_next=distance_next,
            joint_limit=self._joint_limit_reward(),
            residual=residual_reward,
            terminal_type=terminal_type,
            config=self.config,
        )

        if terminal_type == "collision":
            terminated, truncated = True, False
        elif terminal_type == "success":
            terminated, truncated = True, False
        elif terminal_type == "timeout":
            terminated, truncated = False, True
        else:
            terminated, truncated = False, False

        info.update({
            "reward_v2_total": breakdown.total,
            "reward_v2_ik_progress": breakdown.ik_progress,
            "reward_v2_safety_progress": breakdown.safety_progress,
            "reward_v2_zone": breakdown.zone,
            "reward_v2_joint_limit": breakdown.joint_limit,
            "reward_v2_residual": breakdown.residual,
            "reward_v2_terminal": breakdown.terminal,
            "reward_v2_terminal_type": breakdown.terminal_type,
            "D_current": breakdown.D_current,
            "D_next": breakdown.D_next,
            "delta_D": breakdown.delta_D,
            "normalized_ik_progress": breakdown.normalized_ik_progress,
            "distance_current": breakdown.distance_current,
            "distance_next": breakdown.distance_next,
            "danger_current": breakdown.danger_current,
            "danger_next": breakdown.danger_next,
            "raw_safety_progress": breakdown.raw_safety_progress,
            "normalized_safety_progress": breakdown.normalized_safety_progress,
            "old_env_reward": float(old_reward),
            "reward_v2_config": asdict(self.config),
        })

        self._previous_D = self._D_from_observation(observation)
        self._previous_distance = distance_next
        self._episode_done = bool(terminated or truncated)
        return observation, breakdown.total, terminated, truncated, info


__all__ = ["RewardV2Wrapper"]
