"""基于现有障碍环境的第一版 residual-action Gymnasium Wrapper。"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .frozen_base_policy import (
    ACTION_DIM,
    DEFAULT_BASE_MODEL_PATH,
    STATE_DIM,
    FrozenBasePolicy,
)
from .rl_env_6d_obstacle import RobotIK6DEnv as ObstacleIKEnv


POINT_COUNT = 512
POINT_DIM = 3


def validate_residual_scale(residual_scale: float) -> float:
    value = float(residual_scale)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            "residual_scale 必须是 [0, 1] 内的有限数值，"
            f"实际为 {residual_scale!r}"
        )
    return value


def validate_residual_penalty_weight(weight: float) -> float:
    value = float(weight)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "residual_penalty_weight 必须是非负有限数值，"
            f"实际为 {weight!r}"
        )
    return value


def compose_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    residual_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在 normalized action space 中组合 base 和 residual action。

    Returns:
        raw_residual, clipped_residual, scaled_residual, final_action
    """
    scale = validate_residual_scale(residual_scale)
    base = np.asarray(base_action, dtype=np.float32)
    raw_residual = np.asarray(residual_action, dtype=np.float32)
    if base.shape != (ACTION_DIM,):
        raise ValueError(f"base_action shape 应为 ({ACTION_DIM},)，实际为 {base.shape}")
    if raw_residual.shape != (ACTION_DIM,):
        raise ValueError(
            f"residual_action shape 应为 ({ACTION_DIM},)，实际为 {raw_residual.shape}"
        )
    if not np.all(np.isfinite(base)):
        raise ValueError("base_action 包含 NaN 或 Inf")
    if not np.all(np.isfinite(raw_residual)):
        raise ValueError("residual_action 包含 NaN 或 Inf")
    if np.any(base < -1.0) or np.any(base > 1.0):
        raise ValueError("base_action 必须位于 [-1, 1]")

    clipped_residual = np.clip(raw_residual, -1.0, 1.0).astype(np.float32)
    scaled_residual = (scale * clipped_residual).astype(np.float32)
    final_action = np.clip(base + scaled_residual, -1.0, 1.0).astype(np.float32)
    return (
        raw_residual.copy(),
        clipped_residual,
        scaled_residual,
        final_action,
    )


class ResidualObstacleEnv(gym.Wrapper):
    """把 residual action 与冻结 PPO 的 base action 组合后交给原环境。"""

    def __init__(
        self,
        env: gym.Env | None = None,
        *,
        base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
        residual_scale: float = 0.05,
        residual_penalty_weight: float = 1e-4,
    ):
        if env is None:
            env = ObstacleIKEnv()
        super().__init__(env)

        self.residual_scale = validate_residual_scale(residual_scale)
        self.residual_penalty_weight = validate_residual_penalty_weight(
            residual_penalty_weight
        )
        self.frozen_base_policy = FrozenBasePolicy(base_model_path)
        self._current_base_action: np.ndarray | None = None

        original_space = self.env.observation_space
        if not isinstance(original_space, spaces.Dict):
            raise TypeError("底层障碍环境 observation_space 必须是 spaces.Dict")
        if original_space["state"].shape != (STATE_DIM,):
            raise ValueError("底层障碍环境 state 不是 17 维")
        if original_space["point_cloud"].shape != (POINT_COUNT, POINT_DIM):
            raise ValueError("底层障碍环境 point_cloud 不是 (512, 3)")

        self.observation_space = spaces.Dict(
            {
                "state": original_space["state"],
                "point_cloud": original_space["point_cloud"],
                "base_action": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(ACTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    @staticmethod
    def _validate_original_observation(observation: dict) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(observation, dict):
            raise TypeError("底层障碍环境必须返回 Dict observation")
        if "state" not in observation or "point_cloud" not in observation:
            raise KeyError("底层 observation 缺少 state 或 point_cloud")
        state = np.asarray(observation["state"], dtype=np.float32)
        point_cloud = np.asarray(observation["point_cloud"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state shape 应为 ({STATE_DIM},)，实际为 {state.shape}")
        if point_cloud.shape != (POINT_COUNT, POINT_DIM):
            raise ValueError(
                "point_cloud shape 应为 "
                f"({POINT_COUNT}, {POINT_DIM})，实际为 {point_cloud.shape}"
            )
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(point_cloud)):
            raise ValueError("observation 包含 NaN 或 Inf")
        return state, point_cloud

    def _build_residual_observation(self, observation: dict) -> dict[str, np.ndarray]:
        state, point_cloud = self._validate_original_observation(observation)
        base_action = self.frozen_base_policy.predict(state)
        self._current_base_action = base_action.copy()
        residual_observation = {
            "state": state.copy(),
            "point_cloud": point_cloud.copy(),
            "base_action": base_action.copy(),
        }
        if not self.observation_space.contains(residual_observation):
            raise RuntimeError("构造出的 residual observation 不符合 observation_space")
        return residual_observation

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._build_residual_observation(observation), info

    def step(self, residual_action):
        if self._current_base_action is None:
            raise RuntimeError("调用 step() 前必须先调用 reset()")

        base_action = self._current_base_action.copy()
        (
            raw_residual,
            clipped_residual,
            scaled_residual,
            final_action,
        ) = compose_residual_action(
            base_action,
            residual_action,
            self.residual_scale,
        )

        # final_action 是唯一传入原障碍环境的动作；adaptive scale 仅由原环境执行。
        observation, env_reward, terminated, truncated, info = self.env.step(
            final_action
        )
        residual_penalty = -self.residual_penalty_weight * float(
            np.mean(np.square(scaled_residual, dtype=np.float32))
        )
        reward = float(env_reward) + residual_penalty

        info = dict(info)
        info.update(
            {
                "env_reward": float(env_reward),
                "raw_residual_action": raw_residual.copy(),
                "clipped_residual_action": clipped_residual.copy(),
                "scaled_residual": scaled_residual.copy(),
                "base_action": base_action.copy(),
                "final_action": final_action.copy(),
                "residual_scale": self.residual_scale,
                "residual_penalty": residual_penalty,
            }
        )
        next_observation = self._build_residual_observation(observation)
        return next_observation, reward, terminated, truncated, info


__all__ = [
    "ResidualObstacleEnv",
    "compose_residual_action",
    "validate_residual_scale",
]
