"""Pure Reward V2-Balanced mathematics.

This module intentionally has no Gym, PyBullet, Torch, or SB3 dependency.  The
full-precision constants are copied from the locked offline calibration
artifacts so online and offline reward calculations can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TerminalType = Literal["none", "success", "collision", "timeout"]
VALID_TERMINAL_TYPES = frozenset({"none", "success", "collision", "timeout"})


@dataclass(frozen=True)
class RewardV2Config:
    """Locked V2-Balanced parameters from offline calibration."""

    ik_progress_scale: float = 0.3818538138426255
    beta: float = 0.25
    w_ik: float = 0.25
    d_safe: float = 0.15
    safety_scale: float = 0.04055719071763819
    w_safe: float = 0.25
    w_zone: float = 0.025
    success_terminal_bonus: float = 1.0
    timeout_terminal_penalty: float = -1.0
    collision_terminal_penalty: float = -2.0
    joint_limit_scale: float = 0.001

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Reward V2 config values must all be finite")
        if self.ik_progress_scale <= 0.0:
            raise ValueError("ik_progress_scale must be positive")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must satisfy 0 < beta <= 1")
        if self.d_safe <= 0.0 or self.safety_scale <= 0.0:
            raise ValueError("danger and safety scales must be positive")
        if min(self.w_ik, self.w_safe, self.w_zone) < 0.0:
            raise ValueError("Reward V2 weights must be non-negative")
        if self.success_terminal_bonus < 0.0:
            raise ValueError("success_terminal_bonus must be non-negative")
        if self.timeout_terminal_penalty > 0.0:
            raise ValueError("timeout_terminal_penalty must be non-positive")
        if self.collision_terminal_penalty > 0.0:
            raise ValueError("collision_terminal_penalty must be non-positive")
        if self.joint_limit_scale <= 0.0:
            raise ValueError("joint_limit_scale must be positive")


V2_BALANCED_CONFIG = RewardV2Config()


@dataclass(frozen=True)
class RewardV2Breakdown:
    total: float
    ik_progress: float
    safety_progress: float
    zone: float
    joint_limit: float
    residual: float
    terminal: float
    terminal_type: TerminalType
    D_current: float
    D_next: float
    delta_D: float
    normalized_ik_progress: float
    distance_current: float
    distance_next: float
    danger_current: float
    danger_next: float
    raw_safety_progress: float
    normalized_safety_progress: float


def _finite_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def ik_distance(position_error: float, orientation_error: float) -> float:
    """Return the existing MAPPO-IK combined distance metric."""
    position = _finite_float("position_error", position_error)
    orientation = _finite_float("orientation_error", orientation_error)
    cosine_distance = 2.0 - np.cos(orientation)
    return float(position * cosine_distance + (cosine_distance - 1.0))


def danger(distance: float, config: RewardV2Config = V2_BALANCED_CONFIG) -> float:
    """Squared 150 mm danger occupancy used by the offline calibration."""
    clearance = _finite_float("distance", distance)
    normalized = np.clip(
        (config.d_safe - clearance) / config.d_safe,
        0.0,
        1.0,
    )
    return float(normalized * normalized)


def joint_limit_component(
    joints: np.ndarray,
    joint_limits: list[tuple[float, float]] | np.ndarray,
    config: RewardV2Config = V2_BALANCED_CONFIG,
    *,
    tolerance: float = 1e-6,
) -> float:
    """Reproduce the old environment's joint penalty at PPO reward scale."""
    joint_array = np.asarray(joints, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if joint_array.ndim != 1 or limits.shape != (joint_array.size, 2):
        raise ValueError("joint_limits must have shape (number_of_joints, 2)")
    if not np.all(np.isfinite(joint_array)) or not np.all(np.isfinite(limits)):
        raise ValueError("joint values and limits must be finite")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative value")
    at_limit = (joint_array <= limits[:, 0] + tolerance) | (
        joint_array >= limits[:, 1] - tolerance
    )
    return -float(np.count_nonzero(at_limit)) * config.joint_limit_scale


def terminal_component(
    terminal_type: TerminalType,
    config: RewardV2Config = V2_BALANCED_CONFIG,
) -> float:
    if terminal_type not in VALID_TERMINAL_TYPES:
        raise ValueError(f"unsupported terminal_type: {terminal_type!r}")
    if terminal_type == "success":
        return config.success_terminal_bonus
    if terminal_type == "collision":
        return config.collision_terminal_penalty
    if terminal_type == "timeout":
        return config.timeout_terminal_penalty
    return 0.0


def compute_reward_v2(
    *,
    D_current: float,
    D_next: float,
    distance_current: float,
    distance_next: float,
    joint_limit: float = 0.0,
    residual: float = 0.0,
    terminal_type: TerminalType = "none",
    config: RewardV2Config = V2_BALANCED_CONFIG,
) -> RewardV2Breakdown:
    """Compute one V2-Balanced transition and its complete decomposition."""
    current_D = _finite_float("D_current", D_current)
    next_D = _finite_float("D_next", D_next)
    current_distance = _finite_float("distance_current", distance_current)
    next_distance = _finite_float("distance_next", distance_next)
    joint_reward = _finite_float("joint_limit", joint_limit)
    residual_reward = _finite_float("residual", residual)

    delta_D = current_D - next_D
    normalized_ik = float(np.clip(
        delta_D / config.ik_progress_scale,
        -1.0,
        1.0,
    ))
    asymmetric_ik = normalized_ik if normalized_ik >= 0.0 else config.beta * normalized_ik
    ik_reward = config.w_ik * asymmetric_ik

    danger_current = danger(current_distance, config)
    danger_next = danger(next_distance, config)
    raw_safety = danger_current - danger_next
    normalized_safety = float(np.clip(
        raw_safety / config.safety_scale,
        -1.0,
        1.0,
    ))
    safety_reward = config.w_safe * normalized_safety
    zone_reward = -config.w_zone * danger_next
    terminal_reward = terminal_component(terminal_type, config)

    total = (
        ik_reward
        + safety_reward
        + zone_reward
        + joint_reward
        + residual_reward
        + terminal_reward
    )
    return RewardV2Breakdown(
        total=float(total),
        ik_progress=float(ik_reward),
        safety_progress=float(safety_reward),
        zone=float(zone_reward),
        joint_limit=joint_reward,
        residual=residual_reward,
        terminal=float(terminal_reward),
        terminal_type=terminal_type,
        D_current=current_D,
        D_next=next_D,
        delta_D=float(delta_D),
        normalized_ik_progress=normalized_ik,
        distance_current=current_distance,
        distance_next=next_distance,
        danger_current=danger_current,
        danger_next=danger_next,
        raw_safety_progress=float(raw_safety),
        normalized_safety_progress=normalized_safety,
    )


__all__ = [
    "RewardV2Breakdown",
    "RewardV2Config",
    "TerminalType",
    "V2_BALANCED_CONFIG",
    "compute_reward_v2",
    "danger",
    "ik_distance",
    "joint_limit_component",
    "terminal_component",
]
