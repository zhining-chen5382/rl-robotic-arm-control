"""Headroom-aware residual-action wrapper.

This module deliberately reuses the existing residual wrapper's reset and
observation path.  Only normalized-action composition is changed.  The old
direct-add wrapper remains untouched for controlled ablation.
"""

from __future__ import annotations

import numpy as np

from .frozen_base_policy import ACTION_DIM
from .rl_env_6d_residual_obstacle import (
    ResidualObstacleEnv,
    validate_residual_scale,
)


def compose_headroom_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    residual_scale: float = 0.05,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compose a residual using the available room in its requested direction.

    Returns ``raw_residual``, ``clipped_residual``,
    ``nominal_scaled_residual``, ``headroom``, ``effective_residual``,
    ``preclip_final_action`` and ``final_action``.

    ``nominal_scaled_residual`` is used by the unchanged residual penalty.
    ``effective_residual`` is the correction actually added to ``base_action``.
    """
    scale = validate_residual_scale(residual_scale)
    base = np.asarray(base_action, dtype=np.float32)
    raw_residual = np.asarray(residual_action, dtype=np.float32)
    if base.shape != (ACTION_DIM,):
        raise ValueError(
            f"base_action shape must be ({ACTION_DIM},), got {base.shape}"
        )
    if raw_residual.shape != (ACTION_DIM,):
        raise ValueError(
            f"residual_action shape must be ({ACTION_DIM},), "
            f"got {raw_residual.shape}"
        )
    if not np.all(np.isfinite(base)):
        raise ValueError("base_action contains NaN or Inf")
    if not np.all(np.isfinite(raw_residual)):
        raise ValueError("residual_action contains NaN or Inf")
    if np.any(base < -1.0) or np.any(base > 1.0):
        raise ValueError("base_action must be inside [-1, 1]")

    clipped_residual = np.clip(raw_residual, -1.0, 1.0).astype(np.float32)
    nominal_scaled_residual = (scale * clipped_residual).astype(np.float32)
    headroom = np.where(
        clipped_residual >= 0.0,
        np.float32(1.0) - base,
        np.float32(1.0) + base,
    ).astype(np.float32)
    effective_residual = (nominal_scaled_residual * headroom).astype(np.float32)
    preclip_final_action = (base + effective_residual).astype(np.float32)
    final_action = np.clip(preclip_final_action, -1.0, 1.0).astype(np.float32)
    return (
        raw_residual.copy(),
        clipped_residual,
        nominal_scaled_residual,
        headroom,
        effective_residual,
        preclip_final_action,
        final_action,
    )


class HeadroomResidualObstacleEnv(ResidualObstacleEnv):
    """Residual obstacle wrapper with headroom-aware action composition."""

    def step(self, residual_action):
        if self._current_base_action is None:
            raise RuntimeError("reset() must be called before step()")

        base_action = self._current_base_action.copy()
        (
            raw_residual,
            clipped_residual,
            nominal_scaled_residual,
            headroom,
            effective_residual,
            preclip_final_action,
            final_action,
        ) = compose_headroom_residual_action(
            base_action,
            residual_action,
            self.residual_scale,
        )

        # This is the sole action passed to the original obstacle environment.
        # Its adaptive action scale is therefore still applied exactly once.
        observation, env_reward, terminated, truncated, info = self.env.step(
            final_action
        )

        # Keep the old reward definition: penalty is based on the nominal
        # correction, not the headroom-mapped effective correction.
        residual_penalty = -self.residual_penalty_weight * float(
            np.mean(np.square(nominal_scaled_residual, dtype=np.float32))
        )
        reward = float(env_reward) + residual_penalty

        info = dict(info)
        info.update(
            {
                "env_reward": float(env_reward),
                "raw_residual_action": raw_residual.copy(),
                "clipped_residual_action": clipped_residual.copy(),
                "nominal_scaled_residual": nominal_scaled_residual.copy(),
                "headroom": headroom.copy(),
                "effective_residual": effective_residual.copy(),
                "base_action": base_action.copy(),
                "preclip_final_action": preclip_final_action.copy(),
                "final_action": final_action.copy(),
                "residual_scale": self.residual_scale,
                "residual_penalty": residual_penalty,
            }
        )
        next_observation = self._build_residual_observation(observation)
        return next_observation, reward, terminated, truncated, info


__all__ = [
    "HeadroomResidualObstacleEnv",
    "compose_headroom_residual_action",
]
