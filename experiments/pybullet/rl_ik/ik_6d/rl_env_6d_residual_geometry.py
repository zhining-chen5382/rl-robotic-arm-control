"""Headroom residual wrapper with optional explicit Panda FK keypoints."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from .frozen_base_policy import ACTION_DIM, STATE_DIM
from .rl_env_6d_residual_headroom import HeadroomResidualObstacleEnv
from .rl_env_6d_residual_obstacle import POINT_COUNT, POINT_DIM


ROBOT_KEYPOINT_LINK_INDICES = (0, 2, 3, 4, 6, 8)
ROBOT_KEYPOINT_LINK_NAMES = (
    "panda_link1",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link7",
    "panda_hand",
)
ROBOT_KEYPOINT_COUNT = len(ROBOT_KEYPOINT_LINK_INDICES)
ROBOT_KEYPOINT_DIM = 3
GeometryMode = Literal["zero", "actual"]


def validate_geometry_mode(mode: str) -> GeometryMode:
    if mode not in {"zero", "actual"}:
        raise ValueError(
            "geometry_mode must be either 'zero' or 'actual', "
            f"got {mode!r}"
        )
    return mode


def _base_link_world_transform(robot_id: int):
    """Return the URDF base-link frame in world coordinates.

    PyBullet's base pose is the base center-of-mass/inertial frame.  Removing
    the local inertial transform recovers the actual ``panda_link0`` frame.
    """
    world_base_com_position, world_base_com_orientation = (
        p.getBasePositionAndOrientation(robot_id)
    )
    dynamics = p.getDynamicsInfo(robot_id, -1)
    base_to_base_com_position = dynamics[3]
    base_to_base_com_orientation = dynamics[4]
    base_com_to_base = p.invertTransform(
        base_to_base_com_position,
        base_to_base_com_orientation,
    )
    return p.multiplyTransforms(
        world_base_com_position,
        world_base_com_orientation,
        base_com_to_base[0],
        base_com_to_base[1],
    )


def compute_robot_keypoints_base_frame(robot_id: int) -> np.ndarray:
    """Compute six ordered Panda link-frame positions in ``panda_link0`` frame.

    This function only reads robot FK state and robot base/inertial transforms.
    It does not read obstacles, contacts, clearance, rewards, or future state.
    """
    world_base_position, world_base_orientation = _base_link_world_transform(
        robot_id
    )
    base_world_position, base_world_orientation = p.invertTransform(
        world_base_position,
        world_base_orientation,
    )

    keypoints = np.empty(
        (ROBOT_KEYPOINT_COUNT, ROBOT_KEYPOINT_DIM),
        dtype=np.float32,
    )
    for output_index, link_index in enumerate(ROBOT_KEYPOINT_LINK_INDICES):
        link_state = p.getLinkState(robot_id, link_index)
        world_link_frame_position = link_state[4]  # never use COM position [0]
        base_link_position, _ = p.multiplyTransforms(
            base_world_position,
            base_world_orientation,
            world_link_frame_position,
            (0.0, 0.0, 0.0, 1.0),
        )
        keypoints[output_index] = np.asarray(
            base_link_position,
            dtype=np.float32,
        )

    if not np.all(np.isfinite(keypoints)):
        raise RuntimeError("computed robot_keypoints contain NaN or Inf")
    return keypoints


class GeometryResidualObstacleEnv(HeadroomResidualObstacleEnv):
    """Add zero or actual FK keypoints to the verified Headroom observation."""

    def __init__(self, *args, geometry_mode: str, **kwargs):
        self.geometry_mode = validate_geometry_mode(geometry_mode)
        super().__init__(*args, **kwargs)

        original_space = self.env.observation_space
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
                # Infinite bounds avoid imposing an incorrect workspace crop.
                "robot_keypoints": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ROBOT_KEYPOINT_COUNT, ROBOT_KEYPOINT_DIM),
                    dtype=np.float32,
                ),
            }
        )

    def _compute_robot_keypoints(self) -> np.ndarray:
        if self.geometry_mode == "zero":
            return np.zeros(
                (ROBOT_KEYPOINT_COUNT, ROBOT_KEYPOINT_DIM),
                dtype=np.float32,
            )
        robot_id = getattr(self.env.unwrapped, "robot", None)
        if robot_id is None:
            raise RuntimeError("Panda robot is not initialized")
        return compute_robot_keypoints_base_frame(int(robot_id))

    def _build_residual_observation(self, observation: dict) -> dict[str, np.ndarray]:
        state, point_cloud = self._validate_original_observation(observation)
        # FrozenBasePolicy still receives exactly and only the 17-dimensional state.
        base_action = self.frozen_base_policy.predict(state)
        self._current_base_action = base_action.copy()
        residual_observation = {
            "state": state.copy(),
            "point_cloud": point_cloud.copy(),
            "base_action": base_action.copy(),
            "robot_keypoints": self._compute_robot_keypoints(),
        }
        if not self.observation_space.contains(residual_observation):
            raise RuntimeError(
                "constructed geometry residual observation is outside observation_space"
            )
        return residual_observation


__all__ = [
    "GeometryResidualObstacleEnv",
    "ROBOT_KEYPOINT_COUNT",
    "ROBOT_KEYPOINT_DIM",
    "ROBOT_KEYPOINT_LINK_INDICES",
    "ROBOT_KEYPOINT_LINK_NAMES",
    "compute_robot_keypoints_base_frame",
    "validate_geometry_mode",
]
