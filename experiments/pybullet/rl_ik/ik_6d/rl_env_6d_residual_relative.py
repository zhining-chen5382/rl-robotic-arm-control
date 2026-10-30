"""Residual observation with explicit robot-obstacle relation clouds."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pybullet as p
from gymnasium import spaces

from .rl_env_6d_residual_geometry import (
    GeometryResidualObstacleEnv,
    ROBOT_KEYPOINT_COUNT,
    ROBOT_KEYPOINT_DIM,
    _base_link_world_transform,
)
from .rl_env_6d_residual_obstacle import POINT_COUNT, POINT_DIM


RELATION_CHANNELS = 4
RelationMode = Literal["uncentered", "robot_centered"]


def validate_relation_mode(mode: str) -> RelationMode:
    if mode not in {"uncentered", "robot_centered"}:
        raise ValueError(
            "relation_mode must be 'uncentered' or 'robot_centered', "
            f"got {mode!r}"
        )
    return mode


def transform_obstacle_points_to_base_frame(
    robot_id: int,
    obstacle_points_world: np.ndarray,
) -> np.ndarray:
    """Transform world-frame obstacle coordinates into Panda base-link frame."""
    points_world = np.asarray(obstacle_points_world, dtype=np.float32)
    if points_world.shape != (POINT_COUNT, POINT_DIM):
        raise ValueError(
            f"obstacle points must have shape ({POINT_COUNT}, {POINT_DIM}), "
            f"got {points_world.shape}"
        )
    if not np.all(np.isfinite(points_world)):
        raise ValueError("obstacle points contain NaN or Inf")

    world_base_position, world_base_orientation = _base_link_world_transform(
        robot_id
    )
    base_world_position, base_world_orientation = p.invertTransform(
        world_base_position,
        world_base_orientation,
    )
    # Use the already verified PyBullet transform operation for every point.
    # The Windows PyBullet build used by this project raises a native exception
    # in getMatrixFromQuaternion(), so that API is deliberately avoided.
    points_base = np.empty_like(points_world, dtype=np.float32)
    identity = (0.0, 0.0, 0.0, 1.0)
    for index, point_world in enumerate(points_world):
        point_base, _ = p.multiplyTransforms(
            base_world_position,
            base_world_orientation,
            point_world,
            identity,
        )
        points_base[index] = np.asarray(point_base, dtype=np.float32)
    if not np.all(np.isfinite(points_base)):
        raise RuntimeError("base-frame obstacle points contain NaN or Inf")
    return points_base


def compute_relation_cloud(
    obstacle_points_base: np.ndarray,
    robot_keypoints: np.ndarray,
    *,
    relation_mode: str,
) -> np.ndarray:
    """Construct ordered ``[dx, dy, dz, Euclidean distance]`` relation data."""
    mode = validate_relation_mode(relation_mode)
    points = np.asarray(obstacle_points_base, dtype=np.float32)
    keypoints = np.asarray(robot_keypoints, dtype=np.float32)
    if points.shape != (POINT_COUNT, POINT_DIM):
        raise ValueError("obstacle_points_base must have shape (512, 3)")
    if keypoints.shape != (ROBOT_KEYPOINT_COUNT, ROBOT_KEYPOINT_DIM):
        raise ValueError("robot_keypoints must have shape (6, 3)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(keypoints)):
        raise ValueError("relation inputs contain NaN or Inf")

    if mode == "robot_centered":
        vectors = points[None, :, :] - keypoints[:, None, :]
    else:
        vectors = np.broadcast_to(
            points[None, :, :],
            (ROBOT_KEYPOINT_COUNT, POINT_COUNT, POINT_DIM),
        ).copy()
    vectors = np.asarray(vectors, dtype=np.float32)
    distances = np.linalg.norm(vectors, axis=2, keepdims=True).astype(np.float32)
    relation = np.concatenate((vectors, distances), axis=2).astype(np.float32)
    if relation.shape != (ROBOT_KEYPOINT_COUNT, POINT_COUNT, RELATION_CHANNELS):
        raise RuntimeError(f"unexpected relation shape: {relation.shape}")
    if not np.all(np.isfinite(relation)):
        raise RuntimeError("relation_cloud contains NaN or Inf")
    return relation


class RelativeResidualObstacleEnv(GeometryResidualObstacleEnv):
    """Actual K=6 FK geometry plus uncentered or robot-centered relations."""

    def __init__(self, *args, relation_mode: str, **kwargs):
        self.relation_mode = validate_relation_mode(relation_mode)
        # Both arms use actual FK geometry; relation values are the treatment.
        if "geometry_mode" in kwargs:
            raise TypeError("RelativeResidualObstacleEnv fixes geometry_mode='actual'")
        super().__init__(*args, geometry_mode="actual", **kwargs)
        geometry_space = self.observation_space
        self.observation_space = spaces.Dict(
            {
                "state": geometry_space["state"],
                "point_cloud": geometry_space["point_cloud"],
                "base_action": geometry_space["base_action"],
                "robot_keypoints": geometry_space["robot_keypoints"],
                "relation_cloud": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ROBOT_KEYPOINT_COUNT, POINT_COUNT, RELATION_CHANNELS),
                    dtype=np.float32,
                ),
            }
        )

    def _build_residual_observation(self, observation: dict) -> dict[str, np.ndarray]:
        state, point_cloud = self._validate_original_observation(observation)
        # Frozen PPO still receives exactly and only state shape (17,).
        base_action = self.frozen_base_policy.predict(state)
        self._current_base_action = base_action.copy()
        robot_keypoints = self._compute_robot_keypoints()
        robot_id = getattr(self.env.unwrapped, "robot", None)
        if robot_id is None:
            raise RuntimeError("Panda robot is not initialized")
        obstacle_points_base = transform_obstacle_points_to_base_frame(
            int(robot_id), point_cloud
        )
        relation_cloud = compute_relation_cloud(
            obstacle_points_base,
            robot_keypoints,
            relation_mode=self.relation_mode,
        )
        residual_observation = {
            "state": state.copy(),
            "point_cloud": point_cloud.copy(),
            "base_action": base_action.copy(),
            "robot_keypoints": robot_keypoints.copy(),
            "relation_cloud": relation_cloud,
        }
        if not self.observation_space.contains(residual_observation):
            raise RuntimeError("relative residual observation is outside observation_space")
        return residual_observation


__all__ = [
    "RELATION_CHANNELS",
    "RelativeResidualObstacleEnv",
    "compute_relation_cloud",
    "transform_obstacle_points_to_base_frame",
    "validate_relation_mode",
]
