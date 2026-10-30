"""Feature extractor for residual observations with ordered Panda keypoints."""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .residual_pointnet_features import ResidualPointNetEncoder
from .rl_env_6d_residual_geometry import (
    ROBOT_KEYPOINT_COUNT,
    ROBOT_KEYPOINT_DIM,
)


OBSTACLE_FEATURE_DIM = 128
GEOMETRY_FEATURE_DIM = 64


class ResidualGeometryFeatureExtractor(BaseFeaturesExtractor):
    """Fuse state, unchanged PointNet, base action, and ordered geometry MLP."""

    def __init__(self, observation_space):
        state_dim = int(observation_space["state"].shape[0])
        base_action_dim = int(observation_space["base_action"].shape[0])
        if observation_space["point_cloud"].shape != (512, 3):
            raise ValueError("point_cloud observation must have shape (512, 3)")
        if observation_space["robot_keypoints"].shape != (
            ROBOT_KEYPOINT_COUNT,
            ROBOT_KEYPOINT_DIM,
        ):
            raise ValueError("robot_keypoints observation must have shape (6, 3)")

        features_dim = (
            state_dim
            + OBSTACLE_FEATURE_DIM
            + base_action_dim
            + GEOMETRY_FEATURE_DIM
        )
        super().__init__(observation_space, features_dim=features_dim)
        self.pointnet = ResidualPointNetEncoder(output_dim=OBSTACLE_FEATURE_DIM)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(ROBOT_KEYPOINT_COUNT * ROBOT_KEYPOINT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, GEOMETRY_FEATURE_DIM),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        state = observations["state"].float()
        point_cloud = observations["point_cloud"].float()
        base_action = observations["base_action"].float()
        keypoints = observations["robot_keypoints"].float()
        if keypoints.ndim != 3 or keypoints.shape[1:] != (
            ROBOT_KEYPOINT_COUNT,
            ROBOT_KEYPOINT_DIM,
        ):
            raise ValueError(
                "robot_keypoints tensor must have shape (B, 6, 3), "
                f"got {tuple(keypoints.shape)}"
            )
        point_feature = self.pointnet(point_cloud)
        geometry_feature = self.geometry_mlp(keypoints.flatten(start_dim=1))
        return torch.cat(
            (state, point_feature, base_action, geometry_feature),
            dim=1,
        )


__all__ = [
    "GEOMETRY_FEATURE_DIM",
    "OBSTACLE_FEATURE_DIM",
    "ResidualGeometryFeatureExtractor",
]
