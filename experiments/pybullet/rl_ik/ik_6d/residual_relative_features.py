"""Feature extractor for explicit robot-obstacle relation clouds."""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .residual_pointnet_features import ResidualPointNetEncoder
from .rl_env_6d_residual_geometry import ROBOT_KEYPOINT_COUNT, ROBOT_KEYPOINT_DIM
from .rl_env_6d_residual_relative import RELATION_CHANNELS


OBSTACLE_FEATURE_DIM = 128
GEOMETRY_FEATURE_DIM = 64
RELATION_PER_KEYPOINT_DIM = 64
RELATION_FEATURE_DIM = 128


class RobotObstacleRelationEncoder(nn.Module):
    """Shared point encoder with per-keypoint pooling and preserved identities."""

    def __init__(self):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(RELATION_CHANNELS, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, RELATION_PER_KEYPOINT_DIM, kernel_size=1),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(ROBOT_KEYPOINT_COUNT * RELATION_PER_KEYPOINT_DIM, 128),
            nn.ReLU(),
        )

    def forward(self, relation_cloud: torch.Tensor) -> torch.Tensor:
        if relation_cloud.ndim != 4 or relation_cloud.shape[1:] != (
            ROBOT_KEYPOINT_COUNT,
            512,
            RELATION_CHANNELS,
        ):
            raise ValueError(
                "relation_cloud tensor must have shape (B, 6, 512, 4), "
                f"got {tuple(relation_cloud.shape)}"
            )
        batch = relation_cloud.shape[0]
        per_keypoint = relation_cloud.reshape(
            batch * ROBOT_KEYPOINT_COUNT, 512, RELATION_CHANNELS
        ).transpose(1, 2)
        point_features = self.point_mlp(per_keypoint)
        # Pool obstacle points independently for every ordered robot keypoint.
        pooled = torch.max(point_features, dim=2).values
        ordered = pooled.reshape(
            batch, ROBOT_KEYPOINT_COUNT, RELATION_PER_KEYPOINT_DIM
        )
        # Do not pool over K: flatten preserves link-slot identity.
        return self.fusion(ordered.flatten(start_dim=1))


class ResidualRelativeFeatureExtractor(BaseFeaturesExtractor):
    """Fuse state, unchanged PointNet/geometry branches, and relation feature."""

    def __init__(self, observation_space):
        state_dim = int(observation_space["state"].shape[0])
        base_action_dim = int(observation_space["base_action"].shape[0])
        if observation_space["point_cloud"].shape != (512, 3):
            raise ValueError("point_cloud observation must have shape (512, 3)")
        if observation_space["robot_keypoints"].shape != (6, 3):
            raise ValueError("robot_keypoints observation must have shape (6, 3)")
        if observation_space["relation_cloud"].shape != (6, 512, 4):
            raise ValueError("relation_cloud observation must have shape (6, 512, 4)")
        features_dim = (
            state_dim + OBSTACLE_FEATURE_DIM + base_action_dim
            + GEOMETRY_FEATURE_DIM + RELATION_FEATURE_DIM
        )
        super().__init__(observation_space, features_dim=features_dim)
        self.pointnet = ResidualPointNetEncoder(output_dim=OBSTACLE_FEATURE_DIM)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(ROBOT_KEYPOINT_COUNT * ROBOT_KEYPOINT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, GEOMETRY_FEATURE_DIM),
            nn.ReLU(),
        )
        self.relation_encoder = RobotObstacleRelationEncoder()

    def forward(self, observations) -> torch.Tensor:
        state = observations["state"].float()
        point_feature = self.pointnet(observations["point_cloud"].float())
        base_action = observations["base_action"].float()
        keypoints = observations["robot_keypoints"].float()
        if keypoints.ndim != 3 or keypoints.shape[1:] != (6, 3):
            raise ValueError("robot_keypoints tensor must have shape (B, 6, 3)")
        geometry_feature = self.geometry_mlp(keypoints.flatten(start_dim=1))
        relation_feature = self.relation_encoder(
            observations["relation_cloud"].float()
        )
        return torch.cat(
            (state, point_feature, base_action, geometry_feature, relation_feature),
            dim=1,
        )


__all__ = [
    "RELATION_FEATURE_DIM",
    "ResidualRelativeFeatureExtractor",
    "RobotObstacleRelationEncoder",
]
