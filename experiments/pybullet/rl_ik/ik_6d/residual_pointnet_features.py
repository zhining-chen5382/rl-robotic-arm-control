"""第一版 residual PPO 的非特权 PointNet 特征提取器。"""

from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualPointNetEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim 必须大于 0")
        self.output_dim = int(output_dim)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, self.output_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(
                "point_cloud tensor shape 必须为 (B, N, 3)，"
                f"实际为 {tuple(point_cloud.shape)}"
            )
        point_features = self.point_mlp(point_cloud.transpose(1, 2))
        return torch.max(point_features, dim=2).values


class ResidualPointNetFeatureExtractor(BaseFeaturesExtractor):
    """仅融合 state、point_cloud 和 base_action。"""

    def __init__(self, observation_space, point_feature_dim: int = 128):
        state_dim = int(observation_space["state"].shape[0])
        base_action_dim = int(observation_space["base_action"].shape[0])
        point_shape = observation_space["point_cloud"].shape
        if point_shape != (512, 3):
            raise ValueError(f"point_cloud observation 应为 (512, 3)，实际为 {point_shape}")

        features_dim = state_dim + int(point_feature_dim) + base_action_dim
        super().__init__(observation_space, features_dim=features_dim)
        self.pointnet = ResidualPointNetEncoder(output_dim=point_feature_dim)

    def forward(self, observations) -> torch.Tensor:
        state = observations["state"].float()
        point_cloud = observations["point_cloud"].float()
        base_action = observations["base_action"].float()
        point_feature = self.pointnet(point_cloud)
        return torch.cat((state, point_feature, base_action), dim=1)


__all__ = ["ResidualPointNetEncoder", "ResidualPointNetFeatureExtractor"]
