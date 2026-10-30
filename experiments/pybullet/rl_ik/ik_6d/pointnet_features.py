import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ik_6d.pointnet_encoder import PointNetEncoder


class PointNetFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, point_feature_dim=512):

        state_dim = observation_space["state"].shape[0]

        # 最终输出 = 17维 state + 512维 PointNet feature
        features_dim = state_dim + point_feature_dim

        super().__init__(
            observation_space,
            features_dim=features_dim
        )

        self.pointnet = PointNetEncoder(
            output_dim=point_feature_dim
        )

    def forward(self, observations):

        state = observations["state"].float()
        point_cloud = observations["point_cloud"].float()

        # (B, 512, 3)
        #       ↓ PointNet
        # (B, 512)
        point_feature = self.pointnet(point_cloud)

        # (B, 17) + (B, 512)
        #          ↓
        #       (B, 529)
        combined_feature = torch.cat(
            [state, point_feature],
            dim=1
        )

        return combined_feature