import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),

            nn.Conv1d(128, output_dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, point_cloud):
        """
        point_cloud: (batch_size, num_points, 3)

        return:
            global_feature: (batch_size, 512)
        """

        # (B, N, 3) -> (B, 3, N)
        x = point_cloud.transpose(1, 2)

        # 每个点分别提取特征
        x = self.mlp(x)

        # (B, 512, N) -> (B, 512)
        x = torch.max(x, dim=2)[0]

        return x