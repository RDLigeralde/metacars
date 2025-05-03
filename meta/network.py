from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

import torch.nn as nn
import torch

class LIDARConvExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int=256
    ):
        """
        Initial embedding layer that uses CNN on LIDAR, MLP on telemetry

        Args:
            observation_space (gym.spaces.Dict): dict keyed by 'scan', 'odometry'
            features_dim (int, optional): total embedding dim (default 256)
        """
        super().__init__(observation_space, features_dim)

        # assuming 1 X num_beams (projected down from n_agents X num_beams)
        num_beams = observation_space['scan'].shape[1]
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        lidar_embed_dim = self._lidar_embed_dim(num_beams)

        telem_dim = observation_space['odometry'].shape[1]
        self.telemetry_mlp = nn.Sequential(
            nn.Linear(telem_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.embed_layer = nn.Sequential(
            nn.Linear(lidar_embed_dim + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        scan, odom = observations['scan'], observations['odometry']
        state_embed = self.telemetry_mlp(odom).squeeze(1)
        lidar_embed = self.lidar_cnn(scan)
        cat_embed = torch.cat((state_embed, lidar_embed), dim=1)
        return self.embed_layer(cat_embed)

    @torch.inference_mode
    def _lidar_embed_dim(self, num_beams: int) -> int:
        dummy_scan = torch.zeros(1, 1, num_beams) # B X C X N
        return self.lidar_cnn(dummy_scan).shape[1]

