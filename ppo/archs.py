from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

from gymnasium import Space
from typing import Dict

class LidarOdomBlender(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: Space,
        num_agents: int,
        num_odom_layers: int
    ):
        """
        Combines TinyLidarNet from 
        https://arxiv.org/abs/2410.07447
        with MLP for odometry data
        """
        self.num_beams = observation_space['scan'].shape[-1]
        self.num_odom_layers = num_odom_layers
        features_dim = self._conv_outsize() * num_agents
        super().__init__(observation_space, features_dim=features_dim)

        self.convs = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=10, stride=4),
            nn.ReLU(),
            nn.Conv1d(24, 36, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(36, 48, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.odom_mlp = nn.ModuleList()
        if num_odom_layers > 0:
            self.odom_mlp.append(nn.Linear(8 * num_agents, features_dim)) # odom has 8 features
            self.odom_mlp.append(nn.ReLU())
            for _ in range(num_odom_layers - 1):
                self.odom_mlp.append(nn.Linear(features_dim, features_dim))
                self.odom_mlp.append(nn.ReLU())

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        scan, odom = obs["scan"], obs["odometry"] # (n_envs, n_agents, num_beams), (n_envs, n_agents, odom_dim)
        scan = scan.reshape(scan.shape[0], -1).unsqueeze(1) # (n_envs, 1, n_agents * num_beams)
        odom = odom.reshape(odom.shape[0], -1) # (n_envs, n_agents * odom_dim)

        scan = self.convs(scan)
        scan = torch.flatten(scan, start_dim=1)
        if len(self.odom_mlp) > 0:
            for layer in self.odom_mlp:
                odom = layer(odom)
            scan = scan + odom

        return scan

    def _conv_outsize(self) -> int:
        """Gets output size of conv layers for num_beams"""
        length = self.num_beams
        kernels = [10, 8, 4, 3, 3]
        strides = [4, 4, 2, 1, 1]
        for kernel, stride in zip(kernels, strides):
            length = (length - (kernel - 1) - 1) // stride + 1
        return length * 64 # 64 channels in last conv layer
    

class TinyLidarFCN(nn.Module):
    def __init__(self, outsize: int):
        """
        Original action prediction head
        from TinyLidarNet

        Consider using this in place of
        default ActorCriticPolicy head?
        """
        super().__init__()
        self.fcns = nn.Sequential(
            nn.Linear(in_features=outsize, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcns(x)

    
def example():
    import gymnasium as gym
    import numpy as np
    
    large_num = 1e6
    scan_size = 1080
    num_agents = 1
    space = {
        'scan': gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_agents, scan_size), dtype=np.float32
        ),
        'odometry': gym.spaces.Box( # contains all of pose, velocity, and heading
            low=-large_num, high=large_num, shape=(num_agents, 8), dtype=np.float32
        )
    }

    model = LidarOdomBlender(space, num_agents=num_agents, num_odom_layers=2)
    scan = torch.randn(1, scan_size)  # Batch size 1, 1 channel, 1080 beams
    odom = torch.randn(1, 8)        # Batch size 1, 3 odometry features
    obs = {"scan": scan, "odometry": odom}
    output = model(obs)
    print(output.shape)

if __name__ == "__main__":
    example()
