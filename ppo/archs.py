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
        num_odom_layers: int,
        num_cline_layers: int = 0,
        use_convs: bool = True
    ):
        """
        Combines TinyLidarNet from 
        https://arxiv.org/abs/2410.07447
        with MLP for odometry data
        """
        self.num_beams = observation_space['scan'].shape[-1]
        self.n_cline_points = observation_space['cline_ctx'].shape[-2]
        self.num_odom_layers = num_odom_layers
        self.use_convs = use_convs
        self.embed_size = self._conv_outsize() * num_agents
        super().__init__(observation_space, features_dim=self.embed_size)

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
        ) if use_convs else None

        self.odom_mlp = nn.ModuleList()
        if num_odom_layers > 0:
            self.odom_mlp.append(nn.Linear(5 * num_agents, self.embed_size)) # odom has 8 features
            self.odom_mlp.append(nn.ReLU())
            for _ in range(num_odom_layers - 1):
                self.odom_mlp.append(nn.Linear(self.embed_size, self.embed_size))
                self.odom_mlp.append(nn.ReLU())

        self.cline_mlp = nn.ModuleList()
        if num_cline_layers > 0:
            self.cline_mlp.append(nn.Linear(self.n_cline_points * 2 * num_agents, self.embed_size))
            self.cline_mlp.append(nn.ReLU())
            for _ in range(num_cline_layers - 1):
                self.cline_mlp.append(nn.Linear(self.embed_size, self.embed_size))
                self.cline_mlp.append(nn.ReLU())

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        scan, odom, cline = obs["scan"], obs["odometry"], obs["cline_ctx"]
        scan = scan.reshape(scan.shape[0], -1).unsqueeze(1) # (n_envs, 1, num_beams)
        odom = odom.reshape(odom.shape[0], -1) # (n_envs, n_agents * odom_dim)
        cline = cline.reshape(cline.shape[0], -1) # (n_envs, n_agents * n_cline_points * 2)

        if self.use_convs:
            x = self.convs(scan)
            x = torch.flatten(x, start_dim=1)
        else:
            x = torch.zeros((odom.shape[0], self.embed_size), device=odom.device)
        if len(self.odom_mlp) > 0:
            for layer in self.odom_mlp:
                odom = layer(odom)
            x = x + odom
        if len(self.cline_mlp) > 0:
            for layer in self.cline_mlp:
                cline = layer(cline)
            x = x + cline

        return x

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
