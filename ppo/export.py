from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import gymnasium as gym

import torch.nn as nn
import torch

from utils import get_cfg_dicts
import argparse
from rl_env import F110Ego, F110EnvDR

class ONNXPolicy(nn.Module):
    def __init__(self, policy):
        """wraps policy in torch module for ONNX export"""
        super().__init__()
        self.policy = policy

    def _to_dict(self, scan, pose, vel, heading):
        return {
            'scan': scan,
            'pose': pose,
            'vel': vel,
            'heading': heading
        }

    def forward(self, scan, pose, vel, heading):
        obs_dict = self._to_dict(scan, pose, vel, heading)
        return self.policy(obs_dict, deterministic=True)
    
def export_onnx(
    model_path: str,
    config_path: str,
    out_path: str
):
    """saves sb3 policies in ONNX format"""
    env_args, ppo_args, _, _ = get_cfg_dicts(config_path)
    print(ppo_args)
    recurrent = ppo_args.pop('recurrent')
    ppo_args.pop('init_path')

    model_class = RecurrentPPO if recurrent else PPO
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"
    
    env = gym.make('ppo.rl_env:f1tenth-v0-dr', config=env_args)
    model = model_class(policy, env, **ppo_args)
    model.policy.load_state_dict(torch.load(model_path))
    onnx_model = ONNXPolicy(model.policy)

    # frenet observation space w/ 36 beams hardcoded for now
    # maybe grab this from the experiment yaml later
    example_obs = (
        torch.randn(1, 36),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 2)
    )
    input_names = ['scan', 'pose', 'vel', 'heading']

    torch.onnx.export(
        onnx_model,
        example_obs,
        out_path,
        input_names=input_names,
    )

def main():
    parser = argparse.ArgumentParser(description='Export ONNX model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)

    args = parser.parse_args()
    export_onnx(args.model_path, args.config_path, args.out_path)

if __name__ == '__main__':
    main()
