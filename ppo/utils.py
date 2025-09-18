from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym

from wandb.integration.sb3 import WandbCallback
import wandb

import numpy as np
import yaml
import os


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_steps: int):
        """
        Gym wrapper to repeat the same action
        for n_steps in the environment.

        Args:
            env (gym.Env): The environment to wrap.
            n_steps (int): The number of steps to repeat the action.
        """
        super().__init__(env)
        self.n_steps = n_steps

    def step(self, action: np.ndarray):
        total_reward = 0.0
        for _ in range(self.n_steps):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info


def get_cfg_dicts(yml_path):
    """Gets configuration dictionaries from a YAML file"""
    try:
        with open(yml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            world, car, rewards, ppo_params, train_params, log = (
                cfg[key] for key in
                ['world', 'car', 'reward_params', 'ppo_params', 'train_params', 'log']
            )
            world['params'] = car
            world['reward_params'] = rewards
            return world, ppo_params, train_params, log
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None
    

def make_envs(rank: int, global_cfg: dict, render_mode: str, seed: int = 0):
    """
    Custom vecenv creation to ensure that each vecenv
    gets a unique, fixed, track
    """
    tracks = os.listdir(global_cfg['map'])
    n_steps = global_cfg.get('n_steps', 1)
    def _init():
        if os.path.isdir(os.path.join(global_cfg['map'], tracks[0])):
            track = os.path.join(global_cfg['map'], tracks[rank])
        else:
            track = global_cfg['map']
        local_cfg = global_cfg.copy()
        local_cfg['map'] = track
        env = gym.make('f1tenth-v0-legacy', config=local_cfg, render_mode=render_mode)
        if n_steps > 1:
            env = MultiStepWrapper(env, n_steps)
        env = Monitor(env)
        env.reset(seed = seed + rank)
        return env
    
    return _init
  

class CustomWandbCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _on_step(self) -> bool:
        # Log custom metric if present
        for key in self.locals['infos'][0]:
            if 'custom' in key:
                wandb.log({key: self.locals['infos'][0][key]}, step=self.num_timesteps)
        return super()._on_step()