from wandb.integration.sb3 import WandbCallback
import wandb
import yaml
import os

from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

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
    def _init():
        if os.path.isdir(os.path.join(global_cfg['map'], tracks[0])):
            track = os.path.join(global_cfg['map'], tracks[rank])
        else:
            track = global_cfg['map']
        local_cfg = global_cfg.copy()
        local_cfg['map'] = track
        env = gym.make('f1tenth-v0-legacy', config=local_cfg, render_mode=render_mode)
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