from wandb.integration.sb3 import WandbCallback
import wandb

import yaml

def cfg_from_yaml(yml_path):
    """Load a YAML configuration file."""
    try:
        with open(yml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            world, reward, opps, car, ppo_params, train_params, log = (
                cfg[key] for key in [
                    'world', 
                    'reward_params',
                    'opp_params',
                    'car', 
                    'ppo_params', 
                    'train_params', 
                    'log'
                ]
            )
            world['params'] = car
            world['reward_params'] = reward
            return world, ppo_params, train_params, log, opps
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None
    
class CustomWandbCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _on_step(self) -> bool:
        # Log custom metric if present
        for key in self.locals['infos'][0]:
            if 'custom' in key:
                wandb.log({key: self.locals['infos'][0][key]}, step=self.num_timesteps)
        return super()._on_step()