from wandb.integration.sb3 import WandbCallback
import wandb
import yaml

def get_cfg_dicts(yml_path):
    """Gets configuration dictionaries from a YAML file"""
    try:
        with open(yml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            world, car, rewards, ppo_params, train_params, log = (cfg[key] for key in ['world', 'car', 'reward_params', 'ppo_params', 'train_params', 'log'])
            world['params'] = car
            world['reward_params'] = rewards
            return world, ppo_params, train_params, log
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