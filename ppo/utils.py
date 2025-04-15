import yaml

def get_cfg_dicts(yml_path):
    """Gets configuration dictionaries from a YAML file"""
    try:
        with open(yml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            world, car, ppo_params, train_params, log = (cfg[key] for key in ['world', 'car', 'ppo_params', 'train_params', 'log'])
            world['params'] = car
            return world, ppo_params, train_params, log
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None