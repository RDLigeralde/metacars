from rl_env import F110Ego
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from wandb.integration.sb3 import WandbCallback
import wandb

import yml_utils as yml
import argparse
import os

def train(
    env_args: dict,
    ppo_args: dict,
    train_args: dict,
    log_args: dict,
    yml_name: str
):
    if log_args['project_name']:
        run = wandb.init(
            project=log_args['project_name'],
            sync_tensorboard=True,
            save_code=True,
        )
        callback = WandbCallback(
            gradient_save_freq=0, model_save_path=f"models/{yml_name}", verbose=2
        )
    else:
        run = None
        callback = None
    
    tensorboard_log = f"runs/{yml_name}" if log_args.pop('log_tensorboard') else None

    def make_env():
        base = gym.make('ppo:f1tenth-v0-dr', config=env_args)
        return F110Ego(base)

    num_envs = env_args.pop('num_envs')
    if num_envs == 1:
        env = make_env()
    elif num_envs > 1:
        env = make_vec_env(
            make_env,
            n_envs=num_envs
        )
    else:
        num_envs = os.cpu_count()
        env = make_vec_env(
            make_env,
            n_envs=num_envs
        )

    recurrent = ppo_args.pop('recurrent')
    ppo_type = RecurrentPPO if recurrent else PPO # might want to try different learning algorithms later on
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"

    ppo = ppo_type(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        **ppo_args,
        verbose=1
    )

    ppo.learn(
        **train_args,
        callback=callback
    )

    if run:
        run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='Path to the config file'
    )
    args = parser.parse_args()

    env_args, ppo_args, train_args, log_args = yml.get_cfg_dicts(args.config)
    yml_name = os.path.basename(args.config)
    train(
        env_args=env_args,
        ppo_args=ppo_args,
        train_args=train_args,
        log_args=log_args,
        yml_name=os.path.splitext(yml_name)[0]
    )

if __name__ == '__main__':
    main()
