from rl_env import F110Ego, F110EnvDR
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb

from utils import get_cfg_dicts
import argparse
import os

def train(
    env_args: dict,
    ppo_args: dict,
    train_args: dict,
    log_args: dict,
    yml_name: str
):
    model_save_freq = train_args.pop('save_interval')
    if log_args['project_name']:
        run = wandb.init(
            project=log_args['project_name'],
            sync_tensorboard=True,
            save_code=True,
            config={
                'env_args': env_args,
                'ppo_args': ppo_args,
                'train_args': train_args,
            }
        )
        model_save_freq = model_save_freq if model_save_freq else train_args['total_timesteps']
        callback = WandbCallback(
            gradient_save_freq=0, 
            model_save_path=f"models/{yml_name}", 
            model_save_freq=model_save_freq,
            verbose=2
        )
    else:
        run = None
        callback = None

    tensorboard_log = f"runs/{yml_name}" if log_args.pop('log_tensorboard') else None
    render_mode = env_args.pop('render_mode')

    def make_env():
        base = gym.make('ppo:f1tenth-v0-dr', config=env_args, render_mode=render_mode)
        return F110Ego(base)
    
    recurrent = ppo_args.pop('recurrent')
    vec_args = env_args.pop('num_envs')
    num_envs, env_type = vec_args['count'], vec_args['type']

    ppo_type = RecurrentPPO if recurrent else PPO # might want to try different learning algorithms later on
    vec_env_cls = SubprocVecEnv if env_type == 'subproc' else DummyVecEnv
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"
    
    if num_envs == 1:
        env = make_env()
    elif num_envs > 1:
        env = make_vec_env(
            make_env,
            n_envs=num_envs,
            vec_env_cls=vec_env_cls
        )
    else:
        num_envs = os.cpu_count()
        env = make_vec_env(
            make_env,
            n_envs=num_envs,
            vec_env_cls=vec_env_cls
        )

    init_path = ppo_args.pop('init_path')
    ppo = ppo_type(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        seed=env_args['seed'],
        **ppo_args,
        verbose=1
    )

    if init_path:
        ppo.load(init_path)

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

    env_args, ppo_args, train_args, log_args= get_cfg_dicts(args.config)
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
