from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
import wandb

from meta.opponents.opponent import OpponentDriver
from meta_env import F110MultiView

from utils import cfg_from_yaml, CustomWandCallback
from time import gmtime, strftime
import argparse
import os

def train(
    env_args: dict,
    ppo_args: dict,
    train_args: dict,
    log_args: dict,
    yml_name: str,
    run_name: str,
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
        callback = CustomWandCallback(
            gradient_save_freq=0, 
            model_save_path=f"models/{yml_name}/{run_name}", 
            model_save_freq=model_save_freq,
            verbose=2
        )
    else:
        run = None
        callback = None

    tensorboard_log = f"runs/{yml_name}" if log_args.pop('log_tensorboard') else None
    render_mode = env_args.pop('render_mode', None)

    opponents = [OpponentDriver()] # TODO: replace with actual opponents
    def make_env():
        # Create base environment
        base = gym.make(
            id='meta.meta_env:F110Multi-v0',
            config=env_args,
            render_mode=render_mode,
        )
        viewer = F110MultiView(
            env=base, 
            opponents=opponents,
        )
        return viewer
    
    recurrent = ppo_args.pop('recurrent', False)
    vec_args = env_args.pop('num_envs', {'count': 1, 'type': 'dummy'})
    num_envs = vec_args.get('count', 1)
    env_type = vec_args.get('type', 'dummy')

    ppo_type = RecurrentPPO if recurrent else PPO
    vec_env_cls = SubprocVecEnv if env_type == 'subproc' else DummyVecEnv
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"
    
    # Create vectorized environment
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

    init_path = ppo_args.pop('init_path', None)
    if init_path:
        print(f'Loaded previous model from {init_path}')
        ppo = ppo_type.load(
            path=init_path,
            env=env,
            tensorboard_log=tensorboard_log,
            device=ppo_args.get('device', 'cpu')
        )
    else:
        ppo = ppo_type(
            policy=policy,
            env=env,
            tensorboard_log=tensorboard_log,
            seed=env_args.get('seed', 42),
            **ppo_args,
            verbose=1
        )
    
    ppo.learn(
        **train_args,
        callback=callback
    )

    final_model_path = f"models/{yml_name}/{run_name}/final_model"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    ppo.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    if run:
        run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='Path to the config file'
    )
    parser.add_argument('--run_name', type=str, help='Name for distinguishing runs')
    args = parser.parse_args()

    env_args, ppo_args, train_args, log_args = cfg_from_yaml(args.config)
    yml_name = os.path.basename(args.config)
    
    train(
        env_args=env_args,
        ppo_args=ppo_args,
        train_args=train_args,
        log_args=log_args,
        yml_name=os.path.splitext(yml_name)[0],
        run_name=args.run_name
    )

if __name__ == '__main__':
    main()