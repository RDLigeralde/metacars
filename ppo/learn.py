from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import gymnasium as gym
import wandb

from rl_env import F110LegacyViewer, OpponentDriver
from meta.opponents.pure_pursuit import PurePursuit

from utils import get_cfg_dicts, CustomWandbCallback
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
        callback = CustomWandbCallback(
            gradient_save_freq=0, 
            model_save_path=f"models/{yml_name}/{run_name}", 
            model_save_freq=model_save_freq,
            verbose=2
        )
    else:
        run = None
        callback = None

    tensorboard_log = f"runs/{yml_name}" if log_args.pop('log_tensorboard') else None
    render_mode = env_args.pop('render_mode')

    
    def make_env():
        # Create the environment
        base = gym.make(
            'f1tenth-v0-legacy', 
            config=env_args, 
            render_mode=render_mode
        )
        if env_args['num_agents'] == 2:
            conf = {
                'wheelbase': 0.5,
                'lookahead': 0.4,
                'max_reacquire': 1.0,
                'waypoints': 'lup',
                'synthbrake': 0.35,
            }
            conf['s_max'] = env_args['params']['s_max']
            conf['v_max'] = env_args['params']['v_max']
            opponents = [PurePursuit(conf)]
        else:
            opponents = None
        base = F110LegacyViewer(base, render_mode=render_mode, opponents=opponents)
        return Monitor(base)
    
    recurrent = ppo_args.pop('recurrent')
    vec_args = env_args.pop('num_envs')
    num_envs, env_type = vec_args['count'], vec_args['type']

    ppo_type = RecurrentPPO if recurrent else PPO # might want to try different learning algorithms later on
    vec_env_cls = SubprocVecEnv if env_type == 'subproc' else DummyVecEnv
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"
    
    norm_obs, norm_rew = env_args.pop('normalize_observations'), env_args.pop('normalize_rewards')
    norm_obs, norm_rew = False, False
    if num_envs == 1:
        env = make_env()
        if norm_obs or norm_rew:
            env = DummyVecEnv([lambda: env])
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

    if norm_obs or norm_rew:
        env = VecNormalize(
            env,
            norm_obs=norm_obs,
            norm_reward=norm_rew,
            clip_obs=10.0, # TODO: make this a config option
            gamma=env_args['gamma'],
        )

    init_path = ppo_args.pop('init_path')
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
            seed=env_args['seed'],
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
    parser.add_argument('--run_name', type=str, help='Name for distinguishing runs')
    args = parser.parse_args()

    env_args, ppo_args, train_args, log_args= get_cfg_dicts(args.config)
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