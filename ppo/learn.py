from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


from archs import LidarOdomBlender, CenterlineCNN
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import gymnasium as gym
import wandb

import utils

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
    # Register the custom environment first
    # not sure why __init__.py is not donig this for us 
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
        wandb_callback = utils.CustomWandbCallback(
            gradient_save_freq=0, 
            model_save_path=f"models/{yml_name}/{run_name}", 
            model_save_freq=model_save_freq,
            verbose=2
        )
    else:
        run = None
        wandb_callback = None

    tensorboard_log = f"runs/{yml_name}" if log_args.pop('log_tensorboard') else None
    render_mode = env_args.pop('render_mode')

    
    def make_env():
        # Create the environment
        spec = gym.spec('f1tenth-v0-legacy')
        base = spec.make(
            config=env_args, 
            render_mode=render_mode
        )
        return Monitor(base)
    
    recurrent = ppo_args.pop('recurrent')
    vec_args = env_args.pop('num_envs')
    extractor_args = ppo_args.pop('feature_extractor')
    num_envs, env_type = vec_args['count'], vec_args['type']

    ppo_type = RecurrentPPO if recurrent else PPO # might want to try different learning algorithms later on
    vec_env_cls = SubprocVecEnv if env_type == 'subproc' else DummyVecEnv
    vec_env_kwargs = {} if env_type == 'dummy' else {'start_method': 'fork'}
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"
    
    if extractor_args['type'] == 'LidarOdomBlender':
        fe_kwargs = extractor_args['args']
        fe_kwargs['num_agents'] = env_args['num_agents']
        policy_kwargs = dict(
            features_extractor_class=LidarOdomBlender,
            features_extractor_kwargs=fe_kwargs
        )
    elif extractor_args['type'] == "CenterlineCNN":
        fe_kwargs = extractor_args['args']
        fe_kwargs['num_agents'] = env_args['num_agents']
        policy_kwargs = dict(
            features_extractor_class=CenterlineCNN,
            features_extractor_kwargs=fe_kwargs
        )
    else:
        policy_kwargs = None
    
    if num_envs == 1:
        env = make_env()
    else:
        if num_envs < 0:
            num_envs = os.cpu_count()
        env_inits = [
            utils.make_envs(i, env_args, render_mode, seed=env_args['seed']) 
            for i in range(num_envs)
        ]
        env = vec_env_cls(env_inits, **vec_env_kwargs)

    eval_env = make_vec_env(
        make_env,
        n_envs=1,  # Use single env for evaluation
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs
    )

    best_model_save_path = f"models/{yml_name}/{run_name}/best_model"
    os.makedirs(best_model_save_path, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=f"logs/{yml_name}/{run_name}/eval",
        eval_freq=max(model_save_freq // 4, 1000),  # Evaluate 4 times per save interval, minimum every 1000 steps
        n_eval_episodes=10,  # Number of episodes to evaluate
        deterministic=True,
        render=False,
        verbose=1
    )

    # Combine callbacks
    callbacks = [eval_callback]
    if wandb_callback:
        callbacks.append(wandb_callback)
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

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
            policy_kwargs=policy_kwargs,
            env=env,
            tensorboard_log=tensorboard_log,
            seed=env_args['seed'],
            **ppo_args,
            verbose=1
        )
    ppo.learn(
        **train_args,
        callback=callback,
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

    env_args, ppo_args, train_args, log_args = utils.get_cfg_dicts(args.config)
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
    gym.register(
        id="f1tenth-v0-legacy",
        entry_point="rl_env:F110EnvLegacy",
    )
    main()