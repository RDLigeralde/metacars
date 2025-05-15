from meta.opponents.opponent import OpponentDriver
from meta.meta_env import F110MultiView
import gymnasium as gym
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import cfg_from_yaml
from typing import List, Optional
import argparse
import time
import os

def evaluate(
    model_path: str,
    config_path: str,
    opp_dir: str = None,
    n_episodes: int = 10,
    render: bool = True,
    make_video: bool = False,
    deterministic: bool = True,
    verbose: bool = True,
    norm_path: Optional[str] = None,
    MAX_EPISODE_LENGTH: int = 500  # 5 real seconds
):
    """Logs per-episode metrics over n_episodes."""
    env_args, ppo_args, _, log_args, opp_args = cfg_from_yaml(config_path)
    if make_video:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = "none"
    
    # Create the base F110Multi environment
    base_env = gym.make(
        id='meta.meta_env:F110Multi-v0',
        config=env_args,
        render_mode=render_mode,
    )
    
    # Wrap it with the F110MultiView wrapper
    env = F110MultiView(
        env=base_env,
        opp_dir=opp_dir,
        opp_cfg=opp_args,
    )
    env = Monitor(env)
    
    # Check if normalization was used during training

    use_vecnorm = norm_path and os.path.exists(norm_path)
    if use_vecnorm:
        if verbose:
            print(f"Loading normalization statistics from {norm_path}")
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    
    if render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(env, f"video_{time.time()}")
    
    recurrent = ppo_args.pop('recurrent', False)
    ppo_args.pop('init_path', None)
    
    model_class = RecurrentPPO if recurrent else PPO
    
    # Load the model
    model = model_class.load(model_path, env)
    # Set the environment for the model
    model.set_env(env)
    
    total_timestep_rewards = []  
    episode_lengths = []
    overtakes = 0
    crashes = 0
    
    if verbose:
        print(f"Evaluating {model_path} for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        if use_vecnorm:
            obs = env.reset()
        else:
            obs, opp_vmax = env.reset()

        if episode % 20 == 0:
            print(episode)

        episode_steps = 0
        done = False
        info = {}
        lstm_states = None
        
        last_reward = 0
        has_overtaken = False
        has_crashed = False
        
        while not done and episode_steps < MAX_EPISODE_LENGTH:
            episode_steps += 1
            if recurrent:
                action, lstm_states = model.predict(
                    obs, 
                    deterministic=deterministic, 
                    state=lstm_states
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            
            if use_vecnorm:
                obs, reward, terminated, info = env.step(action)
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get('custom/reward_terms/collision') != 0.0:
                    has_crashed = True
                if info.get('custom/reward_terms/overtaking') != 0.0:
                    has_overtaken = True
                total_reward_this_step = info.get('custom/reward_terms/total_timestep_reward', 0.0)
            done = terminated or truncated
            env.render()
            
        if has_overtaken:
            overtakes += 1
        if has_crashed:
            crashes += 1
        episode_lengths.append(episode_steps)
        total_timestep_rewards.append(total_reward_this_step)
        
            
    env.close()
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"Overtakes completed: {overtakes}")
    print(f"Crashes: {crashes}")
    print(f"Episode length: {episode_steps}")
    print(f"All episode lengths so far: {episode_lengths}")
    print(f"Episode Lengths — Mean: {mean_length:.2f}, Std Dev: {std_length:.2f}")

    mean_total_reward = np.mean(total_timestep_rewards)
    std_total_reward = np.std(total_timestep_rewards)

    print(f"Total Timestep Reward (last step) — Mean: {mean_total_reward:.4f}, Std Dev: {std_total_reward:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model in F1TENTH environment with multiple opponents')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file used for training')
    parser.add_argument('--norm_path', type=str, help='Path to the normalization statistics file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--video', action='store_true', help='Record video of the evaluation (requires --no-render)')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--opp_dir', type=str, help='Path to opponent directory', default=None)
    
    args = parser.parse_args()
    
    if not args.norm_path:
        model_dir = os.path.dirname(args.model)
        potential_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_norm_path):
            args.norm_path = potential_norm_path
            print(f"Found normalization file at {args.norm_path}")
    
    args.render = not args.no_render and not args.video
    evaluate(
        model_path=args.model,
        config_path=args.config,
        opp_dir=args.opp_dir,
        n_episodes=args.episodes,
        render=not args.no_render,
        make_video=args.video,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
        norm_path=args.norm_path
    )

if __name__ == '__main__':
    main()