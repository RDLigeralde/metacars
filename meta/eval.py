# from meta.opponents.opponent import OpponentDriver
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
import csv

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
    MAX_EPISODE_LENGTH: int = 1000
):
    """Logs per-episode metrics over n_episodes and writes to a CSV file."""
    env_args, ppo_args, _, log_args, opp_args = cfg_from_yaml(config_path)
    render_mode = "rgb_array" if make_video else ("human" if render else "none")

    base_env = gym.make('meta.meta_env:F110Multi-v0', config=env_args, render_mode=render_mode)
    env = F110MultiView(env=base_env, opp_dir=opp_dir, opp_cfg=opp_args)
    env = Monitor(env)

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
    model = model_class.load(model_path, env)
    model.set_env(env)

    # Prepare log file
    log_file = "evaluation_log.csv"
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode", "Reward", "Length", "Duration(s)",
            "Laps", "Failures", "Laptimes", "Overtakes"
        ])

        if verbose:
            print(f"Evaluating {model_path} for {n_episodes} episodes...")

        for episode in range(n_episodes):
            obs = env.reset() if use_vecnorm else env.reset()[0]
            episode_reward = 0
            episode_steps = 0
            done = False
            lstm_states = None
            laptime_list = []
            overtake_count = 0
            laps = 0
            start_time = time.time()

            while not done and episode_steps < MAX_EPISODE_LENGTH:
                if recurrent:
                    action, lstm_states = model.predict(obs, deterministic=deterministic, state=lstm_states)
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)

                if use_vecnorm:
                    obs, reward, terminated, info = env.step(action)
                    truncated = False
                else:
                    obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1

                if "laptime" in info:
                    laptime_list.append(info["laptime"])
                if "overtakes" in info:
                    overtake_count += info["overtakes"]
                if "laps" in info:
                    laps = max(laps, info["laps"])

                if render:
                    env.render()

            end_time = time.time()
            duration = round(end_time - start_time, 2)
            failed = int(info.get("crashed", False) or (episode_steps >= MAX_EPISODE_LENGTH))

            # Save to CSV
            writer.writerow([
                episode + 1, round(float(episode_reward), 2), episode_steps,
                duration, laps, failed, laptime_list, overtake_count
            ])

            if verbose:
                print(f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_steps}, duration={duration}s, laps={laps}, failures={failed}")

    env.close()
    print(f"Evaluation complete. Results saved to {log_file}")


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