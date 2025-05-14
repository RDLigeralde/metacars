# from meta.opponents.opponent import OpponentDriver
from meta.meta_env import F110MultiView
import gymnasium as gym
import torch.nn as nn
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
from utils import cfg_from_yaml
from typing import List, Optional
import argparse
import time
import os
from learn_mql import ActorNetwork
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

    observation_space = env.observation_space
    print(observation_space)

    action_space = env.action_space
    action_dim = action_space.shape[1]
    # Step 1: Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())


    # Step 2: Initialize the models
    actor = ActorNetwork(observation_space, action_dim, features_dim=256)
    context_encoder = nn.GRU(input_size=551, hidden_size=128, num_layers=2, batch_first=True) # Make sure to match the architecture

    # Step 3: Load the state_dicts
    actor.load_state_dict(checkpoint['actor'])
    context_encoder.load_state_dict(checkpoint['context_encoder'])

    # Step 4: Set them to evaluation mode if needed
    actor.eval()
    context_encoder.eval()

    def get_context_feats_eval(act_rew_obs, context_encoder):
        """
        Given a history of (actions, rewards, obs), encode with GRU and return
        the final hidden vector (batch, context_dim).
        """
        hist_actions, hist_rewards, hist_obs = act_rew_obs

        if hist_rewards.dim() == 1:
            hist_rewards = hist_rewards.unsqueeze(-1)

        # Fix shape: squeeze hist_actions if needed
        if hist_actions.dim() == 4:
            hist_actions = hist_actions.squeeze(2)

        # Fix shape: squeeze obs tensors to (B, H, d_k)
        for k in hist_obs:
            while hist_obs[k].dim() > 3:
                hist_obs[k] = hist_obs[k].squeeze(2)

        obs_seq = torch.cat(
            [hist_obs[k].reshape(hist_obs[k].size(0), hist_obs[k].size(1), -1)
            for k in sorted(hist_obs.keys())],
            dim=2
        )
        # print("DEBUG: Flattened obs_seq shape:", obs_seq.shape)

        # (B, H, 1)
        rew_seq = hist_rewards.unsqueeze(-1)

        # (B, H, A+1+obs_dim)
        seq = torch.cat([hist_actions, rew_seq, obs_seq], dim=2)

        # Pass through GRU
        _, h_n = context_encoder(seq)

        output = h_n.squeeze(0)
        return output


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

            previous_action = np.zeros_like(env.action_space.sample())  # Initialize with zeros
            previous_reward = 0.0
            previous_obs = obs

            H = 10  # History window size

            # Initialize historical buffers as zero tensors
            historical_actions = np.zeros((H, *env.action_space.shape), dtype=np.float32)
            historical_rewards = np.zeros((H,), dtype=np.float32)
            historical_observations = {
                key: np.zeros((H, *value.shape), dtype=np.float32)
                for key, value in obs.items()
            }


            while not done and episode_steps < MAX_EPISODE_LENGTH:

                obs_tensor = {key: torch.tensor(value, dtype=torch.float32)for key, value in obs.items()}

                hist_actions = torch.tensor(historical_actions, dtype=torch.float32).unsqueeze(0)
                hist_rewards = torch.tensor(historical_rewards, dtype=torch.float32).unsqueeze(0)
                hist_obs_tensor = {
                    k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                    for k, v in historical_observations.items()
                }

                if recurrent:
                    action, lstm_states = actor.predict(obs, deterministic=deterministic, state=lstm_states)
                else:
                    action = actor.forward(obs, context_feats=get_context_feats_eval((hist_actions, hist_rewards, hist_obs_tensor), context_encoder))

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