from stable_baselines3.common.monitor import Monitor
from rl_env import F110LegacyViewer, OpponentDriver
import gymnasium as gym
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import torch

from utils import get_cfg_dicts
import argparse
import time


# from f1tenth_gym.envs.f110_env import F110Env

def evaluate(
    model_path: str,
    config_path: str,
    n_episodes: int = 10,
    render: bool = True,
    make_video: bool = False,
    deterministic: bool = True,
    verbose: bool = True,
    MAX_EPISODE_LENGTH: int = 1000#1000 # 10 real seconds
):
    gym.register(
        id="f1tenth-v0-legacy",
        entry_point="rl_env:F110EnvLegacy",
    )

    """Logs per-episode metrics over n_episodes."""
    env_args, ppo_args, _, _ = get_cfg_dicts(config_path)
    if make_video:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = "none"
    
    env = gym.make('f1tenth-v0-legacy', config=env_args, render_mode=render_mode)
    if env_args['num_agents'] == 2:
        opponents = [OpponentDriver()]
    else:
        opponents = None
    env = F110LegacyViewer(env, render_mode=render_mode, opponents=opponents)
    env = Monitor(env)

    # env = F110EnvLegacy(config=env_args, render_mode=render_mode)
    # env = gym.make('f1tenth_gym:f1tenth-v0', config=env_args, render_mode=render_mode)
    if render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(env, f"video_{time.time()}")
    ego_idx = env.unwrapped.ego_idx
    
    recurrent = ppo_args.pop('recurrent')
    ppo_args.pop('init_path')
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"
    model_class = RecurrentPPO if recurrent else PPO
    # print(recurrent)
    model = model_class.load(model_path, env)
    model.set_env(env)
    # model = model_class(policy, env, **ppo_args)
    # model.policy.load_state_dict(torch.load(model_path))
    
    episode_rewards = []
    episode_lengths = []
    episode_durations = []
    laptimes = []
    crashtimes = []
    laps = 0
    failures = 0
    
    if verbose:
        print(f"Evaluating {model_path} for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        info = {}
        lstm_states = None
        
        start_time = time.time()
        
        while not done and episode_steps <= MAX_EPISODE_LENGTH:
            if recurrent:
                action, lstm_states = model.predict(
                    obs, 
                    deterministic=deterministic, 
                    state=lstm_states
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            if render:
                env.render()
        
        episode_duration = time.time() - start_time
        episode_durations.append(episode_duration)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        toggle_list = info['checkpoint_done']
        lap_completed = toggle_list[ego_idx] >= 4 # TODO: check if this is actually a loop closure flag
        
        if lap_completed:
            laps += 1
            laptimes.append(episode_duration)
            if verbose:
                print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Duration={episode_duration:.2f}s")
                print(f"  ✅ Lap completed successfully")
        else:
            failures += 1
            crashtimes.append(episode_duration)
            if verbose:
                print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Duration={episode_duration:.2f}s")
                print(f"  ❌ Episode ended without completing lap (crash inferred)")
    
    if verbose:
        print("\n===== Evaluation Summary =====")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average episode steps: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Average episode duration: {np.mean(episode_durations):.2f} ± {np.std(episode_durations):.2f}s")
        
        print(f"\nSuccessful lap completions: {laps}/{n_episodes} ({laps/n_episodes*100:.1f}%)")
        if laptimes:
            print(f"  Average completion time: {np.mean(laptimes):.2f} ± {np.std(laptimes):.2f}s")
            print(f"  Fastest completion time: {np.min(laptimes):.2f}s")
        
        print(f"\nInferred crashes: {failures}/{n_episodes} ({failures/n_episodes*100:.1f}%)")
        if crashtimes:
            print(f"  Average time before crash: {np.mean(crashtimes):.2f} ± {np.std(crashtimes):.2f}s")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model in F1TENTH environment')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file used for training')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--video', action='store_true', help='Record video of the evaluation (requires --no-render)')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()

    args.render = not args.no_render and not args.video
    evaluate(
        model_path=args.model,
        config_path=args.config,
        n_episodes=args.episodes,
        render=not args.no_render,
        make_video=args.video,
        deterministic=not args.stochastic,
        verbose=not args.quiet
    )

if __name__ == '__main__':
    main()