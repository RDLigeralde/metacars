from meta.opponents.opponent import OpponentDriver
from meta.meta_env import F110MultiView
import gymnasium as gym
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import cfg_from_yaml
from typing import List, Optional
import argparse
import time
import os

def evaluate(
    model_path: str,
    config_path: str,
    opponents: List[OpponentDriver],
    n_episodes: int = 10,
    render: bool = True,
    make_video: bool = False,
    deterministic: bool = True,
    verbose: bool = True,
    norm_path: Optional[str] = None,
    MAX_EPISODE_LENGTH: int = 2000  # 10 real seconds
):
    """Logs per-episode metrics over n_episodes."""
    env_args, ppo_args, _, _ = cfg_from_yaml(config_path)
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
    env = F110MultiView(env=base_env, opponents=opponents)
    
    # Check if normalization was used during training
    if norm_path and os.path.exists(norm_path):
        if verbose:
            print(f"Loading normalization statistics from {norm_path}")
        # We need to vectorize the environment for VecNormalize
        env = DummyVecEnv([lambda: env])
        # Load the saved normalization stats
        env = VecNormalize.load(norm_path, env)
        # During evaluation, don't update the normalization statistics
        env.training = False
        # Don't normalize rewards during evaluation
        env.norm_reward = False
    
    if render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(env, f"video_{time.time()}")
    
    recurrent = ppo_args.pop('recurrent', False)
    ppo_args.pop('init_path', None)
    policy = "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy"
    model_class = RecurrentPPO if recurrent else PPO
    
    # Load the model
    model = model_class.load(model_path, env)
    # Set the environment for the model
    model.set_env(env)
    
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
        # Randomly select an opponent for each episode
        opponent_idx = np.random.randint(len(opponents)) if len(opponents) > 1 else 0
        
        # Reset the environment
        obs, _ = env.reset(opponent_idx=opponent_idx)
        
        episode_reward = 0
        episode_steps = 0
        done = False
        info = {}
        lstm_states = None
        
        start_time = time.time()
        
        while not done:
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
            
            if norm_path and os.path.exists(norm_path):
                original_reward = env.get_original_reward()
                episode_reward += original_reward.item() if hasattr(original_reward, 'item') else original_reward
            else:
                episode_reward += reward
            
            episode_steps += 1
            
            if render and not make_video:
                env.render()
        
        episode_duration = time.time() - start_time
        episode_durations.append(episode_duration)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        toggle_list = info['checkpoint_done']
        lap_completed = toggle_list[0] >= 4  # Assuming 0 as ego idx
        
        opponent_name = opponents[opponent_idx].__class__.__name__
        
        if lap_completed:
            laps += 1
            laptimes.append(episode_duration)
            if verbose:
                print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Duration={episode_duration:.2f}s")
                print(f"  ✅ Lap completed successfully against {opponent_name}")
        else:
            failures += 1
            crashtimes.append(episode_duration)
            if verbose:
                print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Duration={episode_duration:.2f}s")
                print(f"  ❌ Episode ended without completing lap against {opponent_name} (crash inferred)")
    
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
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model in F1TENTH environment with multiple opponents')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file used for training')
    parser.add_argument('--norm_path', type=str, help='Path to the normalization statistics file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--video', action='store_true', help='Record video of the evaluation (requires --no-render)')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    if not args.norm_path:
        model_dir = os.path.dirname(args.model)
        potential_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_norm_path):
            args.norm_path = potential_norm_path
            print(f"Found normalization file at {args.norm_path}")
    
    opponents = [OpponentDriver()]
    
    args.render = not args.no_render and not args.video
    evaluate(
        model_path=args.model,
        config_path=args.config,
        opponents=opponents,
        n_episodes=args.episodes,
        render=not args.no_render,
        make_video=args.video,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
        norm_path=args.norm_path
    )

if __name__ == '__main__':
    main()