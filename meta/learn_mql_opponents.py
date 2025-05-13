import torch
import torch.nn as nn
import gymnasium as gym
from meta_env import F110MultiView
from network import LIDARConvExtractor
from mql import MQL
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import DictReplayBuffer
import numpy as np
from gymnasium.spaces import Box, Dict
from GigaBuffer import GigaBuffer
import wandb

from utils import cfg_from_yaml
import argparse
import os

class CriticNetwork(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_dim: int, features_dim: int = 256):
        """
        Critic network that predicts Q-values using a fully connected architecture
        for the 'heading', 'pose', 'scan', and 'vel' observations.
        """
        super(CriticNetwork, self).__init__()

        # Extract dimensions from observation space
        heading_dim = observation_space['heading'].shape[1]
        pose_dim = observation_space['pose'].shape[1]
        scan_dim = observation_space['scan'].shape[1]
        vel_dim = observation_space['vel'].shape[1]

        # Define separate MLPs for each observation type
        self.heading_mlp = nn.Sequential(
            nn.Linear(heading_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.scan_mlp = nn.Sequential(
            nn.Linear(scan_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.vel_mlp = nn.Sequential(
            nn.Linear(vel_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combine all embeddings with actions
        total_embed_dim = 64 + 64 + 128 + 64 + action_dim
        self.q1_layer = nn.Sequential(
            nn.Linear(total_embed_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1),
        )

        self.q2_layer = nn.Sequential(
            nn.Linear(total_embed_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1),
        )

    def big_squeeze(self, observations, actions):
        heading_embed = self.heading_mlp(observations['heading'].squeeze(1))
        pose_embed = self.pose_mlp(observations['pose'].squeeze(1))
        scan_embed = self.scan_mlp(observations['scan'].squeeze(1))
        vel_embed = self.vel_mlp(observations['vel'].squeeze(1))

        # Concatenate all embeddings with actions
        cat_embed = torch.cat((heading_embed, pose_embed, scan_embed, vel_embed, actions), dim=1)
        return cat_embed


    def forward(self, observations: dict, actions: torch.Tensor, pre_act_rew) -> torch.Tensor:
        cat_embed = self.big_squeeze(observations, actions)
        x1 = self.q1_layer(cat_embed)
        x2 = self.q2_layer(cat_embed)
        return x1,x2
    
    def Q1(self, x, u, pre_act_rew = None):
        '''
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''
        xu = self.big_squeeze(x, u)
        x1 = self.q1_layer(xu)
        return x1
    


class ActorNetwork(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_dim: int, features_dim: int = 256):
        """
        Actor network that predicts actions using a fully connected architecture
        for the 'heading', 'pose', 'scan', and 'vel' observations.
        """
        super(ActorNetwork, self).__init__()

        # Extract dimensions from observation space
        heading_dim = observation_space['heading'].shape[1]
        pose_dim = observation_space['pose'].shape[1]
        scan_dim = observation_space['scan'].shape[1]
        vel_dim = observation_space['vel'].shape[1]

        # Define separate MLPs for each observation type
        self.heading_mlp = nn.Sequential(
            nn.Linear(heading_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.scan_mlp = nn.Sequential(
            nn.Linear(scan_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.vel_mlp = nn.Sequential(
            nn.Linear(vel_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combine all embeddings
        total_embed_dim = 64 + 64 + 128 + 64
        self.action_layer = nn.Sequential(
            nn.Linear(total_embed_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, action_dim),  # Match the action_dim 
            nn.Tanh(),  # Assuming actions are normalized between -1 and 1
        )

    def forward(self, observations: dict, pre_act_rew) -> torch.Tensor:
        # Process each observation type
        heading_embed = self.heading_mlp(observations['heading'].squeeze(1))
        pose_embed = self.pose_mlp(observations['pose'].squeeze(1))
        scan_embed = self.scan_mlp(observations['scan'].squeeze(1))
        vel_embed = self.vel_mlp(observations['vel'].squeeze(1))

        # Concatenate all embeddings
        cat_embed = torch.cat((heading_embed, pose_embed, scan_embed, vel_embed), dim=1)
        return self.action_layer(cat_embed)


def train_mql(env_args: dict, mql_args: dict, train_args: dict, log_args: dict, opp_args: dict, opp_dir: str):
    render_mode = env_args.pop('render_mode', None)

    def make_env():
        base = gym.make(
            id='meta.meta_env:F110Multi-v0',
            config=env_args,
            render_mode=render_mode,
        )
        viewer = F110MultiView(
            env=base,
            opp_dir=opp_dir,
            opp_cfg=opp_args,
        )
        return Monitor(viewer)

    env = DummyVecEnv([make_env])

    observation_space = env.observation_space
    print(observation_space)

    action_space = env.action_space
    action_dim = action_space.shape[1]

    # Initialize actor and critic networks
    actor = ActorNetwork(observation_space, action_dim, features_dim=256)
    actor_target = ActorNetwork(observation_space, action_dim, features_dim=256)
    critic = CriticNetwork(observation_space, action_dim, features_dim=256)
    critic_target = CriticNetwork(observation_space, action_dim, features_dim=256)

    # Initialize MQL
    mql = MQL(
        actor=actor,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        lr=mql_args.get('lr', 1e-3),
        gamma=mql_args.get('gamma', 0.99),
        ptau=mql_args.get('ptau', 0.005),
        policy_noise=mql_args.get('policy_noise', 0.2),
        noise_clip=mql_args.get('noise_clip', 0.5),
        policy_freq=mql_args.get('policy_freq', 2),
        batch_size=mql_args.get('batch_size', 100),
        device=mql_args.get('device', 'cpu'),
        max_action=1.0
    )

    if log_args['project_name']:
        run = wandb.init(
            project=log_args['project_name'],
            name=log_args.get('run_name', 'default_run'),
            config={
                'env_args': env_args,
                'mql_args': mql_args,
                'train_args': train_args,
                'log_args': log_args,
                'opp_args': opp_args,
            },
            save_code=True,
        )
    else:
        run = None

    # print("Observation Space:", env.observation_space)
    # for key, space in env.observation_space.spaces.items():
    #    print(f"Key: {key}, Space: {space}, Dtype: {getattr(space, 'dtype', None)}")

    # training loopno
    replay_buffer = GigaBuffer(
        buffer_size=1000,  # Set buffer size
        observation_space= env.observation_space,
        action_space=env.action_space,
        device=mql_args.get('device', 'cpu'),
        handle_timeout_termination=False,
    )
    iterations = train_args.get('iterations')

    for it in range(iterations):
        obs = env.reset()
        done = False

        # Initialize placeholders for previous and historical values
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

        while not done:
            # Convert observations to tensors
            obs_tensor = {key: torch.tensor(value, dtype=torch.float32) for key, value in obs.items()}

            # Choose action
            action = actor(obs_tensor, None).detach().numpy()

            # Step the environment
            next_obs, reward, done, info = env.step(action)

            # Add to replay buffer with previous and historical values in infos
            replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                prev_action=previous_action, 
                prev_reward=previous_reward,
                prev_obs=previous_obs,
                historical_action=historical_actions,
                historical_reward=historical_rewards,
                historical_obs=historical_observations,
                infos = None
            )

            # Update previous values
            previous_action = action
            previous_reward = reward
            previous_obs = obs

            historical_actions = np.roll(historical_actions, shift=-1, axis=0)
            historical_actions[-1] = action

            historical_rewards = np.roll(historical_rewards, shift=-1, axis=0)
            historical_rewards[-1] = reward

            for key in historical_observations:
                historical_observations[key] = np.roll(historical_observations[key], shift=-1, axis=0)
                historical_observations[key][-1] = obs[key]

            # Limit the size of historical buffers (e.g., keep the last 10 steps)
            if len(historical_actions) > 10:
                historical_actions.pop(0)
            if len(historical_rewards) > 10:
                historical_rewards.pop(0)
            if len(historical_observations) > 10:
                historical_observations.pop(0)

            # Move to the next observation
            obs = next_obs

            # Perform training
            train_metrics = mql.train(replay_buffer=replay_buffer, iterations=train_args.get('train_steps', 2048))

            if run:
                wandb.log({
                    'critic_loss': train_metrics[0]['critic_loss'],
                    'actor_loss': train_metrics[0]['actor_loss']
                })


    # Save the final model
    final_model_path = f"models/{log_args['run_name']}/final_model"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(mql.actor.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--run_name', type=str, help='Name for distinguishing runs')
    parser.add_argument('--opp_dir', type=str, help='Path to opponent directory', default=None)
    args = parser.parse_args()

    env_args, mql_args, train_args, log_args, opp_args = cfg_from_yaml(args.config)

    log_args['run_name'] = args.run_name

    train_mql(
        env_args=env_args,
        mql_args=mql_args,
        train_args=train_args,
        log_args=log_args,
        opp_args=opp_args,
        opp_dir=args.opp_dir,
    )


if __name__ == '__main__':
    main()