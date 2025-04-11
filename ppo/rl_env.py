from gymnasium import spaces
import gymnasium as gym
import numpy as np

from typing import List

class OpponentDriver:
    def __init__(self, **kwargs):
        """Wrapper class for opponent policies"""
        pass

    def drive(self, obs):
        """Drive the car: implemented in subclasses"""
        return np.zeros(2)

class F110Ego(gym.Wrapper):
    def __init__(self, env, opps: List[OpponentDriver] = None):
        """
        f1tenth env wrapper: action space only for ego,
        supports self-play against fixed
        """
        super().__init__(env)
        self.env = env
        self.ego_idx = env.unwrapped.ego_idx
        self.num_agents = env.unwrapped.num_agents

        self.action_space = env.unwrapped.action_type.space
        self.observation_type = env.unwrapped.observation_type
        self.observation_space = self.env.unwrapped.observation_space

        self.opp_idxs = [i for i in range(self.num_agents) if i != self.ego_idx]
        self.opps = opps if opps else [OpponentDriver()] * (self.num_agents - 1)
        
    def step(self, action: np.ndarray):
        """Steps using provided action + opponent policies"""
        actions = np.zeros((self.num_agents, 2))
        actions[self.ego_idx] = action

        opp_idx = 0
        obs = self.observation_type.observe()
        for i in self.opp_idxs:
            actions[i] = self.opps[opp_idx].drive(obs)

        return self.env.step(actions)

    def update_opponent(self, opp, idx):
        """Update opponent policy"""
        self.opps[idx] = opp

