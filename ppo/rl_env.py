from f1tenth_gym.envs import F110Env
import gymnasium as gym
import numpy as np

from typing import List
import os

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
        self.observation_space = env.unwrapped.observation_space

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

class F110EnvDR(F110Env):
    def __init__(
        self,
        config: dict,
        render_mode: str = None,
        reward_idxs: List[int]=None,
        **kwargs         
    ):
        """
        F110Env with support for domain randomization

        Enabled by setting 'param': {'min': val, 'max': val} in config.yml
        instead of static values
        """
        self.config_input = config
        self.params_input = config['params']

        if os.path.exists(config['map']) and os.path.isdir(config['map']):
            tracks = [d for d in os.listdir(config['map']) if os.path.isdir(os.path.join(config['map'], d))]
        else:
            tracks = []

        if len(tracks) > 0:
            self.use_trackgen = True
            self.tracks = tracks
        else:
            self.use_trackgen = False
            self.tracks = None

        config = self._sample_dict(self.config_input)
        config['params'] = self._sample_dict(self.params_input)
        super().__init__(config, render_mode, reward_idxs, **kwargs)

    def _sample_dict(self, params: dict):
        """Sample parameters for domain randomization"""
        pcopy = params.copy()
        for key, val in pcopy.items():
            if isinstance(val, dict) and 'min' in val and 'max' in val: # sample numeric
                pcopy[key] = np.random.uniform(val['min'], val['max'])
            elif self.use_trackgen and key == 'map': # sample track
                pcopy[key] = os.path.join(self.config_input['map'], np.random.choice(self.tracks))
        return pcopy
    
    def reset(self, seed=None, options=None):
        """resets agents, randomizes params"""
        if hasattr(self, 'config_input') and hasattr(self, 'params_input'):
            config = self._sample_dict(self.config_input)
            config['params'] = self._sample_dict(self.params_input)
            self.configure({'params': config['params']})

            for k, v in config.items():
                if k != 'params' and hasattr(self, k):
                    setattr(self, k, v)
        
        if self.use_trackgen:
            self.update_map(config['map'])
        
        return super().reset(seed=seed, options=options)
