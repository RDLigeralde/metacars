import gymnasium as gym
import numpy as np

class MultiStepWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_steps: int):
        """
        Gym wrapper to repeat the same action
        for n_steps in the environment.

        Args:
            env (gym.Env): The environment to wrap.
            n_steps (int): The number of steps to repeat the action.
        """
        super().__init__(env)
        self.n_steps = n_steps
        self.basespace = env.action_space
        self.action_space = gym.spaces.Box(
            low=np.repeat(self.basespace.low, self.n_steps, axis=0),
            high=np.repeat(self.basespace.high, self.n_steps, axis=0),
            dtype=self.basespace.dtype
        )

    def step(self, actions: np.ndarray):
        action_seq = actions.reshape((self.n_steps, self.basespace.shape[0]))
        total_reward = 0.0
        for action in action_seq:
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info