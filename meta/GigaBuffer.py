from stable_baselines3.common.buffers import DictReplayBuffer
import torch
import numpy as np
from collections import namedtuple

GigaSamples = namedtuple(
    "ExtendedReplayBufferSamples",
    [
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "dones",
        "prev_actions",
        "prev_rewards",
        "prev_obs",
        "historical_actions",
        "historical_rewards",
        "historical_obs",
    ]
)


class GigaBuffer(DictReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device=device, **kwargs)

        self.prev_actions = np.zeros((buffer_size, *action_space.shape), dtype=np.float32)
        self.prev_rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.prev_obs = {key: np.zeros((buffer_size, *space.shape), dtype=np.float32)
                         for key, space in observation_space.spaces.items()}

        H = 10 

        self.historical_actions = np.zeros((buffer_size, H, *action_space.shape), dtype=np.float32)
        self.historical_rewards = np.zeros((buffer_size, H), dtype=np.float32)
        self.historical_obs = {
            key: np.zeros((buffer_size, H, *space.shape), dtype=np.float32)
            for key, space in observation_space.spaces.items()
        }

    def size_rb(self, task_id=None):
        return self.size


    def add(self, obs, next_obs, action, reward, done, infos,
            prev_action, prev_obs, prev_reward,
            historical_action, historical_obs, historical_reward):

        # Call parent to add standard fields
        super().add(obs, next_obs, action, reward, done, infos)

        pos = (self.pos - 1) % self.buffer_size

        self.prev_actions[pos] = prev_action
        self.prev_rewards[pos] = prev_reward
        for key in self.prev_obs:
            self.prev_obs[key][pos] = prev_obs[key]

        H = 10

        if infos is not None:
            self.prev_actions[pos] = infos.get("prev_action", np.zeros_like(action))
            self.prev_rewards[pos] = infos.get("prev_reward", 0.0)
            for key in self.prev_obs:
                self.prev_obs[key][pos] = infos.get("prev_obs", obs).get(key, obs[key])

            self.historical_actions[pos] = infos.get("historical_actions", np.zeros((H, *action.shape)))
            self.historical_rewards[pos] = infos.get("historical_rewards", np.zeros((H,)))
            hist_obs = infos.get("historical_obs", {k: np.zeros((H, *v.shape)) for k, v in obs.items()})
            for key in self.historical_obs:
                self.historical_obs[key][pos] = hist_obs[key]


    def sample(self, batch_size, env=None):
        base_sample = super().sample(batch_size, env)


        indices = np.random.randint(0, self.buffer_size, size=batch_size)

        return GigaSamples(
        observations=base_sample.observations,
        next_observations=base_sample.next_observations,
        actions=base_sample.actions,
        rewards=base_sample.rewards,
        dones=base_sample.dones,
        prev_actions=torch.tensor(self.prev_actions[indices]).to(self.device),
        prev_rewards=torch.tensor(self.prev_rewards[indices]).to(self.device),
        prev_obs={key: torch.tensor(self.prev_obs[key][indices]).to(self.device)
                  for key in self.prev_obs},
        historical_actions=torch.tensor(self.historical_actions[indices]).to(self.device),
        historical_rewards=torch.tensor(self.historical_rewards[indices]).to(self.device),
        historical_obs={key: torch.tensor(self.historical_obs[key][indices]).to(self.device)
                        for key in self.historical_obs},
        )

        return batch