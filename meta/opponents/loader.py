from meta.opponents.opponent import OpponentDriver
from stable_baselines3 import PPO
import numpy as np

from typing import Tuple
import glob
import os

class PolicyWrapper(OpponentDriver):
    def __init__(self, policy: PPO):
        """
        Wraps stochastic policies in OpponentDriver interface

        Args:
            policy (PPO): Policy with normalized action space (scaling handled in F110MultiViewer)
        """
        super().__init__()
        self.policy = policy

    def __call__(self, obs, **kwargs) -> np.ndarray:
        action, _ = self.policy.predict(obs, deterministic=False)
        return action

class PolicyLoader:
    def __init__(
        self,
        opp_dir: str,
        entropy_min: float,
        entropy_max: float,
        seed: int = None
    ):
        """
        Randomly samples opponent policy + temperature from a directory of policies
        It may be worth considering a curriculum where faster policies are accessed after slower ones

        Args:
            opp_dir (str): Directory containing opponent policies in format opp_{vmax}.zip
            entropy_min (float): Minimum entropy coefficient
            entropy_max (float): Maximum entropy coefficient
        """
        self.opp_dir = opp_dir
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.policies = {} # name -> (policy, vmax)

        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self._load_policies()

    def sample(self) -> Tuple[PolicyWrapper, float]:
        name = np.random.choice(list(self.policies.keys()))
        policy, vmax = self.policies[name]
        entropy = self.rng.uniform(self.entropy_min, self.entropy_max)
        policy.ent_coef = entropy
        return PolicyWrapper(policy), vmax

    def _load_policies(self) -> None:
        policy_files = glob.glob(f"{self.opp_dir}/*.zip")
        for pf in policy_files:
            name = os.path.basename(pf)
            try:
                name_base = os.path.splitext(name)[0]
                vmax = float(name_base.split('_')[1])
                policy = PPO.load(pf)
                self.policies[name_base] = policy, vmax
            except Exception as e:
                print(f"Error loading policy {pf}: {e}")




