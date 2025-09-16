import gymnasium as gym

gym.register(
    id="f1tenth-v0-legacy",
    entry_point="rl_env:F110EnvLegacy",
)