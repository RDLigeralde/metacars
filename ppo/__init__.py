import gymnasium as gym

gym.register(
    id="f1tenth-v0-dr",
    entry_point="ppo.rl_env:F110EnvDR",
)