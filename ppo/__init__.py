import gymnasium as gym

gym.register(
    id="f1tenth-v0-legacy",
    entry_point="rl_env:F110EnvLegacy",
)
# gym.register(
#     id="f1tenth-v0",
#     entry_point="f1tenth_gym.envs:F110Env",
# )