from gymnasium.envs.registration import register
import gymnasium as gym

register(
    id='F110Multi-v0',
    entry_point='meta.meta_env:F110Multi',
)

register(
    id='F110MultiView-v0',
    entry_point='meta.meta_env:F110MultiView',
    kwargs = {
        'env': lambda: gym.make('F110Multi-v0'),
    }
)