import numpy as np

class OpponentDriver:
    def __init__(self, **kwargs):
        """Wrapper class for opponent policies"""
        pass

    def __call__(self, obs):
        """Drive the car: implemented in subclasses"""
        return np.zeros(2)