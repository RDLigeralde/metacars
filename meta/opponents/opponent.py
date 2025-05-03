import torch.nn as nn
import numpy as np
import torch

from .maf import MAF

class OpponentDriver:
    def __init__(self, **kwargs):
        """Wrapper class for opponent policies"""
        pass

    def __call__(self, obs, **kwargs):
        """Drive the car: implemented in subclasses"""
        return np.zeros(2)
    
class FZeroOpponent(OpponentDriver):
    def __init__(self, model_state_path, cost_weights_path, device='cpu'):
        """
        Initialize the opponent policy with pre-trained MAF weights and cost weights.
        
        Args:
            model_state_path: Path to the model_state_cost_X.pt file
            cost_weights_path: Path to the npz file containing cost weights
            device: torch device to use
        """
        self.device = torch.device(device)
        
        self.N_BLOCKS = 5
        self.INPUT_SIZE = 6
        self.HIDDEN_SIZE = 100
        self.N_HIDDEN = 1
        self.COND_LABEL_SIZE = 101
        self.ACTIVATION_FCN = 'relu'
        self.INPUT_ORDER = 'sequential'
        self.BATCH_NORM = False
        
        self.model = MAF(
            self.N_BLOCKS, 
            self.INPUT_SIZE, 
            self.HIDDEN_SIZE, 
            self.N_HIDDEN, 
            self.COND_LABEL_SIZE, 
            self.ACTIVATION_FCN, 
            self.INPUT_ORDER, 
            self.BATCH_NORM
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_state_path, map_location=self.device))
        self.model.eval()
        
        # Load cost weights
        cost_data = np.load(cost_weights_path)
        self.cost_weights = cost_data['cost_weights']
        
        # Constants from original code
        self.FLOW_S_SHIFT = 5.0
        self.T_SCALE = 0.75
        self.THETA_SCALE = 0.25
        self.V_SCALE = 2.0
        self.NUM_FLOW_SAMPLES = 50
        
    def unnormalize_flow(self, grid):
        grid = grid.copy()
        grid[:, 0] += self.FLOW_S_SHIFT
        grid[:, 1] *= self.T_SCALE
        grid[:, 2] *= self.THETA_SCALE
        grid[:, 3:] *= self.V_SCALE
        return grid
    
    def normalize_flow(self, grid):
        grid = grid.copy()
        grid[:, 0] -= self.FLOW_S_SHIFT
        grid[:, 1] /= self.T_SCALE
        grid[:, 2] /= self.THETA_SCALE
        grid[:, 3:] /= self.V_SCALE
        return grid
    
    @torch.inference_mode()
    def sample_flow(self, observation, n_samples=None):
        """
        Sample flow predictions given an observation.
        
        Args:
            observation: numpy array of shape (COND_LABEL_SIZE,) containing
                        100 LiDAR points + 1 velocity value
            n_samples: number of samples to generate (default: NUM_FLOW_SAMPLES)
        
        Returns:
            numpy array of shape (n_samples, 6) with unnormalized flow predictions
        """
        if n_samples is None:
            n_samples = self.NUM_FLOW_SAMPLES
            
        # Convert observation to torch tensor
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)
        
        # Sample from base distribution
        u = self.model.base_dist.sample((n_samples, 1)).squeeze()
        
        # Generate samples
        samples, _ = self.model.inverse(u, obs_tensor)
        
        # Convert back to numpy and unnormalize
        samples_np = samples.cpu().numpy().astype(np.float64)
        return self.unnormalize_flow(samples_np)
    
    def get_cost_weights(self):
        """Return the cost weights for this policy."""
        return self.cost_weights
    
    def __call__(self, observation, n_samples=None):
        """
        Make the class callable. Equivalent to sample_flow.
        
        Args:
            observation: numpy array of shape (COND_LABEL_SIZE,)
            n_samples: number of samples to generate
        
        Returns:
            numpy array of shape (n_samples, 6) with flow predictions
        """
        return self.sample_flow(observation, n_samples)

