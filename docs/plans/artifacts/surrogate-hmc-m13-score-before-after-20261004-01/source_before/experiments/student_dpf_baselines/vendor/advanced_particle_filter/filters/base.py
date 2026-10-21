"""
Filter base classes and result containers.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterResult:
    """
    Container for filter outputs.
    
    Stores filtered estimates and diagnostics from any filter type.
    
    Attributes:
        means: [T+1, nx] Filtered state means (m_0, m_1, ..., m_T)
        covariances: [T+1, nx, nx] Filtered state covariances (optional)
        
        # Particle filter specific
        particles: [T+1, N, nx] Particle history (optional)
        weights: [T+1, N] Weight history (optional)
        ess: [T] Effective sample size at each step (optional)
        
        # Likelihood
        log_likelihood: Total log marginal likelihood (optional)
        log_likelihood_increments: [T] Per-step log likelihood (optional)
        
        # Diagnostics
        resampled: [T] Boolean mask of resampling events (optional)
    """
    means: np.ndarray
    covariances: Optional[np.ndarray] = None
    
    # Particle filter outputs
    particles: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    ess: Optional[np.ndarray] = None
    
    # Likelihood
    log_likelihood: Optional[float] = None
    log_likelihood_increments: Optional[np.ndarray] = None
    
    # Diagnostics
    resampled: Optional[np.ndarray] = None
    
    @property
    def T(self) -> int:
        """Number of time steps (observations)."""
        return self.means.shape[0] - 1
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.means.shape[1]
    
    def rmse(self, true_states: np.ndarray) -> np.ndarray:
        """
        Compute per-timestep RMSE against true states.
        
        Args:
            true_states: [T+1, nx] True state trajectory
            
        Returns:
            rmse: [T+1] RMSE at each time step
        """
        squared_error = (self.means - true_states) ** 2
        return np.sqrt(np.mean(squared_error, axis=1))
    
    def mse(self, true_states: np.ndarray) -> np.ndarray:
        """
        Compute per-timestep MSE against true states.
        
        Args:
            true_states: [T+1, nx] True state trajectory
            
        Returns:
            mse: [T+1] MSE at each time step
        """
        squared_error = (self.means - true_states) ** 2
        return np.mean(squared_error, axis=1)
    
    def mean_rmse(self, true_states: np.ndarray) -> float:
        """
        Compute average RMSE over all time steps.
        
        Args:
            true_states: [T+1, nx] True state trajectory
            
        Returns:
            Average RMSE (scalar)
        """
        return np.mean(self.rmse(true_states))
    
    def position_rmse(
        self, 
        true_states: np.ndarray, 
        position_indices: Optional[list] = None
    ) -> np.ndarray:
        """
        Compute RMSE for position components only.
        
        Args:
            true_states: [T+1, nx] True state trajectory
            position_indices: List of indices for position components.
                              Default: [0, 1] for 2D position.
                              
        Returns:
            rmse: [T+1] Position RMSE at each time step
        """
        if position_indices is None:
            position_indices = [0, 1]
        
        pos_error = self.means[:, position_indices] - true_states[:, position_indices]
        return np.sqrt(np.sum(pos_error ** 2, axis=1))
    
    def average_ess(self) -> float:
        """Return average ESS if available."""
        if self.ess is None:
            return np.nan
        return np.mean(self.ess)
