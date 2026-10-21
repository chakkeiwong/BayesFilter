"""
State Space Model base class.

Unified model specification for all filters (Kalman, particle, flow).
All functions operate on batched inputs where first axis is batch dimension.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from numpy.random import Generator


@dataclass
class StateSpaceModel:
    """
    State Space Model specification.
    
    Dynamics:    x_t = f(x_{t-1}) + v_t,  v_t ~ N(0, Q)
    Observation: y_t = h(x_t) + w_t,      w_t ~ N(0, R) or custom
    
    All mean functions take batched input [N, nx] and return [N, *].
    Jacobian functions take single input [nx] and return matrix.
    
    Attributes:
        state_dim: State dimension (nx)
        obs_dim: Observation dimension (ny)
        
        initial_mean: [nx] Initial state mean
        initial_cov: [nx, nx] Initial state covariance
        
        dynamics_mean: f(x), maps [N, nx] -> [N, nx]
        dynamics_cov: [nx, nx] Process noise covariance Q
        dynamics_jacobian: F(x) = df/dx, maps [nx] -> [nx, nx]
        
        obs_mean: h(x), maps [N, nx] -> [N, ny]
        obs_cov: [ny, ny] Observation noise covariance R (for Gaussian)
        obs_jacobian: H(x) = dh/dx, maps [nx] -> [ny, nx]
        
        obs_log_prob: Optional custom log p(y|x), maps ([N, nx], [ny]) -> [N]
        obs_sample: Optional custom sampler, maps ([nx], Generator) -> [ny]
    """
    # Dimensions
    state_dim: int
    obs_dim: int
    
    # Initial distribution
    initial_mean: np.ndarray
    initial_cov: np.ndarray
    
    # Dynamics
    dynamics_mean: Callable[[np.ndarray], np.ndarray]
    dynamics_cov: np.ndarray
    dynamics_jacobian: Callable[[np.ndarray], np.ndarray]
    
    # Observation
    obs_mean: Callable[[np.ndarray], np.ndarray]
    obs_cov: np.ndarray
    obs_jacobian: Callable[[np.ndarray], np.ndarray]
    
    # Optional: for non-Gaussian observations
    obs_log_prob: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    obs_sample: Optional[Callable[[np.ndarray, Generator], np.ndarray]] = None
    
    # Precomputed matrices (set by factory functions via __post_init__ or manually)
    _initial_cov_chol: Optional[np.ndarray] = field(default=None, repr=False)
    _dynamics_cov_chol: Optional[np.ndarray] = field(default=None, repr=False)
    _obs_cov_chol: Optional[np.ndarray] = field(default=None, repr=False)
    _obs_cov_inv: Optional[np.ndarray] = field(default=None, repr=False)
    _obs_cov_logdet: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Precompute Cholesky factors and inverses."""
        self._precompute_matrices()
    
    def _precompute_matrices(self):
        """Compute Cholesky factors and related quantities."""
        eps = 1e-8
        nx = self.state_dim
        ny = self.obs_dim
        
        # Initial covariance
        if self._initial_cov_chol is None:
            P0 = self.initial_cov + eps * np.eye(nx)
            self._initial_cov_chol = np.linalg.cholesky(P0)
        
        # Dynamics covariance
        if self._dynamics_cov_chol is None:
            Q = self.dynamics_cov + eps * np.eye(nx)
            self._dynamics_cov_chol = np.linalg.cholesky(Q)
        
        # Observation covariance (for Gaussian case)
        if self._obs_cov_chol is None:
            R = self.obs_cov + eps * np.eye(ny)
            self._obs_cov_chol = np.linalg.cholesky(R)
            self._obs_cov_inv = np.linalg.inv(R)
            self._obs_cov_logdet = np.linalg.slogdet(R)[1]
    
    # -------------------------------------------------------------------------
    # Sampling methods
    # -------------------------------------------------------------------------
    
    def sample_initial(self, n: int, rng: Generator) -> np.ndarray:
        """
        Sample n particles from initial distribution.
        
        Args:
            n: Number of samples
            rng: NumPy random generator
            
        Returns:
            x: [n, nx] samples
        """
        noise = rng.standard_normal((n, self.state_dim))
        return self.initial_mean + noise @ self._initial_cov_chol.T
    
    def sample_dynamics(self, x: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Sample x_t from p(x_t | x_{t-1}).
        
        Args:
            x: [N, nx] current states
            rng: NumPy random generator
            
        Returns:
            x_next: [N, nx] next states
        """
        n = x.shape[0]
        mean = self.dynamics_mean(x)  # [N, nx]
        noise = rng.standard_normal((n, self.state_dim))
        return mean + noise @ self._dynamics_cov_chol.T
    
    def sample_observation(self, x: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Sample y from p(y | x).
        
        Args:
            x: [nx] single state (not batched)
            rng: NumPy random generator
            
        Returns:
            y: [ny] observation
        """
        if self.obs_sample is not None:
            return self.obs_sample(x, rng)
        
        # Default: Gaussian observation
        mean = self.obs_mean(x[np.newaxis, :])[0]  # [ny]
        noise = rng.standard_normal(self.obs_dim)
        return mean + self._obs_cov_chol @ noise
    
    # -------------------------------------------------------------------------
    # Log-probability methods
    # -------------------------------------------------------------------------
    
    def observation_log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute log p(y | x) for all particles.
        
        Args:
            x: [N, nx] particles
            y: [ny] single observation
            
        Returns:
            log_prob: [N] log probabilities
        """
        if self.obs_log_prob is not None:
            return self.obs_log_prob(x, y)
        
        # Default: Gaussian observation
        return self._gaussian_obs_log_prob(x, y)
    
    def _gaussian_obs_log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gaussian observation log-likelihood.
        
        Args:
            x: [N, nx] particles
            y: [ny] observation
            
        Returns:
            log_prob: [N]
        """
        y_pred = self.obs_mean(x)  # [N, ny]
        residual = y - y_pred  # [N, ny]
        
        # Mahalanobis distance: residual @ R_inv @ residual.T
        # Efficient: solve L @ z = residual, then ||z||^2
        solved = np.linalg.solve(self._obs_cov_chol, residual.T)  # [ny, N]
        mahal_sq = np.sum(solved ** 2, axis=0)  # [N]
        
        log_prob = -0.5 * (
            self.obs_dim * np.log(2 * np.pi) + 
            self._obs_cov_logdet + 
            mahal_sq
        )
        return log_prob
    
    def dynamics_log_prob(self, x_next: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute log p(x_next | x) for all particles.
        
        Args:
            x_next: [N, nx] next states
            x: [N, nx] current states
            
        Returns:
            log_prob: [N]
        """
        mean = self.dynamics_mean(x)  # [N, nx]
        residual = x_next - mean  # [N, nx]
        
        solved = np.linalg.solve(self._dynamics_cov_chol, residual.T)  # [nx, N]
        mahal_sq = np.sum(solved ** 2, axis=0)  # [N]
        
        Q_logdet = 2.0 * np.sum(np.log(np.diag(self._dynamics_cov_chol)))
        
        log_prob = -0.5 * (
            self.state_dim * np.log(2 * np.pi) + 
            Q_logdet + 
            mahal_sq
        )
        return log_prob
    
    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------
    
    def simulate(self, T: int, rng: Generator) -> tuple:
        """
        Simulate a trajectory from the model.
        
        Args:
            T: Number of time steps
            rng: NumPy random generator
            
        Returns:
            states: [T+1, nx] states (x_0, ..., x_T)
            observations: [T, ny] observations (y_1, ..., y_T)
        """
        states = np.zeros((T + 1, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Initial state
        states[0] = self.sample_initial(1, rng)[0]
        
        for t in range(T):
            # Dynamics: x_t -> x_{t+1}
            x_curr = states[t:t+1]  # [1, nx]
            states[t + 1] = self.sample_dynamics(x_curr, rng)[0]
            
            # Observation: x_{t+1} -> y_{t+1}
            observations[t] = self.sample_observation(states[t + 1], rng)
        
        return states, observations
    
    def __repr__(self) -> str:
        return f"StateSpaceModel(nx={self.state_dim}, ny={self.obs_dim})"
