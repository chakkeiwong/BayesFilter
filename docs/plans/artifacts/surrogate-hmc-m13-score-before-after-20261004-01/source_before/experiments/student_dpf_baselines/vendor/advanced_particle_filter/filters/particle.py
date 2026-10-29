"""
Particle Filter implementations.

Bootstrap/SIR particle filter with various resampling schemes.
"""

import numpy as np
from typing import Optional, Literal
from numpy.random import Generator, default_rng

from .base import FilterResult
from ..models.base import StateSpaceModel
from ..utils.resampling import (
    systematic_resample,
    stratified_resample,
    multinomial_resample,
    residual_resample,
    effective_sample_size,
    normalize_log_weights,
)


class BootstrapParticleFilter:
    """
    Bootstrap Particle Filter (Sequential Importance Resampling).
    
    Uses the transition prior as proposal distribution.
    """
    
    def __init__(
        self,
        n_particles: int = 1000,
        resample_method: Literal["systematic", "stratified", "multinomial", "residual"] = "systematic",
        resample_criterion: Literal["always", "ess", "never"] = "ess",
        ess_threshold: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_particles: Number of particles
            resample_method: Resampling algorithm
            resample_criterion: When to resample ("always", "ess", "never")
            ess_threshold: ESS threshold as fraction of N (for "ess" criterion)
            seed: Random seed
        """
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.resample_criterion = resample_criterion
        self.ess_threshold = ess_threshold
        self.seed = seed
    
    def _get_resampler(self):
        """Get resampling function based on method."""
        resamplers = {
            "systematic": systematic_resample,
            "stratified": stratified_resample,
            "multinomial": multinomial_resample,
            "residual": residual_resample,
        }
        return resamplers[self.resample_method]
    
    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
        return_particles: bool = False,
        rng: Optional[Generator] = None,
    ) -> FilterResult:
        """
        Run bootstrap particle filter.
        
        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations (y_1, ..., y_T)
            return_particles: If True, store particle history
            rng: Optional random generator (uses self.seed if None)
            
        Returns:
            FilterResult
        """
        if rng is None:
            rng = default_rng(self.seed)
        
        T = observations.shape[0]
        N = self.n_particles
        nx = model.state_dim
        
        resample_fn = self._get_resampler()
        
        # Initialize particles from prior
        particles = model.sample_initial(N, rng)  # [N, nx]
        log_weights = -np.log(N) * np.ones(N)
        
        # Storage
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))
        ess_history = np.zeros(T)
        resampled_history = np.zeros(T, dtype=bool)
        log_likelihood_increments = np.zeros(T)
        
        if return_particles:
            particles_history = np.zeros((T + 1, N, nx))
            weights_history = np.zeros((T + 1, N))
            particles_history[0] = particles
            weights_history[0] = np.exp(log_weights)
        
        # Initial estimate
        weights = np.exp(log_weights)
        means[0] = np.sum(weights[:, np.newaxis] * particles, axis=0)
        diff = particles - means[0]
        covariances[0] = np.einsum('n,ni,nj->ij', weights, diff, diff)
        covariances[0] = 0.5 * (covariances[0] + covariances[0].T)
        
        for t in range(T):
            y = observations[t]
            
            # Propagate particles through dynamics
            particles = model.sample_dynamics(particles, rng)
            
            # Compute log likelihoods
            log_lik = model.observation_log_prob(particles, y)  # [N]
            
            # Update weights
            log_weights = log_weights + log_lik
            
            # Compute log marginal likelihood increment
            log_Z_t = np.max(log_weights) + np.log(np.sum(np.exp(log_weights - np.max(log_weights))))
            log_likelihood_increments[t] = log_Z_t
            
            # Normalize weights
            weights, _ = normalize_log_weights(log_weights)
            log_weights = np.log(weights + 1e-300)
            
            # Compute ESS
            ess = effective_sample_size(weights)
            ess_history[t] = ess
            
            # Compute filtered estimate
            means[t + 1] = np.sum(weights[:, np.newaxis] * particles, axis=0)
            diff = particles - means[t + 1]
            covariances[t + 1] = np.einsum('n,ni,nj->ij', weights, diff, diff)
            covariances[t + 1] = 0.5 * (covariances[t + 1] + covariances[t + 1].T)
            
            # Resample if needed
            do_resample = self._should_resample(ess, N)
            resampled_history[t] = do_resample
            
            if do_resample:
                indices = resample_fn(weights, rng)
                particles = particles[indices]
                log_weights = -np.log(N) * np.ones(N)
                weights = np.ones(N) / N
            
            if return_particles:
                particles_history[t + 1] = particles
                weights_history[t + 1] = weights
        
        result = FilterResult(
            means=means,
            covariances=covariances,
            ess=ess_history,
            log_likelihood=np.sum(log_likelihood_increments),
            log_likelihood_increments=log_likelihood_increments,
            resampled=resampled_history,
        )
        
        if return_particles:
            result.particles = particles_history
            result.weights = weights_history
        
        return result
    
    def _should_resample(self, ess: float, N: int) -> bool:
        """Determine if resampling should occur."""
        if self.resample_criterion == "always":
            return True
        elif self.resample_criterion == "never":
            return False
        elif self.resample_criterion == "ess":
            return ess < self.ess_threshold * N
        else:
            raise ValueError(f"Unknown resample criterion: {self.resample_criterion}")
