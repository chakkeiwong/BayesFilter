"""
Exact Daum-Huang Particle Flow Filters.

Based on Li & Coates (2017): "Particle Filtering With Invertible Particle Flow"
IEEE Transactions on Signal Processing, Vol. 65, No. 15, August 2017.

Implements:
- EDHFlow: Pure EDH flow filter (uniform weights)
- EDHParticleFilter: PF-PF with EDH flow (Algorithm 2, importance weights)
"""

import numpy as np
from typing import Optional, Literal
from numpy.random import Generator, default_rng

from .base import FilterResult
from .kalman import UnscentedKalmanFilter
from ..models.base import StateSpaceModel
from ..utils.resampling import (
    systematic_resample,
    stratified_resample,
    multinomial_resample,
    residual_resample,
    effective_sample_size,
    normalize_log_weights,
)


def generate_lambda_schedule(
    n_steps: int = 29,
    step_ratio: float = 1.2,
) -> np.ndarray:
    """
    Generate exponentially spaced pseudo-time steps for particle flow.
    
    As recommended in Daum & Huang, uses exponentially increasing step sizes
    with ratio 1.2 and 29 steps.
    
    Args:
        n_steps: Number of flow steps (default 29)
        step_ratio: Ratio between consecutive steps (default 1.2)
        
    Returns:
        step_sizes: [n_steps] array of step sizes summing to 1
    """
    if n_steps == 1:
        return np.array([1.0])
    
    r = step_ratio
    eps_1 = (1 - r) / (1 - r ** n_steps)
    step_sizes = eps_1 * (r ** np.arange(n_steps))
    step_sizes = step_sizes / step_sizes.sum()
    
    return step_sizes


class EDHFlow:
    """
    Pure Exact Daum-Huang flow filter (without importance weights).
    
    Uses uniform weights after flow. Suitable for linear or near-linear
    observation models where the flow provides accurate posterior approximation.
    
    For theoretical consistency with proper convergence guarantees, use
    EDHParticleFilter instead.
    """
    
    def __init__(
        self,
        n_particles: int = 500,
        n_flow_steps: int = 29,
        flow_step_ratio: float = 1.2,
        redraw: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_particles: Number of particles
            n_flow_steps: Number of EDH flow steps (default 29)
            flow_step_ratio: Ratio between consecutive step sizes (default 1.2)
            redraw: If True, redraw particles from N(mean, P_updated) at each step
                    (as done in Li & Coates MATLAB code)
            seed: Random seed
        """
        self.n_particles = n_particles
        self.n_flow_steps = n_flow_steps
        self.flow_step_ratio = flow_step_ratio
        self.redraw = redraw
        self.seed = seed
        
        self.step_sizes = generate_lambda_schedule(n_flow_steps, flow_step_ratio)
        self.ukf = UnscentedKalmanFilter()
    
    def _edh_flow_step(
        self,
        particles: np.ndarray,
        eta_bar: np.ndarray,
        eta_bar_0: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        H: np.ndarray,
        lam: float,
        eps: float,
        model: StateSpaceModel,
    ) -> tuple:
        """
        Single EDH flow step for all particles.
        
        Implements equations (10) and (11) from Li & Coates (2017).
        
        Args:
            particles: [N, nx] current particle positions
            eta_bar: [nx] current linearization point (at λ)
            eta_bar_0: [nx] initial mean at λ=0
            P: [nx, nx] predictive covariance
            R: [ny, ny] observation covariance
            z: [ny] observation
            H: [ny, nx] observation Jacobian at eta_bar
            lam: current pseudo-time λ ∈ [0, 1]
            eps: step size ε
            model: StateSpaceModel
            
        Returns:
            particles_new: [N, nx] updated particles
            eta_bar_new: [nx] updated linearization point
            A: [nx, nx] flow matrix for diagnostics
        """
        nx = P.shape[0]
        I = np.eye(nx)
        
        # Equation (10): A(λ) = -0.5 * P @ H.T @ (λ H P H.T + R)^{-1} @ H
        S = lam * H @ P @ H.T + R
        S = 0.5 * (S + S.T)
        
        S_inv_H = np.linalg.solve(S, H)
        A = -0.5 * P @ H.T @ S_inv_H
        
        # Equation (11): b(λ) = (I + 2λA)[(I + λA) P H.T R^{-1} (z - e) + A η̄₀]
        # where e(λ) = h(η̄_λ) - H(λ) η̄_λ
        h_eta_bar = model.obs_mean(eta_bar[np.newaxis, :])[0]
        e = h_eta_bar - H @ eta_bar
        
        R_inv = np.linalg.inv(R)
        R_inv_residual = R_inv @ (z - e)
        
        IplusLamA = I + lam * A
        Iplus2LamA = I + 2 * lam * A
        
        term1 = IplusLamA @ P @ H.T @ R_inv_residual
        term2 = A @ eta_bar_0
        b = Iplus2LamA @ (term1 + term2)
        
        # Update: η += ε(Aη + b)
        drift = particles @ A.T + b
        particles_new = particles + eps * drift
        
        # Update eta_bar using its own drift (matches MATLAB: xp_auxiliary = xp_auxiliary + step*slope.auxiliary)
        # NOT by taking mean of flowed particles
        eta_bar_drift = A @ eta_bar + b
        eta_bar_new = eta_bar + eps * eta_bar_drift
        
        return particles_new, eta_bar_new, A
    
    def _edh_flow(
        self,
        particles: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        model: StateSpaceModel,
        mu_0: np.ndarray = None,
        return_diagnostics: bool = False,
    ):
        """
        Full EDH flow from prior to posterior.
        
        Args:
            particles: [N, nx] particles from predictive distribution
            P: [nx, nx] predictive covariance
            z: [ny] observation
            model: StateSpaceModel
            mu_0: [nx] Initial mean for flow (from UKF prediction). If None, uses particle mean.
            return_diagnostics: If True, return flow diagnostics
            
        Returns:
            particles_flowed: [N, nx] particles after flow
            diagnostics: dict (if return_diagnostics=True)
        """
        R = model.obs_cov
        
        # Use provided mu_0 (from UKF) or fall back to particle mean
        # MATLAB uses vg.mu_0 = propagate(vg.M, no_noise) where vg.M is KF mean
        if mu_0 is None:
            eta_bar_0 = np.mean(particles, axis=0)
        else:
            eta_bar_0 = mu_0.copy()
        
        eta_bar = eta_bar_0.copy()
        
        if return_diagnostics:
            flow_magnitudes = []
            jacobian_conds = []
        
        lam = 0.0
        for eps in self.step_sizes:
            lam += eps
            H = model.obs_jacobian(eta_bar)
            
            particles_old = particles.copy()
            particles, eta_bar, A = self._edh_flow_step(
                particles, eta_bar, eta_bar_0, P, R, z, H, lam, eps, model
            )
            
            if return_diagnostics:
                displacement = np.linalg.norm(particles - particles_old, axis=1).mean()
                flow_magnitudes.append(displacement)
                
                # Jacobian of the mapping is I + εA
                nx = A.shape[0]
                jac = np.eye(nx) + eps * A
                cond = np.linalg.cond(jac)
                jacobian_conds.append(cond)
        
        if return_diagnostics:
            diagnostics = {
                'flow_magnitudes': np.array(flow_magnitudes),
                'jacobian_conds': np.array(jacobian_conds),
            }
            return particles, diagnostics
        
        return particles
    
    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
        Q_override: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
        rng: Optional[Generator] = None,
    ) -> FilterResult:
        """
        Run pure EDH flow filter (uniform weights).
        
        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations
            Q_override: If provided, use this Q for flow computation instead of model.dynamics_cov
            return_diagnostics: If True, store flow diagnostics
            rng: Optional random generator
            
        Returns:
            FilterResult
        """
        if rng is None:
            rng = default_rng(self.seed)
        
        T = observations.shape[0]
        N = self.n_particles
        nx = model.state_dim
        
        particles = model.sample_initial(N, rng)
        
        m_ukf = model.initial_mean.copy()
        P_ukf = model.initial_cov.copy()
        
        # Use Q_override for UKF if provided
        if Q_override is not None:
            from copy import copy
            model_ukf = copy(model)
            model_ukf.dynamics_cov = Q_override
        else:
            model_ukf = model
        
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))
        
        if return_diagnostics:
            all_diagnostics = []
        
        means[0] = particles.mean(axis=0)
        covariances[0] = np.cov(particles.T) if nx > 1 else np.array([[particles.var()]])
        
        for t in range(T):
            z = observations[t]
            
            # Redraw particles from N(mean, P_updated) at start of each step (except first)
            # This matches Li & Coates MATLAB implementation
            if self.redraw and t > 0:
                P_chol = np.linalg.cholesky(P_ukf + 1e-6 * np.eye(nx))
                particles = means[t] + (rng.standard_normal((N, nx)) @ P_chol.T)
            
            m_pred, P_pred = self.ukf.predict(m_ukf, P_ukf, model_ukf)
            particles_prior = model.sample_dynamics(particles, rng)
            
            # Use UKF predicted mean as mu_0 for flow (matches MATLAB: vg.mu_0 = propagate(vg.M, no_noise))
            if return_diagnostics:
                particles_posterior, diag = self._edh_flow(
                    particles_prior, P_pred, z, model, mu_0=m_pred, return_diagnostics=True
                )
                all_diagnostics.append(diag)
            else:
                particles_posterior = self._edh_flow(
                    particles_prior, P_pred, z, model, mu_0=m_pred
                )
            
            # Uniform weights
            means[t + 1] = particles_posterior.mean(axis=0)
            covariances[t + 1] = np.cov(particles_posterior.T) if nx > 1 else np.array([[particles_posterior.var()]])
            
            m_ukf, P_ukf = self.ukf.update(m_pred, P_pred, z, model_ukf)
            particles = particles_posterior
        
        result = FilterResult(
            means=means,
            covariances=covariances,
            ess=np.full(T, float(N)),
        )
        
        if return_diagnostics:
            result.diagnostics = all_diagnostics
        
        return result


class EDHParticleFilter(EDHFlow):
    """
    Particle Flow Particle Filter with Exact Daum-Huang flow.
    
    Implements Algorithm 2 from Li & Coates (2017).
    Uses global linearization at the particle mean with importance weight correction.
    
    The EDH flow migrates particles from the prior to the posterior,
    then importance weights correct for any remaining discrepancy.
    This provides theoretical consistency but may have lower ESS than pure EDH.
    """
    
    def __init__(
        self,
        n_particles: int = 500,
        n_flow_steps: int = 29,
        flow_step_ratio: float = 1.2,
        resample_method: Literal["systematic", "stratified", "multinomial", "residual"] = "systematic",
        resample_criterion: Literal["always", "ess", "never"] = "ess",
        ess_threshold: float = 0.5,
        redraw: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_particles: Number of particles
            n_flow_steps: Number of EDH flow steps (default 29)
            flow_step_ratio: Ratio between consecutive step sizes (default 1.2)
            resample_method: Resampling algorithm
            resample_criterion: When to resample
            ess_threshold: ESS threshold as fraction of N
            redraw: If True, redraw particles from N(mean, P_updated) at each step
            seed: Random seed
        """
        super().__init__(n_particles, n_flow_steps, flow_step_ratio, redraw, seed)
        self.resample_method = resample_method
        self.resample_criterion = resample_criterion
        self.ess_threshold = ess_threshold
    
    def _get_resampler(self):
        """Get resampling function."""
        resamplers = {
            "systematic": systematic_resample,
            "stratified": stratified_resample,
            "multinomial": multinomial_resample,
            "residual": residual_resample,
        }
        return resamplers[self.resample_method]
    
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
    
    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
        Q_override: Optional[np.ndarray] = None,
        return_particles: bool = False,
        return_diagnostics: bool = False,
        rng: Optional[Generator] = None,
    ) -> FilterResult:
        """
        Run EDH Particle Flow Particle Filter.
        
        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations
            Q_override: If provided, use this Q for flow computation instead of model.dynamics_cov
            return_particles: If True, store particle history
            return_diagnostics: If True, store flow diagnostics
            rng: Optional random generator
            
        Returns:
            FilterResult
        """
        if rng is None:
            rng = default_rng(self.seed)
        
        T = observations.shape[0]
        N = self.n_particles
        nx = model.state_dim
        
        resample_fn = self._get_resampler()
        
        particles = model.sample_initial(N, rng)
        log_weights = -np.log(N) * np.ones(N)
        
        m_ukf = model.initial_mean.copy()
        P_ukf = model.initial_cov.copy()
        
        # Use Q_override for UKF if provided
        if Q_override is not None:
            from copy import copy
            model_ukf = copy(model)
            model_ukf.dynamics_cov = Q_override
        else:
            model_ukf = model
        
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
        
        if return_diagnostics:
            all_diagnostics = []
        
        weights = np.exp(log_weights)
        means[0] = np.sum(weights[:, np.newaxis] * particles, axis=0)
        diff = particles - means[0]
        covariances[0] = np.einsum('n,ni,nj->ij', weights, diff, diff)
        covariances[0] = 0.5 * (covariances[0] + covariances[0].T)
        
        for t in range(T):
            z = observations[t]
            
            # Redraw particles from N(mean, P_updated) at start of each step (except first)
            if self.redraw and t > 0:
                P_chol = np.linalg.cholesky(P_ukf + 1e-6 * np.eye(nx))
                particles = means[t] + (rng.standard_normal((N, nx)) @ P_chol.T)
                # Reset weights after redraw
                log_weights = -np.log(N) * np.ones(N)
                weights = np.ones(N) / N
            
            m_pred, P_pred = self.ukf.predict(m_ukf, P_ukf, model_ukf)
            particles_prior = model.sample_dynamics(particles, rng)
            
            # Use UKF predicted mean as mu_0 for flow
            if return_diagnostics:
                particles_posterior, diag = self._edh_flow(
                    particles_prior, P_pred, z, model, mu_0=m_pred, return_diagnostics=True
                )
                all_diagnostics.append(diag)
            else:
                particles_posterior = self._edh_flow(particles_prior, P_pred, z, model, mu_0=m_pred)
            
            # Weight Update (Equation 37)
            log_p_prior = model.dynamics_log_prob(particles_prior, particles)
            log_p_posterior_trans = model.dynamics_log_prob(particles_posterior, particles)
            log_p_obs = model.observation_log_prob(particles_posterior, z)
            
            log_w_increment = log_p_posterior_trans + log_p_obs - log_p_prior
            log_weights = log_weights + log_w_increment
            
            log_Z_t = np.max(log_weights) + np.log(np.sum(np.exp(log_weights - np.max(log_weights))))
            log_likelihood_increments[t] = log_Z_t
            
            weights, _ = normalize_log_weights(log_weights)
            log_weights = np.log(weights + 1e-300)
            
            ess = effective_sample_size(weights)
            ess_history[t] = ess
            
            means[t + 1] = np.sum(weights[:, np.newaxis] * particles_posterior, axis=0)
            diff = particles_posterior - means[t + 1]
            covariances[t + 1] = np.einsum('n,ni,nj->ij', weights, diff, diff)
            covariances[t + 1] = 0.5 * (covariances[t + 1] + covariances[t + 1].T)
            
            m_ukf, P_ukf = self.ukf.update(m_pred, P_pred, z, model_ukf)
            m_ukf = means[t + 1].copy()
            
            do_resample = self._should_resample(ess, N)
            resampled_history[t] = do_resample
            
            if do_resample:
                indices = resample_fn(weights, rng)
                particles_posterior = particles_posterior[indices]
                log_weights = -np.log(N) * np.ones(N)
                weights = np.ones(N) / N
            
            particles = particles_posterior
            
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
        
        if return_diagnostics:
            result.diagnostics = all_diagnostics
        
        return result
