"""
Localized Exact Daum-Huang Particle Flow Filters.

Based on Li & Coates (2017): "Particle Filtering With Invertible Particle Flow"
IEEE Transactions on Signal Processing, Vol. 65, No. 15, August 2017.

Implements:
- LEDHFlow: Pure LEDH flow filter (uniform weights)
- LEDHParticleFilter: PF-PF with LEDH flow (Algorithm 1, importance weights)

LEDH uses per-particle linearization AND per-particle covariance tracking,
which provides better performance for nonlinear observation models compared 
to EDH's global linearization.

Key difference from EDH:
- Each particle has its own predicted covariance P_all[i]
- Each particle has its own initial mean mu_0_all[i]  
- EKF predict/update is run separately for each particle
"""

import numpy as np
from typing import Optional, Literal
from numpy.random import Generator, default_rng

from .base import FilterResult
from .kalman import UnscentedKalmanFilter, ExtendedKalmanFilter
from .edh import generate_lambda_schedule
from ..models.base import StateSpaceModel
from ..utils.resampling import (
    systematic_resample,
    stratified_resample,
    multinomial_resample,
    residual_resample,
    effective_sample_size,
    normalize_log_weights,
)


class LEDHFlow:
    """
    Pure Localized Exact Daum-Huang flow filter (without importance weights).
    
    Uses per-particle linearization AND per-particle covariance tracking
    for better handling of nonlinear observations.
    Uses uniform weights after flow.
    
    For theoretical consistency with proper convergence guarantees, use
    LEDHParticleFilter instead.
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
            n_flow_steps: Number of flow steps (default 29)
            flow_step_ratio: Ratio between consecutive step sizes (default 1.2)
            redraw: If True, redraw particles from N(mean, P_updated) at each step
            seed: Random seed
        """
        self.n_particles = n_particles
        self.n_flow_steps = n_flow_steps
        self.flow_step_ratio = flow_step_ratio
        self.redraw = redraw
        self.seed = seed
        
        self.step_sizes = generate_lambda_schedule(n_flow_steps, flow_step_ratio)
    
    def _ekf_predict(
        self,
        m: np.ndarray,
        P: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        EKF prediction step.
        
        Args:
            m: [nx] Current mean (particle position)
            P: [nx, nx] Current covariance
            model: StateSpaceModel
            
        Returns:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
        """
        F = model.dynamics_jacobian(m)
        Q = model.dynamics_cov
        
        m_pred = model.dynamics_mean(m[np.newaxis, :])[0]
        P_pred = F @ P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        return m_pred, P_pred
    
    def _ekf_update(
        self,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        EKF update step.
        
        Args:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
            y: [ny] Observation
            model: StateSpaceModel
            
        Returns:
            m_upd: [nx] Updated mean
            P_upd: [nx, nx] Updated covariance
        """
        H = model.obs_jacobian(m_pred)
        R = model.obs_cov
        
        y_pred = model.obs_mean(m_pred[np.newaxis, :])[0]
        v = y - y_pred
        
        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)
        
        K = np.linalg.solve(S, H @ P_pred).T
        
        m_upd = m_pred + K @ v
        
        nx = len(m_pred)
        I = np.eye(nx)
        IKH = I - K @ H
        P_upd = IKH @ P_pred @ IKH.T + K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)
        
        # Regularize if needed
        min_eig = np.min(np.linalg.eigvalsh(P_upd))
        if min_eig < 1e-10:
            P_upd = P_upd + (1e-6 - min(min_eig, 0)) * I
        
        return m_upd, P_upd
    
    def _compute_flow_params_particle(
        self,
        eta_bar_i: np.ndarray,
        mu_0_i: np.ndarray,
        P_i: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        lam: float,
        model: StateSpaceModel,
    ) -> tuple:
        """
        Compute flow parameters A^i and b^i for a single particle.
        
        Uses per-particle covariance P_i and per-particle initial mean mu_0_i.
        
        Args:
            eta_bar_i: [nx] Auxiliary flow position for particle i (linearization point)
            mu_0_i: [nx] Initial mean for particle i at λ=0
            P_i: [nx, nx] Predicted covariance for particle i
            R: [ny, ny] Observation covariance
            z: [ny] Observation
            lam: current pseudo-time λ
            model: StateSpaceModel
            
        Returns:
            A_i: [nx, nx] flow matrix for particle i
            b_i: [nx] flow drift for particle i
        """
        nx = P_i.shape[0]
        I = np.eye(nx)
        
        # Linearize at auxiliary flow position η̄^i
        H_i = model.obs_jacobian(eta_bar_i)
        
        # Equation (13): A^i(λ) = -0.5 * P^i @ H^i.T @ (λ H^i P^i H^i.T + R)^{-1} @ H^i
        S_i = lam * H_i @ P_i @ H_i.T + R
        S_i = 0.5 * (S_i + S_i.T)
        
        S_i_inv_H_i = np.linalg.solve(S_i, H_i)
        A_i = -0.5 * P_i @ H_i.T @ S_i_inv_H_i
        
        # Equation (14): b^i(λ) = (I + 2λA^i)[(I + λA^i) P^i H^i.T R^{-1}(z - e^i) + A^i μ_0^i]
        # where e^i(λ) = h(η̄^i_λ) - H^i(λ) η̄^i_λ
        h_eta_bar_i = model.obs_mean(eta_bar_i[np.newaxis, :])[0]
        e_i = h_eta_bar_i - H_i @ eta_bar_i
        
        R_inv = np.linalg.inv(R)
        R_inv_residual = R_inv @ (z - e_i)
        
        IplusLamA = I + lam * A_i
        Iplus2LamA = I + 2 * lam * A_i
        
        term1 = IplusLamA @ P_i @ H_i.T @ R_inv_residual
        term2 = A_i @ mu_0_i  # Use per-particle mu_0
        b_i = Iplus2LamA @ (term1 + term2)
        
        return A_i, b_i
    
    def _ledh_flow_step(
        self,
        particles: np.ndarray,
        eta_bar_aux: np.ndarray,
        mu_0_all: np.ndarray,
        P_all: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        lam: float,
        eps: float,
        model: StateSpaceModel,
    ) -> tuple:
        """
        Single LEDH flow step for all particles with per-particle covariance.
        
        Each particle has its own flow parameters computed at its auxiliary position,
        using its own covariance P_all[i] and initial mean mu_0_all[i].
        
        Args:
            particles: [N, nx] current particle positions
            eta_bar_aux: [N, nx] auxiliary flow positions (linearization points)
            mu_0_all: [N, nx] per-particle initial means at λ=0
            P_all: [N, nx, nx] per-particle predicted covariances
            R: [ny, ny] observation covariance
            z: [ny] observation
            lam: current pseudo-time λ
            eps: step size ε
            model: StateSpaceModel
            
        Returns:
            particles_new: [N, nx] updated particles
            eta_bar_aux_new: [N, nx] updated auxiliary positions
            log_det_jacobians: [N] log |det(I + ε A^i)| for each particle
        """
        N, nx = particles.shape
        particles_new = np.zeros_like(particles)
        eta_bar_aux_new = np.zeros_like(eta_bar_aux)
        log_det_jacobians = np.zeros(N)
        
        I = np.eye(nx)
        
        for i in range(N):
            # Compute flow parameters at auxiliary position with per-particle covariance
            A_i, b_i = self._compute_flow_params_particle(
                eta_bar_aux[i], mu_0_all[i], P_all[i], R, z, lam, model
            )
            
            # Update auxiliary flow: η̄^i = η̄^i + ε(A^i η̄^i + b^i)
            drift_aux = A_i @ eta_bar_aux[i] + b_i
            eta_bar_aux_new[i] = eta_bar_aux[i] + eps * drift_aux
            
            # Update particle: η^i = η^i + ε(A^i η^i + b^i)
            drift = A_i @ particles[i] + b_i
            particles_new[i] = particles[i] + eps * drift
            
            # Log determinant of Jacobian: log |det(I + ε A^i)|
            jac = I + eps * A_i
            sign, logdet = np.linalg.slogdet(jac)
            log_det_jacobians[i] = logdet
        
        return particles_new, eta_bar_aux_new, log_det_jacobians
    
    def _ledh_flow(
        self,
        particles: np.ndarray,
        particles_deterministic: np.ndarray,
        mu_0_all: np.ndarray,
        P_all: np.ndarray,
        z: np.ndarray,
        model: StateSpaceModel,
        return_log_det: bool = False,
        return_diagnostics: bool = False,
    ):
        """
        Full LEDH flow from prior to posterior with per-particle covariance.
        
        Args:
            particles: [N, nx] particles from predictive distribution (with noise)
            particles_deterministic: [N, nx] deterministic predictions (no noise)
            mu_0_all: [N, nx] per-particle initial means
            P_all: [N, nx, nx] per-particle predicted covariances
            z: [ny] observation
            model: StateSpaceModel
            return_log_det: If True, return accumulated log determinants
            return_diagnostics: If True, return flow diagnostics
            
        Returns:
            particles_flowed: [N, nx] particles after flow
            total_log_det: [N] accumulated log |det| (if return_log_det=True)
            diagnostics: dict with flow_magnitudes and jacobian_conds (if return_diagnostics=True)
        """
        N, nx = particles.shape
        R = model.obs_cov
        I = np.eye(nx)
        
        # Initialize auxiliary flows at DETERMINISTIC predictions (no noise)
        # Algorithm 1, line 6: η̄^i = g_k(x^i_{k-1}, 0)
        eta_bar_aux = particles_deterministic.copy()
        
        # Accumulate log determinants
        total_log_det = np.zeros(N)
        
        if return_diagnostics:
            flow_magnitudes = []
            jacobian_conds = []
        
        lam = 0.0
        for eps in self.step_sizes:
            lam += eps
            
            particles_old = particles.copy()
            
            particles, eta_bar_aux, log_det_j = self._ledh_flow_step(
                particles, eta_bar_aux, mu_0_all, P_all, R, z, lam, eps, model
            )
            
            total_log_det += log_det_j
            
            # Normalize to prevent numerical overflow (as in MATLAB code)
            total_log_det = total_log_det - np.max(total_log_det)
            
            if return_diagnostics:
                # Track flow magnitude
                displacement = np.linalg.norm(particles - particles_old, axis=1).mean()
                flow_magnitudes.append(displacement)
                
                # Compute Jacobian condition number for a sample of particles
                # Jacobian of the mapping is I + εA_i
                sample_idx = N // 2
                H_i = model.obs_jacobian(eta_bar_aux[sample_idx])
                S_i = lam * H_i @ P_all[sample_idx] @ H_i.T + R
                S_i = 0.5 * (S_i + S_i.T)
                A_i = -0.5 * P_all[sample_idx] @ H_i.T @ np.linalg.solve(S_i, H_i)
                jac = I + eps * A_i
                cond = np.linalg.cond(jac)
                jacobian_conds.append(cond)
        
        if return_diagnostics:
            diagnostics = {
                'flow_magnitudes': np.array(flow_magnitudes),
                'jacobian_conds': np.array(jacobian_conds),
            }
            if return_log_det:
                return particles, total_log_det, diagnostics
            return particles, diagnostics
        
        if return_log_det:
            return particles, total_log_det
        
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
        Run pure LEDH flow filter with per-particle covariance tracking.
        
        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations
            Q_override: If provided, use this Q for flow computation
            return_diagnostics: If True, return flow diagnostics
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
        
        # Per-particle covariances: P_all[i] is the posterior covariance for particle i
        # Initialize all to the initial covariance
        P_all = np.tile(model.initial_cov, (N, 1, 1))  # [N, nx, nx]
        
        # Use Q_override for EKF if provided
        if Q_override is not None:
            from copy import deepcopy
            model_ekf = deepcopy(model)
            model_ekf.dynamics_cov = Q_override.copy()
            model_ekf._dynamics_cov_chol = np.linalg.cholesky(Q_override + 1e-6 * np.eye(nx))
        else:
            model_ekf = model
        
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))
        
        if return_diagnostics:
            all_diagnostics = []
        
        means[0] = particles.mean(axis=0)
        covariances[0] = np.cov(particles.T) if nx > 1 else np.array([[particles.var()]])
        
        for t in range(T):
            z = observations[t]
            
            # Redraw particles from N(mean, P_mean) at start of each step (except first)
            if self.redraw and t > 0:
                P_mean = np.mean(P_all, axis=0)
                P_chol = np.linalg.cholesky(P_mean + 1e-6 * np.eye(nx))
                particles = means[t] + (rng.standard_normal((N, nx)) @ P_chol.T)
                # Reset covariances
                P_all = np.tile(P_mean, (N, 1, 1))
            
            # Per-particle EKF prediction
            # MATLAB: [M_prior_all(:,i), PP_all(:,:,i)] = ekf_predict1(xp(:,i), PU_all(:,:,i), ...)
            mu_0_all = np.zeros((N, nx))  # Per-particle predicted means
            P_pred_all = np.zeros((N, nx, nx))  # Per-particle predicted covariances
            
            for i in range(N):
                mu_0_all[i], P_pred_all[i] = self._ekf_predict(particles[i], P_all[i], model_ekf)
            
            # Deterministic predictions: g_k(x^i_{k-1}, 0)
            particles_deterministic = model.dynamics_mean(particles)
            
            # Stochastic predictions: g_k(x^i_{k-1}, v_k)
            particles_prior = model.sample_dynamics(particles, rng)
            
            # LEDH flow with per-particle covariance
            if return_diagnostics:
                particles_posterior, diag = self._ledh_flow(
                    particles_prior, particles_deterministic, mu_0_all, P_pred_all, z, model,
                    return_diagnostics=True
                )
                all_diagnostics.append(diag)
            else:
                particles_posterior = self._ledh_flow(
                    particles_prior, particles_deterministic, mu_0_all, P_pred_all, z, model
                )
            
            # Per-particle EKF update
            # MATLAB: [~, PU_all(:,:,i)] = ukf_update1(M_prior_all(:,i), PP_all(:,:,i), z, ...)
            for i in range(N):
                _, P_all[i] = self._ekf_update(mu_0_all[i], P_pred_all[i], z, model_ekf)
            
            # Uniform weights
            means[t + 1] = particles_posterior.mean(axis=0)
            covariances[t + 1] = np.cov(particles_posterior.T) if nx > 1 else np.array([[particles_posterior.var()]])
            
            particles = particles_posterior
        
        result = FilterResult(
            means=means,
            covariances=covariances,
            ess=np.full(T, float(N)),
        )
        
        if return_diagnostics:
            result.diagnostics = all_diagnostics
        
        return result


class LEDHParticleFilter(LEDHFlow):
    """
    LEDH Particle Flow Particle Filter (Algorithm 1 in Li & Coates 2017).
    
    Uses LEDH flow to construct proposal distribution, then performs
    importance sampling with proper weight updates including Jacobian determinant.
    
    Key features:
    - Per-particle linearization
    - Per-particle covariance tracking (PP_all, mu_0_all)
    - Importance weights with Jacobian correction
    - Statistical consistency of particle filters
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
            n_flow_steps: Number of flow steps (default 29)
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
        rng: Optional[Generator] = None,
    ) -> FilterResult:
        """
        Run LEDH Particle Flow Particle Filter with per-particle covariance.
        
        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations
            Q_override: If provided, use this Q for flow computation
            return_particles: If True, store particle history
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
        
        # Per-particle covariances
        P_all = np.tile(model.initial_cov, (N, 1, 1))  # [N, nx, nx]
        
        # Use Q_override for EKF if provided
        if Q_override is not None:
            from copy import deepcopy
            model_ekf = deepcopy(model)
            model_ekf.dynamics_cov = Q_override.copy()
            model_ekf._dynamics_cov_chol = np.linalg.cholesky(Q_override + 1e-6 * np.eye(nx))
        else:
            model_ekf = model
        
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
        
        weights = np.exp(log_weights)
        means[0] = np.sum(weights[:, np.newaxis] * particles, axis=0)
        diff = particles - means[0]
        covariances[0] = np.einsum('n,ni,nj->ij', weights, diff, diff)
        covariances[0] = 0.5 * (covariances[0] + covariances[0].T)
        
        for t in range(T):
            z = observations[t]
            
            # Redraw particles from N(mean, P_mean) at start of each step (except first)
            if self.redraw and t > 0:
                P_mean = np.mean(P_all, axis=0)
                P_chol = np.linalg.cholesky(P_mean + 1e-6 * np.eye(nx))
                particles = means[t] + (rng.standard_normal((N, nx)) @ P_chol.T)
                # Reset weights after redraw
                log_weights = -np.log(N) * np.ones(N)
                weights = np.ones(N) / N
                # Reset covariances
                P_all = np.tile(P_mean, (N, 1, 1))
            
            # Per-particle EKF prediction
            mu_0_all = np.zeros((N, nx))
            P_pred_all = np.zeros((N, nx, nx))
            
            for i in range(N):
                mu_0_all[i], P_pred_all[i] = self._ekf_predict(particles[i], P_all[i], model_ekf)
            
            # Deterministic predictions: g_k(x^i_{k-1}, 0)
            particles_deterministic = model.dynamics_mean(particles)
            
            # Stochastic predictions: g_k(x^i_{k-1}, v_k)
            particles_prior = model.sample_dynamics(particles, rng)
            
            # LEDH flow with log determinants and per-particle covariance
            particles_posterior, total_log_det = self._ledh_flow(
                particles_prior, particles_deterministic, mu_0_all, P_pred_all, z, model,
                return_log_det=True
            )
            
            # Weight Update (Equation 20)
            # w_k^i ∝ p(η_1^i | x_{k-1}^i) p(z_k | η_1^i) |det(...)| / p(η_0^i | x_{k-1}^i) * w_{k-1}^i
            
            log_p_prior = model.dynamics_log_prob(particles_prior, particles)
            log_p_posterior_trans = model.dynamics_log_prob(particles_posterior, particles)
            log_p_obs = model.observation_log_prob(particles_posterior, z)
            
            # Include Jacobian determinant in weight update
            log_w_increment = log_p_posterior_trans + log_p_obs + total_log_det - log_p_prior
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
            
            # Per-particle EKF update
            for i in range(N):
                _, P_all[i] = self._ekf_update(mu_0_all[i], P_pred_all[i], z, model_ekf)
            
            do_resample = self._should_resample(ess, N)
            resampled_history[t] = do_resample
            
            if do_resample:
                indices = resample_fn(weights, rng)
                particles_posterior = particles_posterior[indices]
                P_all = P_all[indices]  # Also resample covariances!
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
        
        return result
