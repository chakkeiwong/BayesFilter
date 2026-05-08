"""
Stochastic Particle Flow Particle Filter (SPF-PF).

Combines:
- Dai & Daum (2022): Stochastic particle flow with stiffness mitigation
  as the PROPOSAL DISTRIBUTION
- Li & Coates (2017): Particle flow particle filter framework
  for IMPORTANCE WEIGHTING and resampling

The SPF-PF uses Dai's stochastic flow (with optimal β* homotopy and
diffusion-based stiffness mitigation) to migrate particles from prior
to posterior, then applies Li's importance weight correction to maintain
theoretical consistency of the particle filter.

Weight update (approximation for small diffusion):
  Since the SPF uses a GLOBAL flow (A, b linearized at ensemble mean),
  the deterministic Jacobian |det(M_det)| is the same for all particles
  and cancels under weight normalization. For small diffusion (tiny step
  sizes), the weight simplifies to Li (2017) Eq. 37:

    w^i_k ∝ p(η^i_1 | x^i_{k-1}) p(z_k | η^i_1) / p(η^i_0 | x^i_{k-1}) · w^i_{k-1}

  This approximation is excellent when step_sizes are tiny (e.g., 1e-9),
  since accumulated diffusion variance ≈ N_λ · dλ · Q_flow ≈ 0.

Architecture per time step:
  1. Propagate particles through dynamics (stochastic: η^i_0 = g(x^i_{k-1}, v))
  2. Compute ensemble P_pred and mean
  3. Solve β*(λ) BVP for optimal homotopy schedule
  4. Run SPF flow (expm split-step with diffusion) → η^i_1
  5. Compute importance weights via Eq. 37
  6. Resample if ESS < threshold
  7. Particles persist across timesteps (no redraw)
"""

import numpy as np
import warnings
from typing import Optional, Literal
from numpy.random import Generator, default_rng
from scipy.linalg import expm as scipy_expm

from .base import FilterResult
from .edh import generate_lambda_schedule
from .stochastic_pff import solve_optimal_beta, StochasticPFFlow
from ..models.base import StateSpaceModel
from ..utils.resampling import (
    systematic_resample,
    stratified_resample,
    multinomial_resample,
    residual_resample,
    effective_sample_size,
    normalize_log_weights,
)


class StochasticPFParticleFilter(StochasticPFFlow):
    """
    Stochastic Particle Flow Particle Filter (SPF-PF).

    Uses Dai (2022) stochastic particle flow as proposal distribution
    within Li (2017) particle flow particle filter framework.

    Inherits from StochasticPFFlow for the flow mechanics (β* schedule,
    drift computation, expm integration, diffusion). Adds importance
    weighting and resampling on top.

    Key differences from StochasticPFFlow:
    - Importance weights instead of uniform weights
    - Resampling when ESS drops
    - No redraw (particles persist, carry velocity/acceleration memory)
    - Ensemble covariance as P_pred (not UKF/EKF tracked)

    Key differences from LEDHParticleFilter:
    - Global flow (not per-particle) → computationally cheaper
    - Stiffness-mitigated drift via optimal β* homotopy
    - Diffusion noise for additional stability
    - No per-particle covariance tracking
    - Weight update uses Eq. 37 (no Jacobian computation needed)
    """

    def __init__(
        self,
        n_particles: int = 100,
        n_flow_steps: int = 29,
        flow_step_ratio: float = 1.2,
        Q_flow_mode: Literal["fixed", "adaptive"] = "adaptive",
        Q_flow_fixed: Optional[np.ndarray] = None,
        beta_schedule: Literal["linear", "optimal"] = "optimal",
        mu: float = 1e-7,
        bvp_method: str = "Radau",
        bvp_max_step: float = 1e-2,
        bvp_clip_ddot: float = 1e8,
        deterministic_dynamics: bool = False,
        integration_method: Literal["euler", "semi_implicit", "heun", "expm"] = "expm",
        resample_method: Literal[
            "systematic", "stratified", "multinomial", "residual"
        ] = "systematic",
        resample_criterion: Literal["always", "ess", "never"] = "ess",
        ess_threshold: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_particles: Number of particles
            n_flow_steps: Number of pseudo-time steps (default 29)
            flow_step_ratio: Ratio between consecutive step sizes (default 1.2)
            Q_flow_mode: "adaptive" for Q = M(λ)^{-1}, "fixed" for user-specified
            Q_flow_fixed: [nx, nx] Fixed diffusion matrix (when Q_flow_mode="fixed")
            beta_schedule: "linear" for β=λ, "optimal" for BVP solution
            mu: Weight for condition number penalty in BVP
            bvp_method: ODE solver for BVP
            bvp_max_step: Max step size for BVP ODE solver
            bvp_clip_ddot: Clamp on β̈ in BVP
            deterministic_dynamics: If True, propagate particles deterministically
            integration_method: SDE integration scheme ("expm" recommended)
            resample_method: Resampling algorithm
            resample_criterion: When to resample ("always", "ess", "never")
            ess_threshold: ESS threshold as fraction of N
            seed: Random seed
        """
        super().__init__(
            n_particles=n_particles,
            n_flow_steps=n_flow_steps,
            flow_step_ratio=flow_step_ratio,
            Q_flow_mode=Q_flow_mode,
            Q_flow_fixed=Q_flow_fixed,
            beta_schedule=beta_schedule,
            mu=mu,
            bvp_method=bvp_method,
            bvp_max_step=bvp_max_step,
            bvp_clip_ddot=bvp_clip_ddot,
            deterministic_dynamics=deterministic_dynamics,
            integration_method=integration_method,
            seed=seed,
        )
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
            raise ValueError(
                f"Unknown resample criterion: {self.resample_criterion}"
            )

    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
        return_particles: bool = False,
        return_diagnostics: bool = False,
        rng: Optional[Generator] = None,
    ) -> FilterResult:
        """
        Run the Stochastic Particle Flow Particle Filter.

        Architecture per time step:
          1. Propagate particles through dynamics (stochastic)
             - η^i_0 = g_k(x^i_{k-1}, v_k)  (with process noise)
          2. Compute ensemble P_pred and mean from propagated particles
          3. Solve β*(λ) BVP for optimal homotopy schedule
          4. Run SPF flow → η^i_1
          5. Weight update (Li 2017 Eq. 37, no Jacobian needed):
             w^i_k ∝ p(η^i_1 | x^i_{k-1}) p(z_k | η^i_1)
                      / p(η^i_0 | x^i_{k-1}) · w^i_{k-1}
          6. Resample if ESS < threshold
          7. Particles persist (no redraw)

        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations
            return_particles: If True, store particle history
            return_diagnostics: If True, store per-step flow diagnostics
            rng: Optional random generator

        Returns:
            FilterResult with means, covariances, ess, log_likelihood, etc.
        """
        if rng is None:
            rng = default_rng(self.seed)

        T = observations.shape[0]
        N = self.n_particles
        nx = model.state_dim

        resample_fn = self._get_resampler()

        # --- Initialize ---
        particles = model.sample_initial(N, rng)
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
            particles_history[0] = particles.copy()
            weights_history[0] = np.exp(log_weights)

        if return_diagnostics:
            all_diagnostics = []

        # Initial estimates
        weights = np.exp(log_weights)
        means[0] = np.sum(weights[:, np.newaxis] * particles, axis=0)
        diff = particles - means[0]
        covariances[0] = np.einsum("n,ni,nj->ij", weights, diff, diff)
        covariances[0] = 0.5 * (covariances[0] + covariances[0].T)

        # --- Main loop ---
        for t in range(T):
            z = observations[t]

            # ---- Step 1: Propagate through dynamics ----
            # Stochastic: η^i_0 = g_k(x^i_{k-1}, v_k)
            particles_prior = model.sample_dynamics(particles, rng)

            # ---- Step 2: Ensemble statistics for flow ----
            x_bar_pred = np.mean(particles_prior, axis=0)
            if nx > 1:
                P_pred = np.cov(particles_prior.T)
            else:
                P_pred = np.array([[particles_prior.var()]])
            # Symmetrize and regularize
            P_pred = 0.5 * (P_pred + P_pred.T)
            min_eig = np.min(np.linalg.eigvalsh(P_pred))
            if min_eig < 1e-10:
                P_pred = P_pred + (1e-8 - min(min_eig, 0)) * np.eye(nx)

            # ---- Step 3: Compute β*(λ) schedule ----
            lam_grid, beta_arr, bdot_arr = self._compute_beta_schedule(
                P_pred, x_bar_pred, model
            )

            # ---- Step 4: SPF flow ----
            if return_diagnostics:
                particles_posterior, diag = self._spf_flow(
                    particles_prior,
                    P_pred,
                    z,
                    model,
                    mu_0=x_bar_pred,
                    lam_grid=lam_grid,
                    beta_arr=beta_arr,
                    bdot_arr=bdot_arr,
                    rng=rng,
                    return_diagnostics=True,
                )
                all_diagnostics.append(diag)
            else:
                particles_posterior = self._spf_flow(
                    particles_prior,
                    P_pred,
                    z,
                    model,
                    mu_0=x_bar_pred,
                    lam_grid=lam_grid,
                    beta_arr=beta_arr,
                    bdot_arr=bdot_arr,
                    rng=rng,
                )

            # ---- Step 5: Weight update (Li 2017 Eq. 37) ----
            #
            # w^i_k ∝ p(η^i_1 | x^i_{k-1}) p(z_k | η^i_1)
            #          / p(η^i_0 | x^i_{k-1}) · w^i_{k-1}
            #
            # No Jacobian determinant needed: global flow → cancels
            # under normalization.
            #
            # p(η^i_1 | x^i_{k-1}) = transition density at flowed particle
            # p(η^i_0 | x^i_{k-1}) = transition density at prior sample
            # p(z_k | η^i_1) = observation likelihood at flowed particle

            log_p_eta0 = model.dynamics_log_prob(particles_prior, particles)
            log_p_eta1 = model.dynamics_log_prob(
                particles_posterior, particles
            )
            log_p_obs = model.observation_log_prob(particles_posterior, z)

            log_w_increment = log_p_eta1 + log_p_obs - log_p_eta0
            log_weights = log_weights + log_w_increment

            # Log-likelihood increment (for model comparison)
            log_max = np.max(log_weights)
            log_Z_t = log_max + np.log(
                np.sum(np.exp(log_weights - log_max))
            )
            log_likelihood_increments[t] = log_Z_t

            # Normalize weights
            weights, _ = normalize_log_weights(log_weights)
            log_weights = np.log(weights + 1e-300)

            # ESS
            ess = effective_sample_size(weights)
            ess_history[t] = ess

            # ---- Weighted estimates ----
            means[t + 1] = np.sum(
                weights[:, np.newaxis] * particles_posterior, axis=0
            )
            diff = particles_posterior - means[t + 1]
            covariances[t + 1] = np.einsum(
                "n,ni,nj->ij", weights, diff, diff
            )
            covariances[t + 1] = 0.5 * (
                covariances[t + 1] + covariances[t + 1].T
            )

            # ---- Step 6: Resample ----
            do_resample = self._should_resample(ess, N)
            resampled_history[t] = do_resample

            if do_resample:
                indices = resample_fn(weights, rng)
                particles_posterior = particles_posterior[indices]
                log_weights = -np.log(N) * np.ones(N)
                weights = np.ones(N) / N

            # ---- Step 7: Particles persist (no redraw) ----
            particles = particles_posterior

            if return_particles:
                particles_history[t + 1] = particles.copy()
                weights_history[t + 1] = weights.copy()

        # --- Build result ---
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
