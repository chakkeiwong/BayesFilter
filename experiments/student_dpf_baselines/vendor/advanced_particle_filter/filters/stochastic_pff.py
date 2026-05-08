"""
Stochastic Particle Flow Filter with Stiffness Mitigation.

Based on Dai & Daum (2022): "Stiffness Mitigation in Stochastic Particle Flow Filters"
IEEE Transactions on Aerospace and Electronic Systems, Vol. 58, No. 4, August 2022.

Implements:
- solve_optimal_beta: BVP solver for optimal homotopy schedule β*(λ)
- StochasticPFFlow: Stochastic particle flow filter with general homotopy

Key equations from Dai & Daum (2022):

  General log-homotopy (Eq 7):
    log p(x, λ) = α(λ) log p_0(x) + β(λ) log p_1(x) - log Γ(λ)

  Stochastic flow SDE (Eq 8):
    dx = f(x, λ) dλ + q(x, λ) dw_λ

  Drift (Eq 10-12), with α + β = 1 (normalization):
    f = K_1 ∇_x log p + K_2 ∇_x log h

    K_1 = (1/2) Q + (β̇/2) S^{-1} A_h S^{-1}                     (Eq 11, simplified)
    K_2 = -β̇ S^{-1}                                                (Eq 12, simplified)

    S = ∇_x ∇_x^T log p = A_0 + β A_h    (Hessian of log p)
    A_0 = -P_pred^{-1}                     (prior Hessian)
    A_h = -H^T R^{-1} H                    (likelihood Hessian)

  Under (A1), f is linear in x:  f = F(λ) x + b(λ)
  Exact drift solution via matrix exponential (split-step):
    x(λ+dλ) = expm(F dλ) x(λ) + F^{-1}(expm(F dλ) - I) b
  Then add diffusion:  x += sqrt(Q) dW

  Optimal homotopy BVP (Theorem 3.1, Eq 26-27):
    d²β*/dλ² = μ ∂κ_ν(M)/∂β,  β*(0) = 0, β*(1) = 1

Architecture (for nonlinear models):
  At each time step t:
    1. Redraw particles from N(x̄_ensemble, P) to enforce Gaussianity (A1)
    2. Propagate particles through dynamics (deterministic or stochastic)
    3. Predict P_pred = Φ P Φ' (propagate covariance through dynamics)
    4. Solve BVP for β*(λ) at predicted ensemble mean
    5. Run stochastic flow SDE with Dai drift (Eq 10-12), re-linearizing H
       at ensemble mean at each λ-step
    6. P propagates forward without Kalman update — the flow itself
       performs the Bayesian update on particles, so P only serves as
       the prior covariance for the next flow step.
    7. x̄ = ensemble mean of flowed particles
"""

import numpy as np
import warnings
from typing import Optional, Literal
from numpy.random import Generator, default_rng
from scipy.linalg import expm as scipy_expm

from .base import FilterResult
from .kalman import UnscentedKalmanFilter, ExtendedKalmanFilter
from .edh import generate_lambda_schedule
from ..models.base import StateSpaceModel


# =============================================================================
# BVP Solver for optimal homotopy schedule
# =============================================================================

def solve_optimal_beta(
    P0_inv: np.ndarray,
    Mh: np.ndarray,
    mu: float = 1e-7,
    n_eval: int = 501,
    bracket: tuple = (-512, 512),
    method: str = "Radau",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float = 1e-2,
    clip_ddot: float = 1e8,
    reg_eps: float = 1e-10,
) -> tuple:
    """
    Solve the BVP for the optimal homotopy schedule β*(λ).

    Solves (Theorem 3.1):
        β''(λ) = μ ∂κ_ν(M) / ∂β,   β(0) = 0,  β(1) = 1

    where M(β) = (1 - β) P0_inv + β (P0_inv + Mh)
               = P0_inv + β Mh
    and κ_ν(M) = tr(M) tr(M^{-1}) is the nuclear-norm condition number
    for SPD matrices.

    Uses shooting method with bisection on β̇(0).

    Args:
        P0_inv: [nx, nx] Prior precision matrix (P_pred^{-1})
        Mh: [nx, nx] Fisher information H^T R^{-1} H
        mu: Weight on condition number penalty (default 1e-7)
        n_eval: Number of output grid points
        bracket: Initial bracket for bisection on β̇(0)
        method: ODE solver method ("Radau" for stiff, "RK45" for non-stiff)
        rtol, atol: ODE solver tolerances
        max_step: Maximum ODE step size
        clip_ddot: Clamp β̈ to prevent divergence
        reg_eps: Regularization added to M for numerical stability

    Returns:
        lam_grid: [n_eval] Lambda grid from 0 to 1
        beta_opt: [n_eval] Optimal β*(λ) values
        bdot_opt: [n_eval] Optimal β̇*(λ) values
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import root_scalar

    nx = P0_inv.shape[0]
    I_nx = np.eye(nx)

    # dM/dβ = Mh (since M = P0_inv + β Mh)
    Mb = Mh

    def build_M(beta):
        """Build curvature matrix M(β) = P0_inv + β Mh."""
        return P0_inv + beta * Mh + reg_eps * I_nx

    def d_kappa_dbeta(beta):
        """
        Derivative of nuclear-norm condition number w.r.t. β.

        For SPD M with nuclear norm: κ_*(M) = tr(M) tr(M^{-1})
        d/dβ [tr(M) tr(M^{-1})] = tr(Mb) tr(M^{-1}) - tr(M) tr(M^{-1} Mb M^{-1})

        See Dai & Daum (2022) Remark 3.2, Eq (28).
        """
        M = build_M(beta)
        M_inv = np.linalg.solve(M, I_nx)
        return float(
            np.trace(Mb) * np.trace(M_inv)
            - np.trace(M) * np.trace(M_inv @ Mb @ M_inv)
        )

    def shoot_rhs(lam, y):
        """RHS of the shooting ODE: [β, β̇] -> [β̇, μ ∂κ/∂β]."""
        beta, betadot = float(y[0]), float(y[1])
        beta_ddot = mu * d_kappa_dbeta(beta)
        beta_ddot = float(np.clip(beta_ddot, -clip_ddot, clip_ddot))
        return np.array([betadot, beta_ddot], dtype=float)

    def integrate_for_s(s):
        """Integrate the ODE with initial β̇(0) = s."""
        y0 = np.array([0.0, float(s)], dtype=float)
        sol = solve_ivp(
            shoot_rhs, (0.0, 1.0), y0,
            method=method, dense_output=True,
            rtol=rtol, atol=atol, max_step=max_step,
        )
        return sol

    def residual(s):
        """Residual: β(1) - 1 for a given β̇(0) = s."""
        sol = integrate_for_s(s)
        if not sol.success or np.any(~np.isfinite(sol.y)):
            return 1e6
        return float(sol.y[0, -1] - 1.0)

    # --- Find bracket for bisection ---
    a, b = bracket
    fa, fb = residual(a), residual(b)

    if fa * fb > 0:
        # Expand bracket
        step = max(abs(a), abs(b))
        for _ in range(20):
            step *= 2.0
            a, b = -step, step
            fa, fb = residual(a), residual(b)
            if fa * fb < 0:
                break
        else:
            warnings.warn(
                "Could not bracket root for optimal β BVP. "
                "Falling back to linear schedule β(λ) = λ.",
                RuntimeWarning,
            )
            lam_grid = np.linspace(0, 1, n_eval)
            return lam_grid, lam_grid.copy(), np.ones(n_eval)

    # --- Bisection ---
    res = root_scalar(residual, bracket=(a, b), method="bisect", xtol=1e-8)

    if not res.converged:
        warnings.warn(
            f"Bisection did not converge (flag={res.flag}). "
            "Using best estimate.",
            RuntimeWarning,
        )

    # --- Final integration with dense output ---
    sol_star = integrate_for_s(res.root)

    if not sol_star.success:
        warnings.warn(
            f"Final ODE integration failed: {sol_star.message}. "
            "Falling back to linear schedule.",
            RuntimeWarning,
        )
        lam_grid = np.linspace(0, 1, n_eval)
        return lam_grid, lam_grid.copy(), np.ones(n_eval)

    lam_grid = np.linspace(0, 1, n_eval)
    beta_opt, bdot_opt = sol_star.sol(lam_grid)

    return lam_grid, beta_opt, bdot_opt


# =============================================================================
# Stochastic Particle Flow Filter
# =============================================================================

class StochasticPFFlow:
    """
    Stochastic Particle Flow Filter (Dai & Daum 2022).

    Migrates particles from prior to posterior via a stochastic differential
    equation with general homotopy. Uses uniform weights (no importance sampling).

    The drift f (Eq 10) decomposes as:
      f = K_1 ∇log p + K_2 ∇log h

    with K_1, K_2 depending on the Hessian S = ∇²log p, the diffusion Q,
    and the homotopy derivatives α̇, β̇.

    Key design choices:
      - Redraw from N(x̄, P_ukf) at each time step to enforce Gaussianity
      - UKF tracks filtering covariance; ensemble mean anchors UKF
      - H re-linearized at ensemble mean at each λ-step
      - Q_flow can be fixed or adaptive (M^{-1} per λ-step)
      - β(λ) schedule: linear (β=λ) or optimal (BVP solution)
    """

    def __init__(
        self,
        n_particles: int = 100,
        n_flow_steps: int = 29,
        step_schedule : Literal["fine","exp"] = "fine",
        flow_step_ratio: Optional[float] = None,
        Q_flow_mode: Literal["fixed", "adaptive"] = "adaptive",
        Q_flow_fixed: Optional[np.ndarray] = None,
        beta_schedule: Literal["linear", "optimal"] = "optimal",
        mu: float = 1e-7,
        bvp_method: str = "Radau",
        bvp_max_step: float = 1e-2,
        bvp_clip_ddot: float = 1e8,
        deterministic_dynamics: bool = False,
        integration_method: Literal["euler", "semi_implicit", "heun", "expm"] = "expm",
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_particles: Number of particles
            n_flow_steps: Number of pseudo-time steps (default 29)
            flow_step_ratio: Ratio between consecutive step sizes (default 1.2)
            Q_flow_mode: "adaptive" for Q = M(λ)^{-1} (Dai Example 2),
                         "fixed" for user-specified Q_flow_fixed
            Q_flow_fixed: [nx, nx] Fixed diffusion matrix (used when Q_flow_mode="fixed")
            beta_schedule: "linear" for β=λ, "optimal" for BVP solution
            mu: Weight for condition number penalty in BVP (default 1e-7)
            bvp_method: ODE solver for BVP ("Radau" recommended for stiff problems)
            bvp_max_step: Max step size for BVP ODE solver
            bvp_clip_ddot: Clamp on β̈ in BVP
            deterministic_dynamics: If True, propagate particles deterministically
                                   (no process noise). Set True for Dai Example 2.
            integration_method: SDE integration scheme for the particle flow:
                "expm" — split-step matrix exponential (default, recommended).
                    Exact solution of the linear drift ODE via expm(F dλ),
                    then additive diffusion. Handles stiffness perfectly.
                "euler" — Euler-Maruyama. Simple but unstable for stiff flows.
                "semi_implicit" — semi-implicit Euler. Better stability than
                    Euler but still approximate.
                "heun" — Heun's predictor-corrector (2nd order drift accuracy).
            seed: Random seed
        """
        self.n_particles = n_particles
        self.n_flow_steps = n_flow_steps
        self.flow_step_ratio = flow_step_ratio
        self.Q_flow_mode = Q_flow_mode
        self.Q_flow_fixed = Q_flow_fixed
        self.beta_schedule = beta_schedule
        self.mu = mu
        self.bvp_method = bvp_method
        self.bvp_max_step = bvp_max_step
        self.bvp_clip_ddot = bvp_clip_ddot
        self.deterministic_dynamics = deterministic_dynamics
        self.integration_method = integration_method
        self.seed = seed
        if step_schedule == 'exp':
            self.step_sizes = generate_lambda_schedule(n_flow_steps, flow_step_ratio)
        else:
            self.step_sizes = 1e-9 * np.ones(self.n_flow_steps)
        self.ukf = UnscentedKalmanFilter()
        self.ekf = ExtendedKalmanFilter()

    # -------------------------------------------------------------------------
    # Homotopy schedule
    # -------------------------------------------------------------------------

    def _compute_beta_schedule(
        self,
        P_pred: np.ndarray,
        eta_bar: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        Compute the homotopy schedule β(λ), β̇(λ).

        For "linear": β = λ, β̇ = 1.
        For "optimal": solve BVP at the current operating point.

        Args:
            P_pred: [nx, nx] Predictive covariance
            eta_bar: [nx] Linearization point for BVP (predicted ensemble mean)
            model: StateSpaceModel

        Returns:
            lam_grid: [n_eval] Lambda grid
            beta_arr: [n_eval] β(λ) values
            bdot_arr: [n_eval] β̇(λ) values
        """
        n_eval = 501

        if self.beta_schedule == "linear":
            lam_grid = np.linspace(0, 1, n_eval)
            return lam_grid, lam_grid.copy(), np.ones(n_eval)

        elif self.beta_schedule == "optimal":
            nx = P_pred.shape[0]
            P0_inv = np.linalg.inv(P_pred + 1e-10 * np.eye(nx))
            H = model.obs_jacobian(eta_bar)
            R_inv = np.linalg.inv(model.obs_cov + 1e-10 * np.eye(model.obs_dim))
            Mh = H.T @ R_inv @ H

            return solve_optimal_beta(
                P0_inv, Mh,
                mu=self.mu,
                n_eval=n_eval,
                method=self.bvp_method,
                max_step=self.bvp_max_step,
                clip_ddot=self.bvp_clip_ddot,
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

    def _interp_beta_bdot(
        self,
        lam: float,
        lam_grid: np.ndarray,
        beta_arr: np.ndarray,
        bdot_arr: np.ndarray,
    ) -> tuple:
        """Interpolate β(λ) and β̇(λ) from precomputed arrays."""
        beta = float(np.interp(lam, lam_grid, beta_arr))
        bdot = float(np.interp(lam, lam_grid, bdot_arr))
        return beta, bdot

    # -------------------------------------------------------------------------
    # Single flow step
    # -------------------------------------------------------------------------

    def _spf_flow_step(
        self,
        particles: np.ndarray,
        x_prior_mean: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
        beta: float,
        bdot: float,
        dlam: float,
        model: StateSpaceModel,
        rng: Generator,
    ) -> np.ndarray:
        """
        Single step of the stochastic particle flow SDE.

        Implements Dai & Daum (2022) Eq (8) with drift from Eq (10)-(12).

        Under the normalization α + β = 1, α̇ = -β̇:

          S = ∇²log p = A_0 + β A_h
          K_1 = (1/2) Q + (β̇/2) S^{-1} A_h S^{-1}
          K_2 = -β̇ S^{-1}
          f = K_1 ∇log p + K_2 ∇log h

        Integration methods:
          "expm": Under (A1), f = F x + b (linear in x). Exact drift via
                  x(λ+dλ) = expm(F dλ) x(λ) + F^{-1}(expm(F dλ) - I) b,
                  then add diffusion noise (split-step).
          "euler": Euler-Maruyama with per-particle nonlinear gradients.
          "semi_implicit": Semi-implicit Euler for stiff flows.
          "heun": Heun's predictor-corrector (2nd order).
        """
        N, nx = particles.shape

        # --- Hessian components ---
        P_inv = np.linalg.inv(P_pred + 1e-10 * np.eye(nx))
        A0 = -P_inv  # ∇²log p_0

        # Linearize observation at ensemble mean
        x_mean = np.mean(particles, axis=0)
        H_mean = model.obs_jacobian(x_mean)
        R_inv = np.linalg.inv(model.obs_cov + 1e-10 * np.eye(model.obs_dim))
        Mh = H_mean.T @ R_inv @ H_mean
        Ah = -Mh  # ∇²log h

        # Full Hessian: S = A0 + β Ah  (since α+β=1, (α+β)A0 + βAh = A0 + βAh)
        S = A0 + beta * Ah
        S = 0.5 * (S + S.T)  # symmetrize

        # S^{-1}
        I_nx = np.eye(nx)
        S_inv = np.linalg.solve(S, I_nx)

        # --- Q_flow for this step ---
        # M = -S = P_inv + β Mh
        if self.Q_flow_mode == "adaptive":
            # Q = M^{-1} = (-S)^{-1} = -S_inv
            # M = -S is positive definite, so M_inv = -S_inv is also PD
            M = -S
            M = 0.5 * (M + M.T)
            Q_flow = np.linalg.inv(M + 1e-10 * I_nx)
            Q_flow = 0.5 * (Q_flow + Q_flow.T)

        elif self.Q_flow_mode == "fixed":
            Q_flow = self.Q_flow_fixed
            if Q_flow is None:
                raise ValueError(
                    "Q_flow_fixed must be provided when Q_flow_mode='fixed'"
                )
        else:
            raise ValueError(f"Unknown Q_flow_mode: {self.Q_flow_mode}")

        # Cholesky of Q_flow for diffusion
        try:
            Q_flow_sqrt = np.linalg.cholesky(Q_flow + 1e-10 * I_nx)
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(Q_flow)
            eigvals = np.maximum(eigvals, 1e-10)
            Q_flow_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

        # --- Drift matrices (shared across particles) ---
        # K_1 = (1/2)Q + (β̇/2) S^{-1} Ah S^{-1}
        S_inv_Ah = S_inv @ Ah
        K1 = 0.5 * Q_flow + (bdot / 2.0) * S_inv_Ah @ S_inv

        # K_2 = -β̇ S^{-1}
        K2 = -bdot * S_inv

        # =================================================================
        # EXPM split-step: exact matrix exponential for drift
        # =================================================================
        if self.integration_method == "expm":
            # Under (A1), f is linear in x. Using the linearized H at
            # ensemble mean:
            #   ∇log p_0(x) = -P_inv (x - m)
            #   ∇log h(x)   = H'R^{-1}(z - Hx)  = H'R^{-1}z - Mh x
            #   ∇log p(x)   = S x + c_post
            # where c_post = P_inv m + β H'R^{-1} z
            #
            # f = K1(Sx + c_post) + K2(H'R^{-1}z - Mh x)
            #   = (K1 S - K2 Mh) x + (K1 c_post + K2 H'R^{-1}z)
            #   = F x + b
            #
            # Exact solution: x(λ+dλ) = expm(F dλ) x + F^{-1}(expm(F dλ) - I) b

            R_inv_local = np.linalg.inv(model.obs_cov + 1e-10 * np.eye(model.obs_dim))
            HtRinv_z = H_mean.T @ R_inv_local @ z
            c_post = P_inv @ x_prior_mean + beta * HtRinv_z

            F_mat = K1 @ S - K2 @ Mh
            b_vec = K1 @ c_post + K2 @ HtRinv_z

            # Matrix exponential of F * dlam
            eFdl = scipy_expm(F_mat * dlam)

            # F^{-1}(expm(F dλ) - I) b  via solve
            try:
                drift_offset = np.linalg.solve(F_mat, (eFdl - I_nx) @ b_vec)
            except np.linalg.LinAlgError:
                # Fallback: Taylor expansion for near-singular F
                drift_offset = dlam * b_vec + 0.5 * dlam**2 * (F_mat @ b_vec)

            # Apply to all particles: exact drift + additive noise
            particles_new = np.empty_like(particles)
            for i in range(N):
                dW = rng.standard_normal(nx) * np.sqrt(dlam)
                particles_new[i] = eFdl @ particles[i] + drift_offset + Q_flow_sqrt @ dW

            return particles_new

        # =================================================================
        # Per-particle methods (euler, semi_implicit, heun)
        # =================================================================
        particles_new = np.empty_like(particles)

        # Precompute semi-implicit Jacobian if needed
        if self.integration_method == "semi_implicit":
            # Approximate Jacobian of drift w.r.t. x:
            #   J ≈ K1 @ A0 + β * K1 @ Ah + K2 @ Ah (from ∇log p, ∇log h dependence on x)
            # Dominant term: K1 @ S (since ∇log p = S(x - m) + β∇log h)
            # Simplification: J_approx = K1 @ S ≈ (1/2)Q_flow@S + (bdot/2)Ah@S_inv@S
            #                            = (1/2)Q_flow@S + (bdot/2)Ah
            # For stability, use the simpler form: J ≈ -0.5*Q_flow @ P_inv
            # which captures the dominant prior-pull stiffness
            J_approx = -0.5 * Q_flow @ P_inv
            IminJdl = I_nx - J_approx * dlam

        for i in range(N):
            xi = particles[i]

            # ∇log p_0 at particle i: A0 (x - m_prior) = -P_inv (x - m_prior)
            grad_log_prior = A0 @ (xi - x_prior_mean)

            # ∇log h at particle i (per-particle Jacobian for accuracy)
            h_i = model.obs_mean(xi[np.newaxis, :])[0]
            H_i = model.obs_jacobian(xi)
            innov_i = z - h_i
            grad_log_lik = H_i.T @ R_inv @ innov_i

            # ∇log p = ∇log p_0 + β ∇log h  (since α+β = 1)
            grad_log_post = grad_log_prior + beta * grad_log_lik

            # Drift: f = K_1 ∇log p + K_2 ∇log h
            drift = K1 @ grad_log_post + K2 @ grad_log_lik

            # Brownian increment
            dW = rng.standard_normal(nx) * np.sqrt(dlam)
            noise = Q_flow_sqrt @ dW

            if self.integration_method == "euler":
                # Euler-Maruyama: x += f Δλ + sqrt(Q) dW
                particles_new[i] = xi + drift * dlam + noise

            elif self.integration_method == "semi_implicit":
                # Semi-implicit: (I - J Δλ)^{-1} (x + f Δλ + noise)
                # Stabilises stiff flows by implicitly damping fast modes
                particles_new[i] = np.linalg.solve(
                    IminJdl, xi + drift * dlam + noise
                )

            elif self.integration_method == "heun":
                # Heun's predictor-corrector (2nd order)
                x_pred = xi + drift * dlam
                # Re-evaluate drift at predicted point
                grad_prior_2 = A0 @ (x_pred - x_prior_mean)
                h_2 = model.obs_mean(x_pred[np.newaxis, :])[0]
                H_2 = model.obs_jacobian(x_pred)
                grad_lik_2 = H_2.T @ R_inv @ (z - h_2)
                grad_post_2 = grad_prior_2 + beta * grad_lik_2
                drift_2 = K1 @ grad_post_2 + K2 @ grad_lik_2
                # Trapezoidal average + noise
                particles_new[i] = xi + 0.5 * (drift + drift_2) * dlam + noise

            else:
                raise ValueError(
                    f"Unknown integration_method: {self.integration_method}"
                )

        return particles_new

    # -------------------------------------------------------------------------
    # Full flow (prior -> posterior)
    # -------------------------------------------------------------------------

    def _spf_flow(
        self,
        particles: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
        model: StateSpaceModel,
        mu_0: np.ndarray,
        lam_grid: np.ndarray,
        beta_arr: np.ndarray,
        bdot_arr: np.ndarray,
        rng: Generator,
        return_diagnostics: bool = False,
    ):
        """
        Full stochastic particle flow from prior to posterior.

        Args:
            particles: [N, nx] particles from predictive distribution
            P_pred: [nx, nx] predictive covariance
            z: [ny] observation
            model: StateSpaceModel
            mu_0: [nx] predicted mean (η̄₀ for gradient computation)
            lam_grid: [n_eval] precomputed lambda grid for β schedule
            beta_arr: [n_eval] precomputed β(λ) values
            bdot_arr: [n_eval] precomputed β̇(λ) values
            rng: Random generator
            return_diagnostics: If True, return flow diagnostics

        Returns:
            particles_flowed: [N, nx]
            diagnostics: dict (if return_diagnostics=True)
        """
        nx = model.state_dim

        if return_diagnostics:
            flow_magnitudes = []
            condition_numbers = []

        lam = 0.0
        for dlam in self.step_sizes:
            lam += dlam
            beta, bdot = self._interp_beta_bdot(lam, lam_grid, beta_arr, bdot_arr)

            particles_old = particles.copy() if return_diagnostics else None

            particles = self._spf_flow_step(
                particles, mu_0, P_pred, z,
                beta, bdot, dlam, model, rng
            )

            if return_diagnostics:
                displacement = np.linalg.norm(particles - particles_old, axis=1).mean()
                flow_magnitudes.append(displacement)

                # Condition number of M = P_inv + β Mh
                P_inv = np.linalg.inv(P_pred + 1e-10 * np.eye(nx))
                H = model.obs_jacobian(np.mean(particles, axis=0))
                R_inv = np.linalg.inv(model.obs_cov + 1e-10 * np.eye(model.obs_dim))
                Mh = H.T @ R_inv @ H
                M = P_inv + beta * Mh + 1e-10 * np.eye(nx)
                condition_numbers.append(np.linalg.cond(M))


        if return_diagnostics:
            diagnostics = {
                "flow_magnitudes": np.array(flow_magnitudes),
                "condition_numbers": np.array(condition_numbers),
            }
            return particles, diagnostics

        return particles

    # -------------------------------------------------------------------------
    # Main filter loop
    # -------------------------------------------------------------------------

    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
        return_diagnostics: bool = False,
        rng: Optional[Generator] = None,
    ) -> FilterResult:
        """
        Run the Stochastic Particle Flow Filter.

        Architecture per time step:
          1. Redraw particles from N(x̄_ensemble, P) [except t=0]
          2. Propagate particles through dynamics
          3. Predict P_pred = Φ P Φ' (no Kalman update)
          4. Compute β*(λ) schedule via BVP at predicted ensemble mean
          5. Run stochastic flow SDE (expm split-step by default)
          6. P propagates forward (P = P_pred, no measurement update)
          7. x̄ = ensemble mean of flowed particles

        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations
            return_diagnostics: If True, store per-step flow diagnostics
            rng: Optional random generator

        Returns:
            FilterResult with means, covariances, diagnostics
        """
        if rng is None:
            rng = default_rng(self.seed)

        T = observations.shape[0]
        N = self.n_particles
        nx = model.state_dim

        # --- Initialize ---
        particles = model.sample_initial(N, rng)

        # EKF state: anchored to ensemble mean
        x_bar = np.mean(particles, axis=0)  # ensemble mean
        P_ukf = model.initial_cov.copy()    # EKF covariance

        # Storage
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))

        means[0] = x_bar.copy()
        covariances[0] = (
            np.cov(particles.T) if nx > 1
            else np.array([[particles.var()]])
        )

        if return_diagnostics:
            all_diagnostics = []

        # --- Main loop ---
        for t in range(T):
            z = observations[t]

            # ---- Step 1: Redraw from N(x̄, P_ekf) ----
            # Enforces Gaussianity assumption (A1) at each time step.
            # At t=0, particles are already from the initial Gaussian.
            # if t > 0:
            #     P_redraw = 0.5 * (P_ukf + P_ukf.T)
            #     # Regularize for Cholesky
            #     min_eig = np.min(np.linalg.eigvalsh(P_redraw))
            #     if min_eig < 1e-10:
            #         P_redraw = P_redraw + (1e-6 - min(min_eig, 0)) * np.eye(nx)
            #     try:
            #         P_chol = np.linalg.cholesky(P_redraw)
            #     except np.linalg.LinAlgError:
            #         eigvals, eigvecs = np.linalg.eigh(P_redraw)
            #         eigvals = np.maximum(eigvals, 1e-8)
            #         P_chol = eigvecs @ np.diag(np.sqrt(eigvals))
            #     particles = x_bar + rng.standard_normal((N, nx)) @ P_chol.T

            # ---- Step 2: Propagate through dynamics ----
            if self.deterministic_dynamics:
                particles_pred = model.dynamics_mean(particles)
            else:
                particles_pred = model.sample_dynamics(particles, rng)

            # Predicted ensemble mean
            # Initial estimates
            x_bar_pred = np.mean(particles_pred, axis=0)
            P_pred = (
                np.cov(particles_pred.T) if nx > 1
                else np.array([[particles_pred.var()]])
            )
            # ---- Step 3: UKF predict (anchored to ensemble mean) ----
            # Use x̄ as UKF mean so that UKF tracks the particle cloud
            # m_pred, P_pred = self.ukf.predict(x_bar, P_ukf, model)


            # ---- Step 4: Compute β*(λ) schedule ----
            lam_grid, beta_arr, bdot_arr = self._compute_beta_schedule(
                P_pred, x_bar_pred, model
            )

            # ---- Step 5: Stochastic particle flow ----
            if return_diagnostics:
                particles_post, diag = self._spf_flow(
                    particles_pred, P_pred, z, model,
                    mu_0=x_bar_pred,
                    lam_grid=lam_grid,
                    beta_arr=beta_arr,
                    bdot_arr=bdot_arr,
                    rng=rng,
                    return_diagnostics=True,
                )
                all_diagnostics.append(diag)
            else:
                particles_post = self._spf_flow(
                    particles_pred, P_pred, z, model,
                    mu_0=x_bar_pred,
                    lam_grid=lam_grid,
                    beta_arr=beta_arr,
                    bdot_arr=bdot_arr,
                    rng=rng,
                )

            # ---- Step 7: Update ensemble mean, store results ----
            x_bar = np.mean(particles_post, axis=0)
                       
            # m_ukf, P_ukf = self.ukf.update(m_pred, P_pred, z, model)

            means[t + 1] = x_bar.copy()
            covariances[t + 1] = (
                np.cov(particles_post.T) if nx > 1
                else np.array([[particles_post.var()]])
            )

            particles = particles_post
            P_ukf = covariances[t + 1] 

        # --- Build result ---
        result = FilterResult(
            means=means,
            covariances=covariances,
            ess=np.full(T, float(N)),  # uniform weights -> ESS = N
        )

        if return_diagnostics:
            result.diagnostics = all_diagnostics

        return result
