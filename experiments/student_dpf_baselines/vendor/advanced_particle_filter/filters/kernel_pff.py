"""
Kernel-Embedded Particle Flow Filter.

Based on Hu & van Leeuwen (2021): "A particle flow filter for high-dimensional
system applications", Quarterly Journal of the Royal Meteorological Society.

Implements:
- KernelPFF: Particle Flow Filter with scalar or matrix-valued kernel

Key equations:
- Flow update (Eq. 7): x_{s+Δs} = x_s + (Δs/Np) * D * Σ_j [K(x_j, x) * ∇log p(x_j|y) + ∇_j · K(x_j, x)]
- Scalar kernel (Eq. 16-17): K(x,z) = exp(-0.5 * (x-z)^T A (x-z)) * I
- Matrix-valued kernel (Eq. 20-21): K(x,z) = diag(K_1, ..., K_nx) where K_d = exp(-0.5 * (x_d - z_d)^2 / (α * σ_d^2))
"""

import numpy as np
from typing import Optional, Literal, Tuple
from numpy.random import Generator, default_rng
from dataclasses import dataclass

from .base import FilterResult
from ..models.base import StateSpaceModel
from ..utils.resampling import effective_sample_size


@dataclass
class PFFDiagnostics:
    """Diagnostics from PFF iterations."""
    n_iterations: int
    final_flow_magnitude: float
    flow_history: Optional[np.ndarray] = None
    step_size_history: Optional[np.ndarray] = None


class KernelPFF:
    """
    Kernel-Embedded Particle Flow Filter.
    
    Transforms particles from prior to posterior using gradient flow in RKHS.
    All particles maintain equal weight throughout (no importance sampling).
    
    Two kernel options:
    - "scalar": Isotropic kernel, same value for all state dimensions
    - "matrix": Diagonal matrix-valued kernel, independent per dimension
    
    The matrix-valued kernel is recommended for high-dimensional systems
    with sparse observations, as it prevents collapse in observed dimensions.
    """
    
    def __init__(
        self,
        n_particles: int = 500,
        kernel_type: Literal["scalar", "matrix"] = "matrix",
        alpha: Optional[float] = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        initial_step_size: float = 0.05,
        step_size_factor: float = 1.4,
        step_increase_patience: int = 20,
        use_preconditioner: bool = True,
        use_localization: bool = False,
        localization_radius: float = 4.0,
        seed: Optional[int] = None,
        store_diagnostics: bool = False,
    ):
        """
        Args:
            n_particles: Number of particles
            kernel_type: "scalar" or "matrix" valued kernel
            alpha: Kernel bandwidth parameter. Default: 1/n_particles
            max_iterations: Maximum pseudo-time iterations
            tolerance: Convergence tolerance on flow magnitude
            initial_step_size: Initial pseudo-time step Δs
            step_size_factor: Factor for adaptive step size (default 1.4)
            step_increase_patience: Steps of decreasing flow before increasing Δs
            use_preconditioner: If True, use prior covariance as preconditioner D
            use_localization: If True, apply Gaspari-Cohn-style localization to B
            localization_radius: Localization radius r_in (Eq. 29 in Hu 2021)
            seed: Random seed
            store_diagnostics: If True, store iteration history
        """
        self.n_particles = n_particles
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.initial_step_size = initial_step_size
        self.step_size_factor = step_size_factor
        self.step_increase_patience = step_increase_patience
        self.use_preconditioner = use_preconditioner
        self.use_localization = use_localization
        self.localization_radius = localization_radius
        self.seed = seed
        self.store_diagnostics = store_diagnostics
    
    def _compute_localization_matrix(self, nx: int) -> np.ndarray:
        """
        Compute Gaspari-Cohn-style localization matrix C.
        
        C[i,j] = exp{ -((i-j)/r_in)^2 }
        
        Following Hu (2021) Eq. (29).
        
        Args:
            nx: State dimension
            
        Returns:
            C: [nx, nx] localization matrix
        """
        i_idx = np.arange(nx)[:, np.newaxis]
        j_idx = np.arange(nx)[np.newaxis, :]
        C = np.exp(-((i_idx - j_idx) / self.localization_radius) ** 2)
        return C
    
    def _compute_prior_statistics(
        self,
        particles: np.ndarray,
        regularization: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute prior mean and covariance from particles.
        
        If use_localization=True, applies Schur product localization:
            B ← B ∘ C
        where C[i,j] = exp{ -((i-j)/r_in)^2 } (Hu 2021, Eq. 28-29)
        
        Args:
            particles: [N, nx] particle positions
            regularization: Small value added to diagonal for numerical stability
            
        Returns:
            x_mean: [nx] ensemble mean
            B: [nx, nx] prior covariance (localized if enabled, regularized)
            B_inv: [nx, nx] inverse of prior covariance
        """
        N, nx = particles.shape
        
        x_mean = np.mean(particles, axis=0)  # [nx]
        
        # Perturbation matrix and covariance
        X = particles - x_mean  # [N, nx]
        B = (X.T @ X) / (N - 1)  # [nx, nx]
        
        # Apply localization if enabled (Eq. 28-29 in Hu 2021)
        if self.use_localization:
            C = self._compute_localization_matrix(nx)
            B = B * C  # Schur (element-wise) product
        
        # Regularize
        B = 0.5 * (B + B.T)  # Symmetrize
        B = B + regularization * np.eye(nx)
        
        # Compute inverse
        B_inv = np.linalg.inv(B)
        
        return x_mean, B, B_inv
    
    def _compute_grad_log_posterior(
        self,
        particles: np.ndarray,
        z: np.ndarray,
        model: StateSpaceModel,
        B_inv: np.ndarray,
        x_mean: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient of log posterior for all particles.
        
        ∇log p(x|y) = ∇log p(y|x) + ∇log p(x)
                    = H^T R^{-1} (y - H(x)) - B^{-1} (x - x_mean)
        
        Args:
            particles: [N, nx] particle positions
            z: [ny] observation
            model: StateSpaceModel
            B_inv: [nx, nx] inverse prior covariance
            x_mean: [nx] prior mean
            
        Returns:
            grad_log_post: [N, nx] gradient of log posterior at each particle
        """
        N, nx = particles.shape
        ny = model.obs_dim
        
        # Observation predictions
        y_pred = model.obs_mean(particles)  # [N, ny]
        residual = z - y_pred  # [N, ny]
        
        # R inverse
        R_inv = np.linalg.inv(model.obs_cov)  # [ny, ny]
        
        # Gradient of log likelihood: H^T R^{-1} (y - H(x))
        # Need Jacobian at each particle
        grad_log_lik = np.zeros((N, nx))
        for i in range(N):
            H_i = model.obs_jacobian(particles[i])  # [ny, nx]
            grad_log_lik[i] = H_i.T @ R_inv @ residual[i]  # [nx]
        
        # Gradient of log prior: -B^{-1} (x - x_mean)
        grad_log_prior = -(particles - x_mean) @ B_inv.T  # [N, nx]
        
        # Total gradient
        grad_log_post = grad_log_lik + grad_log_prior  # [N, nx]
        
        return grad_log_post
    
    def _compute_flow_scalar_kernel(
        self,
        particles: np.ndarray,
        grad_log_post: np.ndarray,
        B: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Compute particle flow using scalar (isotropic) kernel.
        
        K(x, z) = exp(-0.5 * (x-z)^T A (x-z)) * I_nx
        where A = (α * B)^{-1}
        
        Flow: f(x) = (1/N) * Σ_j [K(x_j, x) * ∇log p(x_j|y) + ∇_j · K(x_j, x)]
        
        Args:
            particles: [N, nx] particle positions
            grad_log_post: [N, nx] gradient of log posterior
            B: [nx, nx] prior covariance
            alpha: kernel bandwidth parameter
            
        Returns:
            flow: [N, nx] particle flow (before preconditioner)
        """
        N, nx = particles.shape
        
        # A = (α * B)^{-1}
        A = np.linalg.inv(alpha * B)  # [nx, nx]
        
        # Pairwise differences: diff[i, j] = x_i - x_j
        # Shape: [N, N, nx]
        diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
        
        # Mahalanobis distance squared: (x_i - x_j)^T A (x_i - x_j)
        # diff @ A: [N, N, nx]
        diff_A = np.einsum('ijk,kl->ijl', diff, A)  # [N, N, nx]
        dist_sq = np.einsum('ijk,ijk->ij', diff_A, diff)  # [N, N]
        
        # Scalar kernel: K[i,j] = exp(-0.5 * dist_sq[i,j])
        K = np.exp(-0.5 * dist_sq)  # [N, N]
        
        # Gradient of kernel (Eq. 19): ∇_x K(x, z) = -A^T (x - z) K(x, z)
        # For particle j contributing to flow at particle i:
        # ∇_{x_j} K(x_j, x_i) = -A^T (x_j - x_i) K(x_j, x_i)
        #                     = A^T (x_i - x_j) K(x_j, x_i)  [since diff[i,j] = x_i - x_j]
        # grad_K[i, j] = A^T @ diff[i, j] * K[i, j], shape [N, N, nx]
        grad_K = np.einsum('ij,ijk->ijk', K, diff_A)  # [N, N, nx] - using A symmetry
        
        # Attracting term: (1/N) * Σ_j K[i,j] * ∇log p(x_j|y)
        # K[i,j] weights the gradient at particle j for flow at particle i
        attracting = (K @ grad_log_post) / N  # [N, nx]
        
        # Repelling term: (1/N) * Σ_j ∇_{x_j} · K(x_j, x_i)
        # For scalar kernel, divergence = gradient (since K is scalar * I)
        repelling = np.sum(grad_K, axis=1) / N  # [N, nx]
        
        flow = attracting + repelling  # [N, nx]
        
        return flow
    
    def _compute_flow_matrix_kernel(
        self,
        particles: np.ndarray,
        grad_log_post: np.ndarray,
        B: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Compute particle flow using matrix-valued (diagonal) kernel.
        
        K(x, z) = diag(K_1, ..., K_nx)
        K_d(x, z) = exp(-0.5 * (x_d - z_d)^2 / (α * σ_d^2))
        
        Flow for dimension d:
        f_d(x) = (1/N) * Σ_j [K_d(x_j, x) * ∇log p(x_j|y)_d + ∂K_d/∂x_j]
        
        Args:
            particles: [N, nx] particle positions
            grad_log_post: [N, nx] gradient of log posterior
            B: [nx, nx] prior covariance
            alpha: kernel bandwidth parameter
            
        Returns:
            flow: [N, nx] particle flow (before preconditioner)
        """
        N, nx = particles.shape
        
        # Per-dimension variance: σ_d^2 = B[d, d]
        sigma_sq = alpha * np.diag(B)  # [nx]
        
        # Pairwise differences per dimension: diff[i, j, d] = x_i[d] - x_j[d]
        # Shape: [N, N, nx]
        diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
        
        # Per-dimension kernel: K[i, j, d] = exp(-0.5 * diff[i,j,d]^2 / sigma_sq[d])
        # Shape: [N, N, nx]
        K = np.exp(-0.5 * diff**2 / sigma_sq[np.newaxis, np.newaxis, :])
        
        # Gradient of kernel (Eq. 23):
        # ∂K_d/∂x_j[d] = -(x_j[d] - x_i[d]) / (α * σ_d^2) * K_d(x_j, x_i)
        #              = diff[i,j,d] / sigma_sq[d] * K[i,j,d]  [since diff = x_i - x_j]
        grad_K = diff / sigma_sq[np.newaxis, np.newaxis, :] * K  # [N, N, nx]
        
        # Attracting term: (1/N) * Σ_j K[i,j,d] * ∇log p(x_j|y)_d
        # For each dimension d independently
        # K[:,:,d] @ grad_log_post[:,d] for each d
        attracting = np.einsum('ijd,jd->id', K, grad_log_post) / N  # [N, nx]
        
        # Repelling term: (1/N) * Σ_j ∂K_d/∂x_j
        repelling = np.sum(grad_K, axis=1) / N  # [N, nx]
        
        flow = attracting + repelling  # [N, nx]
        
        return flow
    
    def _pff_update(
        self,
        particles: np.ndarray,
        z: np.ndarray,
        model: StateSpaceModel,
        B_prior: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, PFFDiagnostics]:
        """
        Run PFF iterations to transform particles from prior to posterior.
        
        Args:
            particles: [N, nx] prior particles
            z: [ny] observation
            model: StateSpaceModel
            B_prior: [nx, nx] prior covariance (if None, computed from particles)
            
        Returns:
            particles_posterior: [N, nx] posterior particles
            diagnostics: PFFDiagnostics
        """
        N, nx = particles.shape
        
        # Compute prior statistics
        x_mean, B, B_inv = self._compute_prior_statistics(particles)
        
        # Use provided B_prior if given (e.g., from model)
        if B_prior is not None:
            B = B_prior.copy()
            B = 0.5 * (B + B.T) + 1e-6 * np.eye(nx)
            B_inv = np.linalg.inv(B)
        
        # Kernel bandwidth
        # For scalar kernel, need larger bandwidth to prevent collapse in high dimensions
        # The Mahalanobis distance grows with nx, so alpha should scale accordingly
        # For matrix kernel, 1/N works well as each dimension is independent
        if self.alpha is not None:
            alpha = self.alpha
        elif self.kernel_type == "scalar":
            # Heuristic: alpha ~ nx * 0.1 to compensate for dimension-dependent distance
            alpha = float(nx) * 0.1
        else:
            alpha = 1.0 / N
        
        # Select flow computation function
        if self.kernel_type == "scalar":
            compute_flow = lambda p, g: self._compute_flow_scalar_kernel(p, g, B, alpha)
        else:  # matrix
            compute_flow = lambda p, g: self._compute_flow_matrix_kernel(p, g, B, alpha)
        
        # Preconditioner
        D = B if self.use_preconditioner else np.eye(nx)
        
        # Adaptive step size
        delta_s = self.initial_step_size
        prev_flow_mag = np.inf
        decrease_count = 0
        
        # Storage for diagnostics
        if self.store_diagnostics:
            flow_history = []
            step_history = []
        
        # Main iteration loop
        particles_current = particles.copy()
        
        for iteration in range(self.max_iterations):
            # Compute gradient of log posterior
            grad_log_post = self._compute_grad_log_posterior(
                particles_current, z, model, B_inv, x_mean
            )
            
            # Compute flow
            flow = compute_flow(particles_current, grad_log_post)
            
            # Apply preconditioner: f_precond = D @ f
            flow_precond = flow @ D.T  # [N, nx]
            
            # Flow magnitude for convergence check
            flow_mag = np.mean(np.linalg.norm(flow_precond, axis=1))
            
            if self.store_diagnostics:
                flow_history.append(flow_mag)
                step_history.append(delta_s)
            
            # Check convergence
            if flow_mag < self.tolerance:
                break
            
            # Adaptive step size
            if flow_mag < prev_flow_mag:
                decrease_count += 1
                if decrease_count >= self.step_increase_patience:
                    delta_s *= self.step_size_factor
                    decrease_count = 0
            else:
                delta_s /= self.step_size_factor
                decrease_count = 0
            
            prev_flow_mag = flow_mag
            
            # Update particles
            particles_current = particles_current + delta_s * flow_precond
        
        # Build diagnostics
        diagnostics = PFFDiagnostics(
            n_iterations=iteration + 1,
            final_flow_magnitude=flow_mag,
            flow_history=np.array(flow_history) if self.store_diagnostics else None,
            step_size_history=np.array(step_history) if self.store_diagnostics else None,
        )
        
        return particles_current, diagnostics
    
    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
        return_particles: bool = False,
        return_diagnostics: bool = False,
        rng: Optional[Generator] = None,
    ) -> FilterResult:
        """
        Run Kernel PFF over all time steps.
        
        Args:
            model: StateSpaceModel
            observations: [T, ny] observations
            return_particles: If True, store particle history
            return_diagnostics: If True, store PFF diagnostics per step
            rng: Random number generator
            
        Returns:
            FilterResult
        """
        if rng is None:
            rng = default_rng(self.seed)
        
        T = observations.shape[0]
        N = self.n_particles
        nx = model.state_dim
        
        # Initialize particles from prior
        particles = model.sample_initial(N, rng)  # [N, nx]
        
        # Storage
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))
        
        if return_particles:
            particles_history = np.zeros((T + 1, N, nx))
            particles_history[0] = particles
        
        if return_diagnostics:
            all_diagnostics = []
        
        # Initial estimates
        means[0] = np.mean(particles, axis=0)
        diff = particles - means[0]
        covariances[0] = (diff.T @ diff) / (N - 1)
        
        for t in range(T):
            z = observations[t]
            
            # Propagate particles through dynamics
            particles = model.sample_dynamics(particles, rng)
            
            # PFF update: transform prior particles to posterior
            particles, diag = self._pff_update(particles, z, model)
            
            if return_diagnostics:
                all_diagnostics.append(diag)
            
            # Compute estimates
            means[t + 1] = np.mean(particles, axis=0)
            diff = particles - means[t + 1]
            covariances[t + 1] = (diff.T @ diff) / (N - 1)
            
            if return_particles:
                particles_history[t + 1] = particles
        
        # Build result
        result = FilterResult(
            means=means,
            covariances=covariances,
            ess=np.full(T, float(N)),  # All particles have equal weight
        )
        
        if return_particles:
            result.particles = particles_history
            result.weights = np.full((T + 1, N), 1.0 / N)
        
        if return_diagnostics:
            result.diagnostics = all_diagnostics
        
        return result


class ScalarKernelPFF(KernelPFF):
    """Convenience class for scalar kernel PFF."""
    
    def __init__(self, **kwargs):
        kwargs['kernel_type'] = 'scalar'
        super().__init__(**kwargs)


class MatrixKernelPFF(KernelPFF):
    """Convenience class for matrix-valued kernel PFF."""
    
    def __init__(self, **kwargs):
        kwargs['kernel_type'] = 'matrix'
        super().__init__(**kwargs)
