"""
Flow Filters Module
===================
This module implements advanced particle flow filters that migrate particles
deterministically from the prior to the posterior. This approach effectively
combats the 'curse of dimensionality' and particle degeneracy by avoiding
stochastic sampling in high-stiffness regimes.

Mathematical Context:
---------------------
Traditional Sequential Monte Carlo (SMC) filters rely on proposal distributions
and importance weighting. In high-dimensional spaces or situations with highly
accurate sensors (low observation noise), the likelihood function becomes
incredibly sharp. This causes almost all particles to receive near-zero weights,
leading to severe weight degeneracy.

Particle Flow filters solve this by embedding the filtering problem in a
continuous-time domain using a homotopy parameter (lambda) that goes from
0 (prior) to 1 (posterior). Instead of jumping and re-weighting, particles
"flow" along a deterministic differential equation dictated by the gradient
of the log-homotopy, naturally migrating to the high-probability regions of
the true posterior before any weights are calculated.

Included Methods:
1. Exact Daum-Huang (EDH): Assumes a globally Gaussian prior for flow calculation.
2. Local Exact Daum-Huang (LEDH): Computes localized flow Jacobians per particle.
3. Particle Flow Particle Filter (PFPF): Corrects the flow approximations using
   importance sampling with spatial Jacobian tracking.
4. Kernel Particle Flow Filter (KPFF): Uses RKHS embedding (similar to SVGD).
5. Stochastic Particle Flow Filter (implicitly supported via Q_filter noise).

Author: Joowon Lee
Date: 2026-02-26
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .classical import EKF

# Numerical precision and stability constants preserved from verified code
DTYPE = tf.float64
JITTER = 1e-6
tfd = tfp.distributions

def safe_inv(matrix):
    """
    Numerically stable inversion using Cholesky decomposition.
    Falls back to standard inversion with jitter if the matrix is
    singular or poorly conditioned.

    Mathematical Context:
    ---------------------
    Precision matrices (inverses of covariance matrices) must remain symmetric
    Positive Definite (SPD). Direct inversions compound floating-point errors.
    Cholesky solving (M = L * L^T) maintains this structural property. If the
    matrix loses positive-definiteness, we add Tikhonov regularization
    (jitter * I) to the diagonal to make it invertible.
    """
    try:
        L = tf.linalg.cholesky(matrix)
        return tf.linalg.cholesky_solve(L, tf.eye(tf.shape(matrix)[0], dtype=DTYPE))
    except:
        # Fallback to standard inversion with increased regularization
        return tf.linalg.inv(matrix + tf.eye(tf.shape(matrix)[0], dtype=DTYPE) * JITTER)

class ParticleFlowFilter:
    """
    Unified Particle Flow Filter (Vectorized TensorFlow).
    -----------------------------------------------------
    This class implements the Particle Flow (PF) framework, which avoids the
    'curse of dimensionality' in Sequential Monte Carlo by migrating particles
    deterministically toward high-likelihood regions via a homotopy flow.

    Variants:
    - EDH (Exact Daum-Huang): Uses a global covariance/Jacobian approximation.
    - LEDH (Local EDH): Computes local Jacobians for each particle to handle
      high non-linearity.
    - LEDH_OPT: Implements an optimal time-step schedule (dl) by solving a
      Boundary Value Problem to mitigate numerical stiffness.
    """
    def __init__(self, model, N=100, mode='ledh', is_pfpf=True, mu=0.2):
        self.m = model; self.N = N; self.mode = mode; self.is_pfpf = is_pfpf
        self.dim = model.state_dim
        self.mu = mu # Stiffness mitigation weight scalar (Dai22)
        self.n_steps = 30 # Number of integration steps for the flow

        # Nominal linear-log schedule for standard Li17 modes
        # This maps the homotopy parameter lambda from 0 (prior) to 1 (posterior)
        eps_np = np.logspace(-2, 0, self.n_steps); eps_np /= eps_np.sum()
        self.eps = tf.constant(eps_np, dtype=DTYPE)

        # Internal EKF used to track the 'global' ensemble mean and covariance for flow calculations
        self.ekf = EKF(model)
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N)
        self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        """Initializes the particle ensemble and the internal EKF state."""
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))
        self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        self.ekf.init(x, P)

    def _dynamic_bvp_solver(self, P_inv, H_hess):
        """
        Robust dynamic BVP solver for Dai22 optimal schedule.
        ----------------------------------------------------
        Solves for the optimal discretization of the homotopy parameter lambda.
        It uses a 'Shooting Method' to find an initial velocity v0 that allows
        the schedule to reach beta=1.0 at the final step while minimizing stiffness.

        Mathematical Context:
        ---------------------
        In standard flow filters, lambda increases linearly or logarithmically.
        However, if the observation is highly informative, the flow differential
        equation becomes mathematically "stiff," causing explicit integration
        schemes (like Euler) to explode. This solver computes a custom schedule
        (step sizes 'dl') that takes extremely small steps during periods of high
        matrix stiffness, defined by the condition number of (P^-1 + beta * H^T R^-1 H).
        """
        lambdas = np.linspace(0, 1, self.n_steps)
        dt = 1.0 / (self.n_steps - 1)

        def get_accel(beta_val):
            """Calculates the acceleration of the schedule velocity based on matrix stiffness."""
            b = tf.cast(tf.clip_by_value(beta_val, 0.0, 1.1), DTYPE)
            M = P_inv + b * H_hess

            # Robust inversion: Add jitter to prevent singularity during the shooting simulation
            try:
                # Increased jitter to 1e-4 for Acoustic stability
                M_inv = tf.linalg.inv(M + tf.eye(tf.shape(M)[-1], dtype=DTYPE) * 1e-4)
            except:
                M_inv = tf.linalg.pinv(M) # Fallback to pseudo-inverse

            # Nuclear norm condition number derivative (Dai22 Eq 28)
            # This term penalizes beta-paths that lead to ill-conditioned flow matrices
            t1 = tf.linalg.trace(H_hess) * tf.linalg.trace(M_inv)
            t2 = tf.linalg.trace(M) * tf.linalg.trace(M_inv @ H_hess @ M_inv)
            return -self.mu * (t1 + t2)

        def simulate(v0):
            """Forward simulation of the schedule for a given initial velocity."""
            beta = np.zeros(self.n_steps)
            v = v0
            for i in range(self.n_steps - 1):
                # Euler update for beta
                beta[i+1] = np.clip(beta[i] + v * dt, 0.0, 1.5)
                accel = get_accel(beta[i+1]).numpy()
                v += accel * dt
                # Stability check: stop if velocity explodes or beta becomes NaN
                if not np.isfinite(v) or np.isnan(beta[i+1]):
                    return np.ones(self.n_steps) * 999
            return beta

        # Bisection shooting for v0 such that beta(1) = 1
        low, high = 1.0, 50.0 # Expanded search range
        for _ in range(15):
            v_mid = (low + high) / 2
            res = simulate(v_mid)
            if res[-1] < 1.0:
                low = v_mid
            else:
                high = v_mid

        beta_opt = simulate(low)
        # Ensure the final output is monotonically increasing and bounded between 0 and 1
        beta_opt = np.clip(np.sort(beta_opt), 0.0, 1.0)
        dl_opt = np.diff(beta_opt, prepend=0.0)

        return beta_opt.astype(np.float64), dl_opt.astype(np.float64)

    @tf.function
    def step(self, z):
        """
        Executes a single filtering cycle: Predict -> Flow -> Correction.
        """
        if not self.is_pfpf:
            # For non-PFPF modes (pure flow), re-sample to maintain diversity at each step
            # Pure flow filters do not maintain weights, so they require resampling
            # to reconstruct the prior distribution accurately before propagating.
            mean = tf.reduce_mean(self.X, axis=0)
            cov = tfp.stats.covariance(self.X) + tf.eye(self.dim, dtype=DTYPE)*1e-4
            dist = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(cov))
            self.X.assign(dist.sample(self.N))
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

        # 1. Prediction: Propagate particles through the transition model
        noise = tfd.MultivariateNormalTriL(loc=tf.zeros(self.dim, dtype=DTYPE), scale_tril=tf.linalg.cholesky(self.m.Q_filter)).sample(self.N)

        if hasattr(self.m, 'propagate'):
            # Call the function for Non-Linear models
            self.X.assign(self.m.propagate(self.X) + noise)
        else:
            # Use Matrix Multiplication for Linear models
            self.X.assign(tf.transpose(self.m.F @ tf.transpose(self.X)) + noise)

        # Update internal EKF prior state for flow calculation
        # The EKF approximates the global background covariance (P) necessary for Daum-Huang math.
        x_bar_prop = tf.reduce_mean(self.X, axis=0)
        P_prop = tfp.stats.covariance(self.X) + tf.eye(self.dim, dtype=DTYPE)*1e-4
        self.ekf.x.assign(x_bar_prop); self.ekf.P.assign(P_prop)

        # 2. Schedule Determination: Li17 Nominal vs Dai22 Optimized
        if self.mode == 'ledh_opt':
            P_inv = safe_inv(self.ekf.P)
            H = self.m.jacobian_h(x_bar_prop)
            H_hess = tf.transpose(H) @ self.m.R_inv_filter @ H

            # Solve BVP dynamically to mitigate stiffness via shooting method
            beta_vals, dl_vals = tf.py_function(self._dynamic_bvp_solver, [P_inv, H_hess], [DTYPE, DTYPE])
            beta_vals.set_shape([self.n_steps])
            dl_vals.set_shape([self.n_steps])
        else:
            # Use standard geometric/log schedule
            beta_vals = tf.cumsum(self.eps)
            dl_vals = self.eps

        eta = tf.identity(self.X); log_det = tf.zeros(self.N, dtype=DTYPE)

        # Precompute global Jacobian/Error if using non-localized EDH
        if self.mode == 'edh':
            x_bar = tf.reduce_mean(eta, axis=0)
            H_g = self.m.jacobian_h(x_bar)
            e_g = tf.reshape(self.m.h_func(x_bar), [-1]) - tf.linalg.matvec(H_g, x_bar)

        # 3. Homotopy Loop: Progressively move particles toward the likelihood mode
        for k in range(self.n_steps):
            dl, lam = dl_vals[k], beta_vals[k]

            if self.mode == 'edh':
                # Global Exact Daum-Huang flow equations
                # Assumes the Jacobian H is identical for all particles (evaluated at the mean).
                H, e = H_g, e_g
                S = lam * H @ self.ekf.P @ tf.transpose(H) + self.m.R_filter
                S_inv = safe_inv(S)
                A = -0.5 * self.ekf.P @ tf.transpose(H) @ S_inv @ H
                K_gain = self.ekf.P @ tf.transpose(H) @ self.m.R_inv_filter
                term1 = tf.eye(self.dim, dtype=DTYPE) + 2*lam*A
                term2 = tf.eye(self.dim, dtype=DTYPE) + lam*A
                kze = tf.linalg.matvec(K_gain, z - e); ax = tf.linalg.matvec(A, self.ekf.x)
                inner = tf.linalg.matvec(term2, kze) + ax
                b = tf.linalg.matvec(term1, inner)
                eta += dl * (tf.transpose(A @ tf.transpose(eta)) + b)

            elif self.mode == 'ledh':
                # Localized Flow: Calculate unique Jacobian for every particle
                # This significantly improves performance in highly non-linear measurement spaces.
                def ledh_body(p_i):
                    H_i = self.m.jacobian_h(p_i)
                    e_i = tf.reshape(self.m.h_func(p_i), [-1]) - tf.linalg.matvec(H_i, p_i)
                    S_i = lam * H_i @ self.ekf.P @ tf.transpose(H_i) + self.m.R_filter
                    S_inv_i = safe_inv(S_i)

                    # A_i is the flow matrix, b_i is the flow drift vector
                    A_i = -0.5 * self.ekf.P @ tf.transpose(H_i) @ S_inv_i @ H_i
                    K_gain_i = self.ekf.P @ tf.transpose(H_i) @ self.m.R_inv_filter
                    term1 = tf.eye(self.dim, dtype=DTYPE) + 2*lam*A_i
                    term2 = tf.eye(self.dim, dtype=DTYPE) + lam*A_i
                    kze = tf.linalg.matvec(K_gain_i, z - e_i); ap = tf.linalg.matvec(A_i, p_i)
                    inner = tf.linalg.matvec(term2, kze) + ap
                    b_i = tf.linalg.matvec(term1, inner)

                    # Euler integration step
                    update = dl * (tf.linalg.matvec(A_i, p_i) + b_i)
                    ld_i = 0.0
                    if self.is_pfpf:
                         # Log-determinant of the flow Jacobian for importance weight correction
                         # Necessary because the flow transformation compresses the volume of the
                         # state space, which must be accounted for in the importance weights.
                         sign, logdet = tf.linalg.slogdet(tf.eye(self.dim, dtype=DTYPE) + dl * A_i)
                         ld_i = logdet
                    return p_i + update, ld_i
                eta, ld_batch = tf.map_fn(ledh_body, eta, dtype=(DTYPE, DTYPE))
                if self.is_pfpf: log_det += ld_batch

            elif self.mode == 'ledh_opt':
                # Localized Flow with Dai22 optimized schedule logic
                def ledh_opt_body(p_i):
                    H_i = self.m.jacobian_h(p_i)
                    e_i = tf.reshape(self.m.h_func(p_i), [-1]) - tf.linalg.matvec(H_i, p_i)
                    S_i = lam * H_i @ self.ekf.P @ tf.transpose(H_i) + self.m.R_filter
                    S_inv_i = safe_inv(S_i)

                    # Compute Flow Jacobian (A) and Drift (b)
                    A_i = -0.5 * self.ekf.P @ tf.transpose(H_i) @ S_inv_i @ H_i
                    K_gain_i = self.ekf.P @ tf.transpose(H_i) @ self.m.R_inv_filter

                    # Refined drift following the Dai(22) schedule
                    kze = tf.linalg.matvec(K_gain_i, z - e_i)
                    drift_i = tf.linalg.matvec(A_i, p_i - self.ekf.x) + kze

                    # Update particle and log-det with the optimized step dl
                    p_new = p_i + dl * drift_i
                    ld_i = 0.0
                    if self.is_pfpf:
                        # Improved Jacobian log-det using Tr(A) for speed/stability in small dl regimes
                        # Approximation: log(det(I + dl*A)) ≈ dl * trace(A) for small dl
                        ld_i = dl * tf.linalg.trace(A_i)
                    return p_new, ld_i

                eta, ld_batch = tf.map_fn(ledh_opt_body, eta, dtype=(DTYPE, DTYPE))
                if self.is_pfpf: log_det += ld_batch

        # 4. Importance Sampling Weight Correction (PFPF Framework)
        if self.is_pfpf:
             innov = z - self.m.h_func(eta)
             quad = tf.reduce_sum(tf.matmul(innov, self.m.R_inv_filter) * innov, axis=1)
             log_lik = -0.5 * quad

             # Total log weight = Prior + Likelihood + Jacobian Determinant tracking
             # The log_det term accounts for the spatial compression induced by the deterministic flow.
             log_w = tf.math.log(self.W + 1e-30) + log_lik + log_det
             w = tf.exp(log_w - tf.reduce_max(log_w)); w = w / tf.reduce_sum(w)
             self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))
             self.X.assign(tf.reshape(eta, [self.N, self.dim]))

             # Resample if particle diversity (ESS) drops below threshold
             if self.ess < self.N/2:
                 idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
                 self.X.assign(tf.gather(self.X, idx))
                 self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
             else: self.W.assign(w)

             # Calculate MMSE estimate
             est = tf.reduce_sum(self.X * self.W[:, None], axis=0)
             # Feedback: Update internal EKF with posterior statistics for next time step
             self.ekf.init(est, tf.matmul(tf.transpose((self.X-est)*self.W[:,None]), self.X-est) + tf.eye(self.dim, dtype=DTYPE)*1e-4)
             return est
        else:
             # Pure Flow: Unweighted mean of the particle cloud
             self.X.assign(eta); self.ess.assign(float(self.N))
             est = tf.reduce_mean(self.X, axis=0)
             self.ekf.init(est, tfp.stats.covariance(self.X) + tf.eye(self.dim, dtype=DTYPE)*1e-4)
             return est


class EDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='edh', is_pfpf=False)
        self.n_steps = steps

class LEDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='ledh', is_pfpf=False)
        self.n_steps = steps

class PFPF_EDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='edh', is_pfpf=True)
        self.n_steps = steps

class PFPF_LEDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='ledh', is_pfpf=True)
        self.n_steps = steps

class KPFF:
    """
    Kernel Particle Flow Filter (KPFF) using RKHS embedding.
    Fully TensorFlow implementation.

    Mathematical Context:
    ---------------------
    Unlike Daum-Huang filters which rely on the Fokker-Planck equation and
    assumed Gaussianity, KPFF frames the particle update as a functional
    gradient descent in a Reproducing Kernel Hilbert Space (RKHS).

    This is deeply related to Stein Variational Gradient Descent (SVGD).
    Particles are pushed toward the true posterior by the smoothed gradient of
    the log-posterior, while a repulsive force (the derivative of the kernel)
    prevents particle collapse, maintaining diversity without stochastic resampling.
    """
    def __init__(self, ensemble, y_obs, obs_idx, R_var, kernel_type='matrix'):
        # Convert ensemble to Tensor (not Variable unless persistence is needed in place)
        # Assuming ensemble is updated via update() return or explicit assign in caller
        self.X = tf.Variable(tf.convert_to_tensor(ensemble, dtype=DTYPE), dtype=DTYPE)

        self.Np = tf.shape(self.X)[0]
        self.Nx = tf.shape(self.X)[1]

        self.y = tf.convert_to_tensor(y_obs, dtype=DTYPE)
        self.obs_idx = tf.convert_to_tensor(obs_idx, dtype=tf.int32)

        # Cast scalar R_var
        self.R_inv = tf.cast(1.0 / R_var, DTYPE)
        self.kernel_type = kernel_type

        # Kernel bandwidth heuristic scale factor
        self.alpha = tf.cast(1.0 / tf.cast(self.Np, DTYPE), DTYPE)

        # Prior stats (Static for the flow duration)
        self.prior_mean = tf.reduce_mean(self.X, axis=0)
        self.prior_var = tf.math.reduce_variance(self.X, axis=0) + 1e-4
        self.D = self.prior_var

    @tf.function
    def get_grad_log_posterior(self, x):
        """
        Calculates the exact gradient of the log-posterior for a given state.
        Posterior ~ Prior(x) * Likelihood(y | x)
        """
        # x: (Nx,)
        # 1. Likelihood Gradient
        # Assuming a linear-Gaussian observation y = Hx + v, where H selects specific indices.
        # Gradient = H^T * R^-1 * (y - Hx)
        x_obs = tf.gather(x, self.obs_idx)
        innov = self.y - x_obs
        grad_lik_sub = innov * self.R_inv # Scalar R

        # Scatter back to full state dimension (equivalent to multiplying by H^T)
        indices = tf.expand_dims(self.obs_idx, 1)
        grad_lik = tf.scatter_nd(indices, grad_lik_sub, [self.Nx])

        # 2. Prior Gradient
        # Assuming a Gaussian prior: Gradient = -P_prior^-1 * (x - mu_prior)
        grad_prior = -(x - self.prior_mean) / self.prior_var

        # Combine to get the unnormalized log-posterior gradient
        return grad_lik + grad_prior

    @tf.function
    def compute_flow(self, x_eval):
        """
        Computes the RKHS flow vector for a single particle evaluation point.
        """
        # Compute gradients for all particles
        grad_log_p_list = tf.map_fn(self.get_grad_log_posterior, self.X)

        # Current particle i is x_eval
        diffs = self.X - x_eval # (Np, Nx)

        if self.kernel_type == 'scalar':
            # Isotropic Gaussian RBF Kernel
            dist_sq = tf.reduce_sum(tf.square(diffs) / (self.alpha * self.prior_var), axis=1)
            k_val = tf.exp(-0.5 * dist_sq)

            # grad_k: Derivative of the kernel, acts as a repulsive force between particles
            # -K * diff / (alpha * var)
            # reshape k_val to (Np, 1)
            grad_k = -tf.expand_dims(k_val, 1) * (diffs / (self.alpha * self.prior_var))

            # Stein Variational Operator: Smooths the gradient and adds repulsion
            term = tf.expand_dims(k_val, 1) * grad_log_p_list + grad_k

        elif self.kernel_type == 'matrix':
            # Anisotropic (Matrix) Gaussian RBF Kernel
            dist_sq_vec = tf.square(diffs) / (self.alpha * self.prior_var)
            k_vec = tf.exp(-0.5 * dist_sq_vec)

            grad_k = -k_vec * (diffs / (self.alpha * self.prior_var))

            term = k_vec * grad_log_p_list + grad_k

        flow_sum = tf.reduce_sum(term, axis=0)
        return (self.D * flow_sum) / tf.cast(self.Np, DTYPE)

    @tf.function
    def update(self, n_steps=50, dt=0.01):
        """
        Executes the functional gradient descent over 'n_steps' to flow the ensemble.
        """
        dt_tf = tf.cast(dt, DTYPE)

        def body(i, current_X):
            # 1. Gradients of the log-posterior for the current ensemble
            grad_log_p = tf.map_fn(self.get_grad_log_posterior, current_X)

            # 2. Kernel matrix computation
            # diffs computes pairwise distances between all particles: (Np, Np, Nx)
            X_j = tf.expand_dims(current_X, 0)
            X_i = tf.expand_dims(current_X, 1)
            diffs = X_j - X_i

            scale = self.alpha * self.prior_var

            if self.kernel_type == 'scalar':
                dist_sq = tf.reduce_sum(tf.square(diffs)/scale, axis=2)
                K_mat = tf.exp(-0.5 * dist_sq) # (Np, Np)

                # grad K (repulsive force)
                grad_K = tf.expand_dims(K_mat, 2) * (diffs / scale)

                # Driving force (gradient) + Repulsive force
                term1 = tf.expand_dims(K_mat, 2) * tf.expand_dims(grad_log_p, 0)
                total = term1 + grad_K

            else: # Matrix
                dist_sq_vec = tf.square(diffs)/scale
                K_vec = tf.exp(-0.5 * dist_sq_vec)
                grad_K = K_vec * (diffs / scale)

                term1 = K_vec * tf.expand_dims(grad_log_p, 0)
                total = term1 + grad_K

            # Average the forces over the ensemble and scale by the diffusion tensor (D)
            flow = tf.reduce_mean(total, axis=1) * self.D

            # Euler integration update
            return i + 1, current_X + dt_tf * flow

        _, new_X = tf.while_loop(
            lambda i, x: i < n_steps,
            body,
            [0, self.X]
        )

        self.X.assign(new_X)
