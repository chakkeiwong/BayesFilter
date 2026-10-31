"""
Kernel Flow Module
==================
This module implements the Kernel Particle Flow (KPF) utilizing RKHS
embeddings. Unlike Daum-Huang flows, which rely on local linearization
(Jacobians), KPF uses a collective interaction kernel to migrate the
ensemble as a single manifold.

Methodology:
------------
By projecting the log-posterior gradients into an RKHS, the flow
automatically adapts to the geometry of the particle ensemble. This
is particularly robust for multi-modal distributions or highly
non-linear observation models where Jacobians may be ill-conditioned.

Author: Joowon Lee
Date: 2026-02-26
"""

import tensorflow as tf
import numpy as np

DTYPE = tf.float64

class KernelFlow:
    """
    Collective Ensemble Migration via RKHS Embeddings.

    This class implements the velocity field calculation for the KPFF.
    It supports coordinate-wise bandwidth adaptation (Matrix Kernel)
    to handle states with vastly different variances.
    """

    def __init__(self, prior_mean, prior_var, R_inv, kernel_type='matrix'):
        """
        Initializes the Kernel velocity engine.

        Detailed Parameter Description:
        -------------------------------
        :param prior_mean: The empirical mean of the ensemble [Dim]. Used
            to compute the prior gradient term.
        :param prior_var: The diagonal variance of the ensemble [Dim]. Used
            to set the kernel bandwidth and diffusion coefficient D.
        :param R_inv: The inverse measurement noise covariance.
        :param kernel_type (str):
            - 'scalar': Uses a single bandwidth for all dimensions.
            - 'matrix': Uses a diagonal bandwidth matrix, scaling each
              dimension by its prior variance. Essential for tracking
              problems with mixed units (e.g., position and velocity).
        """
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.R_inv = R_inv
        self.kernel_type = kernel_type

        # Diffusion coefficient D: Acts as a metric tensor that scales
        # the overall flow magnitude based on state uncertainty.
        self.D = prior_var

    @tf.function
    def compute_ensemble_velocity(self, current_X, y_obs, obs_idx, alpha):
        """
        Calculates the kernelized velocity field for the entire ensemble.

        The velocity of each particle is influenced by every other particle
        in the ensemble through the kernel matrix K, ensuring a cohesive
        migration that preserves the distribution's shape.

        Detailed Logic:
        ---------------
        1. **Posterior Gradients**: Computes the grad-log-posterior for
           each particle. This involves a sparse update (tf.scatter_nd)
           for systems where only a subset of states are observed.
        2. **Kernel Interaction (K)**: Computes the Np x Np interaction matrix.
           The matrix 'matrix' kernel type allows for coordinate-specific
           sensitivity.
        3. **RKHS Projection**: Sums the weighted gradients and kernel
           derivatives to find the optimal direction of KL-divergence
           minimization.

        Parameters:
            current_X (tf.Tensor): Current ensemble positions [N, Dim].
            y_obs (tf.Tensor): Current observation.
            obs_idx (tf.Tensor): Indices of observed states.
            alpha (float): Bandwidth scaling parameter.

        Returns:
            tf.Tensor: Velocity vector for each particle [N, Dim].
        """
        Np = tf.shape(current_X)[0]
        Nx = tf.shape(current_X)[1]

        # Nested function for local gradient calculation
        def get_grad_log_posterior(x):
            # 1. Likelihood Gradient
            x_obs = tf.gather(x, obs_idx)
            innov = y_obs - x_obs
            grad_lik_sub = innov * self.R_inv
            indices = tf.expand_dims(obs_idx, 1)
            grad_lik = tf.scatter_nd(indices, grad_lik_sub, [Nx])

            # 2. Prior Gradient
            grad_prior = -(x - self.prior_mean) / self.prior_var
            return grad_lik + grad_prior

        # Map gradient calculation over the ensemble
        grad_log_p_list = tf.map_fn(get_grad_log_posterior, current_X)

        # Compute Pairwise Differences for Kernel Matrix
        # X_j - X_i creates a [N, N, Dim] tensor of relative distances
        X_j = tf.expand_dims(current_X, 0)
        X_i = tf.expand_dims(current_X, 1)
        diffs = X_j - X_i

        # Bandwidth scaling based on ensemble variance and alpha
        scale = alpha * self.prior_var

        if self.kernel_type == 'scalar':
            # Compute Euclidean distances for scalar kernel
            dist_sq = tf.reduce_sum(tf.square(diffs) / scale, axis=2)
            K_mat = tf.exp(-0.5 * dist_sq) # Shape: [N, N]

            # Gradient of the kernel: Derivative relative to particle position
            grad_K = tf.expand_dims(K_mat, 2) * (diffs / scale)

            # Reconstruct velocity: K * grad_log_p + grad_K
            term1 = tf.expand_dims(K_mat, 2) * tf.expand_dims(grad_log_p, 0)
            total = term1 + grad_K
        else:
            # Coordinate-wise Matrix Kernel
            dist_sq_vec = tf.square(diffs) / scale
            K_vec = tf.exp(-0.5 * dist_sq_vec) # Shape: [N, N, Dim]

            grad_K = K_vec * (diffs / scale)
            term1 = K_vec * tf.expand_dims(grad_log_p, 0)
            total = term1 + grad_K

        # Average the influence across all particles (Empirical Mean in RKHS)
        flow = tf.reduce_mean(total, axis=1) * self.D
        return flow
