"""
Exact Daum-Huang (EDH) Flow Solver
==================================
Implements the global velocity field derivation based on the Daum-Huang (2010)
framework. This solver derives a deterministic ODE to migrate particles from
 the prior to the posterior distribution by gradually introducing the
 likelihood information.

Mathematical Context:
---------------------
The EDH solver assumes that the Jacobian of the measurement function is
approximately constant across the particle ensemble. This linearization
leads to a highly efficient, vectorized calculation of the velocity field,
making it particularly suitable for systems with mild non-linearities where
computational speed is a priority.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
from ..classical import DTYPE, safe_inv

class EDHSolver(tf.Module):
    """
    Exact Daum-Huang (EDH) Velocity Field Generator.

    This class computes the drift (velocity) and divergence of the particle
    ensemble at a specific homotopy coordinate (lambda).
    """
    def __init__(self, name=None):
        """
        Initializes the EDH Solver.
        """
        super().__init__(name=name)

    def __call__(self, z, eta, P_prev, lam, model):
        """
        Calculates the velocity field and divergence for the ensemble.

        The velocity field is derived from the requirement that the particle
        density satisfies the Fokker-Planck equation (without the diffusion term)
        to match the log-homotopy gradient of the posterior.

        Args:
            z: Current observation vector.
            eta: Current particle ensemble [N, Dim].
            P_prev: The estimated state covariance from the previous timestep.
            lam: The homotopy parameter (lambda), ranging from 0 to 1.
            model: The state-space model containing the observation function h(x).

        Returns:
            drift: [N, Dim] tensor representing the velocity of each particle.
            div: Scalar tensor of the flow divergence (Trace of the Flow Jacobian).
        """

        # 1. Linearization at the Ensemble Mean
        # -------------------------------------
        # We approximate the measurement Jacobian H at the current centroid of
        # the ensemble. This 'global' linearization enables efficient
        # vectorized matrix operations.
        x_m = tf.reduce_mean(eta, axis=0)
        H = model.jacobian_h(x_m)

        # 2. Compute Homotopy Gain and Flow Matrix
        # ----------------------------------------
        # S represents the 'inflated' innovation covariance at the current
        # stage of the homotopy (lambda).
        # S = lambda * H * P * H^T + R
        S = lam * H @ P_prev @ tf.transpose(H) + model.R_filter

        # K is the homotopy gain matrix, analogous to the Kalman Gain.
        K = P_prev @ tf.transpose(H) @ safe_inv(S)

        # A is the Flow Jacobian matrix. It dictates the 'stretching' or
        # 'compression' of the ensemble space.
        A = -0.5 * K @ H

        # 3. Calculate Vectorized Drift
        # -----------------------------
        # The drift is composed of two forces:
        # a) A * (x - mean): Moves particles based on the local curvature.
        # b) K * (z - h(x)): Pulls particles toward the observed data.
        innov = tf.cast(z, DTYPE) - model.h_func(eta)

        # Vectorized implementation for N particles:
        # eta - x_m: Centering particles around the mean.
        # K @ innov: Innovation pull for each individual particle.
        drift = tf.transpose(A @ tf.transpose(eta - x_m)) + \
                tf.transpose(K @ tf.transpose(innov))

        # 4. Volume Correction (Log-Det Tracking)
        # ---------------------------------------
        # Because the flow is deterministic, we must track the change in the
        # particle density volume to correctly update the importance weights.
        # The divergence of the flow field f is equal to the Trace of its
        # Jacobian matrix A.
        div = tf.linalg.trace(A)

        return drift, div
