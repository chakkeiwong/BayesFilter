"""
Local Exact Daum-Huang (LEDH) Flow Solver
=========================================
Implements a particle-specific velocity field derivation. Unlike the global
EDH solver, this "Local" variant handles high non-linearity by evaluating
the local geometry (Jacobians) for every individual particle in the ensemble.

Mathematical Context:
---------------------
By computing a unique Jacobian H_i and Gain K_i for each particle p_i,
the LEDH solver provides a more accurate manifold migration when the
measurement function h(x) is highly non-linear or has high curvature.
While computationally more intensive than EDH, it significantly reduces
bias in complex state-space transitions.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
from ..classical import DTYPE, safe_inv

class LEDHSolver(tf.Module):
    """
    Local Exact Daum-Huang (LEDH) Velocity Field Generator.

    This class leverages particle-wise linearization to calculate
    trajectories that are sensitive to local likelihood gradients.
    """
    def __init__(self, name=None):
        """
        Initializes the LEDH Solver.
        """
        super().__init__(name=name)

    def __call__(self, z, eta, P_prev, lam, model):
        """
        Calculates localized velocity fields and divergences for the ensemble.

        Args:
            z: Current observation vector.
            eta: Current particle ensemble [N, Dim].
            P_prev: The estimated state covariance from the previous timestep.
            lam: The homotopy parameter (lambda), ranging from 0 to 1.
            model: The state-space model containing the observation function h(x).

        Returns:
            drift: [N, Dim] tensor of localized particle velocities.
            div: [N] tensor of localized flow divergences.
        """
        # Calculate the ensemble mean once to be shared by all local calculations.
        # Group cohesion is maintained by centering local drift around this x_m.
        x_m = tf.reduce_mean(eta, axis=0)

        def compute_particle_velocity(p_i):
            """
            Internal function to compute the flow components for a single particle.
            """
            # 1. Localized Linearization
            # --------------------------
            # Evaluate the Jacobian H_i specifically at the particle's
            # current location p_i rather than the ensemble mean.
            H_i = model.jacobian_h(p_i)

            # 2. Localized Gain Calculations
            # ------------------------------
            # S_i represents the local innovation covariance.
            S_i = lam * H_i @ P_prev @ tf.transpose(H_i) + model.R_filter

            # K_i is the localized homotopy gain.
            K_i = P_prev @ tf.transpose(H_i) @ safe_inv(S_i)

            # A_i is the localized Flow Jacobian.
            A_i = -0.5 * K_i @ H_i

            # 3. Particle-specific Innovation
            # -------------------------------
            # Compute the measurement residual for this specific particle.
            innov_i = tf.cast(z, DTYPE) - model.h_func(p_i)

            # 4. Local Drift Calculation
            # --------------------------
            # f_i = A_i * (p_i - x_m) + K_i * innov_i
            # The first term handles the 'spreading/collapsing' of the ensemble,
            # while the second term 'pulls' the particle toward the observation.
            drift_i = tf.linalg.matvec(A_i, p_i - x_m) + \
                      tf.linalg.matvec(K_i, tf.squeeze(innov_i))

            # Each particle carries its own divergence based on local A_i.
            return drift_i, tf.linalg.trace(A_i)

        # 5. Ensemble Mapping
        # -------------------
        # We use tf.map_fn to apply the localized math across the ensemble.
        # On your Mac Studio / M1 Ultra, Autograph will attempt to parallelize
        # this map across available cores.
        drift, div = tf.map_fn(compute_particle_velocity, eta,
                                dtype=(DTYPE, DTYPE))

        return drift, div
