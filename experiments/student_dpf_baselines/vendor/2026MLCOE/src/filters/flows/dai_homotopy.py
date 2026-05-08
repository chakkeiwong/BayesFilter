"""
Dai Homotopy Module
===================
This module implements the Optimized Homotopy Schedule for Particle Flow,
as proposed by Dai et al. (2022). It specifically addresses 'High-Stiffness'
regimes where observation noise is extremely small, causing standard
linear schedules to fail.

Methodology:
------------
The module solves a Boundary Value Problem (BVP) where the objective is
to find a schedule beta(lambda) such that the condition number of the
Hessian of the log-posterior remains minimized across the flow.
"""

import tensorflow as tf
import numpy as np

DTYPE = tf.float64

class LEDHOptimizedFlow:
    """
    Optimized Particle Flow Engine (Dai 22).

    Utilizes a shooting method to solve for a non-linear schedule that
    'slows down' the flow in regions of high numerical sensitivity.
    """

    def __init__(self, model, n_steps=100, mu=0.2):
        """
        Initializes the BVP-optimized flow engine.

        Detailed Parameter Description:
        -------------------------------
        :param model: The SSM object. The BVP solver relies on the
            `model.P_prior` (prediction uncertainty) and `model.R_filter`
            (measurement precision) to determine the stiffness geometry.
        :param n_steps (int): The number of discretization points for
            the shooting method. Higher values provide a smoother
            schedule but increase initialization time. Defaults to 100.
        :param mu (float): The stiffness penalty parameter.
            - Low mu: Behaves like a standard linear schedule.
            - High mu: Aggressively decelerates the flow when the Hessian
              condition number is high.
        """
        self.m = model
        self.n_steps = n_steps
        self.mu = mu
        self.lambdas = np.linspace(0, 1, n_steps)
        self.dt = 1.0 / (n_steps - 1)

    def solve_dai_schedule(self, x_pred_mean):
        """
        Solves the Dai (22) Boundary Value Problem via the Shooting Method.

        This calculates the 'beta' schedule before the particles are moved.
        The goal is to find beta(lambda) such that beta(0)=0 and beta(1)=1,
        while following the 'least stiffness' acceleration field.
        """
        # Prior and Measurement Hessians
        P_inv = tf.linalg.inv(self.m.P_prior)
        H = self.m.jacobian_h(x_pred_mean)
        R_inv = self.m.R_inv_filter
        H_hess = tf.transpose(H) @ R_inv @ H

        def get_accel(b_val):
            """
            Calculates the acceleration d^2_beta/d_lambda^2.
            This corresponds to the derivative of the nuclear norm proxy
            for the Hessian condition number.
            """
            b = tf.cast(tf.clip_by_value(b_val, 0.0, 1.1), DTYPE)
            M = P_inv + b * H_hess
            M_inv = tf.linalg.inv(M)

            # Trace-based derivatives for numerical efficiency
            t1 = tf.linalg.trace(H_hess) * tf.linalg.trace(M_inv)
            t2 = tf.linalg.trace(M) * tf.linalg.trace(M_inv @ H_hess @ M_inv)
            return -self.mu * (t1 + t2)

        def simulate(v0):
            """Integrates the schedule for a candidate initial velocity v0."""
            beta = np.zeros(self.n_steps)
            v = v0
            for i in range(self.n_steps - 1):
                beta[i+1] = beta[i] + v * self.dt
                v += get_accel(beta[i+1]).numpy() * self.dt
            return beta

        # Bisection search (Shooting Method) to satisfy beta(1) = 1.0
        low, high = 1.0, 30.0
        for _ in range(20):
            v_mid = (low + high) / 2
            if simulate(v_mid)[-1] < 1.0: low = v_mid
            else: high = v_mid

        beta_opt = simulate(low)
        # Numerical gradient of beta for scaling the drift velocity
        dot_beta_opt = np.gradient(beta_opt, self.lambdas)
        return tf.cast(beta_opt, DTYPE), tf.cast(dot_beta_opt, DTYPE)

    def compute_ledh_drift(self, x, z, beta):
        """
        Calculates the velocity and Jacobian gradient for the ensemble.

        This follows the LEDH math but is optimized to follow the
        non-linear schedule calculated by the BVP solver.
        """
        P = self.m.P_prior
        H = self.m.jacobian_h(tf.reduce_mean(x, axis=0))
        R = self.m.R_filter

        # Gain K is scaled by the current 'beta' progression
        S_inv = tf.linalg.inv(beta * H @ P @ tf.transpose(H) + R)
        K = P @ tf.transpose(H) @ S_inv

        # Drift f and its gradient for log-det correction
        f = -K @ (H @ tf.transpose(x) - tf.reshape(z, (-1, 1)))
        grad_f = -K @ H

        return tf.transpose(f), grad_f
