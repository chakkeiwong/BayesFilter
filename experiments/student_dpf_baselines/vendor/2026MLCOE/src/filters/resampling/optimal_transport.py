"""
Optimal Transport Resampling Module
===================================
Implements entropy-regularized Optimal Transport (EOT) via the Sinkhorn
algorithm. This provides a transport-based resampling that is fully
differentiable and mathematically robust.

Mathematical Context:
---------------------
Unlike standard resampling, which is a discrete branching process, Sinkhorn
resampling treats the ensemble transition as a continuous optimal coupling
problem. It finds a transport plan P that minimizes the L2 cost of moving
mass from the weighted prior distribution to a uniform target distribution.
The resulting 'resampled' particles are a barycentric projection of the
original set, ensuring a smooth and differentiable gradient path.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf

class SinkhornResampler:
    """
    Sinkhorn-based Differentiable Resampler.

    Calculates the transport plan that migrates a weighted ensemble to a
    uniform ensemble with minimal displacement cost.
    """

    def __init__(self, epsilon=0.05, n_iter=10, thresh=1e-3):
        """
        Initializes the Sinkhorn engine.

        :param epsilon (float): Regularization parameter for the entropy term.
            Smaller epsilon leads to sharper transport plans but is harder
            to optimize and numerically unstable.
        :param n_iter (int): Maximum number of Sinkhorn unrolling iterations.
        :param thresh (float): Convergence threshold for the scaling factors.
        """
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.thresh = thresh

    def resample(self, x, lw, h=None):
        """
        Executes the Sinkhorn-EOT resampling.

        Detailed Logic:
        ---------------
        1. **Cost Construction**: Computes the L2 distance between particles.
        2. **Log-Domain Sinkhorn**: Updates the dual variables f and g.
           Using the log-sum-exp trick (as in bonus1a_extended.py)
           prevents exponential overflow.
        3. **Barycentric Projection**: Maps particles and (optional) hidden
           states h through the transport plan.

        :param x: Particle states [N, Dim].
        :param lw: Log-importance weights [N].
        :param h: (Optional) Hidden states of an RNN/LSTM [N, Hidden_Dim].
        """
        N = tf.shape(x)[0]
        epsilon = tf.cast(self.epsilon, x.dtype)

        # Build squared distance cost matrix
        C = tf.reduce_sum(tf.square(x[:, None, :] - x[None, :, :]), axis=-1)
        C = C / (tf.stop_gradient(tf.reduce_max(C)) + 1e-6)

        log_a = tf.fill((N,), -tf.math.log(tf.cast(N, x.dtype)))
        log_b = tf.cast(lw, x.dtype)

        # Initialize dual variables
        f = tf.zeros_like(log_a)
        g = tf.zeros_like(log_b)

        # Use a fixed range for unrolling if possible, or remove the break
        # to stay within Autograph's limitations for basic for-loops.
        for _ in range(self.n_iter):
            f = epsilon * (log_a - tf.reduce_logsumexp((g[None, :] - C) / epsilon, axis=1))
            g = epsilon * (log_b - tf.reduce_logsumexp((f[:, None] - C) / epsilon, axis=0))
            # REMOVED: if tf.reduce_mean(tf.abs(g_new - g)) < self.thresh: break
            # Logic: In a graph environment, fixed iterations are more stable for XLA.

        P = tf.exp((f[:, None] + g[None, :] - C) / epsilon)

        res_x = tf.cast(N, x.dtype) * tf.matmul(P, x)
        res_h = None
        if h is not None:
             res_h = tf.nest.map_structure(lambda s: tf.cast(N, x.dtype) * tf.matmul(P, s), h)

        return res_x, res_h
