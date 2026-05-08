"""
Batched Kalman filter log-likelihood for the linear-Gaussian observation model.

Processes B parameter sets simultaneously in one pass — no loop over chains.
This is critical for HMC performance: each HMC iteration evaluates
log p(y | theta) for B chains in parallel.

Model:
    h_{t+1} = mu + Phi (h_t - mu) + eta_t,    eta_t ~ N(0, L L^T)
    y_t     = h_t + nu_t,                      nu_t  ~ N(0, Sigma_obs)

All arguments are traced tensors. Gradients flow to (mu, Phi, Sigma_eta_chol).

TF migration guideline compliance:
  - @tf.function(jit_compile=True) for XLA fusion
  - No tf.while_loop (uses Python for-loop, unrolled at trace time for XLA)
  - tf.float64 throughout
  - No Python control flow on tensor values
  - No tf.random.Generator (deterministic)
"""

import tensorflow as tf


def batched_kalman_log_likelihood(
    mu: tf.Tensor,
    Phi: tf.Tensor,
    Sigma_eta_chol: tf.Tensor,
    Sigma_obs: tf.Tensor,
    observations: tf.Tensor,
    T_static: int,
) -> tf.Tensor:
    """
    Batched exact log p(y_{1:T} | theta) via the Kalman filter.

    Args:
        mu:             [B, d]
        Phi:            [B, d, d]
        Sigma_eta_chol: [B, d, d] lower-triangular
        Sigma_obs:      [d, d] observation noise (shared across batch)
        observations:   [T, d]
        T_static:       Python int — number of time steps (needed for
                        Python for-loop unrolling under jit_compile)

    Returns:
        log_likelihood: [B]
    """
    dtype = mu.dtype
    B = tf.shape(mu)[0]
    d = tf.shape(mu)[1]

    # Dynamics noise covariance: Q = L @ L^T, batched [B, d, d]
    Q = tf.matmul(Sigma_eta_chol, Sigma_eta_chol, transpose_b=True)
    Q = 0.5 * (Q + tf.linalg.matrix_transpose(Q))

    # Observation noise: [d, d] (shared)
    R = 0.5 * (Sigma_obs + tf.transpose(Sigma_obs))

    # Bias: (I - Phi) @ mu, batched
    I_d = tf.eye(d, dtype=dtype)
    I_batch = tf.broadcast_to(I_d, tf.shape(Phi))             # [B, d, d]
    bias = tf.linalg.matvec(I_batch - Phi, mu)                # [B, d]

    # Initial state
    m = mu                                                     # [B, d]
    P = Q + tf.constant(1e-6, dtype=dtype) * I_batch           # [B, d, d]
    ll_accum = tf.zeros([B], dtype=dtype)

    # R broadcast to batch
    R_batch = tf.broadcast_to(R[tf.newaxis, :, :], tf.shape(P))  # [B, d, d]

    pi = tf.constant(3.141592653589793, dtype=dtype)
    d_f = tf.cast(d, dtype)
    eps_eye = tf.constant(1e-10, dtype=dtype) * I_batch

    # Python for-loop: unrolled at trace time for XLA
    for t in range(T_static):
        y = observations[t]                                    # [d]

        # --- Predict ---
        # m_pred = Phi @ m + bias,  batched
        m_pred = tf.linalg.matvec(Phi, m) + bias               # [B, d]

        # P_pred = Phi @ P @ Phi^T + Q
        P_pred = tf.matmul(tf.matmul(Phi, P),
                           Phi, transpose_b=True) + Q           # [B, d, d]
        P_pred = 0.5 * (P_pred + tf.linalg.matrix_transpose(P_pred))

        # --- Update ---
        v = y[tf.newaxis, :] - m_pred                          # [B, d]
        S = P_pred + R_batch                                    # [B, d, d]
        S = 0.5 * (S + tf.linalg.matrix_transpose(S))

        # Kalman gain: K = P_pred @ S^{-1}
        # Solve S @ K^T = P_pred^T  =>  K^T = S^{-1} @ P_pred^T
        K = tf.linalg.matrix_transpose(
            tf.linalg.solve(S, tf.linalg.matrix_transpose(P_pred))
        )                                                       # [B, d, d]

        m = m_pred + tf.linalg.matvec(K, v)                    # [B, d]

        # Joseph form
        IK = I_batch - K                                        # C = I
        P = (tf.matmul(tf.matmul(IK, P_pred),
                       IK, transpose_b=True)
             + tf.matmul(tf.matmul(K, R_batch),
                         K, transpose_b=True))
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))

        # Log-likelihood increment
        S_chol = tf.linalg.cholesky(S + eps_eye)
        S_logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(S_chol)), axis=-1
        )                                                       # [B]
        # Solve L^{-1} v for Mahalanobis
        solved = tf.linalg.triangular_solve(
            S_chol, v[:, :, tf.newaxis], lower=True
        )                                                       # [B, d, 1]
        mahal_sq = tf.reduce_sum(tf.squeeze(solved, -1) ** 2,
                                 axis=-1)                       # [B]

        ll_accum = ll_accum - 0.5 * (
            d_f * tf.math.log(2.0 * pi) + S_logdet + mahal_sq
        )

    return ll_accum
