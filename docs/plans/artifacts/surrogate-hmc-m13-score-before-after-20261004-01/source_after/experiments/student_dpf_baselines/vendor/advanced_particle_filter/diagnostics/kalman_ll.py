"""
Standalone Kalman filter log-likelihood as a pure tensor function.

Unlike TFKalmanFilter (which wraps params in tf.constant via TFStateSpaceModel),
this module takes (mu, Phi, Sigma_eta_chol, Sigma_obs, observations) as raw
tensors. Gradients flow through all of them via GradientTape.

This is the ground-truth gradient for the linear-Gaussian observation model.
Used in diagnostics to validate DPF gradient quality.

TF migration guideline compliance:
  - @tf.function(reduce_retracing=True) on the main function
  - tf.while_loop for the time loop
  - TensorArray for per-step log-likelihood accumulation
  - tf.float64 throughout
  - No Python control flow on tensor values
"""

import tensorflow as tf


@tf.function(reduce_retracing=True)
def kalman_log_likelihood(
    mu: tf.Tensor,
    Phi: tf.Tensor,
    Sigma_eta_chol: tf.Tensor,
    Sigma_obs: tf.Tensor,
    observations: tf.Tensor,
) -> tf.Tensor:
    """
    Exact log p(y_{1:T} | mu, Phi, Sigma_eta_chol) via the Kalman filter
    for the linear-Gaussian observation model:

        h_{t+1} = mu + Phi (h_t - mu) + eta_t,    eta_t ~ N(0, L L^T)
        y_t     = h_t + nu_t,                      nu_t  ~ N(0, Sigma_obs)

    All arguments are traced tensors. Gradients flow to (mu, Phi, Sigma_eta_chol).
    Sigma_obs is treated as fixed (not differentiated), but could be if needed.

    Args:
        mu:             [d]
        Phi:            [d, d]
        Sigma_eta_chol: [d, d] lower-triangular
        Sigma_obs:      [d, d] observation noise covariance (fixed)
        observations:   [T, d]

    Returns:
        log_likelihood: scalar
    """
    dtype = mu.dtype
    d = tf.shape(mu)[0]
    T = tf.shape(observations)[0]

    # Dynamics noise covariance
    Q = Sigma_eta_chol @ tf.transpose(Sigma_eta_chol)
    Q = 0.5 * (Q + tf.transpose(Q))

    # Observation noise
    R = 0.5 * (Sigma_obs + tf.transpose(Sigma_obs))

    # Dynamics: h_{t+1} = Phi h_t + (I - Phi) mu + eta
    #   A = Phi,  bias = (I - Phi) mu
    #   Observation: C = I
    I_d = tf.eye(d, dtype=dtype)
    bias = tf.linalg.matvec(I_d - Phi, mu)               # [d]

    # Initial state: N(mu, Q) as approximation
    m = mu
    P = Q + tf.constant(1e-6, dtype=dtype) * I_d

    # Accumulate log-likelihood
    t0 = tf.constant(0, dtype=tf.int32)
    ll0 = tf.constant(0.0, dtype=dtype)

    def body(t, m, P, ll_accum):
        y = observations[t]                                # [d]

        # --- Predict ---
        m_pred = tf.linalg.matvec(Phi, m) + bias          # [d]
        P_pred = Phi @ P @ tf.transpose(Phi) + Q          # [d, d]
        P_pred = 0.5 * (P_pred + tf.transpose(P_pred))

        # --- Update ---
        # Observation: y = h + nu,  C = I
        # Innovation
        v = y - m_pred                                     # [d]
        S = P_pred + R                                     # [d, d]
        S = 0.5 * (S + tf.transpose(S))

        # Kalman gain: K = P_pred @ S^{-1}
        K = tf.transpose(tf.linalg.solve(S, tf.transpose(P_pred)))  # [d, d]

        m_upd = m_pred + tf.linalg.matvec(K, v)           # [d]

        # Joseph form for numerical stability
        IKC = I_d - K                                      # C = I
        P_upd = IKC @ P_pred @ tf.transpose(IKC) + K @ R @ tf.transpose(K)
        P_upd = 0.5 * (P_upd + tf.transpose(P_upd))

        # Log-likelihood increment: log N(v; 0, S)
        S_chol = tf.linalg.cholesky(
            S + tf.constant(1e-10, dtype=dtype) * I_d
        )
        S_logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(S_chol))
        )
        solved = tf.linalg.triangular_solve(
            S_chol, v[:, tf.newaxis], lower=True
        )
        mahal_sq = tf.reduce_sum(solved ** 2)

        d_f = tf.cast(d, dtype)
        pi = tf.constant(3.141592653589793, dtype=dtype)
        log_lik_t = -0.5 * (d_f * tf.math.log(2.0 * pi) + S_logdet + mahal_sq)

        return t + 1, m_upd, P_upd, ll_accum + log_lik_t

    _, _, _, total_ll = tf.while_loop(
        cond=lambda t, *_: t < T,
        body=body,
        loop_vars=(t0, m, P, ll0),
        maximum_iterations=T,
    )

    return total_ll
