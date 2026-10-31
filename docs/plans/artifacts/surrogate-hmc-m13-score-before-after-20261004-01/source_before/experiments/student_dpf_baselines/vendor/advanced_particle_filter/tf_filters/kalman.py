"""
TensorFlow Kalman Filter implementations.

Mirrors: filters/kalman.py (NumPy version)

- TFKalmanFilter: Standard Kalman filter for linear Gaussian models
- TFExtendedKalmanFilter: EKF for nonlinear models with Gaussian noise
- TFUnscentedKalmanFilter: UKF using sigma points (2n+1 scheme)

All filter() methods are tf.function-compatible. The time loop uses
tf.while_loop for JIT compilation without retracing.
"""

import tensorflow as tf
from typing import Optional

from .base import TFFilterResult
from ..tf_models.base import TFStateSpaceModel


# ============================================================================
# Shared Kalman update kernel (used by KF and EKF)
# ============================================================================

def _kalman_update(m_pred, P_pred, y, H, R):
    """
    Kalman update step — pure TF ops.

    Args:
        m_pred: [nx] Predicted mean
        P_pred: [nx, nx] Predicted covariance
        y: [ny] Observation
        H: [ny, nx] Observation matrix
        R: [ny, ny] Observation noise covariance

    Returns:
        m_upd: [nx] Updated mean
        P_upd: [nx, nx] Updated covariance
        log_lik: Scalar log likelihood of observation
    """
    dtype = m_pred.dtype
    nx = tf.shape(m_pred)[0]
    ny = tf.shape(y)[0]

    # Innovation
    y_pred = tf.linalg.matvec(H, m_pred)  # [ny]
    v = y - y_pred  # [ny]

    # Innovation covariance: S = H @ P_pred @ H.T + R
    S = H @ P_pred @ tf.transpose(H) + R
    S = 0.5 * (S + tf.transpose(S))

    # Kalman gain: K = P_pred @ H.T @ S^{-1}
    # Solve S @ K.T = H @ P_pred  =>  K.T = S^{-1} @ (H @ P_pred)
    K = tf.transpose(tf.linalg.solve(S, H @ P_pred))  # [nx, ny]

    # Updated mean
    m_upd = m_pred + tf.linalg.matvec(K, v)

    # Updated covariance (Joseph form for numerical stability)
    I = tf.eye(nx, dtype=dtype)
    IKH = I - K @ H
    P_upd = IKH @ P_pred @ tf.transpose(IKH) + K @ R @ tf.transpose(K)
    P_upd = 0.5 * (P_upd + tf.transpose(P_upd))

    # Log likelihood
    S_chol = tf.linalg.cholesky(S)
    S_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)))
    solved = tf.linalg.triangular_solve(S_chol, v[:, tf.newaxis], lower=True)
    mahal_sq = tf.reduce_sum(solved ** 2)

    ny_f = tf.cast(ny, dtype)
    pi = tf.constant(3.141592653589793, dtype=dtype)
    log_lik = -0.5 * (ny_f * tf.math.log(2.0 * pi) + S_logdet + mahal_sq)

    return m_upd, P_upd, log_lik


# ============================================================================
# KalmanFilter
# ============================================================================

class TFKalmanFilter:
    """
    Standard Kalman Filter for linear Gaussian models (TF version).

    Optimal for linear dynamics and linear observations with Gaussian noise.
    """

    def __init__(self):
        pass

    def filter(
        self,
        model: TFStateSpaceModel,
        observations: tf.Tensor,
    ) -> TFFilterResult:
        """
        Run Kalman filter on observations.
        Args:
            model: TFStateSpaceModel (should be linear Gaussian)
            observations: [T, ny] Observations

        Returns:
            TFFilterResult with filtered means and covariances
        """
        means, covariances, log_liks = self._filter_impl(model, observations)
        return TFFilterResult(
            means=means,
            covariances=covariances,
            log_likelihood=tf.reduce_sum(log_liks),
            log_likelihood_increments=log_liks,
        )

    @tf.function
    def _filter_impl(self, model, observations):
        # Use tf.while_loop for filtering.
        # A Python for loop with T=100 would create 100 copies of every operation in the graph.
        # tf.while_loop creates one copy that gets reused. Much smaller graph, much faster tracing.
        dtype = model.dtype
        T = tf.shape(observations)[0]
        nx = model.state_dim

        # Get constant model matrices
        dummy_x = tf.zeros([nx], dtype=dtype)
        F = model.dynamics_jacobian(dummy_x)
        H = model.obs_jacobian(dummy_x)
        Q = model.dynamics_cov
        R = model.obs_cov

        # TensorArrays for output
        means_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        covs_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        loglik_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)

        # Initialize
        m = model.initial_mean
        P = model.initial_cov
        means_ta = means_ta.write(0, m)
        covs_ta = covs_ta.write(0, P)

        # Loop body
        def body(t, m, P, means_ta, covs_ta, loglik_ta):
            y = observations[t]

            # Predict
            # Use dynamics_mean (not F @ m) to support affine dynamics
            # h_{t+1} = F h_t + b + noise.  dynamics_mean handles the bias.
            m_pred = model.dynamics_mean(m[tf.newaxis, :])[0]
            P_pred = F @ P @ tf.transpose(F) + Q
            P_pred = 0.5 * (P_pred + tf.transpose(P_pred))

            # Update
            m_upd, P_upd, log_lik = _kalman_update(m_pred, P_pred, y, H, R)

            means_ta = means_ta.write(t + 1, m_upd)
            covs_ta = covs_ta.write(t + 1, P_upd)
            loglik_ta = loglik_ta.write(t, log_lik)

            return t + 1, m_upd, P_upd, means_ta, covs_ta, loglik_ta

        _, _, _, means_ta, covs_ta, loglik_ta = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=body,
            loop_vars=(0, m, P, means_ta, covs_ta, loglik_ta),
        )

        return means_ta.stack(), covs_ta.stack(), loglik_ta.stack()


# ============================================================================
# ExtendedKalmanFilter
# ============================================================================

class TFExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear models (TF version).

    Linearizes dynamics and observation functions around current estimate.
    """

    def __init__(self):
        pass

    def filter(
        self,
        model: TFStateSpaceModel,
        observations: tf.Tensor,
    ) -> TFFilterResult:
        means, covariances, log_liks = self._filter_impl(model, observations)
        return TFFilterResult(
            means=means,
            covariances=covariances,
            log_likelihood=tf.reduce_sum(log_liks),
            log_likelihood_increments=log_liks,
        )

    @tf.function
    def _filter_impl(self, model, observations):
        dtype = model.dtype
        T = tf.shape(observations)[0]
        nx = model.state_dim

        Q = model.dynamics_cov
        R = model.obs_cov

        means_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        covs_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        loglik_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)

        m = model.initial_mean
        P = model.initial_cov
        means_ta = means_ta.write(0, m)
        covs_ta = covs_ta.write(0, P)

        def body(t, m, P, means_ta, covs_ta, loglik_ta):
            y = observations[t]

            # Predict
            F = model.dynamics_jacobian(m)
            m_pred = model.dynamics_mean(m[tf.newaxis, :])[0]
            P_pred = F @ P @ tf.transpose(F) + Q
            P_pred = 0.5 * (P_pred + tf.transpose(P_pred))

            # Update
            H = model.obs_jacobian(m_pred)
            y_pred = model.obs_mean(m_pred[tf.newaxis, :])[0]
            v = y - y_pred

            S = H @ P_pred @ tf.transpose(H) + R
            S = 0.5 * (S + tf.transpose(S))

            K = tf.transpose(tf.linalg.solve(S, H @ P_pred))
            m_upd = m_pred + tf.linalg.matvec(K, v)

            I = tf.eye(nx, dtype=dtype)
            IKH = I - K @ H
            P_upd = IKH @ P_pred @ tf.transpose(IKH) + K @ R @ tf.transpose(K)
            P_upd = 0.5 * (P_upd + tf.transpose(P_upd))

            # Log likelihood
            S_chol = tf.linalg.cholesky(S)
            S_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)))
            solved = tf.linalg.triangular_solve(S_chol, v[:, tf.newaxis], lower=True)
            mahal_sq = tf.reduce_sum(solved ** 2)
            ny_f = tf.cast(tf.shape(y)[0], dtype)
            pi = tf.constant(3.141592653589793, dtype=dtype)
            log_lik = -0.5 * (ny_f * tf.math.log(2.0 * pi) + S_logdet + mahal_sq)

            means_ta = means_ta.write(t + 1, m_upd)
            covs_ta = covs_ta.write(t + 1, P_upd)
            loglik_ta = loglik_ta.write(t, log_lik)

            return t + 1, m_upd, P_upd, means_ta, covs_ta, loglik_ta

        _, _, _, means_ta, covs_ta, loglik_ta = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=body,
            loop_vars=(0, m, P, means_ta, covs_ta, loglik_ta),
        )

        return means_ta.stack(), covs_ta.stack(), loglik_ta.stack()


# ============================================================================
# UnscentedKalmanFilter
# ============================================================================

class TFUnscentedKalmanFilter:
    """
    Unscented Kalman Filter using sigma points (TF version).

    Uses 2n+1 sigma points. Default parameters (alpha=1, beta=2, kappa=3-n)
    match the ekfukf MATLAB toolbox.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: Optional[float] = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self._kappa = kappa  # None means 3-n

    def filter(
        self,
        model: TFStateSpaceModel,
        observations: tf.Tensor,
    ) -> TFFilterResult:
        means, covariances, log_liks = self._filter_impl(model, observations)
        return TFFilterResult(
            means=means,
            covariances=covariances,
            log_likelihood=tf.reduce_sum(log_liks),
            log_likelihood_increments=log_liks,
        )

    def _compute_weights(self, n: int, dtype: tf.DType):
        """Compute sigma point weights (Python-level, called once)."""
        kappa = self._kappa if self._kappa is not None else (3 - n)
        lam = self.alpha ** 2 * (n + kappa) - n

        Wm = tf.fill([2 * n + 1], tf.constant(0.5 / (n + lam), dtype=dtype))
        Wm = tf.tensor_scatter_nd_update(
            Wm, [[0]], [tf.constant(lam / (n + lam), dtype=dtype)]
        )

        Wc = tf.tensor_scatter_nd_update(
            Wm, [[0]],
            [Wm[0] + tf.constant(1 - self.alpha ** 2 + self.beta, dtype=dtype)]
        )

        return Wm, Wc, tf.constant(lam, dtype=dtype)

    @tf.function
    def _filter_impl(self, model, observations):
        dtype = model.dtype
        T = tf.shape(observations)[0]
        nx = model.state_dim
        ny = model.obs_dim

        Q = model.dynamics_cov
        R = model.obs_cov

        Wm, Wc, lam = self._compute_weights(nx, dtype)
        n_sigma = 2 * nx + 1

        means_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        covs_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        loglik_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)

        m = model.initial_mean
        P = model.initial_cov
        means_ta = means_ta.write(0, m)
        covs_ta = covs_ta.write(0, P)

        def sigma_points(m, P):
            """Generate [2n+1, nx] sigma points."""
            scaling = tf.cast(nx, dtype) + lam

            # Regularize P
            P_reg = 0.5 * (P + tf.transpose(P))
            min_eig = tf.reduce_min(tf.linalg.eigvalsh(P_reg))
            P_reg = tf.cond(
                min_eig < 1e-10,
                lambda: P_reg + (1e-6 - tf.minimum(min_eig, 0.0)) * tf.eye(nx, dtype=dtype),
                lambda: P_reg,
            )

            # Fallback scaling
            scaling = tf.cond(
                scaling <= 0.0,
                lambda: tf.cast(nx, dtype),
                lambda: scaling,
            )

            # Cholesky of scaling * P
            L = tf.linalg.cholesky(scaling * P_reg)

            # Build sigma points: [0]=m, [1..n]=m+L[:,i], [n+1..2n]=m-L[:,i]
            cols = tf.transpose(L)  # [nx, nx] -> rows are columns of L
            sigma_plus = m + cols   # [nx, nx] broadcast
            sigma_minus = m - cols  # [nx, nx]
            return tf.concat([m[tf.newaxis, :], sigma_plus, sigma_minus], axis=0)

        def body(t, m, P, means_ta, covs_ta, loglik_ta):
            y = observations[t]

            # ---- Predict ----
            sigma = sigma_points(m, P)  # [2n+1, nx]
            sigma_pred = model.dynamics_mean(sigma)  # [2n+1, nx]

            m_pred = tf.reduce_sum(Wm[:, tf.newaxis] * sigma_pred, axis=0)
            diff = sigma_pred - m_pred
            P_pred = tf.einsum('i,ij,ik->jk', Wc, diff, diff) + Q
            P_pred = 0.5 * (P_pred + tf.transpose(P_pred))

            # ---- Update ----
            sigma2 = sigma_points(m_pred, P_pred)  # [2n+1, nx]
            sigma_y = model.obs_mean(sigma2)  # [2n+1, ny]

            y_pred = tf.reduce_sum(Wm[:, tf.newaxis] * sigma_y, axis=0)
            v = y - y_pred

            diff_x = sigma2 - m_pred  # [2n+1, nx]
            diff_y = sigma_y - y_pred  # [2n+1, ny]

            Pxy = tf.einsum('i,ij,ik->jk', Wc, diff_x, diff_y)  # [nx, ny]
            S = tf.einsum('i,ij,ik->jk', Wc, diff_y, diff_y) + R  # [ny, ny]
            S = 0.5 * (S + tf.transpose(S))

            # Cholesky of S with regularization
            S_chol = tf.linalg.cholesky(S + 1e-10 * tf.eye(ny, dtype=dtype))

            # Kalman gain via Cholesky solve
            temp = tf.linalg.triangular_solve(S_chol, tf.transpose(Pxy), lower=True)
            K = tf.transpose(
                tf.linalg.triangular_solve(tf.transpose(S_chol), temp, lower=False)
            )

            m_upd = m_pred + tf.linalg.matvec(K, v)

            # P_upd = P_pred - K @ S @ K.T  via Cholesky
            KS_chol = K @ S_chol
            P_upd = P_pred - KS_chol @ tf.transpose(KS_chol)
            P_upd = 0.5 * (P_upd + tf.transpose(P_upd))

            # Ensure positive definiteness
            min_eig = tf.reduce_min(tf.linalg.eigvalsh(P_upd))
            P_upd = tf.cond(
                min_eig < 0.0,
                lambda: P_upd + (1e-6 - min_eig) * tf.eye(nx, dtype=dtype),
                lambda: P_upd,
            )

            # Log likelihood
            S_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)))
            solved = tf.linalg.triangular_solve(S_chol, v[:, tf.newaxis], lower=True)
            mahal_sq = tf.reduce_sum(solved ** 2)
            ny_f = tf.cast(ny, dtype)
            pi = tf.constant(3.141592653589793, dtype=dtype)
            log_lik = -0.5 * (ny_f * tf.math.log(2.0 * pi) + S_logdet + mahal_sq)

            means_ta = means_ta.write(t + 1, m_upd)
            covs_ta = covs_ta.write(t + 1, P_upd)
            loglik_ta = loglik_ta.write(t, log_lik)

            return t + 1, m_upd, P_upd, means_ta, covs_ta, loglik_ta

        _, _, _, means_ta, covs_ta, loglik_ta = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=body,
            loop_vars=(0, m, P, means_ta, covs_ta, loglik_ta),
        )

        return means_ta.stack(), covs_ta.stack(), loglik_ta.stack()
