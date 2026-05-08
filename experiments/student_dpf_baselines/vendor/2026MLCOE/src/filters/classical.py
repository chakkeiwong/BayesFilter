"""
Classical and Ensemble Filters Module
=====================================
This module provides TensorFlow-based implementations of foundational
Bayesian filtering algorithms for linear and non-linear state estimation.

Included Filters:
- Kalman Filter (KF): Exact inference for linear Gaussian systems, utilizing
  Joseph-form covariance updates for guaranteed positive semi-definiteness.
- Extended Kalman Filter (EKF): First-order Taylor series linearization for
  mildly non-linear systems.
- Unscented Kalman Filter (UKF): Derivative-free non-linear filtering using
  the Unscented Transform (Sigma Points) to better capture covariance spread.
- Ensemble Square Root Filter (ESRF): A deterministic ensemble-based Kalman
  filter that avoids sampling error during the measurement update via eigenvalue
  decomposition.

Design Principles:
- High Precision: Enforces `float64` arithmetic globally to prevent numerical
  drift in long sequential operations (crucial for Apple Silicon / M1 Ultra).
- Stability: Employs Cholesky-backed safe matrix inversions with fallback jitter.
- Performance: Core prediction and update steps are compiled into optimized
  static graphs using `@tf.function`.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


# Numerical precision and stability constants
DTYPE = tf.float64
JITTER = 1e-6
tfd = tfp.distributions

def safe_inv(matrix):
    """
    Numerically stable inversion for Symmetric Positive Definite (SPD) matrices.

    Mathematical Context:
    ---------------------
    Covariance/Innovation matrices must remain mathematically positive definite.
    Direct inversion (tf.linalg.inv) can compound floating-point errors.
    Cholesky decomposition (L * L^T) is vastly more stable. If the matrix has
    lost positive definiteness due to numerical limits, it falls back to a
    standard inversion with a small Tikhonov regularization (jitter) added to
    the diagonal.
    """
    try:
        L = tf.linalg.cholesky(matrix)
        return tf.linalg.cholesky_solve(L, tf.eye(tf.shape(matrix)[0], dtype=DTYPE))
    except:
        return tf.linalg.inv(matrix + tf.eye(tf.shape(matrix)[0], dtype=DTYPE) * 1e-4)

class KF:
    """
    Kalman Filter (KF).
    -------------------
    Provides exact optimal inference for linear systems with Gaussian noise.
    Uses tf.matmul. Expects state x to be Rank 2 (dim, 1).
    """
    def __init__(self, F, H, Q, R, P0, x0):
        # Cast all system matrices to the global precision format
        self.F = tf.cast(F, DTYPE)
        self.H = tf.cast(H, DTYPE)
        self.Q = tf.cast(Q, DTYPE)
        self.R = tf.cast(R, DTYPE)
        self.dim = self.F.shape[0]
        self.I = tf.eye(self.dim, dtype=DTYPE)

        # Initialize state mean and covariance as trackable tf.Variables
        if isinstance(x0, tf.Variable): self.x = x0
        else: self.x = tf.Variable(tf.cast(x0, DTYPE), dtype=DTYPE)
        if isinstance(P0, tf.Variable): self.P = P0
        else: self.P = tf.Variable(tf.cast(P0, DTYPE), dtype=DTYPE)

    @tf.function
    def predict(self):
        """
        Prediction Step (Time Update).
        Propagates the state mean and covariance through the linear dynamics model.
        """
        x_curr = self.x.read_value()
        P_curr = self.P.read_value()

        # x_{t|t-1} = F * x_{t-1|t-1}
        x_pred = tf.matmul(self.F, x_curr)
        # P_{t|t-1} = F * P_{t-1|t-1} * F^T + Q
        P_pred = tf.matmul(tf.matmul(self.F, P_curr), self.F, transpose_b=True) + self.Q

        self.x.assign(x_pred)
        self.P.assign(P_pred)
        return x_pred, P_pred

    @tf.function
    def update(self, z):
        """
        Measurement Update Step.
        Incorporates a new observation using the Joseph form for stability.
        """
        z = tf.cast(z, DTYPE)
        x_curr = self.x.read_value()
        P_curr = self.P.read_value()

        # Innovation covariance: S = H * P * H^T + R
        S = tf.matmul(tf.matmul(self.H, P_curr), self.H, transpose_b=True) + self.R
        # Optimal Kalman Gain: K = P * H^T * S^-1
        K = tf.matmul(tf.matmul(P_curr, self.H, transpose_b=True), safe_inv(S))

        # Innovation (residual): y = z - H * x
        y_res = z - tf.matmul(self.H, x_curr)
        # State update: x_{t|t} = x_{t|t-1} + K * y
        x_new = x_curr + tf.matmul(K, y_res)

        # Joseph Form Covariance Update: P = (I - KH)P(I - KH)^T + KRK^T
        # This form is computationally heavier but guarantees that P remains
        # symmetric and positive semi-definite despite floating-point rounding.
        I_KH = self.I - tf.matmul(K, self.H)
        term1 = tf.matmul(tf.matmul(I_KH, P_curr), I_KH, transpose_b=True)
        term2 = tf.matmul(tf.matmul(K, self.R), K, transpose_b=True)
        P_new = term1 + term2

        self.x.assign(x_new)
        self.P.assign(P_new)
        return x_new, P_new

class EKF:
    """
    Extended Kalman Filter (EKF).
    -----------------------------
    Approximates non-linear systems by taking the first-order Taylor expansion
    (Jacobian) of the observation function around the current state estimate.
    """
    def __init__(self, model):
        self.m = model
        self.dim = model.state_dim
        self.x = tf.Variable(tf.zeros(self.dim, dtype=DTYPE))
        self.P = tf.Variable(tf.eye(self.dim, dtype=DTYPE))

    def init(self, x, P):
        self.x.assign(x)
        self.P.assign(P)

    @tf.function
    def step(self, z):
        """
        Single recursive EKF step.
        """
        # 1. Predict
        # Non-linear or linear propagation (using F here as a placeholder for f(x))
        self.x.assign(tf.linalg.matvec(self.m.F, self.x))
        # Add slight jitter to Q for stability
        self.P.assign(self.m.F @ self.P @ tf.transpose(self.m.F) + self.m.Q_filter + tf.eye(self.dim, dtype=DTYPE)*1e-4)

        # 2. Update
        # Evaluate the Jacobian of the observation function specifically at the predicted mean
        H = self.m.jacobian_h(self.x)
        pred_obs = tf.reshape(self.m.h_func(self.x), [-1])
        y = z - pred_obs

        # Calculate Gain using the localized Jacobian H
        S = H @ self.P @ tf.transpose(H) + self.m.R_filter
        K = self.P @ tf.transpose(H) @ safe_inv(S)

        self.x.assign_add(tf.linalg.matvec(K, y))

        # Standard Covariance Update: P = (I - KH)P + KRK^T (Joseph form variant)
        JK = tf.eye(self.dim, dtype=DTYPE) - K @ H
        self.P.assign(JK @ self.P @ tf.transpose(JK) + K @ self.m.R_filter @ tf.transpose(K))
        return self.x

class UKF:
    """
    Unscented Kalman Filter (UKF).
    ------------------------------
    Uses the Unscented Transform to map a set of deterministically chosen
    'Sigma Points' through the fully non-linear functions. This captures
    posterior mean and covariance to the 3rd order of a Taylor series
    expansion, outperforming the EKF without requiring analytical Jacobians.
    """
    def __init__(self, model):
        self.m = model
        self.n = model.state_dim
        self.x = tf.Variable(tf.zeros(self.n, dtype=DTYPE))
        self.P = tf.Variable(tf.eye(self.n, dtype=DTYPE))

        # Unscented Transform scaling parameters:
        # alpha: Determines the spread of sigma points around the mean.
        # beta: Incorporates prior knowledge of the distribution (2.0 is optimal for Gaussian).
        # kappa: Secondary scaling parameter.
        self.alpha = 1e-3; self.beta = 2.0; self.kappa = 0.0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

    def init(self, x, P):
        self.x.assign(x)
        self.P.assign(P)

    @tf.function
    def sigma_points(self, x, P):
        """
        Generates 2n + 1 Sigma Points.
        Uses eigenvalue decomposition (eigh) as a more numerically stable
        alternative to Cholesky for positive semi-definite matrices.
        """
        P_sym = (P + tf.transpose(P))/2
        # Eigendecomposition to find the principal axes of the covariance ellipse
        val_chol, mat_chol = tf.linalg.eigh((self.n + self.lam) * P_sym)
        Sig = mat_chol @ tf.linalg.diag(tf.sqrt(tf.maximum(val_chol, 1e-9)))

        # Point 0 is the mean. The rest are +/- the scaled principal axes.
        Xs = [x]
        for i in range(self.n): Xs.append(x + Sig[:, i])
        for i in range(self.n): Xs.append(x - Sig[:, i])
        return tf.stack(Xs)

    @tf.function
    def step(self, z):
        """
        UKF Prediction and Update cycle.
        """
        # Define the recombination weights for mean (Wm) and covariance (Wc)
        Wm_vec = [self.lam / (self.n + self.lam)] + [1/(2*(self.n + self.lam))] * (2*self.n)
        Wc_vec = [self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)] + [1/(2*(self.n + self.lam))] * (2*self.n)
        Wm = tf.constant(Wm_vec, dtype=DTYPE)
        Wc = tf.constant(Wc_vec, dtype=DTYPE)

        # 1. Prediction Step
        # Propagate sigma points through the transition dynamics
        Xsig = self.sigma_points(self.x, self.P)
        Xsig_pred = tf.transpose(self.m.F @ tf.transpose(Xsig))
        x_pred = tf.linalg.matvec(tf.transpose(Xsig_pred), Wm)

        # Reconstruct the predicted covariance
        P_pred = self.m.Q_filter + tf.eye(self.n, dtype=DTYPE)*1e-4
        diff_x = Xsig_pred - x_pred
        P_pred += tf.transpose(diff_x) @ (diff_x * Wc[:, None])

        # 2. Update Step
        # Generate new sigma points around the *predicted* mean/covariance
        Xsig_pred_new = self.sigma_points(x_pred, P_pred)

        # Propagate through the non-linear observation function
        Ysig_pred = self.m.h_func(Xsig_pred_new)
        y_mean = tf.linalg.matvec(tf.transpose(Ysig_pred), Wm)

        # Calculate Measurement Covariance (Py) and Cross-Covariance (Pxy)
        Py = self.m.R_filter + tf.zeros_like(self.m.R_filter)
        diff_y = Ysig_pred - y_mean
        diff_x_new = Xsig_pred_new - x_pred

        Py += tf.transpose(diff_y) @ (diff_y * Wc[:, None])
        Pxy = tf.transpose(diff_x_new) @ (diff_y * Wc[:, None])

        # Calculate Kalman Gain and update state
        K = Pxy @ safe_inv(Py)
        self.x.assign(x_pred + tf.linalg.matvec(K, z - y_mean))
        self.P.assign(P_pred - K @ Py @ tf.transpose(K))
        return self.x


class ESRF:
    """
    Ensemble Square Root Filter (ESRF).
    -----------------------------------
    A deterministic variant of the Ensemble Kalman Filter (EnKF).
    Unlike standard EnKF which adds random noise to the observations to maintain
    correct posterior variance, the ESRF updates the ensemble mean and the
    ensemble anomalies (deviations from the mean) separately. This strictly
    avoids sampling errors during the update step.
    """
    def __init__(self, model, N=100):
        self.m = model; self.N = N
        self.dim = model.state_dim
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))

    def init(self, x, P):
        """Samples the initial ensemble from the prior distribution."""
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))

    @tf.function
    def step(self, z):
        """
        Executes the deterministic ensemble prediction and update.
        """
        # 1. Prediction
        # Add process noise to the ensemble transition
        noise = tfd.MultivariateNormalTriL(loc=tf.zeros(self.dim, dtype=DTYPE), scale_tril=tf.linalg.cholesky(self.m.Q_filter)).sample(self.N)
        X_pred = tf.transpose(self.m.F @ tf.transpose(self.X)) + noise

        # Decompose the predicted ensemble into Mean (x_mean) and Anomalies (A)
        x_mean = tf.reduce_mean(X_pred, axis=0)
        A = tf.transpose(X_pred - x_mean)

        # Map the ensemble to the observation space
        Y_ens = tf.transpose(self.m.h_func(X_pred))
        y_mean = tf.reduce_mean(Y_ens, axis=1)
        # Y_prime represents the observation anomalies
        Y_prime = Y_ens - y_mean[:, None]

        # 2. Mean Update
        # Calculate innovation covariance S using the ensemble approximation
        S = (Y_prime @ tf.transpose(Y_prime)) / (float(self.N) - 1) + self.m.R_filter

        # Calculate Kalman Gain and update the ensemble mean
        term1 = (A @ tf.transpose(Y_prime)) / (float(self.N) - 1)
        K = term1 @ safe_inv(S)
        x_mean_new = x_mean + tf.linalg.matvec(K, z - y_mean)

        # 3. Anomaly Update (The deterministic "Square Root" step)
        # Instead of updating particles directly with perturbed observations,
        # we calculate a symmetric transform matrix T to apply to the anomalies.
        C = tf.transpose(Y_prime) @ self.m.R_inv_filter @ Y_prime / (float(self.N) - 1)

        # Eigenvalue decomposition to compute the square root of the update matrix
        vals, vecs = tf.linalg.eigh(tf.eye(self.N, dtype=DTYPE) + C)
        vals = tf.maximum(vals, 1e-9)
        T = vecs @ tf.linalg.diag(1.0/tf.sqrt(vals)) @ tf.transpose(vecs)

        # Apply the transform to the prior anomalies
        A_new = A @ T

        # Reconstruct the final updated ensemble: X = Mean + Anomalies
        self.X.assign(tf.transpose(x_mean_new[:, None] + A_new))

        # Return the new ensemble mean as the state estimate
        return tf.reduce_mean(self.X, axis=0)
