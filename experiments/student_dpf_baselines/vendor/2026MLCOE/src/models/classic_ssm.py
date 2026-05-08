"""
Classic SSM Module
==================
Updated to strictly follow the BaseSSM abstract interface.

Author: Joowon Lee
Date: 2026-02-26
"""

import tensorflow as tf
import numpy as np
from .base_ssm import BaseSSM

DTYPE = tf.float64

class LinearGaussianSSM:
    """Standard LGSSM Definition using TensorFlow."""
    def __init__(self, dim_x=5, dim_y=5):
        self.dim_x, self.dim_y = dim_x, dim_y
        self.state_dim = dim_x # Attribute required by modular KF
        self.obs_dim = dim_y

        # Stable transition matrix
        A_np = np.random.randn(dim_x, dim_x)
        max_eig = np.max(np.abs(np.linalg.eigvals(A_np)))
        if max_eig > 1.0: A_np /= (max_eig + 0.1)

        # Matricies with library-consistent names
        self.A = tf.constant(A_np, dtype=DTYPE)
        self.C = tf.random.normal((dim_y, dim_x), dtype=DTYPE)
        self.F, self.H = self.A, self.C # Aliases for KF class
        self.Q = tf.eye(dim_x, dtype=DTYPE) * 0.1
        self.R = tf.eye(dim_y, dtype=DTYPE) * 1.0
        self.x0_mean = tf.zeros(dim_x, dtype=DTYPE)
        self.x0_cov = tf.eye(dim_x, dtype=DTYPE) * 5.0

    @tf.function
    def generate(self, T=100):
        x_ta, y_ta = tf.TensorArray(DTYPE, size=T), tf.TensorArray(DTYPE, size=T)
        L0 = tf.linalg.cholesky(self.x0_cov)
        x_curr = self.x0_mean + tf.linalg.matvec(L0, tf.random.normal([self.dim_x], dtype=DTYPE))
        L_Q, L_R = tf.linalg.cholesky(self.Q), tf.linalg.cholesky(self.R)
        for t in tf.range(T):
            x_curr = tf.linalg.matvec(self.A, x_curr) + tf.linalg.matvec(L_Q, tf.random.normal([self.dim_x], dtype=DTYPE))
            y_curr = tf.linalg.matvec(self.C, x_curr) + tf.linalg.matvec(L_R, tf.random.normal([self.dim_y], dtype=DTYPE))
            x_ta, y_ta = x_ta.write(t, x_curr), y_ta.write(t, y_curr)
        return x_ta.stack(), y_ta.stack()


class StochasticVolatilityModel:
    """
    Stochastic Volatility Model.
    """
    def __init__(self, alpha=0.91, sigma_v=1.0, beta=0.5):
        self.state_dim = 1
        self.alpha = alpha
        self.sigma_v = sigma_v
        self.beta = beta

        # Noise Params
        self.Q = np.array([[sigma_v**2]])
        self.R_filter_np = np.array([[np.pi**2 / 2.0]])

        # --- FILTERS.PY COMPATIBILITY ---
        # 1. Transition Matrix F (1x1 for this linear process)
        self.F = tf.constant([[self.alpha]], dtype=DTYPE)

        # 2. Tensor Covariances
        self.Q_filter = tf.constant(self.Q, dtype=DTYPE)
        self.R_filter = tf.constant(self.R_filter_np, dtype=DTYPE)
        self.R_inv_filter = tf.linalg.inv(self.R_filter)

    def generate(self, T=200):
        x = np.zeros((T, 1))
        y = np.zeros((T, 1))
        x[0] = np.random.normal(0, self.sigma_v / np.sqrt(1 - self.alpha**2))
        for t in range(T):
            if t > 0:
                x[t] = self.alpha * x[t-1] + np.random.normal(0, self.sigma_v)
            vol = self.beta * np.exp(x[t] / 2.0)
            y[t] = vol * np.random.normal(0, 1.0)
        return tf.constant(x, dtype=DTYPE), tf.constant(y, dtype=DTYPE)

    def f(self, x):
        return self.alpha * x

    def f_fn(self, x):
        return self.f(x)

    def propagate(self, particles):
        noise = tf.random.normal(tf.shape(particles), stddev=self.sigma_v, dtype=DTYPE)
        return self.alpha * particles + noise

    def log_likelihood(self, y, particles):
        y = tf.reshape(y, [-1])
        x = particles
        var = (self.beta**2) * tf.exp(x)
        const_term = -0.5 * np.log(2 * np.pi)
        log_std_term = -0.5 * tf.math.log(var + 1e-8)
        data_term = -0.5 * tf.square(y) / (var + 1e-8)
        log_prob = const_term + log_std_term + data_term
        return tf.reshape(log_prob, [-1])

    def transform_obs(self, y):
        # Helper to transform y into the linear-Gaussian space expected by h(x)
        return tf.math.log(tf.square(y) + 1e-8)

    def h(self, x):
        # Linearized measurement function (requires transformed y)
        beta_sq = tf.constant(self.beta**2, dtype=DTYPE)
        offset = tf.constant(1.27036, dtype=DTYPE) # E[log(chi^2)]
        return tf.math.log(beta_sq) + x - offset

    def h_func(self, x):
        return self.h(x)

    def h_fn(self, x):
        return self.h(x)

    def f_jacobian(self, x):
        return tf.constant([[self.alpha]], dtype=DTYPE)

    def h_jacobian(self, x):
        return tf.constant([[1.0]], dtype=DTYPE)

    @tf.function
    def jacobian_h(self, x):
        # Alias for filters.py
        return self.h_jacobian(x)


class RangeBearingModel:
    def __init__(self, dt=0.1, sigma_q=0.5, sigma_r=0.5, sigma_theta=0.1, omega=-0.05):
        self.dt = dt
        self.state_dim = 4
        self.omega = omega

        # --- FILTERS.PY COMPATIBILITY ---
        # 1. Covariances (Q_filter, R_filter, R_inv_filter)
        self.Q = tf.eye(4, dtype=DTYPE) * (sigma_q**2)
        self.Q_filter = self.Q # Alias

        self.R_filter = tf.linalg.diag(tf.cast([sigma_r**2, sigma_theta**2], DTYPE))
        self.inv_R = tf.linalg.inv(self.R_filter)
        self.R_inv_filter = self.inv_R # Alias

        # 2. Transition Matrix F
        if abs(omega) > 1e-8:
            sin_w = np.sin(omega * dt)
            cos_w = np.cos(omega * dt)
            F_np = np.array([
                [1, 0, sin_w/omega, -(1-cos_w)/omega],
                [0, 1, (1-cos_w)/omega, sin_w/omega],
                [0, 0, cos_w, -sin_w],
                [0, 0, sin_w, cos_w]
            ])
        else:
            F_np = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])

        self.F_matrix = tf.constant(F_np, dtype=DTYPE)
        self.F = self.F_matrix # Alias for KF/EKF

    def generate(self, T=200):
        x_list = []
        y_list = []
        curr_x = tf.constant([[0.0], [0.0], [4.0], [2.0]], dtype=DTYPE)

        for t in range(T):
            noise_proc = tf.random.normal((4, 1), dtype=DTYPE)
            noise_proc = tf.matmul(tf.linalg.cholesky(self.Q), noise_proc)
            curr_x = tf.matmul(self.F_matrix, curr_x) + noise_proc
            y_clean = self._measurement_fn(curr_x)
            noise_meas = tf.random.normal((2, 1), dtype=DTYPE)
            noise_meas = tf.matmul(tf.linalg.cholesky(self.R_filter), noise_meas)
            curr_y = y_clean + noise_meas
            x_list.append(tf.squeeze(curr_x))
            y_list.append(tf.squeeze(curr_y))

        return tf.stack(x_list), tf.stack(y_list)

    def generate_data(self, steps=200):
        return self.generate(T=steps)

    def _measurement_fn(self, x):
        squeeze_output = False
        # Handle various input shapes (4,1) or (N,4)
        if len(x.shape) == 2 and x.shape[1] == 1:
            x_in = tf.transpose(x)
            squeeze_output = True
        elif len(x.shape) == 1:
             x_in = tf.expand_dims(x, 0)
             squeeze_output = True
        else:
            x_in = x

        px = x_in[:, 0]
        py = x_in[:, 1]
        r = tf.sqrt(px**2 + py**2)
        theta = tf.math.atan2(py, px)
        y = tf.stack([r, theta], axis=1)

        if squeeze_output:
            return tf.transpose(y)
        return y

    def f(self, x):
        if len(x.shape) == 2 and x.shape[1] == 1:
             return tf.matmul(self.F_matrix, x)
        return tf.matmul(x, self.F_matrix, transpose_b=True)

    def f_fn(self, x):
        return self.f(x)

    def h(self, x):
        return self._measurement_fn(x)

    def h_func(self, x):
        return self.h(x)

    def h_fn(self, x):
        return self.h(x)

    @tf.function
    def jacobian_h(self, x_bar):
        """
        Analytic Jacobian H for linearized filters (EDH/LEDH/EKF).
        Expects x_bar shape: (4,)
        """
        # Ensure 1D input
        if len(x_bar.shape) > 1:
            x_bar = tf.squeeze(x_bar)

        px, py = x_bar[0], x_bar[1]
        r2 = px**2 + py**2
        r = tf.sqrt(r2)

        # Stability check
        safe_r = tf.maximum(r, 1e-4)
        safe_r2 = tf.maximum(r2, 1e-8)

        # H = [ [px/r, py/r, 0, 0], [-py/r^2, px/r^2, 0, 0] ]
        row1 = tf.stack([px/safe_r, py/safe_r, 0.0, 0.0])
        row2 = tf.stack([-py/safe_r2, px/safe_r2, 0.0, 0.0])

        return tf.stack([row1, row2])

    def propagate(self, particles):
        x_pred = tf.matmul(particles, self.F_matrix, transpose_b=True)
        noise = tf.random.normal(tf.shape(particles), dtype=DTYPE)
        scale = tf.sqrt(tf.linalg.diag_part(self.Q))
        return x_pred + (noise * scale)

    def log_likelihood(self, y, particles):
        y_flat = tf.reshape(y, [-1])
        preds = self._measurement_fn(particles)
        diff_r = y_flat[0] - preds[:, 0]
        diff_th = y_flat[1] - preds[:, 1]
        diff_th = tf.math.floormod(diff_th + np.pi, 2 * np.pi) - np.pi
        residuals = tf.stack([diff_r, diff_th], axis=1)
        R_diag = tf.linalg.diag_part(self.R_filter)
        weighted_sq = tf.square(residuals) / R_diag
        return -0.5 * tf.reduce_sum(weighted_sq, axis=1)


def tf_cdist(a, b):
    """TensorFlow equivalent of scipy.spatial.distance.cdist"""
    a_sq = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
    b_sq = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(a_sq - 2 * tf.matmul(a, b, transpose_b=True) + tf.transpose(b_sq), 1e-12))


class AcousticTrackingSSM:
    def __init__(self):
        self.n_targets = 4
        self.state_dim = 16
        sensor_locs_np = np.array([[x, y] for x in range(0, 50, 10) for y in range(0, 50, 10)])
        self.sensor_locs = tf.constant(sensor_locs_np, dtype=DTYPE)
        self.obs_dim = 25
        self.Psi = tf.constant(10.0, dtype=DTYPE)
        self.d0 = tf.constant(0.1, dtype=DTYPE)

        self.sigma_w2_true = 0.01
        self.R_true = tf.eye(self.obs_dim, dtype=DTYPE) * self.sigma_w2_true

        q_block_true = np.array([[1/3, 0, 0.5, 0], [0, 1/3, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]]) * 0.1
        self.Q_true = tf.constant(np.kron(np.eye(4), q_block_true), dtype=DTYPE)

        self.sigma_w2_filter = 0.01
        self.R_filter = tf.eye(self.obs_dim, dtype=DTYPE) * self.sigma_w2_filter
        self.R_inv_filter = tf.eye(self.obs_dim, dtype=DTYPE) * (1.0/self.sigma_w2_filter)

        q_block_filter = np.array([[3.0, 0, 0.1, 0], [0, 3.0, 0, 0.1], [0.1, 0, 0.03, 0], [0, 0.1, 0, 0.03]])
        self.Q_filter = tf.constant(np.kron(np.eye(4), q_block_filter), dtype=DTYPE)

        f_block = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.F = tf.constant(np.kron(np.eye(4), f_block), dtype=DTYPE)

    @tf.function
    def h_func(self, x):
        x_in = x
        if len(x.shape) == 1:
            x_in = x[None, :]
        N = tf.shape(x_in)[0]
        y = tf.zeros((N, self.obs_dim), dtype=DTYPE)
        for c in range(self.n_targets):
            pos = x_in[:, c*4 : c*4+2]
            dists = tf_cdist(pos, self.sensor_locs)
            y += self.Psi / (dists + self.d0)
        return y

    @tf.function
    def jacobian_h(self, x):
        sx = self.sensor_locs[:, 0]
        sy = self.sensor_locs[:, 1]

        # Ensure x is strictly 1D (vector of 16) to prevent batch dim leaking
        x = tf.reshape(x, [-1])

        zero = tf.constant(0.0, dtype=DTYPE)

        H_rows = []
        for s in range(self.obs_dim):
            grads = []
            for c in range(self.n_targets):
                ix, iy = c*4, c*4+1
                tx, ty = x[ix], x[iy]

                dist_sq = tf.square(tx - sx[s]) + tf.square(ty - sy[s])
                dist = tf.sqrt(dist_sq)

                # Robust math for the gradient factor
                safe_dist = tf.maximum(dist, 1e-12)
                denom = tf.square(safe_dist + self.d0)
                factor = -self.Psi / (denom * safe_dist)

                # Use tf.cond strictly.
                is_far = dist > 1e-6

                def calc_grad_x(): return factor * (tx - sx[s])
                def calc_grad_y(): return factor * (ty - sy[s])

                val_x = tf.cond(is_far, calc_grad_x, lambda: zero)
                val_y = tf.cond(is_far, calc_grad_y, lambda: zero)

                # Hard assertion that these are scalars
                val_x = tf.ensure_shape(val_x, [])
                val_y = tf.ensure_shape(val_y, [])

                grads.extend([val_x, val_y, zero, zero])

            H_rows.append(tf.stack(grads))

        return tf.stack(H_rows)

class Dai22BearingOnlySSM:
    """Exact Numerical Example from Section 4 of Dai & Daum (2022)."""
    def __init__(self):
        self.state_dim = 2
        # Sensor locations: (3.5, 0) and (-3.5, 0)
        self.sensors = tf.constant([[3.5, 0.0], [-3.5, 0.0]], dtype=DTYPE)
        self.x_truth = tf.constant([4.0, 4.0], dtype=DTYPE)
        self.x_prior = tf.constant([3.0, 5.0], dtype=DTYPE)
        # Prior Covariance: diag(1000.0, 2.0) creates 500:1 stiffness ratio
        self.P_prior = tf.constant([[1000.0, 0.0], [0.0, 2.0]], dtype=DTYPE)

        # In ssm_models.py -> Dai22BearingOnlySSM.__init__
        self.sigma_v2 = 0.04  # Corrected: R = diag(0.04, 0.04)
        self.R_filter = tf.eye(2, dtype=DTYPE) * self.sigma_v2
        self.R_inv_filter = tf.linalg.inv(self.R_filter)
        self.F = tf.eye(2, dtype=DTYPE)
        self.Q_filter = tf.eye(2, dtype=DTYPE) * 1e-9

    @tf.function
    def h_func(self, x):
        xt, yt = x[..., 0:1], x[..., 1:2]
        xi, yi = self.sensors[:, 0], self.sensors[:, 1]
        return tf.atan2(yt - yi, xt - xi) # h_i = arctan((yt-yi)/(xt-xi))

    @tf.function
    def jacobian_h(self, x):
        x = tf.reshape(x, [-1])
        xt, yt = x[0], x[1]
        H_rows = []
        for i in range(2):
            xi, yi = self.sensors[i, 0], self.sensors[i, 1]
            dx, dy = xt - xi, yt - yi
            r_sq = tf.maximum(tf.square(dx) + tf.square(dy), 1e-12)
            H_rows.append(tf.stack([-dy/r_sq, dx/r_sq])) # grad arctan
        return tf.stack(H_rows)


class AndrieuModel:
    """Non-linear SSM from Andrieu et al. (2010)."""
    def __init__(self, sigma_v=np.sqrt(10), sigma_w=1.0):
        self.state_dim = 1
        self.obs_dim = 1
        self.Q_filter = tf.cast([[sigma_v**2]], DTYPE)
        self.R_filter = tf.cast([[sigma_w**2]], DTYPE)
        self.R_inv_filter = tf.linalg.inv(self.R_filter)
        self.F = tf.cast([[0.5]], DTYPE) # Fallback for initialization

    def transition(self, x, n):
        """Eq (14): Non-linear transition with cosine drive."""
        term1 = x / 2.0
        term2 = 25.0 * x / (1.0 + tf.square(x))
        term3 = 8.0 * tf.cos(1.2 * tf.cast(n, DTYPE))
        return term1 + term2 + term3

    def h_func(self, x):
        """Eq (15): Quadratic observation."""
        return tf.reshape(tf.square(x) / 20.0, [-1, 1])

    def jacobian_h(self, x):
        """dh/dx = x / 10 for local flow computation."""
        return tf.reshape(x / 10.0, [1, 1])


class NonlinearBenchmarkSSM:
    def __init__(self, sigma_v_sq=10.0, sigma_w_sq=1.0):
        self.state_dim = 1
        self.sigma_v_sq = tf.cast(sigma_v_sq, DTYPE)
        self.sigma_w_sq = tf.cast(sigma_w_sq, DTYPE)
        self.x0_mean = tf.zeros((1,), dtype=DTYPE)
        self.x0_cov = tf.eye(1, dtype=DTYPE) * 5.0
        self.R_filter = tf.eye(1, dtype=DTYPE) * self.sigma_w_sq
        self.R_inv_filter = tf.eye(1, dtype=DTYPE) / self.sigma_w_sq
        self.current_n = tf.Variable(1.0, dtype=DTYPE)

    def update_parameters(self, theta):
        # Using softplus or abs to ensure positivity during optimization/sampling
        self.sigma_v_sq = tf.abs(tf.cast(theta[0], DTYPE))
        self.Q_filter = tf.eye(1, dtype=DTYPE) * self.sigma_v_sq

    @tf.function
    def propagate(self, X):
        return X/2.0 + 25.0*X/(1.0 + X**2) + 8.0*tf.cos(1.2 * self.current_n)

    @tf.function
    def h_func(self, x): return (x**2) / 20.0

    @tf.function
    def jacobian_h(self, x): return tf.reshape(x / 10.0, (1, 1))
