"""
filters.py
==========
Complete TensorFlow implementation of Bayesian Filters.

Methods Implemented:
1.  Kalman Filter (KF)
2.  Extended Kalman Filter (EKF)
3.  Unscented Kalman Filter (UKF)
4.  Bootstrap Particle Filter (BPF)
5.  Particle Flow Particle Filter (PFPF) - (EDH & LEDH variants)
6.  Exact Daum-Huang Filter (EDH)
7.  Local EDH Filter (LEDH)
8.  PFPF-EDH
9.  PFPF-LEDH
10. Kernel Particle Flow Filter (KPFF)
11.  Ensemble Square Root Filter (ESRF)
12.  Gaussian Sum Monte Carlo (GSMC)
13.  Unscented Particle Filter (UPF)

"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Precision settings
DTYPE = tf.float64
JITTER = 1e-6
tfd = tfp.distributions

def safe_inv(matrix):
    """Numerically stable inversion using Cholesky decomposition."""
    try:
        L = tf.linalg.cholesky(matrix)
        return tf.linalg.cholesky_solve(L, tf.eye(tf.shape(matrix)[0], dtype=DTYPE))
    except:
        return tf.linalg.inv(matrix + tf.eye(tf.shape(matrix)[0], dtype=DTYPE) * 1e-4)

class KF:
    """
    Kalman Filter (KF).
    Uses tf.matmul. Expects state x to be Rank 2 (dim, 1).
    """
    def __init__(self, F, H, Q, R, P0, x0):
        self.F = tf.cast(F, DTYPE)
        self.H = tf.cast(H, DTYPE)
        self.Q = tf.cast(Q, DTYPE)
        self.R = tf.cast(R, DTYPE)
        self.dim = self.F.shape[0]
        self.I = tf.eye(self.dim, dtype=DTYPE)

        if isinstance(x0, tf.Variable): self.x = x0
        else: self.x = tf.Variable(tf.cast(x0, DTYPE), dtype=DTYPE)
        if isinstance(P0, tf.Variable): self.P = P0
        else: self.P = tf.Variable(tf.cast(P0, DTYPE), dtype=DTYPE)

    @tf.function
    def predict(self):
        x_curr = self.x.read_value()
        P_curr = self.P.read_value()

        # Use matmul for Rank 2 (N, 1) vectors
        x_pred = tf.matmul(self.F, x_curr)
        P_pred = tf.matmul(tf.matmul(self.F, P_curr), self.F, transpose_b=True) + self.Q

        self.x.assign(x_pred)
        self.P.assign(P_pred)
        return x_pred, P_pred

    @tf.function
    def update(self, z):
        z = tf.cast(z, DTYPE)
        x_curr = self.x.read_value()
        P_curr = self.P.read_value()

        S = tf.matmul(tf.matmul(self.H, P_curr), self.H, transpose_b=True) + self.R
        K = tf.matmul(tf.matmul(P_curr, self.H, transpose_b=True), safe_inv(S))

        y_res = z - tf.matmul(self.H, x_curr)
        x_new = x_curr + tf.matmul(K, y_res)

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
        # Predict
        self.x.assign(tf.linalg.matvec(self.m.F, self.x))
        self.P.assign(self.m.F @ self.P @ tf.transpose(self.m.F) + self.m.Q_filter + tf.eye(self.dim, dtype=DTYPE)*1e-4)

        # Update
        H = self.m.jacobian_h(self.x)
        pred_obs = tf.reshape(self.m.h_func(self.x), [-1])
        y = z - pred_obs

        S = H @ self.P @ tf.transpose(H) + self.m.R_filter
        K = self.P @ tf.transpose(H) @ safe_inv(S)

        self.x.assign_add(tf.linalg.matvec(K, y))

        JK = tf.eye(self.dim, dtype=DTYPE) - K @ H
        self.P.assign(JK @ self.P @ tf.transpose(JK) + K @ self.m.R_filter @ tf.transpose(K))
        return self.x

class UKF:
    """
    Unscented Kalman Filter (UKF).
    """
    def __init__(self, model):
        self.m = model
        self.n = model.state_dim
        self.x = tf.Variable(tf.zeros(self.n, dtype=DTYPE))
        self.P = tf.Variable(tf.eye(self.n, dtype=DTYPE))
        self.alpha = 1e-3; self.beta = 2.0; self.kappa = 0.0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

    def init(self, x, P):
        self.x.assign(x)
        self.P.assign(P)

    @tf.function
    def sigma_points(self, x, P):
        P_sym = (P + tf.transpose(P))/2
        val_chol, mat_chol = tf.linalg.eigh((self.n + self.lam) * P_sym)
        Sig = mat_chol @ tf.linalg.diag(tf.sqrt(tf.maximum(val_chol, 1e-9)))
        Xs = [x]
        for i in range(self.n): Xs.append(x + Sig[:, i])
        for i in range(self.n): Xs.append(x - Sig[:, i])
        return tf.stack(Xs)

    @tf.function
    def step(self, z):
        Wm_vec = [self.lam / (self.n + self.lam)] + [1/(2*(self.n + self.lam))] * (2*self.n)
        Wc_vec = [self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)] + [1/(2*(self.n + self.lam))] * (2*self.n)
        Wm = tf.constant(Wm_vec, dtype=DTYPE)
        Wc = tf.constant(Wc_vec, dtype=DTYPE)

        Xsig = self.sigma_points(self.x, self.P)
        Xsig_pred = tf.transpose(self.m.F @ tf.transpose(Xsig))
        x_pred = tf.linalg.matvec(tf.transpose(Xsig_pred), Wm)

        P_pred = self.m.Q_filter + tf.eye(self.n, dtype=DTYPE)*1e-4
        diff_x = Xsig_pred - x_pred
        P_pred += tf.transpose(diff_x) @ (diff_x * Wc[:, None])

        Xsig_pred_new = self.sigma_points(x_pred, P_pred)
        Ysig_pred = self.m.h_func(Xsig_pred_new)
        y_mean = tf.linalg.matvec(tf.transpose(Ysig_pred), Wm)

        Py = self.m.R_filter + tf.zeros_like(self.m.R_filter)
        diff_y = Ysig_pred - y_mean
        diff_x_new = Xsig_pred_new - x_pred

        Py += tf.transpose(diff_y) @ (diff_y * Wc[:, None])
        Pxy = tf.transpose(diff_x_new) @ (diff_y * Wc[:, None])

        K = Pxy @ safe_inv(Py)
        self.x.assign(x_pred + tf.linalg.matvec(K, z - y_mean))
        self.P.assign(P_pred - K @ Py @ tf.transpose(K))
        return self.x

class ESRF:
    """
    Ensemble Square Root Filter (ESRF).
    """
    def __init__(self, model, N=100):
        self.m = model; self.N = N
        self.dim = model.state_dim
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))

    def init(self, x, P):
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))

    @tf.function
    def step(self, z):
        noise = tfd.MultivariateNormalTriL(loc=tf.zeros(self.dim, dtype=DTYPE), scale_tril=tf.linalg.cholesky(self.m.Q_filter)).sample(self.N)
        X_pred = tf.transpose(self.m.F @ tf.transpose(self.X)) + noise
        x_mean = tf.reduce_mean(X_pred, axis=0)
        A = tf.transpose(X_pred - x_mean)
        Y_ens = tf.transpose(self.m.h_func(X_pred))
        y_mean = tf.reduce_mean(Y_ens, axis=1)
        Y_prime = Y_ens - y_mean[:, None]

        S = (Y_prime @ tf.transpose(Y_prime)) / (float(self.N) - 1) + self.m.R_filter

        term1 = (A @ tf.transpose(Y_prime)) / (float(self.N) - 1)
        K = term1 @ safe_inv(S)

        x_mean_new = x_mean + tf.linalg.matvec(K, z - y_mean)

        C = tf.transpose(Y_prime) @ self.m.R_inv_filter @ Y_prime / (float(self.N) - 1)
        vals, vecs = tf.linalg.eigh(tf.eye(self.N, dtype=DTYPE) + C)
        vals = tf.maximum(vals, 1e-9)
        T = vecs @ tf.linalg.diag(1.0/tf.sqrt(vals)) @ tf.transpose(vecs)

        A_new = A @ T
        self.X.assign(tf.transpose(x_mean_new[:, None] + A_new))
        return tf.reduce_mean(self.X, axis=0)

class GSMC:
    """
    Gaussian Sum Monte Carlo (GSMC).
    """
    def __init__(self, model, N=100):
        self.m = model; self.N = N; self.esrf = ESRF(model, N)
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N); self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        self.esrf.init(x, P); self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

    @tf.function
    def step(self, z):
        self.esrf.step(z)
        X_prop = self.esrf.X
        preds = self.m.h_func(X_prop)
        res = z - preds
        quad = tf.reduce_sum(tf.matmul(res, self.m.R_inv_filter) * res, axis=1)
        log_lik = -0.5 * quad
        w = tf.exp(log_lik - tf.reduce_max(log_lik)); w = w / tf.reduce_sum(w)

        self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))
        if self.ess < self.N/2:
            idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
            self.esrf.X.assign(tf.gather(self.esrf.X, idx))
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        else: self.W.assign(w)
        return tf.reduce_sum(self.esrf.X * self.W[:, None], axis=0)

class UPF:
    """
    Unscented Particle Filter (UPF).
    """
    def __init__(self, model, N=50):
        self.m = model; self.N = N; self.ukf = UKF(model)
        self.dim = model.state_dim
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))
        self.Ps = tf.Variable(tf.zeros((N, self.dim, self.dim), dtype=DTYPE))
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N); self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))
        self.Ps.assign(tf.tile(P[None, :, :], [self.N, 1, 1]))
        self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

    @tf.function
    def step(self, z):
        def single_ukf_step(inp):
            x_i, P_i = inp; self.ukf.init(x_i, P_i); self.ukf.step(z)
            try: L = tf.linalg.cholesky(self.ukf.P)
            except: L = tf.linalg.cholesky(self.ukf.P + tf.eye(self.dim, dtype=DTYPE)*1e-3)
            samp = self.ukf.x + tf.linalg.matvec(L, tf.random.normal((self.dim,), dtype=DTYPE))
            return samp, self.ukf.P
        X_new, P_new = tf.map_fn(single_ukf_step, (self.X, self.Ps), dtype=(DTYPE, DTYPE))
        self.X.assign(X_new); self.Ps.assign(P_new)

        preds = self.m.h_func(self.X); res = z - preds
        quad = tf.reduce_sum(tf.matmul(res, self.m.R_inv_filter) * res, axis=1)
        log_lik = -0.5 * quad
        w = tf.exp(log_lik - tf.reduce_max(log_lik)); w = w / tf.reduce_sum(w)
        self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))
        if self.ess < self.N/2:
            idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
            self.X.assign(tf.gather(self.X, idx))
            self.Ps.assign(tf.gather(self.Ps, idx))
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        else: self.W.assign(w)
        return tf.reduce_sum(self.X * self.W[:, None], axis=0)

class BPF:
    """
    Bootstrap Particle Filter (BPF).
    """
    def __init__(self, model, N=1000):
        self.m = model; self.N = N
        self.dim = model.state_dim
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N); self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))
        self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

    @tf.function
    def step(self, z):
        noise = tfd.MultivariateNormalTriL(loc=tf.zeros(self.dim, dtype=DTYPE), scale_tril=tf.linalg.cholesky(self.m.Q_filter)).sample(self.N)
        self.X.assign(tf.transpose(self.m.F @ tf.transpose(self.X)) + noise)
        preds = self.m.h_func(self.X); res = z - preds
        quad = tf.reduce_sum(tf.matmul(res, self.m.R_inv_filter) * res, axis=1)
        log_lik = -0.5 * quad
        log_w_prev = tf.math.log(self.W + 1e-30); log_w_unnorm = log_lik + log_w_prev
        w = tf.exp(log_w_unnorm - tf.reduce_max(log_w_unnorm)); w = w / tf.reduce_sum(w)
        self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))
        if self.ess < self.N * 0.1:
            idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
            self.X.assign(tf.gather(self.X, idx))
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        else: self.W.assign(w)
        return tf.reduce_sum(self.X * self.W[:, None], axis=0)

class ParticleFlowFilter:
    """
    Unified Particle Flow Filter (Vectorized TensorFlow).
    Implements EDH (Exact Daum-Huang), LEDH (Local Exact Daum-Huang) and their PFPF variants.
    """
    def __init__(self, model, N=100, mode='ledh', is_pfpf=True):
        self.m = model; self.N = N; self.mode = mode; self.is_pfpf = is_pfpf
        self.dim = model.state_dim
        self.n_steps = 30
        eps_np = np.logspace(-2, 0, self.n_steps); eps_np /= eps_np.sum()
        self.eps = tf.constant(eps_np, dtype=DTYPE)
        self.ekf = EKF(model)
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N); self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))
        self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        self.ekf.init(x, P)

    @tf.function
    def step(self, z):
        if not self.is_pfpf:
            mean = tf.reduce_mean(self.X, axis=0)
            cov = tfp.stats.covariance(self.X) + tf.eye(self.dim, dtype=DTYPE)*1e-4
            dist = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(cov))
            self.X.assign(dist.sample(self.N))
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

        noise = tfd.MultivariateNormalTriL(loc=tf.zeros(self.dim, dtype=DTYPE), scale_tril=tf.linalg.cholesky(self.m.Q_filter)).sample(self.N)
        self.X.assign(tf.transpose(self.m.F @ tf.transpose(self.X)) + noise)

        x_bar_prop = tf.reduce_mean(self.X, axis=0)
        P_prop = tfp.stats.covariance(self.X) + tf.eye(self.dim, dtype=DTYPE)*1e-4
        self.ekf.x.assign(x_bar_prop); self.ekf.P.assign(P_prop)

        # Convert to Tensor
        eta = tf.identity(self.X)

        curr_lam = 0.0; log_det = tf.zeros(self.N, dtype=DTYPE)
        if self.mode == 'edh':
            x_bar = tf.reduce_mean(eta, axis=0)
            H_g = self.m.jacobian_h(x_bar)
            e_g = tf.reshape(self.m.h_func(x_bar), [-1]) - tf.linalg.matvec(H_g, x_bar)

        for k in range(self.n_steps):
            dl = self.eps[k]; curr_lam += dl
            if self.mode == 'edh':
                H, e = H_g, e_g
                S = curr_lam * H @ self.ekf.P @ tf.transpose(H) + self.m.R_filter
                S_inv = safe_inv(S)
                A = -0.5 * self.ekf.P @ tf.transpose(H) @ S_inv @ H
                K_gain = self.ekf.P @ tf.transpose(H) @ self.m.R_inv_filter

                term1 = tf.eye(self.dim, dtype=DTYPE) + 2*curr_lam*A
                term2 = tf.eye(self.dim, dtype=DTYPE) + curr_lam*A

                kze = tf.linalg.matvec(K_gain, z - e)
                ax = tf.linalg.matvec(A, self.ekf.x)

                inner = tf.linalg.matvec(term2, kze) + ax
                b = tf.linalg.matvec(term1, inner)

                flow = tf.transpose(A @ tf.transpose(eta)) + b
                eta += dl * flow
            else: # LEDH
                def ledh_body(p_i):
                    H_i = self.m.jacobian_h(p_i)
                    e_i = tf.reshape(self.m.h_func(p_i), [-1]) - tf.linalg.matvec(H_i, p_i)
                    S_i = curr_lam * H_i @ self.ekf.P @ tf.transpose(H_i) + self.m.R_filter
                    S_inv_i = safe_inv(S_i)
                    A_i = -0.5 * self.ekf.P @ tf.transpose(H_i) @ S_inv_i @ H_i
                    K_gain_i = self.ekf.P @ tf.transpose(H_i) @ self.m.R_inv_filter

                    term1 = tf.eye(self.dim, dtype=DTYPE) + 2*curr_lam*A_i
                    term2 = tf.eye(self.dim, dtype=DTYPE) + curr_lam*A_i

                    kze = tf.linalg.matvec(K_gain_i, z - e_i)
                    ap = tf.linalg.matvec(A_i, p_i)

                    inner = tf.linalg.matvec(term2, kze) + ap
                    b_i = tf.linalg.matvec(term1, inner)

                    update = dl * (tf.linalg.matvec(A_i, p_i) + b_i)
                    ld_i = 0.0
                    if self.is_pfpf:
                         sign, logdet = tf.linalg.slogdet(tf.eye(self.dim, dtype=DTYPE) + dl * A_i)
                         ld_i = logdet
                    return p_i + update, ld_i

                eta, ld_batch = tf.map_fn(ledh_body, eta, dtype=(DTYPE, DTYPE))
                if self.is_pfpf: log_det += ld_batch

        if self.is_pfpf:
             innov = z - self.m.h_func(eta)
             quad = tf.reduce_sum(tf.matmul(innov, self.m.R_inv_filter) * innov, axis=1)
             log_lik = -0.5 * quad
             log_w = tf.math.log(self.W + 1e-30) + log_lik + log_det
             w = tf.exp(log_w - tf.reduce_max(log_w)); w = w / tf.reduce_sum(w)
             self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))
             self.X.assign(eta)
             if self.ess < self.N/2:
                 idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
                 self.X.assign(tf.gather(self.X, idx))
                 self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
             else: self.W.assign(w)
             est = tf.reduce_sum(self.X * self.W[:, None], axis=0)

             diff = self.X - est
             cov_post = tf.matmul(tf.transpose(diff * self.W[:, None]), diff)
             cov_post += tf.eye(self.dim, dtype=DTYPE)*1e-4

             self.ekf.init(est, cov_post)
             return est
        else:
             self.X.assign(eta); self.ess.assign(float(self.N))
             est = tf.reduce_mean(self.X, axis=0)
             cov_post = tfp.stats.covariance(self.X) + tf.eye(self.dim, dtype=DTYPE)*1e-4
             self.ekf.init(est, cov_post)
             return est

class EDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='edh', is_pfpf=False)
        self.n_steps = steps

class LEDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='ledh', is_pfpf=False)
        self.n_steps = steps

class PFPF_EDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='edh', is_pfpf=True)
        self.n_steps = steps

class PFPF_LEDH(ParticleFlowFilter):
    def __init__(self, model, N=100, steps=30):
        super().__init__(model=model, N=N, mode='ledh', is_pfpf=True)
        self.n_steps = steps

class KPFF:
    """
    Kernel Particle Flow Filter (KPFF) using RKHS embedding.
    Fully TensorFlow implementation.
    """
    def __init__(self, ensemble, y_obs, obs_idx, R_var, kernel_type='matrix'):
        # Convert ensemble to Tensor (not Variable unless persistence is needed in place)
        # Assuming ensemble is updated via update() return or explicit assign in caller
        self.X = tf.Variable(tf.convert_to_tensor(ensemble, dtype=DTYPE), dtype=DTYPE)

        self.Np = tf.shape(self.X)[0]
        self.Nx = tf.shape(self.X)[1]

        self.y = tf.convert_to_tensor(y_obs, dtype=DTYPE)
        self.obs_idx = tf.convert_to_tensor(obs_idx, dtype=tf.int32)

        # Cast scalar R_var
        self.R_inv = tf.cast(1.0 / R_var, DTYPE)
        self.kernel_type = kernel_type

        self.alpha = tf.cast(1.0 / tf.cast(self.Np, DTYPE), DTYPE)

        # Prior stats (Static for the flow duration)
        self.prior_mean = tf.reduce_mean(self.X, axis=0)
        self.prior_var = tf.math.reduce_variance(self.X, axis=0) + 1e-4
        self.D = self.prior_var

    @tf.function
    def get_grad_log_posterior(self, x):
        # x: (Nx,)
        # Likelihood grad
        # y - x[obs_idx]
        x_obs = tf.gather(x, self.obs_idx)
        innov = self.y - x_obs
        grad_lik_sub = innov * self.R_inv # Scalar R

        # Scatter back to full dimension
        indices = tf.expand_dims(self.obs_idx, 1)
        grad_lik = tf.scatter_nd(indices, grad_lik_sub, [self.Nx])

        # Prior grad
        grad_prior = -(x - self.prior_mean) / self.prior_var

        return grad_lik + grad_prior

    @tf.function
    def compute_flow(self, x_eval):
        # Compute gradients for all particles
        grad_log_p_list = tf.map_fn(self.get_grad_log_posterior, self.X)

        # Current particle i is x_eval
        diffs = self.X - x_eval # (Np, Nx)

        if self.kernel_type == 'scalar':
            dist_sq = tf.reduce_sum(tf.square(diffs) / (self.alpha * self.prior_var), axis=1)
            k_val = tf.exp(-0.5 * dist_sq)

            # grad_k: -K * diff / (alpha * var)
            # reshape k_val to (Np, 1)
            grad_k = -tf.expand_dims(k_val, 1) * (diffs / (self.alpha * self.prior_var))

            term = tf.expand_dims(k_val, 1) * grad_log_p_list + grad_k

        elif self.kernel_type == 'matrix':
            dist_sq_vec = tf.square(diffs) / (self.alpha * self.prior_var)
            k_vec = tf.exp(-0.5 * dist_sq_vec)

            grad_k = -k_vec * (diffs / (self.alpha * self.prior_var))

            term = k_vec * grad_log_p_list + grad_k

        flow_sum = tf.reduce_sum(term, axis=0)
        return (self.D * flow_sum) / tf.cast(self.Np, DTYPE)

    @tf.function
    def update(self, n_steps=50, dt=0.01):
        dt_tf = tf.cast(dt, DTYPE)

        def body(i, current_X):
            # Since flow depends on the current ensemble configuration,
            # strict PFF evolves them together.
            # We map compute_flow over current_X.

            # Note: My compute_flow uses self.X.
            # Ideally, self.X should track current_X.
            # BUT, changing self.X inside map_fn or loop might be tricky with AutoGraph.
            # Let's perform the all-to-all calc here directly for robustness.

            # 1. Gradients
            grad_log_p = tf.map_fn(self.get_grad_log_posterior, current_X)

            # 2. Kernel matrix
            # diffs: (Np, Np, Nx)
            X_j = tf.expand_dims(current_X, 0)
            X_i = tf.expand_dims(current_X, 1)
            diffs = X_j - X_i

            scale = self.alpha * self.prior_var

            if self.kernel_type == 'scalar':
                dist_sq = tf.reduce_sum(tf.square(diffs)/scale, axis=2)
                K_mat = tf.exp(-0.5 * dist_sq) # (Np, Np)

                # grad K
                grad_K = tf.expand_dims(K_mat, 2) * (diffs / scale)

                term1 = tf.expand_dims(K_mat, 2) * tf.expand_dims(grad_log_p, 0)
                total = term1 + grad_K

            else: # Matrix
                dist_sq_vec = tf.square(diffs)/scale
                K_vec = tf.exp(-0.5 * dist_sq_vec)
                grad_K = K_vec * (diffs / scale)

                term1 = K_vec * tf.expand_dims(grad_log_p, 0)
                total = term1 + grad_K

            flow = tf.reduce_mean(total, axis=1) * self.D

            return i + 1, current_X + dt_tf * flow

        _, new_X = tf.while_loop(
            lambda i, x: i < n_steps,
            body,
            [0, self.X]
        )

        self.X.assign(new_X)
