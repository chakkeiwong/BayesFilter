
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import copy

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Enable eager execution
tf.config.run_functions_eagerly(True)
DTYPE = tf.float64

# ==========================================
# 0. Configuration
# ==========================================
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 11, 'lines.linewidth': 2, 'figure.figsize': (14, 4)})

# ==========================================
# 1. System Dynamics: Lorenz 96 (TensorFlow)
# ==========================================
class Lorenz96TF:
    def __init__(self, D=40, F=8.0, dt=0.01):
        self.D = D
        self.F = tf.constant(F, dtype=DTYPE)
        self.dt = tf.constant(dt, dtype=DTYPE)

    @tf.function
    def f(self, x):
        # x_{i+1} -> roll -1, x_{i-2} -> roll +2, x_{i-1} -> roll +1
        x_plus_1 = tf.roll(x, shift=-1, axis=-1)
        x_minus_2 = tf.roll(x, shift=2, axis=-1)
        x_minus_1 = tf.roll(x, shift=1, axis=-1)
        return (x_plus_1 - x_minus_2) * x_minus_1 - x + self.F

    @tf.function
    def step(self, x):
        k1 = self.f(x)
        k2 = self.f(x + k1 * self.dt / 2.0)
        k3 = self.f(x + k2 * self.dt / 2.0)
        k4 = self.f(x + k3 * self.dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * (self.dt / 6.0)

# ==========================================
# 2. Particle Flow Filter (Robust - TensorFlow)
# ==========================================
class ParticleFlowFilterTF:
    def __init__(self, num_particles, dim, obs_model='linear', sparsity_mask=None):
        self.Np = num_particles
        self.dim = dim
        self.obs_model = obs_model

        # Mask handling for sparse observations
        if sparsity_mask is not None:
            self.mask = tf.constant(sparsity_mask, dtype=tf.bool)
            self.obs_dim = int(np.sum(sparsity_mask))
            # Indices for gather
            self.obs_indices = tf.where(self.mask)[:, 0]
        else:
            self.mask = tf.ones(dim, dtype=tf.bool)
            self.obs_dim = dim
            self.obs_indices = tf.range(dim)

    @tf.function
    def get_H_jacobian(self, x):
        """Returns Jacobian H (obs_dim, state_dim)."""
        if self.obs_model == 'linear':
            # Identity masked
            # Create full eye then gather
            H_full = tf.eye(self.dim, dtype=DTYPE)
            return tf.gather(H_full, self.obs_indices)

        elif self.obs_model == 'square':
            # H = diag(2*x) masked
            H_full = tf.linalg.diag(2.0 * x)
            return tf.gather(H_full, self.obs_indices)

        return tf.zeros((self.obs_dim, self.dim), dtype=DTYPE)

    @tf.function
    def get_predicted_observation(self, x):
        """Returns h(x)."""
        if self.obs_model == 'linear':
            full_obs = x
        elif self.obs_model == 'square':
            full_obs = x**2

        return tf.gather(full_obs, self.obs_indices)

    def run_edh_ledh(self, particles, P, y, R, mode='LEDH', n_steps=20):
        """
        Runs Exact Daum-Huang (EDH) or Local EDH flow.
        Returns updated particles and diagnostics.
        """
        dt_lambda = 1.0 / n_steps
        x = tf.Variable(tf.cast(particles, DTYPE))

        # Casting inputs
        P_curr = tf.cast(P, DTYPE)
        y = tf.cast(y, DTYPE)
        R = tf.cast(R, DTYPE)

        diag_cond_num = []
        diag_flow_mag = []

        for l in range(n_steps):
            lam = l * dt_lambda
            x_mean = tf.reduce_mean(x, axis=0)

            # Temporary storage for drift vectors
            dx_list = []
            max_cond = 0.0

            # Loop over particles (Manual loop or map_fn needed for LEDH dependence on x[i])
            # For diagnostics, manual loop is easier to track max_cond per step

            for i in range(self.Np):
                p_i = x[i]

                # Linearization point
                linearize_at = x_mean if mode == 'EDH' else p_i

                H = self.get_H_jacobian(linearize_at)
                h_val = self.get_predicted_observation(linearize_at)

                # e = h(x) - Hx
                e = h_val - tf.tensordot(H, linearize_at, axes=1)

                # S = lam * H P H^T + R
                HP = tf.matmul(H, P_curr)
                S = lam * tf.matmul(HP, H, transpose_b=True) + R

                # Condition Number Check (using numpy for svd if needed, or tf.svd)
                # tf.linalg.cond is available
                # cond = tf.linalg.cond(S) # Expensive in loop?
                s_vals = tf.linalg.svd(S, compute_uv=False)
                cond = tf.reduce_max(s_vals) / (tf.reduce_min(s_vals) + 1e-12)
                if cond > max_cond: max_cond = cond

                # Invert S
                try:
                    S_inv = tf.linalg.inv(S)
                except:
                    # Fallback regularization
                    S_inv = tf.linalg.inv(S + tf.eye(self.obs_dim, dtype=DTYPE) * 1e-6)

                # A = -0.5 P H^T S^-1 H
                # K = P H^T S^-1  (Note: Code snippet used K in term2 calc differently?)
                # Code: K not explicitly defined but part of term2: P H^T S^-1 (y-e)

                P_HT_Sinv = tf.matmul(tf.matmul(P_curr, H, transpose_b=True), S_inv)
                A = -0.5 * tf.matmul(P_HT_Sinv, H)

                # b term calculation
                # term1 = I + 2*lam*A
                I = tf.eye(self.dim, dtype=DTYPE)
                term1 = I + 2 * lam * A

                # term2 = (I + lam*A) @ (P H^T S^-1 (y-e)) + A @ x_mean
                # innovation = y - e
                innov = y - e
                gain_innov = tf.tensordot(P_HT_Sinv, innov, axes=1) # (dim,)

                # (I + lam*A) @ gain_innov
                part1 = tf.tensordot(I + lam * A, gain_innov, axes=1)

                # A @ x_mean
                part2 = tf.tensordot(A, x_mean, axes=1)

                term2 = part1 + part2

                b = tf.tensordot(term1, term2, axes=1)

                # drift = A x + b
                drift = tf.tensordot(A, p_i, axes=1) + b

                # Soft clipping
                drift = tf.clip_by_value(drift, -1e4, 1e4)
                dx_list.append(drift)

            dx = tf.stack(dx_list)

            # Diagnostics
            diag_cond_num.append(max_cond)
            flow_mags = tf.norm(dx, axis=1)
            diag_flow_mag.append(tf.reduce_mean(flow_mags).numpy())

            # Update
            x.assign_add(dx * dt_lambda)

            # Divergence check
            if tf.reduce_mean(tf.abs(x)) > 1e4:
                break

        return x.numpy(), {'cond_num': np.array(diag_cond_num), 'flow_mag': np.array(diag_flow_mag)}

    def run_kernel_pff(self, particles, P, y, R, kernel_type='matrix', n_steps=50):
        """Matrix/Scalar Kernel PFF implementation."""
        x = tf.Variable(tf.cast(particles, DTYPE))
        dt_lambda = 0.01

        P_curr = tf.cast(P, DTYPE)
        y = tf.cast(y, DTYPE)
        R = tf.cast(R, DTYPE)

        inv_R = tf.linalg.inv(R)
        inv_P = tf.linalg.inv(P_curr + tf.eye(self.dim, dtype=DTYPE) * 1e-6)
        alpha = 1.0 / self.Np

        diag_flow_mag = []

        for l in range(n_steps):
            x_mean = tf.reduce_mean(x, axis=0)

            # Pre-calc gradients
            # Map over particles
            def get_grad(p_i):
                # Grad Prior: -P^-1 (x - mu)
                gp = -tf.tensordot(inv_P, p_i - x_mean, axes=1)

                # Grad Likelihood: H^T R^-1 (y - h(x))
                H = self.get_H_jacobian(p_i)
                h_val = self.get_predicted_observation(p_i)
                innov = y - h_val
                gl = tf.matmul(tf.matmul(H, inv_R, transpose_a=True), tf.reshape(innov, [-1, 1]))
                gl = tf.squeeze(gl)

                return tf.clip_by_value(gp + gl, -1e3, 1e3)

            grad_log_p_all = tf.map_fn(get_grad, x) # (N, dim)

            # Kernel Flow
            # Compute flow f[i] for each particle
            # Matrix kernel logic: k(x,y) depends on (x-y)^T P^-1 (x-y)
            # Need to implement the nested loop or vectorized equivalent

            # Pairwise diffs: (N, N, dim)
            diffs = tf.expand_dims(x, 1) - tf.expand_dims(x, 0)

            # Compute distances for kernel
            # matrix: dist = diff^T (P^-1 / alpha) diff ?
            # Code says: (diff[d]**2) / (sig2 * alpha) per dimension
            # Where sig2 = P[d,d]

            sig2 = tf.linalg.diag_part(P_curr) + 1e-6 # (dim,)

            # Scaled squared diffs: (N, N, dim)
            scaled_sq_diffs = (diffs**2) / (tf.reshape(sig2, [1, 1, -1]) * alpha)
            scaled_sq_diffs = tf.clip_by_value(scaled_sq_diffs, 0, 50)

            if kernel_type == 'matrix':
                # k_vec = exp(-0.5 * dist) per dimension
                k_val = tf.exp(-0.5 * scaled_sq_diffs) # (N, N, dim)

                # grad_k = -k * (diff / (sig2 * alpha))
                grad_k = -k_val * (diffs / (tf.reshape(sig2, [1, 1, -1]) * alpha)) # (N, N, dim)

                # Term: k * grad_log_p_j + grad_k
                # grad_log_p_all is (N, dim). Broadcast to (1, N, dim)
                term = k_val * tf.expand_dims(grad_log_p_all, 0) + grad_k

                # Sum over j (axis 1) -> (N, dim) (kappa_sum)
                kappa_sum = tf.reduce_sum(term, axis=1)

            # Final flow: P @ kappa_sum / N
            f = tf.matmul(kappa_sum, P_curr, transpose_b=True) / self.Np

            f = tf.clip_by_value(f, -50, 50)
            x.assign_add(f * dt_lambda)

            diag_flow_mag.append(np.mean(np.linalg.norm(f.numpy(), axis=1)))

        return x.numpy(), {'flow_mag': diag_flow_mag}

# ==========================================
# 3. Plotting Logic
# ==========================================
def plot_results(experiment_name, data_dict):
    if experiment_name == "A":
        fig, axes = plt.subplots(1, 2)
        ax = axes[0]
        ax.scatter(data_dict['prior'][:,0], data_dict['prior'][:,1], c='gray', alpha=0.3, label='Prior')
        ax.scatter(data_dict['edh'][:,0], data_dict['edh'][:,1], c='red', marker='x', label='EDH')
        ax.scatter(data_dict['ledh'][:,0], data_dict['ledh'][:,1], c='blue', marker='^', alpha=0.5, label='LEDH')
        ax.scatter(data_dict['kpff'][:,0], data_dict['kpff'][:,1], c='green', marker='o', alpha=0.5, label='KPFF')
        t = data_dict['truth']
        ax.scatter(t[0], t[1], c='k', s=200, marker='*', label='Truth')
        ax.set_title("Exp A: Nonlinearity (Phase Space)")
        ax.legend(fontsize=8)

        ax = axes[1]
        sns.kdeplot(data_dict['edh'][:,0], color='red', ax=ax, label='EDH')
        sns.kdeplot(data_dict['ledh'][:,0], color='blue', ax=ax, label='LEDH')
        sns.kdeplot(data_dict['kpff'][:,0], color='green', fill=True, alpha=0.2, ax=ax, label='KPFF')
        ax.axvline(t[0], color='k', linestyle='--')
        ax.set_title("Marginal Density (Dim 0)")
        ax.legend()
        plt.savefig('exp_a_nonlinearity.png')
        print(">> Saved exp_a_nonlinearity.png")

    elif experiment_name == "B":
        fig, axes = plt.subplots(1, 2)
        truth = data_dict['truth']
        ax = axes[0]
        sns.kdeplot(data_dict['edh'][:,0], color='red', ax=ax, label='EDH')
        sns.kdeplot(data_dict['ledh'][:,0], color='blue', ax=ax, label='LEDH')
        sns.kdeplot(data_dict['kpff'][:,0], color='green', fill=True, alpha=0.2, ax=ax, label='KPFF')
        ax.axvline(truth[0], color='k', linestyle='--')
        ax.set_title("Exp B: Observed State Density (L96 d=40)")
        ax.legend()

        ax = axes[1]
        def get_rmse(p): return np.sqrt(np.mean((np.mean(p, axis=0) - truth)**2))
        rmses = [get_rmse(data_dict[k]) for k in ['edh', 'ledh', 'kpff']]
        bars = ax.bar(['EDH', 'LEDH', 'KPFF'], rmses, color=['red', 'blue', 'green'])
        ax.bar_label(bars, fmt='%.2f')
        ax.set_title("RMSE Comparison (High Dim/Sparse)")
        ax.set_ylabel("RMSE")
        plt.savefig('exp_b_sparsity.png')
        print(">> Saved exp_b_sparsity.png")

    elif experiment_name == "C":
        fig, axes = plt.subplots(1, 2)
        diag_edh = data_dict['diag_edh']
        diag_ledh = data_dict['diag_ledh']

        # 1. Condition Number Plot
        ax = axes[0]
        ax.semilogy(diag_edh['cond_num'], 'r-o', label='EDH Cond Num')
        ax.semilogy(diag_ledh['cond_num'], 'b-^', label='LEDH Cond Num')
        ax.axhline(1e15, color='k', ls=':', label='Limit')
        ax.set_title("Exp C: Condition Number (R -> 0)")
        ax.set_xlabel("Pseudo-step")
        ax.legend()

        # 2. Flow Magnitude Plot
        ax = axes[1]
        ax.plot(diag_edh['flow_mag'], 'r--', label='EDH Flow')
        ax.plot(diag_ledh['flow_mag'], 'b-', label='LEDH Flow')
        ax.set_title("Flow Magnitude (Stiffness)")
        ax.legend()
        plt.savefig('exp_c_conditioning.png')
        print(">> Saved exp_c_conditioning.png")

    plt.tight_layout()
    # plt.show() # Commented out for batch run

# ==========================================
# 4. Runners
# ==========================================
def run_all_experiments():
    # --- Exp A: Nonlinearity ---
    print("Running Exp A (Nonlinearity)...")
    dim = 5; Np = 100
    pf = ParticleFlowFilterTF(Np, dim, obs_model='square')
    truth = np.random.randn(dim) + 2.0
    prior = np.random.randn(Np, dim) + truth
    P = np.cov(prior.T)
    y = truth**2 + np.random.normal(0, 0.5, dim)
    R = np.eye(dim) * 0.5

    x_edh, _ = pf.run_edh_ledh(prior, P, y, R, mode='EDH')
    x_ledh, _ = pf.run_edh_ledh(prior, P, y, R, mode='LEDH')
    x_kpff, _ = pf.run_kernel_pff(prior, P, y, R, kernel_type='matrix')

    plot_results("A", {'prior': prior, 'edh': x_edh, 'ledh': x_ledh, 'kpff': x_kpff, 'truth': truth})

    # --- Exp B: Sparsity / High Dim ---
    print("Running Exp B (Sparsity L96 d=40)...")
    dim = 40; Np = 50
    mask = np.zeros(dim, dtype=bool); mask[::4] = True # Observe 10/40
    pf = ParticleFlowFilterTF(Np, dim, obs_model='linear', sparsity_mask=mask)

    truth = np.random.randn(dim)
    prior = np.random.randn(Np, dim) + truth
    P = np.cov(prior.T) + np.eye(dim)*0.01
    y = truth[mask] + np.random.normal(0, 0.5, 10)
    R = np.eye(10) * 0.5

    x_edh, _ = pf.run_edh_ledh(prior, P, y, R, mode='EDH')
    x_ledh, _ = pf.run_edh_ledh(prior, P, y, R, mode='LEDH')
    x_kpff, _ = pf.run_kernel_pff(prior, P, y, R, kernel_type='matrix')

    plot_results("B", {'prior': prior, 'edh': x_edh, 'ledh': x_ledh, 'kpff': x_kpff, 'truth': truth})

    # --- Exp C: Conditioning ---
    print("Running Exp C (Conditioning R->0)...")
    dim = 10; Np = 20
    pf = ParticleFlowFilterTF(Np, dim, obs_model='linear')
    prior = np.random.randn(Np, dim)
    P = np.eye(dim)
    y = np.zeros(dim)

    # Extremely small R -> Singular S -> Stiffness
    R_small = np.eye(dim) * 1e-12

    _, diag_edh = pf.run_edh_ledh(prior, P, y, R_small, mode='EDH')
    _, diag_ledh = pf.run_edh_ledh(prior, P, y, R_small, mode='LEDH')

    plot_results("C", {'diag_edh': diag_edh, 'diag_ledh': diag_ledh})

if __name__ == "__main__":
    run_all_experiments()
