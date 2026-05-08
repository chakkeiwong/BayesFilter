
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns
import copy


# =============================================================================
# TENSORFLOW SETUP
# =============================================================================
tf.config.set_visible_devices([], 'GPU') # Force CPU
DTYPE = tf.float64
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 11, 'lines.linewidth': 2, 'figure.figsize': (14, 4)})

# =============================================================================
# 1. LOCAL PARTICLE FLOW FILTER IMPLEMENTATION (Translated from NumPy)
# =============================================================================
class ParticleFlowFilterTF:
    """
    Standalone TensorFlow implementation of Particle Flow Filters.
    Includes EDH, LEDH, and Kernel PFF methods.
    """
    def __init__(self, num_particles, dim, obs_model='linear', sparsity_mask=None):
        self.Np = num_particles
        self.dim = dim
        self.obs_model = obs_model

        # Mask for linear observations
        if sparsity_mask is not None:
            self.mask_indices = tf.constant(np.where(sparsity_mask)[0], dtype=tf.int32)
            self.obs_dim = len(self.mask_indices)
        else:
            self.mask_indices = tf.range(dim, dtype=tf.int32)
            self.obs_dim = dim

    @tf.function
    def get_H_jacobian(self, x):
        # x shape: (dim,)
        if self.obs_model == 'linear':
            full_eye = tf.eye(self.dim, dtype=DTYPE)
            return tf.gather(full_eye, self.mask_indices)
        elif self.obs_model == 'square':
            # H = diag(2*x)
            # We want the rows corresponding to observed vars.
            # If obs_model is square, we usually assume full observation or mask applies
            # For Exp A (square), dim=5 and fully observed.
            return tf.linalg.diag(2.0 * x)
        return tf.eye(self.dim, dtype=DTYPE)

    @tf.function
    def get_predicted_observation(self, x):
        # x shape: (dim,)
        if self.obs_model == 'linear':
            return tf.gather(x, self.mask_indices)
        elif self.obs_model == 'square':
            return tf.square(x) # Exp A assumes full observation of squares
        return x

    @tf.function
    def run_edh_ledh(self, particles, P, y, R, mode='LEDH', n_steps=20):
        """
        Runs the Exact (EDH) or Local Exact (LEDH) Daum-Huang flow.
        Returns: final_particles, condition_numbers, flow_magnitudes
        """
        dt_lambda = tf.cast(1.0 / n_steps, DTYPE)
        X = tf.convert_to_tensor(particles, dtype=DTYPE)
        y = tf.convert_to_tensor(y, dtype=DTYPE)
        P = tf.convert_to_tensor(P, dtype=DTYPE)
        R = tf.convert_to_tensor(R, dtype=DTYPE)

        # TensorArrays to store diagnostics
        cond_nums = tf.TensorArray(DTYPE, size=n_steps)
        flow_mags = tf.TensorArray(DTYPE, size=n_steps)

        def body(l, current_X, c_nums, f_mags):
            lam = tf.cast(l, DTYPE) * dt_lambda
            x_mean = tf.reduce_mean(current_X, axis=0)

            # --- Per-Particle Loop (Vectorized via map_fn) ---
            def particle_update(x_i):
                # Linearization point
                lin_pt = x_mean if mode == 'EDH' else x_i

                H = self.get_H_jacobian(lin_pt)
                h_val = self.get_predicted_observation(lin_pt)
                e = h_val - tf.linalg.matvec(H, lin_pt)

                # S = lam * H P H^T + R
                HP = tf.matmul(H, P)
                HPHt = tf.matmul(HP, H, transpose_b=True)
                S = lam * HPHt + R

                # Condition Number (for diagnostics)
                # cond(S) = ||S|| * ||S^-1||. Expensive, so we approx or skip if speed needed.
                # For Exp C, we need it. svd is robust.
                s_vals = tf.linalg.svd(S, compute_uv=False)
                cond_val = tf.reduce_max(s_vals) / (tf.reduce_min(s_vals) + 1e-12)

                # Robust Inversion (The Fix for Exp C)
                # S_inv = inv(S + jitter)
                jitter = 1e-6 * tf.eye(self.obs_dim, dtype=DTYPE)
                S_inv = tf.linalg.inv(S + jitter)

                # A = -0.5 * P H^T S^-1 H
                # K_like term = P H^T S^-1
                PHt = tf.transpose(HP)
                K_like = tf.matmul(PHt, S_inv)
                A = -0.5 * tf.matmul(K_like, H)

                # b calculation
                # term1 = I + 2*lam*A
                I = tf.eye(self.dim, dtype=DTYPE)
                term1 = I + 2.0 * lam * A

                # term2_inner = K(y - e) + A*x_mean (Note: K here is P H^T S^-1)
                # Actually formula is: b = (I + 2lamA) [ (I+lamA) K (y-e) + A x_mean ]
                # K in Daum equations often refers to P H^T R^-1 or similar,
                # but here we follow the "Robust" derivation using S_inv directly.

                # Re-deriving 'b' from the NumPy code provided:
                # b = (I + 2*lam*A) @ ( (I + lam*A) @ (K @ (y - e)) + A @ x_mean )
                # where K = P H^T inv_R (Wait, NumPy code used S_inv in A, but inv_R in K?)
                # Let's check the provided NumPy snippet carefully:
                # A = -0.5 * P @ H.T @ S_inv @ H
                # K_np = P @ H.T @ inv_R (This K is the standard Kalman Gain if S was R)
                # BUT wait, the NumPy code actually does:
                # K = self.prior_cov @ H.T @ self.model.inv_R (Requires R inverse!)
                # And S is used ONLY for A.

                # Let's compute inv_R once outside if R is fixed.
                # Assuming R is invertible (Exp C has small R, but not singular R itself, S becomes singular).
                # However, if R -> 0, inv_R explodes.
                # The NumPy code for Exp C sets R_small = 1e-12 * I. inv_R is huge.
                # Let's replicate the logic exactly.

                R_inv = tf.linalg.inv(R + 1e-12 * tf.eye(self.obs_dim, dtype=DTYPE))
                K_np_equiv = tf.matmul(tf.matmul(P, H, transpose_b=True), R_inv)

                innov = y - e
                # (I + lam*A) @ (K @ innov)
                term2_part1 = tf.linalg.matvec(I + lam * A, tf.linalg.matvec(K_np_equiv, innov))
                term2_part2 = tf.linalg.matvec(A, x_mean)

                b = tf.linalg.matvec(term1, term2_part1 + term2_part2)

                drift = tf.linalg.matvec(A, x_i) + b

                # Soft clipping
                drift = tf.clip_by_value(drift, -1e4, 1e4)

                return drift, cond_val

            # Run for all particles
            drifts, conds = tf.map_fn(particle_update, current_X, fn_output_signature=(DTYPE, DTYPE))

            # Update X
            next_X = current_X + drifts * dt_lambda

            # Record diagnostics (max condition, mean flow)
            max_cond = tf.reduce_max(conds)
            mean_flow = tf.reduce_mean(tf.norm(drifts, axis=1))

            c_nums = c_nums.write(l, max_cond)
            f_mags = f_mags.write(l, mean_flow)

            return l + 1, next_X, c_nums, f_mags

        _, final_X, final_conds, final_mags = tf.while_loop(
            lambda l, x, c, f: l < n_steps,
            body,
            [0, X, cond_nums, flow_mags]
        )

        return final_X, final_conds.stack(), final_mags.stack()

    @tf.function
    def run_kernel_pff(self, particles, P, y, R, kernel_type='matrix', n_steps=50):
        dt_lambda = tf.constant(0.01, DTYPE)
        X = tf.convert_to_tensor(particles, dtype=DTYPE)
        y = tf.convert_to_tensor(y, dtype=DTYPE)
        P = tf.convert_to_tensor(P, dtype=DTYPE)
        R_inv = tf.linalg.inv(R)

        # Regularized P inverse
        P_inv = tf.linalg.inv(P + tf.eye(self.dim, dtype=DTYPE)*1e-6)
        alpha = 1.0 / tf.cast(self.Np, DTYPE)

        flow_mags = tf.TensorArray(DTYPE, size=n_steps)

        def body(l, current_X, f_mags):
            x_mean = tf.reduce_mean(current_X, axis=0)

            # 1. Pre-calc Gradients (Vectorized)
            def compute_grad(x_i):
                grad_prior = -tf.linalg.matvec(P_inv, x_i - x_mean)
                H = self.get_H_jacobian(x_i)
                pred = self.get_predicted_observation(x_i)
                innov = y - pred
                grad_lik = tf.linalg.matvec(tf.matmul(H, R_inv, transpose_a=True), innov)
                return tf.clip_by_value(grad_prior + grad_lik, -1e3, 1e3)

            grad_log_p_all = tf.map_fn(compute_grad, current_X) # (N, dim)

            # 2. Kernel Sum (All-to-All)
            # Expand dims for broadcasting: (N, 1, dim) vs (1, N, dim)
            X_i = tf.expand_dims(current_X, 1) # (N, 1, dim)
            X_j = tf.expand_dims(current_X, 0) # (1, N, dim)
            grad_j = tf.expand_dims(grad_log_p_all, 0) # (1, N, dim)

            diff = X_i - X_j # (N, N, dim)

            # Matrix Kernel Logic (Simplified vectorized)
            # We assume P is diagonal-ish or we treat dimensions independently for the 'matrix' kernel
            # logic in the provided NumPy code loop: `dist_sq_d = ... / P[d,d]`
            sig2 = tf.linalg.diag_part(P) + 1e-6 # (dim,)
            sig2_reshaped = tf.reshape(sig2, [1, 1, self.dim])

            # Dist sq per dimension: (diff^2) / (sig2 * alpha)
            dist_sq_d = tf.square(diff) / (sig2_reshaped * alpha)
            dist_sq_d = tf.clip_by_value(dist_sq_d, 0.0, 50.0)
            k_val = tf.exp(-0.5 * dist_sq_d) # (N, N, dim)

            # grad_k term: -k * diff / (sig2 * alpha)
            grad_k = -k_val * (diff / (sig2_reshaped * alpha))

            # Term inside sum: k * grad_log_p + grad_k
            # k_val is (N,N,dim), grad_j is (1,N,dim) -> broadcast ok
            kernel_term = k_val * grad_j + grad_k # (N, N, dim)

            kappa_sum = tf.reduce_sum(kernel_term, axis=1) # Sum over j -> (N, dim)

            # Final flow: P @ kappa_sum / Np
            f = tf.matmul(kappa_sum, P) / tf.cast(self.Np, DTYPE)
            f = tf.clip_by_value(f, -50.0, 50.0)

            next_X = current_X + f * dt_lambda
            mean_mag = tf.reduce_mean(tf.norm(f, axis=1))

            return l+1, next_X, f_mags.write(l, mean_mag)

        _, final_X, final_mags = tf.while_loop(
            lambda l, x, f: l < n_steps,
            body,
            [0, X, flow_mags]
        )

        return final_X, final_mags.stack()

# =============================================================================
# 2. EXPERIMENT RUNNERS
# =============================================================================
def run_all_experiments():
    # --- Exp A: Nonlinearity ---
    print("Running Exp A (Nonlinearity)...")
    dim = 5; Np = 100
    pf = ParticleFlowFilterTF(Np, dim, obs_model='square')

    np.random.seed(42)
    truth = np.random.randn(dim) + 2.0
    prior = np.random.randn(Np, dim) + truth
    P = np.cov(prior.T)
    y = truth**2 + np.random.normal(0, 0.5, dim)
    R = np.eye(dim) * 0.5

    x_edh, _, _ = pf.run_edh_ledh(prior, P, y, R, mode='EDH')
    x_ledh, _, _ = pf.run_edh_ledh(prior, P, y, R, mode='LEDH')
    x_kpff, _ = pf.run_kernel_pff(prior, P, y, R, kernel_type='matrix')

    plot_results("A", {'prior': prior, 'edh': x_edh.numpy(), 'ledh': x_ledh.numpy(), 'kpff': x_kpff.numpy(), 'truth': truth})

    # --- Exp B: Sparsity / High Dim ---
    print("Running Exp B (Sparsity L96 d=40)...")
    dim = 40; Np = 50
    mask = np.zeros(dim, dtype=bool); mask[::4] = True
    pf = ParticleFlowFilterTF(Np, dim, obs_model='linear', sparsity_mask=mask)

    truth = np.random.randn(dim)
    prior = np.random.randn(Np, dim) + truth
    P = np.cov(prior.T) + np.eye(dim)*0.01
    y = truth[mask] + np.random.normal(0, 0.5, 10)
    R = np.eye(10) * 0.5

    x_edh, _, _ = pf.run_edh_ledh(prior, P, y, R, mode='EDH')
    x_ledh, _, _ = pf.run_edh_ledh(prior, P, y, R, mode='LEDH')
    x_kpff, _ = pf.run_kernel_pff(prior, P, y, R, kernel_type='matrix')

    plot_results("B", {'prior': prior, 'edh': x_edh.numpy(), 'ledh': x_ledh.numpy(), 'kpff': x_kpff.numpy(), 'truth': truth})

    # --- Exp C: Conditioning ---
    print("Running Exp C (Conditioning R->0)...")
    dim = 10; Np = 20
    pf = ParticleFlowFilterTF(Np, dim, obs_model='linear')
    prior = np.random.randn(Np, dim)
    P = np.eye(dim)
    y = np.zeros(dim)

    # Extremely small R -> Singular S -> Stiffness
    # In TF, we must use tf.eye to ensure precision
    R_small = np.eye(dim) * 1e-12

    # Run and capture diagnostics
    _, c_edh, f_edh = pf.run_edh_ledh(prior, P, y, R_small, mode='EDH')
    _, c_ledh, f_ledh = pf.run_edh_ledh(prior, P, y, R_small, mode='LEDH')

    plot_results("C", {
        'diag_edh': {'cond_num': c_edh.numpy(), 'flow_mag': f_edh.numpy()},
        'diag_ledh': {'cond_num': c_ledh.numpy(), 'flow_mag': f_ledh.numpy()}
    })

# =============================================================================
# 3. PLOTTING
# =============================================================================
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
        plt.savefig('exp_a_results.png', dpi=300)

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
        plt.savefig('exp_b_results.png', dpi=300)

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
        plt.savefig('exp_c_results.png', dpi=300)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_all_experiments()
