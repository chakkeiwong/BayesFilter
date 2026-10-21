"""
Dai & Daum (2022) - Stiffness Mitigation in Stochastic Particle Flow Filters
Replication of Example 1: Single-step Bayesian update for a static 2D target.

Setup:
  - Two passive IR sensors at [3.5,0] and [-3.5,0], measuring bearing angles
  - True target at [4,4], stationary
  - Prior: N([3,5], diag(1000, 2)) — intentionally poor
  - Measurement: z = [0.4754, 1.1868] (fixed, as in the paper)
  - 20 MC runs using Common Random Numbers (CRN)
  - Compare: straight-line beta(lam)=lam vs optimal beta*(lam)
"""

import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# =============================================================================
# Problem Setup (exactly as in the paper)
# =============================================================================
SENSORS = np.array([[3.5, 0.0], [-3.5, 0.0]])
R_meas = 0.04 * np.eye(2)
x_prior_mean = np.array([3.0, 5.0])
P_prior = np.array([[1000.0, 0.0], [0.0, 2.0]])
x_true = np.array([4.0, 4.0])
z_obs = np.array([0.4754, 1.1868])  # fixed measurement from paper

# Particle flow parameters
Q_flow = np.diag([4.0, 0.4])
MU = 0.2           # stiffness weight
N_PARTICLES = 50
N_LAMBDA = 50      # pseudo-time steps
N_MC = 20          # Monte Carlo runs


# =============================================================================
# Measurement Model
# =============================================================================
def meas_func(x):
    """Bearing angles from each sensor to target at x."""
    z = np.zeros(len(SENSORS))
    for i, s in enumerate(SENSORS):
        z[i] = np.arctan2(x[1] - s[1], x[0] - s[0])
    return z


def meas_jacobian(x):
    """Jacobian H = dh/dx at x."""
    H = np.zeros((len(SENSORS), 2))
    for i, s in enumerate(SENSORS):
        dx = x[0] - s[0]
        dy = x[1] - s[1]
        r2 = dx**2 + dy**2
        H[i, 0] = -dy / r2
        H[i, 1] = dx / r2
    return H


# =============================================================================
# Optimal beta*(lambda) via direct optimization of Eq (25)
# =============================================================================
def solve_optimal_beta(mu=MU, n_pts=N_LAMBDA + 1):
    """
    Minimize J(beta, u) = integral_0^1 [0.5*u^2 + mu*kappa_*(M)] dlam
    subject to dbeta/dlam = u, beta(0)=0, beta(1)=1.

    Uses nuclear norm condition number kappa_*(M) = tr(M)*tr(M^{-1})
    as in Remark 3.2 and the paper's numerical examples.
    """
    P_inv = inv(P_prior)
    R_inv = inv(R_meas)
    H = meas_jacobian(x_prior_mean)
    neg_A0 = P_inv             # = -A0 = P^{-1}
    neg_Ah = H.T @ R_inv @ H  # = -Ah = H^T R^{-1} H

    lam = np.linspace(0, 1, n_pts)
    dlam = lam[1] - lam[0]

    def objective(beta_interior):
        beta = np.concatenate([[0.0], beta_interior, [1.0]])
        u = np.diff(beta) / dlam
        energy = 0.5 * np.sum(u**2) * dlam
        cond_sum = 0.0
        for j in range(n_pts):
            M = neg_A0 + beta[j] * neg_Ah
            ev = np.linalg.eigvalsh(M)
            if np.min(ev) > 0:
                cond_sum += np.sum(ev) * np.sum(1.0 / ev)
            else:
                cond_sum += 1e10
        cond_sum *= mu * dlam
        return energy + cond_sum

    beta_init = np.linspace(0, 1, n_pts)[1:-1]
    res = minimize(objective, beta_init, method='L-BFGS-B',
                   options={'maxiter': 1000, 'ftol': 1e-14})

    beta_opt = np.concatenate([[0.0], res.x, [1.0]])
    beta_dot = np.gradient(beta_opt, lam)

    J_straight = objective(beta_init)
    J_optimal = res.fun

    return lam, beta_opt, beta_dot, J_straight, J_optimal


# =============================================================================
# Stochastic Particle Flow Filter — Single Bayesian Update
# =============================================================================
def spff_single_update(z, beta_sched, beta_dot_sched, lam_sched, rng):
    """
    Single-step Bayesian update: prior p0 -> posterior p1
    using stochastic particle flow with given beta schedule.

    Returns: x_est (posterior mean), P_est (posterior covariance), particles
    """
    n = 2
    N = N_PARTICLES

    # Sample particles from prior
    particles = rng.multivariate_normal(x_prior_mean, P_prior, size=N)

    # Constant Hessians (Gaussian assumption)
    P_inv = inv(P_prior)
    R_inv = inv(R_meas)
    A0 = -P_inv               # nabla^2 log p0
    Q_sqrt = np.diag(np.sqrt(np.diag(Q_flow)))

    n_steps = len(lam_sched) - 1

    for j in range(n_steps):
        dlam = lam_sched[j + 1] - lam_sched[j]
        beta_j = beta_sched[j + 1]
        bdot_j = beta_dot_sched[min(j + 1, len(beta_dot_sched) - 1)]

        # Linearize measurement at current particle cloud mean
        x_mean = np.mean(particles, axis=0)
        H_j = meas_jacobian(x_mean)
        Ah = -H_j.T @ R_inv @ H_j     # nabla^2 log h

        # S = nabla^2 log p = A0 + beta*Ah  (with alpha+beta=1)
        S = A0 + beta_j * Ah
        try:
            S_inv = inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Drift coefficients (Eqs 11-12, with alpha+beta=1, alpha*bdot - adot*beta = bdot)
        K1 = 0.5 * Q_flow + (bdot_j / 2.0) * S_inv @ Ah @ S_inv
        K2 = -bdot_j * S_inv

        # Propagate each particle
        for i in range(N):
            xi = particles[i]

            # Gradients at particle location
            grad_log_p0 = A0 @ (xi - x_prior_mean)

            z_pred = meas_func(xi)
            innov = z - z_pred
            innov = (innov + np.pi) % (2 * np.pi) - np.pi
            grad_log_h = H_j.T @ R_inv @ innov

            # nabla log p = nabla log p0 + beta * nabla log h
            grad_log_p = grad_log_p0 + beta_j * grad_log_h

            # Drift: f = K1 * nabla log p + K2 * nabla log h  (Eq 10)
            f = K1 @ grad_log_p + K2 @ grad_log_h

            # Brownian increment
            dw = rng.randn(n) * np.sqrt(dlam)

            # Euler-Maruyama: dx = f*dlam + q*dw  (Eq 8)
            particles[i] = xi + f * dlam + Q_sqrt @ dw

    x_est = np.mean(particles, axis=0)
    P_est = np.cov(particles.T)
    return x_est, P_est, particles


# =============================================================================
# Condition number utility
# =============================================================================
def compute_kappa_L2(beta):
    """Spectral condition number kappa_2(M(beta))."""
    P_inv = inv(P_prior)
    R_inv = inv(R_meas)
    H = meas_jacobian(x_prior_mean)
    M = P_inv + beta * (H.T @ R_inv @ H)
    ev = np.linalg.eigvalsh(M)
    if np.min(ev) <= 0:
        return 1e10
    return np.max(ev) / np.min(ev)


# =============================================================================
# Stiffness ratio of Jacobian F (Eq 22)
# =============================================================================
def compute_stiffness_ratio(beta, beta_dot):
    """Stiffness ratio R_stiff = |Re(lam_max)| / |Re(lam_min)| of F (Eq 22)."""
    P_inv = inv(P_prior)
    R_inv = inv(R_meas)
    H = meas_jacobian(x_prior_mean)
    neg_Ah = H.T @ R_inv @ H
    A0 = -P_inv
    Ah = -neg_Ah

    S = A0 + beta * Ah
    try:
        S_inv = inv(S)
    except:
        return np.nan

    # F = 0.5*Q*S - (bdot/2)*S^{-1}*Ah  (Eq 22, with alpha+beta=1)
    F = 0.5 * Q_flow @ S - (beta_dot / 2.0) * S_inv @ Ah

    eigvals = np.linalg.eigvals(F)
    re_parts = np.abs(np.real(eigvals))
    re_parts = re_parts[re_parts > 1e-15]
    if len(re_parts) < 2:
        return 1.0
    return np.max(re_parts) / np.min(re_parts)


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    print("=" * 70)
    print("Dai & Daum (2022) Example 1: Stochastic PFF Stiffness Mitigation")
    print("=" * 70)

    # --- 1. Solve for optimal beta ---
    print("\n[1] Solving for optimal beta*(lambda)...")
    t0 = time.time()
    lam, beta_opt, bdot_opt, J_str, J_opt = solve_optimal_beta()
    opt_time = time.time() - t0

    beta_str = lam.copy()
    bdot_str = np.ones_like(lam)

    print(f"    J(straight)  = {J_str:.4f}")
    print(f"    J(optimal)   = {J_opt:.4f}")
    print(f"    Reduction    = {(1 - J_opt/J_str)*100:.1f}%")
    print(f"    Solve time   = {opt_time:.3f}s")
    print(f"    Paper reports: J_straight=4.0, J_optimal=3.4")

    # --- 2. Run 20 MC trials with CRN ---
    print(f"\n[2] Running {N_MC} Monte Carlo trials (CRN), {N_PARTICLES} particles each...")

    rmse_straight = []
    rmse_optimal = []
    trP_straight = []
    trP_optimal = []
    time_straight = []
    time_optimal = []

    print(f"\n    {'MC':>3} {'RMSE_str':>10} {'RMSE_opt':>10} {'trP_str':>10} {'trP_opt':>10}")
    print("    " + "-" * 46)

    for mc in range(N_MC):
        seed = 1000 + mc

        # Straight-line beta = lambda
        rng = np.random.RandomState(seed)  # CRN: same seed
        t0 = time.time()
        x_s, P_s, _ = spff_single_update(z_obs, beta_str, bdot_str, lam, rng)
        time_straight.append(time.time() - t0)
        err_s = np.linalg.norm(x_s - x_true)
        rmse_straight.append(err_s)
        trP_straight.append(np.trace(P_s))

        # Optimal beta*
        rng = np.random.RandomState(seed)  # CRN: same seed
        t0 = time.time()
        x_o, P_o, _ = spff_single_update(z_obs, beta_opt, bdot_opt, lam, rng)
        time_optimal.append(time.time() - t0)
        err_o = np.linalg.norm(x_o - x_true)
        rmse_optimal.append(err_o)
        trP_optimal.append(np.trace(P_o))

        print(f"    {mc+1:>3} {err_s:>10.4f} {err_o:>10.4f} {np.trace(P_s):>10.2f} {np.trace(P_o):>10.2f}")

    rmse_straight = np.array(rmse_straight)
    rmse_optimal = np.array(rmse_optimal)
    trP_straight = np.array(trP_straight)
    trP_optimal = np.array(trP_optimal)

    print("    " + "-" * 46)
    print(f"    {'avg':>3} {np.mean(rmse_straight):>10.4f} {np.mean(rmse_optimal):>10.4f} "
          f"{np.mean(trP_straight):>10.2f} {np.mean(trP_optimal):>10.2f}")

    print(f"\n    Average RMSE reduction: {(1 - np.mean(rmse_optimal)/np.mean(rmse_straight))*100:.1f}%")
    print(f"    Average tr(P) reduction: {(1 - np.mean(trP_optimal)/np.mean(trP_straight))*100:.1f}%")
    print(f"    Avg time (straight): {np.mean(time_straight):.4f}s")
    print(f"    Avg time (optimal):  {np.mean(time_optimal):.4f}s")
    print(f"    Beta* solve overhead: {opt_time:.4f}s")

    # --- 3. Plots ---
    print("\n[3] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dai & Daum (2022) Example 1 — Stochastic PFF with Stiffness Mitigation',
                 fontsize=13, fontweight='bold')

    # (a) beta(lambda) comparison — matches Fig 2(a)
    ax = axes[0, 0]
    ax.plot(lam, beta_str, 'b--', lw=1.5, label=r'$\beta(\lambda)=\lambda$')
    ax.plot(lam, beta_opt, 'r-', lw=2, label=r'optimal $\beta^*(\lambda)$')
    ax.set_xlabel(r'$\lambda$', fontsize=11)
    ax.set_ylabel(r'$\beta(\lambda)$', fontsize=11)
    ax.set_title(r'(a) $\beta(\lambda)$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Deviation e = beta* - lambda — matches Fig 2(b)
    ax = axes[0, 1]
    ax.plot(lam, beta_opt - lam, 'r-', lw=2)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(r'$\lambda$', fontsize=11)
    ax.set_ylabel(r'$e = \beta^*(\lambda) - \lambda$', fontsize=11)
    ax.set_title(r'(b) Deviation $e = \beta^* - \lambda$', fontsize=11)
    ax.grid(True, alpha=0.3)

    # (c) Stiffness ratio R_stiff — matches Fig 2(d)
    ax = axes[1, 0]
    lam_eval = np.linspace(0.02, 0.98, 100)
    Rstiff_str = []
    Rstiff_opt = []
    for l in lam_eval:
        Rstiff_str.append(compute_stiffness_ratio(l, 1.0))
        b = np.interp(l, lam, beta_opt)
        bd = np.interp(l, lam, bdot_opt)
        Rstiff_opt.append(compute_stiffness_ratio(b, bd))
    ax.semilogy(lam_eval, Rstiff_str, 'b--', lw=1.5, label=r'$\beta(\lambda)=\lambda$')
    ax.semilogy(lam_eval, Rstiff_opt, 'r-', lw=2, label=r'optimal $\beta^*(\lambda)$')
    ax.set_xlabel(r'$\lambda$', fontsize=11)
    ax.set_ylabel(r'$R_{\mathrm{stiff}}$', fontsize=11)
    ax.set_title(r'(d) Stiffness ratio $R_{\mathrm{stiff}}$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (d) RMSE per MC run — matches Table I
    ax = axes[1, 1]
    mc_idx = np.arange(1, N_MC + 1)
    width = 0.35
    ax.bar(mc_idx - width/2, rmse_straight, width, color='steelblue', alpha=0.8,
           label=r'$\beta(\lambda)=\lambda$')
    ax.bar(mc_idx + width/2, rmse_optimal, width, color='crimson', alpha=0.8,
           label=r'optimal $\beta^*(\lambda)$')
    ax.axhline(np.mean(rmse_straight), color='steelblue', ls='--', lw=1, alpha=0.7)
    ax.axhline(np.mean(rmse_optimal), color='crimson', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('MC run index', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('(c) RMSE per MC run', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = "dai2022_example1_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {out_path}")
    plt.close()
    print("    Saved: dai2022_example1_results.png")

    # --- Table figure (matching Table I format) ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.axis('off')

    col_labels = ['MC index', r'RMSE ($\beta=\lambda$)', r'RMSE ($\beta^*$)',
                  r'tr(P) ($\beta=\lambda$)', r'tr(P) ($\beta^*$)']
    table_data = []
    for mc in range(N_MC):
        table_data.append([
            str(mc + 1),
            f"{rmse_straight[mc]:.4f}",
            f"{rmse_optimal[mc]:.4f}",
            f"{trP_straight[mc]:.2f}",
            f"{trP_optimal[mc]:.2f}"
        ])
    table_data.append([
        'average',
        f"{np.mean(rmse_straight):.4f}",
        f"{np.mean(rmse_optimal):.4f}",
        f"{np.mean(trP_straight):.2f}",
        f"{np.mean(trP_optimal):.2f}"
    ])

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2E5090')
            cell.set_text_props(color='white', fontweight='bold')
        elif row == len(table_data):  # average row
            cell.set_facecolor('#E8D44D')
            cell.set_text_props(fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D6E4F0')

    ax2.set_title('Table I: Performance Comparison (replicating Dai & Daum 2022, Table I)',
                   fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    out_path = "dai2022_example1_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: dai2022_example1_table.png")

    # --- Computing time comparison ---
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    mc_idx = np.arange(1, N_MC + 1)
    total_opt = np.array(time_optimal) + opt_time / N_MC  # amortize BVP solve
    ax3.plot(mc_idx, time_straight, 'b-s', ms=4, lw=1.5, label=r'$\beta(\lambda)=\lambda$')
    ax3.plot(mc_idx, total_opt, 'r-o', ms=4, lw=1.5, label=r'optimal $\beta^*(\lambda)$')
    ax3.axhline(np.mean(time_straight), color='b', ls='--', alpha=0.5)
    ax3.axhline(np.mean(total_opt), color='r', ls='--', alpha=0.5)
    ax3.set_xlabel('MC run index', fontsize=11)
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Computing Time per MC Run (cf. Fig. 3 in paper)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "dai2022_example1_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: dai2022_example1_time.png")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == '__main__':
    main()
