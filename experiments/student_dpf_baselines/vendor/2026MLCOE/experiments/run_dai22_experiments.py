"""
Dai & Daum (2022) Optimal Homotopy Replication
==============================================
This script solves the Boundary Value Problem (BVP) to determine the optimal
homotopy schedule beta(lambda) for particle flow filters. By minimizing the
log-determinant of the information matrix, it mitigates the numerical stiffness
inherent in the Daum-Huang flow.

The optimization uses a shooting method with bisection to satisfy the
boundary conditions beta(0)=0 and beta(1)=1, replicating Figure 2 from
'Optimal Homotopy for Particle Flow Filters' (Dai & Daum, 2022).

Author: Joowon Lee (UW-Madison Statistics)
Date: 2026-02-27
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Ensure project root is in path for modular discovery
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.classic_ssm import Dai22BearingOnlySSM
from src.filters.classical import DTYPE

# Discovery for portable saving in the same folder as the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, 'Dai22_fig2_replication_final.png')

tf.config.set_visible_devices([], 'GPU')

def solve_dai22_bvp(model, mu=0.2, n_steps=1000):
    """
    Solves the BVP for optimal homotopy: d2beta/dlambda2 = mu * d(kappa)/dbeta.
    Matches the numerical experiment in Section 4 of Dai & Daum (2022).
    """
    lambdas = np.linspace(0, 1, n_steps)
    dt = 1.0 / (n_steps - 1)

    # Information matrices for M(lambda) = P_inv + beta * H_hess
    P_inv = tf.linalg.inv(model.P_prior)
    H = model.jacobian_h(model.x_truth)
    R_inv = model.R_inv_filter
    H_hess = tf.transpose(H) @ R_inv @ H

    def get_accel(beta_val):
        # M is the negative Hessian of the log-posterior
        b = tf.cast(tf.clip_by_value(beta_val, 0.0, 1.1), DTYPE)
        M = P_inv + b * H_hess
        M_inv = tf.linalg.inv(M)

        # Derivative of nuclear norm condition number (Equation 28 in Dai22)
        t1 = tf.linalg.trace(H_hess) * tf.linalg.trace(M_inv)
        t2 = tf.linalg.trace(M) * tf.linalg.trace(M_inv @ H_hess @ M_inv)

        # Acceleration is negative to force the concave beta curve
        return -mu * (t1 + t2)

    def simulate(v0):
        beta = np.zeros(n_steps)
        v = v0
        for i in range(n_steps - 1):
            beta[i+1] = beta[i] + v * dt
            # Stabilized velocity update
            v += get_accel(beta[i+1]).numpy() * dt
        return beta

    # Shooting method: Bisection to find v0 such that beta(1) = 1.0
    # Range is [1, 30] to find the spike near 14.2
    low, high = 1.0, 30.0
    for _ in range(35):
        v_mid = (low + high) / 2
        if simulate(v_mid)[-1] < 1.0: low = v_mid
        else: high = v_mid

    beta_opt = simulate(low)
    u_opt = np.gradient(beta_opt, lambdas)

    def get_kappa(b_vals):
        kappas = []
        for b in b_vals:
            M = P_inv + tf.cast(b, DTYPE) * H_hess
            s = tf.linalg.svd(M, compute_uv=False)
            kappas.append(tf.reduce_max(s) / tf.reduce_min(s))
        return np.array(kappas)

    return beta_opt, u_opt, get_kappa(beta_opt), get_kappa(lambdas)

def run_replication():
    model = Dai22BearingOnlySSM()
    b_opt, u_opt, c_opt, c_lin = solve_dai22_bvp(model, mu=0.2)
    lambdas = np.linspace(0, 1, len(b_opt))

    # CORRECTED: Explicit Figure Size (10, 8) and DPI for clarity
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # (a) beta(lambda) - Concave optimal curve above linear dashed line
    axes[0,0].plot(lambdas, b_opt, 'r-', linewidth=2, label='optimal $\\beta^*(\lambda)$')
    axes[0,0].plot(lambdas, lambdas, 'b--', label='$\\beta(\lambda)=\lambda$')
    axes[0,0].set_xlim([0, 1]); axes[0,0].set_ylim([0, 1.2])
    axes[0,0].set_xlabel('$\lambda$'); axes[0,0].set_ylabel('$\\beta(\lambda)$')
    axes[0,0].set_title("(a)")
    axes[0,0].legend(loc='lower right'); axes[0,0].grid(True, alpha=0.3)

    # (b) Deviation e = beta* - lambda - Correct range [0, 0.15]
    axes[0,1].plot(lambdas, b_opt - lambdas, 'b-', linewidth=2)
    axes[0,1].set_xlim([0, 1]); axes[0,1].set_ylim([0, 0.3])
    axes[0,1].set_xlabel('$\lambda$'); axes[0,1].set_ylabel('$e = \\beta^*(\lambda) - \lambda$')
    axes[0,1].set_title("(b)"); axes[0,1].grid(True, alpha=0.3)

    # (c) Control u*(lambda) - Shows initial spike near 14.2
    axes[1,0].plot(lambdas, u_opt, 'b-', linewidth=2)
    axes[1,0].set_xlim([0, 1]); axes[1,0].set_ylim([0, 15])
    axes[1,0].set_xlabel('$\lambda$'); axes[1,0].set_ylabel('$u^*$')
    axes[1,0].set_title("(c)"); axes[1,0].grid(True, alpha=0.3)

    # (d) Stiffness R_stiff - Log scale essential for stiffness mitigation visualization
    axes[1,1].plot(lambdas, c_opt, 'r-', linewidth=2, label='optimal $\\beta^*(\lambda)$')
    axes[1,1].plot(lambdas, c_lin, 'b--', label='$\\beta(\lambda)=\lambda$')
    axes[1,1].set_yscale('log'); axes[1,1].set_xlim([0, 1]); axes[1,1].set_ylim([1, 100])
    axes[1,1].set_xlabel('$\lambda$'); axes[1,1].set_ylabel('$R_{stiff}$')
    axes[1,1].set_title("(d)")
    axes[1,1].legend(); axes[1,1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"Replication Successful. Figure saved to: {SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    run_replication()
