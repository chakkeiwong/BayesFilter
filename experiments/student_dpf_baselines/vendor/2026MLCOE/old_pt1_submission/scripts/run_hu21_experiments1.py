
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tensorflow as tf
import tensorflow_probability as tfp

# Import the specific filter from your unified library
from filters import KPFF



# =============================================================================
# 1. MODEL & DATA GENERATION (Lorenz 96)
# =============================================================================
def lorenz96_step(x, F=8.0, dt=0.01):
    """
    Vectorized L96 RK4 integration.
    The model follows Equation (24) in the paper.
    """
    k1 = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    x2 = x + k1 * dt / 2
    k2 = (np.roll(x2, -1) - np.roll(x2, 2)) * np.roll(x2, 1) - x2 + F
    x3 = x + k2 * dt / 2
    k3 = (np.roll(x3, -1) - np.roll(x3, 2)) * np.roll(x3, 1) - x3 + F
    x4 = x + k3 * dt
    k4 = (np.roll(x4, -1) - np.roll(x4, 2)) * np.roll(x4, 1) - x4 + F
    return x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)

def generate_experiment_data(Nx=1000, Np=20, seed=42):
    """
    Setup as described in Section 3 (Experimental Design).
    """
    np.random.seed(seed)

    # 1. Truth Spinup
    x_truth = np.full(Nx, 8.0)
    x_truth[0] += 0.01
    for _ in range(100):
        x_truth = lorenz96_step(x_truth, dt=0.05)

    # 2. Prior Ensemble Generation
    prior_ens = np.zeros((Np, Nx))
    for i in range(Np):
        prior_ens[i] = x_truth + np.random.normal(0, np.sqrt(2), Nx)

    # 3. Observations (Linear, every 4th variable)
    obs_every = 4
    obs_idx = np.arange(3, Nx, obs_every)
    Ny = len(obs_idx)
    R_var = 0.5**2  # Error std = 0.5
    y_obs = x_truth[obs_idx] + np.random.normal(0, np.sqrt(R_var), Ny)

    return x_truth, prior_ens, y_obs, obs_idx, R_var

# =============================================================================
# 2. LOCALIZED KALMAN FILTER (For Reference Contours)
# =============================================================================
def get_localized_covariance(P, Nx, radius=4.0):
    """
    Localization using Schur product as described in Eq (28-29).
    """
    ix, jx = np.indices((Nx, Nx))
    d = np.abs(ix - jx)
    d = np.minimum(d, Nx - d)
    rho = np.exp(-(d / radius)**2)
    return P * rho

def get_kf_posterior_stats(prior_ens, y_obs, obs_idx, R_var):
    Nx = prior_ens.shape[1]
    Ny = len(y_obs)

    # Prior Stats
    mu_prior = np.mean(prior_ens, axis=0)
    P_prior = np.cov(prior_ens, rowvar=False)

    # Localization (Crucial for correct contour size in high-dim)
    P_loc = get_localized_covariance(P_prior, Nx, radius=4.0)

    # Kalman Update
    H_P_HT = P_loc[np.ix_(obs_idx, obs_idx)]
    P_HT = P_loc[:, obs_idx]

    R = np.eye(Ny) * R_var
    S = H_P_HT + R
    K = np.linalg.solve(S.T, P_HT.T).T

    y_pred = mu_prior[obs_idx]
    mu_post = mu_prior + K @ (y_obs - y_pred)
    H_P = P_loc[obs_idx, :]
    P_post = P_loc - K @ H_P

    return mu_post, P_post

# =============================================================================
# 3. RUNNER & PLOTTING
# =============================================================================
# Setup
x_true, prior, y_obs, obs_idx, R_var = generate_experiment_data(Nx=1000, Np=20, seed=101)

# Using the imported KPFF class from filters.py
print("Running Matrix Kernel PFF (Fast convergence)...")
pff_mat = KPFF(prior, y_obs, obs_idx, R_var, 'matrix')
pff_mat.update(n_steps=60, dt=0.01)

print("Running Scalar Kernel PFF (Slow convergence - running longer)...")
pff_sca = KPFF(prior, y_obs, obs_idx, R_var, 'scalar')
pff_sca.update(n_steps=800, dt=0.01) # Increased steps to force collapse

# Compute Localized KF Contours for reference
mu_kf, P_kf = get_kf_posterior_stats(prior, y_obs, obs_idx, R_var)

idx_unobs, idx_obs = 18, 19 # x19, x20

def draw_covariance_contours(ax, mean, cov, idx_x, idx_y, stds):
    cov_2d = np.array([[cov[idx_x, idx_x], cov[idx_x, idx_y]],
                       [cov[idx_y, idx_x], cov[idx_y, idx_y]]])
    center = [mean[idx_x], mean[idx_y]]

    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    for std in stds:
        width, height = 2 * std * np.sqrt(vals)
        ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                      facecolor='none', edgecolor='cornflowerblue', linewidth=0.8, alpha=0.6)
        ax.add_patch(ell)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

def plot_subplot(ax, p_prior, p_post, title, box_text):
    # Contours 0.5 to 8 std
    contour_levels = np.arange(0.5, 8.5, 0.75)
    draw_covariance_contours(ax, mu_kf, P_kf, idx_unobs, idx_obs, stds=contour_levels)

    ax.scatter(p_prior[:, idx_unobs], p_prior[:, idx_obs], facecolors='none', edgecolors='k', label='Prior')
    ax.scatter(p_post[:, idx_unobs], p_post[:, idx_obs], c='r', label='Posterior')

    obs_val = y_obs[list(obs_idx).index(idx_obs)]
    ax.axhline(obs_val, color='blue', linestyle='--', alpha=0.4, label='Observation')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f'Unobserved Variable (x{idx_unobs+1})')
    ax.set_ylabel(f'Observed Variable (x{idx_obs+1})')
    ax.text(0.05, 0.95, box_text, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-2, 12)

plot_subplot(axes[0], prior, pff_mat.X, "(a) Matrix-Valued Kernel", "Spread Maintained")
plot_subplot(axes[1], prior, pff_sca.X, "(b) Scalar Kernel", "Collapse in Observed Dim")

axes[1].legend(loc='upper right')
plt.suptitle("PFF Posterior Update: 1000-Dimensional Lorenz 96\n(Replicating Hu & van Leeuwen 2021, Fig 3)", fontsize=14)
plt.tight_layout()
plt.show()
