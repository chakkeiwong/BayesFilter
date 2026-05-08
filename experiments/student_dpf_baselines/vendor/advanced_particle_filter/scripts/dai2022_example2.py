"""
Dai & Daum (2022) Example 2: Stochastic Particle Flow Filter experiment.

3D angle-only target tracking with 9D state, 2 angle measurements.

Key implementation details (discovered through extensive experimentation):
  1. Split-step matrix exponential integration ("expm") for the flow SDE.
     Under assumption (A1), f = Fx + b is linear in x. The exact drift
     solution via expm(F·dλ) handles stiffness perfectly — no step-size
     sensitivity. Diffusion is added separately (additive noise).
  2. P propagates through dynamics without Kalman update. The flow itself
     performs the Bayesian update on particles; P only serves as the prior
     covariance for computing the flow parameters (S, Q, K1, K2).
  3. Paper's original P0 with velocity variance = 1e4 is used. This works
     with propagate-only P because P grows naturally through the dynamics
     coupling (acceleration → velocity → position), staying well-conditioned.

Reference parameters from Dai & Daum (2022):
  - N = 100 particles (flow), 10000 (PFSIR)
  - 20 MC runs, 20 time steps at dt = 1.0
  - Q_flow = M^{-1} = (-∇²log p)^{-1}  (adaptive, Gromov formula)
  - R = 1e-6 I₂, μ = 1e-5, ε = 1e-2
  - s0_truth = [40,40,40,8,0,-3,0,0,0]
  - s0_prior = [50,50,10,10,40,0,0,0,0]
  - P0 = diag(10,10,10,1e4,1e4,1e4,10,10,10)

Usage:
    python dai22_ex2_experiment.py

This is the correct experiment.
"""
import sys, os
import numpy as np
import time
from numpy.random import default_rng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_particle_filter.models.dai22_example2 import make_dai22_example2_ssm
from advanced_particle_filter.filters.stochastic_pff import StochasticPFFlow


# ============================================================================
# Configuration
# ============================================================================
N_PARTICLES   = 100         # particles for SPFF (paper: 100)
N_PFSIR       = 10000       # particles for PFSIR baseline (paper: 10000)
N_FLOW_STEPS  = 29          # lambda steps (expm is step-insensitive)
FLOW_RATIO    = 1.2         # geometric ratio for step sizes
DT_OBS        = 1.0         # measurement interval (seconds)
T_OBS         = 20          # number of measurement steps (paper: 20)
N_MC          = 20          # Monte-Carlo runs (paper: 20)
SEED_DATA     = 3          # seed for truth / observations
SEED_FILTER_0 = 1000        # base seed for filter runs

# Filter settings
VEL_VAR       = 1e4        # velocity variance in P0 (paper: 1e4)
BETA_SCHEDULE = "linear"    # "linear" (β=λ) or "optimal" (BVP)
Q_FLOW_MODE   = "adaptive"  # "adaptive" (Q = M^{-1}) or "fixed"
INTEGRATION   = "expm"      # "expm" (recommended), "euler", "semi_implicit", "heun"
MU_BVP        = 1e-5        # μ for optimal β BVP (paper: 1e-5)


# ============================================================================
# Build model
# ============================================================================
model = make_dai22_example2_ssm(dt=DT_OBS, epsilon=0.1)

# Override P0 if needed (paper uses 1e4 for velocity)
P0 = np.diag([10., 10., 10., VEL_VAR, VEL_VAR, VEL_VAR, 10., 10., 10.])
model.initial_cov = P0


# ============================================================================
# Dynamics — exact matrix exponential (linear system, no substeps needed)
# ============================================================================
from scipy.linalg import expm

nx = model.state_dim
A_cont = model.A_cont
Phi_obs = expm(A_cont * DT_OBS)  # exact transition over dt_obs


def propagate(particles):
    """Propagate particles by one measurement interval (exact)."""
    return particles @ Phi_obs.T


def propagate_truth(state):
    """Propagate a single truth state by one measurement interval (exact)."""
    return Phi_obs @ state


# ============================================================================
# Generate truth & observations
# ============================================================================
data_rng = default_rng(SEED_DATA)
R_chol = np.linalg.cholesky(model.obs_cov + 1e-12 * np.eye(model.obs_dim))

s0_truth = np.array([40., 40., 40., 8., 0., -3., 0., 0., 0.])

true_states = np.zeros((T_OBS + 1, nx))
true_states[0] = s0_truth
observations = np.zeros((T_OBS, model.obs_dim))

for t in range(T_OBS):
    true_states[t + 1] = propagate_truth(true_states[t])
    h_true = model.obs_mean(true_states[t + 1][np.newaxis, :])[0]
    observations[t] = h_true + R_chol @ data_rng.standard_normal(model.obs_dim)


# ============================================================================
# Print experiment settings
# ============================================================================
print("=" * 72)
print("Dai & Daum (2022) Example 2 — Stochastic Particle Flow Filter")
print("=" * 72)
print(f"State dim: {nx}, Obs dim: {model.obs_dim}")
print(f"Truth s0: {s0_truth}")
print(f"Prior s0: {model.initial_mean}")
print(f"P0 diag:  {np.diag(P0)}")
print(f"R = {model.obs_cov[0,0]:.1e} I_{model.obs_dim}")
print(f"N_particles={N_PARTICLES}, N_PFSIR={N_PFSIR}, N_MC={N_MC}")
print(f"n_flow_steps={N_FLOW_STEPS}, integration={INTEGRATION}")
print(f"beta_schedule={BETA_SCHEDULE}, Q_flow={Q_FLOW_MODE}")
print(f"mu_bvp={MU_BVP}")
print()


# ============================================================================
# PFSIR baseline
# ============================================================================
def run_pfsir(seed):
    """Bootstrap particle filter with systematic resampling."""
    rng = default_rng(seed)
    P0_chol = np.linalg.cholesky(P0 + 1e-8 * np.eye(nx))
    R_inv = np.linalg.inv(model.obs_cov)
    particles = model.initial_mean + rng.standard_normal((N_PFSIR, nx)) @ P0_chol.T
    w = np.ones(N_PFSIR) / N_PFSIR
    means = np.zeros((T_OBS + 1, nx))
    means[0] = np.mean(particles, axis=0)

    for t in range(T_OBS):
        z = observations[t]
        particles = propagate(particles)

        # Log-likelihoods
        log_liks = np.zeros(N_PFSIR)
        for i in range(N_PFSIR):
            h_i = model.obs_mean(particles[i:i + 1, :])[0]
            resid = z - h_i
            log_liks[i] = -0.5 * resid @ R_inv @ resid
        log_w = log_liks - np.max(log_liks)
        w = np.exp(log_w)
        w /= w.sum()
        means[t + 1] = particles.T @ w

        # Systematic resampling
        ess = 1.0 / np.sum(w ** 2)
        if ess < N_PFSIR / 2:
            cumw = np.cumsum(w)
            u = (rng.random() + np.arange(N_PFSIR)) / N_PFSIR
            idx = np.searchsorted(cumw, u)
            idx = np.clip(idx, 0, N_PFSIR - 1)
            particles = particles[idx]
            w = np.ones(N_PFSIR) / N_PFSIR

    return means


# ============================================================================
# SPFF with manual multi-rate loop
# ============================================================================
def run_spff(seed):
    """Run SPFF with multi-rate dynamics."""
    rng = default_rng(seed)

    spf = StochasticPFFlow(
        n_particles=N_PARTICLES,
        n_flow_steps=N_FLOW_STEPS,
        flow_step_ratio=FLOW_RATIO,
        Q_flow_mode=Q_FLOW_MODE,
        beta_schedule=BETA_SCHEDULE,
        integration_method=INTEGRATION,
        mu=MU_BVP,
        deterministic_dynamics=True,
        seed=seed,
    )

    result = spf.filter(model, observations, rng=rng)

    return result.means


# ============================================================================
# Monte-Carlo experiment
# ============================================================================
def compute_errors(means):
    """Compute per-step RMSE for position, velocity, acceleration."""
    pos = np.array([1/np.sqrt(3) * np.linalg.norm(means[t+1, :3] - true_states[t+1, :3]) for t in range(T_OBS)])
    vel = np.array([1/np.sqrt(3) * np.linalg.norm(means[t+1, 3:6] - true_states[t+1, 3:6]) for t in range(T_OBS)])
    acc = np.array([1/np.sqrt(3) * np.linalg.norm(means[t+1, 6:9] - true_states[t+1, 6:9]) for t in range(T_OBS)])
    return pos, vel, acc


if __name__ == "__main__":
    spf_pos = np.zeros((N_MC, T_OBS))
    spf_vel = np.zeros((N_MC, T_OBS))
    spf_acc = np.zeros((N_MC, T_OBS))
    pf_pos  = np.zeros((N_MC, T_OBS))
    pf_vel  = np.zeros((N_MC, T_OBS))
    pf_acc  = np.zeros((N_MC, T_OBS))
    spf_times = np.zeros(N_MC)
    pf_times  = np.zeros(N_MC)

    t_total = time.time()

    for mc in range(N_MC):
        seed = SEED_FILTER_0 + mc
        t0 = time.time()

        means_spf = run_spff(seed)
        t_spf = time.time() - t0
        spf_times[mc] = t_spf

        t0 = time.time()
        means_pf = run_pfsir(seed)
        t_pf = time.time() - t0
        pf_times[mc] = t_pf

        spf_pos[mc], spf_vel[mc], spf_acc[mc] = compute_errors(means_spf)
        pf_pos[mc], pf_vel[mc], pf_acc[mc] = compute_errors(means_pf)

        if (mc + 1) % 5 == 0 or mc == 0:
            rp = np.sqrt(np.mean(spf_pos[mc] ** 2))
            rv = np.sqrt(np.mean(spf_vel[mc] ** 2))
            print(f"  MC {mc+1:3d}/{N_MC}: SPFF pos={rp:8.2f} vel={rv:7.2f} "
                  f"({t_spf:.1f}s) | PFSIR ({t_pf:.1f}s)")

    total_time = time.time() - t_total

    # RMSE across MC runs (per time step)
    rmse_spf_pos = np.sqrt(np.mean(spf_pos ** 2, axis=0))
    rmse_spf_vel = np.sqrt(np.mean(spf_vel ** 2, axis=0))
    rmse_spf_acc = np.sqrt(np.mean(spf_acc ** 2, axis=0))
    rmse_pf_pos  = np.sqrt(np.mean(pf_pos ** 2, axis=0))
    rmse_pf_vel  = np.sqrt(np.mean(pf_vel ** 2, axis=0))
    rmse_pf_acc  = np.sqrt(np.mean(pf_acc ** 2, axis=0))

    print(f"\n{'=' * 72}")
    print(f"RESULTS — {N_MC} MC runs, {total_time:.0f}s total")
    print(f"{'=' * 72}")
    print(f"{'t':>3s}  {'SPFF_pos':>10s} {'SPFF_vel':>10s} {'SPFF_acc':>10s}  "
          f"{'PFSIR_pos':>10s} {'PFSIR_vel':>10s} {'PFSIR_acc':>10s}")
    print("-" * 72)
    for t in range(T_OBS):
        print(f"{t+1:3d}  {rmse_spf_pos[t]:10.2f} {rmse_spf_vel[t]:10.2f} {rmse_spf_acc[t]:10.4f}  "
              f"{rmse_pf_pos[t]:10.2f} {rmse_pf_vel[t]:10.2f} {rmse_pf_acc[t]:10.4f}")

    print(f"\nOverall time-averaged RMSE:")
    print(f"  SPFF:  pos={np.sqrt(np.mean(rmse_spf_pos**2)):8.2f}  "
          f"vel={np.sqrt(np.mean(rmse_spf_vel**2)):8.2f}  "
          f"acc={np.sqrt(np.mean(rmse_spf_acc**2)):8.4f}")
    print(f"  PFSIR: pos={np.sqrt(np.mean(rmse_pf_pos**2)):8.2f}  "
          f"vel={np.sqrt(np.mean(rmse_pf_vel**2)):8.2f}  "
          f"acc={np.sqrt(np.mean(rmse_pf_acc**2)):8.4f}")

    # ====================================================================
    # Plot — matching Dai & Daum (2022) Fig. 5 layout
    # ====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_axis = np.arange(1, T_OBS + 1)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(
            r"Comparison of BPF with"
            r"$\beta(\lambda)=\lambda$ for Example 2",
            fontsize=12,
        )

        # ---- (a) Position RMSE ----
        ax = axes[0, 0]
        ax.semilogy(time_axis, rmse_spf_pos, "b-o", markersize=4,
                     label=r"$\beta(\lambda)=\lambda$")
        ax.semilogy(time_axis, rmse_pf_pos, "k--s", markersize=4,
                     label="PFSIR")
        ax.set_xlabel("time")
        ax.set_ylabel("RMSE: Position")
        ax.legend(fontsize=8)
        ax.set_title("(a)")
        ax.grid(True, which="both", alpha=0.3)

        # ---- (b) Velocity RMSE ----
        ax = axes[0, 1]
        ax.semilogy(time_axis, rmse_spf_vel, "b-o", markersize=4,
                     label=r"$\beta(\lambda)=\lambda$")
        ax.semilogy(time_axis, rmse_pf_vel, "k--s", markersize=4,
                     label="PFSIR")
        ax.set_xlabel("time")
        ax.set_ylabel("RMSE: Velocity")
        ax.legend(fontsize=8)
        ax.set_title("(b)")
        ax.grid(True, which="both", alpha=0.3)

        # ---- (c) Acceleration RMSE ----
        ax = axes[1, 0]
        ax.semilogy(time_axis, rmse_spf_acc, "b-o", markersize=4,
                     label=r"$\beta(\lambda)=\lambda$")
        ax.semilogy(time_axis, rmse_pf_acc, "k--s", markersize=4,
                     label="PFSIR")
        ax.set_xlabel("time")
        ax.set_ylabel("RMSE: Acceleration")
        ax.legend(fontsize=8)
        ax.set_title("(c)")
        ax.grid(True, which="both", alpha=0.3)

        # ---- (d) Computing time ----
        ax = axes[1, 1]
        ax.plot(np.arange(1, N_MC + 1), spf_times, "b-o", markersize=4,
                label=r"$\beta(\lambda)=\lambda$")
        ax.plot(np.arange(1, N_MC + 1), pf_times, "k--s", markersize=4,
                label="PFSIR")
        ax.set_xlabel("Monte Carlo run index")
        ax.set_ylabel("computing time (seconds)")
        ax.legend(fontsize=8)
        ax.set_title("(d)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig("dai22_ex2_fig5.png", dpi=150, bbox_inches="tight")
        print(f"\nPlots saved: dai22_ex2_fig5.png / .pdf")
        plt.close(fig)

    except ImportError:
        print("\nmatplotlib not available — skipping plots.")
