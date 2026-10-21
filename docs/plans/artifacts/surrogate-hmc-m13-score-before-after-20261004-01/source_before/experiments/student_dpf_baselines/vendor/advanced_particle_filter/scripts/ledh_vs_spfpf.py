"""
Comparison: LEDH-PFPF vs SPF-PF on Range-Bearing Model.

Benchmarks:
  - Position RMSE over time
  - Velocity RMSE over time
  - ESS over time
  - Runtime per trajectory

Runs 5 independent trajectories with Student-t measurement noise.

Usage:
    python compare_ledh_vs_spfpf.py

"""

import time
import numpy as np
from numpy.random import default_rng

# ---------------------------------------------------------------------------
# Adjust this to point to the parent of your 'filters/' and 'models/' dirs.
# For example, if the layout is:
#   project/
#     filters/
#       __init__.py, ledh.py, stochastic_pff.py, stochastic_pfpf.py, ...
#     models/
#       __init__.py, base.py, range_bearing.py, ...
#
# Then PACKAGE_ROOT = "project"
# ---------------------------------------------------------------------------
# PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PACKAGE_ROOT not in sys.path:
#     sys.path.insert(0, PACKAGE_ROOT)

from advanced_particle_filter.filters.ledh import LEDHParticleFilter
from advanced_particle_filter.filters.stochastic_pfpf import StochasticPFParticleFilter
from advanced_particle_filter.models.range_bearing import make_range_bearing_ssm


# =============================================================================
# Configuration
# =============================================================================

N_TRAJECTORIES = 20
T_STEPS = 40          # Number of time steps per trajectory
BASE_SEED = 2026

# Shared particle filter settings
N_PARTICLES = 100
N_FLOW_STEPS = 29
RESAMPLE_METHOD = "systematic"
RESAMPLE_CRITERION = "ess"
ESS_THRESHOLD = 0.5

# Range-bearing model settings
MODEL_PARAMS = dict(
    dt=0.1,
    q_diag=0.01,
    nu=2.0,          # Student-t degrees of freedom (heavy tails)
    s_r=0.05,        # Range noise scale
    s_th=0.05,       # Bearing noise scale
    m0=(1.0, 0.5, 0.1, 0.1),
    P0_diag=(0.1, 0.1, 0.1, 0.1),
)


# =============================================================================
# Metric computation
# =============================================================================

def compute_pos_rmse(means: np.ndarray, true_states: np.ndarray) -> np.ndarray:
    """
    Position RMSE at each time step.

    Args:
        means: [T+1, nx] filter estimates
        true_states: [T+1, nx] true states

    Returns:
        rmse: [T+1] position RMSE (over px, py)
    """
    pos_err = means[:, :2] - true_states[:, :2]
    return np.sqrt(np.sum(pos_err**2, axis=1))


def compute_vel_rmse(means: np.ndarray, true_states: np.ndarray) -> np.ndarray:
    """
    Velocity RMSE at each time step.

    Args:
        means: [T+1, nx] filter estimates
        true_states: [T+1, nx] true states

    Returns:
        rmse: [T+1] velocity RMSE (over vx, vy)
    """
    vel_err = means[:, 2:4] - true_states[:, 2:4]
    return np.sqrt(np.sum(vel_err**2, axis=1))


# =============================================================================
# Run single filter on single trajectory
# =============================================================================

def run_filter(filter_obj, model, observations, seed):
    """
    Run a filter and return results + runtime.

    Returns:
        result: FilterResult
        elapsed: float (seconds)
    """
    rng = default_rng(seed)
    t0 = time.perf_counter()
    result = filter_obj.filter(model, observations, rng=rng)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# =============================================================================
# Main comparison
# =============================================================================

def main():
    print("=" * 70)
    print("  LEDH-PFPF  vs  SPF-PF  Comparison")
    print("  Range-Bearing Model with Student-t Noise")
    print("=" * 70)
    print()
    print(f"  Trajectories:   {N_TRAJECTORIES}")
    print(f"  Time steps:     {T_STEPS}")
    print(f"  Particles:      {N_PARTICLES}")
    print(f"  Flow steps:     {N_FLOW_STEPS}")
    print(f"  Model:          nu={MODEL_PARAMS['nu']}, "
          f"s_r={MODEL_PARAMS['s_r']}, s_th={MODEL_PARAMS['s_th']}")
    print()

    # --- Build filters ---
    ledh_pf = LEDHParticleFilter(
        n_particles=N_PARTICLES,
        n_flow_steps=N_FLOW_STEPS,
        flow_step_ratio=1.2,
        resample_method=RESAMPLE_METHOD,
        resample_criterion=RESAMPLE_CRITERION,
        ess_threshold=ESS_THRESHOLD,
        redraw=False,
    )

    spf_pf = StochasticPFParticleFilter(
        n_particles=N_PARTICLES,
        n_flow_steps=N_FLOW_STEPS,
        Q_flow_mode="adaptive",
        beta_schedule="linear",
        mu=1e-3,
        deterministic_dynamics=False,
        integration_method="expm",
        resample_method=RESAMPLE_METHOD,
        resample_criterion=RESAMPLE_CRITERION,
        ess_threshold=ESS_THRESHOLD,
    )

    # --- Storage ---
    # [n_traj, T+1]
    pos_rmse_ledh = np.zeros((N_TRAJECTORIES, T_STEPS + 1))
    pos_rmse_spf = np.zeros((N_TRAJECTORIES, T_STEPS + 1))
    vel_rmse_ledh = np.zeros((N_TRAJECTORIES, T_STEPS + 1))
    vel_rmse_spf = np.zeros((N_TRAJECTORIES, T_STEPS + 1))
    # [n_traj, T]
    ess_ledh = np.zeros((N_TRAJECTORIES, T_STEPS))
    ess_spf = np.zeros((N_TRAJECTORIES, T_STEPS))
    # [n_traj]
    runtime_ledh = np.zeros(N_TRAJECTORIES)
    runtime_spf = np.zeros(N_TRAJECTORIES)

    # --- Run ---
    for traj in range(N_TRAJECTORIES):
        print(f"--- Trajectory {traj + 1}/{N_TRAJECTORIES} ---")

        # Build model (fresh each time for reproducibility)
        model = make_range_bearing_ssm(**MODEL_PARAMS)

        # Simulate ground truth
        sim_rng = default_rng(BASE_SEED + traj * 1000)
        true_states, observations = model.simulate(T_STEPS, sim_rng)

        # --- LEDH-PFPF ---
        filter_seed = BASE_SEED + traj * 1000 + 1
        res_ledh, t_ledh = run_filter(
            ledh_pf, model, observations, filter_seed
        )
        pos_rmse_ledh[traj] = compute_pos_rmse(res_ledh.means, true_states)
        vel_rmse_ledh[traj] = compute_vel_rmse(res_ledh.means, true_states)
        ess_ledh[traj] = res_ledh.ess
        runtime_ledh[traj] = t_ledh
        print(f"  LEDH-PFPF:  pos RMSE(end)={pos_rmse_ledh[traj, -1]:.4f}  "
              f"mean ESS={res_ledh.ess.mean():.1f}/{N_PARTICLES}  "
              f"time={t_ledh:.2f}s")

        # --- SPF-PF ---
        res_spf, t_spf = run_filter(
            spf_pf, model, observations, filter_seed
        )
        pos_rmse_spf[traj] = compute_pos_rmse(res_spf.means, true_states)
        vel_rmse_spf[traj] = compute_vel_rmse(res_spf.means, true_states)
        ess_spf[traj] = res_spf.ess
        runtime_spf[traj] = t_spf
        print(f"  SPF-PF:     pos RMSE(end)={pos_rmse_spf[traj, -1]:.4f}  "
              f"mean ESS={res_spf.ess.mean():.1f}/{N_PARTICLES}  "
              f"time={t_spf:.2f}s")
        print()

    # =========================================================================
    # Summary table
    # =========================================================================
    print("=" * 70)
    print("  SUMMARY (mean ± std over trajectories)")
    print("=" * 70)

    def fmt(vals):
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

    # Average RMSE over all time steps, then stats over trajectories
    avg_pos_rmse_ledh = pos_rmse_ledh[:, 1:].mean(axis=1)  # skip t=0
    avg_pos_rmse_spf = pos_rmse_spf[:, 1:].mean(axis=1)
    avg_vel_rmse_ledh = vel_rmse_ledh[:, 1:].mean(axis=1)
    avg_vel_rmse_spf = vel_rmse_spf[:, 1:].mean(axis=1)
    avg_ess_ledh = ess_ledh.mean(axis=1)
    avg_ess_spf = ess_spf.mean(axis=1)

    header = f"{'Metric':<30} {'LEDH-PFPF':>22} {'SPF-PF':>22}"
    print(header)
    print("-" * len(header))
    print(f"{'Avg Pos RMSE':<30} {fmt(avg_pos_rmse_ledh):>22} {fmt(avg_pos_rmse_spf):>22}")
    print(f"{'Final Pos RMSE':<30} {fmt(pos_rmse_ledh[:, -1]):>22} {fmt(pos_rmse_spf[:, -1]):>22}")
    print(f"{'Avg Vel RMSE':<30} {fmt(avg_vel_rmse_ledh):>22} {fmt(avg_vel_rmse_spf):>22}")
    print(f"{'Final Vel RMSE':<30} {fmt(vel_rmse_ledh[:, -1]):>22} {fmt(vel_rmse_spf[:, -1]):>22}")
    print(f"{'Mean ESS':<30} {fmt(avg_ess_ledh):>22} {fmt(avg_ess_spf):>22}")
    print(f"{'Min ESS (across time)':<30} {fmt(ess_ledh.min(axis=1)):>22} {fmt(ess_spf.min(axis=1)):>22}")
    print(f"{'Runtime (s)':<30} {fmt(runtime_ledh):>22} {fmt(runtime_spf):>22}")
    print()

    # =========================================================================
    # Plot
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"LEDH-PFPF vs SPF-PF  |  Range-Bearing (ν={MODEL_PARAMS['nu']})  |  "
            f"{N_PARTICLES} particles, {N_FLOW_STEPS} flow steps, "
            f"{N_TRAJECTORIES} trajectories",
            fontsize=12, fontweight="bold",
        )

        time_axis = np.arange(T_STEPS + 1)
        time_axis_ess = np.arange(1, T_STEPS + 1)

        colors = {"ledh": "#2c5f8a", "spf": "#a0422a"}

        # --- (a) Position RMSE ---
        ax = axes[0, 0]
        mean_ledh = pos_rmse_ledh.mean(axis=0)
        std_ledh = pos_rmse_ledh.std(axis=0)
        mean_spf = pos_rmse_spf.mean(axis=0)
        std_spf = pos_rmse_spf.std(axis=0)

        ax.plot(time_axis, mean_ledh, color=colors["ledh"], lw=2, label="LEDH-PFPF")
        ax.fill_between(time_axis, mean_ledh - std_ledh, mean_ledh + std_ledh,
                         color=colors["ledh"], alpha=0.15)
        ax.plot(time_axis, mean_spf, color=colors["spf"], lw=2, label="SPF-PF")
        ax.fill_between(time_axis, mean_spf - std_spf, mean_spf + std_spf,
                         color=colors["spf"], alpha=0.15)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Position RMSE")
        ax.set_title("(a) Position RMSE")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- (b) Velocity RMSE ---
        ax = axes[0, 1]
        mean_ledh = vel_rmse_ledh.mean(axis=0)
        std_ledh = vel_rmse_ledh.std(axis=0)
        mean_spf = vel_rmse_spf.mean(axis=0)
        std_spf = vel_rmse_spf.std(axis=0)

        ax.plot(time_axis, mean_ledh, color=colors["ledh"], lw=2, label="LEDH-PFPF")
        ax.fill_between(time_axis, mean_ledh - std_ledh, mean_ledh + std_ledh,
                         color=colors["ledh"], alpha=0.15)
        ax.plot(time_axis, mean_spf, color=colors["spf"], lw=2, label="SPF-PF")
        ax.fill_between(time_axis, mean_spf - std_spf, mean_spf + std_spf,
                         color=colors["spf"], alpha=0.15)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Velocity RMSE")
        ax.set_title("(b) Velocity RMSE")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- (c) ESS ---
        ax = axes[1, 0]
        mean_ledh = ess_ledh.mean(axis=0)
        std_ledh = ess_ledh.std(axis=0)
        mean_spf = ess_spf.mean(axis=0)
        std_spf = ess_spf.std(axis=0)

        ax.plot(time_axis_ess, mean_ledh, color=colors["ledh"], lw=2, label="LEDH-PFPF")
        ax.fill_between(time_axis_ess, mean_ledh - std_ledh, mean_ledh + std_ledh,
                         color=colors["ledh"], alpha=0.15)
        ax.plot(time_axis_ess, mean_spf, color=colors["spf"], lw=2, label="SPF-PF")
        ax.fill_between(time_axis_ess, mean_spf - std_spf, mean_spf + std_spf,
                         color=colors["spf"], alpha=0.15)
        ax.axhline(ESS_THRESHOLD * N_PARTICLES, color="gray", ls="--", lw=1,
                    label=f"Resample threshold ({ESS_THRESHOLD * N_PARTICLES:.0f})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("ESS")
        ax.set_title("(c) Effective Sample Size")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # --- (d) Runtime bar chart ---
        ax = axes[1, 1]
        methods = ["LEDH-PFPF", "SPF-PF"]
        means_rt = [runtime_ledh.mean(), runtime_spf.mean()]
        stds_rt = [runtime_ledh.std(), runtime_spf.std()]
        bars = ax.bar(methods, means_rt, yerr=stds_rt, capsize=6,
                       color=[colors["ledh"], colors["spf"]], alpha=0.8,
                       edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("(d) Runtime per Trajectory")
        ax.grid(True, alpha=0.3, axis="y")
        # Add value labels on bars
        for bar, m, s in zip(bars, means_rt, stds_rt):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.1,
                    f"{m:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = "ledh_vs_spfpf_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {out_path}")
        plt.close()

    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    # =========================================================================
    # Per-trajectory detail table
    # =========================================================================
    print()
    print("=" * 70)
    print("  PER-TRAJECTORY DETAIL")
    print("=" * 70)
    header = (f"{'Traj':>4}  "
              f"{'LEDH pos':>10} {'SPF pos':>10}  "
              f"{'LEDH vel':>10} {'SPF vel':>10}  "
              f"{'LEDH ESS':>9} {'SPF ESS':>9}  "
              f"{'LEDH t(s)':>9} {'SPF t(s)':>9}")
    print(header)
    print("-" * len(header))
    for i in range(N_TRAJECTORIES):
        print(
            f"{i+1:>4}  "
            f"{avg_pos_rmse_ledh[i]:>10.4f} {avg_pos_rmse_spf[i]:>10.4f}  "
            f"{avg_vel_rmse_ledh[i]:>10.4f} {avg_vel_rmse_spf[i]:>10.4f}  "
            f"{avg_ess_ledh[i]:>9.1f} {avg_ess_spf[i]:>9.1f}  "
            f"{runtime_ledh[i]:>9.2f} {runtime_spf[i]:>9.2f}"
        )
    print()


if __name__ == "__main__":
    main()

# ======================================================================
#   PER-TRAJECTORY DETAIL
# ======================================================================
# Traj    LEDH pos    SPF pos    LEDH vel    SPF vel   LEDH ESS   SPF ESS  LEDH t(s)  SPF t(s)
# --------------------------------------------------------------------------------------------
#    1      0.1448     0.2625      0.5562     0.8646       17.1      19.5       4.07      0.42
#    2      0.1388     0.1177      0.4971     0.3401       18.2      26.7       4.05      0.41
#    3      0.1189     0.1083      0.7677     0.7434       15.4      22.2       4.03      0.41
#    4      0.1568     0.1345      0.3459     0.4665       18.2      25.8       4.04      0.41
#    5      0.2125     0.1401      0.3620     0.4085       18.0      30.8       4.05      0.41
#    6      0.1823     0.1248      0.4837     0.3315       15.5      25.4       4.06      0.42
#    7      0.2011     0.1512      0.6535     0.5669       18.7      28.1       4.13      0.42
#    8      0.1458     0.1452      0.5649     0.4579       17.6      24.5       4.13      0.41
#    9      0.1400     0.1257      0.5174     0.4723       11.9      20.9       4.10      0.42
#   10      0.1539     0.1425      0.5827     0.7413       15.4      27.2       4.03      0.41
#   11      0.0900     0.1380      0.3343     0.8505       19.9      14.3       4.06      0.43
#   12      0.0830     0.0915      0.7130     0.5609       12.7       8.9       4.00      0.41
#   13      0.0865     0.1754      0.4979     0.8737       18.8      19.0       4.00      0.41
#   14      0.0935     0.0958      0.3900     0.3954       15.2      18.8       4.00      0.41
#   15      0.1851     0.1216      0.3739     0.3492       17.4      28.6       4.06      0.41
#   16      0.1521     0.1559      0.8064     0.3479       12.4      21.7       4.11      0.42
#   17      0.2409     0.2178      0.5728     0.7986       15.8      23.1       4.05      0.42
#   18      0.1262     0.1345      0.4423     0.4639       22.4      22.6       4.01      0.41
#   19      0.0791     0.0982      0.5219     0.4315       16.7      14.9       4.01      0.41
#   20      0.1269     0.1232      0.4379     0.5060       20.4      23.3       4.01      0.41