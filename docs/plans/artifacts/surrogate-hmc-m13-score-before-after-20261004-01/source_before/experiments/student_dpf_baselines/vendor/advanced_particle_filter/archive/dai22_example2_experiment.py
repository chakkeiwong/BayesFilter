"""
Dai & Daum (2022) Example 2: Monte Carlo Replication Experiment.

Replicates Fig. 5 from the paper:
  - RMSE for position, velocity, acceleration vs. time
  - Comparison: optimal β*(λ) vs. baseline β(λ)=λ vs. PFSIR
  - Tests both μ=1e-5 (paper) and μ=1e-7 (our finding)

Usage:
    python dai22_example2_experiment.py

Output:
    - Console: per-run RMSE, average RMSE, timing
    - Plots: saved as PNG files
"""

import numpy as np
import time
import argparse
from numpy.random import default_rng

# ---- Adjust these imports to match your package structure ----
# If running from the repo root with filters/ and models/ as packages:
from advanced_particle_filter.models.dai22_example2 import make_dai22_example2_ssm, simulate_dai22_example2
from advanced_particle_filter.filters.stochastic_pff import StochasticPFFlow
from advanced_particle_filter.filters.stochastic_pff_stable import StochasticPFFlowStable
from advanced_particle_filter.filters.particle import BootstrapParticleFilter
from advanced_particle_filter.filters.base import FilterResult


def compute_component_rmse(means: np.ndarray, true_states: np.ndarray) -> dict:
    """
    Compute RMSE for position, velocity, acceleration components separately.

    Args:
        means: [T+1, 9] filtered means
        true_states: [T+1, 9] truth

    Returns:
        dict with keys 'position', 'velocity', 'acceleration',
        each [T+1] array of RMSE vs time
    """
    pos_err = means[:, 0:3] - true_states[:, 0:3]
    vel_err = means[:, 3:6] - true_states[:, 3:6]
    acc_err = means[:, 6:9] - true_states[:, 6:9]

    return {
        "position": np.sqrt(np.mean(pos_err**2, axis=1)),
        "velocity": np.sqrt(np.mean(vel_err**2, axis=1)),
        "acceleration": np.sqrt(np.mean(acc_err**2, axis=1)),
    }


def run_single_mc(
    mc_idx: int,
    model,
    true_states: np.ndarray,
    observations: np.ndarray,
    configs: dict,
    base_seed: int = 1000,
) -> dict:
    """
    Run all filter configurations for a single MC trial.

    Uses Common Random Numbers (CRN): same Brownian motion samples
    across configurations sharing the same base seed.

    Args:
        mc_idx: MC run index
        model: StateSpaceModel
        true_states: [T+1, 9]
        observations: [T, 2]
        configs: dict of {name: filter_instance}
        base_seed: base seed for CRN

    Returns:
        dict of {name: {'result': FilterResult, 'rmse': dict, 'time': float}}
    """
    results = {}

    for name, filt in configs.items():
        # CRN: same seed for all particle flow filters within one MC run
        # Different seed for PFSIR (it has its own randomness structure)
        rng = default_rng(base_seed + mc_idx)

        t0 = time.time()
        result = filt.filter(model, observations, rng=rng)
        elapsed = time.time() - t0

        rmse = compute_component_rmse(result.means, true_states)

        results[name] = {
            "result": result,
            "rmse": rmse,
            "time": elapsed,
        }

        avg_pos = np.mean(rmse["position"][1:])  # skip t=0
        avg_vel = np.mean(rmse["velocity"][1:])
        avg_acc = np.mean(rmse["acceleration"][1:])
        print(
            f"  MC {mc_idx+1:2d} | {name:30s} | "
            f"pos={avg_pos:8.3f}  vel={avg_vel:8.3f}  acc={avg_acc:8.3f} | "
            f"{elapsed:6.2f}s"
        )

    return results


def run_experiment(
    n_mc: int = 20,
    n_particles_pff: int = 100,
    n_particles_pfsir: int = 10000,
    n_flow_steps: int = 29,
    T: int = 20,
    data_seed: int = 42,
    mc_base_seed: int = 1000,
    run_pfsir: bool = False,
    mu_values: list = None,
):
    """
    Run the full MC experiment.

    Args:
        n_mc: Number of MC runs (paper: 20)
        n_particles_pff: Particles for PFF (paper: 100)
        n_particles_pfsir: Particles for PFSIR (paper: 10000)
        n_flow_steps: Number of E-M flow steps (29)
        T: Number of time steps (20)
        data_seed: Seed for generating truth/observations
        mc_base_seed: Base seed for MC filter runs
        run_pfsir: Whether to include PFSIR comparison
        mu_values: List of μ values to test (default: [1e-5, 1e-7])
    """
    if mu_values is None:
        mu_values = [1e-5, 1e-7]

    # ---- Create model ----
    model = make_dai22_example2_ssm(dt=1, epsilon=1e-2, R_diag=1e-6)

    print("=" * 80)
    print("Dai & Daum (2022) Example 2 Replication")
    print("=" * 80)
    print(f"State dim: {model.state_dim}, Obs dim: {model.obs_dim}")
    print(f"Truth s0: {model.truth_initial_state}")
    print(f"Prior s0: {model.initial_mean}")
    print(f"R = {model.obs_cov[0,0]:.1e} I_2")
    print(f"Phi (1,1) = {model.Phi[0,0]:.6f}")
    print(f"MC runs: {n_mc}, PFF particles: {n_particles_pff}, "
          f"PFSIR particles: {n_particles_pfsir}")
    print(f"Flow steps: {n_flow_steps}, Time steps: {T}")
    print(f"mu values to test: {mu_values}")
    print("=" * 80)

    # ---- Generate truth + observations (shared across all MC runs) ----
    # Each MC run uses the SAME truth trajectory and observations
    # (as in the paper: randomness is only in the filter, not the data)
    data_rng = default_rng(data_seed)
    true_states, observations = simulate_dai22_example2(
        model, T=T, rng=data_rng
    )

    print(f"\nTrue trajectory range:")
    print(f"  Position: x=[{true_states[:,0].min():.1f}, {true_states[:,0].max():.1f}], "
          f"y=[{true_states[:,1].min():.1f}, {true_states[:,1].max():.1f}], "
          f"z=[{true_states[:,2].min():.1f}, {true_states[:,2].max():.1f}]")
    print(f"  Observations (sample): z[0] = {observations[0]}")
    print()

    # ---- Build filter configurations ----
    configs = {}

    # Baseline: β(λ) = λ (linear schedule)
    configs["SPFF (beta=lambda)"] = StochasticPFFlow(
        n_particles=n_particles_pff,
        n_flow_steps=n_flow_steps,
        beta_schedule="linear",
        Q_flow_mode="adaptive",
        deterministic_dynamics=True,
        integration_method='semi_implicit'
    )

    # Optimal β*(λ) for each μ value
    for mu_val in mu_values:
        name = f"SPFF (optimal, mu={mu_val:.0e})"
        configs[name] = StochasticPFFlow(
            n_particles=n_particles_pff,
            n_flow_steps=n_flow_steps,
            beta_schedule="optimal",
            mu=mu_val,
            Q_flow_mode="adaptive",
            deterministic_dynamics=True,
            bvp_method="Radau",
            integration_method='semi_implicit'
        )

    # PFSIR baseline
    if run_pfsir:
        configs["PFSIR (N=10000)"] = BootstrapParticleFilter(
            n_particles=n_particles_pfsir,
            resample_method="systematic",
            resample_criterion="ess",
            ess_threshold=0.5,
        )

    # ---- MC loop ----
    all_results = {name: [] for name in configs}

    for mc in range(n_mc):
        print(f"\n--- MC Run {mc+1}/{n_mc} ---")
        mc_results = run_single_mc(
            mc, model, true_states, observations, configs,
            base_seed=mc_base_seed,
        )
        for name, res in mc_results.items():
            all_results[name].append(res)

    # ---- Aggregate results ----
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (averaged over MC runs)")
    print("=" * 80)

    summary = {}
    for name in configs:
        runs = all_results[name]
        n_runs = len(runs)

        # Stack RMSE arrays: [n_mc, T+1]
        pos_rmse = np.array([r["rmse"]["position"] for r in runs])
        vel_rmse = np.array([r["rmse"]["velocity"] for r in runs])
        acc_rmse = np.array([r["rmse"]["acceleration"] for r in runs])
        times = np.array([r["time"] for r in runs])

        # Average over MC runs
        avg_pos = np.mean(pos_rmse[:, 1:])  # skip t=0
        avg_vel = np.mean(vel_rmse[:, 1:])
        avg_acc = np.mean(acc_rmse[:, 1:])
        avg_time = np.mean(times)

        summary[name] = {
            "pos_rmse_vs_t": np.mean(pos_rmse, axis=0),
            "vel_rmse_vs_t": np.mean(vel_rmse, axis=0),
            "acc_rmse_vs_t": np.mean(acc_rmse, axis=0),
            "pos_rmse_std": np.std(pos_rmse, axis=0),
            "vel_rmse_std": np.std(vel_rmse, axis=0),
            "acc_rmse_std": np.std(acc_rmse, axis=0),
            "avg_pos": avg_pos,
            "avg_vel": avg_vel,
            "avg_acc": avg_acc,
            "avg_time": avg_time,
            "all_times": times,
        }

        print(f"\n{name}:")
        print(f"  Avg RMSE  — pos: {avg_pos:.4f}, vel: {avg_vel:.4f}, acc: {avg_acc:.4f}")
        print(f"  Avg time  — {avg_time:.2f}s (total: {np.sum(times):.1f}s)")

    # ---- Save results ----
    np.savez(
        "dai22_example2_results.npz",
        true_states=true_states,
        observations=observations,
        config_names=list(summary.keys()),
        **{
            f"{name}__pos_rmse": summary[name]["pos_rmse_vs_t"]
            for name in summary
        },
        **{
            f"{name}__vel_rmse": summary[name]["vel_rmse_vs_t"]
            for name in summary
        },
        **{
            f"{name}__acc_rmse": summary[name]["acc_rmse_vs_t"]
            for name in summary
        },
        **{
            f"{name}__avg_time": summary[name]["avg_time"]
            for name in summary
        },
    )
    print("\nResults saved to dai22_example2_results.npz")

    # ---- Plot ----
    try:
        plot_results(summary, T, true_states, observations)
    except ImportError:
        print("matplotlib not available; skipping plots.")

    return summary, true_states, observations


def plot_results(summary, T, true_states, observations):
    """Generate Fig. 5-style plots."""
    import matplotlib.pyplot as plt

    time_axis = np.arange(T + 1)

    # Color/style mapping
    style_map = {}
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    linestyles = ["-", "--", "-.", ":", "-"]
    for i, name in enumerate(summary):
        style_map[name] = {
            "color": colors[i % len(colors)],
            "linestyle": linestyles[i % len(linestyles)],
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Position RMSE
    ax = axes[0, 0]
    for name, s in summary.items():
        ax.semilogy(
            time_axis, s["pos_rmse_vs_t"],
            label=name, **style_map[name], linewidth=1.5,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("RMSE: Position")
    ax.set_title("(a) Position")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Velocity RMSE
    ax = axes[0, 1]
    for name, s in summary.items():
        ax.semilogy(
            time_axis, s["vel_rmse_vs_t"],
            label=name, **style_map[name], linewidth=1.5,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("RMSE: Velocity")
    ax.set_title("(b) Velocity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Acceleration RMSE
    ax = axes[1, 0]
    for name, s in summary.items():
        ax.semilogy(
            time_axis, s["acc_rmse_vs_t"],
            label=name, **style_map[name], linewidth=1.5,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("RMSE: Acceleration")
    ax.set_title("(c) Acceleration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Computing time per MC run
    ax = axes[1, 1]
    names = list(summary.keys())
    avg_times = [summary[n]["avg_time"] for n in names]
    # Shorten labels for bar chart
    short_names = []
    for n in names:
        if "optimal" in n:
            short_names.append(n.replace("SPFF (optimal, ", "opt ").rstrip(")"))
        elif "beta=lambda" in n:
            short_names.append("β=λ")
        elif "PFSIR" in n:
            short_names.append("PFSIR")
        else:
            short_names.append(n[:20])

    bars = ax.bar(short_names, avg_times, color=[style_map[n]["color"] for n in names])
    ax.set_ylabel("Avg time per MC run (s)")
    ax.set_title("(d) Computing Time")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("dai22_example2_fig5.png", dpi=150, bbox_inches="tight")
    print("Plot saved to dai22_example2_fig5.png")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dai (2022) Example 2 MC Experiment"
    )
    parser.add_argument("--n_mc", type=int, default=20, help="Number of MC runs")
    parser.add_argument("--n_particles", type=int, default=100, help="PFF particles")
    parser.add_argument("--n_pfsir", type=int, default=10000, help="PFSIR particles")
    parser.add_argument("--n_flow_steps", type=int, default=29, help="E-M flow steps")
    parser.add_argument("--T", type=int, default=20, help="Time steps")
    parser.add_argument("--no_pfsir", action="store_true", help="Skip PFSIR")
    parser.add_argument("--mu", type=float, nargs="+", default=[1e-5, 1e-7],
                        help="mu values to test")
    parser.add_argument("--data_seed", type=int, default=42, help="Data generation seed")
    parser.add_argument("--mc_seed", type=int, default=1000, help="MC base seed")

    args = parser.parse_args()

    run_experiment(
        n_mc=args.n_mc,
        n_particles_pff=args.n_particles,
        n_particles_pfsir=args.n_pfsir,
        n_flow_steps=args.n_flow_steps,
        T=args.T,
        data_seed=args.data_seed,
        mc_base_seed=args.mc_seed,
        run_pfsir=not args.no_pfsir,
        mu_values=args.mu,
    )
