"""
Acoustic Tracking Experiment - Matching Li & Coates (2017) MATLAB implementation.

Key settings from MATLAB code:
1. Filter initialization: Random m0 ~ N(x0, sigma0^2), particles ~ N(m0, P0)
2. Redraw strategy: Particles redrawn from N(mean, P_updated) at each step
3. Q_real for simulation, Q_filter for filtering
4. T=40 time steps, nTrack=100 trajectories, nAlg_per_track=5 runs
"""

import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import csv
import os
from datetime import datetime

from advanced_particle_filter.models import make_acoustic_ssm, make_acoustic_Q_filter
from advanced_particle_filter.models.base import StateSpaceModel
from advanced_particle_filter.filters import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    BootstrapParticleFilter,
    EDHFlow,
    EDHParticleFilter,
    LEDHFlow,
    LEDHParticleFilter,
)
from advanced_particle_filter.utils import compute_omat


# =============================================================================
# Simulation Parameters (matching MATLAB)
# =============================================================================

# From Acoustic_example_initialization.m
X0_TRUE = np.array([12, 6, 0.001, 0.001,    # Target 1
                    32, 32, -0.001, -0.005,  # Target 2
                    20, 13, -0.1, 0.01,      # Target 3
                    15, 35, 0.002, 0.002])   # Target 4

# sigma0 = 10*[1;1;0.1;0.1] per target
SIGMA0_PER_TARGET = np.array([10.0, 10.0, 1.0, 1.0])

# Surveillance region
SURV_REGION = [0, 0, 40, 40]  # [x_min, y_min, x_max, y_max]


def make_Q_real(num_targets: int = 4) -> np.ndarray:
    """Create Q_real used for trajectory simulation (small noise)."""
    gammavar_real = 0.05
    Gamma_single = np.array([
        [1/3, 0, 0.5, 0],
        [0, 1/3, 0, 0.5],
        [0.5, 0, 1, 0],
        [0, 0.5, 0, 1],
    ])
    Q_single = gammavar_real * Gamma_single
    return np.kron(np.eye(num_targets), Q_single)


def sample_initial_filter_mean(
    x0: np.ndarray,
    sigma0: np.ndarray,
    surv_region: List[float],
    rng: np.random.Generator,
    num_targets: int = 4,
) -> np.ndarray:
    """
    Sample random initial mean for filter (matching AcousticGaussInit.m).
    
    Samples m0 ~ N(x0, diag(sigma0^2)) and rejects if any target is out of bounds.
    """
    dim = len(x0)
    
    while True:
        m0 = x0 + sigma0 * rng.standard_normal(dim)
        
        # Check bounds
        out_of_bound = False
        for i in range(num_targets):
            x_pos = m0[4*i]
            y_pos = m0[4*i + 1]
            if x_pos < surv_region[0] or x_pos > surv_region[2]:
                out_of_bound = True
                break
            if y_pos < surv_region[1] or y_pos > surv_region[3]:
                out_of_bound = True
                break
        
        if not out_of_bound:
            return m0


def generate_trajectory(
    model: StateSpaceModel,
    x0: np.ndarray,
    T: int,
    rng: np.random.Generator,
    num_targets: int = 4,
    surv_region: List[float] = None,
) -> np.ndarray:
    """
    Generate trajectory, rejecting if any target leaves surveillance region.
    
    Matches GenerateTracks.m behavior.
    """
    if surv_region is None:
        surv_region = SURV_REGION
    
    nx = len(x0)
    sim_area_size = surv_region[2]  # Assuming square region
    
    while True:
        states = np.zeros((T + 1, nx))
        states[0] = model.sample_dynamics(x0[np.newaxis, :], rng)[0]
        
        for t in range(1, T + 1):
            states[t] = model.sample_dynamics(states[t-1:t], rng)[0]
        
        # Check bounds (5% to 95% of area)
        out_of_bounds = False
        for i in range(num_targets):
            x_pos = states[:, 4*i]
            y_pos = states[:, 4*i + 1]
            
            if np.any(x_pos < 0.05 * sim_area_size) or np.any(x_pos > 0.95 * sim_area_size):
                out_of_bounds = True
                break
            if np.any(y_pos < 0.05 * sim_area_size) or np.any(y_pos > 0.95 * sim_area_size):
                out_of_bounds = True
                break
        
        if not out_of_bounds:
            return states


def generate_observations(
    model: StateSpaceModel,
    states: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate observations from states."""
    T = states.shape[0] - 1
    ny = model.obs_dim
    
    observations = np.zeros((T, ny))
    for t in range(T):
        y_mean = model.obs_mean(states[t + 1:t + 2])[0]
        observations[t] = y_mean + rng.multivariate_normal(np.zeros(ny), model.obs_cov)
    
    return observations


def create_filter_model(
    base_model: StateSpaceModel,
    filter_m0: np.ndarray,
    P0_diag: np.ndarray,
    Q_filter: np.ndarray,
    num_targets: int = 4,
) -> StateSpaceModel:
    """
    Create a model for filtering with random initial mean and Q_filter.
    
    IMPORTANT: The filter model must use Q_filter (not Q_real) for:
    - Particle propagation (sample_dynamics)
    - Weight calculation (dynamics_log_prob)
    
    This matches the MATLAB implementation where ps.propparams.Q = Q_filter.
    
    Args:
        base_model: Base model (used for observation functions)
        filter_m0: Random initial mean for filter
        P0_diag: Initial covariance diagonal
        Q_filter: Inflated Q for filtering
        num_targets: Number of targets
        
    Returns:
        StateSpaceModel configured for filtering
    """
    from copy import deepcopy
    
    filter_model = deepcopy(base_model)
    filter_model.initial_mean = filter_m0.copy()
    filter_model.initial_cov = np.diag(P0_diag)
    
    # CRITICAL: Set dynamics_cov to Q_filter (not Q_real)
    # This affects both particle propagation and weight calculation
    filter_model.dynamics_cov = Q_filter.copy()
    
    # Update Cholesky decomposition
    filter_model._dynamics_cov_chol = np.linalg.cholesky(
        Q_filter + 1e-6 * np.eye(Q_filter.shape[0])
    )
    
    return filter_model


@dataclass
class TrialResult:
    """Result of a single trial."""
    filter_name: str
    omat_per_step: np.ndarray
    omat_mean: float
    ess_per_step: Optional[np.ndarray]
    ess_mean: float
    runtime: float


def run_single_trial(
    filter_obj,
    filter_name: str,
    sim_model: StateSpaceModel,
    filter_model: StateSpaceModel,
    observations: np.ndarray,
    states: np.ndarray,
    Q_filter: np.ndarray,
    num_targets: int,
    rng: np.random.Generator,
    use_q_override: bool = False,
) -> TrialResult:
    """Run a single filter trial."""
    
    start = time.time()
    
    # Run filter
    if use_q_override and hasattr(filter_obj, 'filter'):
        import inspect
        sig = inspect.signature(filter_obj.filter)
        if 'Q_override' in sig.parameters:
            result = filter_obj.filter(filter_model, observations, Q_override=Q_filter, rng=rng)
        else:
            result = filter_obj.filter(filter_model, observations, rng=rng)
    else:
        if hasattr(filter_obj, 'filter'):
            try:
                result = filter_obj.filter(filter_model, observations, rng=rng)
            except TypeError:
                result = filter_obj.filter(filter_model, observations)
        else:
            raise ValueError(f"Unknown filter type: {type(filter_obj)}")
    
    runtime = time.time() - start
    
    # Compute OMAT
    omat_per_step, omat_mean = compute_omat(states, result.means, num_targets=num_targets)
    
    # ESS
    if hasattr(result, 'ess') and result.ess is not None and not np.all(np.isnan(result.ess)):
        ess_per_step = result.ess
        ess_mean = np.mean(result.ess)
    else:
        ess_per_step = None
        ess_mean = np.nan
    
    return TrialResult(
        filter_name=filter_name,
        omat_per_step=omat_per_step,
        omat_mean=omat_mean,
        ess_per_step=ess_per_step,
        ess_mean=ess_mean,
        runtime=runtime,
    )


def run_experiment(
    T: int = 40,
    n_tracks: int = 10,
    n_runs_per_track: int = 5,
    num_targets: int = 4,
    n_particles: int = 500,
    n_particles_bpf: int = 100000,
    use_redraw: bool = True,
    verbose: bool = True,
) -> Dict[str, List[TrialResult]]:
    """
    Run full experiment matching MATLAB setup.
    
    Args:
        T: Number of time steps (40 in paper)
        n_tracks: Number of different trajectories (100 in paper)
        n_runs_per_track: Number of filter runs per trajectory (5 in paper)
        num_targets: Number of targets
        n_particles: Particles for flow-based filters
        n_particles_bpf: Particles for BPF
        use_redraw: Whether to use redraw strategy
        verbose: Print progress
        
    Returns:
        Dict mapping filter name to list of TrialResult
    """
    
    # Create simulation model with Q_real
    Q_real = make_Q_real(num_targets)
    sim_model = make_acoustic_ssm(num_targets=num_targets, Q=Q_real)
    
    # Q_filter for filtering
    Q_filter = make_acoustic_Q_filter(num_targets=num_targets)
    
    # Initial state parameters
    x0 = X0_TRUE[:4*num_targets]
    sigma0 = np.tile(SIGMA0_PER_TARGET, num_targets)
    P0_diag = sigma0 ** 2
    
    if verbose:
        print("Acoustic Tracking Experiment (Matching MATLAB)")
        print("=" * 60)
        print(f"  T={T}, n_tracks={n_tracks}, n_runs_per_track={n_runs_per_track}")
        print(f"  num_targets={num_targets}, n_particles={n_particles}")
        print(f"  use_redraw={use_redraw}")
        print()
    
    # Define filters
    filters = {
        'EKF': (ExtendedKalmanFilter(), False),
        'UKF': (UnscentedKalmanFilter(), False),
        'EDH Flow': (EDHFlow(n_particles=n_particles, redraw=use_redraw), True),
        'EDH-PF-PF': (EDHParticleFilter(n_particles=n_particles, redraw=use_redraw), False),
        'LEDH Flow': (LEDHFlow(n_particles=n_particles, redraw=use_redraw), True),
        'LEDH-PF-PF': (LEDHParticleFilter(n_particles=n_particles, redraw=use_redraw), False),
        f'BPF (N={n_particles_bpf})': (BootstrapParticleFilter(n_particles=n_particles_bpf), False),
    }
    
    results = {name: [] for name in filters}
    
    total_trials = n_tracks * n_runs_per_track
    trial_count = 0
    
    for track_idx in range(n_tracks):
        # Generate trajectory (same for all runs on this track)
        traj_rng = default_rng(1000 + track_idx)
        states = generate_trajectory(sim_model, x0, T, traj_rng, num_targets)
        observations = generate_observations(sim_model, states, traj_rng)
        
        if verbose:
            print(f"Track {track_idx + 1}/{n_tracks}")
        
        for run_idx in range(n_runs_per_track):
            trial_count += 1
            
            # Random initial mean for filter (different for each run)
            init_rng = default_rng(2000 + track_idx * n_runs_per_track + run_idx)
            filter_m0 = sample_initial_filter_mean(x0, sigma0, SURV_REGION, init_rng, num_targets)
            
            # Create filter model with Q_filter (not Q_real!)
            # This ensures both particle propagation and weight calculation use Q_filter
            filter_model = create_filter_model(sim_model, filter_m0, P0_diag, Q_filter, num_targets)
            
            for filter_name, (filter_obj, use_q_override) in filters.items():
                # Set seed for reproducibility
                if hasattr(filter_obj, 'seed'):
                    filter_obj.seed = 3000 + trial_count
                
                filter_rng = default_rng(3000 + trial_count)
                
                if verbose:
                    print(f"  Run {run_idx + 1}/{n_runs_per_track}, {filter_name}...", end=" ", flush=True)
                
                try:
                    trial_result = run_single_trial(
                        filter_obj, filter_name, sim_model, filter_model,
                        observations, states, Q_filter, num_targets, filter_rng, use_q_override
                    )
                    results[filter_name].append(trial_result)
                    
                    if verbose:
                        print(f"OMAT={trial_result.omat_mean:.4f}, ESS={trial_result.ess_mean:.1f}, "
                              f"time={trial_result.runtime:.2f}s")
                except Exception as e:
                    if verbose:
                        print(f"FAILED: {e}")
                    results[filter_name].append(TrialResult(
                        filter_name=filter_name,
                        omat_per_step=np.full(T, np.nan),
                        omat_mean=np.nan,
                        ess_per_step=None,
                        ess_mean=np.nan,
                        runtime=np.nan,
                    ))
    
    return results


def print_summary(results: Dict[str, List[TrialResult]]):
    """Print summary statistics."""
    print()
    print("=" * 70)
    print("Summary (mean ± std over all trials)")
    print("=" * 70)
    print(f"{'Filter':<20} | {'OMAT':>14} | {'Avg ESS':>14} | {'Time (s)':>10}")
    print("-" * 70)
    
    for name, trials in results.items():
        omats = [t.omat_mean for t in trials if not np.isnan(t.omat_mean)]
        esss = [t.ess_mean for t in trials if not np.isnan(t.ess_mean)]
        times = [t.runtime for t in trials if not np.isnan(t.runtime)]
        
        if omats:
            omat_str = f"{np.mean(omats):.4f} ± {np.std(omats):.4f}"
        else:
            omat_str = "N/A"
        
        if esss:
            ess_str = f"{np.mean(esss):.1f} ± {np.std(esss):.1f}"
        else:
            ess_str = "N/A"
        
        if times:
            time_str = f"{np.mean(times):.2f}"
        else:
            time_str = "N/A"
        
        print(f"{name:<20} | {omat_str:>14} | {ess_str:>14} | {time_str:>10}")


def save_results_to_csv(
    results: Dict[str, List[TrialResult]],
    output_dir: str = ".",
    prefix: str = "acoustic_experiment",
) -> Tuple[str, str]:
    """
    Save experiment results to CSV files.
    
    Creates two files:
    1. {prefix}_summary_{timestamp}.csv - Summary statistics per filter
    2. {prefix}_trials_{timestamp}.csv - All trial results
    
    Args:
        results: Dict mapping filter name to list of TrialResult
        output_dir: Directory to save files
        prefix: Prefix for output filenames
        
    Returns:
        Tuple of (summary_filepath, trials_filepath)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary CSV
    summary_path = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filter_name', 'n_trials', 
            'omat_mean', 'omat_std', 
            'ess_mean', 'ess_std', 
            'runtime_mean', 'runtime_std'
        ])
        
        for name, trials in results.items():
            omats = [t.omat_mean for t in trials if not np.isnan(t.omat_mean)]
            esss = [t.ess_mean for t in trials if not np.isnan(t.ess_mean)]
            times = [t.runtime for t in trials if not np.isnan(t.runtime)]
            
            writer.writerow([
                name,
                len(trials),
                np.mean(omats) if omats else np.nan,
                np.std(omats) if omats else np.nan,
                np.mean(esss) if esss else np.nan,
                np.std(esss) if esss else np.nan,
                np.mean(times) if times else np.nan,
                np.std(times) if times else np.nan,
            ])
    
    # Trials CSV (all individual results)
    trials_path = os.path.join(output_dir, f"{prefix}_trials_{timestamp}.csv")
    with open(trials_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filter_name', 'trial_idx', 'omat_mean', 'ess_mean', 'runtime'
        ])
        
        for name, trials in results.items():
            for idx, trial in enumerate(trials):
                writer.writerow([
                    name,
                    idx,
                    trial.omat_mean,
                    trial.ess_mean,
                    trial.runtime,
                ])
    
    return summary_path, trials_path


def save_results_to_npz(
    results: Dict[str, List[TrialResult]],
    output_dir: str = ".",
    prefix: str = "acoustic_experiment",
) -> str:
    """
    Save experiment results to NPZ file (includes per-step data).
    
    Args:
        results: Dict mapping filter name to list of TrialResult
        output_dir: Directory to save file
        prefix: Prefix for output filename
        
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    npz_path = os.path.join(output_dir, f"{prefix}_full_{timestamp}.npz")
    
    # Prepare data for saving
    data = {}
    filter_names = list(results.keys())
    data['filter_names'] = np.array(filter_names, dtype=object)
    
    for name, trials in results.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        
        data[f'{safe_name}_omat_mean'] = np.array([t.omat_mean for t in trials])
        data[f'{safe_name}_ess_mean'] = np.array([t.ess_mean for t in trials])
        data[f'{safe_name}_runtime'] = np.array([t.runtime for t in trials])
        
        # Per-step OMAT (if all trials have same length)
        omat_per_step = [t.omat_per_step for t in trials if t.omat_per_step is not None]
        if omat_per_step and all(len(o) == len(omat_per_step[0]) for o in omat_per_step):
            data[f'{safe_name}_omat_per_step'] = np.array(omat_per_step)
        
        # Per-step ESS
        ess_per_step = [t.ess_per_step for t in trials if t.ess_per_step is not None]
        if ess_per_step and all(len(e) == len(ess_per_step[0]) for e in ess_per_step):
            data[f'{safe_name}_ess_per_step'] = np.array(ess_per_step)
    
    np.savez(npz_path, **data)
    
    return npz_path


if __name__ == "__main__":
    # Quick test with reduced settings
    results = run_experiment(
        T=40,
        n_tracks=10,
        n_runs_per_track=5,
        num_targets=4,
        n_particles=500,
        n_particles_bpf=int(1e5),  # Reduced for speed
        use_redraw=True,
        verbose=True,
    )
    
    print_summary(results)
    
    # Save results
    output_dir = "pfresults"
    summary_path, trials_path = save_results_to_csv(results, output_dir=output_dir)
    npz_path = save_results_to_npz(results, output_dir=output_dir)
    
    print()
    print("Results saved to:")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Trials CSV:  {trials_path}")
    print(f"  Full NPZ:    {npz_path}")
