"""
Filter Comparison Experiments: EDH vs LEDH vs Matrix Kernel PFF

Experiments:
1. Full observation LGSSM (20D) - EDH/LEDH should win (exact for linear Gaussian)
2. Partial observation LGSSM with Kernel PFF localization ON
3. Partial observation LGSSM with Kernel PFF localization OFF
4. Range-Bearing (nonlinear) with localization ON
5. Range-Bearing (nonlinear) with localization OFF

Based on:
- Li & Coates (2017): EDH and LEDH Particle Flow Particle Filters
- Hu & van Leeuwen (2021): Kernel-Embedded Particle Flow Filter
"""

import numpy as np
import time
from numpy.random import default_rng
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '..')

from advanced_particle_filter.models.base import StateSpaceModel
from advanced_particle_filter.models.range_bearing import make_range_bearing_ssm
from advanced_particle_filter.filters.edh import EDHFlow, generate_lambda_schedule
from advanced_particle_filter.filters.ledh import LEDHFlow
from advanced_particle_filter.filters.kernel_pff import KernelPFF
from advanced_particle_filter.simulation import simulate


# =============================================================================
# Model Factories
# =============================================================================

def make_lgssm(
    nx: int,
    ny: int,
    q: float = 0.1,
    r: float = 0.1,
) -> StateSpaceModel:
    """
    Create linear Gaussian SSM with sparse or full observations.
    
    State: x ∈ R^nx (random walk dynamics)
    Observation: y = C @ x + noise, where C selects first ny states
    """
    A = np.eye(nx)
    Q = q * np.eye(nx)
    
    C = np.zeros((ny, nx))
    for i in range(ny):
        C[i, i] = 1.0
    
    R = r * np.eye(ny)
    m0 = np.zeros(nx)
    P0 = np.eye(nx)
    
    return StateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=m0,
        initial_cov=P0,
        dynamics_mean=lambda x: x @ A.T,
        dynamics_cov=Q,
        dynamics_jacobian=lambda x: A,
        obs_mean=lambda x: (x @ C.T) if x.ndim > 1 else (C @ x),
        obs_cov=R,
        obs_jacobian=lambda x: C,
    )


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class FilterResult:
    """Result from a single filter run on one trajectory."""
    rmse: float
    runtime: float
    # Optional diagnostics (for nonlinear experiments)
    flow_magnitude: Optional[float] = None
    condition_number: Optional[float] = None


@dataclass
class ExperimentResult:
    """Aggregated results across trajectories."""
    filter_name: str
    rmse_mean: float
    rmse_std: float
    runtime_mean: float
    runtime_std: float
    flow_magnitude_mean: Optional[float] = None
    condition_number_mean: Optional[float] = None
    per_trajectory: List[FilterResult] = field(default_factory=list)


# =============================================================================
# Diagnostic Utilities (for nonlinear experiments)
# =============================================================================

def compute_spectral_radius(A: np.ndarray) -> float:
    """Compute spectral radius of matrix A."""
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def compute_condition_number(A: np.ndarray, tol: float = 1e-10) -> float:
    """Compute effective condition number (ratio of non-zero singular values)."""
    if A.size == 0:
        return 1.0
    try:
        s = np.linalg.svd(A, compute_uv=False)
        s_nonzero = s[s > tol * s[0]]
        if len(s_nonzero) < 2:
            return 1.0
        return float(s_nonzero[0] / s_nonzero[-1])
    except:
        return 1e10


def compute_edh_jacobian(eta_bar, P, R, lam, model):
    """Compute EDH flow Jacobian A(λ)."""
    H = model.obs_jacobian(eta_bar)
    S = lam * H @ P @ H.T + R
    S = 0.5 * (S + S.T)
    A = -0.5 * P @ H.T @ np.linalg.solve(S, H)
    return A


# =============================================================================
# Core Experiment Runner
# =============================================================================

def run_filter_with_diagnostics(
    filt,
    model: StateSpaceModel,
    observations: np.ndarray,
    rng,
    filter_name: str,
) -> Tuple[object, Optional[float], Optional[float]]:
    """
    Run filter and extract diagnostics.
    
    Returns:
        result: FilterResult
        flow_magnitude: Mean flow magnitude across time steps (or None)
        condition_number: Mean Jacobian condition number (or None)
    """
    # EDH and LEDH support return_diagnostics
    if filter_name.startswith('EDH') or filter_name.startswith('LEDH'):
        result = filt.filter(model, observations, rng=rng, return_diagnostics=True)
        
        # Extract diagnostics from result
        if hasattr(result, 'diagnostics') and result.diagnostics:
            all_flow_mags = []
            all_cond_nums = []
            for diag in result.diagnostics:
                if 'flow_magnitudes' in diag:
                    all_flow_mags.extend(diag['flow_magnitudes'])
                if 'jacobian_conds' in diag:
                    all_cond_nums.extend(diag['jacobian_conds'])
            
            flow_mag = np.mean(all_flow_mags) if all_flow_mags else None
            cond_num = np.mean(all_cond_nums) if all_cond_nums else None
        else:
            flow_mag = None
            cond_num = None
        
        return result, flow_mag, cond_num
    
    else:  # KernelPFF
        result = filt.filter(model, observations, rng=rng)
        
        # For KernelPFF, we need to estimate diagnostics separately
        # Run a single time step with detailed tracking
        flow_mag, cond_num = estimate_kernel_pff_diagnostics(filt, model, observations, rng)
        
        return result, flow_mag, cond_num


def estimate_kernel_pff_diagnostics(
    filt,
    model: StateSpaceModel,
    observations: np.ndarray,
    rng,
    n_sample_steps: int = 3,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate flow magnitude and Jacobian condition number for Kernel PFF.
    
    Uses finite differences to estimate local Jacobian.
    
    Args:
        filt: KernelPFF filter (already run)
        model: StateSpaceModel
        observations: [T, ny] observations
        rng: random generator
        n_sample_steps: Number of time steps to sample for diagnostics
        
    Returns:
        flow_magnitude: Estimated mean flow magnitude
        condition_number: Estimated mean Jacobian condition number
    """
    T = observations.shape[0]
    nx = model.state_dim
    N = filt.n_particles
    
    # Sample a few time steps
    step_indices = np.linspace(0, T-1, min(n_sample_steps, T), dtype=int)
    
    flow_mags = []
    cond_nums = []
    
    # Create fresh particles
    particles = model.sample_initial(N, rng)
    
    for t in step_indices:
        z = observations[t]
        
        # Propagate to get prior particles
        particles_prior = model.sample_dynamics(particles, rng)
        
        # Compute prior statistics
        x_mean = particles_prior.mean(axis=0)
        B = np.cov(particles_prior.T)
        if B.ndim == 0:
            B = np.array([[B]])
        B = 0.5 * (B + B.T) + 1e-6 * np.eye(nx)
        B_inv = np.linalg.inv(B)
        
        # Compute gradient at particles
        grad = compute_kernel_pff_gradient(particles_prior, z, model, B_inv, x_mean)
        
        # Estimate flow magnitude from gradient
        flow_mag = np.mean(np.linalg.norm(grad, axis=1))
        flow_mags.append(flow_mag)
        
        # Estimate Jacobian condition number via finite differences
        cond_num = estimate_kernel_pff_jacobian_condition(
            particles_prior, z, model, B_inv, x_mean, filt
        )
        cond_nums.append(cond_num)
        
        particles = particles_prior
    
    return (
        np.mean(flow_mags) if flow_mags else None,
        np.mean(cond_nums) if cond_nums else None,
    )


def compute_kernel_pff_gradient(
    particles: np.ndarray,
    z: np.ndarray,
    model: StateSpaceModel,
    B_inv: np.ndarray,
    x_mean: np.ndarray,
) -> np.ndarray:
    """
    Compute gradient of log posterior for Kernel PFF.
    
    ∇log p(x|y) = ∇log p(y|x) + ∇log p(x)
    
    For linear-Gaussian likelihood: ∇log p(y|x) = H^T R^{-1} (y - Hx)
    For Gaussian prior: ∇log p(x) = -B^{-1}(x - x̄)
    """
    N, nx = particles.shape
    H = model.obs_jacobian(particles[0])  # Assume constant or use mean
    R_inv = np.linalg.inv(model.obs_cov)
    
    # Observation mean
    y_pred = model.obs_mean(particles)  # [N, ny]
    residual = z - y_pred  # [N, ny]
    
    # Gradient of log likelihood
    grad_ll = residual @ R_inv @ H  # [N, nx]
    
    # Gradient of log prior
    grad_prior = -(particles - x_mean) @ B_inv  # [N, nx]
    
    return grad_ll + grad_prior


def estimate_kernel_pff_jacobian_condition(
    particles: np.ndarray,
    z: np.ndarray,
    model: StateSpaceModel,
    B_inv: np.ndarray,
    x_mean: np.ndarray,
    filt,
    delta: float = 1e-5,
    n_sample: int = 5,
) -> float:
    """
    Estimate Jacobian condition number for Kernel PFF via finite differences.
    
    Samples a few particles and estimates local Jacobian.
    """
    N, nx = particles.shape
    
    # Sample particles to estimate Jacobian
    indices = np.linspace(0, N-1, min(n_sample, N), dtype=int)
    
    cond_nums = []
    
    for idx in indices:
        x0 = particles[idx]
        
        # Compute Jacobian via finite differences
        J = np.zeros((nx, nx))
        
        # Base gradient
        grad_0 = compute_kernel_pff_gradient(
            x0[np.newaxis, :], z, model, B_inv, x_mean
        )[0]
        
        for j in range(nx):
            x_pert = x0.copy()
            x_pert[j] += delta
            
            grad_pert = compute_kernel_pff_gradient(
                x_pert[np.newaxis, :], z, model, B_inv, x_mean
            )[0]
            
            J[:, j] = (grad_pert - grad_0) / delta
        
        # Condition number
        try:
            cond = np.linalg.cond(np.eye(nx) + 0.01 * J)  # I + εJ approximation
            cond_nums.append(cond)
        except:
            cond_nums.append(1e10)
    
    return np.mean(cond_nums) if cond_nums else 1.0


def run_experiment(
    model: StateSpaceModel,
    filter_configs: List[Tuple[str, dict]],
    T: int = 20,
    n_particles: int = 200,
    n_trajectories: int = 10,
    seed: int = 42,
    track_diagnostics: bool = False,
    verbose: bool = True,
) -> Dict[str, ExperimentResult]:
    """
    Run experiment comparing multiple filters.
    
    Args:
        model: StateSpaceModel
        filter_configs: List of (name, kwargs) for each filter
        T: Time steps per trajectory
        n_particles: Number of particles
        n_trajectories: Number of trajectories
        seed: Random seed
        track_diagnostics: If True, compute flow magnitude and condition number
        verbose: Print progress
        
    Returns:
        Dict mapping filter name to ExperimentResult
    """
    results = {name: [] for name, _ in filter_configs}
    
    for traj_idx in range(n_trajectories):
        traj_seed = seed + traj_idx * 1000
        rng = default_rng(traj_seed)
        
        # Generate trajectory
        trajectory = simulate(model, T, rng=rng)
        states = trajectory.states
        observations = trajectory.observations
        
        if verbose:
            print(f"  Trajectory {traj_idx + 1}/{n_trajectories}", end="")
        
        for name, kwargs in filter_configs:
            # Create filter
            if name.startswith('EDH'):
                filt = EDHFlow(n_particles=n_particles, seed=traj_seed)
            elif name.startswith('LEDH'):
                filt = LEDHFlow(n_particles=n_particles, seed=traj_seed)
            else:  # KernelPFF
                filt = KernelPFF(n_particles=n_particles, seed=traj_seed, **kwargs)
            
            # Run filter (timed)
            t0 = time.time()
            if track_diagnostics:
                result, flow_mag, cond_num = run_filter_with_diagnostics(
                    filt, model, observations, default_rng(traj_seed), name
                )
            else:
                result = filt.filter(model, observations, rng=default_rng(traj_seed))
                flow_mag = None
                cond_num = None
            runtime = time.time() - t0
            
            rmse = result.mean_rmse(states)
            
            results[name].append(FilterResult(
                rmse=rmse,
                runtime=runtime,
                flow_magnitude=flow_mag,
                condition_number=cond_num,
            ))
        
        if verbose:
            print(" - Done")
    
    # Aggregate
    aggregated = {}
    for name, traj_results in results.items():
        rmses = [r.rmse for r in traj_results]
        runtimes = [r.runtime for r in traj_results]
        
        flow_mags = [r.flow_magnitude for r in traj_results if r.flow_magnitude is not None]
        cond_nums = [r.condition_number for r in traj_results if r.condition_number is not None]
        
        aggregated[name] = ExperimentResult(
            filter_name=name,
            rmse_mean=np.mean(rmses),
            rmse_std=np.std(rmses),
            runtime_mean=np.mean(runtimes),
            runtime_std=np.std(runtimes),
            flow_magnitude_mean=np.mean(flow_mags) if flow_mags else None,
            condition_number_mean=np.mean(cond_nums) if cond_nums else None,
            per_trajectory=traj_results,
        )
    
    return aggregated


def print_results(results: Dict[str, ExperimentResult], title: str = ""):
    """Print results table."""
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)
    
    has_diagnostics = any(r.flow_magnitude_mean is not None for r in results.values())
    
    if has_diagnostics:
        print(f"{'Filter':<25} | {'RMSE':>16} | {'Runtime':>14} | {'FlowMag':>10} | {'κ(A)':>10}")
        print("-" * 70)
        for name, res in results.items():
            fm = f"{res.flow_magnitude_mean:.4f}" if res.flow_magnitude_mean else "N/A"
            cn = f"{res.condition_number_mean:.2f}" if res.condition_number_mean else "N/A"
            print(f"{name:<25} | {res.rmse_mean:>6.4f} ± {res.rmse_std:<6.4f} | {res.runtime_mean:>5.3f} ± {res.runtime_std:<4.3f}s | {fm:>10} | {cn:>10}")
    else:
        print(f"{'Filter':<25} | {'RMSE':>16} | {'Runtime':>14}")
        print("-" * 70)
        for name, res in results.items():
            print(f"{name:<25} | {res.rmse_mean:>6.4f} ± {res.rmse_std:<6.4f} | {res.runtime_mean:>5.3f} ± {res.runtime_std:<4.3f}s")
    
    print("=" * 70)


# =============================================================================
# Experiment 1: Full Observation LGSSM
# =============================================================================

def run_experiment_1(
    n_particles: int = 200,
    T: int = 20,
    n_trajectories: int = 10,
    seed: int = 42,
) -> Dict[str, ExperimentResult]:
    """
    Experiment 1: Full observation LGSSM (20D, 100% observed)
    
    Expected: EDH and LEDH should win (exact for linear Gaussian)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Full Observation LGSSM (20D, 100% observed)")
    print("Expected: EDH/LEDH win (exact for linear Gaussian)")
    print("=" * 70)
    
    model = make_lgssm(nx=20, ny=20)
    
    filter_configs = [
        ('EDH', {}),
        ('LEDH', {}),
        ('KernelPFF_loc', {
            'kernel_type': 'matrix',
            'use_localization': True,
            'localization_radius': 4.0,
            'max_iterations': 150,
        }),
    ]
    
    results = run_experiment(
        model, filter_configs,
        T=T, n_particles=n_particles, n_trajectories=n_trajectories, seed=seed,
    )
    
    print_results(results, "Experiment 1: Full Observation LGSSM")
    return results


# =============================================================================
# Experiment 2: Partial Observation with Localization ON
# =============================================================================

def run_experiment_2(
    n_particles: int = 200,
    T: int = 20,
    n_trajectories: int = 10,
    seed: int = 42,
) -> Dict[str, Dict[str, ExperimentResult]]:
    """
    Experiment 2: Partial observation LGSSM with Kernel PFF localization ON
    
    20D state, 20% and 10% observed
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Partial Observation LGSSM - Localization ON")
    print("=" * 70)
    
    all_results = {}
    
    for ny, obs_pct in [(4, "20%"), (2, "10%")]:
        print(f"\n--- {obs_pct} observed ({ny}/20) ---")
        
        model = make_lgssm(nx=20, ny=ny)
        
        filter_configs = [
            ('EDH', {}),
            ('LEDH', {}),
            ('KernelPFF_loc', {
                'kernel_type': 'matrix',
                'use_localization': True,
                'localization_radius': 4.0,
                'max_iterations': 150,
            }),
        ]
        
        results = run_experiment(
            model, filter_configs,
            T=T, n_particles=n_particles, n_trajectories=n_trajectories, seed=seed,
        )
        
        all_results[obs_pct] = results
        print_results(results, f"Experiment 2: {obs_pct} Observed - Localization ON")
    
    return all_results


# =============================================================================
# Experiment 3: Partial Observation with Localization OFF
# =============================================================================

def run_experiment_3(
    n_particles: int = 200,
    T: int = 20,
    n_trajectories: int = 10,
    seed: int = 42,
) -> Dict[str, Dict[str, ExperimentResult]]:
    """
    Experiment 3: Partial observation LGSSM with Kernel PFF localization OFF
    
    Expected: Kernel PFF loses (no observed↔unobserved communication)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Partial Observation LGSSM - Localization OFF")
    print("Expected: Kernel PFF loses (no communication)")
    print("=" * 70)
    
    all_results = {}
    
    for ny, obs_pct in [(4, "20%"), (2, "10%")]:
        print(f"\n--- {obs_pct} observed ({ny}/20) ---")
        
        model = make_lgssm(nx=20, ny=ny)
        
        filter_configs = [
            ('EDH', {}),
            ('LEDH', {}),
            ('KernelPFF_no_loc', {
                'kernel_type': 'matrix',
                'use_localization': False,
                'max_iterations': 150,
            }),
        ]
        
        results = run_experiment(
            model, filter_configs,
            T=T, n_particles=n_particles, n_trajectories=n_trajectories, seed=seed,
        )
        
        all_results[obs_pct] = results
        print_results(results, f"Experiment 3: {obs_pct} Observed - Localization OFF")
    
    return all_results


# =============================================================================
# Experiment 4: Range-Bearing with Localization ON
# =============================================================================

def run_experiment_4(
    n_particles: int = 200,
    T: int = 30,
    n_trajectories: int = 10,
    seed: int = 42,
) -> Dict[str, ExperimentResult]:
    """
    Experiment 4: Range-Bearing (nonlinear) with localization ON
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Range-Bearing (Nonlinear) - Localization ON")
    print("=" * 70)
    
    model = make_range_bearing_ssm(
        dt=0.1,
        q_diag=0.01,
        nu=5.0,
        s_r=0.1,
        s_th=0.05,
        m0=(5.0, 5.0, 0.1, 0.1),
        P0_diag=(1.0, 1.0, 0.5, 0.5),
    )
    
    filter_configs = [
        ('EDH', {}),
        ('LEDH', {}),
        ('KernelPFF_loc', {
            'kernel_type': 'matrix',
            'use_localization': True,
            'localization_radius': 2.0,  # Smaller radius for 4D state
            'max_iterations': 200,
        }),
    ]
    
    results = run_experiment(
        model, filter_configs,
        T=T, n_particles=n_particles, n_trajectories=n_trajectories, seed=seed,
        track_diagnostics=True,
    )
    
    print_results(results, "Experiment 4: Range-Bearing - Localization ON")
    return results


# =============================================================================
# Experiment 5: Range-Bearing with Localization OFF
# =============================================================================

def run_experiment_5(
    n_particles: int = 200,
    T: int = 30,
    n_trajectories: int = 10,
    seed: int = 42,
) -> Dict[str, ExperimentResult]:
    """
    Experiment 5: Range-Bearing (nonlinear) with localization OFF
    
    Expected: Kernel PFF loses (poor stability)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Range-Bearing (Nonlinear) - Localization OFF")
    print("Expected: Kernel PFF loses (poor stability)")
    print("=" * 70)
    
    model = make_range_bearing_ssm(
        dt=0.1,
        q_diag=0.01,
        nu=5.0,
        s_r=0.1,
        s_th=0.05,
        m0=(5.0, 5.0, 0.1, 0.1),
        P0_diag=(1.0, 1.0, 0.5, 0.5),
    )
    
    filter_configs = [
        ('EDH', {}),
        ('LEDH', {}),
        ('KernelPFF_no_loc', {
            'kernel_type': 'matrix',
            'use_localization': False,
            'max_iterations': 200,
        }),
    ]
    
    results = run_experiment(
        model, filter_configs,
        T=T, n_particles=n_particles, n_trajectories=n_trajectories, seed=seed,
        track_diagnostics=True,
    )
    
    print_results(results, "Experiment 5: Range-Bearing - Localization OFF")
    return results


# =============================================================================
# Run All Experiments
# =============================================================================

def run_all_experiments(
    n_particles: int = 200,
    T_lgssm: int = 20,
    T_rb: int = 30,
    n_trajectories: int = 10,
    seed: int = 42,
) -> Dict:
    """Run all experiments and return results."""
    
    all_results = {}
    
    all_results['exp1'] = run_experiment_1(n_particles, T_lgssm, n_trajectories, seed)
    all_results['exp2'] = run_experiment_2(n_particles, T_lgssm, n_trajectories, seed)
    all_results['exp3'] = run_experiment_3(n_particles, T_lgssm, n_trajectories, seed)
    all_results['exp4'] = run_experiment_4(n_particles, T_rb, n_trajectories, seed)
    all_results['exp5'] = run_experiment_5(n_particles, T_rb, n_trajectories, seed)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    
    return all_results
