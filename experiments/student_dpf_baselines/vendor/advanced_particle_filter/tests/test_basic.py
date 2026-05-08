"""
Basic test script for advanced_particle_filter library.

Run: python test_basic.py
"""

import numpy as np
from numpy.random import default_rng

from advanced_particle_filter.models import make_lgssm
from advanced_particle_filter.simulation import simulate
from advanced_particle_filter.filters import (
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    BootstrapParticleFilter,
    EDHFlow,
    EDHParticleFilter,
    LEDHFlow,
    LEDHParticleFilter
)


def test_equivalence_lgssm():
    """Test Linear Gaussian SSM with all filters."""
    print("=" * 60)
    print("Testing Linear Gaussian State Space Model")
    print("=" * 60)
    
    # Define model: 2D position tracking
    dt = 1
    nx, ny = 4, 2
    
    # State: [x, y, vx, vy]
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    
    # Observe: [x, y]
    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    
    # Process noise
    q = 0.1
    Q = q * np.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt],
    ])
    
    # Observation noise
    r = 0.01 # Also check the stability w.r.t. smalle observation noise.
    R = r * np.eye(ny)
    
    # Initial state
    m0 = np.array([0.0, 0.0, 1.0, 0.5])
    P0 = np.eye(nx) * 0.1
    
    # Create model
    model = make_lgssm(A, C, Q, R, m0, P0)
    print(f"Model: {model}")
    
    # Simulate trajectory
    T = 50
    rng = default_rng(42)
    trajectory = simulate(model, T, rng=rng)
    
    print(f"Simulated {T} steps")
    print(f"States shape: {trajectory.states.shape}")
    print(f"Observations shape: {trajectory.observations.shape}")
    
    # Run filters
    kf = KalmanFilter()
    ekf = ExtendedKalmanFilter()
    ukf = UnscentedKalmanFilter()
    bpf = BootstrapParticleFilter(n_particles=1000, seed=123)
    edh_flow = EDHFlow(seed=123)
    edh_pfpf = EDHParticleFilter(n_particles=1000, seed=123)
    ledh_flow = LEDHFlow(seed=123)
    ledh_pfpf = LEDHParticleFilter(n_particles=1000, seed=123)

    result_kf = kf.filter(model, trajectory.observations)
    result_ekf = ekf.filter(model, trajectory.observations)
    result_ukf = ukf.filter(model, trajectory.observations)
    result_bpf = bpf.filter(model, trajectory.observations)
    result_edh_flow = edh_flow.filter(model, trajectory.observations)
    result_edh_pfpf = edh_pfpf.filter(model, trajectory.observations)
    result_ledh_flow = ledh_flow.filter(model, trajectory.observations)
    result_ledh_pfpf = ledh_pfpf.filter(model, trajectory.observations) 

    # Compute RMSEs
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"KF  - Mean RMSE: {result_kf.mean_rmse(trajectory.states):.4f}, "
          f"Log-lik: {result_kf.log_likelihood:.2f}")
    print(f"EKF - Mean RMSE: {result_ekf.mean_rmse(trajectory.states):.4f}, "
          f"Log-lik: {result_ekf.log_likelihood:.2f}")
    print(f"UKF - Mean RMSE: {result_ukf.mean_rmse(trajectory.states):.4f}, "
          f"Log-lik: {result_ukf.log_likelihood:.2f}")
    print(f"BPF - Mean RMSE: {result_bpf.mean_rmse(trajectory.states):.4f}, "
          f"Avg ESS: {result_bpf.average_ess():.1f}")
    print(f"EDH Flow - Mean RMSE: {result_edh_flow.mean_rmse(trajectory.states):.4f}, "
          f"Avg ESS: {result_edh_flow.average_ess():.1f}")
    print(f"EDH PFPF - Mean RMSE: {result_edh_pfpf.mean_rmse(trajectory.states):.4f}, "
          f"Avg ESS: {result_edh_pfpf.average_ess():.1f}")    
    print(f"LEDH Flow - Mean RMSE: {result_ledh_flow.mean_rmse(trajectory.states):.4f}, "
          f"Avg ESS: {result_ledh_flow.average_ess():.1f}")
    print(f"LEDH PFPF - Mean RMSE: {result_ledh_pfpf.mean_rmse(trajectory.states):.4f}, "
          f"Avg ESS: {result_ledh_pfpf.average_ess():.1f}")  
    
    # For LGSSM, KF should be optimal, EKF/UKF/Bootstrap PF should match
    assert np.isclose(result_kf.mean_rmse(trajectory.states), 
                      result_ekf.mean_rmse(trajectory.states), rtol=0.1), \
        "EKF should match KF for linear model"
    assert np.isclose(result_kf.mean_rmse(trajectory.states), 
                      result_ukf.mean_rmse(trajectory.states), rtol=0.1), \
        "UKF should match KF for linear model"
    assert np.isclose(result_kf.mean_rmse(trajectory.states), 
                      result_bpf.mean_rmse(trajectory.states), rtol=0.1), \
        "Bootstrap PF should match KF for linear model"    
    assert np.isclose(result_kf.mean_rmse(trajectory.states), 
                      result_edh_flow.mean_rmse(trajectory.states), rtol=0.1), \
        "EDH flow should match KF for linear model"    
    assert np.isclose(result_kf.mean_rmse(trajectory.states), 
                      result_edh_pfpf.mean_rmse(trajectory.states), rtol=0.1), \
        "EDH PF-PF should match KF for linear model" 
    assert np.isclose(result_kf.mean_rmse(trajectory.states), 
                      result_ledh_flow.mean_rmse(trajectory.states), rtol=0.1), \
        "LEDH flow should match KF for linear model"     
    assert np.isclose(result_kf.mean_rmse(trajectory.states), 
                      result_ledh_pfpf.mean_rmse(trajectory.states), rtol=0.1), \
        "LEDH PF-PF should match KF for linear model"            
    print("\n✓ All filters working correctly!")
    return True


def test_model_simulation():
    """Test model simulation and sampling."""
    print("\n" + "=" * 60)
    print("Testing Model Simulation")
    print("=" * 60)
    
    # Simple 1D random walk
    A = np.array([[1.0]])
    C = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    m0 = np.array([0.0])
    P0 = np.array([[1.0]])
    
    model = make_lgssm(A, C, Q, R, m0, P0)
    
    # Test sampling
    rng = default_rng(0)
    x0 = model.sample_initial(100, rng)
    assert x0.shape == (100, 1), f"Expected (100, 1), got {x0.shape}"
    print(f"Initial samples: mean={x0.mean():.3f}, std={x0.std():.3f}")
    
    # Test dynamics
    x1 = model.sample_dynamics(x0, rng)
    assert x1.shape == (100, 1)
    print(f"Propagated samples: mean={x1.mean():.3f}, std={x1.std():.3f}")
    
    # Test observation
    y = model.sample_observation(x0[0], rng)
    assert y.shape == (1,), f"Expected (1,), got {y.shape}"
    print(f"Observation: {y[0]:.3f}")
    
    # Test log prob
    log_probs = model.observation_log_prob(x0, y)
    assert log_probs.shape == (100,)
    print(f"Log probs: mean={log_probs.mean():.3f}")
    
    print("\n✓ Model simulation working correctly!")
    return True


def test_particle_filter_resampling():
    """Test different resampling schemes."""
    print("\n" + "=" * 60)
    print("Testing Particle Filter Resampling")
    print("=" * 60)
    
    # Simple model
    A = np.array([[0.9]])
    C = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    m0 = np.array([0.0])
    P0 = np.array([[1.0]])
    
    model = make_lgssm(A, C, Q, R, m0, P0)
    trajectory = simulate(model, T=30, seed=42)
    
    methods = ["systematic", "stratified", "multinomial", "residual"]
    
    for method in methods:
        pf = BootstrapParticleFilter(
            n_particles=200,
            resample_method=method,
            resample_criterion="ess",
            ess_threshold=0.5,
            seed=123,
        )
        result = pf.filter(model, trajectory.observations)
        print(f"{method:12s} - RMSE: {result.mean_rmse(trajectory.states):.4f}, "
              f"Avg ESS: {result.average_ess():.1f}, "
              f"Resampled: {result.resampled.sum()}/{len(result.resampled)}")
    
    print("\n✓ All resampling methods working!")
    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Advanced Particle Filter Library Tests")
    print("#" * 60)
    
    tests = [
        test_model_simulation, # test model's dynamics and measurement process under different initial conditions and check their statistics.
        test_equivalence_lgssm, # test different filiters' equivalnce at lgssm.
        test_particle_filter_resampling, # test BPF under different resampling method by comparing filtering statistics.
        # add test:
        # 1. small observation noise stability (Joseph Covariance Update Rule).
        # 2. Kalman Filter Correctness Check by Stable Dynamics (Check the convergence of the filtered covariance matrix).
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#" * 60)
    print(f"# Results: {passed} passed, {failed} failed")
    print("#" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
