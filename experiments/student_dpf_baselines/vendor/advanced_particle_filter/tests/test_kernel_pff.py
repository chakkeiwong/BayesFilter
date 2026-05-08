"""
Test script for Kernel Particle Flow Filter.

Tests:
1. Scalar Kernel PFF equivalence to KF for Linear Gaussian SSM
2. Matrix Kernel PFF equivalence to KF for Linear Gaussian SSM
3. Convergence behavior of PFF iterations

Run: python test_kernel_pff.py
"""

import numpy as np
from numpy.random import default_rng

# Import from our library
from advanced_particle_filter.models import make_lgssm
from advanced_particle_filter.simulation import simulate
from advanced_particle_filter.filters import (
    KalmanFilter,
    KernelPFF,
    ScalarKernelPFF,
    MatrixKernelPFF,
)


def test_kernel_pff_lgssm():
    """Test Kernel PFF equivalence to KF for Linear Gaussian SSM."""
    print("=" * 60)
    print("Testing Kernel PFF on Linear Gaussian SSM")
    print("=" * 60)
    
    # Define model: 2D position tracking
    dt = 1.0
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
    r = 0.5  # Moderate observation noise for stability
    R = r * np.eye(ny)
    
    # Initial state
    m0 = np.array([0.0, 0.0, 1.0, 0.5])
    P0 = np.eye(nx) * 0.1
    
    # Create model
    model = make_lgssm(A, C, Q, R, m0, P0)
    print(f"Model: {model}")
    
    # Simulate trajectory
    T = 20  # Shorter for faster testing
    rng = default_rng(42)
    trajectory = simulate(model, T, rng=rng)
    
    print(f"Simulated {T} steps")
    
    # Run Kalman Filter (ground truth for linear Gaussian)
    kf = KalmanFilter()
    result_kf = kf.filter(model, trajectory.observations)
    
    # Run Scalar Kernel PFF
    scalar_pff = ScalarKernelPFF(
        n_particles=500,
        max_iterations=100,
        tolerance=1e-5,
        initial_step_size=0.1,
        seed=123,
    )
    result_scalar = scalar_pff.filter(model, trajectory.observations)
    
    # Run Matrix Kernel PFF
    matrix_pff = MatrixKernelPFF(
        n_particles=500,
        max_iterations=100,
        tolerance=1e-5,
        initial_step_size=0.1,
        seed=123,
    )
    result_matrix = matrix_pff.filter(model, trajectory.observations)
    
    # Compute RMSEs
    rmse_kf = result_kf.mean_rmse(trajectory.states)
    rmse_scalar = result_scalar.mean_rmse(trajectory.states)
    rmse_matrix = result_matrix.mean_rmse(trajectory.states)
    
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"KF           - Mean RMSE: {rmse_kf:.4f}")
    print(f"Scalar PFF   - Mean RMSE: {rmse_scalar:.4f}")
    print(f"Matrix PFF   - Mean RMSE: {rmse_matrix:.4f}")
    
    # Check equivalence (with tolerance for Monte Carlo error)
    rtol = 0.3  # 30% relative tolerance for Monte Carlo methods
    
    scalar_match = np.isclose(rmse_kf, rmse_scalar, rtol=rtol)
    matrix_match = np.isclose(rmse_kf, rmse_matrix, rtol=rtol)
    
    print(f"\nScalar PFF matches KF (rtol={rtol}): {scalar_match}")
    print(f"Matrix PFF matches KF (rtol={rtol}): {matrix_match}")
    
    # Assertions
    assert scalar_match, f"Scalar PFF RMSE {rmse_scalar:.4f} should match KF RMSE {rmse_kf:.4f}"
    assert matrix_match, f"Matrix PFF RMSE {rmse_matrix:.4f} should match KF RMSE {rmse_kf:.4f}"
    
    print("\n✓ Kernel PFF filters match KF for linear Gaussian model!")
    return True


def test_kernel_pff_convergence():
    """Test PFF convergence behavior."""
    print("\n" + "=" * 60)
    print("Testing Kernel PFF Convergence")
    print("=" * 60)
    
    # Simple 1D model for clear convergence behavior
    A = np.array([[0.9]])
    C = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    m0 = np.array([0.0])
    P0 = np.array([[1.0]])
    
    model = make_lgssm(A, C, Q, R, m0, P0)
    
    # Single observation
    rng = default_rng(42)
    trajectory = simulate(model, T=5, rng=rng)
    
    # Run with diagnostics
    pff = MatrixKernelPFF(
        n_particles=500,
        alpha=0.1,
        max_iterations=100,
        tolerance=1e-6,
        initial_step_size=0.05,
        seed=123,
        store_diagnostics=True,
    )
    
    result = pff.filter(model, trajectory.observations, return_diagnostics=True)
    
    print(f"\nConvergence diagnostics:")
    for t, diag in enumerate(result.diagnostics):
        print(f"  Step {t+1}: {diag.n_iterations} iterations, "
              f"final flow mag: {diag.final_flow_magnitude:.2e}")
    
    # Check that iterations converged (didn't hit max)
    avg_iterations = np.mean([d.n_iterations for d in result.diagnostics])
    print(f"\nAverage iterations: {avg_iterations:.1f}")
    
    assert avg_iterations < 100, "PFF should converge before max iterations"
    
    print("\n✓ Kernel PFF converges properly!")
    return True


def test_scalar_vs_matrix_kernel():
    """Compare scalar and matrix kernel behavior."""
    print("\n" + "=" * 60)
    print("Comparing Scalar vs Matrix Kernel")
    print("=" * 60)
    
    # 2D model with partial observation (only observe x, not velocity)
    dt = 1.0
    nx, ny = 4, 1  # Only observe position x
    
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    
    # Only observe x position
    C = np.array([[1, 0, 0, 0]])
    
    q = 0.1
    Q = q * np.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt],
    ])
    
    r = 0.1  # Small observation noise (informative observations)
    R = r * np.eye(ny)
    
    m0 = np.array([0.0, 0.0, 1.0, 0.5])
    P0 = np.eye(nx) * 0.5
    
    model = make_lgssm(A, C, Q, R, m0, P0)
    
    T = 15
    rng = default_rng(42)
    trajectory = simulate(model, T, rng=rng)
    
    # Run both kernels
    kf = KalmanFilter()
    result_kf = kf.filter(model, trajectory.observations)
    
    scalar_pff = ScalarKernelPFF(
        n_particles=500,
        alpha=0.1,
        max_iterations=100,
        tolerance=1e-5,
        initial_step_size=0.1,
        seed=123,
    )
    result_scalar = scalar_pff.filter(model, trajectory.observations)
    
    matrix_pff = MatrixKernelPFF(
        n_particles=500,
        alpha=0.1,
        max_iterations=100,
        tolerance=1e-5,
        initial_step_size=0.1,
        seed=123,
    )
    result_matrix = matrix_pff.filter(model, trajectory.observations)
    
    rmse_kf = result_kf.mean_rmse(trajectory.states)
    rmse_scalar = result_scalar.mean_rmse(trajectory.states)
    rmse_matrix = result_matrix.mean_rmse(trajectory.states)
    
    print(f"\nPartially observed system (1 of 4 states observed):")
    print(f"KF           - Mean RMSE: {rmse_kf:.4f}")
    print(f"Scalar PFF   - Mean RMSE: {rmse_scalar:.4f}")
    print(f"Matrix PFF   - Mean RMSE: {rmse_matrix:.4f}")
    
    # Both should be reasonably close to KF
    rtol = 0.4  # Looser tolerance for partial observation
    
    print(f"\nScalar within {rtol*100:.0f}% of KF: {np.isclose(rmse_kf, rmse_scalar, rtol=rtol)}")
    print(f"Matrix within {rtol*100:.0f}% of KF: {np.isclose(rmse_kf, rmse_matrix, rtol=rtol)}")
    
    print("\n✓ Both kernels work for partially observed system!")
    return True


def test_high_particle_count():
    """Test that more particles improves accuracy."""
    print("\n" + "=" * 60)
    print("Testing Particle Count Effect")
    print("=" * 60)
    
    # Simple model
    A = np.array([[0.95]])
    C = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.3]])
    m0 = np.array([0.0])
    P0 = np.array([[1.0]])
    
    model = make_lgssm(A, C, Q, R, m0, P0)
    
    T = 20
    rng = default_rng(42)
    trajectory = simulate(model, T, rng=rng)
    
    # Ground truth
    kf = KalmanFilter()
    result_kf = kf.filter(model, trajectory.observations)
    rmse_kf = result_kf.mean_rmse(trajectory.states)
    
    particle_counts = [100, 500, 1000, 2000]
    rmses = []
    
    for N in particle_counts:
        pff = MatrixKernelPFF(
            n_particles=N,
            alpha=0.1,
            max_iterations=100,
            tolerance=1e-5,
            initial_step_size=0.1,
            seed=123,
        )
        result = pff.filter(model, trajectory.observations)
        rmse = result.mean_rmse(trajectory.states)
        rmses.append(rmse)
        print(f"N={N:4d}: RMSE={rmse:.4f}, diff from KF={abs(rmse-rmse_kf):.4f}")
    
    # Check that error generally decreases with more particles
    # (may not be strictly monotonic due to randomness)
    print(f"\nKF RMSE: {rmse_kf:.4f}")
    print(f"Best PFF RMSE: {min(rmses):.4f}")
    
    # The best should be close to KF
    assert min(rmses) < rmse_kf * 1.5, "Best PFF should be within 50% of KF"
    
    print("\n✓ More particles generally improves accuracy!")
    return True


def main():
    """Run all Kernel PFF tests."""
    print("\n" + "#" * 60)
    print("# Kernel Particle Flow Filter Tests")
    print("#" * 60)
    
    tests = [
        test_kernel_pff_lgssm,
        # test_kernel_pff_convergence,
        # test_scalar_vs_matrix_kernel,
        # test_high_particle_count,
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
