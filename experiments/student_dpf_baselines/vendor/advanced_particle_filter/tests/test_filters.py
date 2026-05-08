"""
Test suite for advanced_particle_filter library.

Tests KF, EKF, UKF validity and cross-consistency, plus PF approximation quality.

Run: pytest test_filters.py -v
"""

import pytest
import numpy as np
from numpy.random import default_rng

from advanced_particle_filter.models.base import StateSpaceModel
from advanced_particle_filter.models.linear_gaussian import make_lgssm
from advanced_particle_filter.filters.kalman import (
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
)
from advanced_particle_filter.filters.particle import BootstrapParticleFilter
from advanced_particle_filter.filters.base import FilterResult


# ============================================================================
# Assertion helpers
# ============================================================================

def assert_close(name: str, a, b, atol: float, rtol: float):
    """Check two arrays are close within tolerances."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_abs = np.max(np.abs(a - b))
    max_rel = np.max(np.abs(a - b) / (np.abs(b) + 1e-12))

    passed = np.allclose(a, b, atol=atol, rtol=rtol)
    if not passed:
        pytest.fail(
            f"{name}: FAILED\n"
            f"  max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}\n"
            f"  required: atol={atol:.0e}, rtol={rtol:.0e}"
        )


def assert_symmetric(P: np.ndarray, name: str, atol: float = 1e-10):
    """Check that matrix P is symmetric."""
    diff = np.max(np.abs(P - P.T))
    if diff > atol:
        pytest.fail(f"{name}: Not symmetric, max|P - P.T| = {diff:.3e}")


def assert_positive_definite(P: np.ndarray, name: str, tol: float = 1e-10):
    """Check that matrix P is positive definite via eigenvalues."""
    eigvals = np.linalg.eigvalsh(P)
    min_eig = np.min(eigvals)
    if min_eig < tol:
        pytest.fail(f"{name}: Not positive definite, min eigenvalue = {min_eig:.3e}")


def assert_covariance_valid(Ps: np.ndarray, name: str, atol: float = 1e-10):
    """Check symmetry and positive-definiteness for a sequence of covariances."""
    T = Ps.shape[0]
    for t in range(T):
        assert_symmetric(Ps[t], f"{name}[t={t}]", atol=atol)
        assert_positive_definite(Ps[t], f"{name}[t={t}]")


def check_convergence(Ps: np.ndarray, tol: float = 1e-6, window: int = 10) -> bool:
    """Check if covariance sequence has converged in the last `window` steps."""
    T = Ps.shape[0]
    if T < window + 1:
        return False
    for t in range(T - window, T):
        diff = np.max(np.abs(Ps[t] - Ps[t - 1]))
        if diff > tol:
            return False
    return True


def assert_convergence(Ps: np.ndarray, name: str, tol: float = 1e-6, window: int = 10):
    """Assert covariance sequence has converged."""
    T = Ps.shape[0]
    if T < window + 1:
        pytest.fail(f"{name}: Not enough timesteps ({T}) for convergence check (need {window + 1})")

    max_diffs = []
    for t in range(T - window, T):
        diff = np.max(np.abs(Ps[t] - Ps[t - 1]))
        max_diffs.append(diff)

    if max(max_diffs) > tol:
        pytest.fail(
            f"{name}: Covariance did not converge.\n"
            f"  Last {window} diffs: {[f'{d:.3e}' for d in max_diffs]}\n"
            f"  Required tol: {tol:.0e}"
        )


# ============================================================================
# Model / parameter factories
# ============================================================================

def make_stable_linear_params(nx: int, ny: int, seed: int = 0):
    """Generate a stable linear Gaussian SSM parameter set."""
    rng = np.random.default_rng(seed)

    # Make a stable A (spectral radius < 1)
    A = rng.normal(size=(nx, nx)) * 0.2
    A = A / max(1.2, np.max(np.abs(np.linalg.eigvals(A))))

    C = rng.normal(size=(ny, nx))
    Q = np.diag(rng.uniform(0.05, 0.2, size=nx))
    R = np.diag(rng.uniform(0.05, 0.2, size=ny))
    m0 = rng.normal(size=nx)
    P0 = np.eye(nx)

    return {
        "A": A, "C": C, "Q": Q, "R": R,
        "m0": m0, "P0": P0,
        "nx": nx, "ny": ny,
    }


def simulate_linear_ssm_np(T: int, A, Q, C, R, m0, P0, seed: int = 0):
    """Simulate from linear Gaussian SSM (NumPy).

    Returns:
        xs: [T+1, nx] states (x_0, ..., x_T)
        ys: [T, ny] observations (y_1, ..., y_T)
    """
    rng = np.random.default_rng(seed)
    nx = A.shape[0]
    ny = C.shape[0]

    Q_chol = np.linalg.cholesky(Q)
    R_chol = np.linalg.cholesky(R)
    P0_chol = np.linalg.cholesky(P0)

    xs = np.zeros((T + 1, nx))
    ys = np.zeros((T, ny))

    xs[0] = m0 + P0_chol @ rng.normal(size=nx)

    for t in range(T):
        xs[t + 1] = A @ xs[t] + Q_chol @ rng.normal(size=nx)
        ys[t] = C @ xs[t + 1] + R_chol @ rng.normal(size=ny)

    return xs, ys


def make_ssm_from_params(params) -> StateSpaceModel:
    """Build a StateSpaceModel from a flat parameter dict (linear Gaussian)."""
    A = params["A"]
    C = params["C"]
    Q = params["Q"]
    R = params["R"]
    m0 = params["m0"]
    P0 = params["P0"]
    nx = params["nx"]
    ny = params["ny"]

    return StateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=m0,
        initial_cov=P0,
        dynamics_mean=lambda x: x @ A.T,          # [N, nx] -> [N, nx]
        dynamics_cov=Q,
        dynamics_jacobian=lambda x: A,             # [nx] -> [nx, nx]
        obs_mean=lambda x: x @ C.T,               # [N, nx] -> [N, ny]
        obs_cov=R,
        obs_jacobian=lambda x: C,                  # [nx] -> [ny, nx]
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def random_walk_1d_params():
    """1D random walk parameters."""
    Qval, Rval = 0.2, 0.5
    return {
        "A": np.array([[1.0]]),
        "C": np.array([[1.0]]),
        "Q": np.array([[Qval]]),
        "R": np.array([[Rval]]),
        "m0": np.array([0.3]),
        "P0": np.array([[1.2]]),
        "Qval": Qval, "Rval": Rval,
        "nx": 1, "ny": 1,
    }


@pytest.fixture
def stable_2d_params():
    """Randomly generated stable 2D system."""
    return make_stable_linear_params(nx=2, ny=2, seed=42)


@pytest.fixture
def linear_multidim_params():
    """Higher-dimensional linear system (4D state, 2D obs)."""
    return make_stable_linear_params(nx=4, ny=2, seed=0)


# ============================================================================
# KF Validity Tests
# ============================================================================

class TestKFValidity:

    T = 100
    CONV_TOL = 1e-6
    CONV_WINDOW = 10

    def test_covariance_convergence_1d_random_walk(self, random_walk_1d_params):
        """KF covariance should converge to steady-state for 1D random walk."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=self.T, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=123,
        )

        kf = KalmanFilter()
        result = kf.filter(model, ys)
        assert_convergence(result.covariances, "KF P (1D RW)", tol=self.CONV_TOL, window=self.CONV_WINDOW)

    def test_covariance_convergence_2d(self, stable_2d_params):
        """KF covariance should converge for stable 2D system."""
        params = stable_2d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=self.T, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=456,
        )

        kf = KalmanFilter()
        result = kf.filter(model, ys)
        assert_convergence(result.covariances, "KF P (2D)", tol=self.CONV_TOL, window=self.CONV_WINDOW)

    def test_covariance_symmetry_positive_definite_1d(self, random_walk_1d_params):
        """KF covariances should remain symmetric and positive definite."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=789,
        )

        kf = KalmanFilter()
        result = kf.filter(model, ys)
        assert_covariance_valid(result.covariances, "KF P (1D)")

    def test_covariance_symmetry_positive_definite_multidim(self, linear_multidim_params):
        """KF covariances should remain symmetric and PD for multi-dim system."""
        params = linear_multidim_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=101,
        )

        kf = KalmanFilter()
        result = kf.filter(model, ys)
        assert_covariance_valid(result.covariances, "KF P (multidim)")


# ============================================================================
# EKF Tests
# ============================================================================

class TestEKF:

    def test_ekf_equals_kf_1d_random_walk(self, random_walk_1d_params):
        """EKF should equal KF exactly for linear 1D system."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=100, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=111,
        )

        kf = KalmanFilter()
        ekf = ExtendedKalmanFilter()
        res_kf = kf.filter(model, ys)
        res_ekf = ekf.filter(model, ys)

        assert_close("EKF mean vs KF (1D)", res_ekf.means, res_kf.means, atol=1e-9, rtol=1e-9)
        assert_close("EKF cov vs KF (1D)", res_ekf.covariances, res_kf.covariances, atol=1e-8, rtol=1e-8)

    def test_ekf_equals_kf_multidim(self, linear_multidim_params):
        """EKF should equal KF exactly for linear multi-dim system."""
        params = linear_multidim_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=100, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=222,
        )

        kf = KalmanFilter()
        ekf = ExtendedKalmanFilter()
        res_kf = kf.filter(model, ys)
        res_ekf = ekf.filter(model, ys)

        assert_close("EKF mean vs KF (multi)", res_ekf.means, res_kf.means, atol=1e-9, rtol=1e-9)
        assert_close("EKF cov vs KF (multi)", res_ekf.covariances, res_kf.covariances, atol=1e-8, rtol=1e-8)

    def test_ekf_covariance_valid(self, linear_multidim_params):
        """EKF covariances should remain symmetric and PD."""
        params = linear_multidim_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=333,
        )

        ekf = ExtendedKalmanFilter()
        result = ekf.filter(model, ys)
        assert_covariance_valid(result.covariances, "EKF P")


# ============================================================================
# UKF Tests
# ============================================================================

class TestUKF:

    def test_ukf_equals_kf_1d_random_walk(self, random_walk_1d_params):
        """UKF should equal KF for linear 1D system."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=100, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=444,
        )

        kf = KalmanFilter()
        ukf = UnscentedKalmanFilter()
        res_kf = kf.filter(model, ys)
        res_ukf = ukf.filter(model, ys)

        assert_close("UKF mean vs KF (1D)", res_ukf.means, res_kf.means, atol=1e-9, rtol=1e-9)
        assert_close("UKF cov vs KF (1D)", res_ukf.covariances, res_kf.covariances, atol=1e-8, rtol=1e-8)

    def test_ukf_equals_kf_multidim(self, linear_multidim_params):
        """UKF should equal KF for linear multi-dim system."""
        params = linear_multidim_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=100, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=555,
        )

        kf = KalmanFilter()
        ukf = UnscentedKalmanFilter()
        res_kf = kf.filter(model, ys)
        res_ukf = ukf.filter(model, ys)

        assert_close("UKF mean vs KF (multi)", res_ukf.means, res_kf.means, atol=1e-9, rtol=1e-9)
        assert_close("UKF cov vs KF (multi)", res_ukf.covariances, res_kf.covariances, atol=1e-8, rtol=1e-8)

    def test_ukf_covariance_valid(self, linear_multidim_params):
        """UKF covariances should remain symmetric and PD."""
        params = linear_multidim_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=666,
        )

        ukf = UnscentedKalmanFilter()
        result = ukf.filter(model, ys)
        assert_covariance_valid(result.covariances, "UKF P")


# ============================================================================
# PF Tests
# ============================================================================

class TestPF:

    NUM_PARTICLES = 10**5  # 100k (reduced from 1M for reasonable test runtime)
    MEAN_ATOL = 5e-2
    MEAN_RTOL = 5e-2
    COV_ATOL = 2e-1
    COV_RTOL = 2e-1

    @pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
    def test_pf_approx_kf_1d_random_walk(self, random_walk_1d_params, resampling_method):
        """PF should approximate KF for linear 1D system."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=100, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=777,
        )

        kf = KalmanFilter()
        res_kf = kf.filter(model, ys)

        pf = BootstrapParticleFilter(
            n_particles=self.NUM_PARTICLES,
            resample_method=resampling_method,
            seed=42,
        )
        res_pf = pf.filter(model, ys)

        # Compare means (skip t=0 initial which is identical by construction)
        assert_close(
            f"PF mean vs KF (1D, {resampling_method})",
            res_pf.means[1:], res_kf.means[1:],
            atol=self.MEAN_ATOL, rtol=self.MEAN_RTOL,
        )

    @pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
    def test_pf_approx_kf_multidim(self, linear_multidim_params, resampling_method):
        """PF should approximate KF for linear multi-dim system."""
        params = linear_multidim_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=100, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=888,
        )

        kf = KalmanFilter()
        res_kf = kf.filter(model, ys)

        pf = BootstrapParticleFilter(
            n_particles=self.NUM_PARTICLES,
            resample_method=resampling_method,
            seed=42,
        )
        res_pf = pf.filter(model, ys)

        assert_close(
            f"PF mean vs KF (multi, {resampling_method})",
            res_pf.means[1:], res_kf.means[1:],
            atol=self.MEAN_ATOL, rtol=self.MEAN_RTOL,
        )

    def test_pf_weight_validity(self, random_walk_1d_params):
        """PF weights should be valid (sum to 1, non-negative)."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=999,
        )

        pf = BootstrapParticleFilter(n_particles=1000, seed=42)
        result = pf.filter(model, ys, return_particles=True)

        weights = result.weights  # [T+1, N]
        assert weights is not None, "Weights not returned (pass return_particles=True)"

        # Check non-negative
        assert np.all(weights >= 0), "Weights contain negative values"

        # Check sum to 1 (approximately)
        weight_sums = np.sum(weights, axis=1)  # [T+1]
        assert np.allclose(weight_sums, 1.0, atol=1e-10), \
            f"Weights don't sum to 1: min={weight_sums.min()}, max={weight_sums.max()}"

    def test_pf_reproducibility(self, random_walk_1d_params):
        """PF with same seed should give identical results."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=1010,
        )

        pf1 = BootstrapParticleFilter(n_particles=1000, seed=42)
        res1 = pf1.filter(model, ys)

        pf2 = BootstrapParticleFilter(n_particles=1000, seed=42)
        res2 = pf2.filter(model, ys)

        # Should be exactly equal
        np.testing.assert_array_equal(res1.means, res2.means, err_msg="PF means differ with same seed")
        np.testing.assert_array_equal(
            res1.covariances, res2.covariances, err_msg="PF covs differ with same seed"
        )

        # Run with different seed — should differ
        pf3 = BootstrapParticleFilter(n_particles=1000, seed=99)
        res3 = pf3.filter(model, ys)
        assert not np.allclose(res1.means, res3.means), "PF results identical with different seeds"

    def test_pf_ess_reasonable(self, random_walk_1d_params):
        """PF ESS should stay above 1 and below N throughout."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=1111,
        )

        N = 1000
        pf = BootstrapParticleFilter(n_particles=N, seed=42)
        result = pf.filter(model, ys)

        assert result.ess is not None, "ESS not recorded"
        assert np.all(result.ess >= 1.0), f"ESS below 1: min={result.ess.min()}"
        assert np.all(result.ess <= N), f"ESS above N: max={result.ess.max()}"

    def test_pf_resampling_triggers(self, random_walk_1d_params):
        """PF with ESS-based resampling should trigger resampling at least once."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=100, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=1212,
        )

        pf = BootstrapParticleFilter(
            n_particles=200,  # small N to induce degeneracy
            resample_criterion="ess",
            ess_threshold=0.5,
            seed=42,
        )
        result = pf.filter(model, ys)

        assert result.resampled is not None, "Resampled flags not recorded"
        assert np.any(result.resampled), "No resampling occurred (expected at least once)"

    def test_pf_no_resample_mode(self, random_walk_1d_params):
        """PF with resample_criterion='never' should never resample."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=50, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=1313,
        )

        pf = BootstrapParticleFilter(
            n_particles=500,
            resample_criterion="never",
            seed=42,
        )
        result = pf.filter(model, ys)

        assert result.resampled is not None
        assert not np.any(result.resampled), "Resampling occurred despite 'never' criterion"


# ============================================================================
# Cross-filter consistency on shared data
# ============================================================================

class TestCrossFilterConsistency:
    """All Gaussian filters should agree on linear models."""

    def test_all_filters_agree_1d(self, random_walk_1d_params):
        """KF, EKF, UKF should produce identical results on 1D linear model."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=80, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=2000,
        )

        res_kf = KalmanFilter().filter(model, ys)
        res_ekf = ExtendedKalmanFilter().filter(model, ys)
        res_ukf = UnscentedKalmanFilter().filter(model, ys)

        assert_close("EKF vs KF means (1D)", res_ekf.means, res_kf.means, atol=1e-10, rtol=1e-10)
        assert_close("UKF vs KF means (1D)", res_ukf.means, res_kf.means, atol=1e-9, rtol=1e-9)
        assert_close("EKF vs KF covs (1D)", res_ekf.covariances, res_kf.covariances, atol=1e-10, rtol=1e-10)
        assert_close("UKF vs KF covs (1D)", res_ukf.covariances, res_kf.covariances, atol=1e-8, rtol=1e-8)

    def test_all_filters_agree_multidim(self, linear_multidim_params):
        """KF, EKF, UKF should produce identical results on 4D linear model."""
        params = linear_multidim_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=80, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=3000,
        )

        res_kf = KalmanFilter().filter(model, ys)
        res_ekf = ExtendedKalmanFilter().filter(model, ys)
        res_ukf = UnscentedKalmanFilter().filter(model, ys)

        assert_close("EKF vs KF means (4D)", res_ekf.means, res_kf.means, atol=1e-9, rtol=1e-9)
        assert_close("UKF vs KF means (4D)", res_ukf.means, res_kf.means, atol=1e-9, rtol=1e-9)
        assert_close("EKF vs KF covs (4D)", res_ekf.covariances, res_kf.covariances, atol=1e-8, rtol=1e-8)
        assert_close("UKF vs KF covs (4D)", res_ukf.covariances, res_kf.covariances, atol=1e-8, rtol=1e-8)

    def test_log_likelihood_consistency(self, random_walk_1d_params):
        """All Gaussian filters should report similar log-likelihoods on linear models."""
        params = random_walk_1d_params
        model = make_ssm_from_params(params)
        _, ys = simulate_linear_ssm_np(
            T=80, A=params["A"], Q=params["Q"], C=params["C"],
            R=params["R"], m0=params["m0"], P0=params["P0"], seed=4000,
        )

        res_kf = KalmanFilter().filter(model, ys)
        res_ekf = ExtendedKalmanFilter().filter(model, ys)
        res_ukf = UnscentedKalmanFilter().filter(model, ys)

        assert res_kf.log_likelihood is not None, "KF log-likelihood missing"
        assert res_ekf.log_likelihood is not None, "EKF log-likelihood missing"
        assert res_ukf.log_likelihood is not None, "UKF log-likelihood missing"

        assert_close(
            "EKF vs KF log-lik", res_ekf.log_likelihood, res_kf.log_likelihood,
            atol=1e-6, rtol=1e-6,
        )
        assert_close(
            "UKF vs KF log-lik", res_ukf.log_likelihood, res_kf.log_likelihood,
            atol=1e-5, rtol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
