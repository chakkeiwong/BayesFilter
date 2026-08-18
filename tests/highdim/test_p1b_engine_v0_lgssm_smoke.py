"""P1B smoke: engine v0 vs exact Kalman on small LGSSM (value only).

First rungs of the P1B ladder (n in {1, 2}, T = 8). The full ladder to
n=64 / T=120 runs from docs/benchmarks once these gates pass. Tolerances
declared BEFORE execution per the plan: |TT - Kalman| log-lik gap <= 5e-3
at n=1 and <= 2e-2 at n=2 for the smoke configuration (b=9, r=3, N=512,
s=2, half-width 8) — these are smoke gates for engine correctness, not the
ladder's scientific tolerance (declared in the P1B experiment plan).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

# Both smoke gates are currently EXPECTED FAILURES with a diagnosed
# mechanism: at tau=0 the naive single-sqrt-refit target contains |h_prev|
# kinks that a polynomial L2 fit cannot converge through (fit rms plateau
# ~2.2e-2 across degree/rank); tau>0 smooths but best achieved gap is
# 1.0e-2 vs the declared 5e-3. Repair (branch-decomposed target assembly
# + compression policy) is the standing P1B amendment. See
# docs/plans/bayesfilter-squared-tt-engine-p0-p1a-p1b-smoke-result-2026-08-16.md
_XFAIL_REASON = (
    "P1B smoke: naive sqrt-refit target assembly rejected at declared "
    "tolerance; branch-decomposition repair pending (result note 2026-08-16)"
)

from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DensityKernelAdapter,
    EngineConfig,
    run_value_filter,
)

DTYPE = tf.float64


def _mvn_log_density(x: tf.Tensor, mean: tf.Tensor, covariance: np.ndarray) -> tf.Tensor:
    d = int(covariance.shape[0])
    chol = np.linalg.cholesky(covariance)
    solve = tf.linalg.triangular_solve(
        tf.constant(chol, DTYPE), tf.transpose(x - mean), lower=True
    )
    quad = tf.reduce_sum(tf.square(solve), axis=0)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


def _kalman_log_likelihood(A, Q, H, R, m0, P0, ys) -> float:
    mean, cov = m0.copy(), P0.copy()
    total = 0.0
    for index, y in enumerate(ys):
        if index > 0:
            mean = A @ mean
            cov = A @ cov @ A.T + Q
        innovation_cov = H @ cov @ H.T + R
        innovation = y - H @ mean
        total += float(
            -0.5
            * (
                len(y) * np.log(2.0 * np.pi)
                + np.linalg.slogdet(innovation_cov)[1]
                + innovation @ np.linalg.solve(innovation_cov, innovation)
            )
        )
        gain = cov @ H.T @ np.linalg.inv(innovation_cov)
        mean = mean + gain @ innovation
        cov = cov - gain @ innovation_cov @ gain.T
    return total


def _lgssm_case(n: int, horizon: int, seed: int):
    rng = np.random.default_rng(seed)
    A = 0.7 * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / max(1, n - 1)
    Q = 0.4 * np.eye(n)
    H = np.eye(n)
    R = 0.5 * np.eye(n)
    m0 = np.zeros(n)
    P0 = np.eye(n)
    xs, ys = [], []
    x = rng.multivariate_normal(m0, P0)
    for _t in range(horizon):
        if _t > 0:
            x = A @ x + rng.multivariate_normal(np.zeros(n), Q)
        ys.append(H @ x + rng.multivariate_normal(np.zeros(n), R))
        xs.append(x)
    ys = np.stack(ys)
    adapter = DensityKernelAdapter(
        state_dim=n,
        transition_log_density=lambda xc, xp: _mvn_log_density(
            xc, tf.linalg.matvec(tf.constant(A, DTYPE), xp), Q
        ),
        observation_log_density=lambda xc, y: _mvn_log_density(
            tf.linalg.matvec(tf.constant(H, DTYPE), xc),
            tf.constant(y, DTYPE),
            R,
        ),
        initial_log_density=lambda xc: _mvn_log_density(
            xc, tf.constant(m0, DTYPE), P0
        ),
    )
    exact = _kalman_log_likelihood(A, Q, H, R, m0, P0, ys)
    return adapter, tf.constant(ys, DTYPE), exact


def test_p1b_smoke_n1_matches_kalman_branch_axis() -> None:
    """Branch-axis route (design note 2026-08-16) at the DECLARED 5e-3 gate."""
    from bayesfilter.highdim.squared_tt_engine_v0_tf import run_value_filter_branch_axis

    adapter, ys, exact = _lgssm_case(1, 8, 41)
    config = EngineConfig(
        basis_degree=16, rank=4, row_count=0, sweeps=3, ridge=1e-10,
        tau=1e-6, coordinate_half_width=4.0, seed=90011, quadrature_order=24,
    )
    value, diags = run_value_filter_branch_axis(adapter, ys, config)
    gap = abs(float(value.numpy()) - exact)
    assert np.isfinite(float(value.numpy()))
    assert not any(d["tie_flag"] for d in diags)
    assert gap <= 5e-3, f"n=1 gap {gap} vs declared smoke tolerance 5e-3 (exact {exact})"


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=True)
def test_p1b_smoke_n1_naive_route_rejected() -> None:
    """Historical record: the naive sqrt-refit route fails its declared gate."""
    adapter, ys, exact = _lgssm_case(1, 8, 41)
    config = EngineConfig(
        basis_degree=8, rank=3, row_count=512, sweeps=2, ridge=1e-10,
        tau=0.0, coordinate_half_width=8.0, seed=90011,
    )
    value, diags = run_value_filter(adapter, ys, config)
    gap = abs(float(value.numpy()) - exact)
    assert np.isfinite(float(value.numpy()))
    assert not any(d["tie_flag"] for d in diags)
    assert gap <= 5e-3, f"n=1 gap {gap} vs declared smoke tolerance 5e-3 (exact {exact})"


def test_p1b_smoke_n2_matches_kalman_branch_axis() -> None:
    """Branch-axis route at the DECLARED 2e-2 gate (r=6 rung).

    Diagnostic-scale note: tensor-product quadrature rows are affordable at
    n=2 only; the P1B ladder uses scattered frozen rows (V2 discipline).
    """
    from bayesfilter.highdim.squared_tt_engine_v0_tf import run_value_filter_branch_axis

    adapter, ys, exact = _lgssm_case(2, 8, 42)
    config = EngineConfig(
        basis_degree=12, rank=6, row_count=0, sweeps=3, ridge=1e-10,
        tau=1e-6, coordinate_half_width=3.0, seed=90012, quadrature_order=14,
    )
    value, _diags = run_value_filter_branch_axis(adapter, ys, config)
    gap = abs(float(value.numpy()) - exact)
    assert np.isfinite(float(value.numpy()))
    assert gap <= 2e-2, f"n=2 gap {gap} vs declared smoke tolerance 2e-2 (exact {exact})"
