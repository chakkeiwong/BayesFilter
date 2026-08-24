"""U-RET-1, conversion closure, and the LGSSM oracle gate (C2 ladder 2-4).

Spec: docs/plans/bayesfilter-c2-gaussian-reference-derivation-note-2026-08-24.md
(REVIEWED). Gate definition per review finding F7: compare the
defensive-corrected sum sum_t(log Zhat_t - log(1+tau_t)) against the
exact Kalman log-likelihood; each telescoped increment carries exactly
its own (1+tau_t) factor, so the subtraction is exact closed form.
Degree-0 runs certify the conversion display: a degree-0 basis can only
represent constants, so the gate passes only if the whitened target is
constant (ch38 degree-collapse proposition).

NumPy appears as diagnostic reference machinery only (fixture, Kalman,
quadrature). CPU-only diagnostic: GPU intentionally hidden via
CUDA_VISIBLE_DEVICES=-1 before the TensorFlow import (float64 eager
correctness gates; GPU claim runs belong to the XLA lane).
"""

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.bases import HermiteBasis1D
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.retained_quadratic_form_tf import (
    retained_quadratic_form_from_squared_tt,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    TAU_MIN,
    _hermite_product_basis,
    run_value_filter_branch_axis_gaussian,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DensityKernelAdapter,
    EngineConfig,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64
ORACLE_GATE = 1e-8


def _mvn_log_density(x: tf.Tensor, mean: tf.Tensor, cov: np.ndarray) -> tf.Tensor:
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    chol = np.linalg.cholesky(cov)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    diff = x - mean
    solved = tf.transpose(
        tf.linalg.triangular_solve(
            tf.constant(chol, DTYPE), tf.transpose(diff), lower=True
        )
    )
    quad = tf.reduce_sum(tf.square(solved), axis=-1)
    return -0.5 * (n * math.log(2.0 * math.pi) + logdet + quad)


def _lgssm_fixture(n: int, horizon: int, seed: int):
    """attempt04 fixture family: A=0.7I+0.1 randn/(n-1), Q=0.4I, H=I, R=0.5I."""

    rng = np.random.default_rng(seed)
    A = 0.7 * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / max(1, n - 1)
    Q = 0.4 * np.eye(n)
    H = np.eye(n)
    R = 0.5 * np.eye(n)
    m0 = np.zeros(n)
    P0 = np.eye(n)
    x = rng.multivariate_normal(m0, P0)
    ys = []
    for t in range(horizon):
        if t > 0:
            x = A @ x + rng.multivariate_normal(np.zeros(n), Q)
        ys.append(H @ x + rng.multivariate_normal(np.zeros(n), R))
    ys = np.stack(ys)
    mean, cov = m0.copy(), P0.copy()
    steps = []
    for i, y in enumerate(ys):
        if i > 0:
            mean = A @ mean
            cov = A @ cov @ A.T + Q
        S = H @ cov @ H.T + R
        innov = y - H @ mean
        steps.append(
            float(
                -0.5
                * (
                    n * math.log(2.0 * math.pi)
                    + np.linalg.slogdet(S)[1]
                    + innov @ np.linalg.solve(S, innov)
                )
            )
        )
        K = cov @ H.T @ np.linalg.inv(S)
        mean = mean + K @ innov
        cov = cov - K @ S @ K.T
    adapter = DensityKernelAdapter(
        state_dim=n,
        transition_log_density=lambda xc, xp: _mvn_log_density(
            xc, tf.linalg.matvec(tf.constant(A, DTYPE), xp), Q
        ),
        observation_log_density=lambda xc, y: _mvn_log_density(
            xc, tf.convert_to_tensor(y, DTYPE), R
        ),
        initial_log_density=lambda xc: _mvn_log_density(
            xc, tf.constant(m0, DTYPE), P0
        ),
    )
    model = {"A": A, "Q": Q, "H": H, "R": R, "m0": m0, "P0": P0}
    return adapter, tf.constant(ys, DTYPE), steps, model


def _exact_hint_factories(model):
    """Companion Kalman supplying exact frozen hints (M2-JOINT contract).

    initial_moment_hint(y0): posterior moments of x_0 | y_0.
    predictive_moment_hint(t, y_t): joint filtered moments of
    (x_t, x_{t-1}) | y_{1:t} with the lag-one cross-covariance —
    the paper's Section 5.2 joint bridging object, exact for LGSSM.
    """

    A, Q, H, R = model["A"], model["Q"], model["H"], model["R"]
    state = {"mean": None, "cov": None, "t": None}

    def initial_moment_hint(y0):
        S = H @ model["P0"] @ H.T + R
        K = model["P0"] @ H.T @ np.linalg.inv(S)
        y = np.asarray(y0)
        mean = model["m0"] + K @ (y - H @ model["m0"])
        cov = model["P0"] - K @ S @ K.T
        state["mean"], state["cov"], state["t"] = mean, cov, 0
        return tf.constant(mean, DTYPE), tf.constant(cov, DTYPE)

    def predictive_moment_hint(t, y_t):
        assert state["t"] is not None, "initial hint must be requested first"
        assert t == state["t"] + 1, "hints must be requested in step order"
        m_prev, P_prev = state["mean"], state["cov"]
        mean = A @ m_prev
        cov = A @ P_prev @ A.T + Q
        cross_pred = P_prev @ A.T
        S = H @ cov @ H.T + R
        K = cov @ H.T @ np.linalg.inv(S)
        y = np.asarray(y_t)
        f_mean = mean + K @ (y - H @ mean)
        f_cov = cov - K @ S @ K.T
        J = cross_pred @ H.T @ np.linalg.inv(S)
        p_mean = m_prev + J @ (y - H @ mean)
        p_cov = P_prev - J @ S @ J.T
        cross_f = cross_pred - J @ S @ K.T
        joint_mean = np.concatenate([f_mean, p_mean])
        joint_cov = np.block([[f_cov, cross_f.T], [cross_f, p_cov]])
        state["mean"], state["cov"], state["t"] = f_mean, f_cov, t
        return tf.constant(joint_mean, DTYPE), tf.constant(joint_cov, DTYPE)

    return initial_moment_hint, predictive_moment_hint


def _defensive_corrected(value: tf.Tensor, diagnostics: list[dict]) -> float:
    return float(value.numpy()) - sum(
        math.log1p(d["tau_t"]) for d in diagnostics
    )


def test_u_ret_1_retention_matches_dense_quadrature() -> None:
    """CD22 Prop 2 with M_k = I: Gram-chain retention vs quadrature."""

    degree, rank = 3, 2
    basis2 = _hermite_product_basis(2, degree)
    basis1 = _hermite_product_basis(1, degree)
    ell = degree + 1
    rng = np.random.default_rng(7)
    c1 = tf.constant(rng.standard_normal((1, ell, rank)), DTYPE)
    c2 = tf.constant(rng.standard_normal((rank, ell, 1)), DTYPE)
    form = retained_quadratic_form_from_squared_tt(
        (TTCore(c1), TTCore(c2)),
        basis2,
        split_index=1,
        tau=0.0,
        prefix_basis=basis1,
        coordinate_map=AffineCoordinateMap(
            offset=tf.zeros([1], DTYPE), matrix=tf.eye(1, dtype=DTYPE)
        ),
    )
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / math.sqrt(2.0 * math.pi)
    hermite = HermiteBasis1D(max_degree=degree)
    phi_nodes = hermite.evaluate(tf.constant(nodes, DTYPE)).numpy()

    points = np.linspace(-3.0, 3.0, 11)
    phi_points = hermite.evaluate(tf.constant(points, DTYPE)).numpy()
    # h(u1, u2) = sum_a [phi(u1) @ C1[0]]_a [C2[a] @ phi(u2)]
    left = phi_points @ c1.numpy()[0]            # [P, rank]
    right = np.einsum("aj,nj->an", c2.numpy()[:, :, 0], phi_nodes)  # [rank, N]
    h_grid = left @ right                        # [P, N]
    marginal_quad = (h_grid**2) @ weights        # [P]
    marginal_form = form.quadratic_form_values(
        tf.constant(points[:, None], DTYPE)
    ).numpy()
    assert np.max(np.abs(marginal_form - marginal_quad)) < 1e-12

    z_quad = float(
        weights @ ((phi_nodes @ c1.numpy()[0] @ c2.numpy()[:, :, 0] @ phi_nodes.T) ** 2) @ weights
    )
    assert abs(float(form.z_complete_ref.numpy()) - z_quad) < 1e-12


def _run_oracle(n: int, horizon: int, degree: int, rank: int, rows: int, seed: int):
    adapter, ys, kalman_steps, model = _lgssm_fixture(n, horizon, seed)
    initial_hint, predictive_hint = _exact_hint_factories(model)
    config = EngineConfig(
        basis_degree=degree,
        rank=rank,
        row_count=rows,
        sweeps=3,
        ridge=1e-10,
        tau=1e-6,
        coordinate_half_width=3.0,
        seed=93000 + 10 * n + rank,
        row_design="sobol",
    )
    value, diagnostics = run_value_filter_branch_axis_gaussian(
        adapter,
        ys,
        config,
        predictive_moment_hint=predictive_hint,
        initial_moment_hint=initial_hint,
    )
    gap = abs(_defensive_corrected(value, diagnostics) - sum(kalman_steps))
    return gap, diagnostics


def test_conversion_closure_degree0_n1_t3() -> None:
    """Degree-0 pass is possible only if the whitened target is constant."""

    gap, diagnostics = _run_oracle(n=1, horizon=3, degree=0, rank=1, rows=512, seed=101)
    assert all(d["tau_t"] == TAU_MIN for d in diagnostics)
    assert gap < ORACLE_GATE, f"conversion closure gap {gap:.3e}"


def test_oracle_gate_degree0_n2_t120() -> None:
    gap, _ = _run_oracle(n=2, horizon=120, degree=0, rank=1, rows=512, seed=44)
    assert gap < ORACLE_GATE, f"oracle gate (l=1, r=1) gap {gap:.3e}"


def test_oracle_gate_degree12_rank6_n2_t120() -> None:
    """Fit-of-constant conditioning rung: gate must not degrade."""

    gap, _ = _run_oracle(n=2, horizon=120, degree=12, rank=6, rows=2048, seed=44)
    assert gap < ORACLE_GATE, f"oracle gate (l=13, r=6) gap {gap:.3e}"
