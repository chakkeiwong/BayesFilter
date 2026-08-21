"""I-P2-1 (FD gate) for the full-path manual adjoint score engine.

Shift-parametrized LGSSM family: theta is the transition mean shift
(p = n), so the adapter VJPs are closed-form and the full chain
(retained evaluator -> branch factor -> ALS replay -> normalizer ->
retention, in reverse) is exercised end-to-end. Gate: adjoint gradient
vs centered FD of the SAME value program, relative error <= 1e-6
(program is smooth here; FD step 1e-5).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.squared_tt_adjoint_engine_tf import run_adjoint_score_filter
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DensityKernelAdapter,
    EngineConfig,
    run_value_filter_branch_axis,
)

DTYPE = tf.float64


def _family(n: int, theta: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    a_matrix = 0.7 * np.eye(n)
    q_matrix = 0.4 * np.eye(n)
    r_matrix = 0.5 * np.eye(n)
    m0 = np.zeros(n)
    p0 = np.eye(n)
    q_inv = np.linalg.inv(q_matrix)

    def mvn(x, mean, cov):
        d = int(cov.shape[0])
        chol = np.linalg.cholesky(cov)
        solve = tf.linalg.triangular_solve(
            tf.constant(chol, DTYPE), tf.transpose(x - mean), lower=True
        )
        quad = tf.reduce_sum(tf.square(solve), axis=0)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)

    shift = tf.constant(theta, DTYPE)
    adapter = DensityKernelAdapter(
        state_dim=n,
        transition_log_density=lambda xc, xp: mvn(
            xc, tf.linalg.matvec(tf.constant(a_matrix, DTYPE), xp) + shift, q_matrix
        ),
        observation_log_density=lambda xc, y: mvn(xc, tf.convert_to_tensor(y, DTYPE), r_matrix),
        initial_log_density=lambda xc: mvn(xc, tf.constant(m0, DTYPE), p0),
    )

    def transition_vjp(xc, xp, cot):
        residual = (
            xc - tf.linalg.matvec(tf.constant(a_matrix, DTYPE), xp) - shift
        )
        rows = tf.einsum("nd,de->ne", residual, tf.constant(q_inv, DTYPE))
        return tf.einsum("n,ne->e", cot, rows)

    def observation_vjp(xc, y, cot):
        return tf.zeros([n], DTYPE)

    def initial_vjp(xc, cot):
        return tf.zeros([n], DTYPE)

    # simulate observations at theta=0 truth (data fixed, not theta-dependent)
    x = rng.multivariate_normal(m0, p0)
    ys = []
    for t in range(4):
        if t > 0:
            x = a_matrix @ x + rng.multivariate_normal(np.zeros(n), q_matrix)
        ys.append(x + rng.multivariate_normal(np.zeros(n), r_matrix))
    return adapter, transition_vjp, observation_vjp, initial_vjp, tf.constant(np.stack(ys), DTYPE)


def _config(n: int) -> EngineConfig:
    # rank must keep the retained Gram well-conditioned (score-path claim
    # gate): at n=2 the near-Gaussian filtered law has Gram rank ~2, and a
    # rank-3 factor's near-null Cholesky column rotates erratically with
    # theta (lambda3/lambda1 ~ 1e-13), injecting O(fit-error) value wiggles
    # that invalidate FD (diagnosed 2026-08-17; floor-independent, vanishes
    # at rank 2). Rank selection vs Gram conditioning is a tuning-procedure
    # gate; here we test the gradient in the well-conditioned regime.
    # quadrature_order must exceed basis_degree: with degree-10 bases (11
    # functions per axis), 8 Gauss nodes per axis leave the per-core ALS
    # design column-rank-deficient (cond(N) ~ 3e14, ridge-dominated), and
    # the two independent derivative chains legitimately disagree at ~1e-5
    # through the near-null space. At order 12 cond(N) ~ 1e5 and both
    # gates pass (diagnosed 2026-08-17; adjoint code unchanged).
    return EngineConfig(
        basis_degree=10, rank=3 if n == 1 else 2, row_count=0, sweeps=2,
        ridge=1e-10, tau=1e-6, coordinate_half_width=4.0, seed=90021,
        quadrature_order=14 if n == 1 else 12,
    )


def test_i_p2_1_adjoint_gradient_matches_fd_n1() -> None:
    n = 1
    theta0 = np.zeros(n)
    adapter, tvjp, ovjp, ivjp, ys = _family(n, theta0, 61)
    config = _config(n)
    value, grad = run_adjoint_score_filter(
        adapter, ys, config,
        transition_vjp=tvjp, observation_vjp=ovjp, initial_vjp=ivjp,
        parameter_dim=n,
    )
    # value must equal the value-only engine exactly (same program)
    value_only, _d = run_value_filter_branch_axis(adapter, ys, config)
    assert abs(float((value - value_only).numpy())) <= 1e-12

    step = 1e-5
    fd = np.zeros(n)
    for k in range(n):
        theta_plus = theta0.copy(); theta_plus[k] += step
        theta_minus = theta0.copy(); theta_minus[k] -= step
        ap, *_rest, _ys = _family(n, theta_plus, 61)
        am, *_rest2, _ys2 = _family(n, theta_minus, 61)
        vp, _ = run_value_filter_branch_axis(ap, ys, config)
        vm, _ = run_value_filter_branch_axis(am, ys, config)
        fd[k] = (float(vp.numpy()) - float(vm.numpy())) / (2.0 * step)
    rel = np.linalg.norm(grad.numpy() - fd) / max(1.0, np.linalg.norm(fd))
    assert rel <= 1e-6, f"adjoint vs FD rel {rel}: adjoint {grad.numpy()} fd {fd}"


def test_i_p2_1_adjoint_gradient_matches_fd_n2() -> None:
    n = 2
    theta0 = np.zeros(n)
    adapter, tvjp, ovjp, ivjp, ys = _family(n, theta0, 62)
    config = _config(n)
    value, grad = run_adjoint_score_filter(
        adapter, ys, config,
        transition_vjp=tvjp, observation_vjp=ovjp, initial_vjp=ivjp,
        parameter_dim=n,
    )
    step = 1e-5
    fd = np.zeros(n)
    for k in range(n):
        theta_plus = theta0.copy(); theta_plus[k] += step
        theta_minus = theta0.copy(); theta_minus[k] -= step
        ap, *_r, _y = _family(n, theta_plus, 62)
        am, *_r2, _y2 = _family(n, theta_minus, 62)
        vp, _ = run_value_filter_branch_axis(ap, ys, config)
        vm, _ = run_value_filter_branch_axis(am, ys, config)
        fd[k] = (float(vp.numpy()) - float(vm.numpy())) / (2.0 * step)
    rel = np.linalg.norm(grad.numpy() - fd) / max(1.0, np.linalg.norm(fd))
    assert rel <= 1e-6, f"adjoint vs FD rel {rel}: adjoint {grad.numpy()} fd {fd}"
