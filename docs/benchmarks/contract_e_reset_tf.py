"""TensorFlow helpers for Contract E reset diagnostics.

These helpers are local diagnostic utilities for the LEDH-PFPF-OT Contract E
experiments.  They separate the reset factorization from the full filter so
route audits can check whether a reset path relies on eigensystem coordinates.
"""

from __future__ import annotations

from typing import Any


def contract_e_cholesky_ridge_reset(
    tf: Any,
    *,
    post_flow: Any,
    weights: Any,
    matrix: Any,
    residual_noise: Any,
    rho: Any,
    ridge_rel: Any,
    ridge_abs: Any,
    ridge_escalation: Any,
    ridge_max_attempts: Any,
) -> dict[str, Any]:
    """Apply the opt-in Cholesky-ridge Contract E reset.

    The factorization is intentionally ridge-regularized and full-rank.  It
    restores the ridged covariance contract exactly, while the unridged target
    covariance residual is reported as a diagnostic.
    """

    selection = contract_e_minimal_stabilizing_cholesky_ridge(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=rho,
        ridge_rel=ridge_rel,
        ridge_abs=ridge_abs,
        ridge_escalation=ridge_escalation,
        ridge_max_attempts=ridge_max_attempts,
    )
    value = _contract_e_cholesky_ridge_fixed_ridge_forward(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=rho,
        ridge=selection["ridge"],
    )

    return {
        "particles": value["y_star"],
        "max_covariance_relative_residual": tf.reduce_max(value["cov_residual"]),
        "max_mean_linf_residual": tf.reduce_max(value["mean_residual"]),
        "min_gap_diagnostic": tf.reduce_min(value["min_residual_chol_diag"]),
        "max_condition_proxy": tf.reduce_max(value["condition_proxy"]),
        "min_tilde_positive_diagnostic": tf.reduce_min(value["min_tilde_chol_diag"]),
        "min_rank_margin_diagnostic": tf.reduce_min(value["min_residual_chol_diag"]),
        "min_target_positive_diagnostic": tf.reduce_min(value["min_target_chol_diag"]),
        "max_realized_ridge": tf.reduce_max(selection["ridge"]),
        "ridge_attempts_used": tf.reduce_max(selection["attempts_used"]),
        "ridge_failure": tf.logical_not(tf.reduce_all(value["ok_per_batch"])),
        "realized_ridge_by_batch": selection["ridge"],
        "base_ridge_by_batch": selection["base_ridge"],
        "ridge_attempts_by_batch": selection["attempts_used"],
        "ridge_ok_by_batch": value["ok_per_batch"],
    }


def contract_e_minimal_stabilizing_cholesky_ridge(
    tf: Any,
    *,
    post_flow: Any,
    weights: Any,
    matrix: Any,
    residual_noise: Any,
    rho: Any,
    ridge_rel: Any,
    ridge_abs: Any,
    ridge_escalation: Any,
    ridge_max_attempts: Any,
) -> dict[str, Any]:
    """Choose the smallest per-batch ridge on the configured stability ladder.

    This is a numerical chart-selection rule, not a statistical tuning rule.
    It starts at ``max(ridge_abs, ridge_rel * trace(target_cov) / d)`` and
    escalates only batches whose Cholesky chart remains invalid.  The returned
    ridge should then be treated as a fixed local chart for derivative checks.
    """

    dtype = post_flow.dtype
    state_dim = tf.shape(post_flow)[2]
    state_dim_float = tf.cast(state_dim, dtype)

    def sym(matrix_value: Any) -> Any:
        return 0.5 * (matrix_value + tf.linalg.matrix_transpose(matrix_value))

    mean = tf.reduce_sum(weights[:, :, None] * post_flow, axis=1)
    centered = post_flow - mean[:, None, :]
    target_cov = tf.reduce_sum(
        weights[:, :, None, None] * centered[:, :, :, None] * centered[:, :, None, :],
        axis=1,
    )
    target_scale = tf.linalg.trace(sym(target_cov)) / tf.maximum(
        state_dim_float,
        tf.constant(1.0, dtype=dtype),
    )
    base_ridge = tf.maximum(tf.cast(ridge_abs, dtype), tf.cast(ridge_rel, dtype) * target_scale)
    escalation = tf.cast(ridge_escalation, dtype)
    max_attempts = tf.cast(ridge_max_attempts, tf.int32)

    def evaluate_ok(ridge_value: Any) -> Any:
        value = _contract_e_cholesky_ridge_fixed_ridge_forward(
            tf,
            post_flow=post_flow,
            weights=weights,
            matrix=matrix,
            residual_noise=residual_noise,
            rho=rho,
            ridge=ridge_value,
        )
        return value["ok_per_batch"]

    ok_per_batch = evaluate_ok(base_ridge)
    attempts = tf.ones_like(tf.cast(ok_per_batch, tf.int32), dtype=tf.int32)

    def cond(ridge_value: Any, attempts_used: Any, current_ok: Any) -> Any:
        active = tf.logical_and(tf.logical_not(current_ok), attempts_used < max_attempts)
        return tf.reduce_any(active)

    def body(ridge_value: Any, attempts_used: Any, current_ok: Any) -> tuple[Any, Any, Any]:
        active = tf.logical_and(tf.logical_not(current_ok), attempts_used < max_attempts)
        next_ridge = tf.where(active, ridge_value * escalation, ridge_value)
        next_attempts = tf.where(active, attempts_used + 1, attempts_used)
        next_ok = evaluate_ok(next_ridge)
        return next_ridge, next_attempts, tf.where(active, next_ok, current_ok)

    ridge, attempts, ok_per_batch = tf.while_loop(
        cond,
        body,
        loop_vars=(base_ridge, attempts, ok_per_batch),
        parallel_iterations=1,
        maximum_iterations=max_attempts,
    )
    return {
        "ridge": ridge,
        "base_ridge": base_ridge,
        "attempts_used": attempts,
        "ok_per_batch": ok_per_batch,
        "ridge_failure": tf.logical_not(tf.reduce_all(ok_per_batch)),
    }


def contract_e_cholesky_ridge_reset_fixed_ridge(
    tf: Any,
    *,
    post_flow: Any,
    weights: Any,
    matrix: Any,
    residual_noise: Any,
    rho: Any,
    ridge: Any,
    return_aux: bool = False,
) -> dict[str, Any]:
    """Apply the Cholesky-ridge reset on a fixed-ridge smooth chart."""

    value = _contract_e_cholesky_ridge_fixed_ridge_forward(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=rho,
        ridge=ridge,
    )
    result = {
        "particles": value["y_star"],
        "max_covariance_relative_residual": tf.reduce_max(value["cov_residual"]),
        "max_mean_linf_residual": tf.reduce_max(value["mean_residual"]),
        "min_gap_diagnostic": tf.reduce_min(value["min_residual_chol_diag"]),
        "max_condition_proxy": tf.reduce_max(value["condition_proxy"]),
        "min_tilde_positive_diagnostic": tf.reduce_min(value["min_tilde_chol_diag"]),
        "min_rank_margin_diagnostic": tf.reduce_min(value["min_residual_chol_diag"]),
        "min_target_positive_diagnostic": tf.reduce_min(value["min_target_chol_diag"]),
        "max_realized_ridge": tf.reduce_max(value["ridge"]),
        "ridge_attempts_used": tf.constant(1, dtype=tf.int32),
        "ridge_failure": tf.logical_not(value["ok"]),
    }
    if return_aux:
        result["aux"] = value
    return result


def contract_e_cholesky_ridge_reset_fixed_ridge_vjp(
    tf: Any,
    *,
    post_flow: Any,
    weights: Any,
    matrix: Any,
    residual_noise: Any,
    rho: Any,
    ridge: Any,
    upstream_particles: Any,
) -> dict[str, Any]:
    """Manual VJP for the fixed-ridge Cholesky reset local chart."""

    aux = contract_e_cholesky_ridge_reset_fixed_ridge(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=rho,
        ridge=ridge,
        return_aux=True,
    )["aux"]

    dtype = post_flow.dtype
    batch_size = tf.shape(post_flow)[0]
    num_particles = tf.shape(post_flow)[1]
    num_particles_float = tf.cast(num_particles, dtype)
    sqrt_rho = tf.sqrt(tf.cast(rho, dtype))

    def sym(matrix_value: Any) -> Any:
        return 0.5 * (matrix_value + tf.linalg.matrix_transpose(matrix_value))

    def apply_batch_linear_rows(points: Any, operator: Any) -> Any:
        return tf.reduce_sum(points[:, :, None, :] * operator[:, None, :, :], axis=-1)

    def apply_batch_linear_rows_vjp(points: Any, operator: Any, upstream: Any) -> tuple[Any, Any]:
        points_bar = apply_batch_linear_rows(upstream, tf.linalg.matrix_transpose(operator))
        operator_bar = tf.einsum("bni,bnj->bij", upstream, points)
        return points_bar, operator_bar

    def centered_uniform_vjp(upstream_centered: Any) -> Any:
        return upstream_centered - tf.reduce_mean(upstream_centered, axis=1, keepdims=True)

    def uniform_moments_vjp(points: Any, mean_bar: Any, covariance_bar: Any) -> Any:
        centered = points - tf.reduce_mean(points, axis=1, keepdims=True)
        covariance_bar = sym(covariance_bar)
        mean_part = mean_bar[:, None, :] / num_particles_float
        covariance_part = (2.0 / num_particles_float) * apply_batch_linear_rows(
            centered,
            covariance_bar,
        )
        return mean_part + covariance_part

    def weighted_moments_vjp(
        points: Any,
        local_weights: Any,
        mean: Any,
        mean_bar: Any,
        covariance_bar: Any,
    ) -> tuple[Any, Any]:
        centered = points - mean[:, None, :]
        covariance_bar = sym(covariance_bar)
        points_bar = local_weights[:, :, None] * (
            mean_bar[:, None, :] + 2.0 * apply_batch_linear_rows(centered, covariance_bar)
        )
        weights_bar = tf.reduce_sum(points * mean_bar[:, None, :], axis=2)
        weights_bar = weights_bar + tf.einsum(
            "bni,bij,bnj->bn",
            centered,
            covariance_bar,
            centered,
        )
        return points_bar, weights_bar

    def cholesky_vjp(chol: Any, chol_bar: Any) -> Any:
        product = tf.linalg.matmul(tf.linalg.matrix_transpose(chol), chol_bar)
        lower = tf.linalg.band_part(product, -1, 0)
        diag = tf.linalg.diag(tf.linalg.diag_part(lower))
        phi = lower - 0.5 * diag
        left = tf.linalg.triangular_solve(chol, phi, adjoint=True)
        solved_t = tf.linalg.triangular_solve(
            chol,
            tf.linalg.matrix_transpose(left),
            adjoint=True,
        )
        return sym(tf.linalg.matrix_transpose(solved_t))

    def triangular_solve_adjoint_vjp(chol: Any, rhs: Any, solution: Any, solution_bar: Any) -> tuple[Any, Any]:
        rhs_bar = tf.linalg.triangular_solve(chol, solution_bar)
        solve_matrix_bar = -tf.linalg.matmul(rhs_bar, solution, transpose_b=True)
        chol_bar = tf.linalg.matrix_transpose(solve_matrix_bar)
        return chol_bar, tf.linalg.matrix_transpose(rhs_bar)

    def matmul_vjp(left: Any, right: Any, upstream: Any) -> tuple[Any, Any]:
        left_bar = tf.linalg.matmul(upstream, right, transpose_b=True)
        right_bar = tf.linalg.matmul(left, upstream, transpose_a=True)
        return left_bar, right_bar

    post_flow_bar = tf.zeros_like(post_flow)
    weights_bar = tf.zeros_like(weights)
    matrix_bar = tf.zeros_like(matrix)
    residual_noise_bar = tf.zeros_like(residual_noise)

    target_mean_bar = tf.reduce_sum(upstream_particles, axis=1)
    centered_tilde_bar, affine_bar = apply_batch_linear_rows_vjp(
        aux["centered_tilde"],
        aux["affine"],
        upstream_particles,
    )
    y_tilde_bar = centered_uniform_vjp(centered_tilde_bar)

    solved_bar = tf.linalg.matrix_transpose(affine_bar)
    tilde_chol_bar, target_chol_bar = triangular_solve_adjoint_vjp(
        aux["tilde_chol"],
        tf.linalg.matrix_transpose(aux["target_chol"]),
        aux["solved"],
        solved_bar,
    )

    target_cov_from_affine_bar = cholesky_vjp(aux["target_chol"], target_chol_bar)
    tilde_cov_bar = cholesky_vjp(aux["tilde_chol"], tilde_chol_bar)
    y_tilde_bar = y_tilde_bar + uniform_moments_vjp(
        aux["y_tilde"],
        tf.zeros([batch_size, tf.shape(post_flow)[2]], dtype=dtype),
        tilde_cov_bar,
    )

    y_plus_from_tilde_bar = y_tilde_bar
    xi_bar, b_matrix_bar = apply_batch_linear_rows_vjp(aux["xi"], aux["b_matrix"], y_tilde_bar)
    residual_chol_bar = sqrt_rho * b_matrix_bar
    gap_bar = cholesky_vjp(aux["residual_chol"], residual_chol_bar)
    ridge_bar = (
        tf.linalg.trace(gap_bar)
        + tf.linalg.trace(target_cov_from_affine_bar)
        + tf.linalg.trace(tilde_cov_bar)
    )
    target_cov_bar = target_cov_from_affine_bar + sym(gap_bar)
    plus_cov_bar = -sym(gap_bar)

    y_plus_bar = y_plus_from_tilde_bar + uniform_moments_vjp(
        aux["y_plus"],
        tf.zeros([batch_size, tf.shape(post_flow)[2]], dtype=dtype),
        plus_cov_bar,
    )
    local_matrix_bar, post_flow_from_y_plus_bar = matmul_vjp(matrix, post_flow, y_plus_bar)
    matrix_bar = matrix_bar + local_matrix_bar
    post_flow_bar = post_flow_bar + post_flow_from_y_plus_bar

    local_post_flow_bar, local_weights_bar = weighted_moments_vjp(
        post_flow,
        weights,
        aux["target_mean"],
        target_mean_bar,
        target_cov_bar,
    )
    post_flow_bar = post_flow_bar + local_post_flow_bar
    weights_bar = weights_bar + local_weights_bar

    xi_scale = tf.sqrt(num_particles_float / tf.cast(num_particles - 1, dtype))
    residual_noise_bar = residual_noise_bar + xi_scale * centered_uniform_vjp(xi_bar)

    return {
        "post_flow": post_flow_bar,
        "weights": weights_bar,
        "matrix": matrix_bar,
        "residual_noise": residual_noise_bar,
        "ridge": ridge_bar,
    }


def _contract_e_cholesky_ridge_fixed_ridge_forward(
    tf: Any,
    *,
    post_flow: Any,
    weights: Any,
    matrix: Any,
    residual_noise: Any,
    rho: Any,
    ridge: Any,
) -> dict[str, Any]:
    dtype = post_flow.dtype
    num_particles = tf.shape(post_flow)[1]
    state_dim = tf.shape(post_flow)[2]
    ridge = tf.cast(ridge, dtype)
    ridge = ridge + tf.zeros([tf.shape(post_flow)[0]], dtype=dtype)

    def sym(matrix_value: Any) -> Any:
        return 0.5 * (matrix_value + tf.linalg.matrix_transpose(matrix_value))

    def weighted_moments(points: Any, local_weights: Any) -> tuple[Any, Any]:
        mean = tf.reduce_sum(local_weights[:, :, None] * points, axis=1)
        centered = points - mean[:, None, :]
        covariance = tf.reduce_sum(
            local_weights[:, :, None, None]
            * centered[:, :, :, None]
            * centered[:, :, None, :],
            axis=1,
        )
        return mean, sym(covariance)

    def uniform_moments(points: Any) -> tuple[Any, Any]:
        count = tf.cast(tf.shape(points)[1], dtype)
        uniform_weights = tf.fill([tf.shape(points)[0], tf.shape(points)[1]], 1.0 / count)
        return weighted_moments(points, uniform_weights)

    def apply_batch_linear_rows(points: Any, operator: Any) -> Any:
        return tf.reduce_sum(points[:, :, None, :] * operator[:, None, :, :], axis=-1)

    eye = tf.eye(state_dim, dtype=dtype)[None, :, :]
    ridge_eye = ridge[:, None, None] * eye
    target_mean, target_cov = weighted_moments(post_flow, weights)
    y_plus = tf.linalg.matmul(matrix, post_flow)
    _plus_mean, plus_cov = uniform_moments(y_plus)
    gap = sym(target_cov - plus_cov)
    residual_chol = tf.linalg.cholesky(gap + ridge_eye)
    b_matrix = tf.sqrt(tf.cast(rho, dtype)) * residual_chol
    centered_noise = residual_noise - tf.reduce_mean(residual_noise, axis=1, keepdims=True)
    xi = tf.sqrt(tf.cast(num_particles, dtype) / tf.cast(num_particles - 1, dtype)) * centered_noise
    y_tilde = y_plus + apply_batch_linear_rows(xi, b_matrix)
    tilde_mean, tilde_cov = uniform_moments(y_tilde)
    centered_tilde = y_tilde - tilde_mean[:, None, :]
    target_chol = tf.linalg.cholesky(sym(target_cov) + ridge_eye)
    tilde_chol = tf.linalg.cholesky(sym(tilde_cov) + ridge_eye)
    solved = tf.linalg.triangular_solve(
        tilde_chol,
        tf.linalg.matrix_transpose(target_chol),
        adjoint=True,
    )
    affine = tf.linalg.matrix_transpose(solved)
    y_star = target_mean[:, None, :] + apply_batch_linear_rows(centered_tilde, affine)
    star_mean, star_cov = uniform_moments(y_star)
    cov_norm = tf.norm(target_cov, ord="fro", axis=[-2, -1])
    cov_residual = tf.norm(star_cov - target_cov, ord="fro", axis=[-2, -1]) / tf.maximum(
        cov_norm,
        tf.constant(1.0e-30, dtype=dtype),
    )
    mean_residual = tf.reduce_max(tf.abs(star_mean - target_mean), axis=1)
    min_residual_chol_diag = tf.reduce_min(tf.linalg.diag_part(residual_chol), axis=1)
    min_target_chol_diag = tf.reduce_min(tf.linalg.diag_part(target_chol), axis=1)
    min_tilde_chol_diag = tf.reduce_min(tf.linalg.diag_part(tilde_chol), axis=1)
    max_tilde_chol_diag = tf.reduce_max(tf.linalg.diag_part(tilde_chol), axis=1)
    condition_proxy = max_tilde_chol_diag / tf.maximum(
        min_tilde_chol_diag,
        tf.constant(1.0e-30, dtype=dtype),
    )
    finite_y_star = tf.reduce_all(tf.math.is_finite(y_star), axis=[1, 2])
    ok_per_batch = (
        finite_y_star
        & tf.math.is_finite(cov_residual)
        & tf.math.is_finite(condition_proxy)
        & (min_residual_chol_diag > 0.0)
        & (min_target_chol_diag > 0.0)
        & (min_tilde_chol_diag > 0.0)
    )
    return {
        "target_mean": target_mean,
        "target_cov": target_cov,
        "y_plus": y_plus,
        "plus_cov": plus_cov,
        "gap": gap,
        "residual_chol": residual_chol,
        "b_matrix": b_matrix,
        "xi": xi,
        "y_tilde": y_tilde,
        "tilde_mean": tilde_mean,
        "tilde_cov": tilde_cov,
        "centered_tilde": centered_tilde,
        "target_chol": target_chol,
        "tilde_chol": tilde_chol,
        "solved": solved,
        "affine": affine,
        "y_star": y_star,
        "cov_residual": cov_residual,
        "mean_residual": mean_residual,
        "min_residual_chol_diag": min_residual_chol_diag,
        "condition_proxy": condition_proxy,
        "min_tilde_chol_diag": min_tilde_chol_diag,
        "min_target_chol_diag": min_target_chol_diag,
        "ridge": ridge,
        "ok": tf.reduce_all(ok_per_batch),
        "ok_per_batch": ok_per_batch,
    }
