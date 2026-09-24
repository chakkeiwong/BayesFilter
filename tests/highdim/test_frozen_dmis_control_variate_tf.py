"""Focused tests for the model-independent frozen DMIS candidate route.

These are CPU diagnostics.  They prove finite-bank algebra and tangent parity;
they do not establish C2 likelihood accuracy or proposal efficiency.
"""

from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.highdim.frozen_dmis_control_variate_tf import (
    DTYPE,
    complete_mixture_log_density,
    frozen_importance_directional_estimate,
    frozen_importance_estimate,
    make_compiled_directional_kernel,
    make_compiled_value_kernel,
)


def _component_logs(points: tf.Tensor, means: tuple[float, float]) -> tf.Tensor:
    rows = []
    for mean in means:
        rows.append(
            -0.5 * tf.square(points - tf.constant(mean, DTYPE))
            - tf.constant(0.5 * math.log(2.0 * math.pi), DTYPE)
        )
    return tf.stack(rows, axis=1)


def _student_log_density(
    points: tf.Tensor, location: float, scale: float, degrees: float
) -> tf.Tensor:
    nu = tf.constant(degrees, DTYPE)
    z = (points - tf.constant(location, DTYPE)) / tf.constant(scale, DTYPE)
    return (
        tf.math.lgamma((nu + 1.0) / 2.0)
        - tf.math.lgamma(nu / 2.0)
        - 0.5 * tf.math.log(nu * tf.constant(math.pi, DTYPE))
        - tf.math.log(tf.constant(scale, DTYPE))
        - (nu + 1.0) / 2.0 * tf.math.log1p(tf.square(z) / nu)
    )


def _fixture():
    points = tf.constant([-2.0, -0.5, 0.25, 1.0, 2.0], DTYPE)
    logs = _component_logs(points, (-1.0, 1.25))
    alpha = tf.constant([0.4, 0.6], DTYPE)
    masses = tf.constant([0.4, 0.15, 0.15, 0.15, 0.15], DTYPE)
    # Assemble this denominator independently of the implementation under
    # test, so a shared log-sum-exp error cannot cancel in the expected value.
    q = alpha[0] * tf.exp(logs[:, 0]) + alpha[1] * tf.exp(logs[:, 1])
    integrand = tf.constant([0.8, 1.1, 0.7, 1.4, 0.9], DTYPE)
    target = q * integrand
    control = q * tf.constant([0.75, 1.05, 0.72, 1.35, 0.95], DTYPE)
    # The known control integral is over the target measure, so the finite
    # proposal-bank quadrature uses h/q rather than h itself.
    z_control = tf.reduce_sum(masses * control / q)
    return points, logs, alpha, masses, target, control, z_control, integrand


def test_complete_mixture_uses_all_components_and_is_permutation_invariant() -> None:
    points, logs, alpha, *_ = _fixture()
    direct = complete_mixture_log_density(logs, alpha)
    independent = tf.math.log(
        alpha[0] * tf.exp(logs[:, 0]) + alpha[1] * tf.exp(logs[:, 1])
    )
    permuted = complete_mixture_log_density(
        tf.gather(logs, [1, 0], axis=1), tf.gather(alpha, [1, 0])
    )
    tf.debugging.assert_near(direct, permuted, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_near(direct, independent, atol=1e-14, rtol=1e-14)
    assert bool(tf.reduce_all(tf.math.is_finite(points)).numpy())


def test_deterministic_bank_uses_declared_base_masses() -> None:
    _, logs, alpha, masses, target, _, _, integrand = _fixture()
    result = frozen_importance_estimate(target, logs, alpha, masses)
    expected = tf.reduce_sum(masses * integrand)
    tf.debugging.assert_near(result["dmis_normalizer"], expected, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_near(result["normalizer"], expected, atol=1e-14, rtol=1e-14)
    assert bool(result["valid"].numpy())


def test_control_variate_is_exact_for_the_declared_finite_quadrature() -> None:
    _, logs, alpha, masses, target, control, z_control, integrand = _fixture()
    result = frozen_importance_estimate(
        target,
        logs,
        alpha,
        masses,
        control_values=control,
        control_normalizer=z_control,
    )
    expected = tf.reduce_sum(masses * integrand)
    tf.debugging.assert_near(result["normalizer"], expected, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_near(
        result["residual_second_moment"],
        tf.reduce_sum(
            masses
            * tf.square(
                (target - control)
                / (alpha[0] * tf.exp(logs[:, 0]) + alpha[1] * tf.exp(logs[:, 1]))
            )
        ),
        atol=1e-14,
        rtol=1e-14,
    )
    assert bool(result["valid"].numpy())


def test_control_variate_reduces_the_declared_residual_second_moment() -> None:
    _, logs, alpha, masses, target, control, z_control, _ = _fixture()
    plain = frozen_importance_estimate(target, logs, alpha, masses)
    corrected = frozen_importance_estimate(
        target,
        logs,
        alpha,
        masses,
        control_values=control,
        control_normalizer=z_control,
    )
    plain_second_moment = tf.reduce_sum(
        masses * tf.square(plain["target_weights"])
    )
    assert float(corrected["residual_second_moment"].numpy()) < float(
        plain_second_moment.numpy()
    )


def test_directional_formula_matches_central_difference_including_proposal_tangent() -> None:
    _, logs0, alpha, masses, target0, control0, z0, _ = _fixture()
    target_slope = tf.constant([0.2, -0.1, 0.35, -0.25, 0.15], DTYPE)[:, None]
    control_slope = tf.constant([-0.05, 0.1, 0.08, -0.12, 0.03], DTYPE)[:, None]
    component_slope = tf.constant(
        [[0.11, -0.04], [0.11, -0.04], [0.11, -0.04], [0.11, -0.04], [0.11, -0.04]],
        DTYPE,
    )[:, :, None]
    control_normalizer_slope = tf.constant([0.07], DTYPE)
    analytic = frozen_importance_directional_estimate(
        target0,
        logs0,
        alpha,
        masses,
        target0[:, None] * target_slope,
        component_slope,
        control_values=control0,
        control_normalizer=z0,
        control_tangent=control0[:, None] * control_slope,
        control_normalizer_tangent=control_normalizer_slope,
    )
    step = tf.constant(1.0e-6, DTYPE)

    def value_at(offset: tf.Tensor) -> tf.Tensor:
        target = target0 * tf.exp(target_slope[:, 0] * offset)
        control = control0 * tf.exp(control_slope[:, 0] * offset)
        logs = logs0 + offset * component_slope[:, :, 0]
        return frozen_importance_estimate(
            target,
            logs,
            alpha,
            masses,
            control_values=control,
            control_normalizer=z0 + offset * control_normalizer_slope[0],
        )["normalizer"]

    finite_difference = (value_at(step) - value_at(-step)) / (2.0 * step)
    tf.debugging.assert_near(
        analytic["normalizer_tangent"][0], finite_difference, atol=2e-8, rtol=2e-8
    )
    assert bool(analytic["tangent_valid"].numpy())


def test_compiled_value_and_directional_kernels_match_eager_kernel() -> None:
    _, logs, alpha, masses, target, control, z_control, _ = _fixture()
    eager = frozen_importance_estimate(
        target, logs, alpha, masses, control_values=control, control_normalizer=z_control
    )
    compiled = make_compiled_value_kernel(5, 2, jit_compile=False)(
        target, logs, alpha, masses, control, z_control
    )
    tf.debugging.assert_near(compiled["normalizer"], eager["normalizer"], atol=1e-14, rtol=1e-14)
    tf.debugging.assert_near(
        compiled["complete_mixture_log_density"], eager["complete_mixture_log_density"], atol=1e-14
    )

    target_dot = tf.ones([5, 1], DTYPE) * 0.03
    component_dot = tf.zeros([5, 2, 1], DTYPE)
    control_dot = tf.zeros([5, 1], DTYPE)
    z_dot = tf.zeros([1], DTYPE)
    eager_tangent = frozen_importance_directional_estimate(
        target,
        logs,
        alpha,
        masses,
        target_dot,
        component_dot,
        control_values=control,
        control_normalizer=z_control,
        control_tangent=control_dot,
        control_normalizer_tangent=z_dot,
    )
    compiled_tangent = make_compiled_directional_kernel(5, 2, 1, jit_compile=False)(
        target,
        logs,
        alpha,
        masses,
        control,
        z_control,
        target_dot,
        component_dot,
        control_dot,
        z_dot,
    )
    tf.debugging.assert_near(
        compiled_tangent["normalizer_tangent"],
        eager_tangent["normalizer_tangent"],
        atol=1e-14,
        rtol=1e-14,
    )


def test_invalid_masses_fail_closed_and_negative_cv_is_reported() -> None:
    _, logs, alpha, masses, target, control, z_control, _ = _fixture()
    with pytest.raises(tf.errors.InvalidArgumentError):
        frozen_importance_estimate(
            target,
            logs,
            tf.constant([0.5, 0.6], DTYPE),
            masses,
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        frozen_importance_estimate(
            target,
            logs,
            alpha,
            masses,
            control_values=control,
            control_normalizer=-1.0,
        )
    negative = frozen_importance_estimate(
        tf.zeros_like(target),
        logs,
        alpha,
        masses,
        control_values=control,
        control_normalizer=0.0,
    )
    assert not bool(negative["normalizer_positive"].numpy())
    assert not bool(negative["valid"].numpy())
    assert math.isfinite(float(z_control.numpy()))


def test_student_components_are_accepted_by_the_same_complete_denominator() -> None:
    points = tf.constant([-3.0, -0.75, 0.5, 2.5], DTYPE)
    logs = tf.stack(
        [
            _student_log_density(points, -0.5, 1.2, 5.0),
            _student_log_density(points, 1.0, 0.8, 7.0),
        ],
        axis=1,
    )
    alpha = tf.constant([0.35, 0.65], DTYPE)
    direct = complete_mixture_log_density(logs, alpha)
    permuted = complete_mixture_log_density(
        tf.gather(logs, [1, 0], axis=1), tf.gather(alpha, [1, 0])
    )
    tf.debugging.assert_near(direct, permuted, atol=1e-14, rtol=1e-14)
    assert bool(tf.reduce_all(tf.math.is_finite(logs)).numpy())


def test_zero_density_rows_are_masked_but_unsupported_target_fails_closed() -> None:
    logs = tf.constant(
        [[-float("inf"), -float("inf")], [0.0, -1.0]], DTYPE
    )
    alpha = tf.constant([0.5, 0.5], DTYPE)
    masses = tf.constant([0.5, 0.5], DTYPE)
    target = tf.constant([0.0, 1.0], DTYPE)
    harmless = frozen_importance_estimate(target, logs, alpha, masses)
    assert bool(harmless["finite"].numpy())
    assert bool(harmless["support_valid"].numpy())
    assert bool(harmless["valid"].numpy())
    unsupported = frozen_importance_estimate(
        tf.constant([1.0, 1.0], DTYPE), logs, alpha, masses
    )
    assert not bool(unsupported["support_valid"].numpy())
    assert not bool(unsupported["valid"].numpy())


def test_control_variate_only_requires_support_for_nonzero_residual() -> None:
    logs = tf.constant(
        [[-float("inf"), -float("inf")], [0.0, -1.0]], DTYPE
    )
    alpha = tf.constant([0.5, 0.5], DTYPE)
    masses = tf.constant([0.5, 0.5], DTYPE)
    target = tf.constant([1.0, 1.0], DTYPE)
    control = tf.constant([1.0, 0.0], DTYPE)
    result = frozen_importance_estimate(
        target,
        logs,
        alpha,
        masses,
        control_values=control,
        control_normalizer=tf.constant(0.5, DTYPE),
    )
    assert bool(result["residual_support_valid"].numpy())
    assert not bool(result["target_support_valid"].numpy())
    assert not bool(result["dmis_valid"].numpy())
    assert bool(result["valid"].numpy())


def test_directional_support_fails_closed_without_nan_on_zero_density_rows() -> None:
    logs = tf.constant(
        [[-float("inf"), -float("inf")], [0.0, -1.0]], DTYPE
    )
    alpha = tf.constant([0.5, 0.5], DTYPE)
    masses = tf.constant([0.5, 0.5], DTYPE)
    target = tf.constant([1.0, 1.0], DTYPE)
    control = tf.constant([1.0, 0.0], DTYPE)
    result = frozen_importance_directional_estimate(
        target,
        logs,
        alpha,
        masses,
        tf.zeros([2, 1], DTYPE),
        tf.zeros([2, 2, 1], DTYPE),
        control_values=control,
        control_normalizer=0.5,
    )
    assert not bool(result["tangent_support_valid"].numpy())
    assert not bool(result["tangent_valid"].numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result["normalizer_tangent"])).numpy())
