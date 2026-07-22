from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RESET_PATH = ROOT / "docs/benchmarks/contract_e_reset_tf.py"
DESIGN_PATH = ROOT / (
    "docs/plans/bayesfilter-contract-e-canonical-gradient-migration-"
    "phase1-numerical-statistical-design-freeze-2026-07-13.json"
)
DTYPE = tf.float64


def _load_reset_module():
    spec = importlib.util.spec_from_file_location("contract_e_phase1_reference", RESET_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RESET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    post_flow = tf.constant(
        [[[-1.0, 0.2], [0.0, -0.4], [0.8, 0.6], [1.5, -0.1]]],
        dtype=DTYPE,
    )
    logits = tf.constant([[-0.25, 0.15, -0.75, 0.30]], dtype=DTYPE)
    weights = tf.nn.softmax(logits, axis=1)
    matrix = tf.constant(
        [[[0.50, 0.25, 0.10, 0.15],
          [0.15, 0.40, 0.25, 0.20],
          [0.20, 0.15, 0.45, 0.20],
          [0.15, 0.20, 0.20, 0.45]]],
        dtype=DTYPE,
    )
    residual_noise = tf.constant(
        [[[-0.3, 0.7], [0.5, -0.2], [1.0, 0.1], [-0.4, -0.8]]],
        dtype=DTYPE,
    )
    upstream = tf.reshape(
        tf.linspace(tf.constant(-0.2, DTYPE), tf.constant(0.3, DTYPE), 8),
        [1, 4, 2],
    )
    ridge = tf.constant([1.0e-3], dtype=DTYPE)
    return post_flow, logits, weights, matrix, residual_noise, upstream, ridge


def test_fixed_ridge_manual_vjp_matches_autodiff_for_every_input() -> None:
    reset = _load_reset_module()
    post_flow, logits, weights, matrix, residual_noise, upstream, ridge = _fixture()

    with tf.GradientTape() as tape:
        tape.watch((post_flow, weights, matrix, residual_noise, ridge))
        particles = reset.contract_e_cholesky_ridge_reset_fixed_ridge(
            tf,
            post_flow=post_flow,
            weights=weights,
            matrix=matrix,
            residual_noise=residual_noise,
            rho=tf.constant(1.0, DTYPE),
            ridge=ridge,
        )["particles"]
        objective = tf.reduce_sum(particles * upstream)
    autodiff = tape.gradient(objective, (post_flow, weights, matrix, residual_noise, ridge))
    manual = reset.contract_e_cholesky_ridge_reset_fixed_ridge_vjp(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        upstream_particles=upstream,
    )

    for actual, key in zip(
        autodiff,
        ("post_flow", "weights", "matrix", "residual_noise", "ridge"),
        strict=True,
    ):
        np.testing.assert_allclose(manual[key].numpy(), actual.numpy(), rtol=2e-10, atol=2e-11)


def test_direct_moment_and_weight_paths_are_nonzero_and_not_transport_paths() -> None:
    reset = _load_reset_module()
    post_flow, _logits, weights, matrix, residual_noise, upstream, ridge = _fixture()
    manual = reset.contract_e_cholesky_ridge_reset_fixed_ridge_vjp(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        upstream_particles=upstream,
    )

    assert float(tf.linalg.norm(manual["weights"]).numpy()) > 1e-8
    assert float(tf.linalg.norm(manual["post_flow"]).numpy()) > 1e-8
    assert float(tf.linalg.norm(manual["matrix"]).numpy()) > 1e-8

    direction = tf.constant([[-0.04, 0.03, 0.02, -0.01]], dtype=DTYPE)
    direction -= tf.reduce_mean(direction, axis=1, keepdims=True)
    predicted = tf.reduce_sum(manual["weights"] * direction)
    step = tf.constant(1e-6, DTYPE)

    def objective(local_weights):
        particles = reset.contract_e_cholesky_ridge_reset_fixed_ridge(
            tf,
            post_flow=post_flow,
            weights=local_weights,
            matrix=matrix,
            residual_noise=residual_noise,
            rho=tf.constant(1.0, DTYPE),
            ridge=ridge,
        )["particles"]
        return tf.reduce_sum(particles * upstream)

    fd = (objective(weights + step * direction) - objective(weights - step * direction)) / (2 * step)
    np.testing.assert_allclose(predicted.numpy(), fd.numpy(), rtol=2e-7, atol=2e-9)


def test_fixed_ridge_manual_vjp_matches_directional_fd_for_each_input() -> None:
    reset = _load_reset_module()
    post_flow, _logits, weights, matrix, residual_noise, upstream, ridge = _fixture()
    values = {
        "post_flow": post_flow,
        "weights": weights,
        "matrix": matrix,
        "residual_noise": residual_noise,
        "ridge": ridge,
    }
    directions = {
        "post_flow": tf.reshape(tf.linspace(tf.constant(-0.03, DTYPE), tf.constant(0.04, DTYPE), 8), [1, 4, 2]),
        "weights": tf.constant([[-0.04, 0.03, 0.02, -0.01]], DTYPE),
        "matrix": tf.reshape(tf.linspace(tf.constant(-0.02, DTYPE), tf.constant(0.025, DTYPE), 16), [1, 4, 4]),
        "residual_noise": tf.reshape(tf.linspace(tf.constant(0.035, DTYPE), tf.constant(-0.03, DTYPE), 8), [1, 4, 2]),
        "ridge": tf.constant([2.0e-4], DTYPE),
    }
    directions["weights"] -= tf.reduce_mean(directions["weights"], axis=1, keepdims=True)
    manual = reset.contract_e_cholesky_ridge_reset_fixed_ridge_vjp(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        upstream_particles=upstream,
    )

    def objective(local_values):
        particles = reset.contract_e_cholesky_ridge_reset_fixed_ridge(
            tf,
            post_flow=local_values["post_flow"],
            weights=local_values["weights"],
            matrix=local_values["matrix"],
            residual_noise=local_values["residual_noise"],
            rho=tf.constant(1.0, DTYPE),
            ridge=local_values["ridge"],
        )["particles"]
        return tf.reduce_sum(particles * upstream)

    for name, direction in directions.items():
        predicted = tf.reduce_sum(manual[name] * direction)
        estimates = []
        for step_value in (4.0e-6, 2.0e-6, 1.0e-6, 5.0e-7):
            step = tf.constant(step_value, DTYPE)
            plus = dict(values)
            minus = dict(values)
            plus[name] = values[name] + step * direction
            minus[name] = values[name] - step * direction
            estimates.append((objective(plus) - objective(minus)) / (2.0 * step))
        for fd in estimates[1:]:
            np.testing.assert_allclose(predicted.numpy(), fd.numpy(), rtol=8e-7, atol=5e-9)
        spread = max(float(value.numpy()) for value in estimates) - min(
            float(value.numpy()) for value in estimates
        )
        assert spread <= 2.0e-8 * max(1.0, abs(float(predicted.numpy())))


def test_probability_weight_vjp_is_pulled_back_once_to_logits() -> None:
    reset = _load_reset_module()
    post_flow, logits, _weights, matrix, residual_noise, upstream, ridge = _fixture()
    weights = tf.nn.softmax(logits, axis=1)
    manual = reset.contract_e_cholesky_ridge_reset_fixed_ridge_vjp(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        upstream_particles=upstream,
    )
    expected = weights * (
        manual["weights"]
        - tf.reduce_sum(weights * manual["weights"], axis=1, keepdims=True)
    )
    with tf.GradientTape() as tape:
        tape.watch(logits)
        local_weights = tf.nn.softmax(logits, axis=1)
        particles = reset.contract_e_cholesky_ridge_reset_fixed_ridge(
            tf,
            post_flow=post_flow,
            weights=local_weights,
            matrix=matrix,
            residual_noise=residual_noise,
            rho=tf.constant(1.0, DTYPE),
            ridge=ridge,
        )["particles"]
        objective = tf.reduce_sum(particles * upstream)
    actual = tape.gradient(objective, logits)

    np.testing.assert_allclose(expected.numpy(), actual.numpy(), rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(tf.reduce_sum(expected, axis=1).numpy(), [0.0], atol=2e-16)
    shifted = weights * (
        (manual["weights"] + 7.0)
        - tf.reduce_sum(weights * (manual["weights"] + 7.0), axis=1, keepdims=True)
    )
    np.testing.assert_allclose(shifted.numpy(), expected.numpy(), rtol=1e-13, atol=1e-13)


def test_ridged_identity_and_raw_covariance_residual_formula() -> None:
    reset = _load_reset_module()
    post_flow, _logits, weights, matrix, residual_noise, _upstream, ridge = _fixture()
    aux = reset.contract_e_cholesky_ridge_reset_fixed_ridge(
        tf,
        post_flow=post_flow,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        return_aux=True,
    )["aux"]
    eye = tf.eye(2, batch_shape=[1], dtype=DTYPE)
    left = aux["affine"] @ (aux["tilde_cov"] + ridge[:, None, None] * eye) @ tf.linalg.matrix_transpose(aux["affine"])
    right = aux["target_cov"] + ridge[:, None, None] * eye
    raw_residual = aux["star_cov"] - aux["target_cov"] if "star_cov" in aux else None
    centered = aux["y_star"] - tf.reduce_mean(aux["y_star"], axis=1, keepdims=True)
    measured = tf.einsum("bni,bnj->bij", centered, centered) / tf.cast(tf.shape(centered)[1], DTYPE)
    measured_residual = measured - aux["target_cov"]
    predicted_residual = ridge[:, None, None] * (
        eye - aux["affine"] @ tf.linalg.matrix_transpose(aux["affine"])
    )

    assert raw_residual is None
    np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(measured_residual.numpy(), predicted_residual.numpy(), rtol=2e-10, atol=2e-11)
    assert float(tf.linalg.norm(measured_residual).numpy()) > 0.0


def test_row_quotient_jvp_and_vjp_formulas() -> None:
    numerator = tf.constant([[[1.2, -0.6], [0.3, 1.5], [-0.4, 0.8]]], DTYPE)
    mass = tf.constant([[0.8, 1.15, 0.55]], DTYPE)
    d_numerator = tf.constant([[[0.05, -0.02], [-0.03, 0.04], [0.02, 0.01]]], DTYPE)
    d_mass = tf.constant([[0.01, -0.02, 0.015]], DTYPE)
    upstream = tf.constant([[[0.3, -0.1], [0.2, 0.4], [-0.5, 0.25]]], DTYPE)
    cloud = numerator / mass[:, :, None]
    d_cloud = (d_numerator - cloud * d_mass[:, :, None]) / mass[:, :, None]
    numerator_bar = upstream / mass[:, :, None]
    mass_bar = -tf.reduce_sum(upstream * cloud, axis=2) / mass

    primal_pairing = tf.reduce_sum(upstream * d_cloud)
    adjoint_pairing = tf.reduce_sum(numerator_bar * d_numerator) + tf.reduce_sum(mass_bar * d_mass)
    np.testing.assert_allclose(primal_pairing.numpy(), adjoint_pairing.numpy(), rtol=1e-14, atol=1e-14)
    assert float(tf.linalg.norm(mass_bar).numpy()) > 0.0


def test_design_freeze_has_no_optional_stopping_or_silent_gradient_margin() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    stats = design["lgssm_statistical_design"]
    assert stats["design"] == "fixed_five_seed_paired_common_random_numbers_no_optional_stopping"
    assert stats["confidence_family"]["familywise_level"] == 0.95
    assert stats["confidence_family"]["components"] == 6
    assert stats["gradient_equivalence"]["relative_margin"] is None
    assert stats["gradient_equivalence"]["historical_one_percent_reused"] is False
    fd = design["finite_difference_design"]
    assert fd["screen_role"] == "heuristic_only_implementation_screen_not_a_confidence_interval"
    assert fd["relative_screen_inequality"] == "relative_error <= 0.05*sqrt(p)"
    assert fd["endpoint_value_absolute_error_bounds"] is None
    assert "bitwise equal" in fd["endpoint_construction"]
    assert "actual ratio" in fd["truncation_estimate"]
    assert "floor((first_index+last_index)/2)" in fd["plateau_rule"]
    assert "BLOCKED" in fd["status"]
    coverage = stats["confidence_family"]["coverage_note"]
    assert "at least 95%" in coverage
    assert "not equality" in coverage
    for name, gate in design["numerical_vetoes"].items():
        if name not in {"finite_values", "cholesky"}:
            assert gate.get("threshold_formula") is None or "threshold_formula" not in gate
