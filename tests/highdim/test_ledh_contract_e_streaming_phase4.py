from __future__ import annotations

import inspect
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_streaming_tf as streaming
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase4-small-fixture-freeze-2026-07-13.json"
)
DTYPE = tf.float64
ARCHIVAL_WRONG_ARTIFACT_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase4-local-parity-diagnostics-2026-07-14.json"
)


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fraction(value: str) -> float:
    return float(Fraction(value))


def _tensor(value: Any) -> tf.Tensor:
    if isinstance(value, str):
        return tf.constant(_fraction(value), DTYPE)
    if isinstance(value, list):
        def convert(item: Any) -> Any:
            if isinstance(item, list):
                return [convert(child) for child in item]
            return _fraction(item)

        return tf.constant(convert(value), DTYPE)
    raise TypeError(type(value))


def _inputs() -> tuple[tf.Tensor, ...]:
    fixture = _fixture()
    weights = _tensor(fixture["normalized_weights"])
    return (
        _tensor(fixture["scaled_geometry"]),
        _tensor(fixture["source_particles"]),
        tf.math.log(weights),
        weights,
        _tensor(fixture["residual_design"]),
        _tensor(fixture["ridge"]),
        _tensor(fixture["epsilon"]),
        _tensor(fixture["epsilon0"]),
        _tensor(fixture["scaling"]),
    )


def _base_direction() -> tuple[tf.Tensor, ...]:
    fixture = _fixture()
    direction = fixture["direction"]
    weights = _tensor(fixture["normalized_weights"])
    d_logw = _tensor(direction["normalized_log_weights"])
    return (
        _tensor(direction["scaled_geometry"]),
        _tensor(direction["source_particles"]),
        d_logw,
        weights * d_logw,
        _tensor(direction["residual_design"]),
        _tensor(direction["ridge"]),
        _tensor(direction["epsilon0"]),
    )


def _directions() -> tuple[tf.Tensor, ...]:
    return tuple(tf.stack([value, -2 * value], axis=-1) for value in _base_direction())


def _kwargs(index: int = 0) -> dict[str, int]:
    del index
    particle_count = len(_fixture()["source_particles"][0])
    return {
        "steps": int(_fixture()["finite_sinkhorn_steps"]),
        "row_chunk_size": particle_count,
        "col_chunk_size": particle_count,
    }


def _dense_plan(
    scaled_geometry: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
) -> tf.Tensor:
    particle_count = tf.shape(scaled_geometry)[1]
    log_n = tf.math.log(tf.cast(particle_count, DTYPE))
    uniform_log_weight = -log_n * tf.ones_like(normalized_log_weights)
    cost = annealed_transport_tf._filterflow_exact_cost(  # noqa: SLF001
        scaled_geometry, scaled_geometry
    )
    alpha, beta, _running, _iterations = (
        annealed_transport_tf._filterflow_manual_dense_finite_sinkhorn_outputs(  # noqa: SLF001
            normalized_log_weights,
            uniform_log_weight,
            cost,
            cost,
            cost,
            cost,
            epsilon=epsilon,
            epsilon0=epsilon0,
            scaling=scaling,
            steps=_kwargs()["steps"],
        )
    )
    return annealed_transport_tf._filterflow_exact_transport_from_potentials(  # noqa: SLF001
        scaled_geometry,
        alpha,
        beta,
        epsilon,
        normalized_log_weights,
        tf.cast(particle_count, DTYPE),
    )


def _dense_contract_e_forward(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
) -> dict[str, tf.Tensor]:
    plan = _dense_plan(
        scaled_geometry,
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
    )
    numerator = tf.linalg.matmul(plan, source_particles)
    mass = tf.reduce_sum(plan, axis=2)
    quotient = numerator / mass[:, :, None]
    reset = streaming.cloud_reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        source_particles,
        normalized_weights,
        quotient,
        residual_design,
        ridge,
    )
    return {
        "plan": plan,
        "numerator": numerator,
        "mass": mass,
        "quotient": quotient,
        "particles": reset["particles"],
    }


def _max_abs(left: tf.Tensor, right: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(left - right)).numpy())


def _diagnostic(value: float) -> dict[str, Any]:
    return {"float64_hex": float(value).hex(), "descriptive_decimal": float(value)}


def _collect_phase4_local_diagnostics() -> dict[str, Any]:
    inputs = _inputs()
    directions = _directions()
    (
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        epsilon,
        epsilon0,
        scaling,
    ) = inputs
    upstream = _tensor(_fixture()["output_particle_cotangent"])
    stream_forward = streaming._contract_e_streaming_forward_core(  # noqa: SLF001
        *inputs, **_kwargs()
    )
    dense_forward = _dense_contract_e_forward(*inputs)
    manual_jvp = streaming._contract_e_streaming_jvp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        *directions,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )["particles"]
    manual_vjp = streaming._contract_e_streaming_vjp_core(  # noqa: SLF001
        *inputs, upstream, **_kwargs()
    )
    watched = [scaled, particles, logw, weights, residual, ridge, epsilon0]
    names = (
        "scaled_geometry",
        "source_particles",
        "normalized_log_weights",
        "normalized_weights_probability",
        "residual_design",
        "ridge",
        "epsilon0",
    )
    manual_values = (
        manual_vjp["scaled_geometry"],
        manual_vjp["source_particles"],
        manual_vjp["normalized_log_weights_transport"],
        manual_vjp["normalized_weights_probability"],
        manual_vjp["residual_design"],
        manual_vjp["ridge"],
        manual_vjp["epsilon0"],
    )
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(watched)
        stream_particles = streaming._contract_e_streaming_forward_core(  # noqa: SLF001
            *inputs, **_kwargs()
        )["particles"]
        dense_particles = _dense_contract_e_forward(*inputs)["particles"]
        stream_objective = tf.reduce_sum(stream_particles * upstream)
        dense_objective = tf.reduce_sum(dense_particles * upstream)
    stream_automatic_vjp = tape.gradient(
        stream_objective,
        watched,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    dense_automatic_vjp = tape.gradient(
        dense_objective,
        watched,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    del tape

    stream_automatic_jvp = []
    dense_automatic_jvp = []
    duality_differences = []
    for index in range(2):
        tangent = [value[..., index] for value in directions]
        with tf.autodiff.ForwardAccumulator(watched, tangent) as accumulator:
            stream_particles = streaming._contract_e_streaming_forward_core(  # noqa: SLF001
                *inputs, **_kwargs()
            )["particles"]
        stream_automatic_jvp.append(accumulator.jvp(stream_particles))
        with tf.autodiff.ForwardAccumulator(watched, tangent) as accumulator:
            dense_particles = _dense_contract_e_forward(*inputs)["particles"]
        dense_automatic_jvp.append(accumulator.jvp(dense_particles))
        primal_pairing = tf.reduce_sum(upstream * manual_jvp[:, :, :, index])
        adjoint_pairing = sum(
            (
                tf.reduce_sum(value * tangent_value)
                for value, tangent_value in zip(
                    manual_values, tangent, strict=True
                )
            ),
            tf.constant(0.0, DTYPE),
        )
        duality_differences.append(
            abs(float(primal_pairing.numpy()) - float(adjoint_pairing.numpy()))
        )

    quotient = stream_forward["quotient"]
    quotient_jvp = streaming._streaming_row_quotient_jvp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        directions[0],
        directions[1],
        directions[2],
        directions[6],
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    quotient_vjp = streaming._streaming_row_quotient_vjp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        epsilon,
        epsilon0,
        scaling,
        manual_vjp["reset"]["transported_particles"],
        **_kwargs(),
    )
    return {
        "fixture_status": _fixture()["status"],
        "mass": {
            "minimum": _diagnostic(float(tf.reduce_min(quotient["mass"]).numpy())),
            "maximum": _diagnostic(float(tf.reduce_max(quotient["mass"]).numpy())),
            "maximum_row_residual": _diagnostic(
                float(tf.reduce_max(quotient["row_residual_by_batch"]).numpy())
            ),
            "all_finite_positive": bool(
                tf.reduce_all(quotient["valid_chart"]).numpy()
            ),
            "not_all_bitwise_one": bool(
                tf.reduce_any(tf.not_equal(quotient["mass"], 1.0)).numpy()
            ),
            "mass_tangent_max_abs": _diagnostic(
                float(tf.reduce_max(tf.abs(quotient_jvp["mass_tangent"])).numpy())
            ),
            "mass_cotangent_max_abs": _diagnostic(
                float(tf.reduce_max(tf.abs(quotient_vjp["mass_bar"])).numpy())
            ),
        },
        "stream_vs_dense": {
            "classification": "DESCRIPTIVE_NO_JUSTIFIED_KERNEL_ERROR_BOUND",
            "forward_max_abs": {
                "numerator": _diagnostic(
                    _max_abs(quotient["numerator"], dense_forward["numerator"])
                ),
                "mass": _diagnostic(
                    _max_abs(quotient["mass"], dense_forward["mass"])
                ),
                "quotient": _diagnostic(
                    _max_abs(quotient["particles"], dense_forward["quotient"])
                ),
                "contract_e_particles": _diagnostic(
                    _max_abs(stream_forward["particles"], dense_forward["particles"])
                ),
            },
            "jvp_max_abs_by_direction": [
                _diagnostic(_max_abs(manual_jvp[:, :, :, index], value))
                for index, value in enumerate(dense_automatic_jvp)
            ],
            "vjp_max_abs_by_input": {
                name: _diagnostic(_max_abs(value, dense_value))
                for name, value, dense_value in zip(
                    names, manual_values, dense_automatic_vjp, strict=True
                )
            },
        },
        "manual_vs_same_graph_autodiff": {
            "classification": "DESCRIPTIVE_NO_JUSTIFIED_GENERAL_FORWARD_ERROR_BOUND",
            "jvp_max_abs_by_direction": [
                _diagnostic(_max_abs(manual_jvp[:, :, :, index], value))
                for index, value in enumerate(stream_automatic_jvp)
            ],
            "vjp_max_abs_by_input": {
                name: _diagnostic(_max_abs(value, automatic))
                for name, value, automatic in zip(
                    names, manual_values, stream_automatic_vjp, strict=True
                )
            },
        },
        "exact_engineering_checks": {
            "standalone_quotient_defining_identities": "PASSED_IN_FOCUSED_TEST",
            "jvp_vjp_duality_abs_difference_by_direction": [
                _diagnostic(value) for value in duality_differences
            ],
            "nonzero_mass_tangent": bool(
                tf.reduce_any(tf.not_equal(quotient_jvp["mass_tangent"], 0.0)).numpy()
            ),
            "nonzero_mass_cotangent": bool(
                tf.reduce_any(tf.not_equal(quotient_vjp["mass_bar"], 0.0)).numpy()
            ),
            "generic_payload_shape": list(
                manual_vjp["quotient"]["constant_payload"].shape
            ),
            "direct_source_path_nonzero": bool(
                tf.reduce_any(
                    tf.not_equal(manual_vjp["source_particles_direct"], 0.0)
                ).numpy()
            ),
            "transport_source_path_nonzero": bool(
                tf.reduce_any(
                    tf.not_equal(manual_vjp["source_particles_transport"], 0.0)
                ).numpy()
            ),
            "direct_weight_path_nonzero": bool(
                tf.reduce_any(
                    tf.not_equal(manual_vjp["normalized_log_weights_moment"], 0.0)
                ).numpy()
            ),
            "transport_weight_path_nonzero": bool(
                tf.reduce_any(
                    tf.not_equal(manual_vjp["normalized_log_weights_transport"], 0.0)
                ).numpy()
            ),
        },
        "local_status": (
            "EXACT_QUOTIENT_AND_DUALITY_CHECKS_PASSED_"
            "EXACT_POLICY_CHUNK_ONLY"
        ),
    }


def test_standalone_row_quotient_exact_formulas_and_invalid_mass() -> None:
    numerator = tf.constant(
        [[[1.25, -0.5], [0.25, 1.5], [-0.75, 0.625]]], DTYPE
    )
    mass = tf.constant([[0.5, 1.0, 0.5]], DTYPE)
    numerator_tangent = tf.constant(
        [[[[1 / 16], [-1 / 32]], [[-1 / 32], [1 / 16]], [[1 / 64], [1 / 32]]]],
        DTYPE,
    )
    mass_tangent = tf.constant([[[1 / 64], [-1 / 32], [1 / 64]]], DTYPE)
    upstream = tf.constant(
        [[[1 / 4, -1 / 8], [1 / 8, 1 / 4], [-1 / 2, 1 / 4]]], DTYPE
    )
    jvp = streaming._row_quotient_jvp_core(  # noqa: SLF001
        numerator, mass, numerator_tangent, mass_tangent
    )
    vjp = streaming._row_quotient_vjp_core(  # noqa: SLF001
        numerator, mass, upstream
    )
    tf.debugging.assert_equal(jvp["particles"] * mass[:, :, None], numerator)
    tf.debugging.assert_equal(
        jvp["particles_tangent"] * mass[:, :, None, None],
        numerator_tangent
        - jvp["particles"][:, :, :, None] * mass_tangent[:, :, None, :],
    )
    tf.debugging.assert_equal(
        vjp["numerator_bar"] * mass[:, :, None], upstream
    )
    tf.debugging.assert_equal(
        vjp["mass_bar"] * mass,
        -tf.reduce_sum(upstream * vjp["particles"], axis=2),
    )
    primal_pairing = tf.reduce_sum(upstream * jvp["particles_tangent"][:, :, :, 0])
    adjoint_pairing = tf.reduce_sum(vjp["numerator_bar"] * numerator_tangent[:, :, :, 0])
    adjoint_pairing += tf.reduce_sum(vjp["mass_bar"] * mass_tangent[:, :, 0])
    tf.debugging.assert_equal(primal_pairing, adjoint_pairing)

    invalid = streaming._row_quotient_forward_core(  # noqa: SLF001
        numerator, tf.constant([[0.5, 0.0, -0.25]], DTYPE)
    )
    assert not bool(tf.reduce_all(invalid["valid_chart"]).numpy())


def test_generic_payload_vjp_uses_payload_width_and_matches_autodiff() -> None:
    scaled, particles, logw, _weights, _residual, _ridge, epsilon, epsilon0, scaling = _inputs()
    augmented = streaming._augmented_payload(particles)  # noqa: SLF001
    upstream = tf.concat(
        [
            _tensor(_fixture()["output_particle_cotangent"]),
            tf.constant([[[0.03], [-0.02], [0.01], [0.04]], [[-0.01], [0.025], [-0.015], [0.02]]], DTYPE),
        ],
        axis=2,
    )
    float_n = tf.cast(tf.shape(scaled)[1], DTYPE)
    log_n = tf.math.log(float_n)
    uniform_logw = -log_n * tf.ones_like(logw)
    alpha, beta = annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_total_vjp(  # noqa: SLF001
        logw,
        uniform_logw,
        scaled,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    manual = annealed_transport_tf._filterflow_streaming_transport_from_potentials_vjp(  # noqa: SLF001
        scaled,
        augmented,
        alpha,
        beta,
        epsilon,
        logw,
        float_n,
        upstream,
        row_chunk_size=_kwargs()["row_chunk_size"],
        col_chunk_size=_kwargs()["col_chunk_size"],
    )
    watched = [scaled, augmented, alpha, beta, logw]
    with tf.GradientTape() as tape:
        tape.watch(watched)
        value, _ = annealed_transport_tf._filterflow_streaming_transport_from_potentials(  # noqa: SLF001
            scaled,
            augmented,
            alpha,
            beta,
            epsilon,
            logw,
            float_n,
            row_chunk_size=_kwargs()["row_chunk_size"],
            col_chunk_size=_kwargs()["col_chunk_size"],
        )
        objective = tf.reduce_sum(value * upstream)
    automatic = tape.gradient(
        objective, watched, unconnected_gradients=tf.UnconnectedGradients.ZERO
    )
    assert manual[0].shape[-1] == 2
    assert manual[1].shape[-1] == 3
    for actual, expected in zip(manual, automatic, strict=True):
        tf.debugging.assert_near(actual, expected, atol=2e-14, rtol=2e-14)


def test_streaming_quotient_has_nonunit_mass_and_both_mass_derivative_paths() -> None:
    scaled, particles, logw, _weights, _residual, _ridge, epsilon, epsilon0, scaling = _inputs()
    d_scaled, d_particles, d_logw, _d_weights, _d_residual, _d_ridge, d_epsilon0 = _directions()
    forward = streaming._streaming_row_quotient_forward_core(  # noqa: SLF001
        scaled, particles, logw, epsilon, epsilon0, scaling, **_kwargs()
    )
    jvp = streaming._streaming_row_quotient_jvp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        d_scaled,
        d_particles,
        d_logw,
        d_epsilon0,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    upstream = _tensor(_fixture()["output_particle_cotangent"])
    vjp = streaming._streaming_row_quotient_vjp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        epsilon,
        epsilon0,
        scaling,
        upstream,
        **_kwargs(),
    )
    assert bool(tf.reduce_all(forward["valid_chart"]).numpy())
    assert bool(tf.reduce_any(tf.not_equal(forward["mass"], 1.0)).numpy())
    assert bool(tf.reduce_any(tf.not_equal(jvp["mass_tangent"], 0.0)).numpy())
    assert bool(tf.reduce_any(tf.not_equal(vjp["mass_bar"], 0.0)).numpy())
    assert vjp["constant_payload"].shape == forward["mass"].shape
    assert bool(tf.reduce_all(tf.math.is_finite(vjp["constant_payload"])).numpy())

    for index in range(2):
        primal = tf.reduce_sum(upstream * jvp["particles_tangent"][:, :, :, index])
        adjoint = tf.reduce_sum(vjp["scaled_geometry"] * d_scaled[:, :, :, index])
        adjoint += tf.reduce_sum(vjp["particles"] * d_particles[:, :, :, index])
        adjoint += tf.reduce_sum(vjp["normalized_log_weights"] * d_logw[:, :, index])
        adjoint += tf.reduce_sum(vjp["epsilon0"] * d_epsilon0[:, index])
        tf.debugging.assert_near(primal, adjoint, atol=5e-14, rtol=5e-14)


def test_streaming_quotient_matches_dense_finite_plan_descriptively() -> None:
    scaled, particles, logw, _weights, _residual, _ridge, epsilon, epsilon0, scaling = _inputs()
    stream = streaming._streaming_row_quotient_forward_core(  # noqa: SLF001
        scaled, particles, logw, epsilon, epsilon0, scaling, **_kwargs()
    )
    plan = _dense_plan(scaled, logw, epsilon, epsilon0, scaling)
    dense_numerator = tf.linalg.matmul(plan, particles)
    dense_mass = tf.reduce_sum(plan, axis=2)
    dense_particles = dense_numerator / dense_mass[:, :, None]
    diagnostics = {
        "numerator": _max_abs(stream["numerator"], dense_numerator),
        "mass": _max_abs(stream["mass"], dense_mass),
        "particles": _max_abs(stream["particles"], dense_particles),
    }
    assert all(tf.math.is_finite(value) for value in diagnostics.values())


def test_terminal_balance_repairs_both_consumed_plan_marginals() -> None:
    scaled, particles, logw, weights, residual, ridge, epsilon, epsilon0, scaling = _inputs()
    broken = streaming._streaming_row_quotient_forward_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        epsilon,
        epsilon0,
        scaling,
        steps=2,
        balance_steps=0,
        row_chunk_size=4,
        col_chunk_size=4,
    )
    repaired = streaming._contract_e_streaming_forward_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        epsilon,
        epsilon0,
        scaling,
        steps=20,
        balance_steps=100,
        row_chunk_size=4,
        col_chunk_size=4,
    )
    assert not bool(tf.reduce_all(broken["marginal_valid"]).numpy())
    assert bool(tf.reduce_all(repaired["quotient"]["marginal_valid"]).numpy())
    tf.debugging.assert_less_equal(
        repaired["quotient"]["maximum_row_absolute_residual"],
        repaired["quotient"]["marginal_roundoff_tolerance"],
    )
    tf.debugging.assert_less_equal(
        repaired["quotient"]["maximum_post_quotient_column_absolute_residual"],
        repaired["quotient"]["marginal_roundoff_tolerance"],
    )
    tf.debugging.assert_greater_equal(
        repaired["reset"]["gap_eigenvalues"],
        -repaired["quotient"]["marginal_roundoff_tolerance"][:, None],
    )


def test_terminal_balance_manual_jvp_vjp_match_autodiff() -> None:
    scaled, _particles, logw, _weights, _residual, _ridge, epsilon, _epsilon0, _scaling = _inputs()
    d_scaled, _d_particles, d_logw, _d_weights, _d_residual, _d_ridge, _d_epsilon0 = _directions()
    initial = tf.zeros_like(logw)
    d_initial = tf.zeros_like(d_logw)
    upstream = tf.constant(
        [[0.2, -0.1, 0.3, -0.4], [-0.3, 0.25, 0.15, -0.1]], DTYPE
    )
    value, tangent = (
        annealed_transport_tf._filterflow_streaming_terminal_balance_potential_jvp(  # noqa: SLF001
            logw,
            scaled,
            initial,
            d_logw,
            d_scaled,
            d_initial,
            epsilon,
            balance_steps=5,
            row_chunk_size=4,
            col_chunk_size=4,
        )
    )
    initial_bar, logw_bar, scaled_bar = (
        annealed_transport_tf._filterflow_streaming_terminal_balance_potential_vjp(  # noqa: SLF001
            logw,
            scaled,
            initial,
            epsilon,
            upstream,
            balance_steps=5,
            row_chunk_size=4,
            col_chunk_size=4,
        )
    )
    with tf.GradientTape() as tape:
        tape.watch((initial, logw, scaled))
        automatic_value = (
            annealed_transport_tf._filterflow_streaming_terminal_balance_potential(  # noqa: SLF001
                logw,
                scaled,
                initial,
                epsilon,
                balance_steps=5,
                row_chunk_size=4,
                col_chunk_size=4,
            )
        )
        objective = tf.reduce_sum(automatic_value * upstream)
    automatic = tape.gradient(objective, (initial, logw, scaled))
    tf.debugging.assert_equal(value, automatic_value)
    for manual, expected in zip(
        (initial_bar, logw_bar, scaled_bar), automatic, strict=True
    ):
        tf.debugging.assert_near(manual, expected, atol=5e-14, rtol=5e-14)
    for index in range(2):
        primal = tf.reduce_sum(upstream * tangent[:, :, index])
        adjoint = tf.reduce_sum(initial_bar * d_initial[:, :, index])
        adjoint += tf.reduce_sum(logw_bar * d_logw[:, :, index])
        adjoint += tf.reduce_sum(scaled_bar * d_scaled[:, :, :, index])
        tf.debugging.assert_near(primal, adjoint, atol=5e-14, rtol=5e-14)


def test_fused_contract_e_reuses_one_ot_state_and_matches_separated_route() -> None:
    scaled, particles, logw, weights, residual, ridge, epsilon, epsilon0, scaling = _inputs()
    d_scaled, d_particles, d_logw, d_weights, d_residual, d_ridge, d_epsilon0 = _directions()
    forward = streaming._contract_e_streaming_forward_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    tangent = streaming._contract_e_streaming_jvp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        d_scaled,
        d_particles,
        d_logw,
        d_weights,
        d_residual,
        d_ridge,
        d_epsilon0,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    fused = streaming._contract_e_streaming_forward_jvp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        d_scaled,
        d_particles,
        d_logw,
        d_weights,
        d_residual,
        d_ridge,
        d_epsilon0,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    tf.debugging.assert_near(fused["particles"], forward["particles"], atol=5e-14, rtol=5e-14)
    tf.debugging.assert_near(
        fused["particles_tangent"], tangent["particles"], atol=5e-13, rtol=5e-13
    )
    tf.debugging.assert_near(
        fused["quotient"]["mass"], forward["quotient"]["mass"], atol=5e-14, rtol=5e-14
    )
    assert int(fused["work"]["sinkhorn_state_constructions"]) == 1
    assert int(fused["work"]["terminal_balance_state_constructions"]) == 1
    assert int(fused["work"]["transport_tile_sweeps"]) == 1
    assert int(fused["work"]["marginal_tile_sweeps"]) == 0
    assert int(fused["work"]["diagnostic_solver_reconstructions"]) == 0


def test_fused_state_reduces_probability_mass_outside_payload_gemm() -> None:
    source = inspect.getsource(streaming._balanced_transport_forward_jvp_state_core)
    assert "row_mass = tf.reduce_sum(transport, axis=2)" in source
    assert "row_mass_tangent = tf.reduce_sum(transport_tangent, axis=2)" in source
    assert 'tf.einsum("bij,bjd->bid", transport, payload)' not in source


def test_contract_e_composed_jvp_vjp_and_weight_coordinates_match_autodiff() -> None:
    scaled, particles, logw, weights, residual, ridge, epsilon, epsilon0, scaling = _inputs()
    d_scaled, d_particles, d_logw, d_weights, d_residual, d_ridge, d_epsilon0 = _directions()
    jvp = streaming._contract_e_streaming_jvp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        d_scaled,
        d_particles,
        d_logw,
        d_weights,
        d_residual,
        d_ridge,
        d_epsilon0,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    upstream = _tensor(_fixture()["output_particle_cotangent"])
    vjp = streaming._contract_e_streaming_vjp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        epsilon,
        epsilon0,
        scaling,
        upstream,
        **_kwargs(),
    )
    tf.debugging.assert_equal(
        vjp["source_particles"],
        vjp["source_particles_direct"] + vjp["source_particles_transport"],
    )
    tf.debugging.assert_equal(
        vjp["normalized_log_weights_moment"],
        weights * vjp["normalized_weights_probability"],
    )
    tf.debugging.assert_equal(
        vjp["normalized_log_weights"],
        vjp["normalized_log_weights_transport"]
        + vjp["normalized_log_weights_moment"],
    )
    moment_corrected = vjp["normalized_log_weights_moment"] - weights * tf.reduce_sum(
        vjp["normalized_log_weights_moment"], axis=1, keepdims=True
    )
    shifted_probability = vjp["normalized_weights_probability"] + 0.75
    shifted_moment = weights * shifted_probability
    shifted_corrected = shifted_moment - weights * tf.reduce_sum(
        shifted_moment, axis=1, keepdims=True
    )
    tf.debugging.assert_near(
        moment_corrected, shifted_corrected, atol=5e-16, rtol=5e-16
    )

    input_names = (
        "scaled_geometry",
        "source_particles",
        "normalized_log_weights_transport",
        "normalized_weights_probability",
        "residual_design",
        "ridge",
        "epsilon0",
    )
    tangents = (
        d_scaled,
        d_particles,
        d_logw,
        d_weights,
        d_residual,
        d_ridge,
        d_epsilon0,
    )
    for index in range(2):
        primal = tf.reduce_sum(upstream * jvp["particles"][:, :, :, index])
        adjoint = sum(
            (
                tf.reduce_sum(vjp[name] * tangent[..., index])
                for name, tangent in zip(input_names, tangents, strict=True)
            ),
            tf.constant(0.0, DTYPE),
        )
        tf.debugging.assert_near(primal, adjoint, atol=2e-13, rtol=2e-13)


def test_archived_chunk_tilings_are_rejected_by_contract_e() -> None:
    archived = _fixture()["chunk_tilings"][0]
    with tf.test.TestCase().assertRaisesRegex(ValueError, "wrong under"):
        streaming._contract_e_streaming_forward_core(  # noqa: SLF001
            *_inputs(),
            steps=int(_fixture()["finite_sinkhorn_steps"]),
            row_chunk_size=int(archived["row_chunk_size"]),
            col_chunk_size=int(archived["col_chunk_size"]),
        )


def test_old_phase4_artifact_is_preserved_but_scientifically_ineligible() -> None:
    artifact = json.loads(
        ARCHIVAL_WRONG_ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == (
        "bayesfilter.contract_e_canonical_gradient_migration.phase4_local.v1"
    )
    assert artifact["status"] == (
        "EXACT_QUOTIENT_AND_DUALITY_CHECKS_PASSED_"
        "GENERAL_DENSE_AUTODIFF_CHUNK_PARITY_INCONCLUSIVE"
    )
    assert "chunk_tiling" in artifact["diagnostics"]
    assert artifact["status"].endswith("CHUNK_PARITY_INCONCLUSIVE")


def test_owned_boundaries_and_public_cpu_xla_wrappers() -> None:
    source = inspect.getsource(streaming)
    for forbidden in (
        "import numpy",
        "GradientTape",
        "ForwardAccumulator",
        "tf.linalg.inv",
        "tf.linalg.eigh",
        "tf.maximum(",
        "clip_by",
        "stopped_scale_keys",
        "transport_matrix",
    ):
        assert forbidden not in source
    for name in (
        "contract_e_streaming_forward_tf",
        "contract_e_streaming_jvp_tf",
        "contract_e_streaming_vjp_tf",
    ):
        assert getattr(streaming, name)._jit_compile is True

    inputs = _inputs()
    scaled, particles, logw, weights, residual, ridge, epsilon, epsilon0, scaling = inputs
    eager = streaming._contract_e_streaming_forward_core(  # noqa: SLF001
        *inputs, **_kwargs()
    )["particles"]
    compiled = streaming.contract_e_streaming_forward_tf(*inputs, **_kwargs())
    tf.debugging.assert_near(compiled, eager, atol=2e-14, rtol=2e-14)

    directions = _directions()
    compiled_jvp = streaming.contract_e_streaming_jvp_tf(
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        *directions,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )
    eager_jvp = streaming._contract_e_streaming_jvp_core(  # noqa: SLF001
        scaled,
        particles,
        logw,
        weights,
        residual,
        ridge,
        *directions,
        epsilon,
        epsilon0,
        scaling,
        **_kwargs(),
    )["particles"]
    tf.debugging.assert_near(compiled_jvp, eager_jvp, atol=2e-14, rtol=2e-14)

    upstream = _tensor(_fixture()["output_particle_cotangent"])
    compiled_vjp = streaming.contract_e_streaming_vjp_tf(
        *inputs, upstream, **_kwargs()
    )
    eager_vjp = streaming._contract_e_streaming_vjp_core(  # noqa: SLF001
        *inputs, upstream, **_kwargs()
    )
    for name, value in compiled_vjp.items():
        tf.debugging.assert_near(value, eager_vjp[name], atol=2e-14, rtol=2e-14)
