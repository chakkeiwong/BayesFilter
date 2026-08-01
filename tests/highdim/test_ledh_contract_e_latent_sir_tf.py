from __future__ import annotations

import inspect

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim import ledh_contract_e_latent_sir_tf as candidate
from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
    dense_latent_sir_value_and_manual_score,
    prepare_reduced_dense_grids,
    reduced_latent_preclip_sir_model,
)


DTYPE = tf.float64


def _prepared(*, reset: bool = True):
    particles = 8
    observations = tf.constant([[0.15], [0.1]], DTYPE)
    initial_noise = tf.constant(
        [
            [-1.5, -1.0],
            [-1.0, 0.5],
            [-0.5, 1.0],
            [-0.2, -1.5],
            [0.2, 1.5],
            [0.5, -0.5],
            [1.0, -1.0],
            [1.5, 1.0],
        ],
        DTYPE,
    )[None, :, :]
    transition_noise = tf.constant(
        [
            [-1.2, 0.2],
            [-0.8, -0.3],
            [-0.4, 0.7],
            [-0.1, -0.8],
            [0.1, 0.8],
            [0.4, -0.7],
            [0.8, 0.3],
            [1.2, -0.2],
        ],
        DTYPE,
    )[None, None, :, :]
    residual = tf.constant(
        [
            [-1.0, -0.75],
            [-0.75, 0.25],
            [-0.5, 0.5],
            [-0.25, -1.0],
            [0.25, 1.0],
            [0.5, -0.5],
            [0.75, -0.25],
            [1.0, 0.75],
        ],
        DTYPE,
    )
    tf.debugging.assert_near(tf.reduce_sum(residual, axis=0), tf.zeros([2], DTYPE))
    return {
        "observations": observations,
        "initial_noise": initial_noise,
        "transition_noise": transition_noise,
        "fixed_reset_mask": tf.constant([[reset, reset]], tf.bool),
        "residual_design": tf.tile(residual[None, None, :, :], [1, 2, 1, 1]),
        "prepared_ridge": tf.constant([[1.0e-6, 1.0e-6]], DTYPE),
        "epsilon": tf.constant(0.25, DTYPE),
        "scaling": tf.constant(0.9, DTYPE),
    }


def _prepared_quantile(particle_count: int, *, reset: bool = True):
    probabilities = (tf.cast(tf.range(particle_count), DTYPE) + 0.5) / tf.cast(
        particle_count, DTYPE
    )
    quantiles = tfp.distributions.Normal(
        tf.constant(0.0, DTYPE), tf.constant(1.0, DTYPE)
    ).quantile(probabilities)
    initial_noise = tf.stack(
        [quantiles, tf.roll(quantiles, shift=max(1, particle_count // 3), axis=0)],
        axis=1,
    )
    transition_noise = tf.stack(
        [tf.roll(quantiles, shift=particle_count // 3, axis=0), -quantiles], axis=1
    )
    centered_index = tf.cast(tf.range(particle_count), DTYPE) - 0.5 * tf.cast(
        particle_count - 1, DTYPE
    )
    residual = tf.stack(
        [centered_index, tf.roll(centered_index, shift=particle_count // 2, axis=0)],
        axis=1,
    )
    residual /= tf.reduce_max(tf.abs(residual))
    residual -= tf.reduce_mean(residual, axis=0, keepdims=True)
    centered_noise = initial_noise - tf.reduce_mean(initial_noise, axis=0, keepdims=True)
    covariance = tf.einsum("ni,nj->ij", centered_noise, centered_noise) / tf.cast(
        particle_count, DTYPE
    )
    tf.debugging.assert_positive(tf.linalg.eigvalsh(covariance))
    return {
        "observations": tf.constant([[0.15], [0.1]], DTYPE),
        "initial_noise": initial_noise[None, :, :],
        "transition_noise": transition_noise[None, None, :, :],
        "fixed_reset_mask": tf.constant([[reset, reset]], tf.bool),
        "residual_design": tf.tile(residual[None, None, :, :], [1, 2, 1, 1]),
        "prepared_ridge": tf.constant([[1.0e-6, 1.0e-6]], DTYPE),
        "epsilon": tf.constant(0.25, DTYPE),
        "scaling": tf.constant(0.9, DTYPE),
    }


def _prepared_austria_for_identity_test():
    model = candidate.latent_preclip_zhao_cui_sir_austria_model()
    state_dimension = model.state_dim()
    observation_dimension = model.observation_dim()
    particle_count = 16
    initial_noise = tf.random.stateless_normal(
        [1, particle_count, state_dimension],
        seed=[20260716, 9101],
        dtype=DTYPE,
    )
    transition_noise = tf.random.stateless_normal(
        [1, 1, particle_count, state_dimension],
        seed=[20260716, 9102],
        dtype=DTYPE,
    )
    residual_design = tf.random.stateless_normal(
        [1, 2, particle_count, state_dimension],
        seed=[20260716, 9103],
        dtype=DTYPE,
    )
    residual_design -= tf.reduce_mean(residual_design, axis=2, keepdims=True)
    return {
        "observations": tf.zeros([2, observation_dimension], DTYPE),
        "initial_noise": initial_noise,
        "transition_noise": transition_noise,
        "fixed_reset_mask": tf.ones([1, 2], tf.bool),
        "residual_design": residual_design,
        "prepared_ridge": tf.fill([1, 2], tf.constant(1.0e-6, DTYPE)),
        "epsilon": tf.constant(0.25, DTYPE),
        "scaling": tf.constant(0.9, DTYPE),
    }


def _prepared_two_node_registered_route(horizon: int):
    model = candidate.latent_preclip_two_node_spatial_sir_model()
    state_dimension = model.state_dim()
    observation_dimension = model.observation_dim()
    particle_count = 8
    initial_noise = tf.random.stateless_normal(
        [1, particle_count, state_dimension],
        seed=[20260716, 9201],
        dtype=DTYPE,
    )
    transition_noise = tf.random.stateless_normal(
        [1, max(1, horizon - 1), particle_count, state_dimension],
        seed=[20260716, 9202],
        dtype=DTYPE,
    )
    residual_design = tf.random.stateless_normal(
        [1, horizon, particle_count, state_dimension],
        seed=[20260716, 9203],
        dtype=DTYPE,
    )
    residual_design -= tf.reduce_mean(residual_design, axis=2, keepdims=True)
    return {
        "observations": tf.zeros([horizon, observation_dimension], DTYPE),
        "initial_noise": initial_noise,
        "transition_noise": transition_noise,
        "fixed_reset_mask": tf.ones([1, horizon], tf.bool),
        "residual_design": residual_design,
        "prepared_ridge": tf.fill([1, horizon], tf.constant(1.0e-6, DTYPE)),
        "epsilon": tf.constant(0.25, DTYPE),
        "scaling": tf.constant(0.9, DTYPE),
    }


def _automatic_score(theta: tf.Tensor, prepared) -> tf.Tensor:
    model = reduced_latent_preclip_sir_model()
    spec = candidate.static_spec_from_model(model)
    tensors = candidate._as_prepared_tensors(prepared, spec)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = candidate.latent_sir_contract_e_value_and_score_core(
            theta,
            tensors,
            spec,
            steps=candidate.CANONICAL_ANNEALING_STEPS,
            balance_steps=candidate.CANONICAL_BALANCE_STEPS,
            row_chunk_size=8,
            col_chunk_size=8,
        )["objective"]
    score = tape.gradient(value, theta)
    assert score is not None
    return score


def _finite_difference(theta: tf.Tensor, prepared, step: float = 1.0e-5) -> tf.Tensor:
    model = reduced_latent_preclip_sir_model()
    spec = candidate.static_spec_from_model(model)
    tensors = candidate._as_prepared_tensors(prepared, spec)
    values = []
    for index in range(candidate.PARAMETER_COUNT):
        direction = tf.one_hot(index, candidate.PARAMETER_COUNT, dtype=DTYPE)
        plus = candidate.latent_sir_contract_e_value_and_score_core(
            theta + step * direction,
            tensors,
            spec,
            steps=candidate.CANONICAL_ANNEALING_STEPS,
            balance_steps=candidate.CANONICAL_BALANCE_STEPS,
            row_chunk_size=8,
            col_chunk_size=8,
        )["objective"]
        minus = candidate.latent_sir_contract_e_value_and_score_core(
            theta - step * direction,
            tensors,
            spec,
            steps=candidate.CANONICAL_ANNEALING_STEPS,
            balance_steps=candidate.CANONICAL_BALANCE_STEPS,
            row_chunk_size=8,
            col_chunk_size=8,
        )["objective"]
        values.append((plus - minus) / (2.0 * step))
    return tf.stack(values)


def test_candidate_uses_contract_e_chol_and_manual_total_score_matches_ad_fd() -> None:
    theta = tf.constant([0.03, -0.02, 0.04], DTYPE)
    prepared = _prepared(reset=True)
    model = reduced_latent_preclip_sir_model()
    spec = candidate.static_spec_from_model(model)
    tensors = candidate._as_prepared_tensors(prepared, spec)
    result = candidate.latent_sir_contract_e_value_and_score_core(
        theta,
        tensors,
        spec,
        steps=candidate.CANONICAL_ANNEALING_STEPS,
        balance_steps=candidate.CANONICAL_BALANCE_STEPS,
        row_chunk_size=8,
        col_chunk_size=8,
    )
    automatic = _automatic_score(theta, prepared)
    finite_difference = _finite_difference(theta, prepared)

    assert candidate.CANDIDATE_ROUTE_ID == "contract_e_chol_latent_preclip_sir_candidate_v1"
    assert candidate.CANDIDATE_STATUS == "candidate_not_canonical_not_admitted"
    assert bool(tf.reduce_all(result["valid_chart"]).numpy())
    assert bool(tf.reduce_all(result["reset_valid_history"]).numpy())
    assert bool(tf.reduce_all(result["minimum_mass_history"] > 0.0).numpy())
    tf.debugging.assert_near(result["score"], automatic, atol=2.0e-8, rtol=2.0e-8)
    tf.debugging.assert_near(result["score"], finite_difference, atol=2.0e-6, rtol=2.0e-6)


def test_active_reset_changes_next_step_and_total_score() -> None:
    theta = tf.constant([0.03, -0.02, 0.04], DTYPE)
    model = reduced_latent_preclip_sir_model()
    spec = candidate.static_spec_from_model(model)
    active_tensors = candidate._as_prepared_tensors(_prepared(reset=True), spec)
    inactive_tensors = candidate._as_prepared_tensors(_prepared(reset=False), spec)
    active = candidate.latent_sir_contract_e_value_and_score_core(
        theta, active_tensors, spec, steps=2, row_chunk_size=8, col_chunk_size=8
    )
    inactive = candidate.latent_sir_contract_e_value_and_score_core(
        theta, inactive_tensors, spec, steps=2, row_chunk_size=8, col_chunk_size=8
    )
    assert abs(float((active["objective"] - inactive["objective"]).numpy())) > 1.0e-7
    assert float(tf.reduce_max(tf.abs(active["score"] - inactive["score"])).numpy()) > 1.0e-7


def test_clipping_boundary_is_a_fail_closed_score_chart() -> None:
    latent = tf.constant([[[0.0, 0.2], [-0.3, 0.1]]], DTYPE)
    tangent = tf.ones([1, 2, 2, candidate.PARAMETER_COUNT], DTYPE)
    physical, physical_tangent, away = candidate._physical_state_and_tangent(
        latent, tangent, time_index=1
    )
    tf.debugging.assert_equal(away, [False])
    tf.debugging.assert_equal(physical[:, :, 0], tf.zeros([1, 2], DTYPE))
    tf.debugging.assert_equal(
        physical_tangent[:, :, 0, :],
        tf.zeros([1, 2, candidate.PARAMETER_COUNT], DTYPE),
    )


def test_nonzero_susceptible_states_do_not_use_an_arbitrary_boundary_band() -> None:
    latent = tf.constant([[[1.0e-15, 0.2], [-1.0e-15, 0.1]]], DTYPE)
    tangent = tf.ones([1, 2, 2, candidate.PARAMETER_COUNT], DTYPE)
    _, physical_tangent, away = candidate._physical_state_and_tangent(
        latent, tangent, time_index=1
    )
    tf.debugging.assert_equal(away, [True])
    tf.debugging.assert_equal(
        physical_tangent[:, :, 0, :],
        tf.constant(
            [[[1.0] * candidate.PARAMETER_COUNT, [0.0] * candidate.PARAMETER_COUNT]],
            DTYPE,
        ),
    )


def test_initial_state_bypasses_clipping_boundary_chart() -> None:
    latent = tf.constant([[[0.0, 0.2], [-0.3, 0.1]]], DTYPE)
    tangent = tf.ones([1, 2, 2, candidate.PARAMETER_COUNT], DTYPE)
    physical, physical_tangent, away = candidate._physical_state_and_tangent(
        latent, tangent, time_index=0
    )
    tf.debugging.assert_equal(away, [True])
    tf.debugging.assert_equal(physical, latent)
    tf.debugging.assert_equal(physical_tangent, tangent)


def test_candidate_factory_defaults_to_xla_jit() -> None:
    parameter = inspect.signature(
        candidate.make_latent_sir_contract_e_candidate
    ).parameters["jit_compile"]
    assert parameter.default is True


def test_registered_canonical_callable_matches_candidate_core() -> None:
    theta = tf.constant([0.03, -0.02, 0.04], DTYPE)
    prepared = _prepared_austria_for_identity_test()
    model = candidate.latent_preclip_zhao_cui_sir_austria_model()
    spec = candidate.static_spec_from_model(model)
    tensors = candidate._as_prepared_tensors(prepared, spec)
    expected = candidate.latent_sir_contract_e_value_and_score_core(
        theta,
        tensors,
        spec,
        steps=candidate.CANONICAL_STEPS,
        balance_steps=candidate.CANONICAL_BALANCE_STEPS,
        row_chunk_size=16,
        col_chunk_size=16,
    )
    actual = candidate.latent_sir_contract_e_canonical_value_and_score_tf.python_function(
        theta,
        tensors["observations"],
        tensors["initial_noise"],
        tensors["transition_noise"],
        tensors["fixed_reset_mask"],
        tensors["residual_design"],
        tensors["prepared_ridge"],
        tensors["epsilon"],
        tensors["scaling"],
    )
    for name in (
        "objective",
        "score",
        "per_batch_log_likelihood",
        "per_batch_score",
        "increment_history",
        "increment_score_history",
        "valid_chart",
        "reset_valid_history",
        "minimum_mass_history",
        "clip_boundary_away_history",
    ):
        tf.debugging.assert_equal(actual[name], expected[name])


def test_registered_two_node_route_accepts_t1_then_t2_trace_sequence() -> None:
    theta = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model = candidate.latent_preclip_two_node_spatial_sir_model()
    for horizon in (1, 2):
        prepared = _prepared_two_node_registered_route(horizon)
        result = candidate.latent_sir_two_node_contract_e_value_and_score_tf(
            theta,
            prepared["observations"],
            prepared["initial_noise"],
            prepared["transition_noise"],
            prepared["fixed_reset_mask"],
            prepared["residual_design"],
            prepared["prepared_ridge"],
            prepared["epsilon"],
            prepared["scaling"],
        )
        assert result["increment_history"].shape == (1, horizon)
        assert result["increment_score_history"].shape == (
            1,
            horizon,
            candidate.PARAMETER_COUNT,
        )
        for name in (
            "flow_valid_history",
            "geometry_valid_history",
            "quotient_valid_history",
            "reset_finite_history",
            "reset_factor_positive_history",
            "quotient_row_residual_history",
            "quotient_column_residual_history",
            "quotient_column_residual_scale_history",
            "quotient_post_column_residual_history",
        ):
            assert result[name].shape == (1, horizon)
        assert result["covariance_gap_eigenvalue_history"].shape == (
            1,
            horizon,
            model.state_dim(),
        )
        tf.debugging.assert_equal(
            result["reset_valid_history"],
            result["quotient_valid_history"]
            & result["reset_finite_history"]
            & result["reset_factor_positive_history"],
        )


def test_candidate_and_dense_reference_are_same_target_but_separate_programs() -> None:
    theta = tf.constant([0.03, -0.02, 0.04], DTYPE)
    model = reduced_latent_preclip_sir_model()
    grids = prepare_reduced_dense_grids(
        model, theta, time_steps=1, order=33, radius=7.0
    )
    reference = dense_latent_sir_value_and_manual_score(
        model, theta, _prepared()["observations"], grids
    )
    spec = candidate.static_spec_from_model(model)
    result = candidate.latent_sir_contract_e_value_and_score_core(
        theta,
        candidate._as_prepared_tensors(_prepared(), spec),
        spec,
        steps=2,
        row_chunk_size=8,
        col_chunk_size=8,
    )
    assert bool(tf.math.is_finite(reference["objective"]).numpy())
    assert bool(tf.math.is_finite(result["objective"]).numpy())
    assert candidate.CANDIDATE_STATUS == "candidate_not_canonical_not_admitted"


def test_particle_ladder_is_finite_and_records_descriptive_reference_gaps() -> None:
    theta = tf.constant([0.03, -0.02, 0.04], DTYPE)
    model = reduced_latent_preclip_sir_model()
    spec = candidate.static_spec_from_model(model)
    grids = prepare_reduced_dense_grids(
        model, theta, time_steps=1, order=33, radius=7.0
    )
    reference = dense_latent_sir_value_and_manual_score(
        model, theta, tf.constant([[0.15], [0.1]], DTYPE), grids
    )
    gaps = []
    for particle_count in (8, 16):
        prepared = _prepared_quantile(particle_count)
        result = candidate.latent_sir_contract_e_value_and_score_core(
            theta,
            candidate._as_prepared_tensors(prepared, spec),
            spec,
            steps=2,
            row_chunk_size=particle_count,
            col_chunk_size=particle_count,
        )
        assert bool(tf.reduce_all(result["valid_chart"]).numpy())
        assert bool(tf.reduce_all(tf.math.is_finite(result["score"])).numpy())
        gaps.append(abs(float((result["objective"] - reference["objective"]).numpy())))
    assert all(bool(tf.math.is_finite(gap).numpy()) for gap in gaps)
    # The ladder is descriptive. A finite-particle sequence need not be monotone.
    assert min(gaps) < 0.1
