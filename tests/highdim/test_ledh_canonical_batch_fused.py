"""Fused batch lane parity gates (NeuTra-eligibility prerequisite).

P-1 fused: batch-size-1 value AND score parity vs the single-cloud
canonical authority; multi-row independence; tf.function compilability
(the batch-native requirement). Declared tolerance: rtol 5e-4 (op-order),
recorded errors.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

DTYPE = tf.float64


def _single_model():
    def transition_mean_fn(theta, points):
        return points + theta[0] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return (
            tf.sin(points)
            + d_points
            + theta[0] * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def _fused_model():
    def transition_mean_fn(theta, points):
        # per-point theta rows [M, P]
        return points + theta[:, 0:1] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points, d_theta):
        return (
            d_theta[:, 0:1] * tf.sin(points)
            + d_points
            + theta[:, 0:1] * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def _fixture(seed: int, n: int = 6, horizon: int = 3):
    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((n, 2)), DTYPE)
    covs = tf.constant(np.stack([np.eye(2)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, 2)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, 2)), DTYPE)
    return initial, covs, noises, observations


def test_fused_batch_size_one_parity():
    initial, covs, noises, observations = _fixture(91)
    single = _single_model()
    value_ref, score_ref = canonical_value_and_analytical_score(
        single, tf.constant([0.6], DTYPE), initial, covs, noises,
        observations, flow_substeps=10, with_score=True,
    )
    fused = _fused_model()
    value_b, score_b, diag = canonical_batch_fused_value_score(
        fused,
        tf.constant([[0.6]], DTYPE),
        tf.constant([[1.0]], DTYPE),
        initial, covs, noises, observations,
        substeps=10,
    )
    assert bool(diag["program_valid"][0].numpy())
    v_err = abs(float(value_b[0].numpy()) - float(value_ref.numpy()))
    s_err = abs(float(score_b[0].numpy()) - float(score_ref[0].numpy()))
    assert v_err < 5.0e-4 * max(abs(float(value_ref.numpy())), 1.0), v_err
    assert s_err < 5.0e-4 * max(abs(float(score_ref[0].numpy())), 1.0), s_err


def test_fused_rows_independent_and_distinct():
    initial, covs, noises, observations = _fixture(93)
    fused = _fused_model()
    value_b, score_b, _ = canonical_batch_fused_value_score(
        fused,
        tf.constant([[0.6], [0.9]], DTYPE),
        tf.constant([[1.0], [1.0]], DTYPE),
        initial, covs, noises, observations,
        substeps=10,
    )
    value_row0, _, _ = canonical_batch_fused_value_score(
        fused,
        tf.constant([[0.6]], DTYPE),
        tf.constant([[1.0]], DTYPE),
        initial, covs, noises, observations,
        substeps=10,
    )
    assert abs(
        float(value_b[0].numpy()) - float(value_row0[0].numpy())
    ) < 1.0e-12, "row 0 changed when row 1 appended"
    assert abs(
        float(value_b[0].numpy()) - float(value_b[1].numpy())
    ) > 1.0e-6, "distinct theta rows produced identical values"


def test_fused_lane_is_tf_function_compilable():
    """Batch-native requirement: the fused program must trace under
    tf.function with no Python-loop-over-rows dependence on batch size
    (rows enter only through tensor shapes)."""
    initial, covs, noises, observations = _fixture(97)
    fused = _fused_model()
    compiled = tf.function(
        lambda t, d: canonical_batch_fused_value_score(
            fused, t, d, initial, covs, noises, observations, substeps=8
        ),
        autograph=False,
    )
    value_g, score_g, _ = compiled(
        tf.constant([[0.6], [0.8], [1.1]], DTYPE),
        tf.constant([[1.0], [1.0], [1.0]], DTYPE),
    )
    assert value_g.shape == (3,)
    assert bool(tf.reduce_all(tf.math.is_finite(value_g)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score_g)).numpy())


def test_fused_multi_direction_matches_swept():
    """Phase 2: K=P in one call must match P swept single-direction calls."""
    initial, covs, noises, observations = _fixture(101, n=6, horizon=3)
    fused = _fused_model()

    # One call with K=3 directions (standard basis)
    theta = tf.constant([[0.6]], DTYPE)
    directions = tf.constant([[[1.0], [0.0], [0.0]]], DTYPE)  # [B, K, P] with K=3, P=1

    # For P=1 model, we only have 1 parameter, so K=1 for meaningful test
    # Use K=3 with 3 different direction magnitudes
    directions = tf.constant([[[1.0], [0.5], [0.25]]], DTYPE)

    value_multi, score_multi, diag = canonical_batch_fused_value_score(
        fused, theta, directions, initial, covs, noises, observations, substeps=10
    )

    assert diag["program_valid"][0].numpy()
    assert score_multi.shape == (1, 3)

    # Swept calls
    swept_scores = []
    for k in range(3):
        _, score_k, _ = canonical_batch_fused_value_score(
            fused,
            theta,
            directions[:, k, :],  # [B, P]
            initial, covs, noises, observations,
            substeps=10,
        )
        swept_scores.append(float(score_k[0].numpy()))

    # Compare
    for k in range(3):
        err = abs(float(score_multi[0, k].numpy()) - swept_scores[k])
        rtol = 5.0e-4 * max(abs(swept_scores[k]), 1.0)
        assert err < rtol, f"direction {k}: error {err} exceeds rtol {rtol}"


def test_fused_multi_direction_rank_two_backward_compatible():
    """Phase 2: rank-2 theta_directions returns rank-1 score (backward compat)."""
    initial, covs, noises, observations = _fixture(103)
    fused = _fused_model()

    # Rank-2 input
    value, score, diag = canonical_batch_fused_value_score(
        fused,
        tf.constant([[0.6], [0.8]], DTYPE),
        tf.constant([[1.0], [1.0]], DTYPE),  # [B, P]
        initial, covs, noises, observations,
        substeps=10,
    )

    assert diag["program_valid"].shape == (2,)
    assert value.shape == (2,)
    assert score.shape == (2,), f"Expected rank-1 score, got shape {score.shape}"
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_fused_multi_direction_graph_compilable():
    """Phase 2: tf.function with K>1 must compile."""
    initial, covs, noises, observations = _fixture(107)
    fused = _fused_model()

    compiled = tf.function(
        lambda t, d: canonical_batch_fused_value_score(
            fused, t, d, initial, covs, noises, observations, substeps=8
        ),
        autograph=True,
        experimental_relax_shapes=True,
    )

    # K=3 directions
    directions = tf.constant([[[1.0], [0.5], [0.25]]], DTYPE)  # [B, K, P]
    value, score, _ = compiled(
        tf.constant([[0.7]], DTYPE),
        directions,
    )

    assert value.shape == (1,)
    assert score.shape == (1, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


_FULL_UNION = {
    "reset_policy": "contract_e",
    "reset_epsilon": 2.0,
    "reset_sinkhorn_steps": 8,
    "reset_balance_steps": 8,
    "reset_ridge": 1.0e-5,
    "correction_steps": 2,
    "correction_strength": 0.2,
    "correction_lm_damping": 1.0e-3,
    "correction_lm_scale_floor": 1.0e-6,
    "correction_trust_radius": 0.1,
    "pairwise_steps": 2,
    "pairwise_strength": 0.02,
    "pairwise_rms_cap": 2.0,
    "coordinate_cap": 0.98,
    "coordinate_cap_power": 8,
}


def _reset_design(n: int) -> tf.Tensor:
    basis = np.concatenate([np.eye(2), -np.eye(2)])
    return tf.constant(np.tile(basis, (n // 4, 1)), DTYPE)


def test_full_union_batch_size_one_multi_direction_parity():
    initial, covs, noises, observations = _fixture(211, n=16, horizon=2)
    design = _reset_design(16)
    theta = tf.constant([0.6], DTYPE)
    value_ref, score_ref = canonical_value_and_analytical_score(
        _single_model(),
        theta,
        initial,
        covs,
        noises,
        observations,
        flow_substeps=10,
        with_score=True,
        reset_design=design,
        **_FULL_UNION,
    )
    directions = tf.constant([[[1.0], [0.5], [0.25]]], DTYPE)
    value, score, diagnostics = canonical_batch_fused_value_score(
        _fused_model(),
        theta[None, :],
        directions,
        initial,
        covs,
        noises,
        observations,
        substeps=10,
        reset_design=design,
        **_FULL_UNION,
    )

    np.testing.assert_allclose(value[0].numpy(), value_ref.numpy(), rtol=5.0e-7)
    np.testing.assert_allclose(
        score[0].numpy(),
        score_ref[0].numpy() * directions[0, :, 0].numpy(),
        rtol=5.0e-7,
        atol=2.0e-12,
    )
    assert bool(diagnostics["program_valid"][0].numpy())


def test_full_union_rows_are_isolated_for_multiple_directions():
    initial, covs, noises, observations = _fixture(223, n=16, horizon=2)
    design = _reset_design(16)
    theta = tf.constant([[0.45], [0.8], [1.05]], DTYPE)
    directions = tf.constant(
        [[[1.0], [0.5]], [[0.25], [1.0]], [[0.75], [0.4]]], DTYPE
    )
    baseline_value, baseline_score, _ = canonical_batch_fused_value_score(
        _fused_model(),
        theta,
        directions,
        initial,
        covs,
        noises,
        observations,
        substeps=8,
        reset_design=design,
        **_FULL_UNION,
    )

    for changed_row in range(3):
        changed_theta = tf.tensor_scatter_nd_add(
            theta, [[changed_row, 0]], [tf.constant(0.17, DTYPE)]
        )
        changed_value, changed_score, _ = canonical_batch_fused_value_score(
            _fused_model(),
            changed_theta,
            directions,
            initial,
            covs,
            noises,
            observations,
            substeps=8,
            reset_design=design,
            **_FULL_UNION,
        )
        assert not np.array_equal(
            changed_value[changed_row].numpy(), baseline_value[changed_row].numpy()
        )
        for row in range(3):
            if row == changed_row:
                continue
            np.testing.assert_array_equal(
                changed_value[row].numpy(), baseline_value[row].numpy()
            )
            np.testing.assert_array_equal(
                changed_score[row].numpy(), baseline_score[row].numpy()
            )


def test_full_union_is_tf_function_compilable():
    initial, covs, noises, observations = _fixture(227, n=16, horizon=2)
    design = _reset_design(16)

    @tf.function(autograph=False)
    def compiled(theta, directions):
        return canonical_batch_fused_value_score(
            _fused_model(),
            theta,
            directions,
            initial,
            covs,
            noises,
            observations,
            substeps=8,
            reset_design=design,
            **_FULL_UNION,
        )

    value, score, diagnostics = compiled(
        tf.constant([[0.6], [0.9]], DTYPE),
        tf.constant([[[1.0], [0.5]], [[0.25], [1.0]]], DTYPE),
    )
    assert value.shape == (2,)
    assert score.shape == (2, 2)
    assert bool(tf.reduce_all(diagnostics["program_valid"]).numpy())


def test_annealed_full_union_multi_direction_parity_and_row_isolation():
    initial, covs, noises, observations = _fixture(229, n=16, horizon=2)
    design = _reset_design(16)
    theta = tf.constant([[0.55], [0.85]], DTYPE)
    directions = tf.constant([[[1.0], [0.4]], [[0.25], [1.0]]], DTYPE)
    annealed = {"annealed_stages": 3, "annealed_seed": 29}

    values, scores, diagnostics = canonical_batch_fused_value_score(
        _fused_model(),
        theta,
        directions,
        initial,
        covs,
        noises,
        observations,
        substeps=8,
        reset_design=design,
        **_FULL_UNION,
        **annealed,
    )

    for row in range(2):
        value_ref, score_ref = canonical_value_and_analytical_score(
            _single_model(),
            theta[row],
            initial,
            covs,
            noises,
            observations,
            flow_substeps=8,
            with_score=True,
            reset_design=design,
            **_FULL_UNION,
            **annealed,
        )
        np.testing.assert_allclose(values[row].numpy(), value_ref.numpy(), rtol=1.0e-12)
        np.testing.assert_allclose(
            scores[row].numpy(),
            score_ref[0].numpy() * directions[row, :, 0].numpy(),
            rtol=1.0e-12,
            atol=2.0e-12,
        )

    changed_theta = tf.tensor_scatter_nd_add(
        theta, [[1, 0]], [tf.constant(0.19, DTYPE)]
    )
    changed_values, changed_scores, changed_diagnostics = (
        canonical_batch_fused_value_score(
            _fused_model(),
            changed_theta,
            directions,
            initial,
            covs,
            noises,
            observations,
            substeps=8,
            reset_design=design,
            **_FULL_UNION,
            **annealed,
        )
    )
    np.testing.assert_array_equal(changed_values[0].numpy(), values[0].numpy())
    np.testing.assert_array_equal(changed_scores[0].numpy(), scores[0].numpy())
    assert not np.array_equal(changed_values[1].numpy(), values[1].numpy())
    assert bool(tf.reduce_all(diagnostics["program_valid"]).numpy())
    assert bool(tf.reduce_all(changed_diagnostics["program_valid"]).numpy())


def test_annealed_full_union_is_tf_function_compilable():
    initial, covs, noises, observations = _fixture(233, n=16, horizon=2)
    design = _reset_design(16)

    @tf.function(autograph=False)
    def compiled(theta, directions):
        return canonical_batch_fused_value_score(
            _fused_model(),
            theta,
            directions,
            initial,
            covs,
            noises,
            observations,
            substeps=8,
            reset_design=design,
            annealed_stages=3,
            annealed_seed=31,
            **_FULL_UNION,
        )

    values, scores, diagnostics = compiled(
        tf.constant([[0.6], [0.9]], DTYPE),
        tf.constant([[[1.0], [0.5]], [[0.25], [1.0]]], DTYPE),
    )
    assert values.shape == (2,)
    assert scores.shape == (2, 2)
    assert bool(tf.reduce_all(diagnostics["program_valid"]).numpy())

