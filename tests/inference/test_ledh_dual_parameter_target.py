"""Tests for dual-parameter LEDH HMC target (Corollary 5.2 surrogate-force).

Authority: Phase 3 Task 3.1a of ledh-surrogate-hmc-unified-program-2026-09-06.md
Date: 2026-09-11
"""

import numpy as np
import tensorflow as tf
import pytest

from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
    make_dual_parameter_target,
)
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import PerPointScoreModel


def _make_simple_lgssm_model(d: int = 3) -> PerPointScoreModel:
    """Create simple diagonal LGSSM for testing."""

    def transition_fn(theta, x_prev):
        phi = tf.nn.sigmoid(theta[:3])  # AR coefficients [3]
        return phi * x_prev  # Diagonal transition

    def observation_fn(theta, x):
        return x  # Identity observation

    def transition_score(theta, x_prev, x_curr):
        phi = tf.nn.sigmoid(theta[:3])
        return tf.concat([x_prev * (phi * (1 - phi)), tf.zeros([2], dtype=theta.dtype)], axis=0)

    def observation_score(theta, x, y):
        return tf.zeros_like(theta)

    return PerPointScoreModel(
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        transition_score=transition_score,
        observation_score=observation_score,
    )


@pytest.fixture
def lgssm_fixture():
    """LGSSM d=3 T=10 frozen fixture for testing."""
    d = 3
    T = 10
    N = 64

    model = _make_simple_lgssm_model(d)

    # True parameters
    theta_true = tf.constant([0.5, 0.3, 0.2, 0.1, 0.05], dtype=tf.float32)

    # Frozen particles
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=tf.float32) * 0.1
    initial_covariances = tf.tile(tf.eye(d, dtype=tf.float32)[None, :, :], [N, 1, 1]) * 0.01

    # Frozen noises
    noises = generator.normal([T, N, d], dtype=tf.float32) * 0.1

    # Frozen observations
    observations = generator.normal([T, d], dtype=tf.float32) * 0.5

    return model, theta_true, initial_states, initial_covariances, noises, observations


def test_dual_parameter_target_value_is_exact(lgssm_fixture):
    """Value uses exact parameters, not biased parameters."""
    model, theta_true, initial_states, initial_covariances, noises, observations = (
        lgssm_fixture
    )

    # Shared parameters
    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=tf.eye(3, dtype=tf.float32),
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=4,
        correction_strength=0.2,
        correction_trust_radius=0.5,
        pairwise_steps=0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    # Exact parameters
    exact_params = {
        **shared_params,
        'reset_ridge': 1e-5,
        'correction_lm_damping': 1e-2,
        'correction_lm_scale_floor': 1e-4,
    }

    # Biased parameters (100× larger damping)
    biased_params = {
        **shared_params,
        'reset_ridge': 1e-3,  # 100× larger
        'correction_lm_damping': 1.0,  # 100× larger
        'correction_lm_scale_floor': 1e-2,  # 100× larger
    }

    target = DualParameterLEDHTarget(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params=exact_params,
        biased_params=biased_params,
    )

    # Evaluate dual-parameter target
    value_dual, score_dual = target(theta_true)

    # Evaluate exact-only target
    value_exact_only = target.value_only(theta_true)

    # Value from dual target must match value from exact-only
    np.testing.assert_allclose(
        value_dual.numpy(),
        value_exact_only.numpy(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="Dual-parameter value does not match exact value",
    )


def test_dual_parameter_target_score_is_biased(lgssm_fixture):
    """Score uses biased parameters, differs from exact score."""
    model, theta_true, initial_states, initial_covariances, noises, observations = (
        lgssm_fixture
    )

    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=tf.eye(3, dtype=tf.float32),
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=4,
        correction_strength=0.2,
        correction_trust_radius=0.5,
        pairwise_steps=0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    exact_params = {
        **shared_params,
        'reset_ridge': 1e-5,
        'correction_lm_damping': 1e-2,
        'correction_lm_scale_floor': 1e-4,
    }

    biased_params = {
        **shared_params,
        'reset_ridge': 1e-3,
        'correction_lm_damping': 1.0,
        'correction_lm_scale_floor': 1e-2,
    }

    target = DualParameterLEDHTarget(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params=exact_params,
        biased_params=biased_params,
    )

    # Score from dual-parameter target
    _, score_biased = target(theta_true)

    # Score from exact parameters only
    score_exact = target.score_only(theta_true, exact=True)

    # Biased score must differ from exact score
    score_diff_norm = tf.norm(score_biased - score_exact).numpy()
    assert score_diff_norm > 1e-6, (
        f"Biased score is too close to exact score (diff={score_diff_norm}). "
        "Expected larger difference with 100× damping ratio."
    )


def test_dual_parameter_target_determinism(lgssm_fixture):
    """Same theta, same noises → same value and score (deterministic)."""
    model, theta_true, initial_states, initial_covariances, noises, observations = (
        lgssm_fixture
    )

    target = make_dual_parameter_target(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        damping_ratio=100.0,
        substeps=8,
        reset_policy="contract_e",
        reset_design=tf.eye(3, dtype=tf.float32),
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    # Evaluate twice
    value1, score1 = target(theta_true)
    value2, score2 = target(theta_true)

    # Must be bitwise identical (deterministic)
    np.testing.assert_array_equal(
        value1.numpy(),
        value2.numpy(),
        err_msg="Value is not deterministic",
    )

    np.testing.assert_array_equal(
        score1.numpy(),
        score2.numpy(),
        err_msg="Score is not deterministic",
    )


def test_dual_parameter_target_shape_consistency(lgssm_fixture):
    """Value and score have consistent shapes."""
    model, theta_true, initial_states, initial_covariances, noises, observations = (
        lgssm_fixture
    )

    target = make_dual_parameter_target(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        damping_ratio=100.0,
        substeps=8,
        reset_policy="contract_e",
        reset_design=tf.eye(3, dtype=tf.float32),
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    # Single theta [P]
    value_single, score_single = target(theta_true)
    assert value_single.shape.rank == 0, "Single theta should return scalar value"
    assert score_single.shape == theta_true.shape, (
        f"Score shape {score_single.shape} != theta shape {theta_true.shape}"
    )

    # Batch theta [B, P]
    theta_batch = tf.stack([theta_true, theta_true + 0.1], axis=0)
    value_batch, score_batch = target(theta_batch)
    assert value_batch.shape == (2,), f"Batch value shape {value_batch.shape} != (2,)"
    assert score_batch.shape == theta_batch.shape, (
        f"Batch score shape {score_batch.shape} != theta_batch shape {theta_batch.shape}"
    )


def test_make_dual_parameter_target_factory():
    """Factory function creates valid target with damping ratio."""
    d = 3
    T = 10
    N = 64
    model = _make_simple_lgssm_model(d)
    theta_true = tf.constant([0.5, 0.3, 0.2, 0.1, 0.05], dtype=tf.float32)

    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=tf.float32) * 0.1
    initial_covariances = tf.tile(tf.eye(d, dtype=tf.float32)[None, :, :], [N, 1, 1]) * 0.01
    noises = generator.normal([T, N, d], dtype=tf.float32) * 0.1
    observations = generator.normal([T, d], dtype=tf.float32) * 0.5

    # Test various damping ratios
    for damping_ratio in [1.0, 10.0, 100.0, 1000.0]:
        target = make_dual_parameter_target(
            model=model,
            initial_states=initial_states,
            initial_covariances=initial_covariances,
            noises=noises,
            observations=observations,
            damping_ratio=damping_ratio,
            base_reset_ridge=1e-5,
            base_lm_damping=1e-2,
            substeps=8,
            reset_policy="contract_e",
            reset_design=tf.eye(3, dtype=tf.float32),
            reset_epsilon=2.0,
            reset_sinkhorn_steps=8,
            reset_balance_steps=8,
            correction_steps=4,
            correction_strength=0.2,
            correction_lm_scale_floor=1e-4,
            correction_trust_radius=0.5,
            pairwise_steps=0,
            coordinate_cap=0.0,
            annealed_stages=1,
            annealed_seed=0,
        )

        # Verify parameters are set correctly
        assert target.exact_params['reset_ridge'] == 1e-5
        assert target.exact_params['correction_lm_damping'] == 1e-2
        assert target.biased_params['reset_ridge'] == 1e-5 * damping_ratio
        assert target.biased_params['correction_lm_damping'] == 1e-2 * damping_ratio

        # Should be callable
        value, score = target(theta_true)
        assert tf.math.is_finite(value)
        assert tf.reduce_all(tf.math.is_finite(score))


def test_dual_parameter_requires_frozen_noises():
    """Constructor rejects non-tensor noises (must be frozen)."""
    d = 3
    N = 64
    model = _make_simple_lgssm_model(d)

    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=tf.float32) * 0.1
    initial_covariances = tf.tile(tf.eye(d, dtype=tf.float32)[None, :, :], [N, 1, 1]) * 0.01
    observations = generator.normal([10, d], dtype=tf.float32) * 0.5

    with pytest.raises(TypeError, match="noises must be a Tensor"):
        DualParameterLEDHTarget(
            model=model,
            initial_states=initial_states,
            initial_covariances=initial_covariances,
            noises=None,  # Invalid: not a tensor
            observations=observations,
            exact_params={'reset_ridge': 1e-5},
            biased_params={'reset_ridge': 1e-3},
        )


def test_dual_parameter_requires_matching_keys():
    """Constructor rejects mismatched parameter keys."""
    d = 3
    T = 10
    N = 64
    model = _make_simple_lgssm_model(d)

    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=tf.float32) * 0.1
    initial_covariances = tf.tile(tf.eye(d, dtype=tf.float32)[None, :, :], [N, 1, 1]) * 0.01
    noises = generator.normal([T, N, d], dtype=tf.float32) * 0.1
    observations = generator.normal([T, d], dtype=tf.float32) * 0.5

    with pytest.raises(ValueError, match="Parameter key mismatch"):
        DualParameterLEDHTarget(
            model=model,
            initial_states=initial_states,
            initial_covariances=initial_covariances,
            noises=noises,
            observations=observations,
            exact_params={'reset_ridge': 1e-5, 'substeps': 8},
            biased_params={'reset_ridge': 1e-3},  # Missing 'substeps'
        )
