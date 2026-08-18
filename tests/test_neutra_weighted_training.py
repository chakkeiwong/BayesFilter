"""Analytic mechanics tests for weighted forward-KL NeuTra training."""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    MatchedReverseKLNeuTraTrainer,
    WEIGHTED_NEUTRA_NONCLAIMS,
    WeightedDenseIAFTransport,
    WeightedForwardKLNeuTraTrainer,
    WeightedNeuTraConfig,
)


DTYPE = tf.float64


def _config(**overrides) -> WeightedNeuTraConfig:
    values = {
        "dimension": 2,
        "hidden_layers": (8, 8),
        "stages": 2,
        "activation": "tanh",
        "initialization_scale": 0.02,
        "initialization_seed": (20260811, 9101),
        "learning_rate": 1.0e-3,
        "jit_compile": True,
    }
    values.update(overrides)
    return WeightedNeuTraConfig(**values)


def test_initial_transport_preserves_standard_normal_density() -> None:
    transport = WeightedDenseIAFTransport(_config())
    rows = tf.constant(((0.0, 0.0), (1.0, -2.0), (-0.5, 0.25)), DTYPE)
    physical, logdet = transport.forward_and_logdet(rows)
    recovered, inverse_forward_logdet = transport.inverse_and_forward_logdet(physical)
    expected = -0.5 * (
        tf.reduce_sum(tf.square(rows), axis=1)
        + tf.constant(2.0 * math.log(2.0 * math.pi), DTYPE)
    )
    # Two identity autoregressive stages contain one fixed coordinate reversal.
    tf.debugging.assert_near(
        physical, tf.reverse(rows, axis=(-1,)), atol=1.0e-15, rtol=1.0e-15
    )
    tf.debugging.assert_near(recovered, rows, atol=1.0e-15, rtol=1.0e-15)
    tf.debugging.assert_near(logdet, tf.zeros(3, DTYPE), atol=1.0e-15, rtol=0.0)
    tf.debugging.assert_near(
        inverse_forward_logdet, tf.zeros(3, DTYPE), atol=1.0e-15, rtol=0.0
    )
    tf.debugging.assert_near(transport.log_prob(rows), expected, atol=1.0e-14)


def test_zero_initialized_scale_linear_skip_preserves_identity() -> None:
    config = _config(
        stages=2,
        stage_scale_linear_skip=(True, False),
    )
    transport = WeightedDenseIAFTransport(config)
    rows = tf.constant(((0.0, 0.0), (1.0, -2.0), (-0.5, 0.25)), DTYPE)
    physical, logdet = transport.forward_and_logdet(rows)
    tf.debugging.assert_near(
        physical, tf.reverse(rows, axis=(-1,)), atol=1.0e-15, rtol=1.0e-15
    )
    tf.debugging.assert_near(logdet, tf.zeros(3, DTYPE), atol=1.0e-15, rtol=0.0)
    assert transport.stages[0].scale_linear_skip_weight is not None
    tf.debugging.assert_equal(
        transport.stages[0].scale_linear_skip_weight,
        tf.zeros((2, 2), DTYPE),
    )
    assert transport.stages[1].scale_linear_skip_weight is None


def test_scale_linear_skip_mask_is_strictly_autoregressive() -> None:
    transport = WeightedDenseIAFTransport(
        _config(
            dimension=4,
            stages=1,
            hidden_layers=(8,),
            stage_scale_linear_skip=(True,),
        )
    )
    stage = transport.stages[0]
    expected_mask = tf.constant(
        (
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        DTYPE,
    )
    tf.debugging.assert_equal(stage.scale_linear_skip_mask, expected_mask)
    assert stage.scale_linear_skip_weight is not None
    stage.scale_linear_skip_weight.assign(tf.ones((4, 4), DTYPE))
    values = tf.constant(((1.0, 10.0, 100.0, 1000.0),), DTYPE)
    tf.debugging.assert_equal(
        stage._scale_linear_skip(values),
        tf.constant(((0.0, 1.0, 11.0, 111.0),), DTYPE),
    )


def test_inverse_roundtrip_and_logdet_hold_after_parameter_perturbation() -> None:
    transport = WeightedDenseIAFTransport(_config(stages=3))
    for index, variable in enumerate(transport.trainable_variables):
        seed = tf.constant((20260811, 9200 + index), tf.int32)
        variable.assign(
            variable
            + tf.random.stateless_normal(variable.shape, seed=seed, dtype=DTYPE) * 0.03
        )
    latent = tf.random.stateless_normal((17, 2), seed=(20260811, 9301), dtype=DTYPE)
    physical, forward_logdet = transport.forward_and_logdet(latent)
    recovered, recovered_forward_logdet = transport.inverse_and_forward_logdet(physical)
    tf.debugging.assert_near(recovered, latent, atol=2.0e-12, rtol=2.0e-12)
    tf.debugging.assert_near(
        recovered_forward_logdet, forward_logdet, atol=2.0e-12, rtol=2.0e-12
    )


def test_stage_specific_scale_caps_preserve_roundtrip_and_scores() -> None:
    config = _config(stages=3, s_max=1.0, stage_s_max=(3.0, 0.5, 0.5))
    transport = WeightedDenseIAFTransport(config)
    assert [stage.s_max for stage in transport.stages] == [3.0, 0.5, 0.5]
    assert config.manifest_payload()["stage_s_max"] == [3.0, 0.5, 0.5]
    for index, variable in enumerate(transport.trainable_variables):
        variable.assign_add(
            tf.random.stateless_normal(
                variable.shape, seed=(20260814, 9300 + index), dtype=DTYPE
            )
            * tf.constant(0.02, DTYPE)
        )
    latent = tf.random.stateless_normal((11, 2), seed=(20260814, 9401), dtype=DTYPE)
    output_score = tf.random.stateless_normal((11, 2), seed=(20260814, 9402), dtype=DTYPE)
    physical, logdet = transport.forward_and_logdet(latent)
    recovered, inverse_logdet = transport.inverse_and_forward_logdet(physical)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent)
        mapped = transport.forward_batch(latent)
        pullback_objective = tf.reduce_sum(mapped * output_score)
        logdet_objective = tf.reduce_sum(transport.log_abs_det_jacobian_batch(latent))
    expected_pullback = tape.gradient(pullback_objective, latent)
    expected_logdet_score = tape.gradient(logdet_objective, latent)
    tf.debugging.assert_near(recovered, latent, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(inverse_logdet, logdet, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(
        transport.pullback_score_batch(latent, output_score),
        expected_pullback,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    tf.debugging.assert_near(
        transport.log_abs_det_jacobian_score_batch(latent),
        expected_logdet_score,
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_scale_linear_skip_preserves_roundtrip_and_matches_autodiff_scores() -> None:
    config = _config(
        stages=3,
        s_max=1.0,
        stage_s_max=(4.0, 0.5, 0.5),
        stage_scale_linear_skip=(True, False, False),
    )
    transport = WeightedDenseIAFTransport(config)
    assert config.manifest_payload()["stage_scale_linear_skip"] == [True, False, False]
    for index, variable in enumerate(transport.trainable_variables):
        variable.assign_add(
            tf.random.stateless_normal(
                variable.shape, seed=(20260814, 9500 + index), dtype=DTYPE
            )
            * tf.constant(0.02, DTYPE)
        )
    latent = tf.random.stateless_normal((13, 2), seed=(20260814, 9601), dtype=DTYPE)
    output_score = tf.random.stateless_normal((13, 2), seed=(20260814, 9602), dtype=DTYPE)
    physical, logdet = transport.forward_and_logdet(latent)
    recovered, inverse_logdet = transport.inverse_and_forward_logdet(physical)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent)
        mapped = transport.forward_batch(latent)
        pullback_objective = tf.reduce_sum(mapped * output_score)
        logdet_objective = tf.reduce_sum(transport.log_abs_det_jacobian_batch(latent))
    tf.debugging.assert_near(recovered, latent, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(inverse_logdet, logdet, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(
        transport.pullback_score_batch(latent, output_score),
        tape.gradient(pullback_objective, latent),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    tf.debugging.assert_near(
        transport.log_abs_det_jacobian_score_batch(latent),
        tape.gradient(logdet_objective, latent),
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_unbounded_scale_linear_exactly_represents_funnel_map() -> None:
    dimension = 4
    transport = WeightedDenseIAFTransport(
        _config(
            dimension=dimension,
            stages=1,
            hidden_layers=(8,),
            s_max=0.5,
            stage_unbounded_scale_linear=(True,),
        )
    )
    stage = transport.stages[0]
    assert stage.unbounded_scale_linear_weight is not None
    exact_weight = tf.concat(
        (
            tf.concat(
                (
                    tf.zeros((1, 1), DTYPE),
                    tf.ones((1, dimension - 1), DTYPE),
                ),
                axis=1,
            ),
            tf.zeros((dimension - 1, dimension), DTYPE),
        ),
        axis=0,
    )
    stage.unbounded_scale_linear_weight.assign(exact_weight)
    latent = tf.random.stateless_normal(
        (17, dimension), seed=(20260814, 9651), dtype=DTYPE
    )
    physical, logdet = transport.forward_and_logdet(latent)
    expected = tf.concat(
        (
            latent[:, :1],
            latent[:, 1:] * tf.exp(latent[:, :1]),
        ),
        axis=1,
    )
    recovered, inverse_logdet = transport.inverse_and_forward_logdet(physical)
    tf.debugging.assert_near(physical, expected, atol=2.0e-14, rtol=2.0e-14)
    tf.debugging.assert_near(
        logdet,
        tf.cast(dimension - 1, DTYPE) * latent[:, 0],
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    tf.debugging.assert_near(recovered, latent, atol=2.0e-13, rtol=2.0e-13)
    tf.debugging.assert_near(inverse_logdet, logdet, atol=2.0e-13, rtol=2.0e-13)


@pytest.mark.parametrize(
    "stages, hidden_width, permutation_policy, stage_caps",
    (
        (1, 100, "full_reverse", (4.0,)),
        (3, 100, "root_preserving_reverse", (4.0, 0.5, 0.5)),
        (3, 100, "full_reverse", (4.0, 0.5, 0.5)),
        (3, 200, "root_preserving_reverse", (4.0, 0.5, 0.5)),
    ),
)
def test_campaign_architectures_contain_exact_funnel_map(
    stages: int,
    hidden_width: int,
    permutation_policy: str,
    stage_caps: tuple[float, ...],
) -> None:
    dimension = 100
    transport = WeightedDenseIAFTransport(
        _config(
            dimension=dimension,
            hidden_layers=(hidden_width, hidden_width),
            stages=stages,
            stage_s_max=stage_caps,
            stage_unbounded_scale_linear=(True,) + (False,) * (stages - 1),
            permutation_policy=permutation_policy,
        )
    )
    first_stage = transport.stages[0]
    assert first_stage.unbounded_scale_linear_weight is not None
    exact_weight = tf.concat(
        (
            tf.concat(
                (tf.zeros((1, 1), DTYPE), tf.ones((1, dimension - 1), DTYPE)),
                axis=1,
            ),
            tf.zeros((dimension - 1, dimension), DTYPE),
        ),
        axis=0,
    )
    first_stage.unbounded_scale_linear_weight.assign(exact_weight)
    latent = tf.random.stateless_normal(
        (17, dimension), seed=(20260815, 1301), dtype=DTYPE
    )
    physical, logdet = transport.forward_and_logdet(latent)
    expected = tf.concat(
        (latent[:, :1], latent[:, 1:] * tf.exp(latent[:, :1])), axis=1
    )
    recovered, inverse_logdet = transport.inverse_and_forward_logdet(physical)
    tf.debugging.assert_near(physical, expected, atol=3.0e-13, rtol=3.0e-13)
    tf.debugging.assert_near(
        logdet,
        tf.cast(dimension - 1, DTYPE) * latent[:, 0],
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    tf.debugging.assert_near(recovered, latent, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(inverse_logdet, logdet, atol=3.0e-12, rtol=3.0e-12)


def test_unbounded_scale_linear_matches_autodiff_scores() -> None:
    transport = WeightedDenseIAFTransport(
        _config(
            stages=3,
            stage_s_max=(4.0, 0.5, 0.5),
            stage_unbounded_scale_linear=(True, False, False),
        )
    )
    for index, variable in enumerate(transport.trainable_variables):
        variable.assign_add(
            tf.random.stateless_normal(
                variable.shape, seed=(20260814, 9660 + index), dtype=DTYPE
            )
            * tf.constant(0.02, DTYPE)
        )
    latent = tf.random.stateless_normal((13, 2), seed=(20260814, 9681), dtype=DTYPE)
    output_score = tf.random.stateless_normal((13, 2), seed=(20260814, 9682), dtype=DTYPE)
    physical, logdet = transport.forward_and_logdet(latent)
    recovered, inverse_logdet = transport.inverse_and_forward_logdet(physical)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent)
        mapped = transport.forward_batch(latent)
        pullback_objective = tf.reduce_sum(mapped * output_score)
        logdet_objective = tf.reduce_sum(transport.log_abs_det_jacobian_batch(latent))
    tf.debugging.assert_near(recovered, latent, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(inverse_logdet, logdet, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(
        transport.pullback_score_batch(latent, output_score),
        tape.gradient(pullback_objective, latent),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    tf.debugging.assert_near(
        transport.log_abs_det_jacobian_score_batch(latent),
        tape.gradient(logdet_objective, latent),
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_unbounded_scale_linear_config_and_variables_roundtrip() -> None:
    config = _config(
        stages=3,
        stage_s_max=(4.0, 0.5, 0.5),
        stage_unbounded_scale_linear=(True, False, False),
    )
    source = WeightedDenseIAFTransport(config)
    for index, variable in enumerate(source.trainable_variables):
        variable.assign_add(
            tf.random.stateless_normal(
                variable.shape, seed=(20260814, 9700 + index), dtype=DTYPE
            )
            * tf.constant(0.01, DTYPE)
        )
    payload = dict(config.manifest_payload())
    payload.pop("schema")
    restored = WeightedDenseIAFTransport(WeightedNeuTraConfig(**payload))
    for destination, value in zip(
        restored.trainable_variables, source.trainable_variables, strict=True
    ):
        destination.assign(value)
    latent = tf.random.stateless_normal((7, 2), seed=(20260814, 9801), dtype=DTYPE)
    source_physical, source_logdet = source.forward_and_logdet(latent)
    restored_physical, restored_logdet = restored.forward_and_logdet(latent)
    tf.debugging.assert_equal(restored_physical, source_physical)
    tf.debugging.assert_equal(restored_logdet, source_logdet)


@pytest.mark.parametrize(
    "permutation_policy",
    ("full_reverse", "root_preserving_reverse"),
)
def test_permutation_policy_preserves_roundtrip_and_matches_autodiff_scores(
    permutation_policy: str,
) -> None:
    transport = WeightedDenseIAFTransport(
        _config(
            dimension=4,
            hidden_layers=(8, 8),
            stages=3,
            stage_s_max=(4.0, 0.5, 0.5),
            stage_unbounded_scale_linear=(True, False, False),
            permutation_policy=permutation_policy,
        )
    )
    for index, variable in enumerate(transport.trainable_variables):
        variable.assign_add(
            tf.random.stateless_normal(
                variable.shape, seed=(20260815, 1100 + index), dtype=DTYPE
            )
            * tf.constant(0.02, DTYPE)
        )
    latent = tf.random.stateless_normal((13, 4), seed=(20260815, 1201), dtype=DTYPE)
    output_score = tf.random.stateless_normal(
        (13, 4), seed=(20260815, 1202), dtype=DTYPE
    )
    physical, logdet = transport.forward_and_logdet(latent)
    recovered, inverse_logdet = transport.inverse_and_forward_logdet(physical)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent)
        mapped = transport.forward_batch(latent)
        pullback_objective = tf.reduce_sum(mapped * output_score)
        logdet_objective = tf.reduce_sum(
            transport.log_abs_det_jacobian_batch(latent)
        )
    tf.debugging.assert_near(recovered, latent, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(inverse_logdet, logdet, atol=3.0e-12, rtol=3.0e-12)
    tf.debugging.assert_near(
        transport.pullback_score_batch(latent, output_score),
        tape.gradient(pullback_objective, latent),
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    tf.debugging.assert_near(
        transport.log_abs_det_jacobian_score_batch(latent),
        tape.gradient(logdet_objective, latent),
        atol=3.0e-11,
        rtol=3.0e-11,
    )


def test_root_preserving_permutation_keeps_root_and_reverses_children() -> None:
    transport = WeightedDenseIAFTransport(
        _config(
            dimension=5,
            stages=2,
            permutation_policy="root_preserving_reverse",
        )
    )
    rows = tf.constant(((1.0, 2.0, 3.0, 4.0, 5.0),), DTYPE)
    physical, logdet = transport.forward_and_logdet(rows)
    tf.debugging.assert_equal(
        physical, tf.constant(((1.0, 5.0, 4.0, 3.0, 2.0),), DTYPE)
    )
    tf.debugging.assert_equal(logdet, tf.zeros((1,), DTYPE))


def test_invalid_permutation_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported permutation_policy"):
        _config(permutation_policy="rotate")


@pytest.mark.parametrize(
    "stage_s_max, message",
    (
        ((3.0, 0.5), "match stages"),
        ((3.0, 0.0, 0.5), "finite and positive"),
    ),
)
def test_stage_specific_scale_caps_reject_invalid_values(stage_s_max, message) -> None:
    with pytest.raises(ValueError, match=message):
        _config(stages=3, stage_s_max=stage_s_max)


@pytest.mark.parametrize(
    "stage_scale_linear_skip, message",
    (
        ((True, False), "match stages"),
        ((True, 1, False), "must be booleans"),
    ),
)
def test_stage_scale_linear_skip_rejects_invalid_values(
    stage_scale_linear_skip, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(stages=3, stage_scale_linear_skip=stage_scale_linear_skip)


@pytest.mark.parametrize(
    "stage_unbounded_scale_linear, message",
    (
        ((True, False), "match stages"),
        ((True, 1, False), "must be booleans"),
    ),
)
def test_stage_unbounded_scale_linear_rejects_invalid_values(
    stage_unbounded_scale_linear, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(stages=3, stage_unbounded_scale_linear=stage_unbounded_scale_linear)


def test_pre_cap_and_unbounded_scale_linear_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(
            stages=2,
            stage_scale_linear_skip=(True, False),
            stage_unbounded_scale_linear=(True, False),
        )


def test_weighted_loss_matches_manual_reduction_and_weight_diagnostics() -> None:
    trainer = WeightedForwardKLNeuTraTrainer(_config())
    rows = tf.constant(((0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (-1.0, -1.0)), DTYPE)
    log_weights = tf.math.log(tf.constant((0.1, 0.2, 0.3, 0.4), DTYPE))
    validation = trainer.validation_batch(rows, log_weights)
    normalized = tf.nn.softmax(log_weights)
    expected = tf.reduce_sum(normalized * -trainer.log_prob(rows))
    tf.debugging.assert_near(validation.loss, expected, atol=1.0e-13)
    tf.debugging.assert_near(validation.normalized_weights, normalized, atol=1.0e-14)
    tf.debugging.assert_near(
        validation.effective_sample_size,
        tf.math.reciprocal(tf.reduce_sum(tf.square(normalized))),
        atol=1.0e-13,
    )
    assert float(validation.maximum_normalized_weight.numpy()) == pytest.approx(0.4)


def test_validation_reuses_inverse_identity_without_changing_log_density() -> None:
    trainer = WeightedForwardKLNeuTraTrainer(_config(jit_compile=False))
    rows = tf.random.stateless_normal((9, 2), seed=(20260814, 9201), dtype=DTYPE)
    weights = tf.zeros((9,), DTYPE)
    validation = trainer.validation_batch(rows, weights)
    latent, forward_logdet = trainer.inverse_and_forward_logdet(rows)
    expected_negative_log_prob = (
        0.5 * tf.reduce_sum(tf.square(latent), axis=-1)
        + tf.constant(math.log(2.0 * math.pi), DTYPE)
        + forward_logdet
    )
    tf.debugging.assert_near(
        validation.per_sample_negative_log_prob,
        expected_negative_log_prob,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    tf.debugging.assert_near(
        validation.per_sample_negative_log_prob,
        -trainer.log_prob(rows),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_weighted_gradient_matches_finite_difference() -> None:
    trainer = WeightedForwardKLNeuTraTrainer(_config(jit_compile=False))
    rows = tf.constant(((0.3, -0.7), (1.2, 0.5), (-1.1, 0.8), (0.4, 1.3)), DTYPE)
    log_weights = tf.math.log(tf.constant((0.1, 0.2, 0.3, 0.4), DTYPE))
    variable = trainer.variables[-1]
    index = 2
    with tf.GradientTape() as tape:
        loss = trainer.validation_batch(rows, log_weights).loss
    analytic = tape.gradient(loss, variable)[index]
    original = variable.read_value()
    epsilon = tf.constant(1.0e-5, DTYPE)
    direction = tf.one_hot(index, variable.shape[0], dtype=DTYPE)
    variable.assign(original + epsilon * direction)
    plus = trainer.validation_batch(rows, log_weights).loss
    variable.assign(original - epsilon * direction)
    minus = trainer.validation_batch(rows, log_weights).loss
    variable.assign(original)
    numeric = (plus - minus) / (2.0 * epsilon)
    tf.debugging.assert_near(analytic, numeric, atol=2.0e-7, rtol=2.0e-6)


def test_xla_weighted_update_is_finite_and_replays_from_same_initialization() -> None:
    rows = tf.random.stateless_normal((32, 2), seed=(20260811, 9401), dtype=DTYPE)
    log_weights = tf.linspace(tf.constant(-2.0, DTYPE), tf.constant(1.0, DTYPE), 32)
    left = WeightedForwardKLNeuTraTrainer(_config())
    right = WeightedForwardKLNeuTraTrainer(_config())
    first = left.train_step(rows, log_weights)
    replay = right.train_step(rows, log_weights)
    tf.debugging.assert_near(first.loss, replay.loss, atol=1.0e-14, rtol=1.0e-14)
    assert int(first.step.numpy()) == 1
    assert bool(tf.math.is_finite(first.gradient_norm).numpy())
    for left_variable, right_variable in zip(left.variables, right.variables):
        tf.debugging.assert_near(
            left_variable, right_variable, atol=1.0e-14, rtol=1.0e-14
        )
    assert "weighted particles are not an unweighted posterior archive" in WEIGHTED_NEUTRA_NONCLAIMS


def test_invalid_weights_and_shapes_fail_closed() -> None:
    trainer = WeightedForwardKLNeuTraTrainer(_config(jit_compile=False))
    with pytest.raises(ValueError, match="log_weights"):
        trainer.train_step(tf.zeros((4, 2), DTYPE), tf.zeros(3, DTYPE))
    with pytest.raises(tf.errors.InvalidArgumentError):
        trainer.validation_batch(
            tf.zeros((4, 2), DTYPE),
            tf.constant((0.0, 0.0, float("nan"), 0.0), DTYPE),
        )


def test_matched_reverse_kl_comparator_uses_same_transport_and_xla() -> None:
    config = _config()

    def target(rows: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(rows - 0.5), axis=1)

    trainer = MatchedReverseKLNeuTraTrainer(config, target)
    latent = tf.random.stateless_normal((32, 2), seed=(20260811, 9501), dtype=DTYPE)
    result = trainer.train_step(latent)
    assert int(result.step.numpy()) == 1
    assert bool(tf.math.is_finite(result.loss).numpy())
    assert len(trainer.variables) == len(WeightedDenseIAFTransport(config).trainable_variables)
