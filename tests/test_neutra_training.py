from __future__ import annotations

import json
from pathlib import Path
import os
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_batching import (
    batch_native_value_status_target_fn,
    bind_batch_native_neutra_target,
)
from bayesfilter.inference.neutra_training import (
    NeuTraTrainingError,
    PlainDenseIAFTrainingConfig,
    PlainDenseIAFTransport,
    train_plain_dense_iaf,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    load_deterministic_lgssm_exact_target,
)


TARGET_SIGNATURE = "a" * 64


class GaussianAdapter:
    parameter_dim = 3

    def __init__(self) -> None:
        self.call_count = 0

    def value_score_capability(self):
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="test_gaussian",
            evidence_path="tests/test_neutra_training.py",
            target_scope="test_gaussian",
        )

    def log_prob_and_grad(self, theta):
        self.call_count += 1
        values = tf.convert_to_tensor(theta, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

    def neutra_batch_log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        self.call_count += 1
        value = -0.5 * tf.reduce_sum(tf.square(values), axis=-1)
        score = -values
        leading = tf.shape(values)[:-1]
        return value, score, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
            "innovation_condition_estimate": tf.ones(leading, tf.float64),
        }


class StatusGaussianAdapter(GaussianAdapter):
    def __init__(self, *, valid: bool = True) -> None:
        super().__init__()
        self.valid = bool(valid)

    def log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        value, score = self.log_prob_and_grad(values)
        leading = tf.shape(values)[:-1]
        valid = tf.fill(leading, tf.constant(self.valid))
        return value, score, {
            "status_code": tf.where(
                valid,
                tf.zeros(leading, tf.int32),
                tf.ones(leading, tf.int32),
            ),
            "valid_pre_regularized_score": valid,
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
            "innovation_condition_estimate": tf.ones(leading, tf.float64),
        }


    def neutra_batch_log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        self.call_count += 1
        value = -0.5 * tf.reduce_sum(tf.square(values), axis=-1)
        score = -values
        leading = tf.shape(values)[:-1]
        valid = tf.fill(leading, tf.constant(self.valid))
        return value, score, {
            "status_code": tf.where(
                valid,
                tf.zeros(leading, tf.int32),
                tf.ones(leading, tf.int32),
            ),
            "valid_pre_regularized_score": valid,
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
            "innovation_condition_estimate": tf.ones(leading, tf.float64),
        }


class MissingBatchMethodAdapter(GaussianAdapter):
    neutra_batch_log_prob_and_grad_status = None


class NoConditionEstimateGaussianAdapter(GaussianAdapter):
    def neutra_batch_log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        value = -0.5 * tf.reduce_sum(tf.square(values), axis=-1)
        leading = tf.shape(values)[:-1]
        return value, -values, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
        }


def _config(tmp_path, *, output="run", steps=6):
    return PlainDenseIAFTrainingConfig(
        target_signature=TARGET_SIGNATURE,
        dimension=3,
        affine_center=(0.1, -0.2, 0.3),
        affine_factor=((1.2, 0.0, 0.0), (0.1, 0.9, 0.0), (-0.2, 0.05, 1.1)),
        output_dir=tmp_path / output,
        seed=(20260713, 17),
        hidden_layers=(4, 4),
        stage_count=3,
        steps=steps,
        batch_size=8,
        learning_rate=1.0e-3,
        checkpoint_every=3,
        heartbeat_every=1,
        device="/CPU:0",
        require_gpu=False,
    )


def _exact_config(tmp_path, *, output: str, steps: int):
    bundle = load_deterministic_lgssm_exact_target()
    dimension = bundle.adapter.parameter_dim
    center = tuple(float(value) for value in bundle.raw_truth.numpy())
    config = PlainDenseIAFTrainingConfig(
        target_signature=bundle.target_signature,
        dimension=dimension,
        affine_center=center,
        affine_factor=tuple(
            tuple(0.01 * float(row == column) for column in range(dimension))
            for row in range(dimension)
        ),
        output_dir=tmp_path / output,
        seed=(20260714, 1),
        hidden_layers=(2,),
        stage_count=1,
        steps=steps,
        batch_size=2,
        learning_rate=1.0e-3,
        checkpoint_every=steps,
        heartbeat_every=1,
        device="/CPU:0",
        require_gpu=False,
    )
    return bundle, config


def test_training_rejects_missing_batch_binding_before_side_effect(tmp_path) -> None:
    adapter = MissingBatchMethodAdapter()
    config = _config(tmp_path, output="missing-batch-binding", steps=1)

    with pytest.raises(NeuTraTrainingError, match="requires bound method"):
        train_plain_dense_iaf(adapter=adapter, config=config)

    assert adapter.call_count == 0
    assert not config.output_dir.exists()


def test_training_runs_exact_batch_native_lgssm_one_step(tmp_path) -> None:
    bundle, config = _exact_config(
        tmp_path, output="batch-native-lgssm", steps=1
    )

    result = train_plain_dense_iaf(adapter=bundle.adapter, config=config)

    assert result.completed_steps == 1
    binding = result.runtime_metadata["batch_native_target"]
    assert binding["method_name"] == (
        "neutra_batch_log_prob_and_grad_status"
    )
    assert binding["scalar_fallback_used"] is False
    assert binding["row_mapped_scalar_target_used"] is False
    assert result.runtime_metadata["sample_axis_python_loop_used"] is False
    assert result.records[0]["target_status_all_valid"] is True


def test_training_records_missing_condition_estimate_as_unavailable(tmp_path) -> None:
    config = _config(tmp_path, output="no-condition-estimate", steps=1)

    result = train_plain_dense_iaf(
        adapter=NoConditionEstimateGaussianAdapter(), config=config
    )

    assert result.records[0]["target_condition_estimate_available"] is False
    assert result.records[0]["target_max_innovation_condition_estimate"] is None


def test_exact_target_reverse_kl_gradient_matches_finite_difference(tmp_path) -> None:
    bundle, config = _exact_config(
        tmp_path, output="exact-gradient", steps=1
    )
    flow = PlainDenseIAFTransport(config)
    binding = bind_batch_native_neutra_target(
        bundle.adapter,
        target_signature=bundle.target_signature,
    )
    target = batch_native_value_status_target_fn(binding)
    z = tf.random.stateless_normal(
        (config.batch_size, config.dimension),
        seed=(20260714, 41),
        dtype=tf.float64,
    )

    def exact_loss(use_reviewed_gradient: bool):
        theta, logdet = flow.forward_and_logdet(z)
        if use_reviewed_gradient:
            value, _status = target(theta)
        else:
            value, _score, _status = (
                bundle.adapter.neutra_batch_log_prob_and_grad_status(theta)
            )
        return -tf.reduce_mean(value + logdet)

    variable = flow.trainable_variables[-1]
    index = 18
    with tf.GradientTape() as tape:
        loss = exact_loss(True)
    observed = tape.gradient(loss, variable)[index]
    original = variable.read_value()
    epsilon = tf.constant(1.0e-5, tf.float64)
    direction = tf.one_hot(index, tf.shape(variable)[0], dtype=tf.float64)
    variable.assign(original + epsilon * direction)
    plus = exact_loss(False)
    variable.assign(original - epsilon * direction)
    minus = exact_loss(False)
    variable.assign(original)
    numeric = (plus - minus) / (2.0 * epsilon)

    assert float(observed.numpy()) == pytest.approx(
        float(numeric.numpy()), rel=3e-5, abs=3e-6
    )


def test_exact_target_five_step_training_is_deterministic(tmp_path) -> None:
    left_bundle, left_config = _exact_config(
        tmp_path, output="exact-five-left", steps=5
    )
    right_bundle, right_config = _exact_config(
        tmp_path, output="exact-five-right", steps=5
    )
    left = train_plain_dense_iaf(
        adapter=left_bundle.adapter,
        config=left_config,
    )
    right = train_plain_dense_iaf(
        adapter=right_bundle.adapter,
        config=right_config,
    )
    left_state = json.loads(left.state_path.read_text(encoding="utf-8"))
    right_state = json.loads(right.state_path.read_text(encoding="utf-8"))

    assert left_state["trainable_variables"] == right_state["trainable_variables"]
    assert left_state["adam_first_moments"] == right_state["adam_first_moments"]
    assert left_state["adam_second_moments"] == right_state["adam_second_moments"]
    assert left_state["completed_steps"] == right_state["completed_steps"] == 5
    assert all(row["target_status_all_valid"] for row in left.records)
    assert left.runtime_metadata["compiled_training_program_invocations"] == 1
    assert left.runtime_metadata["program_step_count"] == 5


def test_training_config_rejects_singleton_batch_before_output(tmp_path) -> None:
    output = tmp_path / "singleton-batch"
    with pytest.raises(ValueError, match="greater than one"):
        replace(_config(tmp_path), output_dir=output, batch_size=1)
    assert not output.exists()


def test_trainable_flow_matches_frozen_reload(tmp_path) -> None:
    config = _config(tmp_path, steps=2)
    result = train_plain_dense_iaf(
        adapter=GaussianAdapter(),
        config=config,
        freeze_transport_id="test-plain-dense-iaf",
    )
    payload = json.loads(result.frozen_payload_path.read_text(encoding="utf-8"))
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=TARGET_SIGNATURE
    )
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    flow = PlainDenseIAFTransport(config)
    for variable, value in zip(flow.trainable_variables, state["trainable_variables"]):
        variable.assign(tf.constant(value, tf.float64))
    z = tf.random.stateless_normal((5, 3), seed=(20260713, 18), dtype=tf.float64)

    expected, expected_logdet = flow.forward_and_logdet(z)
    np.testing.assert_allclose(
        loaded.transport.forward_batch(z).numpy(), expected.numpy(), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        loaded.transport.log_abs_det_jacobian_batch(z).numpy(),
        expected_logdet.numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_training_records_combined_target_status(tmp_path) -> None:
    result = train_plain_dense_iaf(
        adapter=StatusGaussianAdapter(),
        config=_config(tmp_path, output="status", steps=2),
    )

    assert all(record["target_status_available"] for record in result.records)
    assert all(record["target_status_all_valid"] for record in result.records)
    assert all(record["target_status_nonvalid_count"] == 0 for record in result.records)


def test_training_uses_one_graph_native_program_and_terminal_checkpoint(tmp_path) -> None:
    config = _config(tmp_path, output="graph-native", steps=6)
    result = train_plain_dense_iaf(adapter=GaussianAdapter(), config=config)

    assert result.runtime_metadata["compiled_training_program_invocations"] == 1
    assert result.runtime_metadata["compiled_training_control_flow"] == "tf_while_loop"
    assert result.runtime_metadata["checkpoint_policy"] == (
        "terminal_only_graph_native_v1"
    )
    assert any(
        "While" in item for item in result.runtime_metadata["graph_operation_types"]
    )
    assert tuple(path.name for path in config.output_dir.glob("checkpoint_step_*.json")) == (
        "checkpoint_step_000006.json",
    )


def test_training_rejects_invalid_combined_target_status(tmp_path) -> None:
    with pytest.raises(NeuTraTrainingError, match="invalid exact target status"):
        train_plain_dense_iaf(
            adapter=StatusGaussianAdapter(valid=False),
            config=_config(tmp_path, output="invalid-status", steps=1),
        )


def test_uninterrupted_and_resumed_training_are_bitwise_equal(tmp_path) -> None:
    full = train_plain_dense_iaf(adapter=GaussianAdapter(), config=_config(tmp_path, output="full"))
    partial_config = _config(tmp_path, output="resume")
    partial = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=partial_config, stop_after_steps=3
    )
    resumed = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=partial_config, resume_from=partial.state_path
    )
    full_state = json.loads(full.state_path.read_text(encoding="utf-8"))
    resumed_state = json.loads(resumed.state_path.read_text(encoding="utf-8"))

    assert full_state["trainable_variables"] == resumed_state["trainable_variables"]
    assert full_state["adam_first_moments"] == resumed_state["adam_first_moments"]
    assert full_state["adam_second_moments"] == resumed_state["adam_second_moments"]


def test_resume_rejects_config_mismatch(tmp_path) -> None:
    config = _config(tmp_path, output="resume-mismatch")
    partial = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=config, stop_after_steps=3
    )
    changed = replace(config, learning_rate=2.0e-3)
    with pytest.raises(NeuTraTrainingError, match="config mismatch"):
        train_plain_dense_iaf(
            adapter=GaussianAdapter(), config=changed, resume_from=partial.state_path
        )


def test_lower_rate_repair_resume_has_explicit_lineage(tmp_path) -> None:
    parent = _config(tmp_path, output="repair-parent")
    partial = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=parent, stop_after_steps=3
    )
    child = replace(
        parent,
        output_dir=tmp_path / "repair-child",
        learning_rate=5.0e-4,
    )
    repaired = train_plain_dense_iaf(
        adapter=GaussianAdapter(),
        config=child,
        resume_repair_from=partial.state_path,
    )
    state = json.loads(repaired.state_path.read_text(encoding="utf-8"))

    assert repaired.resumed
    assert state["repair_lineage"]["repair_type"] == (
        "single_lower_learning_rate_retry"
    )
    assert state["repair_lineage"]["parent_state_hash"] == partial.state_hash


def test_lower_rate_repair_rejects_other_config_changes(tmp_path) -> None:
    parent = _config(tmp_path, output="repair-reject-parent")
    partial = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=parent, stop_after_steps=3
    )
    child = replace(
        parent,
        output_dir=tmp_path / "repair-reject-child",
        learning_rate=5.0e-4,
        batch_size=16,
    )
    with pytest.raises(NeuTraTrainingError, match="may change only"):
        train_plain_dense_iaf(
            adapter=GaussianAdapter(),
            config=child,
            resume_repair_from=partial.state_path,
        )


def test_infrastructure_resume_uses_fresh_output_and_preserves_state(tmp_path) -> None:
    full = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=_config(tmp_path, output="infra-full")
    )
    parent_config = _config(tmp_path, output="infra-parent")
    partial = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=parent_config, stop_after_steps=3
    )
    child_config = replace(parent_config, output_dir=tmp_path / "infra-child")
    resumed = train_plain_dense_iaf(
        adapter=GaussianAdapter(),
        config=child_config,
        resume_infrastructure_from=partial.state_path,
    )
    full_state = json.loads(full.state_path.read_text(encoding="utf-8"))
    resumed_state = json.loads(resumed.state_path.read_text(encoding="utf-8"))

    assert resumed_state["trainable_variables"] == full_state["trainable_variables"]
    assert resumed_state["adam_first_moments"] == full_state["adam_first_moments"]
    assert resumed_state["adam_second_moments"] == full_state["adam_second_moments"]
    assert resumed_state["repair_lineage"]["repair_type"] == (
        "infrastructure_resume_same_config_fresh_output_v1"
    )
    assert resumed_state["repair_lineage"]["scientific_configuration_changed"] is False


def test_segmented_training_matches_uninterrupted_and_freezes_only_terminal(tmp_path) -> None:
    from bayesfilter.inference.neutra_training import (
        train_plain_dense_iaf_infrastructure_segments,
    )

    full_config = _config(tmp_path, output="segmented-full", steps=6)
    full = train_plain_dense_iaf(
        adapter=GaussianAdapter(),
        config=full_config,
        freeze_transport_id="segmented-full",
    )
    segmented_config = _config(tmp_path, output="segmented", steps=6)
    segmented = train_plain_dense_iaf_infrastructure_segments(
        adapter=GaussianAdapter(),
        config=segmented_config,
        segment_steps=2,
        freeze_transport_id="segmented",
    )
    full_state = json.loads(full.state_path.read_text(encoding="utf-8"))
    segmented_state = json.loads(
        segmented.final_result.state_path.read_text(encoding="utf-8")
    )

    assert segmented_state["trainable_variables"] == full_state["trainable_variables"]
    assert segmented_state["adam_first_moments"] == full_state["adam_first_moments"]
    assert segmented_state["adam_second_moments"] == full_state["adam_second_moments"]
    assert {row["config_hash"] for row in segmented.segment_rows} == {
        segmented_config.config_hash
    }
    assert [row["completed_steps"] for row in segmented.segment_rows] == [2, 4, 6]
    assert [row["terminal"] for row in segmented.segment_rows] == [False, False, True]
    assert [row["frozen_payload_path"] is not None for row in segmented.segment_rows] == [
        False,
        False,
        True,
    ]
    assert segmented.segment_rows[1]["parent_state_path"] == str(
        Path(segmented.segment_rows[0]["state_path"]).resolve()
    )


def test_infrastructure_resume_rejects_existing_output(tmp_path) -> None:
    parent_config = _config(tmp_path, output="infra-existing-parent")
    partial = train_plain_dense_iaf(
        adapter=GaussianAdapter(), config=parent_config, stop_after_steps=3
    )
    child_dir = tmp_path / "infra-existing-child"
    child_dir.mkdir()
    with pytest.raises(NeuTraTrainingError, match="fresh output"):
        train_plain_dense_iaf(
            adapter=GaussianAdapter(),
            config=replace(parent_config, output_dir=child_dir),
            resume_infrastructure_from=partial.state_path,
        )


def test_no_overwrite_freeze_and_checkpoint(tmp_path) -> None:
    config = _config(tmp_path, output="no-overwrite", steps=2)
    train_plain_dense_iaf(
        adapter=GaussianAdapter(),
        config=config,
        freeze_transport_id="test-no-overwrite",
    )
    blocked_adapter = GaussianAdapter()
    with pytest.raises(NeuTraTrainingError, match="training after freeze"):
        train_plain_dense_iaf(
            adapter=blocked_adapter,
            config=config,
            freeze_transport_id="test-no-overwrite",
        )
    assert blocked_adapter.call_count == 0


def test_reverse_kl_gradient_matches_finite_difference_at_one_weight(tmp_path) -> None:
    config = _config(tmp_path, output="gradient", steps=1)
    flow = PlainDenseIAFTransport(config)
    adapter = GaussianAdapter()
    z = tf.random.stateless_normal((16, 3), seed=(20260713, 20), dtype=tf.float64)

    def loss():
        theta, logdet = flow.forward_and_logdet(z)
        value, _ = adapter.log_prob_and_grad(theta)
        return -tf.reduce_mean(value + logdet)

    variable = flow.trainable_variables[0]
    index = (0, 0)
    with tf.GradientTape() as tape:
        observed_loss = loss()
    gradient = tape.gradient(observed_loss, variable)[index]
    original = float(variable[index].numpy())
    epsilon = 1.0e-5
    plus = variable.numpy()
    plus[index] = original + epsilon
    variable.assign(plus)
    loss_plus = float(loss().numpy())
    minus = variable.numpy()
    minus[index] = original - epsilon
    variable.assign(minus)
    loss_minus = float(loss().numpy())
    numeric = (loss_plus - loss_minus) / (2.0 * epsilon)

    assert float(gradient.numpy()) == pytest.approx(numeric, rel=2e-4, abs=2e-5)


def test_frozen_explicit_score_matches_autodiff_and_xla(tmp_path) -> None:
    config = _config(tmp_path, output="score", steps=2)
    result = train_plain_dense_iaf(
        adapter=GaussianAdapter(),
        config=config,
        freeze_transport_id="test-score-dense-iaf",
    )
    payload = json.loads(result.frozen_payload_path.read_text(encoding="utf-8"))
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=TARGET_SIGNATURE
    )
    transport = loaded.transport
    z = tf.random.stateless_normal((4, 3), seed=(20260713, 23), dtype=tf.float64)
    theta_score = tf.random.stateless_normal(
        (4, 3), seed=(20260713, 24), dtype=tf.float64
    )

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        theta = transport.forward_batch(z)
        logdet = transport.log_abs_det_jacobian_batch(z)
        pullback_objective = tf.reduce_sum(theta * theta_score)
        logdet_objective = tf.reduce_sum(logdet)
    expected_pullback = tape.gradient(pullback_objective, z)
    expected_logdet_score = tape.gradient(logdet_objective, z)

    actual_pullback = transport.pullback_score_batch(z, theta_score)
    actual_logdet_score = transport.log_abs_det_jacobian_score_batch(z)
    np.testing.assert_allclose(
        actual_pullback.numpy(), expected_pullback.numpy(), rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        actual_logdet_score.numpy(),
        expected_logdet_score.numpy(),
        rtol=1e-11,
        atol=1e-11,
    )

    @tf.function(jit_compile=True)
    def compiled(z_arg, score_arg):
        return (
            transport.pullback_score_batch(z_arg, score_arg),
            transport.log_abs_det_jacobian_score_batch(z_arg),
        )

    compiled_pullback, compiled_logdet_score = compiled(z, theta_score)
    np.testing.assert_allclose(
        compiled_pullback.numpy(), actual_pullback.numpy(), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        compiled_logdet_score.numpy(), actual_logdet_score.numpy(), rtol=1e-14, atol=1e-14
    )
