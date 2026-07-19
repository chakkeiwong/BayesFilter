from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import (
    InvalidNeuTraArtifact,
    finalize_dense_iaf_neutra_artifact_payload,
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (
    DSGE_PAPER_LR_BOUNDARIES,
    DSGE_PAPER_NEUTRA_FAMILY,
    SSL_LSTM_CAPACITY_NEUTRA_FAMILY,
    SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
    NeuTraReverseKLTrainer,
    NeuTraTrainerConfig,
    NeuTraTrainingError,
    dsge_paper_neutra_config,
    ssl_lstm_capacity_neutra_config,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.neutra_training import _stable_hash


DSGE_ROOT = Path("/home/ubuntu/python/dsge_hmc")
DSGE_COMMIT = "d94566c9f70b3143e599a56eba7cb461ff2bda88"
TARGET_SIGNATURE = "1" * 64
ADAPTER_SIGNATURE = "2" * 64
PARAMETER_NAMES = ("alpha", "beta", "gamma", "delta")
TRANSLATION = (0.35, -0.08, 0.65, 0.05)


class _ParityTarget:
    parameter_dim = 4
    parameter_names = PARAMETER_NAMES

    def __init__(self, *, names=PARAMETER_NAMES, center=TRANSLATION, chart="identity"):
        self.parameter_names = tuple(names)
        self.config = SimpleNamespace(
            prior_center=tf.constant(center, tf.float64),
            signature_payload=lambda: {
                "parameter_transform": {
                    "orientation": chart,
                    "inverse_orientation": chart,
                }
            },
        )
        self.precision = tf.constant(
            [
                [1.5, -0.2, 0.1, 0.0],
                [-0.2, 1.1, 0.0, 0.2],
                [0.1, 0.0, 0.9, -0.15],
                [0.0, 0.2, -0.15, 1.3],
            ],
            tf.float64,
        )

    def target_signature(self):
        return TARGET_SIGNATURE

    def adapter_signature(self):
        return ADAPTER_SIGNATURE

    def batch_value_and_score(self, theta):
        centered = theta - tf.constant(TRANSLATION, theta.dtype)
        score = -tf.matmul(centered, self.precision)
        value = 0.5 * tf.reduce_sum(centered * score, axis=-1)
        return value, score


def _config(**changes):
    values = {
        "dimension": 4,
        "fixed_translation": TRANSLATION,
        "target_parameter_names": PARAMETER_NAMES,
        "target_signature": TARGET_SIGNATURE,
        "target_adapter_signature": ADAPTER_SIGNATURE,
        "initialization_seed": (20260715, 4101),
        "jit_compile": False,
    }
    values.update(changes)
    return dsge_paper_neutra_config(**values)


def _capacity_config(**changes):
    values = {
        "dimension": 4,
        "fixed_translation": TRANSLATION,
        "target_parameter_names": PARAMETER_NAMES,
        "target_signature": TARGET_SIGNATURE,
        "target_adapter_signature": ADAPTER_SIGNATURE,
        "initialization_seed": (20260715, 4101),
        "jit_compile": False,
    }
    values.update(changes)
    return ssl_lstm_capacity_neutra_config(**values)


def _tuned_capacity_config(**changes):
    values = {
        "dimension": 4,
        "fixed_translation": TRANSLATION,
        "target_parameter_names": PARAMETER_NAMES,
        "target_signature": TARGET_SIGNATURE,
        "target_adapter_signature": ADAPTER_SIGNATURE,
        "learning_rate": 1.0e-3,
        "initialization_scale": 0.01,
        "gradient_clip_norm": 5.0,
        "initialization_seed": (20260715, 4101),
        "jit_compile": False,
    }
    values.update(changes)
    return ssl_lstm_tuned_capacity_neutra_config(**values)


def _explicit_variables(trainer):
    rows = []
    for index, variable in enumerate(trainer.variables):
        size = int(tf.size(variable).numpy())
        values = np.linspace(
            -0.035 + 0.001 * index,
            0.041 + 0.001 * index,
            num=size,
            dtype=np.float64,
        ).reshape(variable.shape)
        rows.append(values)
        variable.assign(values)
    return tuple(rows)


def _dsge_imports():
    actual = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=DSGE_ROOT, text=True
    ).strip()
    if actual != DSGE_COMMIT:
        pytest.fail(f"required dsge_hmc commit mismatch: {actual}")
    source = str(DSGE_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from dsge_hmc.estimation import (  # noqa: PLC0415
        AffineTransport,
        ComposedTransport,
        DenseAutoregressiveIAFTransport,
        MixingLinearTransport,
    )
    from dsge_hmc.estimation._flow_training import (  # noqa: PLC0415
        collect_trainable_variables,
        make_learning_rate_schedule,
    )

    return (
        AffineTransport,
        ComposedTransport,
        DenseAutoregressiveIAFTransport,
        MixingLinearTransport,
        collect_trainable_variables,
        make_learning_rate_schedule,
    )


def _dsge_flow_and_variables(rows, *, hidden_layers=(4, 4)):
    (
        AffineTransport,
        ComposedTransport,
        DenseAutoregressiveIAFTransport,
        MixingLinearTransport,
        collect_trainable_variables,
        _,
    ) = _dsge_imports()
    reverse = np.eye(4, dtype=np.float64)[::-1]
    components = []
    for stage in range(3):
        components.append(
            DenseAutoregressiveIAFTransport(
                4,
                hidden_layers=hidden_layers,
                activation="elu",
                s_max=1.0,
                trainable=True,
                seed=100 + stage,
                scale=0.02,
            )
        )
        if stage < 2:
            components.append(MixingLinearTransport(4, W=reverse))
    components.append(AffineTransport(TRANSLATION, scale=np.ones(4)))
    flow = ComposedTransport(components)
    variables = tuple(collect_trainable_variables(flow))
    assert len(variables) == len(rows)
    for variable, values in zip(variables, rows):
        variable.assign(values)
    return flow, variables


def test_capacity_preset_changes_width_only_and_preserves_source_preset() -> None:
    source = _config()
    capacity = _capacity_config()
    assert source.family == DSGE_PAPER_NEUTRA_FAMILY
    assert source.hidden_layers == (4, 4)
    assert capacity.family == SSL_LSTM_CAPACITY_NEUTRA_FAMILY
    assert capacity.hidden_layers == (32, 32)
    source_payload = dict(source.manifest_payload())
    capacity_payload = dict(capacity.manifest_payload())
    for key in ("family", "hidden_layers"):
        source_payload.pop(key)
        capacity_payload.pop(key)
    assert capacity_payload == source_payload

    source_trainer = NeuTraReverseKLTrainer(_ParityTarget(), source)
    capacity_trainer = NeuTraReverseKLTrainer(_ParityTarget(), capacity)
    assert sum(int(tf.size(value)) for value in source_trainer.variables) == 240
    assert sum(int(tf.size(value)) for value in capacity_trainer.variables) == 4440


def test_capacity_operator_matches_dsge_generic_32x32_math() -> None:
    target = _ParityTarget()
    trainer = NeuTraReverseKLTrainer(target, _capacity_config())
    rows = _explicit_variables(trainer)
    flow, dsge_variables = _dsge_flow_and_variables(rows, hidden_layers=(32, 32))
    z = _base_rows()

    actual_theta, actual_logdet = trainer.forward_and_logdet(z)
    expected_theta = flow.forward_batch(z)
    expected_logdet = flow.log_abs_det_jacobian_batch(z)
    np.testing.assert_allclose(actual_theta, expected_theta, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(actual_logdet, expected_logdet, rtol=0.0, atol=5e-12)

    actual_result, actual_gradients = trainer.loss_and_gradients(z)
    with tf.GradientTape() as tape:
        expected_theta = flow.forward_batch(z)
        expected_logdet = flow.log_abs_det_jacobian_batch(z)
        target_value, _ = target.batch_value_and_score(expected_theta)
        expected_loss = tf.reduce_mean(-target_value - expected_logdet)
    expected_gradients = tape.gradient(expected_loss, dsge_variables)
    np.testing.assert_allclose(actual_result.loss, expected_loss, rtol=0.0, atol=5e-12)
    for actual, expected in zip(actual_gradients, expected_gradients):
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-14)


def _base_rows():
    return tf.constant(
        [
            [0.2, -0.4, 0.6, -0.8],
            [-1.1, 0.7, 0.3, -0.2],
            [0.05, 0.15, -0.25, 0.35],
        ],
        tf.float64,
    )


def test_required_sibling_commit_and_exact_cross_repository_math_parity() -> None:
    target = _ParityTarget()
    trainer = NeuTraReverseKLTrainer(target, _config())
    rows = _explicit_variables(trainer)
    flow, dsge_variables = _dsge_flow_and_variables(rows)
    z = _base_rows()

    actual_theta, actual_logdet = trainer.forward_and_logdet(z)
    expected_theta = flow.forward_batch(z)
    expected_logdet = flow.log_abs_det_jacobian_batch(z)
    np.testing.assert_allclose(actual_theta, expected_theta, rtol=0.0, atol=2e-15)
    # dsge_hmc's fixed permutation logdet uses a 1e-12 Cholesky nudge.
    np.testing.assert_allclose(actual_logdet, expected_logdet, rtol=0.0, atol=5e-12)

    actual_result, actual_gradients = trainer.loss_and_gradients(z)
    with tf.GradientTape() as tape:
        expected_theta = flow.forward_batch(z)
        expected_logdet = flow.log_abs_det_jacobian_batch(z)
        target_value, _ = target.batch_value_and_score(expected_theta)
        expected_loss = tf.reduce_mean(-target_value - expected_logdet)
    expected_gradients = tape.gradient(expected_loss, dsge_variables)
    np.testing.assert_allclose(actual_result.loss, expected_loss, rtol=0.0, atol=5e-12)
    for actual, expected in zip(actual_gradients, expected_gradients):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)


def test_one_keras_adam_update_matches_dsge_runner_semantics() -> None:
    target = _ParityTarget()
    trainer = NeuTraReverseKLTrainer(target, _config())
    rows = _explicit_variables(trainer)
    flow, dsge_variables = _dsge_flow_and_variables(rows)
    z = _base_rows()
    *_, make_learning_rate_schedule = _dsge_imports()
    schedule = make_learning_rate_schedule(
        lr=0.01,
        n_steps=5000,
        schedule="paper_piecewise",
        final_frac=1.0,
        warmup_frac=0.0,
        decay_frac=1.0,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    optimizer.build(dsge_variables)
    with tf.GradientTape() as tape:
        theta = flow.forward_batch(z)
        logdet = flow.log_abs_det_jacobian_batch(z)
        target_value, _ = target.batch_value_and_score(theta)
        loss = tf.reduce_mean(-target_value - logdet)
    gradients = tape.gradient(loss, dsge_variables)
    clipped = [
        tf.clip_by_norm(
            tf.where(tf.math.is_finite(value), value, tf.zeros_like(value)), 10.0
        )
        for value in gradients
    ]
    optimizer.apply_gradients(zip(clipped, dsge_variables))

    result = trainer.train_step(z)
    assert int(result.step.numpy()) == 1
    for actual, expected in zip(trainer.variables, dsge_variables):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)


def test_exact_schedule_boundaries_and_per_variable_clipping() -> None:
    trainer = NeuTraReverseKLTrainer(_ParityTarget(), _config())
    assert DSGE_PAPER_LR_BOUNDARIES == (999, 3999)
    expected = {
        0: 0.01,
        998: 0.01,
        999: 0.01,
        1000: 0.001,
        3998: 0.001,
        3999: 0.001,
        4000: 0.0001,
    }
    for iteration, value in expected.items():
        np.testing.assert_allclose(trainer.learning_rate_at(iteration), value, rtol=1e-6)

    class LargeScoreTarget(_ParityTarget):
        def batch_value_and_score(self, theta):
            return tf.zeros(tf.shape(theta)[:-1], tf.float64), tf.ones_like(theta) * 1e6

    clipped = NeuTraReverseKLTrainer(LargeScoreTarget(), _config())
    _explicit_variables(clipped)
    result, gradients = clipped.loss_and_gradients(_base_rows())
    assert bool(result.clipping_applied.numpy()) is True
    assert sum(float(tf.linalg.norm(value).numpy()) > 10.0 for value in gradients) > 1
    assert float(result.clipped_gradient_norm.numpy()) > 10.0


def test_strict_preset_and_live_target_chart_fail_closed() -> None:
    base = _config()
    payload = dict(base.__dict__)
    for key, value in (
        ("activation", "tanh"),
        ("hidden_layers", (8, 8)),
        ("learning_rate_schedule", "constant"),
        ("gradient_clip_mode", "global"),
        ("epsilon", 1e-8),
        ("beta1", 0.8),
        ("beta2", 0.99),
        ("stages", 2),
        ("fixed_translation", ()),
    ):
        mutated = {**payload, key: value}
        with pytest.raises(ValueError, match="preset mismatch|fixed_translation"):
            NeuTraTrainerConfig(**mutated)

    with pytest.raises(NeuTraTrainingError, match="names/order"):
        NeuTraReverseKLTrainer(
            _ParityTarget(names=tuple(reversed(PARAMETER_NAMES))), base
        )
    with pytest.raises(NeuTraTrainingError, match="not identity-oriented"):
        NeuTraReverseKLTrainer(_ParityTarget(chart="log"), base)
    with pytest.raises(NeuTraTrainingError, match="fixed translation mismatch"):
        NeuTraReverseKLTrainer(_ParityTarget(center=(0.0, 0.0, 0.0, 0.0)), base)


def _mutated_payload(payload, mutator):
    changed = copy.deepcopy(payload)
    mutator(changed)
    for component in changed["components"]:
        component.pop("topology_hash", None)
        component.pop("tensor_hash", None)
        component.pop("component_hash", None)
    for key in ("topology_hash", "tensor_hash", "transport_hash"):
        changed.pop(key, None)
    return finalize_dense_iaf_neutra_artifact_payload(changed)


@pytest.mark.parametrize(
    "mutator,match",
    (
        (
            lambda row: (row["components"].pop(0), row["component_order"].pop(0)),
            "component order mismatch",
        ),
        (
            lambda row: row["components"][1].update(
                {"matrix": np.eye(4).tolist()}
            ),
            "reverse mixing mismatch",
        ),
        (
            lambda row: row["components"][0].update({"activation": "tanh"}),
            "activation mismatch",
        ),
        (
            lambda row: row["components"][-1].update(
                {"offset": [0.0, 0.0, 0.0, 0.0]}
            ),
            "translation mismatch",
        ),
        (
            lambda row: row["components"][-1].update(
                {"scale": [1.0, 1.0, 1.0, 1.01]}
            ),
            "output scale",
        ),
        (
            lambda row: row["component_order"].__setitem__(
                slice(0, 2), list(reversed(row["component_order"][:2]))
            ),
            "component order mismatch",
        ),
    ),
)
def test_serialized_procedure_mutations_are_rejected(mutator, match) -> None:
    trainer = NeuTraReverseKLTrainer(_ParityTarget(), _config())
    _explicit_variables(trainer)
    payload = trainer.frozen_transport_payload(
        transport_id="dsge-procedure-parity-fixture",
        target_signature=TARGET_SIGNATURE,
    )
    mutated = _mutated_payload(payload, mutator)
    with pytest.raises(InvalidNeuTraArtifact, match=match):
        load_frozen_neutra_artifact(
            mutated,
            expected_target_signature=TARGET_SIGNATURE,
        )


def test_six_component_serialization_inverse_scores_and_exact_resume() -> None:
    target = _ParityTarget()
    trainer = NeuTraReverseKLTrainer(target, _config())
    _explicit_variables(trainer)
    z = _base_rows()
    trainer.train_step(z)
    state = trainer.state_payload()
    payload = trainer.frozen_transport_payload(
        transport_id="dsge-procedure-parity-fixture",
        target_signature=TARGET_SIGNATURE,
    )
    assert [row["kind"] for row in payload["components"]] == [
        "dense_autoregressive_iaf",
        "mixing_linear",
        "dense_autoregressive_iaf",
        "mixing_linear",
        "dense_autoregressive_iaf",
        "affine",
    ]
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=TARGET_SIGNATURE
    )
    theta, logdet = trainer.forward_and_logdet(z)
    np.testing.assert_allclose(loaded.transport.forward_batch(z), theta, atol=0.0)
    np.testing.assert_allclose(
        loaded.transport.log_abs_det_jacobian_batch(z), logdet, atol=0.0
    )
    np.testing.assert_allclose(
        loaded.transport.inverse_theta_to_z_batch(theta), z, atol=2e-14
    )

    score = tf.constant(
        [[0.2, -0.3, 0.4, -0.5], [0.1, 0.2, -0.1, -0.2], [1.0, 0.0, 0.5, -0.5]],
        tf.float64,
    )
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        mapped = loaded.transport.forward_batch(z)
        logdet = loaded.transport.log_abs_det_jacobian_batch(z)
        contraction = tf.reduce_sum(mapped * score)
        total_logdet = tf.reduce_sum(logdet)
    np.testing.assert_allclose(
        loaded.transport.pullback_score_batch(z, score),
        tape.gradient(contraction, z),
        rtol=2e-13,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        loaded.transport.log_abs_det_jacobian_score_batch(z),
        tape.gradient(total_logdet, z),
        rtol=2e-13,
        atol=2e-14,
    )

    expected = trainer.train_step(z)
    expected_variables = tuple(value.numpy().copy() for value in trainer.variables)
    resumed = NeuTraReverseKLTrainer(target, _config())
    resumed.restore_state(state)
    actual = resumed.train_step(z)
    np.testing.assert_array_equal(actual.loss, expected.loss)
    for left, right in zip(resumed.variables, expected_variables):
        np.testing.assert_array_equal(left.numpy(), right)


def test_capacity_serialization_resume_and_procedure_label_are_exact() -> None:
    target = _ParityTarget()
    trainer = NeuTraReverseKLTrainer(target, _capacity_config())
    z = _base_rows()
    trainer.train_step(z)
    state = trainer.state_payload()
    payload = trainer.frozen_transport_payload(
        transport_id="ssl-lstm-capacity-32x32-fixture",
        target_signature=TARGET_SIGNATURE,
    )
    assert payload["procedure"] == "bayesfilter_ssl_lstm_capacity_32x32_neutra_v1"
    assert [
        row.get("hidden_layers")
        for row in payload["components"]
        if row["kind"] == "dense_autoregressive_iaf"
    ] == [[32, 32]] * 3
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=TARGET_SIGNATURE
    )
    theta, logdet = trainer.forward_and_logdet(z)
    np.testing.assert_array_equal(loaded.transport.forward_batch(z), theta)
    np.testing.assert_array_equal(
        loaded.transport.log_abs_det_jacobian_batch(z), logdet
    )
    np.testing.assert_allclose(
        loaded.transport.inverse_theta_to_z_batch(theta), z, rtol=0.0, atol=3e-14
    )

    resumed = NeuTraReverseKLTrainer(target, _capacity_config())
    resumed.restore_state(state)
    expected = trainer.train_step(z)
    actual = resumed.train_step(z)
    np.testing.assert_array_equal(actual.loss, expected.loss)
    for left, right in zip(resumed.variables, trainer.variables):
        np.testing.assert_array_equal(left, right)

    mislabeled = copy.deepcopy(payload)
    mislabeled["procedure"] = "dsge_hmc_rotemberg_sgu_plain_neutra_v1"
    for key in ("topology_hash", "tensor_hash", "transport_hash"):
        mislabeled.pop(key)
    mislabeled = finalize_dense_iaf_neutra_artifact_payload(mislabeled)
    with pytest.raises(InvalidNeuTraArtifact, match="hidden layers mismatch"):
        load_frozen_neutra_artifact(
            mislabeled, expected_target_signature=TARGET_SIGNATURE
        )

    wrong_width = copy.deepcopy(payload)
    wrong_width["components"][0]["hidden_layers"] = [4, 4]
    for key in ("topology_hash", "tensor_hash", "transport_hash"):
        wrong_width.pop(key)
    wrong_width = finalize_dense_iaf_neutra_artifact_payload(wrong_width)
    with pytest.raises(InvalidNeuTraArtifact, match="hidden layers mismatch"):
        load_frozen_neutra_artifact(
            wrong_width, expected_target_signature=TARGET_SIGNATURE
        )


def test_tuned_capacity_mutable_learning_rate_resume_and_label_are_exact() -> None:
    target = _ParityTarget()
    config = _tuned_capacity_config()
    trainer = NeuTraReverseKLTrainer(target, config)
    assert config.family == SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY
    np.testing.assert_allclose(trainer.learning_rate_at(0), 1.0e-3)
    trainer.train_step(_base_rows())
    optimizer_before = tuple(value.numpy().copy() for value in trainer.optimizer.variables)
    iteration_before = int(trainer.optimizer.iterations.numpy())
    trainer.set_learning_rate(5.0e-4)
    assert int(trainer.optimizer.iterations.numpy()) == iteration_before
    np.testing.assert_allclose(trainer.learning_rate_at(100), 5.0e-4)
    for before, after in zip(optimizer_before[2:], trainer.optimizer.variables[2:]):
        np.testing.assert_array_equal(before, after.numpy())

    state = trainer.state_payload()
    resumed = NeuTraReverseKLTrainer(target, config)
    resumed.restore_state(state)
    np.testing.assert_allclose(resumed.learning_rate_at(0), 5.0e-4)
    expected = trainer.train_step(_base_rows())
    actual = resumed.train_step(_base_rows())
    np.testing.assert_array_equal(actual.loss, expected.loss)
    for left, right in zip(resumed.variables, trainer.variables):
        np.testing.assert_array_equal(left, right)

    payload = trainer.frozen_transport_payload(
        transport_id="ssl-lstm-tuned-capacity-32x32-fixture",
        target_signature=TARGET_SIGNATURE,
    )
    assert payload["procedure"] == (
        "bayesfilter_ssl_lstm_tuned_capacity_32x32_neutra_v1"
    )
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=TARGET_SIGNATURE
    )
    theta, logdet = trainer.forward_and_logdet(_base_rows())
    np.testing.assert_array_equal(loaded.transport.forward_batch(_base_rows()), theta)
    np.testing.assert_array_equal(
        loaded.transport.log_abs_det_jacobian_batch(_base_rows()), logdet
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("learning_rate", 3.0e-3),
        ("initialization_scale", 0.03),
        ("gradient_clip_norm", 7.0),
    ),
)
def test_tuned_capacity_search_contract_fails_closed(field, value) -> None:
    with pytest.raises(ValueError, match="outside search contract"):
        _tuned_capacity_config(**{field: value})


def test_mutable_learning_rate_is_restricted_to_tuned_family() -> None:
    trainer = NeuTraReverseKLTrainer(_ParityTarget(), _capacity_config())
    with pytest.raises(NeuTraTrainingError, match="restricted"):
        trainer.set_learning_rate(1.0e-3)


def test_strict_serialization_rejects_target_signature_drift() -> None:
    trainer = NeuTraReverseKLTrainer(_ParityTarget(), _config())
    with pytest.raises(NeuTraTrainingError, match="target_signature"):
        trainer.frozen_transport_payload(
            transport_id="signature-drift",
            target_signature="3" * 64,
        )


def test_strict_path_rejects_nonfinite_gradient_instead_of_zeroing_it() -> None:
    class ExplodingScoreTarget(_ParityTarget):
        def batch_value_and_score(self, theta):
            return tf.zeros(tf.shape(theta)[:-1], tf.float64), tf.ones_like(theta) * 1e308

    trainer = NeuTraReverseKLTrainer(ExplodingScoreTarget(), _config())
    _explicit_variables(trainer)
    with pytest.raises(tf.errors.InvalidArgumentError, match="reverse-KL surrogate"):
        trainer.loss_and_gradients(_base_rows())


def test_malformed_valid_checkpoint_does_not_partially_mutate_trainer() -> None:
    trainer = NeuTraReverseKLTrainer(_ParityTarget(), _config())
    _explicit_variables(trainer)
    before = tuple(variable.numpy().copy() for variable in trainer.variables)
    state = copy.deepcopy(trainer.state_payload())
    state["variables"][0] = [0.0]
    state.pop("state_hash")
    state["state_hash"] = _stable_hash(state)
    with pytest.raises(NeuTraTrainingError, match=r"variables\[0\] shape mismatch"):
        trainer.restore_state(state)
    for variable, expected in zip(trainer.variables, before):
        np.testing.assert_array_equal(variable.numpy(), expected)
