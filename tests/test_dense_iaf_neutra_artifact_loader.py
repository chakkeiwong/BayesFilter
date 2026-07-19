from __future__ import annotations

import copy
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    FixedTransportValueScoreAdapter,
    FrozenDenseIAFTransport,
    InvalidNeuTraArtifact,
    finalize_dense_iaf_neutra_artifact_payload,
    load_frozen_neutra_artifact,
    stable_frozen_neutra_artifact_signature,
)
from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    stable_ssm_target_signature,
)


SCHEMA = "bayesfilter.neutra.dense_iaf_frozen_transport.v1"


def _target_signature() -> str:
    problem = BayesianSSMProblem(
        problem_id="dense-iaf-toy-ssm",
        static_shape=SSMStaticShape(
            horizon=3,
            state_dim=1,
            observation_dim=1,
            innovation_dim=1,
            parameter_dim=2,
        ),
        data_signature=SSMDataSignature(
            dataset_id="dense-iaf-toy-data",
            observation_shape=(3, 1),
            data_hash="sha256:dense-iaf-data",
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            "model_id": "dense-iaf-toy-model",
            "model_hash": "sha256:dense-iaf-model",
            "capabilities": ("transition_mean", "observation_mean"),
        },
    )
    chart = ParameterChart(
        parameter_names=("alpha", "beta"),
        unconstrained_dim=2,
        constrained_shape=(2,),
        transform_manifest={
            "transform_id": "identity-chart",
            "transform_hash": "sha256:dense-iaf-chart",
        },
        log_jacobian_convention="not_included",
    )
    prior = ParameterPrior(
        prior_manifest={
            "prior_id": "toy-gaussian-prior",
            "prior_hash": "sha256:dense-iaf-prior",
        },
        support_policy="unbounded",
        log_density_authority="graph_native",
    )
    filter_program = FilterProgram(
        filter_id="toy-deterministic-filter",
        required_model_capabilities=("transition_mean", "observation_mean"),
        deterministic_target_policy="deterministic",
        approximation_semantics="deterministic_approximation",
        filter_manifest={
            "filter_id": "toy-deterministic-filter",
            "filter_hash": "sha256:dense-iaf-filter",
        },
    )
    return stable_ssm_target_signature(
        SSMTargetContract(
            problem=problem,
            chart=chart,
            prior=prior,
            filter_program=filter_program,
        )
    )


def _raw_payload(**overrides):
    values = {
        "schema": SCHEMA,
        "transport_id": "dense-iaf-synthetic-transport",
        "dimension": 2,
        "target_signature": _target_signature(),
        "log_jacobian_available": True,
        "component_order": ("dense",),
        "components": (
            {
                "component_id": "dense",
                "kind": "dense_autoregressive_iaf",
                "dim": 2,
                "hidden_layers": (2,),
                "activation": "tanh",
                "s_max": 1.0,
                "masks_policy": "legacy_degree_masks_v1",
                "dtype": "float64",
                "weights": (
                    ((0.5, -0.25), (0.75, 0.1)),
                    ((0.2, -0.4, 0.3, -0.2), (0.1, 0.6, -0.5, 0.7)),
                ),
                "biases": (
                    (0.05, -0.1),
                    (0.02, -0.03, 0.04, -0.05),
                ),
            },
        ),
        "training_state_hash": "sha256:dense-iaf-synthetic-training",
        "nonclaims": (
            "frozen dense-IAF transport artifact loader only",
            "no NeuTra training claim",
            "no HMC tuning or sampling claim",
            "no posterior convergence claim",
            "no scientific validity claim",
            "no default policy change",
        ),
    }
    values.update(overrides)
    return values


def _payload(**overrides):
    return finalize_dense_iaf_neutra_artifact_payload(_raw_payload(**overrides))


def _load(payload=None):
    return load_frozen_neutra_artifact(
        _payload() if payload is None else payload,
        expected_target_signature=_target_signature(),
    )


def _load_components(components, component_order):
    return _load(
        finalize_dense_iaf_neutra_artifact_payload(
            _raw_payload(
                components=components,
                component_order=component_order,
            )
        )
    )


def _expected_forward_and_logdet(z):
    z = np.asarray(z, dtype=np.float64)
    w0 = np.array([[0.5, -0.25], [0.75, 0.1]], dtype=np.float64)
    b0 = np.array([0.05, -0.1], dtype=np.float64)
    w1 = np.array(
        [[0.2, -0.4, 0.3, -0.2], [0.1, 0.6, -0.5, 0.7]],
        dtype=np.float64,
    )
    b1 = np.array([0.02, -0.03, 0.04, -0.05], dtype=np.float64)
    mask0 = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    mask1 = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    h = np.tanh(z @ (w0 * mask0) + b0)
    raw = h @ (w1 * mask1) + b1
    scale_log = np.tanh(raw[:, :2])
    shift = raw[:, 2:]
    return z * np.exp(scale_log) + shift, np.sum(scale_log, axis=-1)


def test_dense_iaf_loader_accepts_synthetic_payload_and_manifest() -> None:
    artifact = _load()

    assert isinstance(artifact.transport, FrozenDenseIAFTransport)
    assert artifact.manifest.schema == SCHEMA
    assert artifact.manifest.dimension == 2
    assert artifact.binding.target_signature == _target_signature()
    assert artifact.binding.transport_manifest["topology_hash"] == artifact.manifest.topology_hash
    assert artifact.binding.transport_manifest["tensor_hash"] == artifact.manifest.tensor_hash
    assert stable_frozen_neutra_artifact_signature(artifact) == artifact.artifact_signature
    assert "no HMC tuning or sampling claim" in artifact.manifest.nonclaims


def test_dense_iaf_loader_forward_and_logdet_match_fixture() -> None:
    artifact = _load()
    z = tf.constant([[0.2, -0.4], [0.1, 0.3]], dtype=tf.float64)

    actual = artifact.transport.forward_batch(z)
    actual_logdet = artifact.transport.log_abs_det_jacobian_batch(z)
    expected, expected_logdet = _expected_forward_and_logdet(z.numpy())

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual_logdet.numpy(),
        expected_logdet,
        rtol=1e-12,
        atol=1e-12,
    )


def test_dense_iaf_inverse_and_manual_scores_match_debug_autodiff() -> None:
    transport = _load().transport
    z = tf.constant(
        [[0.2, -0.4], [0.1, 0.3], [2.5, -3.0]],
        dtype=tf.float64,
    )
    theta_score = tf.constant(
        [[0.7, -1.1], [-0.2, 0.5], [1.3, 0.4]],
        dtype=tf.float64,
    )

    theta = transport.forward_z_to_theta_batch(z)
    roundtrip = transport.inverse_theta_to_z_batch(theta)
    np.testing.assert_allclose(roundtrip.numpy(), z.numpy(), rtol=1e-12, atol=1e-12)

    with tf.GradientTape() as tape:
        tape.watch(z)
        debug_objective = tf.reduce_sum(transport.forward_batch(z) * theta_score)
    debug_pullback = tape.gradient(debug_objective, z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        debug_logdet = tf.reduce_sum(transport.log_abs_det_jacobian_batch(z))
    debug_logdet_score = tape.gradient(debug_logdet, z)

    np.testing.assert_allclose(
        transport.pullback_score_batch(z, theta_score).numpy(),
        debug_pullback.numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        transport.log_abs_det_jacobian_score_batch(z).numpy(),
        debug_logdet_score.numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        transport.forward_z_to_theta(z[0]).numpy(),
        theta[0].numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        transport.inverse_theta_to_z(theta[0]).numpy(),
        z[0].numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transport.pullback_score(z[0], theta_score[0]).numpy(),
        debug_pullback[0].numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        transport.log_abs_det_jacobian_score(z[0]).numpy(),
        debug_logdet_score[0].numpy(),
        rtol=2e-12,
        atol=2e-12,
    )


def test_nonsymmetric_linear_and_affine_transpose_conventions() -> None:
    mixing_matrix = np.array([[2.0, 1.0], [-0.5, 1.5]], dtype=np.float64)
    affine_matrix = np.array([[1.0, -0.75], [0.25, 2.0]], dtype=np.float64)
    offset = np.array([0.3, -0.2], dtype=np.float64)
    components = (
        {
            "component_id": "mix",
            "kind": "mixing_linear",
            "dim": 2,
            "dtype": "float64",
            "matrix": mixing_matrix.tolist(),
        },
        {
            "component_id": "affine",
            "kind": "affine_dense",
            "dim": 2,
            "dtype": "float64",
            "offset": offset.tolist(),
            "matrix": affine_matrix.tolist(),
        },
    )
    transport = _load_components(components, ("mix", "affine")).transport
    z = tf.constant([[0.4, -0.6], [-0.2, 0.8]], dtype=tf.float64)
    theta_score = tf.constant([[0.7, -1.2], [1.1, 0.3]], dtype=tf.float64)

    expected_theta = (z.numpy() @ mixing_matrix) @ affine_matrix.T + offset
    expected_pullback = (theta_score.numpy() @ affine_matrix) @ mixing_matrix.T
    expected_logdet = np.linalg.slogdet(mixing_matrix)[1] + np.linalg.slogdet(
        affine_matrix
    )[1]

    np.testing.assert_allclose(
        transport.forward_batch(z).numpy(), expected_theta, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        transport.pullback_score_batch(z, theta_score).numpy(),
        expected_pullback,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transport.log_abs_det_jacobian_batch(z).numpy(),
        np.full(2, expected_logdet),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        transport.log_abs_det_jacobian_score_batch(z).numpy(),
        np.zeros((2, 2)),
    )
    np.testing.assert_allclose(
        transport.inverse_theta_to_z_batch(expected_theta).numpy(),
        z.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_nested_composition_order_and_batch_permutation() -> None:
    children = (
        {
            "component_id": "child_scale",
            "kind": "affine",
            "dim": 2,
            "dtype": "float64",
            "offset": [0.1, -0.2],
            "scale": [1.5, -0.75],
        },
        copy.deepcopy(_raw_payload()["components"][0]),
    )
    components = (
        {
            "component_id": "nested",
            "kind": "composed",
            "dim": 2,
            "dtype": "float64",
            "children": children,
        },
    )
    transport = _load_components(components, ("nested",)).transport
    z = tf.constant([[0.2, -0.4], [0.1, 0.3], [-1.2, 0.8]], dtype=tf.float64)
    theta_score = tf.constant([[0.7, -1.1], [-0.2, 0.5], [0.4, 0.9]], dtype=tf.float64)
    permutation = tf.constant([2, 0, 1], dtype=tf.int32)

    theta = transport.forward_batch(z)
    score = transport.pullback_score_batch(z, theta_score)
    logdet_score = transport.log_abs_det_jacobian_score_batch(z)
    np.testing.assert_allclose(
        transport.inverse_theta_to_z_batch(theta).numpy(),
        z.numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        transport.forward_batch(tf.gather(z, permutation)).numpy(),
        tf.gather(theta, permutation).numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        transport.pullback_score_batch(
            tf.gather(z, permutation), tf.gather(theta_score, permutation)
        ).numpy(),
        tf.gather(score, permutation).numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        transport.log_abs_det_jacobian_score_batch(
            tf.gather(z, permutation)
        ).numpy(),
        tf.gather(logdet_score, permutation).numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_dense_iaf_manual_scores_match_directional_finite_differences() -> None:
    transport = _load().transport
    z = tf.constant([[0.25, -0.35]], dtype=tf.float64)
    direction = tf.constant([[0.6, -0.8]], dtype=tf.float64)
    theta_score = tf.constant([[0.7, -1.1]], dtype=tf.float64)
    step = tf.constant(1.0e-6, dtype=tf.float64)

    plus = z + step * direction
    minus = z - step * direction
    objective_plus = tf.reduce_sum(transport.forward_batch(plus) * theta_score)
    objective_minus = tf.reduce_sum(transport.forward_batch(minus) * theta_score)
    finite_difference = (objective_plus - objective_minus) / (2.0 * step)
    manual_directional = tf.reduce_sum(
        transport.pullback_score_batch(z, theta_score) * direction
    )
    logdet_finite_difference = (
        tf.reduce_sum(transport.log_abs_det_jacobian_batch(plus))
        - tf.reduce_sum(transport.log_abs_det_jacobian_batch(minus))
    ) / (2.0 * step)
    logdet_manual_directional = tf.reduce_sum(
        transport.log_abs_det_jacobian_score_batch(z) * direction
    )

    np.testing.assert_allclose(
        manual_directional.numpy(), finite_difference.numpy(), rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(
        logdet_manual_directional.numpy(),
        logdet_finite_difference.numpy(),
        rtol=2e-8,
        atol=2e-9,
    )


def test_dense_iaf_pullback_rejects_shape_mismatch() -> None:
    transport = _load().transport
    z = tf.zeros((2, 2), dtype=tf.float64)
    with pytest.raises(ValueError, match="shape must match"):
        transport.pullback_score_batch(
            z,
            tf.zeros((1, 2), dtype=tf.float64),
        )


def test_identity_and_diagonal_affine_closure() -> None:
    identity_component = {
        "component_id": "identity",
        "kind": "affine",
        "dim": 2,
        "dtype": "float64",
        "offset": [0.0, 0.0],
        "scale": [1.0, 1.0],
    }
    diagonal_component = {
        "component_id": "diagonal",
        "kind": "affine",
        "dim": 2,
        "dtype": "float64",
        "offset": [0.3, -0.2],
        "scale": [1.5, -0.75],
    }
    z = tf.constant([[0.2, -0.4], [0.1, 0.3]], dtype=tf.float64)
    score = tf.constant([[0.7, -1.1], [-0.2, 0.5]], dtype=tf.float64)

    identity = _load_components((identity_component,), ("identity",)).transport
    np.testing.assert_array_equal(identity.forward_batch(z).numpy(), z.numpy())
    np.testing.assert_array_equal(
        identity.inverse_theta_to_z_batch(z).numpy(), z.numpy()
    )
    np.testing.assert_array_equal(
        identity.pullback_score_batch(z, score).numpy(), score.numpy()
    )
    np.testing.assert_array_equal(
        identity.log_abs_det_jacobian_score_batch(z).numpy(),
        np.zeros((2, 2)),
    )

    diagonal = _load_components((diagonal_component,), ("diagonal",)).transport
    expected = np.array([0.3, -0.2]) + z.numpy() * np.array([1.5, -0.75])
    np.testing.assert_allclose(
        diagonal.forward_batch(z).numpy(), expected, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        diagonal.inverse_theta_to_z_batch(expected).numpy(),
        z.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagonal.pullback_score_batch(z, score).numpy(),
        score.numpy() * np.array([1.5, -0.75]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("activation", ("elu", "relu"))
def test_supported_activation_manual_scores_match_debug_autodiff(activation) -> None:
    payload = copy.deepcopy(_raw_payload())
    payload["components"][0]["activation"] = activation
    transport = _load(finalize_dense_iaf_neutra_artifact_payload(payload)).transport
    z = tf.constant([[0.2, -0.4], [-0.3, 0.7]], dtype=tf.float64)
    theta_score = tf.constant([[0.7, -1.1], [-0.2, 0.5]], dtype=tf.float64)

    with tf.GradientTape() as tape:
        tape.watch(z)
        objective = tf.reduce_sum(transport.forward_batch(z) * theta_score)
    debug_pullback = tape.gradient(objective, z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        logdet = tf.reduce_sum(transport.log_abs_det_jacobian_batch(z))
    debug_logdet_score = tape.gradient(logdet, z)

    np.testing.assert_allclose(
        transport.pullback_score_batch(z, theta_score).numpy(),
        debug_pullback.numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        transport.log_abs_det_jacobian_score_batch(z).numpy(),
        debug_logdet_score.numpy(),
        rtol=2e-12,
        atol=2e-12,
    )


def test_dense_iaf_reload_replays_all_closure_operations() -> None:
    payload = _payload()
    first = _load(payload).transport
    second = _load(copy.deepcopy(payload)).transport
    z = tf.constant([[0.2, -0.4], [2.5, -3.0]], dtype=tf.float64)
    theta_score = tf.constant([[0.7, -1.1], [1.3, 0.4]], dtype=tf.float64)
    first_rows = (
        first.forward_batch(z),
        first.inverse_theta_to_z_batch(first.forward_batch(z)),
        first.log_abs_det_jacobian_batch(z),
        first.pullback_score_batch(z, theta_score),
        first.log_abs_det_jacobian_score_batch(z),
    )
    second_rows = (
        second.forward_batch(z),
        second.inverse_theta_to_z_batch(second.forward_batch(z)),
        second.log_abs_det_jacobian_batch(z),
        second.pullback_score_batch(z, theta_score),
        second.log_abs_det_jacobian_score_batch(z),
    )
    for first_row, second_row in zip(first_rows, second_rows):
        np.testing.assert_array_equal(first_row.numpy(), second_row.numpy())


def test_dense_iaf_closure_integrates_with_fixed_transport_target() -> None:
    class GaussianBaseAdapter:
        def log_prob_and_grad(self, theta):
            values = tf.convert_to_tensor(theta, dtype=tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

    transport = _load().transport
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=GaussianBaseAdapter(),
        transport=transport,
        target_scope="dense-iaf-closure-fixture",
        require_batch_native=True,
    )
    z = tf.constant([[0.2, -0.4], [0.1, 0.3]], dtype=tf.float64)
    actual_value, actual_score = adapter.log_prob_and_grad_batch(z)

    with tf.GradientTape() as tape:
        tape.watch(z)
        theta = transport.forward_batch(z)
        expected_value = -0.5 * tf.reduce_sum(tf.square(theta), axis=-1)
        expected_value += transport.log_abs_det_jacobian_batch(z)
        expected_total = tf.reduce_sum(expected_value)
    expected_score = tape.gradient(expected_total, z)

    np.testing.assert_allclose(
        actual_value.numpy(), expected_value.numpy(), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        actual_score.numpy(), expected_score.numpy(), rtol=2e-12, atol=2e-12
    )


def test_dense_iaf_loader_rejects_target_signature_mismatch_and_legacy_identity() -> None:
    with pytest.raises(InvalidNeuTraArtifact, match="target_signature mismatch"):
        load_frozen_neutra_artifact(
            _payload(),
            expected_target_signature="0" * 64,
        )

    historical_style = _payload(target_signature="legacy-rotemberg-target-name")
    with pytest.raises(InvalidNeuTraArtifact, match="sha256"):
        load_frozen_neutra_artifact(
            historical_style,
            expected_target_signature=_target_signature(),
        )


def test_dense_iaf_loader_rejects_individual_hash_tampering() -> None:
    for field in ("topology_hash", "tensor_hash", "transport_hash"):
        tampered = dict(_payload())
        tampered[field] = "0" * 64
        with pytest.raises(InvalidNeuTraArtifact, match=f"{field} mismatch"):
            load_frozen_neutra_artifact(
                tampered,
                expected_target_signature=_target_signature(),
            )


def test_dense_iaf_loader_rejects_nonfinite_and_shape_mismatch() -> None:
    nonfinite = _raw_payload()
    component = dict(nonfinite["components"][0])
    weights = [list(row_group) for row_group in component["weights"]]
    weights[0] = [list(row) for row in weights[0]]
    weights[0][0] = (float("nan"), -0.25)
    component["weights"] = tuple(tuple(tuple(row) for row in group) for group in weights)
    nonfinite["components"] = (component,)
    with pytest.raises(InvalidNeuTraArtifact, match="finite"):
        finalize_dense_iaf_neutra_artifact_payload(nonfinite)

    bad_shape = copy.deepcopy(_payload())
    bad_shape["components"][0]["weights"][0] = ((0.5,), (0.75,))
    bad_shape = finalize_dense_iaf_neutra_artifact_payload(bad_shape)
    with pytest.raises(InvalidNeuTraArtifact, match="shape mismatch"):
        load_frozen_neutra_artifact(
            bad_shape,
            expected_target_signature=_target_signature(),
        )


def test_dense_iaf_loader_rejects_process_local_identity_and_component_semantics() -> None:
    with pytest.raises(InvalidNeuTraArtifact, match="process-local"):
        finalize_dense_iaf_neutra_artifact_payload(
            _raw_payload(transport_id=f"object at 0x{id(object()):x}")
        )

    unsupported = copy.deepcopy(_payload())
    unsupported["components"][0]["kind"] = "real_nvp"
    unsupported = finalize_dense_iaf_neutra_artifact_payload(unsupported)
    with pytest.raises(InvalidNeuTraArtifact, match="unsupported component kind"):
        load_frozen_neutra_artifact(
            unsupported,
            expected_target_signature=_target_signature(),
        )

    bad_policy = copy.deepcopy(_payload())
    bad_policy["components"][0]["masks_policy"] = "legacy_runtime_mask_object"
    bad_policy = finalize_dense_iaf_neutra_artifact_payload(bad_policy)
    with pytest.raises(InvalidNeuTraArtifact, match="masks_policy"):
        load_frozen_neutra_artifact(
            bad_policy,
            expected_target_signature=_target_signature(),
        )

    for kind, field in (("mixing_linear", "matrix"), ("affine_dense", "matrix")):
        singular_component = {
            "component_id": "singular",
            "kind": kind,
            "dim": 2,
            "dtype": "float64",
            field: [[1.0, 2.0], [2.0, 4.0]],
        }
        if kind == "affine_dense":
            singular_component["offset"] = [0.0, 0.0]
        singular = finalize_dense_iaf_neutra_artifact_payload(
            _raw_payload(
                components=(singular_component,),
                component_order=("singular",),
            )
        )
        with pytest.raises(InvalidNeuTraArtifact, match="nonsingular"):
            _load(singular)


def test_dense_iaf_loader_rejects_summary_only_historical_artifacts() -> None:
    with pytest.raises(InvalidNeuTraArtifact, match="schema"):
        load_frozen_neutra_artifact(
            {
                "schema_version": 1,
                "kind": "neutra_paper_style_at_baseline",
                "replay_state_path": "docs/plans/artifacts/legacy/paper_dense_iaf_replay.json",
                "row": {"candidate_arm": "paper_dense_iaf"},
            },
            expected_target_signature=_target_signature(),
        )
