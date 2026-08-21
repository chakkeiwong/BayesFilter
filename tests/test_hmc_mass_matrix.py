from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference.mass_matrix import (
    covariance_from_negative_hessian,
    covariance_from_precision,
    regularize_precision,
)


def test_hessian_mass_builder_preserves_positive_definite_precision():
    precision = np.array([[4.0, 1.0], [1.0, 3.0]])

    result = covariance_from_precision(
        precision,
        source="analytic_map_hessian",
        jitter=0.0,
    )

    np.testing.assert_allclose(result.covariance, np.linalg.inv(precision))
    np.testing.assert_allclose(result.regularized_precision, precision)
    assert result.matrix_kind == "dense"
    assert result.precision_eigen_summary["positive"] is True
    assert result.covariance_eigen_summary["positive"] is True
    assert result.regularization_report["clipped_eigenvalue_count"] == 0
    assert result.regularization_report["silent_eigenvalue_reflection"] is False


def test_hessian_mass_builder_regularizes_indefinite_precision_with_metadata():
    precision = np.array([[2.0, 0.0], [0.0, -0.5]])

    result = covariance_from_negative_hessian(
        precision,
        source="fd_psb_hessian",
        jitter=0.0,
        eigenvalue_floor=0.25,
    )

    assert result.precision_eigen_summary["positive"] is True
    assert result.covariance_eigen_summary["positive"] is True
    assert result.regularization_report["raw_nonpositive_eigenvalue_count"] == 1
    assert result.regularization_report["clipped_eigenvalue_count"] == 1
    assert result.regularization_report["effective_eigenvalue_floor"] == pytest.approx(
        0.25
    )
    np.testing.assert_allclose(
        result.regularized_precision,
        np.array([[2.0, 0.0], [0.0, 0.25]]),
    )


def test_hessian_mass_builder_records_symmetry_projection_metadata():
    precision = np.array([[2.0, 0.25], [0.0, 3.0]])

    result = covariance_from_precision(
        precision,
        source="finite_difference_hessian",
        jitter=0.0,
    )

    assert result.regularization_report["symmetry_projection"] == (
        "average_with_transpose"
    )
    assert result.regularization_report["input_asymmetric"] is True
    assert result.regularization_report["input_asymmetry_max_abs"] == pytest.approx(
        0.25
    )
    np.testing.assert_allclose(
        result.regularized_precision,
        tf.linalg.matrix_transpose(result.regularized_precision),
    )


def test_mass_matrix_runtime_module_is_tensorflow_only() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "bayesfilter"
        / "inference"
        / "mass_matrix.py"
    ).read_text(encoding="utf-8")

    assert "import numpy" not in source
    assert "import tensorflow as tf" in source


def test_hessian_mass_builder_fails_closed_on_nonfinite_precision():
    with pytest.raises(ValueError, match="precision must be finite"):
        regularize_precision(np.array([[1.0, np.nan], [np.nan, 2.0]]))


def test_precomputed_mass_artifact_records_hessian_regularization_metadata():
    precision = np.array([[2.0, 0.0], [0.0, -0.5]])

    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.array([0.0, 0.0]),
        negative_hessian=precision,
        adapter_signature="test_hessian_mass_adapter_v1",
        covariance_source="fd_psb_hessian",
        jitter=0.0,
        eigenvalue_floor=0.25,
    )

    np.testing.assert_allclose(
        artifact.covariance,
        np.array([[0.5, 0.0], [0.0, 4.0]]),
    )
    np.testing.assert_allclose(artifact.factor @ artifact.factor.T, artifact.covariance)
    assert artifact.covariance_source == "fd_psb_hessian"
    assert artifact.eigen_summary["positive"] is True
    assert artifact.precision_eigen_summary["positive"] is True
    assert artifact.regularization_report["raw_nonpositive_eigenvalue_count"] == 1
    assert artifact.regularization_report["silent_eigenvalue_reflection"] is False
    assert "no posterior convergence claim" in artifact.nonclaims


def test_hessian_mass_builder_rejects_adapter_signature_mismatch():
    class Adapter:
        parameter_dim = 2

        def parameter_names(self):
            return ("a", "b")

    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.zeros(2),
        negative_hessian=np.eye(2),
        adapter_signature="not_the_adapter_signature",
        jitter=0.0,
    )

    with pytest.raises(ValueError, match="adapter signature mismatch"):
        artifact.validate_for_adapter(Adapter(), expected_dim=2)


def test_hessian_mass_builder_diagonal_fallback_records_provenance():
    result = covariance_from_precision(
        np.array([[4.0, 1.0], [1.0, 3.0]]),
        source="analytic_map_hessian",
        jitter=0.0,
        dense=False,
    )

    assert result.matrix_kind == "diagonal"
    np.testing.assert_allclose(result.covariance, np.diag([0.25, 1.0 / 3.0]))
    assert result.regularization_report["diagonal_fallback_used"] is True
    assert (
        result.regularization_report["diagonal_fallback_source"]
        == "regularized_precision_diagonal"
    )


def test_precomputed_mass_artifact_diagonal_fallback_records_provenance():
    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.zeros(2),
        negative_hessian=np.array([[4.0, 1.0], [1.0, 3.0]]),
        adapter_signature="test_diagonal_fallback_adapter_v1",
        jitter=0.0,
        dense=False,
    )

    assert artifact.regularization_report["diagonal_fallback_used"] is True
    assert (
        artifact.regularization_report["diagonal_fallback_source"]
        == "regularized_precision_diagonal"
    )
    payload = artifact.signature_payload()
    assert payload["regularization_report"]["diagonal_fallback_used"] is True


def test_precomputed_mass_artifact_payload_round_trips_json_arrays():
    artifact = PrecomputedMassArtifact.from_covariance(
        position=np.array([0.25, -0.5]),
        covariance=np.array([[1.0, 0.2], [0.2, 1.5]], dtype=float),
        adapter_signature="roundtrip_adapter_v1",
        position_role="posterior_mode",
        covariance_source="reviewed_regularized_covariance",
        source="phase2_reviewed_mass",
        jitter=0.0,
        regularization_report={"method": "none", "jitter": 0.0},
        reconstruction_rtol=1.0e-9,
        reconstruction_atol=1.0e-11,
    )

    payload = artifact.to_payload(include_arrays=True)
    assert payload["eigen_summary"]["positive"] is True
    payload["eigen_summary"] = {"finite": False, "positive": False}
    loaded = json.loads(json.dumps(payload, sort_keys=True))
    restored = PrecomputedMassArtifact.from_payload(
        loaded,
        expected_adapter_signature="roundtrip_adapter_v1",
        expected_dim=2,
    )

    assert payload["artifact_type"] == "bayesfilter_precomputed_mass_artifact"
    assert payload["schema_version"] == 1
    assert payload["include_arrays"] is True
    assert payload["position_role"] == "posterior_mode"
    assert payload["covariance_source"] == "reviewed_regularized_covariance"
    assert payload["matrix_used_for_square_root"] == "regularized_covariance"
    assert payload["factor_orientation"] == "row_right_transpose"
    assert payload["source"] == "phase2_reviewed_mass"
    assert payload["log_jacobian_convention"] == "constant_omitted"
    assert payload["nonclaims"] == list(artifact.nonclaims)
    assert payload["regularization_report"] == {"method": "none", "jitter": 0.0}
    assert payload["reconstruction_rtol"] == pytest.approx(1.0e-9)
    assert payload["reconstruction_atol"] == pytest.approx(1.0e-11)
    np.testing.assert_allclose(restored.position, artifact.position)
    np.testing.assert_allclose(restored.covariance, artifact.covariance)
    np.testing.assert_allclose(restored.factor, artifact.factor)
    assert restored.eigen_summary["finite"] is True
    assert restored.eigen_summary["positive"] is True
    assert restored.adapter_signature == artifact.adapter_signature
    assert restored.position.flags.writeable is False
    assert restored.covariance.flags.writeable is False
    assert restored.factor.flags.writeable is False


def test_precomputed_mass_artifact_payload_metadata_only_omits_arrays():
    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.zeros(2),
        negative_hessian=np.eye(2),
        adapter_signature="metadata_only_adapter_v1",
        jitter=0.0,
    )

    payload = artifact.to_payload(include_arrays=False)

    assert payload["include_arrays"] is False
    assert "position" not in payload
    assert "covariance" not in payload
    assert "factor" not in payload
    expected = json.loads(json.dumps(artifact.signature_payload(), sort_keys=True))
    for key, value in expected.items():
        assert payload[key] == value
    with pytest.raises(ValueError, match="missing required array fields"):
        PrecomputedMassArtifact.from_payload(payload)


def test_precomputed_mass_artifact_payload_rejects_stale_adapter_signature():
    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.zeros(2),
        negative_hessian=np.eye(2),
        adapter_signature="fresh_adapter_v1",
        jitter=0.0,
    )
    payload = artifact.to_payload(include_arrays=True)

    with pytest.raises(ValueError, match="adapter signature mismatch"):
        PrecomputedMassArtifact.from_payload(
            payload,
            expected_adapter_signature="stale_adapter_v1",
        )


def test_precomputed_mass_artifact_payload_rejects_dimension_mismatch():
    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.zeros(2),
        negative_hessian=np.eye(2),
        adapter_signature="dimension_adapter_v1",
        jitter=0.0,
    )
    payload = artifact.to_payload(include_arrays=True)

    with pytest.raises(ValueError, match="dimension mismatch"):
        PrecomputedMassArtifact.from_payload(payload, expected_dim=3)


def test_precomputed_mass_artifact_payload_rejects_wrong_schema():
    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.zeros(2),
        negative_hessian=np.eye(2),
        adapter_signature="schema_adapter_v1",
        jitter=0.0,
    )
    payload = dict(artifact.to_payload(include_arrays=True))

    wrong_type = dict(payload)
    wrong_type["artifact_type"] = "local_client_payload"
    with pytest.raises(ValueError, match="artifact_type mismatch"):
        PrecomputedMassArtifact.from_payload(wrong_type)

    wrong_version = dict(payload)
    wrong_version["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version mismatch"):
        PrecomputedMassArtifact.from_payload(wrong_version)


def test_precomputed_mass_artifact_payload_rejects_corrupted_covariance_and_factor():
    artifact = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=np.array([[1.0, 0.1], [0.1, 1.0]], dtype=float),
        adapter_signature="corruption_adapter_v1",
        covariance_source="test_covariance",
        jitter=0.0,
    )
    covariance_payload = dict(artifact.to_payload(include_arrays=True))
    covariance_payload["covariance"] = [[1.0, 0.0], [0.0, 0.0]]
    covariance_payload["factor"] = [[1.0, 0.0], [0.0, 0.0]]
    factor_payload = dict(artifact.to_payload(include_arrays=True))
    factor_payload["factor"] = [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="positive definite"):
        PrecomputedMassArtifact.from_payload(covariance_payload)
    with pytest.raises(ValueError, match="factor must reconstruct covariance"):
        PrecomputedMassArtifact.from_payload(factor_payload)
