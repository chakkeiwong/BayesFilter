from __future__ import annotations

import sys

import numpy as np
import pytest

from bayesfilter.inference import (
    BackendParityGate,
    BackendParityRow,
    FactorizationFailure,
    HMC_TUNING_POLICY_LABELS,
    HMCTuningPolicy,
    LatentAffineHMCTransform,
    PrecomputedMassArtifact,
    PrecomputedMAP,
    PriorSupportError,
    TargetFailurePolicy,
    UnsupportedValueScoreAuthority,
    ValueScoreCapability,
    classify_fixed_kernel_screen_with_tuning_policy,
    classify_hmc_tuning_diagnostic,
    classify_target_failure_mode,
    _make_hmc_target_log_prob_fn,
    evaluate_target_with_failure_policy,
    classify_hmc_screen,
    covariance_from_precision,
    latent_to_position,
    latent_value_and_score,
    normalize_hmc_tuning_policy,
    require_executable_tuning_policy,
    screen_hmc_diagnostics,
    stable_adapter_signature,
    static_unroll_chain_value_and_score,
    summarize_hmc_diagnostics,
    summarize_log_accept_ratios,
    validate_precomputed_map,
    value_score_capability,
)
from bayesfilter.runtime import (
    CandidateResult,
    EvidenceManifest,
    RunManifest,
    WorkerManifest,
    build_trusted_gpu_snapshot,
    configs_match_exact,
    ensure_cpu_only_env,
    ensure_gpu_memory_growth_env,
    select_first_tie_candidate,
    select_preferred_gpu,
    stable_config_hash,
)


class DebugValueScoreAdapter:
    parameter_dim = 2

    def parameter_names(self):
        return ("alpha", "beta")

    def log_prob_and_grad(self, theta):
        theta = np.asarray(theta, dtype=float)
        return -0.5 * float(theta @ theta), -theta


class ReviewedAdapter(DebugValueScoreAdapter):
    def log_prob(self, theta):
        value, _grad = self.log_prob_and_grad(theta)
        return value

    def value_score_capability(self):
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="toy_graph",
            nonclaims=("toy fixture only",),
        )


class ReviewedTapeAdapter(DebugValueScoreAdapter):
    def log_prob(self, theta):
        value, _grad = self.log_prob_and_grad(theta)
        return value

    def value_score_capability(self):
        return ValueScoreCapability(
            "reviewed_gradient_tape_xla_exception",
            True,
            evidence_path="docs/plans/reviewed-target.md",
            target_scope="toy-target",
            nonclaims=("target-specific exception only",),
        )


class ProcessLocalSignatureAdapter(DebugValueScoreAdapter):
    def adapter_signature(self):
        return f"{self!r}:id({id(self)})"


def test_cpu_only_env_hides_gpu_before_framework_import(monkeypatch):
    env = {}
    ensure_cpu_only_env(env)

    assert env["CUDA_VISIBLE_DEVICES"] == "-1"

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="before framework import"):
        ensure_cpu_only_env()


def test_gpu_memory_growth_env_is_required_before_framework_import(monkeypatch):
    env = {}
    ensure_gpu_memory_growth_env(env)
    assert env["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
    with pytest.raises(RuntimeError, match="before framework import"):
        ensure_gpu_memory_growth_env()


def test_value_score_authority_fails_closed_for_xla_and_unknown_labels():
    capability = value_score_capability(DebugValueScoreAdapter())
    assert capability.value_score_authority == "gradient_tape_fallback"
    assert capability.xla_hmc_ready is False

    with pytest.raises(ValueError, match="XLA HMC requires"):
        _make_hmc_target_log_prob_fn(DebugValueScoreAdapter(), use_xla=True)

    with pytest.raises(UnsupportedValueScoreAuthority):
        ValueScoreCapability("locally_named_authority", True)  # type: ignore[arg-type]


def test_reviewed_gradient_tape_exception_must_be_scoped():
    with pytest.raises(ValueError, match="scoped"):
        ValueScoreCapability("reviewed_gradient_tape_xla_exception", True)

    capability = ValueScoreCapability(
        "reviewed_gradient_tape_xla_exception",
        True,
        evidence_path="docs/plans/reviewed-target.md",
        target_scope="toy-target",
        nonclaims=("target-specific exception only",),
    )
    assert capability.is_accepted_xla_hmc_authority is True
    assert capability.is_accepted_full_chain_xla_diagnostic_authority is False


def test_full_chain_xla_diagnostic_authority_requires_scoped_target_authority():
    with pytest.raises(ValueError, match="accepted target-XLA authority"):
        ValueScoreCapability(
            "gradient_tape_fallback",
            False,
            full_chain_xla_diagnostic_ready=True,
        )

    with pytest.raises(ValueError, match="target_scope"):
        ValueScoreCapability(
            "graph_native",
            True,
            evidence_path="docs/plans/full-chain-xla.md",
            full_chain_xla_diagnostic_ready=True,
        )

    capability = ValueScoreCapability(
        "graph_native",
        True,
        evidence_path="docs/plans/full-chain-xla.md",
        target_scope="toy-target",
        nonclaims=("tiny full-chain diagnostic only",),
        full_chain_xla_diagnostic_ready=True,
    )
    assert capability.is_accepted_xla_hmc_authority is True
    assert capability.is_accepted_full_chain_xla_diagnostic_authority is True


def test_reviewed_gradient_tape_exception_is_bound_to_target_scope():
    target = _make_hmc_target_log_prob_fn(
        ReviewedTapeAdapter(),
        use_xla=True,
        target_scope="toy-target",
    )
    assert target(np.array([1.0, 2.0])) == pytest.approx(-2.5)

    with pytest.raises(ValueError, match="target_scope mismatch"):
        _make_hmc_target_log_prob_fn(
            ReviewedTapeAdapter(),
            use_xla=True,
            target_scope="other-target",
        )


def test_xla_target_allows_reviewed_graph_native_adapter():
    target = _make_hmc_target_log_prob_fn(ReviewedAdapter(), use_xla=True)

    assert target(np.array([1.0, 2.0])) == pytest.approx(-2.5)


def test_static_unroll_chain_value_score_preserves_order_and_matches_scalar():
    adapter = ReviewedAdapter()
    chain = np.array([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]])
    values, scores = static_unroll_chain_value_and_score(adapter, chain, use_xla=True)

    expected_values = []
    expected_scores = []
    for row in chain:
        value, score = adapter.log_prob_and_grad(row)
        expected_values.append(value)
        expected_scores.append(score)

    np.testing.assert_allclose(values, np.asarray(expected_values))
    np.testing.assert_allclose(scores, np.asarray(expected_scores))
    assert float(values[0]) < float(values[2])


def test_static_unroll_chain_value_score_fails_closed_without_reviewed_authority():
    with pytest.raises(ValueError, match="XLA chain-batched"):
        static_unroll_chain_value_and_score(
            DebugValueScoreAdapter(),
            np.zeros((2, 2)),
            use_xla=True,
        )


def test_static_unroll_chain_value_score_requires_static_rank_two():
    with pytest.raises(ValueError, match="rank 2"):
        static_unroll_chain_value_and_score(
            ReviewedAdapter(),
            np.zeros(2),
            use_xla=True,
        )


def test_precomputed_map_validates_shape_signature_and_avoids_process_ids():
    adapter = DebugValueScoreAdapter()
    signature = stable_adapter_signature(adapter)
    assert "object at" not in signature
    assert str(id(adapter)) not in signature

    artifact = PrecomputedMAP(
        position=np.zeros(2),
        covariance=np.eye(2),
        adapter_signature=signature,
    )
    position, covariance = validate_precomputed_map(artifact, adapter, expected_dim=2)
    np.testing.assert_allclose(position, np.zeros(2))
    np.testing.assert_allclose(covariance, np.eye(2))

    with pytest.raises(ValueError, match="dimension"):
        validate_precomputed_map(artifact, adapter, expected_dim=3)
    with pytest.raises(ValueError, match="signature"):
        validate_precomputed_map(
            PrecomputedMAP(np.zeros(2), np.eye(2), "stale"),
            adapter,
        )


def test_precomputed_map_rejects_covariance_shape_mismatch():
    adapter = DebugValueScoreAdapter()
    signature = stable_adapter_signature(adapter)

    with pytest.raises(ValueError, match="covariance"):
        validate_precomputed_map(
            PrecomputedMAP(np.zeros(2), np.ones((2, 1)), signature),
            adapter,
            expected_dim=2,
        )


def test_precomputed_mass_artifact_from_negative_hessian_builds_transform():
    adapter = DebugValueScoreAdapter()
    signature = stable_adapter_signature(adapter)
    precision = np.array([[4.0, 1.0], [1.0, 3.0]])
    artifact = PrecomputedMassArtifact.from_negative_hessian(
        position=np.array([0.25, -0.5]),
        negative_hessian=precision,
        adapter_signature=signature,
        covariance_source="analytic_full_hessian",
        jitter=0.0,
    )

    position, covariance, factor = artifact.validate_for_adapter(adapter, expected_dim=2)
    np.testing.assert_allclose(position, np.array([0.25, -0.5]))
    np.testing.assert_allclose(covariance, np.linalg.inv(precision))
    np.testing.assert_allclose(factor @ factor.T, covariance)
    assert artifact.position_role == "map"
    assert artifact.covariance_source == "analytic_full_hessian"
    assert artifact.matrix_used_for_square_root == "covariance_from_negative_hessian"
    assert artifact.factor_orientation == "row_right_transpose"
    assert artifact.eigen_summary["positive"] is True

    transform = artifact.build_latent_transform()
    z = np.array([0.1, -0.2])
    np.testing.assert_allclose(
        transform.latent_to_position(z),
        artifact.position + z @ artifact.factor.T,
    )


def test_precomputed_mass_artifact_preserves_non_map_position_role():
    adapter = DebugValueScoreAdapter()
    covariance = np.array([[2.0, 0.25], [0.25, 1.0]])
    artifact = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=covariance,
        adapter_signature=stable_adapter_signature(adapter),
        position_role="diagnostic_center_not_map",
        covariance_source="matched_dgp_hessian_scaled_initialization",
        matrix_used_for_square_root="Sigma_phi_reg",
        jitter=0.0,
    )

    assert artifact.position_role == "diagnostic_center_not_map"
    assert artifact.matrix_used_for_square_root == "Sigma_phi_reg"
    assert "no production MAP quality claim" in artifact.nonclaims


def test_precomputed_mass_artifact_rejects_stale_and_process_local_signature():
    adapter = DebugValueScoreAdapter()
    artifact = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=np.eye(2),
        adapter_signature="stale",
        covariance_source="fixture",
        jitter=0.0,
    )

    with pytest.raises(ValueError, match="signature"):
        artifact.validate_for_adapter(adapter)
    with pytest.raises(ValueError, match="process-local"):
        PrecomputedMassArtifact.from_covariance(
            position=np.zeros(2),
            covariance=np.eye(2),
            adapter_signature="DebugAdapter object at 0x1234",
            covariance_source="fixture",
            jitter=0.0,
        )


def test_precomputed_mass_artifact_rejects_shape_and_reconstruction_mismatch():
    signature = stable_adapter_signature(DebugValueScoreAdapter())
    with pytest.raises(ValueError, match="factor shape"):
        PrecomputedMassArtifact(
            position=np.zeros(2),
            covariance=np.eye(2),
            factor=np.ones((2, 1)),
            adapter_signature=signature,
            covariance_source="fixture",
        )
    with pytest.raises(ValueError, match="reconstruct covariance"):
        PrecomputedMassArtifact(
            position=np.zeros(2),
            covariance=np.eye(2),
            factor=2.0 * np.eye(2),
            adapter_signature=signature,
            covariance_source="fixture",
        )
    with pytest.raises(ValueError, match="positive definite"):
        PrecomputedMassArtifact(
            position=np.zeros(2),
            covariance=np.zeros((2, 2)),
            factor=np.zeros((2, 2)),
            adapter_signature=signature,
            covariance_source="fixture",
        )


def test_explicit_adapter_signature_rejects_process_local_identity():
    with pytest.raises(ValueError, match="process-local"):
        stable_adapter_signature(ProcessLocalSignatureAdapter())


def test_dense_whitening_uses_documented_latent_score_orientation():
    center = np.array([1.0, -2.0])
    factor = np.array([[2.0, 0.0], [3.0, 4.0]])
    z = np.array([0.5, -1.0])

    np.testing.assert_allclose(latent_to_position(z, center, factor), center + z @ factor.T)

    def value_and_grad(theta):
        return float(np.sum(theta)), np.array([7.0, 11.0])

    _value, latent_score = latent_value_and_score(z, center, factor, value_and_grad)
    np.testing.assert_allclose(latent_score, factor.T @ np.array([7.0, 11.0]))


def test_latent_affine_transform_records_metadata_and_inverts_dense_factor():
    center = np.array([1.0, -2.0])
    factor = np.array([[2.0, 0.0], [3.0, 4.0]])
    transform = LatentAffineHMCTransform(
        center=center,
        factor=factor,
        covariance_provenance="analytic_full_hessian_cov",
        log_jacobian_convention="constant_omitted",
    )

    z = np.array([0.5, -1.0])
    theta = transform.latent_to_position(z)
    np.testing.assert_allclose(theta, center + z @ factor.T)
    np.testing.assert_allclose(transform.position_to_latent(theta), z)
    np.testing.assert_allclose(
        transform.theta_score_to_latent_score(np.array([7.0, 11.0])),
        factor.T @ np.array([7.0, 11.0]),
    )
    assert transform.factor_orientation == "row_right_transpose"
    assert transform.covariance_provenance == "analytic_full_hessian_cov"
    assert transform.log_jacobian_convention == "constant_omitted"


def test_latent_affine_transform_batched_row_vector_parity():
    center = np.array([1.0, -2.0])
    factor = np.array([[2.0, 0.0], [3.0, 4.0]])
    transform = LatentAffineHMCTransform(center, factor, covariance_provenance="test")
    z = np.array([[0.5, -1.0], [2.0, 0.25]])
    grad_theta = np.array([[7.0, 11.0], [-3.0, 5.0]])

    np.testing.assert_allclose(transform.latent_to_position(z), center + z @ factor.T)
    np.testing.assert_allclose(transform.position_to_latent(center + z @ factor.T), z)
    np.testing.assert_allclose(
        transform.theta_score_to_latent_score(grad_theta),
        grad_theta @ factor,
    )


def test_latent_affine_transform_rejects_shape_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="square"):
        LatentAffineHMCTransform(np.zeros(2), np.ones((2, 1)), covariance_provenance="test")
    with pytest.raises(ValueError, match="finite"):
        LatentAffineHMCTransform(np.zeros(2), np.array([[1.0, np.nan], [0.0, 1.0]]), covariance_provenance="test")
    with pytest.raises(ValueError, match="factor_orientation"):
        LatentAffineHMCTransform(
            np.zeros(2),
            np.eye(2),
            factor_orientation="column_left",
            covariance_provenance="test",
        )

    transform = LatentAffineHMCTransform(np.zeros(2), np.eye(2), covariance_provenance="test")
    with pytest.raises(ValueError, match="trailing dimension"):
        transform.latent_to_position(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        transform.theta_score_to_latent_score(np.array([1.0, np.inf]))


def test_backend_parity_gate_accepts_same_target_gaussian_value_score_rows():
    x = np.array([0.25, -0.75])
    value = -0.5 * float(x @ x)
    score = -x
    rows = (
        BackendParityRow(
            backend_name="analytic",
            target_scope="toy_gaussian",
            value=value,
            score=score,
            position=x,
        ),
        BackendParityRow(
            backend_name="matrix_form",
            target_scope="toy_gaussian",
            value=value + 2.0e-12,
            score=score + np.array([1.0e-10, -1.0e-10]),
            position=x,
        ),
    )
    result = BackendParityGate(
        rows,
        baseline_backend_name="analytic",
        value_atol=1.0e-9,
        score_atol=1.0e-8,
        require_score_parity=True,
    ).evaluate()

    assert result.passed is True
    assert result.baseline_backend_name == "analytic"
    assert result.baseline_selection == "provided"
    assert result.checks["hessian_explanatory_only"] is True
    assert "posterior convergence" in result.nonclaims[2]


def test_backend_parity_gate_fails_closed_on_target_and_derivative_mismatch():
    rows = (
        BackendParityRow("a", "target_a", value=1.0, score=np.array([1.0])),
        BackendParityRow("b", "target_b", value=1.0, score=np.array([1.0])),
    )
    result = BackendParityGate(rows, require_score_parity=True).evaluate()

    assert result.passed is False
    assert "target_scope_matches" in result.failed_checks

    with pytest.raises(ValueError, match="score must be a derivative"):
        BackendParityRow(
            "bad_score",
            "target_a",
            value=1.0,
            score=np.array([1.0]),
            derivative_target_scope="target_b",
        )


def test_backend_parity_gate_classifies_shape_and_branch_policy_mismatches():
    shape_result = BackendParityGate(
        (
            BackendParityRow("a", "target", value=1.0, score=np.ones(2)),
            BackendParityRow("b", "target", value=1.0, score=np.ones(3)),
        ),
        require_score_parity=True,
    ).evaluate()
    branch_result = BackendParityGate(
        (
            BackendParityRow("a", "target", value=1.0, branch_label="valid"),
            BackendParityRow(
                "b",
                "target",
                value=1.0,
                branch_label="invalid_support",
                failure_policy_label="finite_low_density",
            ),
        ),
    ).evaluate()

    assert shape_result.passed is False
    assert "shape_parity" in shape_result.failed_checks
    assert "score_parity" in shape_result.failed_checks
    assert branch_result.passed is False
    assert "branch_policy_parity" in branch_result.failed_checks


def test_backend_parity_gate_treats_hessian_as_explanatory_by_default():
    rows = (
        BackendParityRow("a", "target", value=1.0, hessian=np.eye(2)),
        BackendParityRow("b", "target", value=1.0, hessian=2.0 * np.eye(2)),
    )
    result = BackendParityGate(rows, hessian_atol=1.0e-12).evaluate()

    assert result.passed is True
    assert result.checks["hessian_parity"] is True
    assert result.max_hessian_abs_diff == pytest.approx(1.0)
    assert result.row_results[1]["hessian_parity_explanatory_passed"] is False
    with pytest.raises(ValueError, match="reviewed_hessian_contract"):
        BackendParityGate(rows, hessian_role="hard_reviewed")


def test_backend_parity_gate_does_not_compare_hessians_across_coordinates():
    rows = (
        BackendParityRow(
            "parameter_hessian",
            "target",
            value=1.0,
            hessian=np.eye(2),
            coordinate_scope="parameter",
        ),
        BackendParityRow(
            "latent_hessian",
            "target",
            value=1.0,
            hessian=3.0 * np.eye(2),
            coordinate_scope="latent",
        ),
    )
    result = BackendParityGate(rows).evaluate()
    reviewed = BackendParityGate(
        rows,
        hessian_role="hard_reviewed",
        reviewed_hessian_contract="fixture_contract",
    ).evaluate()

    assert result.passed is True
    assert result.max_hessian_abs_diff is None
    assert result.row_results[1]["hessian_coordinate_scope_matches"] is False
    assert result.row_results[1]["hessian_parity_explanatory_passed"] is False
    assert reviewed.passed is False
    assert "hessian_parity" in reviewed.failed_checks


def test_backend_parity_gate_validates_dense_latent_wrapper_rows():
    center = np.array([1.0, -0.5])
    factor = np.array([[2.0, 0.25], [0.5, 1.5]])
    transform = LatentAffineHMCTransform(
        center=center,
        factor=factor,
        covariance_provenance="fixture",
    )
    z = np.array([0.2, -0.1])
    theta = transform.latent_to_position(z)
    theta_score = -theta
    latent_score = transform.theta_score_to_latent_score(theta_score)
    value = -0.5 * float(theta @ theta)
    rows = (
        BackendParityRow(
            "direct_latent_chain_rule",
            "latent_wrapped_gaussian",
            value=value,
            score=latent_score,
            coordinate_scope="latent",
            latent_position=z,
            role="latent_wrapper_expected",
        ),
        BackendParityRow(
            "hmc_wrapper",
            "latent_wrapped_gaussian",
            value=value,
            score=factor.T @ theta_score,
            coordinate_scope="latent",
            latent_position=z,
            role="latent_wrapper_actual",
        ),
    )
    result = BackendParityGate(rows, require_score_parity=True).evaluate()

    assert result.passed is True
    assert result.max_score_abs_diff == pytest.approx(0.0)


def test_backend_parity_gate_requires_labeled_target_changing_regularization():
    unlabeled = BackendParityGate(
        (
            BackendParityRow("a", "target", value=1.0),
            BackendParityRow(
                "b",
                "target",
                value=1.0,
                regularization_changes_target=True,
            ),
        )
    ).evaluate()
    labeled = BackendParityGate(
        (
            BackendParityRow("a", "target", value=1.0),
            BackendParityRow(
                "b",
                "target",
                value=1.0,
                regularization_changes_target=True,
                regularization_label="singular_floor_changes_target",
            ),
        )
    ).evaluate()

    assert unlabeled.passed is False
    assert "regularization_target_changes_labeled" in unlabeled.failed_checks
    assert labeled.passed is True


def test_target_failure_policy_valid_gaussian_does_not_use_fallback():
    policy = TargetFailurePolicy(
        "toy_gaussian",
        fallback_log_prob=-123.0,
        fallback_gradient_value=0.0,
    )

    def gaussian(theta):
        theta = np.asarray(theta, dtype=float)
        return -0.5 * float(theta @ theta), -theta

    evaluation = evaluate_target_with_failure_policy(
        gaussian,
        np.array([0.25, -0.5]),
        policy,
    )
    classification = classify_target_failure_mode(evaluation)

    assert evaluation.fallback_used is False
    assert evaluation.branch_label == "valid"
    assert evaluation.value == pytest.approx(-0.15625)
    np.testing.assert_allclose(evaluation.score, np.array([-0.25, 0.5]))
    assert classification.classification == "valid_target_region"


def test_target_failure_policy_declared_support_error_gets_finite_fallback():
    policy = TargetFailurePolicy(
        "toy_support",
        fallback_log_prob=-1.0e6,
        fallback_gradient_value=0.0,
    )

    def invalid(_theta):
        raise PriorSupportError("outside support", details={"parameter": "rho"})

    evaluation = evaluate_target_with_failure_policy(
        invalid,
        np.array([2.0, 3.0]),
        policy,
    )
    classification = classify_target_failure_mode(evaluation)

    assert evaluation.fallback_used is True
    assert evaluation.value == pytest.approx(-1.0e6)
    assert evaluation.value_finite is True
    assert evaluation.score_finite is True
    np.testing.assert_allclose(evaluation.score, np.zeros(2))
    assert evaluation.branch_label == "fallback_prior_support"
    assert evaluation.failure_label == "prior_support"
    assert evaluation.diagnostics["exception_type"] == "PriorSupportError"
    assert classification.classification == "target_boundary"


def test_target_failure_policy_nonfinite_value_gradient_is_ambiguous():
    policy = TargetFailurePolicy(
        "toy_nonfinite",
        fallback_log_prob=-1.0e5,
        fallback_gradient_value=0.0,
    )

    def nonfinite(theta):
        theta = np.asarray(theta, dtype=float)
        return float("nan"), np.full_like(theta, np.nan)

    evaluation = evaluate_target_with_failure_policy(
        nonfinite,
        np.array([1.0, -1.0]),
        policy,
    )
    classification = classify_target_failure_mode(evaluation)

    assert evaluation.fallback_used is True
    assert evaluation.failure_label == "nonfinite_value_gradient"
    assert evaluation.branch_label == "fallback_nonfinite_value_gradient"
    assert evaluation.classification == "ambiguous_nonfinite_fallback"
    assert classification.classification == "ambiguous_nonfinite_target_or_backend"


def test_target_failure_policy_does_not_mask_programmer_or_shape_errors():
    policy = TargetFailurePolicy("toy_errors")

    def programmer_error(_theta):
        raise RuntimeError("bug")

    def shape_error(_theta):
        return 1.0, np.ones(3)

    with pytest.raises(RuntimeError, match="bug"):
        evaluate_target_with_failure_policy(
            programmer_error,
            np.ones(2),
            policy,
        )
    with pytest.raises(ValueError, match="score shape"):
        evaluate_target_with_failure_policy(
            shape_error,
            np.ones(2),
            policy,
        )


def test_target_failure_policy_labels_are_bounded_and_backend_breakdown_is_separate():
    with pytest.raises(ValueError, match="unknown target failure labels"):
        TargetFailurePolicy("toy", allowed_failure_labels=("local_label",))
    with pytest.raises(ValueError, match="unknown target branch labels"):
        TargetFailurePolicy("toy", allowed_branch_labels=("fallback_local",))

    policy = TargetFailurePolicy("toy_factorization")

    def factorization_failure(_theta):
        raise FactorizationFailure("chol failed")

    evaluation = evaluate_target_with_failure_policy(
        factorization_failure,
        np.ones(2),
        policy,
    )
    classification = classify_target_failure_mode(evaluation)

    assert evaluation.branch_label == "fallback_factorization_failure"
    assert evaluation.classification == "backend_numerical_fallback"
    assert classification.classification == "backend_numerical_breakdown"


def test_target_failure_policy_classifies_sampler_energy_error_after_valid_target():
    policy = TargetFailurePolicy("toy_sampler")

    def gaussian(theta):
        theta = np.asarray(theta, dtype=float)
        return -0.5 * float(theta @ theta), -theta

    evaluation = evaluate_target_with_failure_policy(
        gaussian,
        np.ones(2),
        policy,
    )
    classification = classify_target_failure_mode(
        evaluation,
        sampler_diagnostics={
            "required_arrays_finite": True,
            "zero_divergences": False,
            "log_accept_nonfinite_count_zero": True,
        },
    )

    assert evaluation.fallback_used is False
    assert classification.classification == "sampler_energy_error"


def test_mass_matrix_records_covariance_provenance():
    result = covariance_from_precision(
        np.eye(2),
        source="analytic_full_hessian",
        jitter=0.0,
    )

    assert result.source == "analytic_full_hessian"
    assert result.matrix_kind == "dense"
    np.testing.assert_allclose(result.covariance, np.eye(2))


def test_hmc_diagnostics_distinguish_unavailable_from_zero_divergences():
    unavailable = summarize_hmc_diagnostics(is_accepted=[True, False], divergences=None)
    zero = summarize_hmc_diagnostics(is_accepted=[True, True], divergences=[False, False])

    assert unavailable.divergence_status == "unavailable"
    assert unavailable.divergence_count is None
    assert zero.divergence_status == "available"
    assert zero.divergence_count == 0
    assert "convergence" in zero.nonclaims[0]


def test_log_accept_summary_counts_nonfinite_and_threshold_divergences():
    summary = summarize_log_accept_ratios([0.0, 1200.0, -1500.0, np.nan, np.inf], divergence_threshold=1000.0)

    assert summary.status == "available"
    assert summary.finite_count == 3
    assert summary.nonfinite_count == 2
    assert summary.divergence_count == 2
    assert summary.max_abs_log_accept_ratio == pytest.approx(1500.0)


def test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero():
    screen = screen_hmc_diagnostics(
        sample_chain_returned=True,
        hmc_error_absent=True,
        required_arrays_finite=True,
        log_accept_ratio=None,
        divergences=None,
        acceptance_rate_by_chain=None,
    )

    assert screen.passed is False
    assert "zero_divergences" in screen.unavailable_diagnostics
    assert "log_accept_nonfinite_count_zero" in screen.unavailable_diagnostics
    assert screen.checks["zero_divergences"] is False
    assert screen.diagnostic_roles["zero_divergences"] == "hard_veto"


def test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence():
    screen = screen_hmc_diagnostics(
        sample_chain_returned=True,
        hmc_error_absent=True,
        required_arrays_finite=True,
        log_accept_ratio=[0.0, 1200.0, -1500.0],
        divergences=None,
        acceptance_rate_by_chain=[0.5, 0.75],
    )

    assert "zero_divergences" in screen.unavailable_diagnostics
    assert screen.checks["zero_divergences"] is False
    assert screen.checks["log_accept_nonfinite_count_zero"] is True
    assert screen.diagnostic_roles["zero_divergences"] == "hard_veto"


def test_hmc_screen_classifies_fixed_kernel_acceptance_one_as_promotion_veto():
    screen = screen_hmc_diagnostics(
        sample_chain_returned=True,
        hmc_error_absent=True,
        required_arrays_finite=True,
        log_accept_ratio=[0.0, 1.0e-8, -1.0e-8],
        divergences=[False, False, False],
        acceptance_rate_by_chain=[1.0, 1.0, 1.0, 1.0],
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
        latent_initial_scale_zero=True,
        use_xla_false=True,
        compile_chain_with_xla_false=True,
    )
    classification = classify_hmc_screen(
        screen,
        acceptance_rate_by_chain=[1.0, 1.0, 1.0, 1.0],
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
        step_size=0.0005,
        num_leapfrog_steps=1,
        max_abs_log_accept_ratio=1.0e-8,
    )

    assert screen.passed is False
    assert classification.failure_class == "fixed_kernel_conservative_acceptance_veto"
    assert classification.diagnostic_role == "promotion_veto_repair_trigger"
    assert classification.screen_veto == "acceptance_rate_by_chain_strictly_between_0_05_and_0_99"
    assert "not posterior convergence" in classification.interpretation


def test_hmc_tuning_policy_labels_are_bounded_and_fixed_default_reproduces_current_behavior():
    assert set(HMC_TUNING_POLICY_LABELS) == {
        "fixed_kernel_screen",
        "dual_averaging_step_size",
        "fixed_mass_dual_averaging",
        "windowed_mass_adaptation",
        "fixed_trajectory_tuning",
        "windowed_mass_adaptation_future",
        "manual_ladder_diagnostic",
    }

    policy = normalize_hmc_tuning_policy(None)
    same_policy = normalize_hmc_tuning_policy("fixed_kernel_no_adaptation")

    assert policy == same_policy
    assert policy.label == "fixed_kernel_screen"
    assert policy.adaptation_policy == "fixed_kernel_no_adaptation"
    assert policy.num_adaptation_steps == 0
    assert policy.target_accept_prob is None
    assert policy.enabled is False
    assert policy.implemented is True
    assert "no posterior convergence claim" in policy.nonclaims


def test_hmc_tuning_policy_raw_adaptation_and_future_window_fail_closed():
    with pytest.raises(ValueError, match="fail-closed"):
        normalize_hmc_tuning_policy("dual_averaging")
    with pytest.raises(ValueError, match="label"):
        HMCTuningPolicy("locally_named_policy", "dual_averaging")

    windowed = HMCTuningPolicy.windowed_mass_adaptation_future()
    manual = HMCTuningPolicy.manual_ladder_diagnostic()

    with pytest.raises(ValueError, match="not implemented"):
        require_executable_tuning_policy(windowed)
    with pytest.raises(ValueError, match="not implemented"):
        require_executable_tuning_policy(manual)


def test_hmc_tuning_policy_dual_averaging_requires_reviewed_policy_object():
    policy = HMCTuningPolicy.dual_averaging_step_size(
        num_adaptation_steps=3,
        target_accept_prob=0.75,
        source="tests/test_common_inference_runtime_contracts.py",
    )
    executable = require_executable_tuning_policy(policy)

    assert executable.uses_dual_averaging is True
    assert executable.enabled is True
    assert executable.adaptation_policy == "dual_averaging_step_size"
    assert executable.target_accept_prob == pytest.approx(0.75)
    assert any("no posterior convergence claim" in item for item in executable.nonclaims)

    with pytest.raises(ValueError, match="adaptation steps"):
        require_executable_tuning_policy(
            HMCTuningPolicy.dual_averaging_step_size(
                num_adaptation_steps=0,
                target_accept_prob=0.75,
                source="tests/test_common_inference_runtime_contracts.py",
            )
        )


def test_hmc_tuning_policy_classifies_fixed_screen_like_existing_hmc_screen():
    screen = screen_hmc_diagnostics(
        sample_chain_returned=True,
        hmc_error_absent=True,
        required_arrays_finite=True,
        log_accept_ratio=[0.0, 1.0e-8, -1.0e-8],
        divergences=[False, False, False],
        acceptance_rate_by_chain=[1.0, 1.0, 1.0, 1.0],
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
    )
    legacy = classify_hmc_screen(
        screen,
        acceptance_rate_by_chain=[1.0, 1.0, 1.0, 1.0],
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
        step_size=0.0005,
        num_leapfrog_steps=1,
        max_abs_log_accept_ratio=1.0e-8,
    )
    explicit = classify_fixed_kernel_screen_with_tuning_policy(
        HMCTuningPolicy.fixed_kernel_screen(),
        screen,
        acceptance_rate_by_chain=[1.0, 1.0, 1.0, 1.0],
        step_size=0.0005,
        num_leapfrog_steps=1,
        max_abs_log_accept_ratio=1.0e-8,
    )

    assert explicit.failure_class == legacy.failure_class
    assert explicit.diagnostic_role == legacy.diagnostic_role
    assert explicit.screen_veto == legacy.screen_veto
    assert "not posterior convergence" in explicit.interpretation


def test_hmc_tuning_diagnostic_does_not_treat_target_invalidity_as_success():
    target_policy = TargetFailurePolicy("toy-target")

    def invalid_target(theta):
        raise PriorSupportError("outside support")

    evaluation = evaluate_target_with_failure_policy(
        invalid_target,
        np.zeros(2),
        target_policy,
    )
    classification = classify_hmc_tuning_diagnostic(
        HMCTuningPolicy.dual_averaging_step_size(
            num_adaptation_steps=3,
            target_accept_prob=0.75,
            source="tests/test_common_inference_runtime_contracts.py",
        ),
        target_evaluation=evaluation,
    )

    assert classification["passed"] is False
    assert classification["classification"] == "target_invalidity_not_tuning_success"
    assert classification["diagnostic_role"] == "hard_veto"
    assert "no posterior convergence claim" in classification["nonclaims"]


def test_hmc_screen_classifies_divergence_and_nonfinite_log_accept_as_hard_veto():
    screen = screen_hmc_diagnostics(
        sample_chain_returned=True,
        hmc_error_absent=True,
        required_arrays_finite=True,
        log_accept_ratio=[0.0, 1200.0, np.nan],
        divergences=[False, True, False],
        acceptance_rate_by_chain=[0.75, 0.625, 1.0, 1.0],
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
    )
    classification = classify_hmc_screen(
        screen,
        acceptance_rate_by_chain=[0.75, 0.625, 1.0, 1.0],
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
    )

    assert screen.checks["zero_divergences"] is False
    assert screen.checks["log_accept_nonfinite_count_zero"] is False
    assert classification.failure_class == "hmc_hard_veto_before_promotion"
    assert classification.diagnostic_role == "hard_veto"
    assert "zero_divergences" in classification.screen_failed_checks
    assert "log_accept_nonfinite_count_zero" in classification.screen_failed_checks


def test_stale_artifact_matching_includes_candidate_parallel_settings():
    base = {"target": "toy", "candidate_parallel": {"enabled": True, "workers": 8}}
    same = {"candidate_parallel": {"workers": 8, "enabled": True}, "target": "toy"}
    drifted = {"target": "toy", "candidate_parallel": {"enabled": True, "workers": 16}}

    assert configs_match_exact(base, same)
    assert not configs_match_exact(base, drifted)


def test_candidate_selection_preserves_identity_order_and_ties():
    completed = (
        CandidateResult(2, 0.2, 4, 1.0),
        CandidateResult(0, 0.1, 2, 1.0),
        CandidateResult(1, 0.1, 4, 1.0),
    )

    selected = select_first_tie_candidate(completed)
    assert selected.candidate_index == 0
    assert selected.step_size == pytest.approx(0.1)


def test_gpu_selection_policy_prefers_gpu1_then_gpu0_without_probe():
    preferred = select_preferred_gpu([0, 1], busy_gpu_ids=[])
    untrusted_busy = select_preferred_gpu([0, 1], busy_gpu_ids=[1])
    trusted_fallback = select_preferred_gpu(
        [0, 1],
        busy_gpu_ids=[1],
        trusted_or_escalated=True,
    )

    assert preferred.selected_gpu == 1
    assert preferred.fallback_used is False
    assert untrusted_busy.selected_gpu == 1
    assert "gpu_busy_evidence_not_trusted" in untrusted_busy.veto_reasons
    assert trusted_fallback.selected_gpu == 0
    assert trusted_fallback.fallback_used is True


def test_gpu_selection_snapshot_requires_trusted_busy_evidence():
    snapshot = build_trusted_gpu_snapshot(
        [
            {"index": 0, "memory_used_mb": 100, "memory_total_mb": 32000},
            {"index": 1, "memory_used_mb": 20000, "memory_total_mb": 32000},
        ],
        trusted_or_escalated=True,
        source="synthetic-test",
    )
    selected = select_preferred_gpu([], gpu_snapshot=snapshot)

    assert selected.selected_gpu == 0
    assert selected.reason == "preferred_gpu_busy_fallback_selected"


def test_gpu_selection_applies_strict_utilization_and_free_memory_rule():
    snapshot = build_trusted_gpu_snapshot(
        [
            {
                "index": 0,
                "utilization_gpu_pct": 49,
                "memory_free_mb": 8193,
            },
            {
                "index": 1,
                "utilization_gpu_pct": 50,
                "memory_free_mb": 16000,
            },
        ],
        trusted_or_escalated=True,
        source="synthetic-test",
    )
    selected = select_preferred_gpu(
        (),
        gpu_snapshot=snapshot,
        busy_memory_fraction=1.0,
        busy_utilization_pct=50,
        minimum_free_memory_mb=8192,
    )

    assert selected.selected_gpu == 0
    assert selected.reason == "preferred_gpu_busy_fallback_selected"

    snapshot["gpus"][1]["utilization_gpu_pct"] = 49
    snapshot["gpus"][1]["memory_free_mb"] = 8192
    selected = select_preferred_gpu(
        (),
        gpu_snapshot=snapshot,
        busy_memory_fraction=1.0,
        busy_utilization_pct=50,
        minimum_free_memory_mb=8192,
    )
    assert selected.selected_gpu == 0

    snapshot["gpus"][1]["memory_free_mb"] = 8193
    selected = select_preferred_gpu(
        (),
        gpu_snapshot=snapshot,
        busy_memory_fraction=1.0,
        busy_utilization_pct=50,
        minimum_free_memory_mb=8192,
    )
    assert selected.selected_gpu == 1
    assert selected.reason == "preferred_gpu_selected_trusted"


def test_gpu_selection_uses_trusted_fallback_when_gpu1_is_not_visible():
    snapshot = build_trusted_gpu_snapshot(
        [{"index": 0, "utilization_gpu_pct": 0, "memory_free_mb": 16000}],
        trusted_or_escalated=True,
        source="synthetic-test",
    )
    selected = select_preferred_gpu(
        (),
        gpu_snapshot=snapshot,
        busy_memory_fraction=1.0,
        busy_utilization_pct=50,
        minimum_free_memory_mb=8192,
    )

    assert selected.selected_gpu == 0
    assert selected.reason == "preferred_gpu_not_visible_fallback_selected"
    assert selected.fallback_used is True


def _evidence_manifest_payload(run_scope: str = "no_hmc_parity") -> dict[str, object]:
    return {
        "run_scope": run_scope,
        "git_state": {"commit": "abc123", "status_short": "M tests/example.py"},
        "command": ("python", "-m", "pytest", "tests/example.py"),
        "environment": {"python": "3.13", "package": "bayesfilter"},
        "cpu_gpu_status": "CPU-only; CUDA_VISIBLE_DEVICES=-1",
        "data_hash": "sha256:fixture",
        "target_scope": "toy-target",
        "backend": "toy-backend",
        "transform_signature": {"factor_orientation": "row_right_transpose"},
        "map_covariance_source": "analytic_full_hessian",
        "tuning_policy": "fixed_kernel_screen",
        "diagnostic_policy": "bounded_hmc_screen",
        "result_paths": {"json": "results/toy.json", "note": "docs/plans/toy.md"},
        "nonclaims": (
            "manifest completeness is not posterior convergence evidence",
            "no GPU readiness claim",
        ),
    }


def test_evidence_manifest_is_json_stable_and_wraps_existing_runtime_authority():
    run_manifest = RunManifest(
        run_id="run-1",
        command=("python", "-m", "toy"),
        git_commit="abc123",
        config_hash="config-hash",
        artifact_root="results/toy",
        cpu_gpu_status="CPU-only",
    )
    worker_manifest = WorkerManifest(
        worker_id="worker-1",
        command=("python", "-m", "toy"),
        pid=123,
        git_commit="abc123",
        artifact_root="results/toy",
        normalized_config={"alpha": 1},
        program_signature="program",
        device_policy={"CUDA_VISIBLE_DEVICES": "-1"},
        thread_caps={"intra": 1},
        worker_config={"timeout": 30},
    )
    manifest = EvidenceManifest(
        **_evidence_manifest_payload("hmc"),
        run_manifest=run_manifest,
        worker_manifest=worker_manifest,
    )
    equivalent = EvidenceManifest(
        **_evidence_manifest_payload("hmc"),
        run_manifest=run_manifest,
        worker_manifest=worker_manifest,
    )

    assert manifest.payload()["artifact_type"] == "bayesfilter_evidence_manifest"
    assert manifest.payload()["run_manifest"]["run_id"] == "run-1"
    assert manifest.payload()["worker_manifest"]["worker_id"] == "worker-1"
    assert manifest.manifest_hash == equivalent.manifest_hash
    assert manifest.manifest_hash == stable_config_hash(manifest.payload())
    assert "no GPU readiness claim" in manifest.markdown_note()


def test_evidence_manifest_rejects_process_local_identity_and_missing_fields():
    payload = _evidence_manifest_payload()
    payload["target_scope"] = "ToyAdapter object at 0x1234"
    with pytest.raises(ValueError, match="process-local"):
        EvidenceManifest(**payload)

    missing = _evidence_manifest_payload()
    missing["result_paths"] = {}
    with pytest.raises(ValueError, match="result_paths"):
        EvidenceManifest(**missing)


def test_evidence_manifest_required_scopes_are_supported_without_scientific_claims():
    for scope in ("hmc", "target_only", "no_hmc_parity"):
        manifest = EvidenceManifest(**_evidence_manifest_payload(scope))
        assert manifest.run_scope == scope
        assert "posterior convergence" in manifest.nonclaims[0]

    with pytest.raises(ValueError, match="run_scope"):
        EvidenceManifest(**_evidence_manifest_payload("macrofinance_only"))


def test_new_core_contracts_do_not_export_convergence_claims():
    capability = value_score_capability(DebugValueScoreAdapter())
    diagnostics = summarize_hmc_diagnostics(divergences=None)

    assert not hasattr(capability, "convergence_claim")
    assert not hasattr(diagnostics, "convergence_claim")
