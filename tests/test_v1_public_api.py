import os
from pathlib import Path
import subprocess
import sys

import bayesfilter
import bayesfilter.linear


V1_PUBLIC_SYMBOLS = {
    "TFLinearGaussianStateSpace",
    "TFLinearGaussianStateSpaceDerivatives",
    "TFLinearGaussianStateSpaceFirstDerivatives",
    "StationaryLGSSMFirstDerivativeCoverage",
    "continuous_lyapunov_first_derivatives_tf",
    "continuous_lyapunov_solution_tf",
    "diffusion_from_cholesky_first_derivatives_tf",
    "first_to_full_linear_gaussian_derivatives",
    "matrix_exponential_frechet_tf",
    "stationary_lgssm_first_derivative_coverage",
    "stationary_lgssm_from_continuous_first_derivatives_tf",
    "tf_linear_gaussian_log_likelihood",
    "tf_kalman_log_likelihood",
    "tf_masked_kalman_log_likelihood",
    "tf_qr_linear_gaussian_log_likelihood",
    "tf_qr_linear_gaussian_score",
    "tf_qr_sqrt_kalman_log_likelihood",
    "tf_qr_sqrt_masked_kalman_log_likelihood",
    "tf_qr_linear_gaussian_score_hessian",
    "tf_qr_sqrt_kalman_score_hessian",
    "tf_qr_sqrt_masked_kalman_score_hessian",
    "tf_svd_linear_gaussian_log_likelihood",
    "tf_svd_linear_gaussian_score_first_order",
    "tf_svd_linear_gaussian_score_first_order_graph_status",
    "tf_svd_linear_gaussian_score_hessian",
    "tf_svd_linear_gaussian_score_hessian_graph_status",
    "tf_svd_kalman_log_likelihood",
    "tf_svd_masked_kalman_log_likelihood",
    "TFStructuralStateSpace",
    "structural_block_metadata",
    "structural_filter_diagnostics",
    "structural_filter_metadata",
    "pointwise_deterministic_residuals",
    "affine_structural_to_linear_gaussian_tf",
    "make_affine_structural_tf",
    "tf_batched_svd_sigma_point_value_and_score_custom_gradient",
    "tf_svd_sigma_point_log_likelihood",
    "tf_svd_sigma_point_log_likelihood_with_rule",
    "tf_svd_sigma_point_filter",
    "tf_svd_sigma_point_placement",
    "tf_svd_sigma_point_score_with_rule",
    "tf_svd_cubature_score",
    "tf_principal_sqrt_ukf_score",
    "tf_svd_ukf_score",
    "tf_unit_sigma_point_rule",
    "tf_cut4g_sigma_point_rule",
    "tf_svd_cut4_log_likelihood",
    "tf_svd_cut4_filter",
    "tf_svd_cut4_score",
    "TFStructuralFirstDerivatives",
}


COMMON_INFERENCE_RUNTIME_SYMBOLS = {
    "InvalidNeuTraBatchTarget",
    "NeuTraBatchTargetBinding",
    "NEUTRA_BATCHING_SCHEMA",
    "batch_native_value_status_target_fn",
    "bind_batch_native_neutra_target",
    "require_batch_native_neutra_target",
    "NeuTraTrainingError",
    "PlainDenseIAFTrainingConfig",
    "PlainDenseIAFTrainingResult",
    "PlainDenseIAFTransport",
    "train_plain_dense_iaf",
    "CandidateResult",
    "BackendParityGate",
    "BackendParityResult",
    "BackendParityRow",
    "BatchValueScoreMetadata",
    "BatchValueScoreResult",
    "CovariancePositiveDefiniteError",
    "EVIDENCE_MANIFEST_SCOPES",
    "EvidenceManifest",
    "GPUSelection",
    "HMCFailureClassification",
    "HMC_TUNING_POLICY_LABELS",
    "HMCDiagnosticSummary",
    "HMCLogAcceptSummary",
    "HMCScreenResult",
    "HMCSampleArchiveManifest",
    "HMCStreamingSampleArchiveSink",
    "HMCTuningDiagnosticResult",
    "HMCTuningPolicy",
    "HessianPosteriorAdapter",
    "FactorizationFailure",
    "LatentAffineHMCTransform",
    "MassMatrixResult",
    "PartialResultSnapshot",
    "PosteriorAdapter",
    "PrecomputedMassArtifact",
    "PrecomputedMAP",
    "PriorSupportError",
    "QUADRATIC_MAP_COVARIANCE_NONCLAIMS",
    "QuadraticMapCovarianceLocatorConfig",
    "QuadraticMapCovarianceMassConfig",
    "QuadraticMapCovarianceResult",
    "ReducerRowStatus",
    "RunManifest",
    "SolveResidualError",
    "StageEvent",
    "StationarityError",
    "TIMING_BUCKET_NAMES",
    "TargetFailureClassification",
    "TargetFailurePolicy",
    "TargetPolicyEvaluation",
    "TargetRegionError",
    "TimingBucket",
    "TimeoutRecord",
    "UnsupportedValueScoreAuthority",
    "ValueScoreCapability",
    "ValueScorePosteriorAdapter",
    "FixedTrajectoryCandidateResult",
    "FixedTrajectoryTuningConfig",
    "FixedTrajectoryTuningResult",
    "BOOTSTRAP_SCREEN_NONCLAIMS",
    "InternalSegmentHMCRunResult",
    "InternalSegmentHMCRunner",
    "InternalSegmentHMCRunnerConfig",
    "RetainedSampleHMCArchiveConfig",
    "RetainedSampleHMCArchiveRunResult",
    "RetainedSampleHMCArchiveRunner",
    "FROZEN_STEP_TRAJECTORY_STAGE_NONCLAIMS",
    "GEOMETRY_INITIALIZATION_NONCLAIMS",
    "FixedTransportValueScoreAdapter",
    "GENERIC_HMC_TUNING_NONCLAIMS",
    "GenericHMCCandidateEvaluation",
    "GenericHMCCandidateResult",
    "GenericHMCTuningArtifact",
    "GenericHMCTuningConfig",
    "GenericHMCTuningResult",
    "HMCGeometryInitializationConfig",
    "HMCGeometryInitializationResult",
    "HMCBootstrapRepairRound",
    "HMCBootstrapScreenConfig",
    "HMCBootstrapScreenResult",
    "HMCFixedMassStepStageConfig",
    "HMCFixedMassStepStageResult",
    "HMCFrozenStepTrajectoryStageConfig",
    "HMCFrozenStepTrajectoryStageResult",
    "HMCKernelTuningConfig",
    "HMCKernelTuningResult",
    "HMCTuneVerifyRepairAttempt",
    "HMCTuneVerifyRepairLoopConfig",
    "HMCTuneVerifyRepairLoopResult",
    "HMCWindowedMassStageConfig",
    "HMCWindowedMassStageResult",
    "HMC_KERNEL_TUNING_PUBLIC_NONCLAIMS",
    "WelfordCovarianceResult",
    "WindowedMassAdaptationConfig",
    "WindowedMassAdaptationResult",
    "WindowedMassUpdate",
    "WindowedWarmupWindow",
    "WINDOWED_MASS_STAGE_NONCLAIMS",
    "FIXED_MASS_STEP_STAGE_NONCLAIMS",
    "TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS",
    "VALID_REDUCER_STATUSES",
    "WorkerManifest",
    "WorkerRecord",
    "append_heartbeat",
    "append_stage_event",
    "assert_cpu_only_env",
    "build_worker_manifest",
    "canonical_candidate_order",
    "build_internal_segment_hmc_runner",
    "build_retained_sample_hmc_archive_runner",
    "build_windowed_warmup_schedule",
    "classify_target_failure_mode",
    "classify_fixed_kernel_screen_with_tuning_policy",
    "classify_hmc_screen",
    "classify_hmc_tuning_diagnostic",
    "configs_match_exact",
    "covariance_from_negative_hessian",
    "covariance_from_precision",
    "ensure_cpu_only_env",
    "evaluate_target_with_failure_policy",
    "evaluate_batch_native_value_score",
    "estimate_quadratic_map_covariance",
    "initialize_hmc_kernel_geometry",
    "latent_to_position",
    "latent_value_and_score",
    "make_timing_bucket",
    "normalize_hmc_tuning_policy",
    "orchestrate_generic_hmc_tuning",
    "production_leapfrog_count",
    "record_timeout",
    "record_worker_result",
    "regularize_covariance",
    "reduce_worker_artifacts",
    "select_first_tie_candidate",
    "select_preferred_gpu",
    "require_executable_tuning_policy",
    "run_fixed_trajectory_tuning_diagnostic",
    "run_hmc_bootstrap_screen",
    "run_hmc_fixed_mass_step_stage",
    "run_hmc_frozen_step_trajectory_stage",
    "run_hmc_tune_verify_repair_loop",
    "run_hmc_windowed_mass_stage",
    "tune_hmc_kernel",
    "run_generic_hmc_tuning_orchestration",
    "run_gaussian_dual_averaging_diagnostic",
    "reviewed_value_score_target_fn",
    "run_windowed_mass_adaptation_diagnostic",
    "screen_hmc_diagnostics",
    "stable_adapter_signature",
    "stale_artifacts_match_exact",
    "stable_config_hash",
    "stale_match_payload",
    "summarize_hmc_diagnostics",
    "summarize_log_accept_ratios",
    "validate_precomputed_map",
    "validate_windowed_shrinkage_target",
    "value_score_capability",
    "welford_covariance",
    "whitening_from_covariance",
    "write_partial_result_snapshot",
    "write_evidence_manifest",
    "write_hmc_sample_archive",
    "write_worker_manifest",
}


def test_v1_public_api_symbols_are_top_level_importable() -> None:
    missing = sorted(name for name in V1_PUBLIC_SYMBOLS if not hasattr(bayesfilter, name))

    assert missing == []
    assert V1_PUBLIC_SYMBOLS.issubset(set(bayesfilter.__all__))


def test_linear_public_score_symbol_is_subpackage_importable() -> None:
    assert hasattr(bayesfilter.linear, "tf_qr_linear_gaussian_score")
    assert "tf_qr_linear_gaussian_score" in bayesfilter.linear.__all__


def test_common_inference_runtime_symbols_are_top_level_importable() -> None:
    missing = sorted(
        name for name in COMMON_INFERENCE_RUNTIME_SYMBOLS if not hasattr(bayesfilter, name)
    )

    assert missing == []
    assert COMMON_INFERENCE_RUNTIME_SYMBOLS.issubset(set(bayesfilter.__all__))


def test_v1_public_api_import_does_not_import_external_clients() -> None:
    forbidden_prefixes = ("dsge_hmc", "MacroFinance", "macrofinance")
    imported_clients = sorted(
        name for name in sys.modules if name.startswith(forbidden_prefixes)
    )

    assert imported_clients == []


def test_top_level_import_does_not_import_tensorflow_until_symbol_resolution() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    code = (
        "import json, sys; "
        "import bayesfilter; "
        "print(json.dumps({"
        "'tensorflow': 'tensorflow' in sys.modules, "
        "'tensorflow_probability': 'tensorflow_probability' in sys.modules, "
        "'has_all': hasattr(bayesfilter, '__all__')"
        "}, sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == (
        '{"has_all": true, "tensorflow": false, '
        '"tensorflow_probability": false}'
    )


def test_inference_package_import_remains_lazy() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    code = (
        "import json, sys; "
        "import bayesfilter.inference as inference; "
        "print(json.dumps({"
        "'tensorflow': 'tensorflow' in sys.modules, "
        "'tensorflow_probability': 'tensorflow_probability' in sys.modules, "
        "'has_training_symbol': 'train_plain_dense_iaf' in inference.__all__"
        "}, sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == (
        '{"has_training_symbol": true, "tensorflow": false, '
        '"tensorflow_probability": false}'
    )
