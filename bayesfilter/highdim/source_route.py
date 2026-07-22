"""Clean-room source-route contracts for Zhao--Cui-style filtering."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.bases import (
    BoundedInterval,
    LegendreBasis1D,
    ProductBasis,
    p85_author_sir_lagrangep_algebraic_product_basis_spec,
    p85_legacy_legendre_product_basis_spec,
)
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    HighDimStatus,
    MassMeasure,
    MeasureConvention,
    assert_density_matches_mass,
    assert_tf_float64,
    freeze_mapping,
)
from bayesfilter.highdim.fitting import FixedTTFitConfig, FixedTTFitSampleBatch, FixedTTFitter
from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    SquaredTTMarginal,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.tt import TTCore


class P70FixedFitDiagnosticError(RuntimeError):
    """Carry failed fixed-fit diagnostics without making the fit admissible."""

    def __init__(self, message: str, *, status: HighDimStatus, payload: Mapping[str, object]):
        super().__init__(str(message))
        self.status = status
        self.payload = freeze_mapping(payload)


SOURCE_FAITHFUL_ROUTE_LABEL = "source_faithful_filtering"
GRADIENT_ADAPTATION_ROUTE_LABEL = "gradient_bearing_adaptation"

# HMC evaluates the same value program at changing parameter values.  Any
# route that rebuilds/adapts its approximation as theta changes is therefore
# not an admissible HMC route, even when it remains useful for diagnostics.
ZHAO_CUI_FIXED_VARIANT_ROUTE_ID = "fixed_variant_zhao_cui_source_route"
ZHAO_CUI_HMC_DEFAULT_ROUTE_ID = ZHAO_CUI_FIXED_VARIANT_ROUTE_ID
ZHAO_CUI_HMC_ROUTE_POLICY_ID = "zhao_cui_fixed_variant_hmc_default_v1"
ZHAO_CUI_FIXED_VARIANT_ROUTE_IDS = (
    ZHAO_CUI_FIXED_VARIANT_ROUTE_ID,
    "zhao_cui_exact_transformed_sv_fixed_branch_tt",
    "zhao_cui_ksc_mixture_fixed_branch_tt",
    "zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt",
    "zhao_cui_sir_d18_local_complete_data_manual_component",
)
ZHAO_CUI_HISTORICAL_ROUTE_IDS = (
    "adaptive_author_full_sol",
    "adaptive_author_pre_sol",
    "diagnostic_historical_retained_grid",
    "multistate_nonlinear_fixed_design_tt_value_path",
    "multistate_nonlinear_fixed_design_tt_score_path",
    "scalar_nonlinear_fixed_design_tt_value_path",
    "scalar_nonlinear_fixed_design_tt_score_path",
    "zhao_cui_predator_prey_t20_multistate_fixed_design_tt",
    "zhao_cui_lgssm_exact_oracle_affine_adapter",
    "zhao_cui_sir_d18_local_complete_data_manual_component",
    "zhao_cui_fixed_adjacent_state_squared_tt_v1",
)
ZHAO_CUI_HISTORICAL_ROUTE_STATUS = "historical_diagnostic_only"
ZHAO_CUI_HMC_ROUTE_STATUS = "default_hmc_fixed_variant"
ZHAO_CUI_HMC_ROUTE_BLOCK_STATUS = "blocked_hmc_route_not_fixed_variant"


def zhao_cui_hmc_route_policy(route_id: str | None = None) -> Mapping[str, object]:
    """Return the sole HMC/default Zhao--Cui route decision.

    ``route_id=None`` selects the repository default.  An explicit legacy or
    adaptive route is never silently substituted: it is classified as
    historical and blocked for HMC-facing use.  Historical artifacts remain
    readable through their own diagnostic readers.
    """

    requested = ZHAO_CUI_HMC_DEFAULT_ROUTE_ID if route_id is None else str(route_id)
    if requested in ZHAO_CUI_FIXED_VARIANT_ROUTE_IDS:
        return {
            "policy_id": ZHAO_CUI_HMC_ROUTE_POLICY_ID,
            "requested_route_id": requested,
            "selected_route_id": requested,
            "default_route_id": ZHAO_CUI_HMC_DEFAULT_ROUTE_ID,
            "status": ZHAO_CUI_HMC_ROUTE_STATUS,
            "hmc_eligible": True,
            "historical_diagnostic_only": False,
            "fail_closed": True,
        }
    if requested in ZHAO_CUI_HISTORICAL_ROUTE_IDS:
        status = ZHAO_CUI_HMC_ROUTE_BLOCK_STATUS
        historical = True
    else:
        status = ZHAO_CUI_HMC_ROUTE_BLOCK_STATUS
        historical = False
    return {
        "policy_id": ZHAO_CUI_HMC_ROUTE_POLICY_ID,
        "requested_route_id": requested,
        "selected_route_id": None,
        "default_route_id": ZHAO_CUI_HMC_DEFAULT_ROUTE_ID,
        "status": status,
        "hmc_eligible": False,
        "historical_diagnostic_only": historical,
        "fail_closed": True,
    }


def require_zhao_cui_hmc_route(route_id: str | None = None) -> str:
    """Resolve the fixed-variant route or fail closed for HMC use."""

    decision = zhao_cui_hmc_route_policy(route_id)
    if decision["hmc_eligible"] is not True:
        raise ValueError(
            "HMC-facing Zhao-Cui evaluation requires the fixed-variant source "
            f"route {ZHAO_CUI_HMC_DEFAULT_ROUTE_ID!r}; requested "
            f"{decision['requested_route_id']!r} is historical/diagnostic only"
        )
    return str(decision["selected_route_id"])


SOURCE_ROUTE_REQUIRED_OPERATION_IDS = (
    "initialize_samples",
    "push_samples",
    "augment_current_previous_state",
    "ess_enhancement_gate",
    "weighted_recenter_computeL",
    "previous_retained_object_marginalization",
    "shifted_target_construction",
    "transport_fit",
    "normalizer_update",
    "retained_sample_generation",
    "proposal_correction",
)
SOURCE_ROUTE_FORBIDDEN_DRIFT_MARKERS = (
    "pairwise_grid_transition",
    "all_grid_pairwise_transition",
    "multistate_grid_pairwise_transition",
    "retained_grid_only_route",
    "all_grid_retained_storage",
    "multistate_tt_grid_retained_storage",
    "local_neighborhood_rank_multiplier",
    "q_power_dependency_width",
)

P58_M9_READY_STATUS = "PASS_P58_M9_SOURCE_ROUTE_PIPELINE_READY_FOR_PHASE9_LAUNCH"
P58_M9_BLOCK_MISSING_ASSEMBLY_STATUS = (
    "BLOCK_P58_M9_SOURCE_ROUTE_PIPELINE_STILL_MISSING_ASSEMBLY"
)
P58_M9_BLOCK_SOURCE_DRIFT_STATUS = "BLOCK_P58_M9_SOURCE_ROUTE_PIPELINE_SOURCE_DRIFT"
P58_M9_BLOCK_HUMAN_REQUIRED_STATUS = (
    "BLOCK_P58_M9_SOURCE_ROUTE_PIPELINE_HUMAN_REQUIRED"
)

P58_M9_REQUIRED_ASSEMBLY_FLAGS = (
    "has_author_sir_callback",
    "has_fixed_ttsirt_fit_artifacts",
    "has_fixed_ttsirt_transports",
    "has_frozen_reference_samples",
    "has_source_route_step_specs",
    "has_sequential_retained_carry",
    "has_previous_marginal_evidence",
    "has_m9_runner_manifest_path",
)

P58_M9_ALLOWED_COMPARATOR_TIERS = (
    "d18_execution_only",
    "d18_same_route_rank_convergence",
    "d18_correctness_candidate",
)

P58_M9_SOURCE_ROUTE_PIPELINE_KIND = "author_sir_fixed_ttsirt_source_route"
P58_M9_AUTHOR_SIR_TARGET_ID = "zhao_cui_sir_austria_d18"
P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS = (
    "PASS_P83_MINIMAL_SOURCE_ROUTE_TRANSPORT_SLICE"
)
P83_MINIMAL_TRANSPORT_SLICE_BLOCK_STATUS = (
    "BLOCK_P83_MINIMAL_SOURCE_ROUTE_TRANSPORT_SLICE"
)
P90_VALUE_BRIDGE_TARGET_ID = P58_M9_AUTHOR_SIR_TARGET_ID
P90_VALUE_BRIDGE_COMPARATOR_LABEL = "author_formula_replay"
P90_VALUE_BRIDGE_TOLERANCE_VERSION = "p90.value_bridge.tolerances.v1"
P90_VALUE_BRIDGE_PHYSICAL_ORDERING = ("theta", "x_t", "x_t_minus_1")

P59_9A_PASS_STATUS = "PASS_P59_9A_AUTHOR_SIR_36D_TARGET_FIT_PREP"
P59_9A_BLOCK_STATUS = "BLOCK_P59_9A_AUTHOR_SIR_36D_TARGET_FIT_PREP"
P59_9A_AUTHOR_SIR_TARGET_DIMENSION = 36
P85_AUTHOR_SIR_BASIS_DOMAIN_CONFIG_STATUS = "PASS_P85_AUTHOR_SIR_BASIS_DOMAIN_CONFIG"
P85_AUTHOR_SIR_BASIS_DOMAIN_FULL_FIT_BLOCK_STATUS = (
    "BLOCK_P85_AUTHOR_SIR_BASIS_DOMAIN_FULL_FIT_REQUIRES_DOWNSTREAM_MAPPING_REPAIR"
)
P59_9C_PASS_STATUS = "PASS_P59_9C_PRECONDITIONED_ROUTE_INTEGRATION"
P59_9C_BLOCK_STATUS = "BLOCK_P59_9C_PRECONDITIONED_ROUTE_INTEGRATION"
P59_9C_FULL_ROUTE_SELECTED = "full_route_selected"
P59_9C_PRECONDITIONED_ROUTE_REQUIRED = "preconditioned_route_required"
P59_9C_PRECONDITIONED_ROUTE_DEFERRED = "preconditioned_route_deferred_with_source_reason"
P59_9C_ROUTE_DECISION_BLOCKED = "route_decision_blocked"
P59_9C_NOT_REQUIRED_PRECONDITIONED_STATUS = (
    "not_required_author_sir_mainscript_selects_full_sol"
)
P57_M8_PRECONDITIONED_PASS_STATUS = "PASS_P57_M8_PRECONDITIONED_ALGORITHM5"
P59_9B_PASS_STATUS = "PASS_P59_9B_SOURCE_ROUTE_STEP_SPEC_ASSEMBLY"
P59_9B_BLOCK_STATUS = "BLOCK_P59_9B_SOURCE_ROUTE_STEP_SPEC_ASSEMBLY"
P59_9D_PASS_STATUS = "PASS_P59_9D_RUNNER_MANIFEST_PATH"
P59_9D_BLOCK_STATUS = "BLOCK_P59_9D_RUNNER_MANIFEST_PATH"
P59_9D_DEFAULT_MANIFEST_PATH = (
    "docs/plans/bayesfilter-highdim-zhao-cui-p59-9d-runner-readiness-manifest-2026-06-11.json"
)
P59_9E_D18_EXECUTION_ONLY_PASS_STATUS = "PASS_P59_9E_D18_EXECUTION_ONLY"
P59_9E_BLOCK_STATUS = "BLOCK_P59_9E_VALIDATION_LADDER"
P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT = 9
P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT_SOURCE = (
    "p70_row_adequacy_hard_minimum_d36_degree0_rank1"
)
P60_D18_RANK_CONVERGENCE_PASS_STATUS = "PASS_P60_D18_SAME_ROUTE_RANK_CONVERGENCE"
P60_D18_RANK_CONVERGENCE_BLOCK_STATUS = "BLOCK_P60_D18_SAME_ROUTE_RANK_CONVERGENCE"
P66_FIXED_BRANCH_VALIDATION_LADDER_READY_STATUS = (
    "READY_FIXED_BRANCH_VALIDATION_LADDER_SCHEMA"
)
P66_ADJACENT_LADDER_STABLE_PASS_STATUS = "PASS_ADJACENT_LADDER_DIAGNOSTICS_STABLE"
P66_SOURCE_ROUTE_INVARIANT_DRIFT_BLOCK_STATUS = "BLOCK_SOURCE_ROUTE_INVARIANT_DRIFT"
P66_FIXED_BRANCH_DEFENSIVE_ONLY_BLOCK_STATUS = "BLOCK_FIXED_BRANCH_DEFENSIVE_ONLY"
P66_FIT_DESIGN_UNDERDETERMINED_BLOCK_STATUS = (
    "BLOCK_FIT_DESIGN_UNDERDETERMINED_FOR_CONVERGENCE"
)
P66_ADJACENT_RANK_LADDER_NOT_STABLE_BLOCK_STATUS = (
    "BLOCK_ADJACENT_RANK_LADDER_NOT_STABLE"
)
P66_ADJACENT_DEGREE_LADDER_NOT_STABLE_BLOCK_STATUS = (
    "BLOCK_ADJACENT_DEGREE_LADDER_NOT_STABLE"
)
P66_VALIDATION_LADDER_SCOPE_BLOCK_STATUS = "BLOCK_VALIDATION_LADDER_IMPLEMENTATION_SCOPE"
P66_CANDIDATE_ADMISSIBLE_STATUS = "PASS_FIXED_BRANCH_ADMISSIBLE_NONCOLLAPSED"
P66_SENTINEL_WARN_STATUS = "WARN_SENTINEL_BRANCH_DIFFERS_FROM_CANDIDATE"
P66_SENTINEL_PASS_STATUS = "PASS_SENTINEL_BRANCH_WITHIN_OLD_P60_THRESHOLDS"
P66_SAMPLE_ADEQUATE_STATUS = "PASS_SAMPLE_ADEQUATE_FOR_DIAGNOSTIC"
P66_RANK_LADDER_STABLE_STATUS = "PASS_ADJACENT_RANK_LADDER_STABLE"
P66_DEGREE_LADDER_STABLE_STATUS = "PASS_ADJACENT_DEGREE_LADDER_STABLE"
P66_RANK_LADDER_UNDERDETERMINED_STATUS = "SKIP_ADJACENT_RANK_LADDER_UNDERDETERMINED"
P66_DEGREE_LADDER_UNDERDETERMINED_STATUS = (
    "SKIP_ADJACENT_DEGREE_LADDER_UNDERDETERMINED"
)
P66_RANK_LADDER_SCHEMA_ONLY_STATUS = "SCHEMA_ONLY_ADJACENT_RANK_LADDER_NOT_EXECUTED"
P66_DEGREE_LADDER_SCHEMA_ONLY_STATUS = "SCHEMA_ONLY_ADJACENT_DEGREE_LADDER_NOT_EXECUTED"
P62_AUTHOR_TTSIRT_EXECUTABLE_DEFAULT_TAU = 1e-8
P63_AUTHOR_SIR_SOURCE_FIT_DATA_MODE = "source_pushed_computeL_resampled_local_fit"
P63_AUTHOR_SIR_FIXED_VARIANT_RESAMPLING = "deterministic_systematic_quantile"
P63_AUTHOR_SIR_EXPANSION_FACTOR = 4.0
P64_DEFENSIVE_ONLY_SQRT_NORMALIZER_TOL = 1e-14
P65_FIXED_BRANCH_INITIALIZATION_RULE = "fixed_hmc_constant_path_weighted_mean"
P70_FIXED_BRANCH_INITIALIZATION_RULE = "fixed_hmc_seeded_channel_paths_v1"
P70_SEEDED_CHANNEL_EPSILON = 1e-6
P70_FIXED_BRANCH_MAX_SWEEPS = 4
P70_FIT_RIDGE = 1e-10
P70_CONDITION_NUMBER_WARNING = 1e10
P70_CONDITION_NUMBER_VETO = 1e14
P70_CHANNEL_ACTIVITY_ABS_TOL = 1e-12
P70_CHANNEL_ACTIVITY_REL_TOL = 1e-8
P70_DEFENSIVE_ONLY_SQRT_NORMALIZER_TOL = 1e-14
P70_FIT_MASS_FRACTION_MIN = 1e-6
P70_LOG_INCREMENT_ABS_BOUND = 1e6
P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO = 10.0
P72_PASS_STATUS = "PASS_P72_SUPPORT_CERTIFIED_LOWER_GATE"
P72_BLOCK_STATUS = "BLOCK_P72_SUPPORT_CERTIFIED_LOWER_GATE"
P72_WARN_STATUS = "WARN_P72_SUPPORT_CERTIFIED_LOWER_GATE"
P72_GUARD_STEP1_PRIOR_SEED = 7321
P72_GUARD_STEP1_PROCESS_SEED = 7601
P72_GUARD_STEP2_PROCESS_SEED = 7602
P72_AUDIT_STEP1_HOLDOUT_PRIOR_SEED = 7301
P72_AUDIT_STEP1_HOLDOUT_PROCESS_SEED = 7401
P72_AUDIT_STEP1_REPLAY_PRIOR_SEED = 7311
P72_AUDIT_STEP1_REPLAY_PROCESS_SEED = 7501
P72_AUDIT_STEP2_HOLDOUT_PROCESS_SEED = 7402
P72_AUDIT_STEP2_REPLAY_PROCESS_SEED = 7502
P72_LINE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
P72_GUARD_WEIGHT_ALPHA = 1.0
P72_SHAPE_PENALTY_WEIGHT = 0.0
P72_RESIDUAL_RMS_REL_VETO = 10.0
P72_RESIDUAL_MAX_REL_VETO = 1e3
P72_LINE_GROWTH_REL_VETO = 1e3
P72_CONDITION_NUMBER_ADMISSION = 1e10
P72_EFFECTIVE_RANK_TOL = 1e-12
P72_EFFECTIVE_RANK_MIN = 1.0
P72_SQRT_SQUARE_NORMALIZER_FLOOR = 1e-14
P72_FIT_MASS_FRACTION_MIN = 1e-6
P72_LOG_NORMALIZER_ABS_BOUND = 1e6
P73_PASS_STATUS = "PASS_P73_DENSITY_AWARE_RENEWED_SUPPORT_LOWER_GATE"
P73_BLOCK_STATUS = "BLOCK_P73_DENSITY_AWARE_RENEWED_SUPPORT_LOWER_GATE"
P73_WARN_STATUS = "WARN_P73_DENSITY_AWARE_RENEWED_SUPPORT_LOWER_GATE"
P73_RENEWAL_COUNT = 1
P73_LAMBDA_CE = 0.1
P73_DENSITY_AWARE_OBJECTIVE_STATUS = "included_as_opt_in_diagnostic_arm"
P73_B_OPTIMIZER_BLOCKED = "P73_B_OPTIMIZER_BLOCKED_NONLINEAR_OBJECTIVE_NOT_IMPLEMENTED"
P73_EPS_LOG = 1e-300
P65_FIXED_BRANCH_ADAPTATION_CLASS = "fixed_hmc_adaptation"
P65_ZERO_SQRT_TT_CORE_NORM_TOL = 1e-12
P69_HOLDOUT_REPLAY_AVAILABLE_STATUS = "PASS_HOLDOUT_REPLAY_DIAGNOSTICS_AVAILABLE"
P69_HOLDOUT_REPLAY_DIAGNOSTIC_ONLY_STATUS = "WARN_HOLDOUT_REPLAY_DIAGNOSTIC_ONLY"
P69_HOLDOUT_REPLAY_MISSING_STATUS = "BLOCK_HOLDOUT_REPLAY_DIAGNOSTICS_MISSING"
P69_HOLDOUT_REPLAY_NONFINITE_STATUS = "BLOCK_HOLDOUT_REPLAY_NONFINITE"
P69_HOLDOUT_REPLAY_ROUTE_MISMATCH_STATUS = "BLOCK_HOLDOUT_REPLAY_ROUTE_MISMATCH"
P69_BRANCH_IDENTITY_DRIFT_STATUS = "BLOCK_BRANCH_IDENTITY_DRIFT"
P69_HOLDOUT_REPLAY_ROUTE_CHANGE_STATUS = "BLOCK_HOLDOUT_REPLAY_DESIGN_NEEDS_ROUTE_CHANGE"

_P62_AUTHOR_TTSIRT_DEFENSIVE_TAU_SOURCE_ANCHORS = (
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/TTSIRT.m:185-188",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:116-120",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:85",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_potential_reference.m:20-33",
)
_P63_AUTHOR_SIR_FIT_DATA_SOURCE_ANCHORS = (
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:22-30",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:49-66",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:84-99",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/computeL.m:1-35",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m:45-55",
)

_P59_9C_AUTHOR_SIR_SOURCE_ANCHORS = (
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:53-56",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:9-18",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-38",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pre_sol.m:1-10",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/tensordot/precond.m:43-56",
)
_P59_9B_AUTHOR_SIR_SOURCE_ANCHORS = (
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:14-56",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-38",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72-93",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:101-130",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:132-135",
)
_P59_9D_AUTHOR_SIR_SOURCE_ANCHORS = (
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-25",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:26-30",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:32-39",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:54-87",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:101-130",
)

_FORBIDDEN_SOURCE_STORAGE_KINDS = frozenset(
    {
        "scalar_dense_grid",
        "scalar_tt_grid",
        "multistate_tt_grid",
        "all_axes_tensor_product_grid",
    }
)

_FORBIDDEN_SOURCE_TRANSITION_INTERFACES = frozenset(
    {
        "pairwise_grid_transition",
        "all_grid_pairwise_transition",
        "multistate_grid_pairwise_transition",
    }
)


@dataclass(frozen=True)
class SourceRouteCoordinateFrame:
    """Affine frame carried by the clean-room source route."""

    mu: tf.Tensor
    matrix: tf.Tensor
    expansion_factor: float

    def __post_init__(self) -> None:
        mu = tf.convert_to_tensor(self.mu, dtype=tf.float64)
        matrix = tf.convert_to_tensor(self.matrix, dtype=tf.float64)
        if mu.shape.rank != 1 or matrix.shape.rank != 2:
            raise ValueError(f"SourceRouteCoordinateFrame: {HighDimStatus.INVALID_SHAPE.value}")
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != mu.shape[0]:
            raise ValueError(f"SourceRouteCoordinateFrame: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("mu", mu)
        assert_tf_float64("matrix", matrix)
        if not bool(
            tf.reduce_all(tf.math.is_finite(mu)).numpy()
            and tf.reduce_all(tf.math.is_finite(matrix)).numpy()
        ):
            raise ValueError(f"SourceRouteCoordinateFrame: {HighDimStatus.NONFINITE_VALUE.value}")
        det = tf.linalg.det(matrix)
        if not bool(tf.math.is_finite(det).numpy()) or bool((tf.abs(det) <= 0.0).numpy()):
            raise ValueError(f"SourceRouteCoordinateFrame: {HighDimStatus.NONFINITE_VALUE.value}")
        if float(self.expansion_factor) <= 0.0:
            raise ValueError("expansion_factor must be positive")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "expansion_factor", float(self.expansion_factor))

    @property
    def dimension(self) -> int:
        return int(self.mu.shape[0])

    def log_abs_det(self) -> tf.Tensor:
        return tf.math.log(tf.abs(tf.linalg.det(self.matrix)))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteCoordinateFrame",
            "mu": self.mu,
            "matrix": self.matrix,
            "expansion_factor": float(self.expansion_factor),
            "log_abs_det": self.log_abs_det(),
        }


@dataclass(frozen=True)
class SourceRouteSampleDiagnostics:
    """Sample and weight diagnostics carried by a source-route retained object."""

    sample_count: int
    effective_sample_size: tf.Tensor
    enhancement_attempts: int = 0

    def __post_init__(self) -> None:
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be positive")
        ess = tf.convert_to_tensor(self.effective_sample_size, dtype=tf.float64)
        if ess.shape.rank != 0:
            raise ValueError(f"effective_sample_size: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.math.is_finite(ess).numpy()) or bool((ess <= 0.0).numpy()):
            raise ValueError(f"effective_sample_size: {HighDimStatus.NONFINITE_VALUE.value}")
        if bool((ess > float(self.sample_count) + 1e-9).numpy()):
            raise ValueError("effective_sample_size cannot exceed sample_count")
        if int(self.enhancement_attempts) < 0:
            raise ValueError("enhancement_attempts must be nonnegative")
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "effective_sample_size", ess)
        object.__setattr__(self, "enhancement_attempts", int(self.enhancement_attempts))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteSampleDiagnostics",
            "sample_count": int(self.sample_count),
            "effective_sample_size": self.effective_sample_size,
            "enhancement_attempts": int(self.enhancement_attempts),
        }


@dataclass(frozen=True)
class SourceRouteNormalizerContribution:
    """Named normalizer terms for the clean-room source route."""

    log_transport_normalizer: tf.Tensor
    shift_constant: tf.Tensor
    log_abs_det_policy: str

    def __post_init__(self) -> None:
        log_z = tf.convert_to_tensor(self.log_transport_normalizer, dtype=tf.float64)
        shift = tf.convert_to_tensor(self.shift_constant, dtype=tf.float64)
        for name, value in (
            ("log_transport_normalizer", log_z),
            ("shift_constant", shift),
        ):
            if value.shape.rank != 0:
                raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
            if not bool(tf.math.is_finite(value).numpy()):
                raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
        if str(self.log_abs_det_policy) not in ("included_in_target", "separate_term"):
            raise ValueError("log_abs_det_policy must be included_in_target or separate_term")
        object.__setattr__(self, "log_transport_normalizer", log_z)
        object.__setattr__(self, "shift_constant", shift)
        object.__setattr__(self, "log_abs_det_policy", str(self.log_abs_det_policy))

    def log_increment(self) -> tf.Tensor:
        return self.log_transport_normalizer - self.shift_constant

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteNormalizerContribution",
            "log_transport_normalizer": self.log_transport_normalizer,
            "shift_constant": self.shift_constant,
            "log_abs_det_policy": self.log_abs_det_policy,
            "log_increment": self.log_increment(),
        }


@dataclass(frozen=True)
class SourceRouteSampleBatch:
    """Weighted sample batch used by the clean-room source route."""

    samples: tf.Tensor
    log_weights: tf.Tensor
    time_index: int
    route_label: str
    sample_origin: str

    def __post_init__(self) -> None:
        samples = tf.convert_to_tensor(self.samples, dtype=tf.float64)
        log_weights = tf.convert_to_tensor(self.log_weights, dtype=tf.float64)
        if samples.shape.rank != 2:
            raise ValueError(f"samples: {HighDimStatus.INVALID_SHAPE.value}")
        if log_weights.shape != (int(samples.shape[1]),):
            raise ValueError(f"log_weights: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("samples", samples)
        assert_tf_float64("log_weights", log_weights)
        if not bool(
            tf.reduce_all(tf.math.is_finite(samples)).numpy()
            and tf.reduce_all(tf.math.is_finite(log_weights)).numpy()
        ):
            raise ValueError(f"SourceRouteSampleBatch: {HighDimStatus.NONFINITE_VALUE.value}")
        if int(self.time_index) < 0:
            raise ValueError("time_index must be nonnegative")
        if not str(self.route_label).strip():
            raise ValueError("route_label must be nonempty")
        if not str(self.sample_origin).strip():
            raise ValueError("sample_origin must be nonempty")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "log_weights", log_weights)
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "route_label", str(self.route_label))
        object.__setattr__(self, "sample_origin", str(self.sample_origin))

    @property
    def dimension(self) -> int:
        return int(self.samples.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[1])

    def normalized_log_weights(self) -> tf.Tensor:
        return normalize_log_weights(self.log_weights)

    def effective_sample_size(self) -> tf.Tensor:
        return effective_sample_size_from_log_weights(self.log_weights)

    def diagnostics(self, *, enhancement_attempts: int = 0) -> SourceRouteSampleDiagnostics:
        return SourceRouteSampleDiagnostics(
            sample_count=self.sample_count,
            effective_sample_size=self.effective_sample_size(),
            enhancement_attempts=int(enhancement_attempts),
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteSampleBatch",
            "samples": self.samples,
            "log_weights": self.log_weights,
            "time_index": int(self.time_index),
            "route_label": self.route_label,
            "sample_origin": self.sample_origin,
            "normalized_log_weights": self.normalized_log_weights(),
            "effective_sample_size": self.effective_sample_size(),
        }


@dataclass(frozen=True)
class SourceRoutePushResult:
    """Clean-room source-route push output for one filtering step."""

    propagated_batch: SourceRouteSampleBatch
    augmented_batch: SourceRouteSampleBatch
    diagnostics: SourceRouteSampleDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.propagated_batch, SourceRouteSampleBatch):
            raise TypeError("propagated_batch must be SourceRouteSampleBatch")
        if not isinstance(self.augmented_batch, SourceRouteSampleBatch):
            raise TypeError("augmented_batch must be SourceRouteSampleBatch")
        if not isinstance(self.diagnostics, SourceRouteSampleDiagnostics):
            raise TypeError("diagnostics must be SourceRouteSampleDiagnostics")
        if self.propagated_batch.sample_count != self.augmented_batch.sample_count:
            raise ValueError("propagated and augmented sample counts must match")
        if self.propagated_batch.time_index != self.augmented_batch.time_index:
            raise ValueError("propagated and augmented time indices must match")
        if self.propagated_batch.route_label != SOURCE_FAITHFUL_ROUTE_LABEL:
            raise ValueError("source push result must use source_faithful_filtering")
        if self.augmented_batch.route_label != SOURCE_FAITHFUL_ROUTE_LABEL:
            raise ValueError("source push result must use source_faithful_filtering")

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRoutePushResult",
            "propagated_batch": self.propagated_batch.manifest_payload(),
            "augmented_batch": self.augmented_batch.manifest_payload(),
            "diagnostics": self.diagnostics.manifest_payload(),
        }


@dataclass(frozen=True)
class _P59AuthorSIRSourceFitData:
    """Source-derived fixed-variant fit data for one author-SIR step."""

    time_index: int
    frame: SourceRouteCoordinateFrame
    local_fit_points: tf.Tensor
    target_values: tf.Tensor
    negative_log_values: tf.Tensor
    shift_constant: tf.Tensor
    fit_weights: tf.Tensor
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, SourceRouteCoordinateFrame):
            raise TypeError("frame must be SourceRouteCoordinateFrame")
        local = tf.convert_to_tensor(self.local_fit_points, dtype=tf.float64)
        targets = _finite_vector("target_values", self.target_values)
        negative = _finite_vector("negative_log_values", self.negative_log_values)
        weights = _finite_vector("fit_weights", self.fit_weights)
        shift = tf.convert_to_tensor(self.shift_constant, dtype=tf.float64)
        if local.shape.rank != 2 or int(local.shape[0]) != self.frame.dimension:
            raise ValueError(f"local_fit_points: {HighDimStatus.INVALID_SHAPE.value}")
        if targets.shape != (int(local.shape[1]),):
            raise ValueError(f"target_values: {HighDimStatus.INVALID_SHAPE.value}")
        if negative.shape != targets.shape or weights.shape != targets.shape:
            raise ValueError(f"fit data vectors: {HighDimStatus.INVALID_SHAPE.value}")
        if shift.shape.rank != 0 or not bool(tf.math.is_finite(shift).numpy()):
            raise ValueError(f"shift_constant: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("local_fit_points", local)
        if not bool(tf.reduce_all(tf.math.is_finite(local)).numpy()):
            raise ValueError(f"local_fit_points: {HighDimStatus.NONFINITE_VALUE.value}")
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "local_fit_points", local)
        object.__setattr__(self, "target_values", targets)
        object.__setattr__(self, "negative_log_values", negative)
        object.__setattr__(self, "shift_constant", shift)
        object.__setattr__(self, "fit_weights", weights)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))


@dataclass(frozen=True)
class SourceRouteTarget:
    """One-step source-route target in local coordinates."""

    negative_log_physical_density_fn: object
    coordinate_frame: SourceRouteCoordinateFrame
    shift_constant: tf.Tensor
    time_index: int
    target_family: str
    source_terms: tuple[str, ...]
    log_abs_det_policy: str = "included_in_target"

    def __post_init__(self) -> None:
        if not callable(self.negative_log_physical_density_fn):
            raise TypeError("negative_log_physical_density_fn must be callable")
        if not isinstance(self.coordinate_frame, SourceRouteCoordinateFrame):
            raise TypeError("coordinate_frame must be SourceRouteCoordinateFrame")
        shift = tf.convert_to_tensor(self.shift_constant, dtype=tf.float64)
        if shift.shape.rank != 0:
            raise ValueError(f"shift_constant: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.math.is_finite(shift).numpy()):
            raise ValueError(f"shift_constant: {HighDimStatus.NONFINITE_VALUE.value}")
        if int(self.time_index) < 1:
            raise ValueError("time_index must be positive for source-route targets")
        if str(self.target_family) not in ("full", "preconditioner", "residual"):
            raise ValueError("target_family must be full, preconditioner, or residual")
        terms = tuple(str(term) for term in self.source_terms)
        if not terms:
            raise ValueError("source_terms must be nonempty")
        for required in ("prior_or_previous", "transition", "likelihood"):
            if required not in terms:
                raise ValueError(f"source_terms missing {required}")
        if str(self.log_abs_det_policy) != "included_in_target":
            raise ValueError("P55 source target requires included_in_target determinant policy")
        object.__setattr__(self, "shift_constant", shift)
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "target_family", str(self.target_family))
        object.__setattr__(self, "source_terms", terms)
        object.__setattr__(self, "log_abs_det_policy", str(self.log_abs_det_policy))

    def physical_points_from_reference(self, reference_points: tf.Tensor) -> tf.Tensor:
        reference = tf.convert_to_tensor(reference_points, dtype=tf.float64)
        if reference.shape.rank != 2:
            raise ValueError(f"reference_points: {HighDimStatus.INVALID_SHAPE.value}")
        if int(reference.shape[0]) != self.coordinate_frame.dimension:
            raise ValueError(f"reference_points: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(reference)).numpy()):
            raise ValueError(f"reference_points: {HighDimStatus.NONFINITE_VALUE.value}")
        return (
            tf.linalg.matmul(self.coordinate_frame.matrix, reference)
            + self.coordinate_frame.mu[:, tf.newaxis]
        )

    def negative_log_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        physical = self.physical_points_from_reference(reference_points)
        negative_log_physical = _finite_vector(
            "negative_log_physical_density",
            self.negative_log_physical_density_fn(physical),
        )
        if negative_log_physical.shape != (int(physical.shape[1]),):
            raise ValueError(
                f"negative_log_physical_density: {HighDimStatus.INVALID_SHAPE.value}"
            )
        local_negative_log = (
            negative_log_physical
            - self.coordinate_frame.log_abs_det()
        )
        return source_route_shifted_negative_log_target(
            negative_log_target=local_negative_log,
            shift_constant=self.shift_constant,
        )

    def log_target_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        return -self.negative_log_density(reference_points)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteTarget",
            "time_index": int(self.time_index),
            "target_family": self.target_family,
            "source_terms": self.source_terms,
            "shift_constant": self.shift_constant,
            "log_abs_det_policy": self.log_abs_det_policy,
            "coordinate_frame": self.coordinate_frame.manifest_payload(),
        }


@dataclass(frozen=True)
class SourceRouteTransportProtocol:
    """Protocol wrapper for source-route transport objects."""

    transport_object: object
    route_label: str = SOURCE_FAITHFUL_ROUTE_LABEL

    def __post_init__(self) -> None:
        if str(self.route_label) != SOURCE_FAITHFUL_ROUTE_LABEL:
            raise ValueError("transport protocol must use source_faithful_filtering")
        required_methods = (
            "manifest_payload",
            "inverse_transport",
            "forward_transport",
            "conditional_inverse_transport",
            "eval_pdf",
            "potential",
            "proposal_log_density",
            "marginalize",
            "log_normalizer",
        )
        missing = tuple(
            method for method in required_methods
            if not callable(getattr(self.transport_object, method, None))
        )
        if missing:
            raise TypeError(
                "transport_object missing required methods: " + ", ".join(missing)
            )
        payload = self.transport_object.manifest_payload()
        if not isinstance(payload, Mapping):
            raise TypeError("transport_object manifest_payload() must return a mapping")
        contract_level = str(payload.get("source_contract_level", ""))
        if contract_level not in ("contract_test_double", "fixed_ttsirt"):
            raise ValueError(
                "transport_object manifest_payload() must declare "
                "source_contract_level as contract_test_double or fixed_ttsirt"
            )
        if contract_level == "fixed_ttsirt":
            for key in ("tt_cores_declared", "defensive_density_declared"):
                if payload.get(key) is not True:
                    raise ValueError(f"fixed_ttsirt transport must declare {key}")
        log_z = tf.convert_to_tensor(self.transport_object.log_normalizer(), dtype=tf.float64)
        if log_z.shape.rank != 0 or not bool(tf.math.is_finite(log_z).numpy()):
            raise ValueError(f"log_normalizer: {HighDimStatus.NONFINITE_VALUE.value}")
        object.__setattr__(self, "route_label", str(self.route_label))

    def inverse_transport(self, reference_points: tf.Tensor) -> tf.Tensor:
        result = tf.convert_to_tensor(
            self.transport_object.inverse_transport(reference_points),
            dtype=tf.float64,
        )
        if result.shape != tf.convert_to_tensor(reference_points, dtype=tf.float64).shape:
            raise ValueError(f"inverse_transport: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(result)).numpy()):
            raise ValueError(f"inverse_transport: {HighDimStatus.NONFINITE_VALUE.value}")
        return result

    def log_reference_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        if not callable(getattr(self.transport_object, "log_reference_density", None)):
            raise TypeError(
                "log_reference_density is diagnostic-only and unavailable for this source transport"
            )
        values = _finite_vector(
            "log_reference_density",
            self.transport_object.log_reference_density(reference_points),
        )
        reference = tf.convert_to_tensor(reference_points, dtype=tf.float64)
        if reference.shape.rank != 2 or values.shape != (int(reference.shape[1]),):
            raise ValueError(f"log_reference_density: {HighDimStatus.INVALID_SHAPE.value}")
        return values

    def forward_transport(self, local_points: tf.Tensor) -> tf.Tensor:
        result = tf.convert_to_tensor(
            self.transport_object.forward_transport(local_points),
            dtype=tf.float64,
        )
        if result.shape != tf.convert_to_tensor(local_points, dtype=tf.float64).shape:
            raise ValueError(f"forward_transport: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(result)).numpy()):
            raise ValueError(f"forward_transport: {HighDimStatus.NONFINITE_VALUE.value}")
        return result

    def conditional_inverse_transport(
        self,
        conditioning_points: tf.Tensor,
        reference_points: tf.Tensor,
    ) -> tf.Tensor:
        result = tf.convert_to_tensor(
            self.transport_object.conditional_inverse_transport(
                conditioning_points,
                reference_points,
            ),
            dtype=tf.float64,
        )
        reference = tf.convert_to_tensor(reference_points, dtype=tf.float64)
        if result.shape != reference.shape:
            raise ValueError(
                f"conditional_inverse_transport: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if not bool(tf.reduce_all(tf.math.is_finite(result)).numpy()):
            raise ValueError(
                f"conditional_inverse_transport: {HighDimStatus.NONFINITE_VALUE.value}"
            )
        return result

    def eval_pdf(self, local_points: tf.Tensor) -> tf.Tensor:
        values = _finite_vector("eval_pdf", self.transport_object.eval_pdf(local_points))
        local = tf.convert_to_tensor(local_points, dtype=tf.float64)
        if local.shape.rank != 2 or values.shape != (int(local.shape[1]),):
            raise ValueError(f"eval_pdf: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(values > 0.0).numpy()):
            raise ValueError(f"eval_pdf: {HighDimStatus.NONFINITE_VALUE.value}")
        return values

    def potential(self, local_points: tf.Tensor) -> tf.Tensor:
        values = _finite_vector("potential", self.transport_object.potential(local_points))
        local = tf.convert_to_tensor(local_points, dtype=tf.float64)
        if local.shape.rank != 2 or values.shape != (int(local.shape[1]),):
            raise ValueError(f"potential: {HighDimStatus.INVALID_SHAPE.value}")
        return values

    def proposal_log_density(
        self,
        *,
        local_points: tf.Tensor,
        reference_points: tf.Tensor,
    ) -> tf.Tensor:
        values = _finite_vector(
            "proposal_log_density",
            self.transport_object.proposal_log_density(
                local_points=local_points,
                reference_points=reference_points,
            ),
        )
        local = tf.convert_to_tensor(local_points, dtype=tf.float64)
        reference = tf.convert_to_tensor(reference_points, dtype=tf.float64)
        if (
            local.shape.rank != 2
            or reference.shape.rank != 2
            or local.shape != reference.shape
            or values.shape != (int(local.shape[1]),)
        ):
            raise ValueError(f"proposal_log_density: {HighDimStatus.INVALID_SHAPE.value}")
        return values

    def marginalize(self, keep_axes: tuple[int, ...]) -> object:
        result = self.transport_object.marginalize(tuple(int(axis) for axis in keep_axes))
        if result is None:
            raise ValueError(f"marginalize: {HighDimStatus.INVALID_SHAPE.value}")
        return result

    def log_normalizer(self) -> tf.Tensor:
        value = tf.convert_to_tensor(self.transport_object.log_normalizer(), dtype=tf.float64)
        if value.shape.rank != 0 or not bool(tf.math.is_finite(value).numpy()):
            raise ValueError(f"log_normalizer: {HighDimStatus.NONFINITE_VALUE.value}")
        return value

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteTransportProtocol",
            "route_label": self.route_label,
            "source_contract_level": self.transport_object.manifest_payload().get(
                "source_contract_level"
            ),
            "required_source_methods": (
                "inverse_transport",
                "forward_transport",
                "conditional_inverse_transport",
                "eval_pdf",
                "potential",
                "proposal_log_density",
                "marginalize",
                "log_normalizer",
            ),
            "proposal_density_semantics": "transport_eval_pdf_equivalent_on_local_samples",
            "transport_object": self.transport_object.manifest_payload(),
            "log_normalizer": self.log_normalizer(),
        }


@dataclass(frozen=True)
class SourceRouteRetainedSampleResult:
    """Retained sample generation plus source proposal-correction output."""

    retained_batch: SourceRouteSampleBatch
    proposal_log_density: tf.Tensor
    target_log_density: tf.Tensor
    correction_log_weights: tf.Tensor
    diagnostics: SourceRouteSampleDiagnostics
    normalizer: SourceRouteNormalizerContribution

    def __post_init__(self) -> None:
        if not isinstance(self.retained_batch, SourceRouteSampleBatch):
            raise TypeError("retained_batch must be SourceRouteSampleBatch")
        proposal, target = _finite_same_shape_vectors(
            "proposal_log_density",
            self.proposal_log_density,
            "target_log_density",
            self.target_log_density,
        )
        correction = _finite_vector("correction_log_weights", self.correction_log_weights)
        if correction.shape != proposal.shape:
            raise ValueError(
                f"correction_log_weights: {HighDimStatus.INVALID_SHAPE.value}"
            )
        tf.debugging.assert_near(correction, target - proposal, atol=1e-10)
        if not isinstance(self.diagnostics, SourceRouteSampleDiagnostics):
            raise TypeError("diagnostics must be SourceRouteSampleDiagnostics")
        if not isinstance(self.normalizer, SourceRouteNormalizerContribution):
            raise TypeError("normalizer must be SourceRouteNormalizerContribution")
        if self.retained_batch.sample_count != self.diagnostics.sample_count:
            raise ValueError("diagnostics sample_count must match retained batch")
        object.__setattr__(self, "proposal_log_density", proposal)
        object.__setattr__(self, "target_log_density", target)
        object.__setattr__(self, "correction_log_weights", correction)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteRetainedSampleResult",
            "retained_batch": self.retained_batch.manifest_payload(),
            "proposal_log_density": self.proposal_log_density,
            "target_log_density": self.target_log_density,
            "correction_log_weights": self.correction_log_weights,
            "diagnostics": self.diagnostics.manifest_payload(),
            "normalizer": self.normalizer.manifest_payload(),
        }


@dataclass(frozen=True)
class SourceRouteOneStepResult:
    """P55 one-step source reapproximation boundary result."""

    target: SourceRouteTarget
    transport: SourceRouteTransportProtocol
    retained_samples: SourceRouteRetainedSampleResult
    time_index: int
    sequential_status: str = "one_step_t1_only"

    def __post_init__(self) -> None:
        if not isinstance(self.target, SourceRouteTarget):
            raise TypeError("target must be SourceRouteTarget")
        if not isinstance(self.transport, SourceRouteTransportProtocol):
            raise TypeError("transport must be SourceRouteTransportProtocol")
        if not isinstance(self.retained_samples, SourceRouteRetainedSampleResult):
            raise TypeError("retained_samples must be SourceRouteRetainedSampleResult")
        if int(self.time_index) != 1 or self.target.time_index != 1:
            raise ValueError(
                "P55 one-step source reapproximation only supports t=1; "
                "previous retained-object marginalization is not implemented"
            )
        if str(self.sequential_status) != "one_step_t1_only":
            raise ValueError("sequential_status must be one_step_t1_only")
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "sequential_status", str(self.sequential_status))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteOneStepResult",
            "time_index": int(self.time_index),
            "sequential_status": self.sequential_status,
            "target": self.target.manifest_payload(),
            "transport": self.transport.manifest_payload(),
            "retained_samples": self.retained_samples.manifest_payload(),
        }


@dataclass(frozen=True)
class SourceRouteOperationRecord:
    """Audit row for one source-route operation."""

    operation_id: str
    source_anchor: str
    implementation_status: str
    route_label: str = SOURCE_FAITHFUL_ROUTE_LABEL
    notes: str = ""

    def __post_init__(self) -> None:
        operation = str(self.operation_id)
        if operation not in SOURCE_ROUTE_REQUIRED_OPERATION_IDS:
            raise ValueError("unknown source-route operation_id")
        if not str(self.source_anchor).strip():
            raise ValueError("source_anchor must be nonempty")
        status = str(self.implementation_status)
        if status not in {"implemented", "partial", "blocked"}:
            raise ValueError("implementation_status must be implemented, partial, or blocked")
        if not str(self.route_label).strip():
            raise ValueError("route_label must be nonempty")
        object.__setattr__(self, "operation_id", operation)
        object.__setattr__(self, "source_anchor", str(self.source_anchor))
        object.__setattr__(self, "implementation_status", status)
        object.__setattr__(self, "route_label", str(self.route_label))
        object.__setattr__(self, "notes", str(self.notes))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "operation_id": self.operation_id,
            "source_anchor": self.source_anchor,
            "implementation_status": self.implementation_status,
            "route_label": self.route_label,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SourceRouteImplementationAudit:
    """Drift-aware audit of source-route operation coverage."""

    records: tuple[SourceRouteOperationRecord, ...]
    drift_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise ValueError("records must be nonempty")
        if not all(isinstance(row, SourceRouteOperationRecord) for row in records):
            raise TypeError("records must contain SourceRouteOperationRecord")
        markers = tuple(str(marker) for marker in self.drift_markers)
        for marker in markers:
            if marker not in SOURCE_ROUTE_FORBIDDEN_DRIFT_MARKERS:
                raise ValueError("unknown source-route drift marker")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "drift_markers", markers)

    @property
    def missing_required_operations(self) -> tuple[str, ...]:
        present = {row.operation_id for row in self.records}
        return tuple(
            operation
            for operation in SOURCE_ROUTE_REQUIRED_OPERATION_IDS
            if operation not in present
        )

    @property
    def incomplete_required_operations(self) -> tuple[str, ...]:
        return tuple(
            row.operation_id
            for row in self.records
            if row.implementation_status != "implemented"
        )

    @property
    def status(self) -> str:
        if self.drift_markers:
            return "BLOCK_SOURCE_ROUTE_DRIFT"
        if self.missing_required_operations or self.incomplete_required_operations:
            return "BLOCK_SOURCE_ROUTE_INCOMPLETE"
        return "PASS_SOURCE_ROUTE_OPERATION_COVERAGE"

    def assert_no_drift(self) -> None:
        if self.drift_markers:
            raise ValueError("source-route drift markers present")

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteImplementationAudit",
            "required_operation_ids": SOURCE_ROUTE_REQUIRED_OPERATION_IDS,
            "records": [row.manifest_payload() for row in self.records],
            "missing_required_operations": self.missing_required_operations,
            "incomplete_required_operations": self.incomplete_required_operations,
            "drift_markers": self.drift_markers,
            "status": self.status,
        }


@dataclass(frozen=True)
class SourceRouteSequentialDensityComponents:
    """Source density components for one sequential fixed-HMC step."""

    parameter_dim: int
    state_dim: int
    transition_log_density_fn: object
    likelihood_log_density_fn: object
    prior_log_density_fn: object | None = None
    source_anchor: str = "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:76-130"

    def __post_init__(self) -> None:
        d = int(self.parameter_dim)
        m = int(self.state_dim)
        if d < 0 or m <= 0:
            raise ValueError("parameter_dim must be nonnegative and state_dim positive")
        if not callable(self.transition_log_density_fn):
            raise TypeError("transition_log_density_fn must be callable")
        if not callable(self.likelihood_log_density_fn):
            raise TypeError("likelihood_log_density_fn must be callable")
        if self.prior_log_density_fn is not None and not callable(self.prior_log_density_fn):
            raise TypeError("prior_log_density_fn must be callable when provided")
        if not str(self.source_anchor).strip():
            raise ValueError("source_anchor must be nonempty")
        object.__setattr__(self, "parameter_dim", d)
        object.__setattr__(self, "state_dim", m)
        object.__setattr__(self, "source_anchor", str(self.source_anchor))

    def negative_log_physical_density(
        self,
        *,
        physical_points: tf.Tensor,
        time_index: int,
        previous_retained_object: SourceRouteRetainedObject | None,
    ) -> tf.Tensor:
        return source_route_sequential_negative_log_physical_density(
            physical_points=physical_points,
            time_index=int(time_index),
            parameter_dim=self.parameter_dim,
            state_dim=self.state_dim,
            transition_log_density_fn=self.transition_log_density_fn,
            likelihood_log_density_fn=self.likelihood_log_density_fn,
            prior_log_density_fn=self.prior_log_density_fn,
            previous_retained_object=previous_retained_object,
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteSequentialDensityComponents",
            "parameter_dim": int(self.parameter_dim),
            "state_dim": int(self.state_dim),
            "has_prior_log_density_fn": self.prior_log_density_fn is not None,
            "source_anchor": self.source_anchor,
            "source_terms": ("prior_or_previous", "transition", "likelihood"),
        }


@dataclass(frozen=True)
class SourceRouteValueBridgeBinding:
    """Fail-closed identity binding for the P90 same-target value bridge."""

    target_id: str
    time_index: int
    parameter_dim: int
    state_dim: int
    physical_shape: tuple[int, int]
    physical_ordering: tuple[str, ...]
    previous_retained_hash: str | None
    previous_marginal_keep_axes: tuple[int, ...] | None
    previous_marginal_input_axes: tuple[int, ...] | None
    basis_family: str
    basis_order: int
    basis_elements: int
    tt_rank_tuple: tuple[int, ...]
    sample_count: int
    seed: int
    transport_branch_hash: str
    coordinate_frame_hash: str
    transition_log_density_id: str
    likelihood_log_density_id: str
    prior_log_density_id: str | None
    comparator_label: str = P90_VALUE_BRIDGE_COMPARATOR_LABEL
    tolerance_version: str = P90_VALUE_BRIDGE_TOLERANCE_VERSION

    def __post_init__(self) -> None:
        d = int(self.parameter_dim)
        m = int(self.state_dim)
        if str(self.target_id) != P90_VALUE_BRIDGE_TARGET_ID:
            raise ValueError("P90 value bridge target_id mismatch")
        if int(self.time_index) < 1:
            raise ValueError("time_index must be positive")
        if d < 0 or m <= 0:
            raise ValueError("parameter_dim must be nonnegative and state_dim positive")
        shape = tuple(int(item) for item in self.physical_shape)
        if len(shape) != 2 or shape[0] != d + 2 * m or shape[1] <= 0:
            raise ValueError(f"physical_shape: {HighDimStatus.INVALID_SHAPE.value}")
        ordering = tuple(str(item) for item in self.physical_ordering)
        if ordering != P90_VALUE_BRIDGE_PHYSICAL_ORDERING:
            raise ValueError("P90 value bridge physical ordering mismatch")
        keep = (
            None
            if self.previous_marginal_keep_axes is None
            else tuple(int(axis) for axis in self.previous_marginal_keep_axes)
        )
        input_axes = (
            None
            if self.previous_marginal_input_axes is None
            else tuple(int(axis) for axis in self.previous_marginal_input_axes)
        )
        previous_hash = (
            None
            if self.previous_retained_hash is None
            else str(self.previous_retained_hash)
        )
        if int(self.time_index) == 1:
            if previous_hash is not None or keep is not None or input_axes is not None:
                raise ValueError("t=1 bridge binding cannot include previous retained state")
            if self.prior_log_density_id is None:
                raise ValueError("t=1 bridge binding requires prior_log_density_id")
        else:
            if previous_hash is None:
                raise ValueError("t>1 bridge binding requires previous_retained_hash")
            expected_keep = tuple(range(d + m))
            expected_input_axes = tuple(range(d)) + tuple(range(d + m, d + 2 * m))
            if keep != expected_keep:
                raise ValueError("P90 value bridge previous keep axes mismatch")
            if input_axes != expected_input_axes:
                raise ValueError("P90 value bridge previous input axes mismatch")
            if self.prior_log_density_id is not None:
                raise ValueError("t>1 bridge binding must not include prior_log_density_id")
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be positive")
        if not str(self.basis_family).strip():
            raise ValueError("basis_family must be nonempty")
        if int(self.basis_order) < 0:
            raise ValueError("basis_order must be nonnegative")
        if int(self.basis_elements) < 0:
            raise ValueError("basis_elements must be nonnegative")
        ranks = tuple(int(rank) for rank in self.tt_rank_tuple)
        if not ranks or any(rank <= 0 for rank in ranks):
            raise ValueError("tt_rank_tuple must contain positive ranks")
        for field_name, value in (
            ("transport_branch_hash", self.transport_branch_hash),
            ("coordinate_frame_hash", self.coordinate_frame_hash),
            ("transition_log_density_id", self.transition_log_density_id),
            ("likelihood_log_density_id", self.likelihood_log_density_id),
            ("comparator_label", self.comparator_label),
            ("tolerance_version", self.tolerance_version),
        ):
            if not str(value).strip():
                raise ValueError(f"{field_name} must be nonempty")
        if str(self.comparator_label) != P90_VALUE_BRIDGE_COMPARATOR_LABEL:
            raise ValueError("P90 value bridge comparator label mismatch")
        if str(self.tolerance_version) != P90_VALUE_BRIDGE_TOLERANCE_VERSION:
            raise ValueError("P90 value bridge tolerance version mismatch")
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "parameter_dim", d)
        object.__setattr__(self, "state_dim", m)
        object.__setattr__(self, "physical_shape", shape)
        object.__setattr__(self, "physical_ordering", ordering)
        object.__setattr__(self, "previous_retained_hash", previous_hash)
        object.__setattr__(self, "previous_marginal_keep_axes", keep)
        object.__setattr__(self, "previous_marginal_input_axes", input_axes)
        object.__setattr__(self, "basis_family", str(self.basis_family))
        object.__setattr__(self, "basis_order", int(self.basis_order))
        object.__setattr__(self, "basis_elements", int(self.basis_elements))
        object.__setattr__(self, "tt_rank_tuple", ranks)
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "transport_branch_hash", str(self.transport_branch_hash))
        object.__setattr__(self, "coordinate_frame_hash", str(self.coordinate_frame_hash))
        object.__setattr__(
            self,
            "transition_log_density_id",
            str(self.transition_log_density_id),
        )
        object.__setattr__(
            self,
            "likelihood_log_density_id",
            str(self.likelihood_log_density_id),
        )
        object.__setattr__(
            self,
            "prior_log_density_id",
            None if self.prior_log_density_id is None else str(self.prior_log_density_id),
        )
        object.__setattr__(self, "comparator_label", str(self.comparator_label))
        object.__setattr__(self, "tolerance_version", str(self.tolerance_version))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteValueBridgeBinding",
            "target_id": self.target_id,
            "time_index": int(self.time_index),
            "parameter_dim": int(self.parameter_dim),
            "state_dim": int(self.state_dim),
            "physical_shape": self.physical_shape,
            "physical_ordering": self.physical_ordering,
            "previous_retained_hash": self.previous_retained_hash,
            "previous_marginal_keep_axes": self.previous_marginal_keep_axes,
            "previous_marginal_input_axes": self.previous_marginal_input_axes,
            "basis_family": self.basis_family,
            "basis_order": int(self.basis_order),
            "basis_elements": int(self.basis_elements),
            "tt_rank_tuple": self.tt_rank_tuple,
            "sample_count": int(self.sample_count),
            "seed": int(self.seed),
            "transport_branch_hash": self.transport_branch_hash,
            "coordinate_frame_hash": self.coordinate_frame_hash,
            "transition_log_density_id": self.transition_log_density_id,
            "likelihood_log_density_id": self.likelihood_log_density_id,
            "prior_log_density_id": self.prior_log_density_id,
            "comparator_label": self.comparator_label,
            "tolerance_version": self.tolerance_version,
        }

    @property
    def binding_hash(self) -> str:
        return BranchManifest(
            version="p90.value_bridge.binding.v1",
            payload=self.manifest_payload(),
        ).sha256().value


@dataclass(frozen=True)
class SourceRouteAuthorFormulaReplayResult:
    """Independent author-formula replay of one source-route value."""

    binding: SourceRouteValueBridgeBinding
    physical_points: tf.Tensor
    prior_or_previous_log_density: tf.Tensor
    transition_log_density: tf.Tensor
    likelihood_log_density: tf.Tensor
    negative_log_density: tf.Tensor
    previous_marginal_local_points: tf.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, SourceRouteValueBridgeBinding):
            raise TypeError("binding must be SourceRouteValueBridgeBinding")
        points = tf.convert_to_tensor(self.physical_points, dtype=tf.float64)
        if points.shape.rank != 2 or tuple(int(item) for item in points.shape) != self.binding.physical_shape:
            raise ValueError(f"physical_points: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("physical_points", points)
        if not bool(tf.reduce_all(tf.math.is_finite(points)).numpy()):
            raise ValueError(f"physical_points: {HighDimStatus.NONFINITE_VALUE.value}")
        prior, transition = _finite_same_shape_vectors(
            "prior_or_previous_log_density",
            self.prior_or_previous_log_density,
            "transition_log_density",
            self.transition_log_density,
        )
        _, likelihood = _finite_same_shape_vectors(
            "transition_log_density",
            transition,
            "likelihood_log_density",
            self.likelihood_log_density,
        )
        expected = -prior - transition - likelihood
        negative = _finite_vector("negative_log_density", self.negative_log_density)
        if negative.shape != expected.shape:
            raise ValueError(f"negative_log_density: {HighDimStatus.INVALID_SHAPE.value}")
        tf.debugging.assert_near(negative, expected, atol=1e-10)
        local = (
            None
            if self.previous_marginal_local_points is None
            else tf.convert_to_tensor(self.previous_marginal_local_points, dtype=tf.float64)
        )
        if self.binding.time_index == 1:
            if local is not None:
                raise ValueError("t=1 bridge replay cannot carry previous marginal local points")
        else:
            if local is None:
                raise ValueError("t>1 bridge replay requires previous marginal local points")
            keep = self.binding.previous_marginal_keep_axes
            if (
                keep is None
                or local.shape.rank != 2
                or int(local.shape[0]) != len(keep)
                or int(local.shape[1]) != int(points.shape[1])
            ):
                raise ValueError(
                    f"previous_marginal_local_points: {HighDimStatus.INVALID_SHAPE.value}"
                )
            if not bool(tf.reduce_all(tf.math.is_finite(local)).numpy()):
                raise ValueError(
                    f"previous_marginal_local_points: {HighDimStatus.NONFINITE_VALUE.value}"
                )
        object.__setattr__(self, "physical_points", points)
        object.__setattr__(self, "prior_or_previous_log_density", prior)
        object.__setattr__(self, "transition_log_density", transition)
        object.__setattr__(self, "likelihood_log_density", likelihood)
        object.__setattr__(self, "negative_log_density", negative)
        object.__setattr__(self, "previous_marginal_local_points", local)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteAuthorFormulaReplayResult",
            "binding": self.binding.manifest_payload(),
            "binding_hash": self.binding.binding_hash,
            "prior_or_previous_log_density": self.prior_or_previous_log_density,
            "transition_log_density": self.transition_log_density,
            "likelihood_log_density": self.likelihood_log_density,
            "negative_log_density": self.negative_log_density,
            "previous_marginal_local_points": self.previous_marginal_local_points,
        }


@dataclass(frozen=True)
class SourceRouteDerivativeBinding:
    """Same-branch derivative binding for P90 deterministic carry tests."""

    value_bridge_binding: SourceRouteValueBridgeBinding
    value_bridge_binding_hash: str
    parameter_indices: tuple[int, ...]
    derivative_tolerance_version: str = "p90.derivative_carry.tolerances.v1"
    fixed_ttsirt_transport_derivative_status: str = (
        "BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.value_bridge_binding, SourceRouteValueBridgeBinding):
            raise TypeError("value_bridge_binding must be SourceRouteValueBridgeBinding")
        if str(self.value_bridge_binding_hash) != self.value_bridge_binding.binding_hash:
            raise ValueError("P90 derivative binding hash mismatch")
        indices = tuple(int(index) for index in self.parameter_indices)
        if not indices:
            raise ValueError("parameter_indices must be nonempty")
        if any(index < 0 or index >= self.value_bridge_binding.parameter_dim for index in indices):
            raise ValueError("parameter_indices out of range")
        if str(self.derivative_tolerance_version) != "p90.derivative_carry.tolerances.v1":
            raise ValueError("P90 derivative tolerance version mismatch")
        if not str(self.fixed_ttsirt_transport_derivative_status).startswith("BLOCK_"):
            raise ValueError("fixed TTSIRT transport derivative status must remain a blocker")
        object.__setattr__(self, "value_bridge_binding_hash", str(self.value_bridge_binding_hash))
        object.__setattr__(self, "parameter_indices", indices)
        object.__setattr__(
            self,
            "derivative_tolerance_version",
            str(self.derivative_tolerance_version),
        )
        object.__setattr__(
            self,
            "fixed_ttsirt_transport_derivative_status",
            str(self.fixed_ttsirt_transport_derivative_status),
        )

    @property
    def target_id(self) -> str:
        return self.value_bridge_binding.target_id

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteDerivativeBinding",
            "value_bridge_binding_hash": self.value_bridge_binding_hash,
            "target_id": self.target_id,
            "parameter_indices": self.parameter_indices,
            "derivative_tolerance_version": self.derivative_tolerance_version,
            "fixed_ttsirt_transport_derivative_status": (
                self.fixed_ttsirt_transport_derivative_status
            ),
            "value_bridge_binding": self.value_bridge_binding.manifest_payload(),
        }


@dataclass(frozen=True)
class SourceRouteComponentDerivativeCarry:
    """One component score in the P90 deterministic derivative carry."""

    binding: SourceRouteDerivativeBinding
    component_name: str
    component_value: tf.Tensor
    parameter_score: tf.Tensor
    callable_identity: str
    classification: str
    owner_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding, SourceRouteDerivativeBinding):
            raise TypeError("binding must be SourceRouteDerivativeBinding")
        name = str(self.component_name)
        if name not in (
            "prior",
            "previous_marginal",
            "transition",
            "likelihood",
            "negative_log_assembly",
            "proposal_correction",
            "normalizer",
            "transport",
        ):
            raise ValueError("unsupported derivative component_name")
        value = _finite_vector("component_value", self.component_value)
        score = tf.convert_to_tensor(self.parameter_score, dtype=tf.float64)
        if score.shape.rank != 2:
            raise ValueError(f"parameter_score: {HighDimStatus.INVALID_SHAPE.value}")
        if int(score.shape[0]) != int(value.shape[0]):
            raise ValueError(f"parameter_score: {HighDimStatus.INVALID_SHAPE.value}")
        if int(score.shape[1]) != len(self.binding.parameter_indices):
            raise ValueError(f"parameter_score: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("parameter_score", score)
        if not bool(tf.reduce_all(tf.math.is_finite(score)).numpy()):
            raise ValueError(f"parameter_score: {HighDimStatus.NONFINITE_VALUE.value}")
        if not str(self.callable_identity).strip():
            raise ValueError("callable_identity must be nonempty")
        if str(self.classification) not in (
            "local_parameterized_sir_component",
            "source_formula_assembly",
            "fixed_hmc_adaptation_blocker",
            "source_backed_transport_blocker",
        ):
            raise ValueError("unsupported derivative classification")
        status = str(self.owner_status)
        if not (status.startswith("READY_") or status.startswith("BLOCK_")):
            raise ValueError("owner_status must be READY_* or BLOCK_*")
        object.__setattr__(self, "component_name", name)
        object.__setattr__(self, "component_value", value)
        object.__setattr__(self, "parameter_score", score)
        object.__setattr__(self, "callable_identity", str(self.callable_identity))
        object.__setattr__(self, "classification", str(self.classification))
        object.__setattr__(self, "owner_status", status)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteComponentDerivativeCarry",
            "component_name": self.component_name,
            "component_value": self.component_value,
            "parameter_score": self.parameter_score,
            "callable_identity": self.callable_identity,
            "classification": self.classification,
            "owner_status": self.owner_status,
            "binding_hash": self.binding.value_bridge_binding_hash,
        }


@dataclass(frozen=True)
class SourceRoutePreviousMarginalDerivativeCarry:
    """Carry previous-marginal affine state plus explicit derivative blocker."""

    binding: SourceRouteDerivativeBinding
    previous_retained_hash: str
    keep_axes: tuple[int, ...]
    input_axes: tuple[int, ...]
    local_prefix_points: tf.Tensor
    log_density: tf.Tensor
    affine_log_det: tf.Tensor
    derivative_owner_status: str = (
        "BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.binding, SourceRouteDerivativeBinding):
            raise TypeError("binding must be SourceRouteDerivativeBinding")
        value_binding = self.binding.value_bridge_binding
        if str(self.previous_retained_hash) != value_binding.previous_retained_hash:
            raise ValueError("P90 previous marginal derivative retained hash mismatch")
        keep = tuple(int(axis) for axis in self.keep_axes)
        input_axes = tuple(int(axis) for axis in self.input_axes)
        if keep != value_binding.previous_marginal_keep_axes:
            raise ValueError("P90 previous marginal derivative keep axes mismatch")
        if input_axes != value_binding.previous_marginal_input_axes:
            raise ValueError("P90 previous marginal derivative input axes mismatch")
        local = tf.convert_to_tensor(self.local_prefix_points, dtype=tf.float64)
        if local.shape.rank != 2 or int(local.shape[0]) != len(keep):
            raise ValueError(f"local_prefix_points: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("local_prefix_points", local)
        if not bool(tf.reduce_all(tf.math.is_finite(local)).numpy()):
            raise ValueError(f"local_prefix_points: {HighDimStatus.NONFINITE_VALUE.value}")
        log_density = _finite_vector("log_density", self.log_density)
        if int(log_density.shape[0]) != int(local.shape[1]):
            raise ValueError(f"log_density: {HighDimStatus.INVALID_SHAPE.value}")
        log_det = tf.convert_to_tensor(self.affine_log_det, dtype=tf.float64)
        if log_det.shape.rank != 0 or not bool(tf.math.is_finite(log_det).numpy()):
            raise ValueError(f"affine_log_det: {HighDimStatus.NONFINITE_VALUE.value}")
        status = str(self.derivative_owner_status)
        if not status.startswith("BLOCK_"):
            raise ValueError("previous marginal derivative owner must remain a blocker")
        object.__setattr__(self, "previous_retained_hash", str(self.previous_retained_hash))
        object.__setattr__(self, "keep_axes", keep)
        object.__setattr__(self, "input_axes", input_axes)
        object.__setattr__(self, "local_prefix_points", local)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "affine_log_det", log_det)
        object.__setattr__(self, "derivative_owner_status", status)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRoutePreviousMarginalDerivativeCarry",
            "binding_hash": self.binding.value_bridge_binding_hash,
            "previous_retained_hash": self.previous_retained_hash,
            "keep_axes": self.keep_axes,
            "input_axes": self.input_axes,
            "local_prefix_points": self.local_prefix_points,
            "log_density": self.log_density,
            "affine_log_det": self.affine_log_det,
            "derivative_owner_status": self.derivative_owner_status,
        }


def source_route_negative_log_assembly_derivative_carry(
    *,
    binding: SourceRouteDerivativeBinding,
    prior_or_previous: SourceRouteComponentDerivativeCarry,
    transition: SourceRouteComponentDerivativeCarry,
    likelihood: SourceRouteComponentDerivativeCarry,
) -> SourceRouteComponentDerivativeCarry:
    """Assemble the negative-log score carry from component log-density scores."""

    components = (prior_or_previous, transition, likelihood)
    if any(component.binding != binding for component in components):
        raise ValueError("P90 derivative assembly binding hash mismatch")
    if prior_or_previous.component_name not in ("prior", "previous_marginal"):
        raise ValueError("prior_or_previous component must be prior or previous_marginal")
    if transition.component_name != "transition":
        raise ValueError("transition component mismatch")
    if likelihood.component_name != "likelihood":
        raise ValueError("likelihood component mismatch")
    score = -(
        prior_or_previous.parameter_score
        + transition.parameter_score
        + likelihood.parameter_score
    )
    value = -(
        prior_or_previous.component_value
        + transition.component_value
        + likelihood.component_value
    )
    return SourceRouteComponentDerivativeCarry(
        binding=binding,
        component_name="negative_log_assembly",
        component_value=value,
        parameter_score=score,
        callable_identity="source_route_negative_log_assembly",
        classification="source_formula_assembly",
        owner_status="READY_P90_DETERMINISTIC_ASSEMBLY_DERIVATIVE_CARRY",
    )


@dataclass(frozen=True)
class SourceRouteSequentialStepSpec:
    """Frozen inputs for one fixed-HMC replay step of the source route."""

    target: SourceRouteTarget
    transport: SourceRouteTransportProtocol
    reference_samples: tf.Tensor
    measure_convention: MeasureConvention
    density_components: SourceRouteSequentialDensityComponents
    previous_marginal_keep_axes: tuple[int, ...] | None = None
    previous_marginal_input_axes: tuple[int, ...] | None = None
    storage_kind: str = "source_transport_object"
    transition_interface: str = "sample_propagation"

    def __post_init__(self) -> None:
        if not isinstance(self.target, SourceRouteTarget):
            raise TypeError("target must be SourceRouteTarget")
        if not isinstance(self.transport, SourceRouteTransportProtocol):
            raise TypeError("transport must be SourceRouteTransportProtocol")
        assert_density_matches_mass(self.measure_convention)
        if not isinstance(self.density_components, SourceRouteSequentialDensityComponents):
            raise TypeError("density_components must be SourceRouteSequentialDensityComponents")
        reference = tf.convert_to_tensor(self.reference_samples, dtype=tf.float64)
        if reference.shape.rank != 2:
            raise ValueError(f"reference_samples: {HighDimStatus.INVALID_SHAPE.value}")
        if int(reference.shape[0]) != self.target.coordinate_frame.dimension:
            raise ValueError(f"reference_samples: {HighDimStatus.INVALID_SHAPE.value}")
        if int(reference.shape[1]) <= 0:
            raise ValueError("reference_samples must contain at least one column")
        assert_tf_float64("reference_samples", reference)
        if not bool(tf.reduce_all(tf.math.is_finite(reference)).numpy()):
            raise ValueError(f"reference_samples: {HighDimStatus.NONFINITE_VALUE.value}")
        keep_axes = (
            None
            if self.previous_marginal_keep_axes is None
            else tuple(int(axis) for axis in self.previous_marginal_keep_axes)
        )
        input_axes = (
            None
            if self.previous_marginal_input_axes is None
            else tuple(int(axis) for axis in self.previous_marginal_input_axes)
        )
        if self.target.time_index == 1:
            if keep_axes is not None:
                raise ValueError("t=1 must not declare previous_marginal_keep_axes")
            if input_axes is not None:
                raise ValueError("t=1 must not declare previous_marginal_input_axes")
        else:
            if keep_axes is None:
                raise ValueError("t>1 requires previous_marginal_keep_axes")
            if input_axes is None:
                raise ValueError("t>1 requires previous_marginal_input_axes")
            if not keep_axes:
                raise ValueError("previous_marginal_keep_axes must be nonempty")
            if tuple(sorted(keep_axes)) != keep_axes or len(set(keep_axes)) != len(keep_axes):
                raise ValueError("previous_marginal_keep_axes must be unique and increasing")
            if any(axis < 0 for axis in keep_axes):
                raise ValueError("previous_marginal_keep_axes must be nonnegative")
            if keep_axes != tuple(range(len(keep_axes))):
                raise ValueError(
                    "source previous marginalization requires prefix keep axes"
                )
            if len(input_axes) != len(keep_axes):
                raise ValueError(
                    "previous_marginal_input_axes must match keep_axes length"
                )
            if tuple(sorted(input_axes)) != input_axes or len(set(input_axes)) != len(input_axes):
                raise ValueError("previous_marginal_input_axes must be unique and increasing")
            if any(axis < 0 for axis in input_axes):
                raise ValueError("previous_marginal_input_axes must be nonnegative")
            if input_axes[-1] >= self.target.coordinate_frame.dimension:
                raise ValueError(
                    f"previous_marginal_input_axes: {HighDimStatus.INVALID_SHAPE.value}"
                )
        _validate_route_storage_contract(
            route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
            storage_kind=str(self.storage_kind),
            transition_interface=str(self.transition_interface),
        )
        object.__setattr__(self, "reference_samples", reference)
        object.__setattr__(self, "previous_marginal_keep_axes", keep_axes)
        object.__setattr__(self, "previous_marginal_input_axes", input_axes)
        object.__setattr__(self, "storage_kind", str(self.storage_kind))
        object.__setattr__(self, "transition_interface", str(self.transition_interface))

    @property
    def time_index(self) -> int:
        return self.target.time_index

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteSequentialStepSpec",
            "time_index": int(self.time_index),
            "target": self.target.manifest_payload(),
            "transport": self.transport.manifest_payload(),
            "reference_samples": self.reference_samples,
            "density_components": self.density_components.manifest_payload(),
            "measure_convention": {
                "density_measure": self.measure_convention.density_measure.value,
                "mass_measure": self.measure_convention.mass_measure.value,
                "reference_weight_name": self.measure_convention.reference_weight_name,
                "physical_coordinate_name": self.measure_convention.physical_coordinate_name,
                "reference_coordinate_name": self.measure_convention.reference_coordinate_name,
                "dtype_name": self.measure_convention.dtype_name,
            },
            "previous_marginal_keep_axes": self.previous_marginal_keep_axes,
            "previous_marginal_input_axes": self.previous_marginal_input_axes,
            "storage_kind": self.storage_kind,
            "transition_interface": self.transition_interface,
        }


@dataclass(frozen=True)
class SourceRouteSequentialStepResult:
    """One step of the fixed-HMC source-route sequential loop."""

    spec: SourceRouteSequentialStepSpec
    target: SourceRouteTarget
    retained_samples: SourceRouteRetainedSampleResult
    retained_object: SourceRouteRetainedObject
    previous_retained_object: SourceRouteRetainedObject | None
    previous_marginal_density: SourceRoutePreviousMarginalDensityResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, SourceRouteSequentialStepSpec):
            raise TypeError("spec must be SourceRouteSequentialStepSpec")
        if not isinstance(self.target, SourceRouteTarget):
            raise TypeError("target must be SourceRouteTarget")
        if self.target.time_index != self.spec.time_index:
            raise ValueError("target time_index must match spec")
        if not isinstance(self.retained_samples, SourceRouteRetainedSampleResult):
            raise TypeError("retained_samples must be SourceRouteRetainedSampleResult")
        if not isinstance(self.retained_object, SourceRouteRetainedObject):
            raise TypeError("retained_object must be SourceRouteRetainedObject")
        if (
            self.retained_object.sample_diagnostics.sample_count
            != self.retained_samples.diagnostics.sample_count
            or self.retained_object.sample_diagnostics.enhancement_attempts
            != self.retained_samples.diagnostics.enhancement_attempts
        ):
            raise ValueError("retained_object diagnostics must match retained_samples")
        tf.debugging.assert_near(
            self.retained_object.sample_diagnostics.effective_sample_size,
            self.retained_samples.diagnostics.effective_sample_size,
            atol=1e-10,
        )
        tf.debugging.assert_near(
            self.retained_object.normalizer.log_transport_normalizer,
            self.retained_samples.normalizer.log_transport_normalizer,
            atol=1e-10,
        )
        tf.debugging.assert_near(
            self.retained_object.normalizer.shift_constant,
            self.retained_samples.normalizer.shift_constant,
            atol=1e-10,
        )
        if (
            self.retained_object.normalizer.log_abs_det_policy
            != self.retained_samples.normalizer.log_abs_det_policy
        ):
            raise ValueError("retained_object normalizer must match retained_samples")
        if self.spec.time_index == 1:
            if self.previous_retained_object is not None:
                raise ValueError("t=1 cannot consume a previous retained object")
            if self.previous_marginal_density is not None:
                raise ValueError("t=1 cannot have a previous marginal density")
        else:
            if not isinstance(self.previous_retained_object, SourceRouteRetainedObject):
                raise TypeError("t>1 requires a previous retained object")
            if not isinstance(
                self.previous_marginal_density,
                SourceRoutePreviousMarginalDensityResult,
            ):
                raise TypeError("t>1 requires a previous marginal density")
            if (
                self.previous_marginal_density.previous_retained_object
                is not self.previous_retained_object
            ):
                raise ValueError("previous marginal density must use previous retained object")
        if self.retained_object.samples.shape != self.retained_samples.retained_batch.samples.shape:
            raise ValueError(f"retained_object samples: {HighDimStatus.INVALID_SHAPE.value}")
        tf.debugging.assert_near(
            self.retained_object.samples,
            self.retained_samples.retained_batch.samples,
            atol=1e-10,
        )
        tf.debugging.assert_near(
            self.retained_object.log_weights,
            self.retained_samples.retained_batch.log_weights,
            atol=1e-10,
        )

    @property
    def time_index(self) -> int:
        return self.spec.time_index

    @property
    def normalizer_increment(self) -> tf.Tensor:
        return self.retained_samples.normalizer.log_increment()

    def manifest_payload(self) -> Mapping[str, object]:
        previous_hash = (
            None
            if self.previous_retained_object is None
            else self.previous_retained_object.branch_identity.hash.value
        )
        previous_marginal_density_payload = (
            None
            if self.previous_marginal_density is None
            else self.previous_marginal_density.manifest_payload()
        )
        return {
            "family": "SourceRouteSequentialStepResult",
            "time_index": int(self.time_index),
            "normalizer_increment": self.normalizer_increment,
            "previous_retained_hash": previous_hash,
            "previous_marginal_keep_axes": self.spec.previous_marginal_keep_axes,
            "previous_marginal_input_axes": self.spec.previous_marginal_input_axes,
            "previous_marginal_density": previous_marginal_density_payload,
            "spec": self.spec.manifest_payload(),
            "target": self.target.manifest_payload(),
            "retained_samples": self.retained_samples.manifest_payload(),
            "retained_object_hash": self.retained_object.branch_identity.hash.value,
        }


@dataclass(frozen=True)
class SourceRouteSequentialResult:
    """Fixed-HMC replayable sequential source-route result."""

    steps: tuple[SourceRouteSequentialStepResult, ...]
    branch_audit: SourceRouteImplementationAudit
    sequential_status: str = "sequential_fixed_hmc_source_loop"

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if len(steps) < 2:
            raise ValueError("sequential source loop requires at least two steps")
        if not all(isinstance(step, SourceRouteSequentialStepResult) for step in steps):
            raise TypeError("steps must contain SourceRouteSequentialStepResult")
        expected_times = tuple(range(1, len(steps) + 1))
        actual_times = tuple(step.time_index for step in steps)
        if actual_times != expected_times:
            raise ValueError("sequential steps must be consecutive starting at t=1")
        if not isinstance(self.branch_audit, SourceRouteImplementationAudit):
            raise TypeError("branch_audit must be SourceRouteImplementationAudit")
        if self.branch_audit.status != "PASS_SOURCE_ROUTE_OPERATION_COVERAGE":
            raise ValueError("branch_audit must pass source-route operation coverage")
        if str(self.sequential_status) != "sequential_fixed_hmc_source_loop":
            raise ValueError("sequential_status must be sequential_fixed_hmc_source_loop")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "sequential_status", str(self.sequential_status))

    @property
    def log_marginal_likelihood(self) -> tf.Tensor:
        increments = [step.normalizer_increment for step in self.steps]
        return tf.reduce_sum(tf.stack(increments))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteSequentialResult",
            "sequential_status": self.sequential_status,
            "step_count": len(self.steps),
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "branch_audit": self.branch_audit.manifest_payload(),
            "steps": [step.manifest_payload() for step in self.steps],
        }


@dataclass(frozen=True)
class P58M9SourceRoutePipelineReadiness:
    """Phase-9 launch-readiness gate for the author SIR source-route pipeline."""

    status: str
    blockers: tuple[str, ...]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        allowed = {
            P58_M9_READY_STATUS,
            P58_M9_BLOCK_MISSING_ASSEMBLY_STATUS,
            P58_M9_BLOCK_SOURCE_DRIFT_STATUS,
            P58_M9_BLOCK_HUMAN_REQUIRED_STATUS,
        }
        if status not in allowed:
            raise ValueError("unknown P58 M9 readiness status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P58_M9_READY_STATUS and blockers:
            raise ValueError("ready status cannot carry blockers")
        if status != P58_M9_READY_STATUS and not blockers:
            raise ValueError("blocked status requires at least one blocker")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    @property
    def ready_for_phase9_launch(self) -> bool:
        return self.status == P58_M9_READY_STATUS

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P58M9SourceRoutePipelineReadiness",
            "status": self.status,
            "ready_for_phase9_launch": self.ready_for_phase9_launch,
            "blockers": self.blockers,
            "manifest": self.manifest,
            "required_assembly_flags": P58_M9_REQUIRED_ASSEMBLY_FLAGS,
            "allowed_comparator_tiers": P58_M9_ALLOWED_COMPARATOR_TIERS,
            "nonclaims": (
                "no d18 correctness without comparator-tier result",
                "no d50 or d100 scaling claim",
                "no HMC readiness",
                "no adaptive Zhao-Cui parity",
                "no S&P 500 reproduction",
            ),
        }


@dataclass(frozen=True)
class P83MinimalTransportSliceReadiness:
    """Phase-3 fail-closed gate for the minimal fixed-TTSIRT transport slice."""

    status: str
    blockers: tuple[str, ...]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        allowed = {
            P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS,
            P83_MINIMAL_TRANSPORT_SLICE_BLOCK_STATUS,
        }
        if status not in allowed:
            raise ValueError("unknown P83 minimal transport slice readiness status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS and blockers:
            raise ValueError("ready status cannot carry blockers")
        if status != P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS and not blockers:
            raise ValueError("blocked status requires at least one blocker")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    @property
    def ready_for_p83_minimal_slice(self) -> bool:
        return self.status == P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P83MinimalTransportSliceReadiness",
            "status": self.status,
            "ready_for_p83_minimal_slice": self.ready_for_p83_minimal_slice,
            "blockers": self.blockers,
            "manifest": self.manifest,
            "nonclaims": (
                "no production KR closure",
                "no d18 correctness",
                "no author-scale fit quality",
                "no derivative readiness",
                "no LEDH readiness",
                "no HMC readiness",
            ),
        }


def p83_minimal_transport_slice_readiness(
    manifest: Mapping[str, object],
) -> P83MinimalTransportSliceReadiness:
    """Fail closed unless the P83 minimal fixed-TTSIRT slice is honest.

    This metadata gate is intentionally narrower than P58/P59 launch readiness.
    It only checks that a positive-defensive fixed-TTSIRT transport exposes the
    retained-marginal, CDF, proposal, and nonclaim metadata required by P83.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    data = dict(manifest)
    transport_data = data.get("transport_object")
    if isinstance(transport_data, Mapping):
        transport = dict(transport_data)
    else:
        transport = data
    blockers: list[str] = []

    if str(transport.get("source_contract_level", "")) != "fixed_ttsirt":
        blockers.append("missing_fixed_ttsirt_contract_level")
    if transport.get("tt_cores_declared") is not True:
        blockers.append("missing_tt_cores_declared")
    if transport.get("defensive_density_declared") is not True:
        blockers.append("missing_defensive_density_declared")
    if transport.get("defensive_mass_positive") is not True:
        blockers.append("missing_positive_defensive_mass")
    if (
        str(transport.get("proposition2_marginal_backend", ""))
        != "paired_core_mass_contraction_prefix_suffix"
    ):
        blockers.append("missing_paired_core_marginal_backend")
    if (
        str(transport.get("conditional_cdf_backend", ""))
        != "numerical_grid_trapezoid_bisection"
    ):
        blockers.append("missing_numerical_grid_cdf_backend")
    if (
        str(transport.get("conditional_cdf_route_class", ""))
        != "fixed_hmc_adaptation_diagnostic_approximation"
    ):
        blockers.append("missing_diagnostic_cdf_route_class")
    if transport.get("production_kr_closure") is not False:
        blockers.append("production_kr_closure_not_false")
    if str(transport.get("proposal_density_backend", "")) != "eval_pdf_on_local_samples":
        blockers.append("missing_eval_pdf_proposal_backend")

    nonclaims = tuple(str(item) for item in transport.get("p83_nonclaims", ()))
    for required in (
        "no production KR closure",
        "no d18 correctness",
        "no derivative readiness",
        "no LEDH readiness",
        "no HMC readiness",
    ):
        if required not in nonclaims:
            blockers.append(f"missing_p83_nonclaim:{required}")

    strings = _flatten_manifest_strings(transport)
    for marker in SOURCE_ROUTE_FORBIDDEN_DRIFT_MARKERS:
        if any(marker in text for text in strings):
            blockers.append(f"forbidden_source_route_marker:{marker}")

    if blockers:
        return P83MinimalTransportSliceReadiness(
            status=P83_MINIMAL_TRANSPORT_SLICE_BLOCK_STATUS,
            blockers=tuple(blockers),
            manifest=data,
        )
    return P83MinimalTransportSliceReadiness(
        status=P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS,
        blockers=(),
        manifest=data,
    )


def _flatten_manifest_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        rows: list[str] = []
        for key, child in value.items():
            rows.extend(_flatten_manifest_strings(key))
            rows.extend(_flatten_manifest_strings(child))
        return tuple(rows)
    if isinstance(value, (tuple, list, set, frozenset)):
        rows = []
        for child in value:
            rows.extend(_flatten_manifest_strings(child))
        return tuple(rows)
    if isinstance(value, str):
        return (value,)
    return ()


def p58_m9_source_route_pipeline_readiness(
    manifest: Mapping[str, object],
) -> P58M9SourceRoutePipelineReadiness:
    """Fail-closed launch gate for P58/P57-M9 source-route execution.

    The gate is intentionally metadata-only.  It prevents Phase 9 launch from
    being claimed when the available evidence is a contract-double loop, UKF or
    memory proxy, old local/operator/all-grid route, or unassembled source
    surface.  It does not build the missing d=18 pipeline.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    data = dict(manifest)
    blockers: list[str] = []
    drift: list[str] = []

    target_id = str(data.get("target_id", ""))
    pipeline_kind = str(data.get("pipeline_kind", ""))
    if target_id != P58_M9_AUTHOR_SIR_TARGET_ID:
        blockers.append("missing_author_sir_d18_target")
    if pipeline_kind != P58_M9_SOURCE_ROUTE_PIPELINE_KIND:
        blockers.append("missing_author_sir_fixed_ttsirt_source_route_pipeline")

    for flag in P58_M9_REQUIRED_ASSEMBLY_FLAGS:
        if data.get(flag) is not True:
            blockers.append(f"missing_{flag}")

    comparator_tier = str(data.get("m9_comparator_tier", ""))
    if comparator_tier not in P58_M9_ALLOWED_COMPARATOR_TIERS:
        blockers.append("missing_valid_m9_comparator_tier")

    rank_policy_status = str(data.get("rank_policy_status", ""))
    if rank_policy_status != "PASS_P57_M7_SOURCE_FAITHFUL_RANK_UKF_CALIBRATION":
        blockers.append("missing_p57_m7_rank_policy_pass")

    if data.get("preconditioned_route_required") is True:
        preconditioned_status = str(data.get("preconditioned_route_status", ""))
        if preconditioned_status != "PASS_P57_M8_PRECONDITIONED_ALGORITHM5":
            blockers.append("missing_required_preconditioned_route_pass")

    if data.get("uses_contract_test_double") is True:
        drift.append("contract_test_double_cannot_launch_phase9")
    if data.get("uses_ukf_as_comparator") is True:
        drift.append("ukf_proxy_cannot_launch_phase9")
    if data.get("uses_rank_memory_proxy_as_comparator") is True:
        drift.append("rank_memory_proxy_cannot_launch_phase9")

    route_class = str(data.get("route_class", ""))
    storage_kind = str(data.get("storage_kind", ""))
    transition_interface = str(data.get("transition_interface", ""))
    if route_class and route_class != "fixed_ttsirt_source_route":
        drift.append("route_class_is_not_fixed_ttsirt_source_route")
    for marker in SOURCE_ROUTE_FORBIDDEN_DRIFT_MARKERS:
        if marker in (route_class, storage_kind, transition_interface):
            drift.append(f"forbidden_source_route_marker:{marker}")

    if drift:
        return P58M9SourceRoutePipelineReadiness(
            status=P58_M9_BLOCK_SOURCE_DRIFT_STATUS,
            blockers=tuple(drift + blockers),
            manifest=data,
        )
    if blockers:
        return P58M9SourceRoutePipelineReadiness(
            status=P58_M9_BLOCK_MISSING_ASSEMBLY_STATUS,
            blockers=tuple(blockers),
            manifest=data,
        )
    return P58M9SourceRoutePipelineReadiness(
        status=P58_M9_READY_STATUS,
        blockers=(),
        manifest=data,
    )


@dataclass(frozen=True)
class P59AuthorSIRRouteDecisionResult:
    """P59-9c source-cited full/preconditioned route decision gate."""

    status: str
    blockers: tuple[str, ...]
    route_decision: str
    preconditioned_route_required: bool
    preconditioned_route_status: str
    source_anchors: tuple[str, ...]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in (P59_9C_PASS_STATUS, P59_9C_BLOCK_STATUS):
            raise ValueError("unknown P59-9c status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P59_9C_PASS_STATUS and blockers:
            raise ValueError("P59-9c pass cannot carry blockers")
        if status == P59_9C_BLOCK_STATUS and not blockers:
            raise ValueError("P59-9c block requires at least one blocker")
        route_decision = str(self.route_decision)
        allowed_decisions = {
            P59_9C_FULL_ROUTE_SELECTED,
            P59_9C_PRECONDITIONED_ROUTE_REQUIRED,
            P59_9C_PRECONDITIONED_ROUTE_DEFERRED,
            P59_9C_ROUTE_DECISION_BLOCKED,
        }
        if route_decision not in allowed_decisions:
            raise ValueError("unknown P59-9c route decision")
        preconditioned_required = bool(self.preconditioned_route_required)
        if route_decision == P59_9C_FULL_ROUTE_SELECTED and preconditioned_required:
            raise ValueError("full route decision cannot require preconditioned route")
        if (
            route_decision == P59_9C_PRECONDITIONED_ROUTE_REQUIRED
            and not preconditioned_required
        ):
            raise ValueError("preconditioned route decision must require preconditioned route")
        if (
            status == P59_9C_PASS_STATUS
            and preconditioned_required
            and str(self.preconditioned_route_status) != P57_M8_PRECONDITIONED_PASS_STATUS
        ):
            raise ValueError("P59-9c pass with preconditioned route requires P57-M8 pass")
        anchors = tuple(str(anchor) for anchor in self.source_anchors)
        if not anchors:
            raise ValueError("P59-9c source anchors must be nonempty")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "route_decision", route_decision)
        object.__setattr__(
            self, "preconditioned_route_required", preconditioned_required
        )
        object.__setattr__(
            self, "preconditioned_route_status", str(self.preconditioned_route_status)
        )
        object.__setattr__(self, "source_anchors", anchors)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    @property
    def unlocks_step_spec_assembly(self) -> bool:
        return self.status == P59_9C_PASS_STATUS

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P59AuthorSIRRouteDecisionResult",
            "status": self.status,
            "blockers": self.blockers,
            "route_decision": self.route_decision,
            "preconditioned_route_required": self.preconditioned_route_required,
            "preconditioned_route_status": self.preconditioned_route_status,
            "unlocks_step_spec_assembly": self.unlocks_step_spec_assembly,
            "source_anchors": self.source_anchors,
            "manifest": self.manifest,
        }


def p59_author_sir_route_decision(
    *,
    source_root: str | Path = "third_party/audit/zhao_cui_tensor_ssm_p10/source",
    target_id: str = P58_M9_AUTHOR_SIR_TARGET_ID,
) -> P59AuthorSIRRouteDecisionResult:
    """Decide whether author Austria SIR Phase 9 uses ``full_sol`` or ``pre_sol``.

    The decision is intentionally source-backed and fail-closed.  For the
    author SIR row, ``eg3_sir/mainscript.m`` constructs ``full_sol``; therefore
    P59-9b and P59-9d may assemble the full source route without requiring the
    P57-M8 preconditioned surface.  If the source files are absent or
    contradictory, this helper blocks instead of guessing.
    """

    root = Path(source_root)
    files = {
        "mainscript": root / "eg3_sir" / "mainscript.m",
        "full_sol": root / "models" / "full_sol.m",
        "pre_sol": root / "models" / "pre_sol.m",
        "precond": root / "models" / "tensordot" / "precond.m",
    }
    blockers: list[str] = []
    texts: dict[str, str] = {}
    for label, path in files.items():
        try:
            texts[label] = path.read_text(encoding="utf-8")
        except OSError:
            blockers.append(f"missing_source_file:{label}")
            texts[label] = ""

    if str(target_id) != P58_M9_AUTHOR_SIR_TARGET_ID:
        blockers.append("unsupported_target_id_for_p59_9c_author_sir_route")

    mainscript = texts["mainscript"]
    full_source = texts["full_sol"]
    pre_source = texts["pre_sol"]
    precond_source = texts["precond"]
    mainscript_selects_full = (
        "full_sol(myModel" in mainscript and "solve(mySol_unbounded)" in mainscript
    )
    mainscript_selects_pre = "pre_sol(" in mainscript
    full_route_boundary = (
        "classdef full_sol < Y_sol" in full_source
        and "model.d+2*model.m" in full_source
        and "eval_irt(sol.SIRTs{t}" in full_source
        and "eval_pdf(sol.SIRTs{t}, r)" in full_source
    )
    pre_route_boundary = (
        "classdef pre_sol < full_sol" in pre_source
        and "function sol = pre_sol" in pre_source
        and "sol.precond" in pre_source
    )
    precond_boundary = (
        "function [C, Sigmak] = precond" in precond_source
        and "Sigma1\\Sigma2full" in precond_source
        and "Sigmak" in precond_source
    )

    if not mainscript_selects_full:
        blockers.append("author_sir_mainscript_does_not_select_full_sol")
    if mainscript_selects_pre:
        blockers.append("author_sir_mainscript_also_mentions_pre_sol")
    if not full_route_boundary:
        blockers.append("full_sol_source_boundary_not_verified")
    if not pre_route_boundary:
        blockers.append("pre_sol_boundary_not_verified")
    if not precond_boundary:
        blockers.append("precond_boundary_not_verified")

    evidence = {
        "mainscript_selects_full_sol": mainscript_selects_full,
        "mainscript_selects_pre_sol": mainscript_selects_pre,
        "full_sol_source_boundary_verified": full_route_boundary,
        "pre_sol_boundary_verified": pre_route_boundary,
        "precond_boundary_verified": precond_boundary,
    }
    if blockers:
        return _p59_9c_block_result(blockers=tuple(blockers), evidence=evidence)

    manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P59-9c",
        "artifact_role": "source_route_decision_gate",
        "route_decision": P59_9C_FULL_ROUTE_SELECTED,
        "source_selected_matlab_constructor": "full_sol",
        "source_selected_python_route": "full_route",
        "route_class": "fixed_ttsirt_source_route",
        "preconditioned_route_required": False,
        "preconditioned_route_status": P59_9C_NOT_REQUIRED_PRECONDITIONED_STATUS,
        "source_evidence": evidence,
        "source_anchors": _P59_9C_AUTHOR_SIR_SOURCE_ANCHORS,
        "unlocks_after_consumed_by": ("P59-9b", "P59-9d"),
        "fail_closed_rule": (
            "P59-9b and P59-9d must block if this route-decision artifact is absent "
            "or if route_decision is not full_route_selected."
        ),
        "nonclaims": (
            "no Phase-9 validation launch",
            "no d18 filtering accuracy claim",
            "no preconditioned route claim for author SIR",
            "no UKF route substitute",
            "no adaptive Zhao-Cui parity claim",
        ),
    }
    return P59AuthorSIRRouteDecisionResult(
        status=P59_9C_PASS_STATUS,
        blockers=(),
        route_decision=P59_9C_FULL_ROUTE_SELECTED,
        preconditioned_route_required=False,
        preconditioned_route_status=P59_9C_NOT_REQUIRED_PRECONDITIONED_STATUS,
        source_anchors=_P59_9C_AUTHOR_SIR_SOURCE_ANCHORS,
        manifest=manifest,
    )


def _p59_9c_block_result(
    *,
    blockers: tuple[str, ...],
    evidence: Mapping[str, object],
) -> P59AuthorSIRRouteDecisionResult:
    manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P59-9c",
        "artifact_role": "blocked_source_route_decision_gate",
        "route_decision": P59_9C_ROUTE_DECISION_BLOCKED,
        "preconditioned_route_required": False,
        "preconditioned_route_status": "undecided",
        "blockers": tuple(blockers),
        "source_evidence": dict(evidence),
        "source_anchors": _P59_9C_AUTHOR_SIR_SOURCE_ANCHORS,
        "fail_closed_rule": "P59-9b and P59-9d must not proceed.",
        "nonclaims": (
            "no Phase-9 validation launch",
            "no d18 filtering accuracy claim",
        ),
    }
    return P59AuthorSIRRouteDecisionResult(
        status=P59_9C_BLOCK_STATUS,
        blockers=tuple(blockers),
        route_decision=P59_9C_ROUTE_DECISION_BLOCKED,
        preconditioned_route_required=False,
        preconditioned_route_status="undecided",
        source_anchors=_P59_9C_AUTHOR_SIR_SOURCE_ANCHORS,
        manifest=manifest,
    )


@dataclass(frozen=True)
class P59AuthorSIR36DTargetFitPrepResult:
    """Bounded author-SIR source-route 36D target/fit preparation artifact."""

    status: str
    blockers: tuple[str, ...]
    target_dimension: int
    state_dimension: int
    parameter_dimension: int
    time_index: int
    sample_count: int
    negative_log_values: tf.Tensor
    shift_constant: tf.Tensor
    fit_status: str
    fit_branch_hash: str | None
    density_branch_hash: str | None
    transport_manifest: Mapping[str, object] | None
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in (P59_9A_PASS_STATUS, P59_9A_BLOCK_STATUS):
            raise ValueError("unknown P59-9a status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P59_9A_PASS_STATUS and blockers:
            raise ValueError("P59-9a pass cannot carry blockers")
        if status == P59_9A_BLOCK_STATUS and not blockers:
            raise ValueError("P59-9a block requires at least one blocker")
        values = tf.convert_to_tensor(self.negative_log_values, dtype=tf.float64)
        shift = tf.convert_to_tensor(self.shift_constant, dtype=tf.float64)
        if values.shape.rank != 1 or int(values.shape[0]) != int(self.sample_count):
            raise ValueError(f"negative_log_values: {HighDimStatus.INVALID_SHAPE.value}")
        if shift.shape.rank != 0:
            raise ValueError(f"shift_constant: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(
            tf.reduce_all(tf.math.is_finite(values)).numpy()
            and tf.math.is_finite(shift).numpy()
        ):
            raise ValueError(f"P59AuthorSIR36DTargetFitPrepResult: {HighDimStatus.NONFINITE_VALUE.value}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "target_dimension", int(self.target_dimension))
        object.__setattr__(self, "state_dimension", int(self.state_dimension))
        object.__setattr__(self, "parameter_dimension", int(self.parameter_dimension))
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "negative_log_values", values)
        object.__setattr__(self, "shift_constant", shift)
        object.__setattr__(self, "fit_status", str(self.fit_status))
        object.__setattr__(
            self,
            "fit_branch_hash",
            None if self.fit_branch_hash is None else str(self.fit_branch_hash),
        )
        object.__setattr__(
            self,
            "density_branch_hash",
            None if self.density_branch_hash is None else str(self.density_branch_hash),
        )
        object.__setattr__(
            self,
            "transport_manifest",
            None if self.transport_manifest is None else freeze_mapping(self.transport_manifest),
        )
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P59AuthorSIR36DTargetFitPrepResult",
            "status": self.status,
            "blockers": self.blockers,
            "target_dimension": int(self.target_dimension),
            "state_dimension": int(self.state_dimension),
            "parameter_dimension": int(self.parameter_dimension),
            "time_index": int(self.time_index),
            "sample_count": int(self.sample_count),
            "negative_log_values": self.negative_log_values,
            "shift_constant": self.shift_constant,
            "fit_status": self.fit_status,
            "fit_branch_hash": self.fit_branch_hash,
            "density_branch_hash": self.density_branch_hash,
            "transport_manifest": self.transport_manifest,
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class P59AuthorSIRStepSpecAssemblyResult:
    """P59-9b bounded author-SIR source-route step-spec assembly artifact."""

    status: str
    blockers: tuple[str, ...]
    route_decision: P59AuthorSIRRouteDecisionResult
    prep_result: P59AuthorSIR36DTargetFitPrepResult | None
    step_specs: tuple[SourceRouteSequentialStepSpec, ...]
    sequential_result: SourceRouteSequentialResult | None
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in (P59_9B_PASS_STATUS, P59_9B_BLOCK_STATUS):
            raise ValueError("unknown P59-9b status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P59_9B_PASS_STATUS and blockers:
            raise ValueError("P59-9b pass cannot carry blockers")
        if status == P59_9B_BLOCK_STATUS and not blockers:
            raise ValueError("P59-9b block requires at least one blocker")
        if not isinstance(self.route_decision, P59AuthorSIRRouteDecisionResult):
            raise TypeError("route_decision must be P59AuthorSIRRouteDecisionResult")
        if self.prep_result is not None and not isinstance(
            self.prep_result,
            P59AuthorSIR36DTargetFitPrepResult,
        ):
            raise TypeError("prep_result must be P59AuthorSIR36DTargetFitPrepResult")
        specs = tuple(self.step_specs)
        if status == P59_9B_PASS_STATUS:
            if len(specs) < 2:
                raise ValueError("P59-9b pass requires at least two step specs")
            if not all(isinstance(spec, SourceRouteSequentialStepSpec) for spec in specs):
                raise TypeError("step_specs must contain SourceRouteSequentialStepSpec")
            if self.sequential_result is None:
                raise ValueError("P59-9b pass requires sequential_result")
            if not isinstance(self.sequential_result, SourceRouteSequentialResult):
                raise TypeError("sequential_result must be SourceRouteSequentialResult")
        elif self.sequential_result is not None:
            raise ValueError("P59-9b block cannot carry sequential_result")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "step_specs", specs)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    @property
    def ready_for_runner_manifest_path(self) -> bool:
        return self.status == P59_9B_PASS_STATUS

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P59AuthorSIRStepSpecAssemblyResult",
            "status": self.status,
            "blockers": self.blockers,
            "ready_for_runner_manifest_path": self.ready_for_runner_manifest_path,
            "route_decision": self.route_decision.manifest_payload(),
            "prep_result": (
                None if self.prep_result is None else self.prep_result.manifest_payload()
            ),
            "step_specs": [spec.manifest_payload() for spec in self.step_specs],
            "sequential_result": (
                None
                if self.sequential_result is None
                else self.sequential_result.manifest_payload()
            ),
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class P59AuthorSIRRunnerManifestResult:
    """P59-9d runner path plus P58 readiness manifest artifact."""

    status: str
    blockers: tuple[str, ...]
    assembly_result: P59AuthorSIRStepSpecAssemblyResult | None
    readiness: P58M9SourceRoutePipelineReadiness
    manifest_path: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in (P59_9D_PASS_STATUS, P59_9D_BLOCK_STATUS):
            raise ValueError("unknown P59-9d status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P59_9D_PASS_STATUS and blockers:
            raise ValueError("P59-9d pass cannot carry blockers")
        if status == P59_9D_BLOCK_STATUS and not blockers:
            raise ValueError("P59-9d block requires at least one blocker")
        if self.assembly_result is not None and not isinstance(
            self.assembly_result,
            P59AuthorSIRStepSpecAssemblyResult,
        ):
            raise TypeError("assembly_result must be P59AuthorSIRStepSpecAssemblyResult")
        if not isinstance(self.readiness, P58M9SourceRoutePipelineReadiness):
            raise TypeError("readiness must be P58M9SourceRoutePipelineReadiness")
        if status == P59_9D_PASS_STATUS:
            if self.readiness.status != P58_M9_READY_STATUS:
                raise ValueError("P59-9d pass requires P58 ready status")
            if self.assembly_result is None:
                raise ValueError("P59-9d pass requires assembly_result")
            if self.assembly_result.status != P59_9B_PASS_STATUS:
                raise ValueError("P59-9d pass requires P59-9b pass")
        if not str(self.manifest_path).strip():
            raise ValueError("manifest_path must be nonempty")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "manifest_path", str(self.manifest_path))
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    @property
    def ready_for_validation_ladder(self) -> bool:
        return self.status == P59_9D_PASS_STATUS and self.readiness.ready_for_phase9_launch

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P59AuthorSIRRunnerManifestResult",
            "status": self.status,
            "blockers": self.blockers,
            "ready_for_validation_ladder": self.ready_for_validation_ladder,
            "manifest_path": self.manifest_path,
            "readiness": self.readiness.manifest_payload(),
            "assembly_result": (
                None
                if self.assembly_result is None
                else self.assembly_result.manifest_payload()
            ),
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class P59AuthorSIRValidationLadderResult:
    """P59-9e validation-ladder result with explicit tier boundary."""

    status: str
    blockers: tuple[str, ...]
    tier: str
    runner_result: P59AuthorSIRRunnerManifestResult | None
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in (P59_9E_D18_EXECUTION_ONLY_PASS_STATUS, P59_9E_BLOCK_STATUS):
            raise ValueError("unknown P59-9e status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P59_9E_D18_EXECUTION_ONLY_PASS_STATUS and blockers:
            raise ValueError("P59-9e pass cannot carry blockers")
        if status == P59_9E_BLOCK_STATUS and not blockers:
            raise ValueError("P59-9e block requires at least one blocker")
        tier = str(self.tier)
        if tier not in P58_M9_ALLOWED_COMPARATOR_TIERS:
            raise ValueError("unknown P59-9e tier")
        if self.runner_result is not None and not isinstance(
            self.runner_result,
            P59AuthorSIRRunnerManifestResult,
        ):
            raise TypeError("runner_result must be P59AuthorSIRRunnerManifestResult")
        if status == P59_9E_D18_EXECUTION_ONLY_PASS_STATUS:
            if tier != "d18_execution_only":
                raise ValueError("only d18_execution_only pass is implemented")
            if self.runner_result is None or self.runner_result.status != P59_9D_PASS_STATUS:
                raise ValueError("P59-9e pass requires P59-9d pass")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P59AuthorSIRValidationLadderResult",
            "status": self.status,
            "blockers": self.blockers,
            "tier": self.tier,
            "runner_result": (
                None if self.runner_result is None else self.runner_result.manifest_payload()
            ),
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class P60AuthorSIRSameRouteRankComparatorResult:
    """P60-2 same-route rank-comparator result with explicit nonclaims."""

    status: str
    blockers: tuple[str, ...]
    low_result: P59AuthorSIRStepSpecAssemblyResult | None
    high_result: P59AuthorSIRStepSpecAssemblyResult | None
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in (
            P60_D18_RANK_CONVERGENCE_PASS_STATUS,
            P60_D18_RANK_CONVERGENCE_BLOCK_STATUS,
        ):
            raise ValueError("unknown P60-2 status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if status == P60_D18_RANK_CONVERGENCE_PASS_STATUS and blockers:
            raise ValueError("P60-2 pass cannot carry blockers")
        if status == P60_D18_RANK_CONVERGENCE_BLOCK_STATUS and not blockers:
            raise ValueError("P60-2 block requires at least one blocker")
        for name, value in (
            ("low_result", self.low_result),
            ("high_result", self.high_result),
        ):
            if value is not None and not isinstance(value, P59AuthorSIRStepSpecAssemblyResult):
                raise TypeError(f"{name} must be P59AuthorSIRStepSpecAssemblyResult")
        if status == P60_D18_RANK_CONVERGENCE_PASS_STATUS:
            if self.low_result is None or self.high_result is None:
                raise ValueError("P60-2 pass requires low and high comparator rows")
            if (
                self.low_result.status != P59_9B_PASS_STATUS
                or self.high_result.status != P59_9B_PASS_STATUS
            ):
                raise ValueError("P60-2 pass requires P59-9b pass rows")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P60AuthorSIRSameRouteRankComparatorResult",
            "status": self.status,
            "blockers": self.blockers,
            "low_result": (
                None if self.low_result is None else self.low_result.manifest_payload()
            ),
            "high_result": (
                None if self.high_result is None else self.high_result.manifest_payload()
            ),
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class P66AuthorSIRFixedBranchValidationLadderResult:
    """P66 fixed-branch validation ladder result with explicit nonclaims."""

    status: str
    blockers: tuple[str, ...]
    candidate_result: P59AuthorSIRStepSpecAssemblyResult | None
    sentinel_result: P60AuthorSIRSameRouteRankComparatorResult | None
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        status = str(self.status)
        allowed = {
            P66_FIXED_BRANCH_VALIDATION_LADDER_READY_STATUS,
            P66_ADJACENT_LADDER_STABLE_PASS_STATUS,
            P66_SOURCE_ROUTE_INVARIANT_DRIFT_BLOCK_STATUS,
            P66_FIXED_BRANCH_DEFENSIVE_ONLY_BLOCK_STATUS,
            P66_FIT_DESIGN_UNDERDETERMINED_BLOCK_STATUS,
            P66_ADJACENT_RANK_LADDER_NOT_STABLE_BLOCK_STATUS,
            P66_ADJACENT_DEGREE_LADDER_NOT_STABLE_BLOCK_STATUS,
            P66_VALIDATION_LADDER_SCOPE_BLOCK_STATUS,
        }
        if status not in allowed:
            raise ValueError("unknown P66 status")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        ready_or_pass = status in (
            P66_FIXED_BRANCH_VALIDATION_LADDER_READY_STATUS,
            P66_ADJACENT_LADDER_STABLE_PASS_STATUS,
        )
        if ready_or_pass and blockers:
            raise ValueError("P66 ready/pass cannot carry blockers")
        if not ready_or_pass and not blockers:
            raise ValueError("P66 block requires at least one blocker")
        if self.candidate_result is not None and not isinstance(
            self.candidate_result,
            P59AuthorSIRStepSpecAssemblyResult,
        ):
            raise TypeError("candidate_result must be P59AuthorSIRStepSpecAssemblyResult")
        if self.sentinel_result is not None and not isinstance(
            self.sentinel_result,
            P60AuthorSIRSameRouteRankComparatorResult,
        ):
            raise TypeError("sentinel_result must be P60AuthorSIRSameRouteRankComparatorResult")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "P66AuthorSIRFixedBranchValidationLadderResult",
            "status": self.status,
            "blockers": self.blockers,
            "candidate_result": (
                None
                if self.candidate_result is None
                else self.candidate_result.manifest_payload()
            ),
            "sentinel_result": (
                None if self.sentinel_result is None else self.sentinel_result.manifest_payload()
            ),
            "manifest": self.manifest,
        }


def p59_author_sir_36d_target_fit_prep(
    *,
    sample_count: int = 6,
    fit_degree: int = 0,
    fit_rank: int = 1,
    ridge: float = 1e-8,
) -> P59AuthorSIR36DTargetFitPrepResult:
    """Build bounded P59-9a author-SIR 36D source-route fit evidence.

    This is deliberately a preparation artifact.  It checks the source-route
    target dimension and finite target/fixed-TT-fit plumbing for the author SIR
    row; it does not claim filtering accuracy or Phase-9 validation success.
    """

    model = zhao_cui_sir_austria_model()
    d = model.parameter_dim()
    m = model.state_dim()
    target_dim = d + 2 * m
    blockers: list[str] = []
    if d != 0:
        blockers.append("author_sir_parameter_dim_not_zero")
    if m != 18:
        blockers.append("author_sir_state_dim_not_18")
    if target_dim != P59_9A_AUTHOR_SIR_TARGET_DIMENSION:
        blockers.append("source_route_target_dimension_not_36")
    if int(sample_count) < 2:
        blockers.append("sample_count_must_be_at_least_2")
    if int(fit_degree) < 0:
        blockers.append("fit_degree_must_be_nonnegative")
    if int(fit_rank) <= 0:
        blockers.append("fit_rank_must_be_positive")
    if blockers:
        return _p59_9a_block_result(
            blockers=tuple(blockers),
            target_dimension=target_dim,
            state_dimension=m,
            parameter_dimension=d,
        )

    observations = model.simulate(final_time=1, seed=5901)[1]
    fit_data = _p59_author_sir_source_fit_data_for_step(
        model=model,
        observations=observations,
        time_index=1,
        fit_sample_count=int(sample_count),
    )
    coordinate_frame = fit_data.frame
    negative_log_values = fit_data.negative_log_values
    if not bool(tf.reduce_all(tf.math.is_finite(negative_log_values)).numpy()):
        return _p59_9a_block_result(
            blockers=("nonfinite_author_sir_36d_target_values",),
            target_dimension=target_dim,
            state_dimension=m,
            parameter_dimension=d,
            negative_log_values=negative_log_values,
        )

    shift_constant = fit_data.shift_constant
    target_values = fit_data.target_values
    if not bool(tf.reduce_all(tf.math.is_finite(target_values)).numpy()):
        return _p59_9a_block_result(
            blockers=("nonfinite_author_sir_36d_sqrt_target_values",),
            target_dimension=target_dim,
            state_dimension=m,
            parameter_dimension=d,
            negative_log_values=negative_log_values,
            shift_constant=shift_constant,
        )

    convention = _p59_reference_convention()
    legacy_basis_config = p85_legacy_legendre_product_basis_spec(
        dimension=target_dim,
        fit_degree=int(fit_degree),
        convention=convention,
    )
    author_basis_config = p85_author_sir_lagrangep_algebraic_product_basis_spec(
        dimension=target_dim,
        convention=convention,
    )
    product_basis = ProductBasis(
        [
            LegendreBasis1D(BoundedInterval(-1.0, 1.0), int(fit_degree))
            for _ in range(target_dim)
        ],
        convention,
    )
    fit_config = FixedTTFitConfig(
        ranks=_source_route_rank_tuple(target_dim, int(fit_rank)),
        ridge=float(ridge),
        max_sweeps=1,
        sweep_order=tuple(range(target_dim)),
        row_budget=max(int(sample_count), 1),
        column_budget=(int(fit_degree) + 1) * int(fit_rank) * int(fit_rank),
        dense_matrix_byte_budget=2**22,
        normal_matrix_byte_budget=2**24,
        condition_number_warning=1e12,
        condition_number_veto=1e16,
        holdout_tolerance=1e6,
    )
    initial_cores = _source_route_constant_path_initial_cores(
        ranks=fit_config.ranks,
        basis_dim=int(fit_degree) + 1,
        constant_value=_weighted_mean_target_value(target_values, fit_data.fit_weights),
    )
    fit_result = FixedTTFitter().fit(
        product_basis=product_basis,
        samples=FixedTTFitSampleBatch(
            points=tf.transpose(fit_data.local_fit_points),
            target_values=target_values,
            weights=fit_data.fit_weights,
        ),
        config=fit_config,
        initial_cores=initial_cores,
        branch_seed="p59-9a-author-sir-36d-bounded-fit",
        measure_convention=convention,
        initialization_rule=P65_FIXED_BRANCH_INITIALIZATION_RULE,
    )
    fit_quality_diagnostics = _p59_fixed_ttsirt_fit_quality_diagnostics(fit_result)
    if fit_result.status is not HighDimStatus.OK:
        return _p59_9a_block_result(
            blockers=(f"fit_status_{fit_result.status.value}",),
            target_dimension=target_dim,
            state_dimension=m,
            parameter_dimension=d,
            negative_log_values=negative_log_values,
            shift_constant=shift_constant,
            fit_status=fit_result.status.value,
            fit_branch_hash=fit_result.branch_hash.value,
        )

    defensive = TensorProductReferenceDensity(product_basis, convention)
    defensive_tau = _p59_author_sir_defensive_tau_tensor()
    density_identity = SquaredTTDensity.expected_branch_identity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=defensive_tau,
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=convention,
    )
    density = SquaredTTDensity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=defensive_tau,
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=convention,
        branch_identity=density_identity,
    )
    transport = FixedTTSIRTTransport(
        density=density,
        cdf_config=KRCDFConfig(
            grid_size=5,
            bisection_steps=4,
            monotonicity_tolerance=1e-10,
            bracket_tolerance=1e-10,
            denominator_floor=1e-12,
            max_floor_count=0,
        ),
    )
    transport_manifest = transport.manifest_payload()
    manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P59-9a",
        "artifact_role": "bounded_preparation_evidence_only",
        "source_contract_level": "fixed_ttsirt",
        "source_anchors": (
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:14-56",
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:11-39",
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:76-130",
            *_P62_AUTHOR_TTSIRT_DEFENSIVE_TAU_SOURCE_ANCHORS,
            *_P63_AUTHOR_SIR_FIT_DATA_SOURCE_ANCHORS,
        ),
        "fit_data_mode": P63_AUTHOR_SIR_SOURCE_FIT_DATA_MODE,
        "fit_data_manifest": fit_data.manifest,
        "defensive_tau": P62_AUTHOR_TTSIRT_EXECUTABLE_DEFAULT_TAU,
        "defensive_tau_source": "author_executable_ttsirt_default",
        "source_declared_tau_unwired": 10.0,
        "source_executable_ttsirt_default_tau": P62_AUTHOR_TTSIRT_EXECUTABLE_DEFAULT_TAU,
        "parameter_dimension": d,
        "state_dimension": m,
        "target_dimension": target_dim,
        "source_target_order": "[theta, x_t, x_{t-1}]",
        "coordinate_frame_dimension": coordinate_frame.dimension,
        "coordinate_frame_log_abs_det": coordinate_frame.log_abs_det(),
        "author_sir_d_plus_2m_dimension_check": target_dim == 36,
        "sample_count": int(sample_count),
        "rank_tuple": fit_config.ranks,
        "fit_degree": int(fit_degree),
        "fit_rank": int(fit_rank),
        "basis_domain_config": legacy_basis_config.manifest_payload(),
        "author_basis_domain_config_available": author_basis_config.manifest_payload(),
        "author_basis_domain_config_status": P85_AUTHOR_SIR_BASIS_DOMAIN_CONFIG_STATUS,
        "author_basis_domain_full_fit_status": P85_AUTHOR_SIR_BASIS_DOMAIN_FULL_FIT_BLOCK_STATUS,
        "fixed_branch_adaptation_class": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "fit_initialization_rule": P65_FIXED_BRANCH_INITIALIZATION_RULE,
        "fit_initialization_rule_source": "docs/plans/bayesfilter-highdim-zhao-cui-p50-chapter-discipline-rewrite-2026-06-12.tex:prop:p50-constant-path-initialization",
        "fit_status": fit_result.status.value,
        "fit_quality_diagnostics": fit_quality_diagnostics,
        "fit_branch_hash": fit_result.branch_hash.value,
        "density_branch_hash": density.branch_identity.hash.value,
        "transport_family": transport_manifest["family"],
        "transport_dimension": transport_manifest["dimension"],
        "nonclaims": (
            "no Phase-9 validation launch",
            "no d18 filtering accuracy claim",
            "no same-route rank convergence claim",
            "no d50 or d100 scaling claim",
            "no HMC production readiness claim",
            "legacy bounded Legendre route remains local_gap/diagnostic_legendre_route",
            "author Lagrangep(4,8) AlgebraicMapping(1) config is not fitted in P85",
        ),
    }
    return P59AuthorSIR36DTargetFitPrepResult(
        status=P59_9A_PASS_STATUS,
        blockers=(),
        target_dimension=target_dim,
        state_dimension=m,
        parameter_dimension=d,
        time_index=1,
        sample_count=int(sample_count),
        negative_log_values=negative_log_values,
        shift_constant=shift_constant,
        fit_status=fit_result.status.value,
        fit_branch_hash=fit_result.branch_hash.value,
        density_branch_hash=density.branch_identity.hash.value,
        transport_manifest=transport_manifest,
        manifest=manifest,
    )


def _p59_author_sir_source_density_callbacks(model, observation: tf.Tensor):
    theta = tf.zeros([0], dtype=tf.float64)
    y_t = tf.convert_to_tensor(observation, dtype=tf.float64)

    def prior_log_density(prior_points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(prior_points, dtype=tf.float64)
        if points.shape.rank != 2 or int(points.shape[0]) != model.state_dim():
            raise ValueError(f"prior_points: {HighDimStatus.INVALID_SHAPE.value}")
        return model.initial_log_density(theta, tf.transpose(points))

    def transition_log_density(points: tf.Tensor, time_index: int) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        x_t = tf.transpose(values[: model.state_dim(), :])
        x_previous = tf.transpose(values[model.state_dim() :, :])
        return model.transition_log_density(theta, x_previous, x_t, t=int(time_index))

    def likelihood_log_density(points: tf.Tensor, time_index: int) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        x_t = tf.transpose(values[: model.state_dim(), :])
        return model.observation_log_density(theta, x_t, y_t, t=int(time_index))

    return prior_log_density, transition_log_density, likelihood_log_density


def _p59_author_sir_36d_coordinate_frame(model) -> SourceRouteCoordinateFrame:
    return _p59_author_sir_36d_coordinate_frame_for_time(model, time_index=1)


def _p59_author_sir_uniform_log_weights(sample_count: int) -> tf.Tensor:
    n = int(sample_count)
    if n <= 0:
        raise ValueError("sample_count must be positive")
    return tf.fill(
        [n],
        -tf.math.log(tf.cast(n, tf.float64)),
    )


def _p59_author_sir_prior_sample_batch(
    *,
    model,
    sample_count: int,
    seed: int,
) -> SourceRouteSampleBatch:
    n = int(sample_count)
    if n <= 0:
        raise ValueError("sample_count must be positive")
    generator = tf.random.Generator.from_seed(int(seed))
    initial_chol = tf.linalg.cholesky(model.initial_covariance)
    noise = generator.normal([n, model.state_dim()], dtype=tf.float64)
    x0 = model.initial_mean[tf.newaxis, :] + tf.linalg.matmul(
        noise,
        initial_chol,
        transpose_b=True,
    )
    theta = tf.zeros([model.parameter_dim(), n], dtype=tf.float64)
    samples = tf.concat([theta, tf.transpose(x0)], axis=0)
    return SourceRouteSampleBatch(
        samples=samples,
        log_weights=_p59_author_sir_uniform_log_weights(n),
        time_index=0,
        route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
        sample_origin="author_prior_samples_fixed_seed",
    )


def _p59_author_sir_batch_from_previous_retained_prefix(
    previous_retained_object: SourceRouteRetainedObject,
    *,
    parameter_dim: int,
    state_dim: int,
    time_index: int,
) -> SourceRouteSampleBatch:
    if not isinstance(previous_retained_object, SourceRouteRetainedObject):
        raise TypeError("previous_retained_object must be SourceRouteRetainedObject")
    d = int(parameter_dim)
    m = int(state_dim)
    samples = previous_retained_object.samples[: d + m, :]
    return SourceRouteSampleBatch(
        samples=samples,
        log_weights=_p59_author_sir_uniform_log_weights(int(samples.shape[1])),
        time_index=int(time_index),
        route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
        sample_origin="author_retained_prefix_uniform_weight_fixed_variant",
    )


def _p59_author_sir_source_push_result(
    *,
    model,
    previous_batch: SourceRouteSampleBatch,
    observation: tf.Tensor,
    time_index: int,
    process_noise_seed: int,
) -> SourceRoutePushResult:
    n = previous_batch.sample_count
    m = model.state_dim()
    generator = tf.random.Generator.from_seed(int(process_noise_seed))
    transition_noise = generator.normal([n, m], dtype=tf.float64)
    y_t = tf.convert_to_tensor(observation, dtype=tf.float64)

    def transition_fn(previous_samples: tf.Tensor, step_index: int) -> tf.Tensor:
        values = tf.convert_to_tensor(previous_samples, dtype=tf.float64)
        theta = tf.transpose(values[: model.parameter_dim(), :])
        previous_state = tf.transpose(values[model.parameter_dim() :, :])
        pushed = model.transition_push_from_standard_normal(
            theta,
            previous_state,
            transition_noise,
            t=int(step_index),
        )
        return tf.transpose(pushed)

    def log_likelihood_fn(propagated_samples: tf.Tensor, step_index: int) -> tf.Tensor:
        values = tf.convert_to_tensor(propagated_samples, dtype=tf.float64)
        theta = tf.transpose(values[: model.parameter_dim(), :])
        current_state = tf.transpose(values[model.parameter_dim() :, :])
        return model.observation_log_density(
            theta,
            current_state,
            y_t,
            t=int(step_index),
        )

    return source_route_push_and_augment_samples(
        previous_batch=previous_batch,
        transition_fn=transition_fn,
        log_likelihood_fn=log_likelihood_fn,
        parameter_dim=model.parameter_dim(),
        state_dim=m,
        time_index=int(time_index),
    )


def _p59_author_sir_deterministic_weighted_resample(
    *,
    samples: tf.Tensor,
    log_weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    values = tf.convert_to_tensor(samples, dtype=tf.float64)
    weights = tf.exp(normalize_log_weights(log_weights))
    if values.shape.rank != 2 or weights.shape != (int(values.shape[1]),):
        raise ValueError(f"resample inputs: {HighDimStatus.INVALID_SHAPE.value}")
    n = int(values.shape[1])
    positions = (
        tf.cast(tf.range(n), tf.float64) + tf.constant(0.5, dtype=tf.float64)
    ) / tf.cast(n, tf.float64)
    cdf = tf.cumsum(weights)
    cdf = tf.concat(
        [cdf[:-1], tf.ones([1], dtype=tf.float64)],
        axis=0,
    )
    indices = tf.searchsorted(cdf, positions, side="left", out_type=tf.int32)
    indices = tf.minimum(indices, tf.fill([n], n - 1))
    return tf.gather(values, indices, axis=1), indices


def _p59_author_sir_source_fit_data_for_step(
    *,
    model,
    observations: tf.Tensor,
    time_index: int,
    fit_sample_count: int,
    previous_retained_object: SourceRouteRetainedObject | None = None,
) -> _P59AuthorSIRSourceFitData:
    t = int(time_index)
    n = int(fit_sample_count)
    if t < 1:
        raise ValueError("time_index must be positive")
    if n < 2:
        raise ValueError("fit_sample_count must be at least 2")
    d = model.parameter_dim()
    m = model.state_dim()
    if t == 1:
        previous_batch = _p59_author_sir_prior_sample_batch(
            model=model,
            sample_count=n,
            seed=6301,
        )
        previous_for_density = None
    else:
        if previous_retained_object is None:
            raise TypeError("t>1 requires previous_retained_object")
        previous_batch = _p59_author_sir_batch_from_previous_retained_prefix(
            previous_retained_object,
            parameter_dim=d,
            state_dim=m,
            time_index=t - 1,
        )
        previous_for_density = previous_retained_object
    if previous_batch.sample_count != n:
        raise ValueError("source_fit_data_previous_batch_count_mismatch")
    push = _p59_author_sir_source_push_result(
        model=model,
        previous_batch=previous_batch,
        observation=tf.convert_to_tensor(observations[t], dtype=tf.float64),
        time_index=t,
        process_noise_seed=6400 + t,
    )
    frame = source_route_recenter(
        samples=push.augmented_batch.samples,
        log_weights=push.augmented_batch.log_weights,
        expansion_factor=P63_AUTHOR_SIR_EXPANSION_FACTOR,
        covariance_jitter=1e-5,
        use_quantile_scale=True,
    )
    resampled, resample_indices = _p59_author_sir_deterministic_weighted_resample(
        samples=push.augmented_batch.samples,
        log_weights=push.augmented_batch.log_weights,
    )
    local_unclipped = tf.linalg.solve(
        frame.matrix,
        resampled - frame.mu[:, tf.newaxis],
    )
    clipped_mask = tf.abs(local_unclipped) > tf.constant(1.0, dtype=tf.float64)
    clip_fraction = tf.reduce_mean(tf.cast(clipped_mask, tf.float64))
    if bool((clip_fraction >= 1.0).numpy()):
        raise ValueError("source_fit_data_all_local_entries_clipped")
    local_fit_points = tf.clip_by_value(local_unclipped, -1.0, 1.0)
    physical_fit_points = (
        tf.linalg.matmul(frame.matrix, local_fit_points)
        + frame.mu[:, tf.newaxis]
    )
    prior_log_density, transition_log_density, likelihood_log_density = (
        _p59_author_sir_source_density_callbacks(model, observations[t])
    )
    components = SourceRouteSequentialDensityComponents(
        parameter_dim=d,
        state_dim=m,
        transition_log_density_fn=transition_log_density,
        likelihood_log_density_fn=likelihood_log_density,
        prior_log_density_fn=prior_log_density if t == 1 else None,
    )
    negative_log_physical = components.negative_log_physical_density(
        physical_points=physical_fit_points,
        time_index=t,
        previous_retained_object=previous_for_density,
    )
    local_negative_log = negative_log_physical - frame.log_abs_det()
    shift = tf.reduce_min(local_negative_log)
    shifted = source_route_shifted_negative_log_target(
        negative_log_target=local_negative_log,
        shift_constant=shift,
    )
    target_values = tf.exp(-0.5 * shifted)
    if not bool(tf.reduce_all(tf.math.is_finite(target_values)).numpy()):
        raise ValueError("nonfinite_source_fit_target_values")
    manifest = {
        "fit_data_mode": P63_AUTHOR_SIR_SOURCE_FIT_DATA_MODE,
        "fit_data_source": "pushed_weighted_augmented_samples",
        "coordinate_frame_source": "source_computeL_weighted_augmented_samples",
        "fixed_variant_resampling": P63_AUTHOR_SIR_FIXED_VARIANT_RESAMPLING,
        "source_anchors": _P63_AUTHOR_SIR_FIT_DATA_SOURCE_ANCHORS,
        "time_index": t,
        "fit_sample_count": n,
        "source_push_sample_origin": push.augmented_batch.sample_origin,
        "source_push_ess": push.diagnostics.effective_sample_size,
        "source_push_weight_policy": (
            "uniform previous weights then observation likelihood update"
        ),
        "resample_indices": resample_indices,
        "expansion_factor": P63_AUTHOR_SIR_EXPANSION_FACTOR,
        "coordinate_frame_log_abs_det": frame.log_abs_det(),
        "coordinate_frame_hash": _p69_frame_hash(frame),
        "local_clip_fraction": clip_fraction,
        "local_max_abs_before_clip": tf.reduce_max(tf.abs(local_unclipped)),
        "target_value_min": tf.reduce_min(target_values),
        "target_value_max": tf.reduce_max(target_values),
        "nonclaims": (
            "deterministic fixed-variant replacement for MATLAB random datasample",
            "bounded local basis clips fit data when needed",
            "no AlgebraicMapping(1) parity claim",
            "no adaptive Zhao-Cui parity claim",
        ),
    }
    return _P59AuthorSIRSourceFitData(
        time_index=t,
        frame=frame,
        local_fit_points=local_fit_points,
        target_values=target_values,
        negative_log_values=local_negative_log,
        shift_constant=shift,
        fit_weights=tf.ones([n], dtype=tf.float64),
        manifest=manifest,
    )


def _p69_author_sir_source_diagnostic_data_for_step(
    *,
    model,
    observations: tf.Tensor,
    time_index: int,
    diagnostic_sample_count: int,
    frame: SourceRouteCoordinateFrame,
    shift_constant: tf.Tensor,
    previous_retained_object: SourceRouteRetainedObject | None = None,
    prior_seed: int = 7301,
    process_noise_seed: int = 7401,
    construction: str,
) -> _P59AuthorSIRSourceFitData:
    t = int(time_index)
    n = int(diagnostic_sample_count)
    if t < 1:
        raise ValueError("time_index must be positive")
    if n < 2:
        raise ValueError("diagnostic_sample_count must be at least 2")
    if not isinstance(frame, SourceRouteCoordinateFrame):
        raise TypeError("frame must be SourceRouteCoordinateFrame")
    d = model.parameter_dim()
    m = model.state_dim()
    if t == 1:
        previous_batch = _p59_author_sir_prior_sample_batch(
            model=model,
            sample_count=n,
            seed=int(prior_seed),
        )
        previous_for_density = None
    else:
        if previous_retained_object is None:
            raise TypeError("t>1 requires previous_retained_object")
        previous_batch = _p59_author_sir_batch_from_previous_retained_prefix(
            previous_retained_object,
            parameter_dim=d,
            state_dim=m,
            time_index=t - 1,
        )
        previous_for_density = previous_retained_object
        if previous_batch.sample_count != n:
            raise ValueError("diagnostic_previous_batch_count_mismatch")
    push = _p59_author_sir_source_push_result(
        model=model,
        previous_batch=previous_batch,
        observation=tf.convert_to_tensor(observations[t], dtype=tf.float64),
        time_index=t,
        process_noise_seed=int(process_noise_seed),
    )
    resampled, resample_indices = _p59_author_sir_deterministic_weighted_resample(
        samples=push.augmented_batch.samples,
        log_weights=push.augmented_batch.log_weights,
    )
    local_unclipped = tf.linalg.solve(
        frame.matrix,
        resampled - frame.mu[:, tf.newaxis],
    )
    clipped_mask = tf.abs(local_unclipped) > tf.constant(1.0, dtype=tf.float64)
    clip_fraction = tf.reduce_mean(tf.cast(clipped_mask, tf.float64))
    if bool((clip_fraction >= 1.0).numpy()):
        raise ValueError("diagnostic_data_all_local_entries_clipped")
    local_points = tf.clip_by_value(local_unclipped, -1.0, 1.0)
    physical_points = (
        tf.linalg.matmul(frame.matrix, local_points)
        + frame.mu[:, tf.newaxis]
    )
    prior_log_density, transition_log_density, likelihood_log_density = (
        _p59_author_sir_source_density_callbacks(model, observations[t])
    )
    components = SourceRouteSequentialDensityComponents(
        parameter_dim=d,
        state_dim=m,
        transition_log_density_fn=transition_log_density,
        likelihood_log_density_fn=likelihood_log_density,
        prior_log_density_fn=prior_log_density if t == 1 else None,
    )
    negative_log_physical = components.negative_log_physical_density(
        physical_points=physical_points,
        time_index=t,
        previous_retained_object=previous_for_density,
    )
    local_negative_log = negative_log_physical - frame.log_abs_det()
    shifted = source_route_shifted_negative_log_target(
        negative_log_target=local_negative_log,
        shift_constant=shift_constant,
    )
    target_values = tf.exp(-0.5 * shifted)
    if not bool(tf.reduce_all(tf.math.is_finite(target_values)).numpy()):
        raise ValueError("nonfinite_source_diagnostic_target_values")
    manifest = {
        "fit_data_mode": P63_AUTHOR_SIR_SOURCE_FIT_DATA_MODE,
        "diagnostic_role": "post_fit_diagnostic_only",
        "diagnostic_classification": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "diagnostic_construction": str(construction),
        "fit_data_source": "pushed_weighted_augmented_samples",
        "coordinate_frame_source": "reuse_fitted_source_computeL_frame",
        "fixed_variant_resampling": P63_AUTHOR_SIR_FIXED_VARIANT_RESAMPLING,
        "source_anchors": _P63_AUTHOR_SIR_FIT_DATA_SOURCE_ANCHORS,
        "time_index": t,
        "diagnostic_sample_count": n,
        "diagnostic_prior_seed": int(prior_seed) if t == 1 else None,
        "diagnostic_process_noise_seed": int(process_noise_seed),
        "source_push_sample_origin": push.augmented_batch.sample_origin,
        "source_push_ess": push.diagnostics.effective_sample_size,
        "resample_indices": resample_indices,
        "coordinate_frame_log_abs_det": frame.log_abs_det(),
        "coordinate_frame_hash": _p69_frame_hash(frame),
        "shift_constant_source": "reuse_fitted_branch_shift",
        "shift_constant": tf.convert_to_tensor(shift_constant, dtype=tf.float64),
        "local_clip_fraction": clip_fraction,
        "local_max_abs_before_clip": tf.reduce_max(tf.abs(local_unclipped)),
        "target_value_min": tf.reduce_min(target_values),
        "target_value_max": tf.reduce_max(target_values),
        "nonclaims": (
            "post-fit diagnostic only",
            "not passed into FixedTTFitter.fit",
            "no adaptive Zhao-Cui parity claim",
        ),
    }
    return _P59AuthorSIRSourceFitData(
        time_index=t,
        frame=frame,
        local_fit_points=local_points,
        target_values=target_values,
        negative_log_values=local_negative_log,
        shift_constant=tf.convert_to_tensor(shift_constant, dtype=tf.float64),
        fit_weights=tf.ones([n], dtype=tf.float64),
        manifest=manifest,
    )


def _p71_optional_author_sir_diagnostic_data_for_step(
    **kwargs,
) -> _P59AuthorSIRSourceFitData | None:
    try:
        return _p69_author_sir_source_diagnostic_data_for_step(**kwargs)
    except ValueError as exc:
        if str(exc) != "diagnostic_data_all_local_entries_clipped":
            raise
        return None


def _p59_author_sir_36d_coordinate_frame_for_time(
    model,
    *,
    time_index: int,
) -> SourceRouteCoordinateFrame:
    if int(time_index) < 1:
        raise ValueError("time_index must be positive")
    previous = model.initial_mean
    current = model.transition_mean(previous)[0]
    for _ in range(2, int(time_index) + 1):
        previous = current
        current = model.transition_mean(previous)[0]
    center = tf.concat([model.transition_mean(model.initial_mean)[0], model.initial_mean], axis=0)
    if int(time_index) != 1:
        center = tf.concat([current, previous], axis=0)
    scale = tf.concat(
        [
            tf.sqrt(tf.linalg.diag_part(model.process_covariance)),
            tf.sqrt(tf.linalg.diag_part(model.initial_covariance)),
        ],
        axis=0,
    )
    matrix = tf.linalg.diag(scale)
    return SourceRouteCoordinateFrame(
        mu=center,
        matrix=matrix,
        expansion_factor=1.0,
    )


def _p59_author_sir_reference_points(sample_count: int, target_dim: int) -> tf.Tensor:
    columns = []
    base = tf.linspace(
        tf.constant(-0.75, dtype=tf.float64),
        tf.constant(0.75, dtype=tf.float64),
        int(target_dim),
    )
    for index in range(int(sample_count)):
        shift = tf.cast(index, tf.float64) / tf.cast(max(int(sample_count) - 1, 1), tf.float64)
        shifted = tf.math.floormod(base + 0.17 * shift + 1.0, 2.0) - 1.0
        columns.append(shifted)
    return tf.stack(columns, axis=1)


def _p59_reference_convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _p59_9a_block_result(
    *,
    blockers: tuple[str, ...],
    target_dimension: int,
    state_dimension: int,
    parameter_dimension: int,
    negative_log_values: tf.Tensor | None = None,
    shift_constant: tf.Tensor | None = None,
    fit_status: str = "not_attempted",
    fit_branch_hash: str | None = None,
) -> P59AuthorSIR36DTargetFitPrepResult:
    sample_values = (
        tf.zeros([1], dtype=tf.float64)
        if negative_log_values is None
        else tf.convert_to_tensor(negative_log_values, dtype=tf.float64)
    )
    shift = (
        tf.constant(0.0, dtype=tf.float64)
        if shift_constant is None
        else tf.convert_to_tensor(shift_constant, dtype=tf.float64)
    )
    return P59AuthorSIR36DTargetFitPrepResult(
        status=P59_9A_BLOCK_STATUS,
        blockers=tuple(blockers),
        target_dimension=int(target_dimension),
        state_dimension=int(state_dimension),
        parameter_dimension=int(parameter_dimension),
        time_index=1,
        sample_count=int(sample_values.shape[0]),
        negative_log_values=sample_values,
        shift_constant=shift,
        fit_status=str(fit_status),
        fit_branch_hash=fit_branch_hash,
        density_branch_hash=None,
        transport_manifest=None,
        manifest={
            "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
            "pipeline_phase": "P59-9a",
            "artifact_role": "blocked_preparation_evidence",
            "blockers": tuple(blockers),
            "target_dimension": int(target_dimension),
            "state_dimension": int(state_dimension),
            "parameter_dimension": int(parameter_dimension),
            "nonclaims": (
                "no Phase-9 validation launch",
                "no d18 filtering accuracy claim",
            ),
        },
    )


def p59_author_sir_step_spec_assembly(
    *,
    route_decision: P59AuthorSIRRouteDecisionResult | None = None,
    prep_result: P59AuthorSIR36DTargetFitPrepResult | None = None,
    sample_count: int = 3,
    fit_sample_count: int = 6,
    fit_degree: int = 0,
    fit_rank: int = 1,
    ridge: float = 1e-8,
) -> P59AuthorSIRStepSpecAssemblyResult:
    """Assemble bounded two-step P59-9b author-SIR source-route specs."""

    if route_decision is not None and not isinstance(
        route_decision,
        P59AuthorSIRRouteDecisionResult,
    ):
        raise TypeError("route_decision must be P59AuthorSIRRouteDecisionResult")
    if prep_result is not None and not isinstance(
        prep_result,
        P59AuthorSIR36DTargetFitPrepResult,
    ):
        raise TypeError("prep_result must be P59AuthorSIR36DTargetFitPrepResult")

    route = route_decision or p59_author_sir_route_decision()
    prep = prep_result or p59_author_sir_36d_target_fit_prep(
        sample_count=int(fit_sample_count),
        fit_degree=int(fit_degree),
        fit_rank=int(fit_rank),
        ridge=float(ridge),
    )
    blockers: list[str] = []
    if route.status != P59_9C_PASS_STATUS:
        blockers.append("missing_p59_9c_route_decision_pass")
    if route.route_decision != P59_9C_FULL_ROUTE_SELECTED:
        blockers.append("p59_9c_route_not_full_route_selected")
    if route.preconditioned_route_required:
        blockers.append("p59_9b_rewrite_required_for_preconditioned_route")
    if prep.status != P59_9A_PASS_STATUS:
        blockers.append("missing_p59_9a_target_fit_prep_pass")
    if prep.target_dimension != P59_9A_AUTHOR_SIR_TARGET_DIMENSION:
        blockers.append("p59_9a_target_dimension_not_36")
    if prep.transport_manifest is None:
        blockers.append("missing_p59_9a_fixed_ttsirt_transport_manifest")
    elif prep.transport_manifest.get("source_contract_level") != "fixed_ttsirt":
        blockers.append("p59_9a_transport_not_fixed_ttsirt")
    if int(sample_count) <= 0:
        blockers.append("sample_count_must_be_positive")
    if int(fit_sample_count) < 2:
        blockers.append("fit_sample_count_must_be_at_least_2")
    if int(fit_degree) < 0:
        blockers.append("fit_degree_must_be_nonnegative")
    if int(fit_rank) <= 0:
        blockers.append("fit_rank_must_be_positive")
    if blockers:
        return _p59_9b_block_result(
            blockers=tuple(blockers),
            route_decision=route,
            prep_result=prep,
        )

    model = zhao_cui_sir_austria_model()
    d = model.parameter_dim()
    m = model.state_dim()
    target_dim = d + 2 * m
    observations = model.simulate(final_time=2, seed=5901)[1]
    retained_reference = _p59_author_sir_unit_reference_points(
        int(sample_count),
        target_dim,
    )
    convention = _p59_reference_convention()

    prior_log_density, transition_log_density, likelihood_t1 = (
        _p59_author_sir_source_density_callbacks(model, observations[1])
    )
    _, _, likelihood_t2 = _p59_author_sir_source_density_callbacks(model, observations[2])
    fit_data1 = _p59_author_sir_source_fit_data_for_step(
        model=model,
        observations=observations,
        time_index=1,
        fit_sample_count=int(fit_sample_count),
    )
    frame1 = fit_data1.frame
    shift1 = fit_data1.shift_constant
    holdout_data1 = _p71_optional_author_sir_diagnostic_data_for_step(
        model=model,
        observations=observations,
        time_index=1,
        diagnostic_sample_count=int(fit_sample_count),
        frame=frame1,
        shift_constant=shift1,
        prior_seed=7301,
        process_noise_seed=7401,
        construction="p69_step1_holdout_distinct_diagnostic_seed",
    )
    replay_data1 = _p71_optional_author_sir_diagnostic_data_for_step(
        model=model,
        observations=observations,
        time_index=1,
        diagnostic_sample_count=int(fit_sample_count),
        frame=frame1,
        shift_constant=shift1,
        prior_seed=7311,
        process_noise_seed=7501,
        construction="p69_step1_replay_distinct_diagnostic_seed",
    )
    transport1, fit_hash1, density_hash1, fit_quality1, holdout_replay1 = _p59_fixed_ttsirt_transport_from_values(
        local_fit_points=fit_data1.local_fit_points,
        target_values=fit_data1.target_values,
        fit_weights=fit_data1.fit_weights,
        fit_data_manifest=fit_data1.manifest,
        holdout_local_points=None if holdout_data1 is None else holdout_data1.local_fit_points,
        holdout_target_values=None if holdout_data1 is None else holdout_data1.target_values,
        holdout_weights=None if holdout_data1 is None else holdout_data1.fit_weights,
        holdout_manifest=None if holdout_data1 is None else holdout_data1.manifest,
        replay_local_points=None if replay_data1 is None else replay_data1.local_fit_points,
        replay_target_values=None if replay_data1 is None else replay_data1.target_values,
        replay_weights=None if replay_data1 is None else replay_data1.fit_weights,
        replay_manifest=None if replay_data1 is None else replay_data1.manifest,
        target_dim=target_dim,
        fit_degree=int(fit_degree),
        fit_rank=int(fit_rank),
        ridge=float(ridge),
        branch_seed="p59-9b-author-sir-step1-fixed-ttsirt",
        convention=convention,
    )
    components1 = SourceRouteSequentialDensityComponents(
        parameter_dim=d,
        state_dim=m,
        transition_log_density_fn=transition_log_density,
        likelihood_log_density_fn=likelihood_t1,
        prior_log_density_fn=prior_log_density,
    )
    components2 = SourceRouteSequentialDensityComponents(
        parameter_dim=d,
        state_dim=m,
        transition_log_density_fn=transition_log_density,
        likelihood_log_density_fn=likelihood_t2,
        prior_log_density_fn=None,
    )
    spec1 = SourceRouteSequentialStepSpec(
        target=_p59_author_sir_source_route_target(
            frame=frame1,
            shift_constant=shift1,
            time_index=1,
            components=components1,
        ),
        transport=SourceRouteTransportProtocol(transport1),
        reference_samples=retained_reference,
        measure_convention=convention,
        density_components=components1,
    )
    retained1 = _p59_retained_object_from_spec(spec1)
    fit_retained_reference = (
        retained_reference
        if int(fit_sample_count) == int(sample_count)
        else _p59_author_sir_unit_reference_points(
            int(fit_sample_count),
            target_dim,
        )
    )
    fit_spec1 = SourceRouteSequentialStepSpec(
        target=spec1.target,
        transport=spec1.transport,
        reference_samples=fit_retained_reference,
        measure_convention=convention,
        density_components=components1,
    )
    fit_retained1 = (
        retained1
        if int(fit_sample_count) == int(sample_count)
        else _p59_retained_object_from_spec(fit_spec1)
    )
    fit_data2 = _p59_author_sir_source_fit_data_for_step(
        model=model,
        observations=observations,
        time_index=2,
        fit_sample_count=int(fit_sample_count),
        previous_retained_object=fit_retained1,
    )
    frame2 = fit_data2.frame
    shift2 = fit_data2.shift_constant
    holdout_data2 = _p71_optional_author_sir_diagnostic_data_for_step(
        model=model,
        observations=observations,
        time_index=2,
        diagnostic_sample_count=int(fit_sample_count),
        frame=frame2,
        shift_constant=shift2,
        previous_retained_object=fit_retained1,
        process_noise_seed=7402,
        construction="p69_step2_holdout_distinct_diagnostic_seed",
    )
    replay_data2 = _p71_optional_author_sir_diagnostic_data_for_step(
        model=model,
        observations=observations,
        time_index=2,
        diagnostic_sample_count=int(fit_sample_count),
        frame=frame2,
        shift_constant=shift2,
        previous_retained_object=fit_retained1,
        process_noise_seed=7502,
        construction="p69_step2_replay_distinct_diagnostic_seed",
    )
    transport2, fit_hash2, density_hash2, fit_quality2, holdout_replay2 = _p59_fixed_ttsirt_transport_from_values(
        local_fit_points=fit_data2.local_fit_points,
        target_values=fit_data2.target_values,
        fit_weights=fit_data2.fit_weights,
        fit_data_manifest=fit_data2.manifest,
        holdout_local_points=None if holdout_data2 is None else holdout_data2.local_fit_points,
        holdout_target_values=None if holdout_data2 is None else holdout_data2.target_values,
        holdout_weights=None if holdout_data2 is None else holdout_data2.fit_weights,
        holdout_manifest=None if holdout_data2 is None else holdout_data2.manifest,
        replay_local_points=None if replay_data2 is None else replay_data2.local_fit_points,
        replay_target_values=None if replay_data2 is None else replay_data2.target_values,
        replay_weights=None if replay_data2 is None else replay_data2.fit_weights,
        replay_manifest=None if replay_data2 is None else replay_data2.manifest,
        target_dim=target_dim,
        fit_degree=int(fit_degree),
        fit_rank=int(fit_rank),
        ridge=float(ridge),
        branch_seed="p59-9b-author-sir-step2-fixed-ttsirt",
        convention=convention,
    )
    previous_keep_axes = tuple(range(d + m))
    previous_input_axes = tuple(range(d + m, d + 2 * m))
    spec2 = SourceRouteSequentialStepSpec(
        target=_p59_author_sir_placeholder_target(
            frame=frame2,
            shift_constant=shift2,
            time_index=2,
        ),
        transport=SourceRouteTransportProtocol(transport2),
        reference_samples=retained_reference,
        measure_convention=convention,
        density_components=components2,
        previous_marginal_keep_axes=previous_keep_axes,
        previous_marginal_input_axes=previous_input_axes,
    )
    sequential = source_route_run_sequential_fixed_hmc(step_specs=(spec1, spec2))
    previous_marginal_present = sequential.steps[1].previous_marginal_density is not None
    if not previous_marginal_present:
        return _p59_9b_block_result(
            blockers=("missing_time2_previous_marginal_evidence",),
            route_decision=route,
            prep_result=prep,
            step_specs=(spec1, spec2),
        )

    p58_partial_manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_kind": P58_M9_SOURCE_ROUTE_PIPELINE_KIND,
        "route_class": "fixed_ttsirt_source_route",
        "storage_kind": "source_transport_object",
        "transition_interface": "sample_propagation",
        "m9_comparator_tier": "d18_execution_only",
        "rank_policy_status": "PASS_P57_M7_SOURCE_FAITHFUL_RANK_UKF_CALIBRATION",
        "preconditioned_route_required": False,
        "preconditioned_route_status": route.preconditioned_route_status,
        "uses_contract_test_double": False,
        "uses_ukf_as_comparator": False,
        "uses_rank_memory_proxy_as_comparator": False,
        "has_author_sir_callback": True,
        "has_fixed_ttsirt_fit_artifacts": True,
        "has_fixed_ttsirt_transports": True,
        "has_frozen_reference_samples": True,
        "has_source_route_step_specs": True,
        "has_sequential_retained_carry": True,
        "has_previous_marginal_evidence": True,
        "has_m9_runner_manifest_path": False,
    }
    p58_status = p58_m9_source_route_pipeline_readiness(p58_partial_manifest)
    manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P59-9b",
        "artifact_role": "bounded_source_route_step_spec_assembly",
        "route_decision_status": route.status,
        "route_decision": route.route_decision,
        "p59_9a_status": prep.status,
        "step_count": 2,
        "target_dimension": target_dim,
        "state_dimension": m,
        "parameter_dimension": d,
        "source_target_order": "[theta, x_t, x_{t-1}]",
        "fit_data_mode": P63_AUTHOR_SIR_SOURCE_FIT_DATA_MODE,
        "fit_data_manifests": (fit_data1.manifest, fit_data2.manifest),
        "previous_marginal_keep_axes": previous_keep_axes,
        "previous_marginal_input_axes": previous_input_axes,
        "previous_marginal_axis_interpretation": (
            "keep previous retained prefix [theta, x_{t-1}] and feed current "
            "target suffix coordinates x_{t-1}"
        ),
        "transport_families": (
            spec1.transport.manifest_payload()["transport_object"]["family"],
            spec2.transport.manifest_payload()["transport_object"]["family"],
        ),
        "transport_source_contract_levels": (
            spec1.transport.manifest_payload()["source_contract_level"],
            spec2.transport.manifest_payload()["source_contract_level"],
        ),
        "fit_branch_hashes": (fit_hash1, fit_hash2),
        "density_branch_hashes": (density_hash1, density_hash2),
        "fit_quality_diagnostics_by_step": (fit_quality1, fit_quality2),
        "holdout_replay_diagnostics_by_step": (holdout_replay1, holdout_replay2),
        "defensive_tau": P62_AUTHOR_TTSIRT_EXECUTABLE_DEFAULT_TAU,
        "defensive_tau_source": "author_executable_ttsirt_default",
        "source_declared_tau_unwired": 10.0,
        "source_executable_ttsirt_default_tau": P62_AUTHOR_TTSIRT_EXECUTABLE_DEFAULT_TAU,
        "defensive_tau_source_anchors": _P62_AUTHOR_TTSIRT_DEFENSIVE_TAU_SOURCE_ANCHORS,
        "fixed_branch_adaptation_class": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "fit_initialization_rule": P70_FIXED_BRANCH_INITIALIZATION_RULE,
        "fit_initialization_rule_source": "docs/plans/bayesfilter-highdim-zhao-cui-p70-phase4-nondegenerate-fitting-design-result-2026-06-16.md",
        "fit_degree": int(fit_degree),
        "fit_rank": int(fit_rank),
        "rank_tuple": _source_route_rank_tuple(target_dim, int(fit_rank)),
        "sequential_status": sequential.sequential_status,
        "time2_previous_marginal_present": previous_marginal_present,
        "source_anchors": _P59_9B_AUTHOR_SIR_SOURCE_ANCHORS,
        "p58_partial_manifest": p58_partial_manifest,
        "p58_readiness_status_after_9b": p58_status.status,
        "p58_expected_remaining_blocker": "missing_has_m9_runner_manifest_path",
        "bounded_transport_note": (
            "Step specs use bounded fixed-TTSIRT transports to prove assembly and "
            "previous-marginal wiring; this is not rank convergence or validation."
        ),
        "nonclaims": (
            "no Phase-9 validation launch",
            "no d18 filtering accuracy claim",
            "no same-route rank convergence claim",
            "no d50 or d100 scaling claim",
            "no HMC production readiness claim",
            "no adaptive Zhao-Cui parity claim",
        ),
    }
    return P59AuthorSIRStepSpecAssemblyResult(
        status=P59_9B_PASS_STATUS,
        blockers=(),
        route_decision=route,
        prep_result=prep,
        step_specs=(spec1, spec2),
        sequential_result=sequential,
        manifest=manifest,
    )


def _p59_9b_block_result(
    *,
    blockers: tuple[str, ...],
    route_decision: P59AuthorSIRRouteDecisionResult,
    prep_result: P59AuthorSIR36DTargetFitPrepResult | None,
    step_specs: tuple[SourceRouteSequentialStepSpec, ...] = (),
) -> P59AuthorSIRStepSpecAssemblyResult:
    return P59AuthorSIRStepSpecAssemblyResult(
        status=P59_9B_BLOCK_STATUS,
        blockers=tuple(blockers),
        route_decision=route_decision,
        prep_result=prep_result,
        step_specs=tuple(step_specs),
        sequential_result=None,
        manifest={
            "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
            "pipeline_phase": "P59-9b",
            "artifact_role": "blocked_source_route_step_spec_assembly",
            "blockers": tuple(blockers),
            "route_decision": route_decision.route_decision,
            "route_decision_status": route_decision.status,
            "p59_9a_status": None if prep_result is None else prep_result.status,
            "fail_closed_rule": "P59-9d and P59-9e must not proceed.",
            "nonclaims": (
                "no Phase-9 validation launch",
                "no d18 filtering accuracy claim",
            ),
        },
    )


def _p59_retained_object_from_spec(spec: SourceRouteSequentialStepSpec) -> SourceRouteRetainedObject:
    retained = source_route_generate_retained_samples(
        target=spec.target,
        transport=spec.transport,
        reference_samples=spec.reference_samples,
        time_index=spec.time_index,
    )
    diagnostics = {
        "phase": "P59-9b-preassembly-step1",
        "time_index": int(spec.time_index),
        "source_anchor": "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:33-38",
        "purpose": "materialize previous retained object before fitting bounded time-2 target",
    }
    identity = source_route_retained_object_identity(
        transport_object=spec.transport.transport_object,
        coordinate_frame=spec.target.coordinate_frame,
        samples=retained.retained_batch.samples,
        log_weights=retained.retained_batch.log_weights,
        sample_diagnostics=retained.diagnostics,
        normalizer=retained.normalizer,
        measure_convention=spec.measure_convention,
        route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
        storage_kind=spec.storage_kind,
        transition_interface=spec.transition_interface,
        diagnostics=diagnostics,
    )
    return SourceRouteRetainedObject(
        transport_object=spec.transport.transport_object,
        coordinate_frame=spec.target.coordinate_frame,
        samples=retained.retained_batch.samples,
        log_weights=retained.retained_batch.log_weights,
        sample_diagnostics=retained.diagnostics,
        normalizer=retained.normalizer,
        measure_convention=spec.measure_convention,
        route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
        storage_kind=spec.storage_kind,
        transition_interface=spec.transition_interface,
        branch_identity=identity,
        diagnostics=diagnostics,
    )


def _p59_author_sir_source_route_target(
    *,
    frame: SourceRouteCoordinateFrame,
    shift_constant: tf.Tensor,
    time_index: int,
    components: SourceRouteSequentialDensityComponents,
) -> SourceRouteTarget:
    def negative_log_physical(points: tf.Tensor) -> tf.Tensor:
        return components.negative_log_physical_density(
            physical_points=points,
            time_index=int(time_index),
            previous_retained_object=None,
        )

    return build_source_route_target(
        negative_log_physical_density_fn=negative_log_physical,
        coordinate_frame=frame,
        shift_constant=shift_constant,
        time_index=int(time_index),
    )


def _p59_author_sir_placeholder_target(
    *,
    frame: SourceRouteCoordinateFrame,
    shift_constant: tf.Tensor,
    time_index: int,
) -> SourceRouteTarget:
    def negative_log_physical(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank != 2:
            raise ValueError(f"physical_points: {HighDimStatus.INVALID_SHAPE.value}")
        return tf.zeros([int(values.shape[1])], dtype=tf.float64)

    return build_source_route_target(
        negative_log_physical_density_fn=negative_log_physical,
        coordinate_frame=frame,
        shift_constant=shift_constant,
        time_index=int(time_index),
    )


def _p59_fixed_ttsirt_transport_from_values(
    *,
    local_fit_points: tf.Tensor,
    target_values: tf.Tensor,
    fit_weights: tf.Tensor | None = None,
    fit_data_manifest: Mapping[str, object] | None = None,
    holdout_local_points: tf.Tensor | None = None,
    holdout_target_values: tf.Tensor | None = None,
    holdout_weights: tf.Tensor | None = None,
    holdout_manifest: Mapping[str, object] | None = None,
    replay_local_points: tf.Tensor | None = None,
    replay_target_values: tf.Tensor | None = None,
    replay_weights: tf.Tensor | None = None,
    replay_manifest: Mapping[str, object] | None = None,
    target_dim: int,
    fit_degree: int,
    fit_rank: int,
    ridge: float,
    branch_seed: str,
    convention: MeasureConvention,
) -> tuple[FixedTTSIRTTransport, str, str, Mapping[str, object], Mapping[str, object]]:
    product_basis = ProductBasis(
        [
            LegendreBasis1D(BoundedInterval(-1.0, 1.0), int(fit_degree))
            for _ in range(int(target_dim))
        ],
        convention,
    )
    fit_config = FixedTTFitConfig(
        ranks=_source_route_rank_tuple(int(target_dim), int(fit_rank)),
        ridge=float(ridge),
        max_sweeps=P70_FIXED_BRANCH_MAX_SWEEPS,
        sweep_order=_p70_canonical_alternating_sweep_order(int(target_dim)),
        row_budget=max(int(tf.shape(local_fit_points)[1]), 1),
        column_budget=(int(fit_degree) + 1) * int(fit_rank) * int(fit_rank),
        dense_matrix_byte_budget=2**22,
        normal_matrix_byte_budget=2**24,
        condition_number_warning=P70_CONDITION_NUMBER_WARNING,
        condition_number_veto=P70_CONDITION_NUMBER_VETO,
        holdout_tolerance=1e6,
    )
    weights = (
        tf.ones([int(tf.shape(local_fit_points)[1])], dtype=tf.float64)
        if fit_weights is None
        else tf.convert_to_tensor(fit_weights, dtype=tf.float64)
    )
    target_tensor = tf.convert_to_tensor(target_values, dtype=tf.float64)
    row_adequacy = _p70_row_adequacy_diagnostics(
        row_count=int(tf.shape(local_fit_points)[1]),
        target_dim=int(target_dim),
        fit_degree=int(fit_degree),
        fit_rank=int(fit_rank),
    )
    if row_adequacy["status"] == "branch_fit_row_adequacy_failed":
        raise ValueError("branch_fit_row_adequacy_failed")
    initial_cores = _source_route_seeded_channel_initial_cores(
        ranks=fit_config.ranks,
        basis_dim=int(fit_degree) + 1,
        constant_value=_weighted_mean_target_value(target_tensor, weights),
    )
    fit_result = FixedTTFitter().fit(
        product_basis=product_basis,
        samples=FixedTTFitSampleBatch(
            points=tf.transpose(tf.convert_to_tensor(local_fit_points, dtype=tf.float64)),
            target_values=target_tensor,
            weights=weights,
        ),
        config=fit_config,
        initial_cores=initial_cores,
        branch_seed=str(branch_seed),
        measure_convention=convention,
        initialization_rule=P70_FIXED_BRANCH_INITIALIZATION_RULE,
    )
    channel_activity = _p70_channel_activity_diagnostics(
        cores=fit_result.fitted_tt.cores,
        target_dim=int(target_dim),
        fit_rank=int(fit_rank),
    )
    fit_quality_diagnostics = dict(_p59_fixed_ttsirt_fit_quality_diagnostics(fit_result))
    fit_diagnostics = {} if fit_result.diagnostics is None else dict(fit_result.diagnostics)
    fit_quality_diagnostics["p70_fixed_fitting_policy"] = _p70_fixed_fitting_policy_payload(
        target_dim=int(target_dim),
        fit_degree=int(fit_degree),
        fit_rank=int(fit_rank),
        ridge=float(ridge),
        row_adequacy=row_adequacy,
        channel_activity=channel_activity,
        stabilization_policy=fit_diagnostics.get("stabilization_policy"),
        stabilization_diagnostics_summary=fit_diagnostics.get(
            "stabilization_diagnostics_summary"
        ),
    )
    if fit_result.status is not HighDimStatus.OK:
        message = f"fixed_ttsirt_fit_status_{fit_result.status.value}"
        raise P70FixedFitDiagnosticError(
            message,
            status=fit_result.status,
            payload={
                "message": message,
                "fit_status": fit_result.status.value,
                "termination_reason": fit_result.termination_reason,
                "stop_condition_triggered": fit_result.stop_condition_triggered,
                "fit_quality_diagnostics": fit_quality_diagnostics,
                "core_update_statuses": tuple(dict(record) for record in fit_result.core_update_statuses),
                "fit_branch_hash": fit_result.branch_hash.value,
                "rank_tuple": fit_config.ranks,
                "fit_degree": int(fit_degree),
                "fit_rank": int(fit_rank),
                "target_dim": int(target_dim),
                "initialization_rule": P70_FIXED_BRANCH_INITIALIZATION_RULE,
                "ridge": float(ridge),
                "max_sweeps": P70_FIXED_BRANCH_MAX_SWEEPS,
                "sweep_order": _p70_canonical_alternating_sweep_order(int(target_dim)),
                "condition_number_warning": P70_CONDITION_NUMBER_WARNING,
                "condition_number_veto": P70_CONDITION_NUMBER_VETO,
                "p70_fixed_fitting_policy": fit_quality_diagnostics["p70_fixed_fitting_policy"],
                "stabilization_policy": fit_diagnostics.get("stabilization_policy"),
                "stabilization_diagnostics_summary": fit_diagnostics.get(
                    "stabilization_diagnostics_summary"
                ),
                "stabilization_policy_id": fit_diagnostics.get("stabilization_policy_id"),
                "objective_preserving_column_scaling": fit_diagnostics.get(
                    "objective_preserving_column_scaling"
                ),
                "column_scale_floor": fit_diagnostics.get("column_scale_floor"),
                "condition_number_gate_target": fit_diagnostics.get(
                    "condition_number_gate_target"
                ),
                "original_unscaled_normal_condition_role": fit_diagnostics.get(
                    "original_unscaled_normal_condition_role"
                ),
                "failed_fit_remains_inadmissible": True,
                "transport_returned": False,
                "nonclaims": (
                    "failed fit is not admissible",
                    "diagnostic capture is not numerical repair",
                    "no threshold retuning",
                    "no Phase 6 rerun authorization",
                    "no bug-fixed claim",
                ),
            },
        )
    defensive = TensorProductReferenceDensity(product_basis, convention)
    defensive_tau = _p59_author_sir_defensive_tau_tensor()
    density_identity = SquaredTTDensity.expected_branch_identity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=defensive_tau,
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=convention,
    )
    density = SquaredTTDensity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=defensive_tau,
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=convention,
        branch_identity=density_identity,
    )
    holdout_replay_diagnostics = _p69_post_fit_holdout_replay_diagnostics(
        fit_result=fit_result,
        density=density,
        local_fit_points=local_fit_points,
        target_values=target_tensor,
        fit_weights=weights,
        fit_data_manifest=fit_data_manifest,
        holdout_local_points=holdout_local_points,
        holdout_target_values=holdout_target_values,
        holdout_weights=holdout_weights,
        holdout_manifest=holdout_manifest,
        replay_local_points=replay_local_points,
        replay_target_values=replay_target_values,
        replay_weights=replay_weights,
        replay_manifest=replay_manifest,
    )
    return (
        FixedTTSIRTTransport(
            density=density,
            cdf_config=KRCDFConfig(
                grid_size=5,
                bisection_steps=4,
                monotonicity_tolerance=1e-10,
                bracket_tolerance=1e-10,
                denominator_floor=1e-12,
                max_floor_count=0,
            ),
        ),
        fit_result.branch_hash.value,
        density.branch_identity.hash.value,
        fit_quality_diagnostics,
        holdout_replay_diagnostics,
    )


def _p69_post_fit_holdout_replay_diagnostics(
    *,
    fit_result,
    density: SquaredTTDensity,
    local_fit_points: tf.Tensor,
    target_values: tf.Tensor,
    fit_weights: tf.Tensor,
    fit_data_manifest: Mapping[str, object] | None,
    holdout_local_points: tf.Tensor | None,
    holdout_target_values: tf.Tensor | None,
    holdout_weights: tf.Tensor | None,
    holdout_manifest: Mapping[str, object] | None,
    replay_local_points: tf.Tensor | None,
    replay_target_values: tf.Tensor | None,
    replay_weights: tf.Tensor | None,
    replay_manifest: Mapping[str, object] | None,
) -> Mapping[str, object]:
    fit_hash_before = fit_result.branch_hash.value
    density_hash_before = density.branch_identity.hash.value
    fit_point_hash = _p69_hash_tensor("p69_fit_points_hash.v1", local_fit_points)
    fit_target_hash = _p69_hash_tensor("p69_fit_targets_hash.v1", target_values)
    fit_weight_hash = _p69_hash_tensor("p69_fit_weights_hash.v1", fit_weights)
    coordinate_frame_hash = (
        None
        if fit_data_manifest is None
        else fit_data_manifest.get("coordinate_frame_hash")
    )
    holdout = _p69_post_fit_diagnostic_channel(
        channel="holdout",
        fitted_tt=fit_result.fitted_tt,
        points=holdout_local_points,
        targets=holdout_target_values,
        weights=holdout_weights,
        manifest=holdout_manifest,
        fit_point_hash=fit_point_hash,
    )
    replay = _p69_post_fit_diagnostic_channel(
        channel="replay",
        fitted_tt=fit_result.fitted_tt,
        points=replay_local_points,
        targets=replay_target_values,
        weights=replay_weights,
        manifest=replay_manifest,
        fit_point_hash=fit_point_hash,
    )
    fit_hash_after = fit_result.branch_hash.value
    density_hash_after = density.branch_identity.hash.value
    branch_unchanged = (
        fit_hash_before == fit_hash_after
        and density_hash_before == density_hash_after
    )
    any_missing = holdout["supplied"] is not True or replay["supplied"] is not True
    route_mismatch = (
        not any_missing
        and not (
            holdout["route_match"] is True
            and replay["route_match"] is True
        )
    )
    if not branch_unchanged:
        status = P69_BRANCH_IDENTITY_DRIFT_STATUS
    elif any_missing:
        status = P69_HOLDOUT_REPLAY_MISSING_STATUS
    elif route_mismatch:
        status = P69_HOLDOUT_REPLAY_ROUTE_MISMATCH_STATUS
    elif holdout["nonfinite"] is True or replay["nonfinite"] is True:
        status = P69_HOLDOUT_REPLAY_NONFINITE_STATUS
    elif holdout["available"] is not True or replay["available"] is not True:
        status = P69_HOLDOUT_REPLAY_MISSING_STATUS
    else:
        status = P69_HOLDOUT_REPLAY_AVAILABLE_STATUS
    return {
        "diagnostic_role": "post_fit_diagnostic_only",
        "diagnostic_classification": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "status": status,
        "warning_status": P69_HOLDOUT_REPLAY_DIAGNOSTIC_ONLY_STATUS,
        "fit_point_count": int(tf.shape(local_fit_points)[1]),
        "fit_point_hash": fit_point_hash,
        "fit_target_hash": fit_target_hash,
        "fit_weight_hash": fit_weight_hash,
        "coordinate_frame_hash": coordinate_frame_hash,
        "holdout_available": holdout["available"],
        "holdout_status": holdout["status"],
        "holdout_point_count": holdout["point_count"],
        "holdout_point_hash": holdout["point_hash"],
        "holdout_target_hash": holdout["target_hash"],
        "holdout_weight_hash": holdout["weight_hash"],
        "holdout_construction": holdout["construction"],
        "holdout_disjoint_from_fit": holdout["disjoint_from_fit"],
        "holdout_residual": holdout["residual"],
        "holdout_residual_available": holdout["residual_available"],
        "holdout_nonfinite": holdout["nonfinite"],
        "holdout_threshold_role": "diagnostic_only_unless_predeclared",
        "replay_available": replay["available"],
        "replay_status": replay["status"],
        "replay_identity": replay["identity"],
        "replay_point_count": replay["point_count"],
        "replay_point_hash": replay["point_hash"],
        "replay_target_hash": replay["target_hash"],
        "replay_weight_hash": replay["weight_hash"],
        "replay_disjoint_from_fit": replay["disjoint_from_fit"],
        "replay_residual": replay["residual"],
        "replay_residual_available": replay["residual_available"],
        "replay_nonfinite": replay["nonfinite"],
        "fit_branch_hash_before_diagnostic": fit_hash_before,
        "fit_branch_hash_after_diagnostic": fit_hash_after,
        "density_branch_hash_before_diagnostic": density_hash_before,
        "density_branch_hash_after_diagnostic": density_hash_after,
        "branch_identity_unchanged_by_diagnostics": branch_unchanged,
        "source_route_invariants": {
            "fit_data_mode": (
                None if fit_data_manifest is None else fit_data_manifest.get("fit_data_mode")
            ),
            "holdout_route_match": holdout["route_match"],
            "replay_route_match": replay["route_match"],
            "route_mismatch": route_mismatch,
        },
        "fixed_branch_adaptation_class": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "nonclaims": (
            "post-fit diagnostics do not change fitted cores",
            "holdout/replay residuals are not filtering correctness",
            "no adaptive Zhao-Cui parity claim",
        ),
    }


def _p69_post_fit_diagnostic_channel(
    *,
    channel: str,
    fitted_tt,
    points: tf.Tensor | None,
    targets: tf.Tensor | None,
    weights: tf.Tensor | None,
    manifest: Mapping[str, object] | None,
    fit_point_hash: str,
) -> Mapping[str, object]:
    if points is None or targets is None:
        return {
            "available": False,
            "supplied": False,
            "status": P69_HOLDOUT_REPLAY_MISSING_STATUS,
            "point_count": 0,
            "point_hash": None,
            "target_hash": None,
            "weight_hash": None,
            "construction": "not_supplied",
            "identity": "not_supplied",
            "disjoint_from_fit": False,
            "residual": None,
            "residual_available": False,
            "nonfinite": False,
            "route_match": False,
        }
    local = tf.convert_to_tensor(points, dtype=tf.float64)
    target = tf.convert_to_tensor(targets, dtype=tf.float64)
    weight = (
        tf.ones([int(tf.shape(local)[1])], dtype=tf.float64)
        if weights is None
        else tf.convert_to_tensor(weights, dtype=tf.float64)
    )
    if local.shape.rank != 2 or target.shape != (int(local.shape[1]),) or weight.shape != target.shape:
        raise ValueError(f"{channel}_diagnostic: {HighDimStatus.INVALID_SHAPE.value}")
    point_hash = _p69_hash_tensor(f"p69_{channel}_points_hash.v1", local)
    target_hash = _p69_hash_tensor(f"p69_{channel}_targets_hash.v1", target)
    weight_hash = _p69_hash_tensor(f"p69_{channel}_weights_hash.v1", weight)
    prediction = fitted_tt.evaluate(tf.transpose(local))
    residual = _p69_weighted_rms_residual(prediction, target, weight)
    residual_finite = bool(tf.math.is_finite(residual).numpy())
    route_match = bool(
        manifest is not None
        and manifest.get("fit_data_mode") == P63_AUTHOR_SIR_SOURCE_FIT_DATA_MODE
        and manifest.get("diagnostic_role") == "post_fit_diagnostic_only"
        and manifest.get("diagnostic_classification") == P65_FIXED_BRANCH_ADAPTATION_CLASS
    )
    return {
        "available": residual_finite and route_match,
        "supplied": True,
        "status": (
            P69_HOLDOUT_REPLAY_AVAILABLE_STATUS
            if residual_finite and route_match
            else (
                P69_HOLDOUT_REPLAY_ROUTE_MISMATCH_STATUS
                if residual_finite
                else P69_HOLDOUT_REPLAY_NONFINITE_STATUS
            )
        ),
        "point_count": int(local.shape[1]),
        "point_hash": point_hash,
        "target_hash": target_hash,
        "weight_hash": weight_hash,
        "construction": (
            "not_recorded"
            if manifest is None
            else str(manifest.get("diagnostic_construction", "not_recorded"))
        ),
        "identity": f"p69_{channel}_post_fit_replay.v1",
        "disjoint_from_fit": point_hash != fit_point_hash,
        "residual": residual,
        "residual_available": residual_finite,
        "nonfinite": not residual_finite,
        "route_match": route_match,
    }


def _p69_weighted_rms_residual(
    predicted: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    pred = tf.convert_to_tensor(predicted, dtype=tf.float64)
    values = tf.convert_to_tensor(target, dtype=tf.float64)
    weight = tf.convert_to_tensor(weights, dtype=tf.float64)
    if pred.shape != values.shape or weight.shape != values.shape:
        raise ValueError(f"diagnostic residual: {HighDimStatus.INVALID_SHAPE.value}")
    denominator = tf.reduce_sum(weight)
    return tf.sqrt(tf.reduce_sum(weight * tf.square(pred - values)) / denominator)


def _p69_frame_hash(frame: SourceRouteCoordinateFrame) -> str:
    return BranchManifest(
        version="p69_coordinate_frame_hash.v1",
        payload=frame.manifest_payload(),
    ).sha256().value


def _p69_hash_tensor(version: str, value: tf.Tensor) -> str:
    return BranchManifest(
        version=str(version),
        payload={"value": tf.convert_to_tensor(value, dtype=tf.float64)},
    ).sha256().value


def p72_support_certified_policy() -> Mapping[str, object]:
    """Return the frozen Phase-2 policy for the P72 lower-gate lane."""

    return {
        "policy_id": "p72_support_certified_fixed_fit_v1",
        "classification": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "source_faithfulness_status": "extension_or_invention_for_guard_audit_gates",
        "fit_ridge": P70_FIT_RIDGE,
        "low_level_condition_number_veto": P70_CONDITION_NUMBER_VETO,
        "p72_condition_number_admission": P72_CONDITION_NUMBER_ADMISSION,
        "guard_weight_alpha": P72_GUARD_WEIGHT_ALPHA,
        "shape_penalty_weight": P72_SHAPE_PENALTY_WEIGHT,
        "line_fractions": P72_LINE_FRACTIONS,
        "residual_rms_rel_veto": P72_RESIDUAL_RMS_REL_VETO,
        "residual_max_rel_veto": P72_RESIDUAL_MAX_REL_VETO,
        "line_growth_rel_veto": P72_LINE_GROWTH_REL_VETO,
        "sqrt_square_normalizer_floor": P72_SQRT_SQUARE_NORMALIZER_FLOOR,
        "fit_mass_fraction_min": P72_FIT_MASS_FRACTION_MIN,
        "log_normalizer_abs_bound": P72_LOG_NORMALIZER_ABS_BOUND,
        "effective_rank_tol": P72_EFFECTIVE_RANK_TOL,
        "effective_rank_min": P72_EFFECTIVE_RANK_MIN,
        "guard_seeds": {
            "step1_prior": P72_GUARD_STEP1_PRIOR_SEED,
            "step1_process": P72_GUARD_STEP1_PROCESS_SEED,
            "step2_process": P72_GUARD_STEP2_PROCESS_SEED,
        },
        "audit_seeds": {
            "step1_holdout_prior": P72_AUDIT_STEP1_HOLDOUT_PRIOR_SEED,
            "step1_holdout_process": P72_AUDIT_STEP1_HOLDOUT_PROCESS_SEED,
            "step1_replay_prior": P72_AUDIT_STEP1_REPLAY_PRIOR_SEED,
            "step1_replay_process": P72_AUDIT_STEP1_REPLAY_PROCESS_SEED,
            "step2_holdout_process": P72_AUDIT_STEP2_HOLDOUT_PROCESS_SEED,
            "step2_replay_process": P72_AUDIT_STEP2_REPLAY_PROCESS_SEED,
        },
        "nonclaims": (
            "not a source-faithful adaptive Zhao-Cui reproduction",
            "focused tests are not repaired diagnostic evidence",
            "no d18 validation claim",
            "no HMC readiness claim",
        ),
    }


def _p72_normalize_set_weights(weights: tf.Tensor) -> tf.Tensor:
    tensor = tf.convert_to_tensor(weights, dtype=tf.float64)
    if tensor.shape.rank != 1:
        raise ValueError(f"weights: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"weights: {HighDimStatus.NONFINITE_VALUE.value}")
    total = tf.reduce_sum(tensor)
    if bool(tf.reduce_any(tensor < 0.0).numpy()) or bool((total <= 0.0).numpy()):
        raise ValueError(f"weights: {HighDimStatus.NONFINITE_VALUE.value}")
    return tensor / total


def p72_training_batch_from_fit_and_guard(
    *,
    fit_points: tf.Tensor,
    fit_target_values: tf.Tensor,
    fit_weights: tf.Tensor,
    guard_points: tf.Tensor,
    guard_target_values: tf.Tensor,
    guard_weights: tf.Tensor,
    alpha_guard: float = P72_GUARD_WEIGHT_ALPHA,
) -> tuple[FixedTTFitSampleBatch, Mapping[str, object]]:
    """Concatenate fit and guard sets; audit data is deliberately absent."""

    fit = tf.convert_to_tensor(fit_points, dtype=tf.float64)
    guard = tf.convert_to_tensor(guard_points, dtype=tf.float64)
    fit_targets = tf.convert_to_tensor(fit_target_values, dtype=tf.float64)
    guard_targets = tf.convert_to_tensor(guard_target_values, dtype=tf.float64)
    if fit.shape.rank != 2 or guard.shape.rank != 2:
        raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
    if fit.shape[0] != guard.shape[0]:
        raise ValueError(f"guard_points: {HighDimStatus.INVALID_SHAPE.value}")
    if fit_targets.shape != (int(fit.shape[1]),) or guard_targets.shape != (int(guard.shape[1]),):
        raise ValueError(f"target_values: {HighDimStatus.INVALID_SHAPE.value}")
    fit_set_weights = _p72_normalize_set_weights(fit_weights)
    guard_set_weights = _p72_normalize_set_weights(guard_weights)
    alpha = float(alpha_guard)
    if alpha < 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha_guard must be finite nonnegative")
    points = tf.transpose(tf.concat([fit, guard], axis=1))
    targets = tf.concat([fit_targets, guard_targets], axis=0)
    weights = tf.concat([fit_set_weights, alpha * guard_set_weights], axis=0)
    manifest = {
        "policy_id": "p72_fit_guard_training_batch.v1",
        "fit_point_count": int(fit.shape[1]),
        "guard_point_count": int(guard.shape[1]),
        "audit_point_count_used_for_training": 0,
        "audit_exclusion_provenance": "Z_audit never concatenated into FixedTTFitSampleBatch",
        "alpha_guard": alpha,
        "fit_weight_mass": float(tf.reduce_sum(fit_set_weights).numpy()),
        "guard_weight_mass_before_alpha": float(tf.reduce_sum(guard_set_weights).numpy()),
        "guard_weight_mass_after_alpha": float((alpha * tf.reduce_sum(guard_set_weights)).numpy()),
        "fit_point_hash": _p69_hash_tensor("p72_fit_points_hash.v1", fit),
        "guard_point_hash": _p69_hash_tensor("p72_guard_points_hash.v1", guard),
        "training_point_hash": _p69_hash_tensor("p72_training_points_hash.v1", tf.transpose(points)),
        "fit_target_hash": _p69_hash_tensor("p72_fit_targets_hash.v1", fit_targets),
        "guard_target_hash": _p69_hash_tensor("p72_guard_targets_hash.v1", guard_targets),
        "training_target_hash": _p69_hash_tensor("p72_training_targets_hash.v1", targets),
        "fit_weight_hash": _p69_hash_tensor("p72_fit_weights_hash.v1", fit_set_weights),
        "guard_weight_hash": _p69_hash_tensor("p72_guard_weights_hash.v1", guard_set_weights),
        "training_weight_hash": _p69_hash_tensor("p72_training_weights_hash.v1", weights),
        "nonclaims": (
            "guard augmentation is a fixed-HMC support-stabilization gate",
            "audit data is not used for coefficient selection",
        ),
    }
    return FixedTTFitSampleBatch(points, targets, weights), manifest


def p72_support_clipping_coverage(
    *,
    role: str,
    points: tf.Tensor | None,
    fit_points: tf.Tensor,
    clip_fraction: float | None = None,
    local_max_abs_before_clip: float | None = None,
) -> Mapping[str, object]:
    """Finite-cloud support and clipping diagnostics for a P72 cloud role."""

    reasons: list[str] = []
    warnings: list[str] = []
    if points is None:
        fit = tf.convert_to_tensor(fit_points, dtype=tf.float64)
        fit_hash = (
            _p69_hash_tensor("p72_support_fit_points_hash.v1", fit)
            if bool(tf.reduce_all(tf.math.is_finite(fit)).numpy())
            else None
        )
        return {
            "role": str(role),
            "status": "block",
            "reasons": ("missing_points",),
            "warnings": (),
            "point_count": 0,
            "point_hash": None,
            "fit_point_hash": fit_hash,
        }
    local = tf.convert_to_tensor(points, dtype=tf.float64)
    fit = tf.convert_to_tensor(fit_points, dtype=tf.float64)
    if local.shape.rank != 2 or fit.shape.rank != 2 or local.shape[0] != fit.shape[0]:
        raise ValueError(f"{role}_support: {HighDimStatus.INVALID_SHAPE.value}")
    point_count = int(local.shape[1])
    fit_count = int(fit.shape[1])
    fit_is_finite = bool(tf.reduce_all(tf.math.is_finite(fit)).numpy())
    if point_count <= 0 or fit_count <= 0:
        reasons.append("empty_cloud")
    local_is_finite = bool(tf.reduce_all(tf.math.is_finite(local)).numpy())
    if not local_is_finite:
        reasons.append("nonfinite_cloud")
    if not fit_is_finite:
        reasons.append("nonfinite_fit_cloud")
    clipped = None if clip_fraction is None else float(clip_fraction)
    max_abs = None if local_max_abs_before_clip is None else float(local_max_abs_before_clip)
    if clipped is not None:
        if not math.isfinite(clipped):
            reasons.append("nonfinite_clip_fraction")
        elif clipped >= 1.0:
            reasons.append("all_clipped_cloud")
        elif clipped > 0.0:
            warnings.append("positive_clip_fraction")
    if max_abs is not None and (not math.isfinite(max_abs)):
        reasons.append("nonfinite_local_max_abs_before_clip")
    if point_count > 0 and fit_count > 0 and local_is_finite and fit_is_finite:
        distances = tf.norm(local[:, :, tf.newaxis] - fit[:, tf.newaxis, :], axis=0)
        nearest = tf.reduce_min(distances, axis=1)
        fit_pairwise = tf.norm(fit[:, :, tf.newaxis] - fit[:, tf.newaxis, :], axis=0)
        large = tf.constant(1e300, dtype=tf.float64)
        fit_leave_one_out = (
            tf.reduce_min(fit_pairwise + tf.eye(fit_count, dtype=tf.float64) * large, axis=1)
            if fit_count > 1
            else tf.zeros([fit_count], dtype=tf.float64)
        )
    else:
        nearest = tf.constant([], dtype=tf.float64)
        fit_leave_one_out = tf.constant([], dtype=tf.float64)
    nearest_max = None if int(nearest.shape[0]) <= 0 else float(tf.reduce_max(nearest).numpy())
    fit_loo_max = None if int(fit_leave_one_out.shape[0]) <= 0 else float(tf.reduce_max(fit_leave_one_out).numpy())
    if nearest_max is not None and fit_loo_max is not None and fit_loo_max > 0.0:
        if nearest_max > 10.0 * fit_loo_max:
            warnings.append("cloud_far_from_fit_support")
    status = "block" if reasons else ("warn" if warnings else "pass")
    return {
        "role": str(role),
        "status": status,
        "reasons": tuple(reasons),
        "warnings": tuple(warnings),
        "point_count": point_count,
        "point_hash": (
            _p69_hash_tensor(f"p72_{role}_support_points_hash.v1", local)
            if local_is_finite
            else None
        ),
        "fit_point_count": fit_count,
        "fit_point_hash": (
            _p69_hash_tensor("p72_support_fit_points_hash.v1", fit)
            if fit_is_finite
            else None
        ),
        "nearest_fit_distance_min": None if int(nearest.shape[0]) <= 0 else float(tf.reduce_min(nearest).numpy()),
        "nearest_fit_distance_median": None if int(nearest.shape[0]) <= 0 else _p60_tensor_median_float(nearest),
        "nearest_fit_distance_max": nearest_max,
        "fit_leave_one_out_distance_max": fit_loo_max,
        "clip_fraction": clipped,
        "point_any_saturated_fraction": (
            None
            if point_count <= 0
            else float(
                tf.reduce_mean(
                    tf.cast(tf.reduce_any(tf.abs(local) >= 1.0, axis=0), tf.float64)
                ).numpy()
            )
        ),
        "local_max_abs_before_clip": max_abs,
        "effective_support_role": "finite_cloud_diagnostic_not_continuum_support",
    }


def p72_guard_line_points(
    *,
    fit_points: tf.Tensor,
    guard_points: tf.Tensor,
    line_fractions: tuple[float, ...] = P72_LINE_FRACTIONS,
) -> tuple[tf.Tensor, Mapping[str, object]]:
    """Build deterministic fit-to-guard line probes for guard augmentation."""

    fit = tf.convert_to_tensor(fit_points, dtype=tf.float64)
    guard = tf.convert_to_tensor(guard_points, dtype=tf.float64)
    if fit.shape.rank != 2 or guard.shape.rank != 2 or fit.shape[0] != guard.shape[0]:
        raise ValueError(f"line_points: {HighDimStatus.INVALID_SHAPE.value}")
    if int(fit.shape[1]) <= 0 or int(guard.shape[1]) <= 0:
        raise ValueError("line point construction requires nonempty clouds")
    center = tf.reduce_mean(fit, axis=1, keepdims=True)
    distances = tf.norm(guard - center, axis=0)
    sorted_indices = tf.argsort(distances)
    selected = tf.gather(
        sorted_indices,
        tf.constant(
            [0, int(guard.shape[1]) // 2, int(guard.shape[1]) - 1],
            dtype=tf.int32,
        ),
    )
    endpoints = tf.gather(guard, selected, axis=1)
    starts = tf.repeat(center, repeats=int(endpoints.shape[1]), axis=1)
    fractions = tuple(float(value) for value in line_fractions)
    pieces = []
    for fraction in fractions:
        frac = tf.constant(fraction, dtype=tf.float64)
        pieces.append((1.0 - frac) * starts + frac * endpoints)
    raw_line_points = tf.concat(pieces, axis=1)
    unique_columns = []
    unique_start_indices = []
    seen_columns = set()
    endpoint_count = int(endpoints.shape[1])
    for column_index, column in enumerate(tf.transpose(raw_line_points).numpy()):
        key = tuple(float(f"{float(value):.17g}") for value in column)
        if key in seen_columns:
            continue
        seen_columns.add(key)
        unique_columns.append(column)
        unique_start_indices.append(column_index % endpoint_count)
    line_points = tf.transpose(tf.constant(unique_columns, dtype=tf.float64))
    manifest = {
        "policy_id": "p72_guard_line_points.v1",
        "line_fractions": fractions,
        "selected_guard_indices": tuple(int(i) for i in selected.numpy()),
        "line_start_indices": tuple(int(i) for i in unique_start_indices),
        "line_point_count": int(line_points.shape[1]),
        "raw_line_point_count": int(raw_line_points.shape[1]),
        "line_hash": _p69_hash_tensor("p72_guard_line_points_hash.v1", line_points),
        "duplicate_removal": "deterministic_exact_column_key_on_fixed_line_cloud",
    }
    return line_points, manifest


def p72_line_probe_diagnostics(
    *,
    fitted_tt,
    line_points: tf.Tensor,
    line_target_values: tf.Tensor,
    start_prediction_values: tf.Tensor | None = None,
    line_start_indices: tuple[int, ...] | list[int] | tf.Tensor | None = None,
    target_scale: float,
) -> Mapping[str, object]:
    """Evaluate P72 absolute and endpoint-growth line gates."""

    points = tf.convert_to_tensor(line_points, dtype=tf.float64)
    targets = tf.convert_to_tensor(line_target_values, dtype=tf.float64)
    if points.shape.rank != 2 or targets.shape != (int(points.shape[1]),):
        raise ValueError(f"line_probe: {HighDimStatus.INVALID_SHAPE.value}")
    predictions = tf.convert_to_tensor(fitted_tt.evaluate(tf.transpose(points)), dtype=tf.float64)
    if predictions.shape != targets.shape:
        raise ValueError(f"line_probe_prediction: {HighDimStatus.INVALID_SHAPE.value}")
    residual = tf.abs(predictions - targets)
    scale = max(float(target_scale), 1e-300)
    reasons: list[str] = []
    max_abs = float(tf.reduce_max(tf.abs(predictions)).numpy())
    max_residual = float(tf.reduce_max(residual).numpy())
    rms_residual = float(tf.sqrt(tf.reduce_mean(tf.square(residual))).numpy())
    if not math.isfinite(max_abs) or not math.isfinite(max_residual) or not math.isfinite(rms_residual):
        reasons.append("line_nonfinite")
    if max_abs > P72_LINE_GROWTH_REL_VETO * scale:
        reasons.append("line_absolute_value_veto")
    if max_residual > P72_RESIDUAL_MAX_REL_VETO * scale:
        reasons.append("line_max_residual_veto")
    if rms_residual > P72_RESIDUAL_RMS_REL_VETO * scale:
        reasons.append("line_rms_residual_veto")
    growth_ratio = None
    line_start_indices_available = False
    if start_prediction_values is not None:
        starts = tf.convert_to_tensor(start_prediction_values, dtype=tf.float64)
        if starts.shape.rank != 1 or starts.shape[0] is None or int(starts.shape[0]) <= 0:
            raise ValueError(f"line_start_predictions: {HighDimStatus.INVALID_SHAPE.value}")
        start_scale = tf.maximum(tf.abs(starts), tf.constant(scale, dtype=tf.float64))
        if line_start_indices is None:
            denominators = tf.ones_like(predictions) * tf.reduce_max(start_scale)
        else:
            indices = tf.reshape(tf.convert_to_tensor(line_start_indices, dtype=tf.int32), [-1])
            if int(indices.shape[0]) != int(points.shape[1]):
                raise ValueError(f"line_start_indices: {HighDimStatus.INVALID_SHAPE.value}")
            denominators = tf.gather(start_scale, indices)
            line_start_indices_available = True
        ratios = tf.abs(predictions) / denominators
        growth_ratio = float(tf.reduce_max(ratios).numpy())
        if not math.isfinite(growth_ratio) or growth_ratio > P72_LINE_GROWTH_REL_VETO:
            reasons.append("line_endpoint_growth_veto")
    return {
        "status": "block" if reasons else "pass",
        "reasons": tuple(reasons),
        "line_point_count": int(points.shape[1]),
        "line_point_hash": _p69_hash_tensor("p72_line_probe_points_hash.v1", points),
        "line_target_hash": _p69_hash_tensor("p72_line_probe_targets_hash.v1", targets),
        "line_prediction_hash": _p69_hash_tensor("p72_line_probe_predictions_hash.v1", predictions),
        "target_scale": scale,
        "line_prediction_max_abs": max_abs,
        "line_residual_max_abs": max_residual,
        "line_residual_rms": rms_residual,
        "endpoint_growth_ratio_max": growth_ratio,
        "line_start_indices_available": line_start_indices_available,
        "direct_target_evaluation_required": True,
    }


def p72_full_normalizer_gate(
    normalizer_terms: Mapping[str, object],
) -> Mapping[str, object]:
    """Apply mandatory P72 normalizer-admission checks."""

    reasons: list[str] = []
    normalizer_exception = normalizer_terms.get("normalizer_exception")
    if normalizer_exception not in (None, "", "None"):
        reasons.append("normalizer_exception_veto")

    def finite_positive(name: str) -> float | None:
        if name not in normalizer_terms:
            reasons.append(f"missing_{name}")
            return None
        try:
            value = float(normalizer_terms[name])
        except (TypeError, ValueError):
            reasons.append(f"nonfinite_{name}")
            return None
        if not math.isfinite(value):
            reasons.append(f"nonfinite_{name}")
        elif value <= 0.0:
            reasons.append(f"nonpositive_{name}")
        return value

    mixture = finite_positive("mixture_normalizer")
    sqrt_square = finite_positive("sqrt_square_normalizer")
    tau = finite_positive("defensive_tau")
    defensive = finite_positive("defensive_normalizer")
    if sqrt_square is not None and sqrt_square <= P72_SQRT_SQUARE_NORMALIZER_FLOOR:
        reasons.append("sqrt_square_normalizer_floor_veto")
    fit_mass_fraction = None
    if mixture is not None and sqrt_square is not None:
        fit_mass_fraction = sqrt_square / mixture
        if fit_mass_fraction < P72_FIT_MASS_FRACTION_MIN:
            reasons.append("fit_mass_fraction_veto")
    log_value = normalizer_terms.get("log_transport_normalizer")
    if log_value is None:
        reasons.append("missing_log_transport_normalizer")
        log_float = None
    else:
        try:
            log_float = float(log_value)
        except (TypeError, ValueError):
            log_float = None
        if log_float is None or not math.isfinite(log_float):
            reasons.append("nonfinite_log_transport_normalizer")
        elif abs(log_float) > P72_LOG_NORMALIZER_ABS_BOUND:
            reasons.append("log_transport_normalizer_abs_bound_veto")
    return {
        "status": "block" if reasons else "pass",
        "reasons": tuple(reasons),
        "mixture_normalizer": mixture,
        "sqrt_square_normalizer": sqrt_square,
        "defensive_tau": tau,
        "defensive_normalizer": defensive,
        "transport_normalizer": normalizer_terms.get("transport_normalizer"),
        "transport_normalizer_floor": normalizer_terms.get("transport_normalizer_floor"),
        "normalizer_exception": normalizer_exception,
        "fit_mass_fraction": fit_mass_fraction,
        "log_transport_normalizer": log_float,
        "thresholds": {
            "sqrt_square_normalizer_floor": P72_SQRT_SQUARE_NORMALIZER_FLOOR,
            "fit_mass_fraction_min": P72_FIT_MASS_FRACTION_MIN,
            "log_normalizer_abs_bound": P72_LOG_NORMALIZER_ABS_BOUND,
        },
    }


def p72_condition_effective_rank_gate(
    records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
) -> Mapping[str, object]:
    """P72 wrapper/admission gate over already-recorded solve diagnostics."""

    reasons: list[str] = []
    condition_values: list[float] = []
    effective_ranks: list[float] = []
    for record in records:
        condition_raw = record.get(
            "scaled_augmented_condition_number",
            record.get("condition_number"),
        )
        if isinstance(condition_raw, (int, float)):
            condition = float(condition_raw)
            condition_values.append(condition)
            if not math.isfinite(condition) or condition > P72_CONDITION_NUMBER_ADMISSION:
                reasons.append("p72_condition_admission_veto")
        singular_values = record.get("scaled_augmented_singular_values")
        if singular_values is not None:
            values = tf.reshape(tf.convert_to_tensor(singular_values, dtype=tf.float64), [-1])
            if int(values.shape[0]) == 0 or not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
                reasons.append("p72_effective_rank_unavailable")
                continue
            max_sv = tf.reduce_max(values)
            active = tf.reduce_sum(
                tf.cast(values > P72_EFFECTIVE_RANK_TOL * max_sv, tf.float64)
            )
            rank_value = float(active.numpy())
            effective_ranks.append(rank_value)
            if rank_value < P72_EFFECTIVE_RANK_MIN:
                reasons.append("p72_effective_rank_veto")
        elif "effective_rank" in record:
            rank_value = float(record["effective_rank"])
            effective_ranks.append(rank_value)
            if not math.isfinite(rank_value) or rank_value < P72_EFFECTIVE_RANK_MIN:
                reasons.append("p72_effective_rank_veto")
    if not condition_values:
        reasons.append("p72_condition_unavailable")
    if not effective_ranks:
        reasons.append("p72_effective_rank_unavailable")
    return {
        "status": "block" if reasons else "pass",
        "reasons": tuple(dict.fromkeys(reasons)),
        "condition_number_max": None if not condition_values else max(condition_values),
        "condition_number_admission": P72_CONDITION_NUMBER_ADMISSION,
        "effective_rank_min": None if not effective_ranks else min(effective_ranks),
        "effective_rank_threshold": P72_EFFECTIVE_RANK_MIN,
        "low_level_solver_veto_unchanged": P70_CONDITION_NUMBER_VETO,
        "p72_gate_role": "wrapper_admission_not_low_level_solver_abort",
    }


def p72_provenance_manifest(
    *,
    branch_identity: str,
    target_values: tf.Tensor,
    frame_hash: str,
    shift_constant: tf.Tensor | float,
    fit_points: tf.Tensor,
    guard_points: tf.Tensor,
    audit_points: tf.Tensor,
    line_points: tf.Tensor,
) -> Mapping[str, object]:
    """Record branch/frame/target/cloud hashes for P72 diagnostics."""

    return {
        "policy_id": "p72_provenance_manifest.v1",
        "branch_identity": str(branch_identity),
        "target_hash": _p69_hash_tensor("p72_provenance_target_hash.v1", target_values),
        "frame_hash": str(frame_hash),
        "shift_constant": float(tf.convert_to_tensor(shift_constant, dtype=tf.float64).numpy()),
        "fit_cloud_hash": _p69_hash_tensor("p72_provenance_fit_cloud_hash.v1", fit_points),
        "guard_cloud_hash": _p69_hash_tensor("p72_provenance_guard_cloud_hash.v1", guard_points),
        "audit_cloud_hash": _p69_hash_tensor("p72_provenance_audit_cloud_hash.v1", audit_points),
        "line_hash": _p69_hash_tensor("p72_provenance_line_hash.v1", line_points),
        "audit_exclusion_provenance": "audit clouds and audit-line points excluded from coefficient selection",
        "classification": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "nonclaims": (
            "P72 guard/audit/line gates are not source-faithful Zhao-Cui operations",
            "hash provenance is not validation evidence",
        ),
    }


def p72_gate_summary(
    *,
    residual_gates: Mapping[str, object],
    support_gates: Mapping[str, Mapping[str, object]],
    normalizer_gate: Mapping[str, object],
    line_gate: Mapping[str, object],
    condition_gate: Mapping[str, object],
    rank_activity: Mapping[str, object],
    provenance: Mapping[str, object],
) -> Mapping[str, object]:
    """Combine P72 support-certified lower-gate components."""

    reasons: list[str] = []
    for role, gate in support_gates.items():
        if gate.get("status") == "block":
            reasons.append(f"{role}_support_block")
    for name, gate in (
        ("normalizer", normalizer_gate),
        ("line", line_gate),
        ("condition_effective_rank", condition_gate),
    ):
        if gate.get("status") == "block":
            reasons.append(f"{name}_block")
    if rank_activity.get("status") != "ok":
        reasons.append("rank_channel_activity_failed")
    rms = residual_gates.get("rms_relative")
    max_rel = residual_gates.get("max_relative")
    if rms is None:
        reasons.append("residual_rms_missing")
    else:
        rms_float = float(rms)
        if not math.isfinite(rms_float) or rms_float > P72_RESIDUAL_RMS_REL_VETO:
            reasons.append("residual_rms_veto")
    if max_rel is None:
        reasons.append("residual_max_missing")
    else:
        max_float = float(max_rel)
        if not math.isfinite(max_float) or max_float > P72_RESIDUAL_MAX_REL_VETO:
            reasons.append("residual_max_veto")
    required_provenance = (
        "branch_identity",
        "target_hash",
        "frame_hash",
        "shift_constant",
        "fit_cloud_hash",
        "guard_cloud_hash",
        "audit_cloud_hash",
        "line_hash",
        "audit_exclusion_provenance",
    )
    missing = tuple(name for name in required_provenance if name not in provenance)
    if missing:
        reasons.append("provenance_missing")
    return {
        "status": P72_BLOCK_STATUS if reasons else P72_PASS_STATUS,
        "reasons": tuple(dict.fromkeys(reasons)),
        "policy": p72_support_certified_policy(),
        "residual_gates": dict(residual_gates),
        "support_gates": {str(key): dict(value) for key, value in support_gates.items()},
        "normalizer_gate": dict(normalizer_gate),
        "line_gate": dict(line_gate),
        "condition_effective_rank_gate": dict(condition_gate),
        "rank_activity": dict(rank_activity),
        "provenance": dict(provenance),
        "missing_provenance_fields": missing,
        "nonclaims": (
            "gate pass is not d18 validation",
            "gate pass is not HMC readiness",
            "gate pass is not adaptive Zhao-Cui source-faithful parity",
        ),
    }


_P73_RENEWAL_ROLES = (
    "fit",
    "guard",
    "guard_line",
    "audit",
    "audit_line",
    "enrichment",
    "fresh",
)
_P73_GUARD_ENRICHMENT_CHANNELS = ("guard", "guard_line")
_P73_AUDIT_CHANNELS = ("audit", "audit_line")
_P73_REQUIRED_PROVENANCE_FIELDS = (
    "point_id",
    "cloud_hash",
    "role",
    "created_round",
    "entered_training_round",
    "audit_round",
    "source_channel",
    "parent_point_ids",
    "seed_or_constructor_label",
)


def p73_density_aware_renewal_policy() -> Mapping[str, object]:
    """Return the frozen opt-in policy for P73 renewed-support diagnostics."""

    p72_policy = p72_support_certified_policy()
    return {
        "policy_id": "p73_density_aware_renewed_support_v1",
        "classification": {
            "squared_tt_density_route": P65_FIXED_BRANCH_ADAPTATION_CLASS,
            "staged_renewal": "extension_or_invention",
            "density_aware_cross_entropy": "extension_or_invention",
            "strict_audit_exclusion": "extension_or_invention",
        },
        "p73_a_role": "mandatory_renewal_only_square_root_regression",
        "p73_b_role": "opt_in_density_aware_diagnostic_arm",
        "density_aware_objective_status": P73_DENSITY_AWARE_OBJECTIVE_STATUS,
        "p73_b_optimizer_status": P73_B_OPTIMIZER_BLOCKED,
        "renewal_count": P73_RENEWAL_COUNT,
        "lambda_ce": P73_LAMBDA_CE,
        "eps_log": P73_EPS_LOG,
        "line_penalty_weight": 0.0,
        "condition_penalty_weight": 0.0,
        "rank_promotion": "deferred",
        "no_audit_predicate": "NO_AUDIT_COEFFICIENT_SELECTION",
        "training_rule": "coefficients_selected_from_F_1_only",
        "enrichment_rule": "E_0_may_use_guard_or_guard_line_never_audit",
        "inherited_p72_thresholds": {
            "fit_rms_relative_veto": P72_RESIDUAL_RMS_REL_VETO,
            "guard_rms_relative_veto": P72_RESIDUAL_RMS_REL_VETO,
            "audit_rms_relative_veto": P72_RESIDUAL_RMS_REL_VETO,
            "residual_max_relative_veto": P72_RESIDUAL_MAX_REL_VETO,
            "line_growth_relative_veto": P72_LINE_GROWTH_REL_VETO,
            "condition_number_admission": P72_CONDITION_NUMBER_ADMISSION,
            "effective_rank_tol": P72_EFFECTIVE_RANK_TOL,
            "effective_rank_min": P72_EFFECTIVE_RANK_MIN,
            "sqrt_square_normalizer_floor": P72_SQRT_SQUARE_NORMALIZER_FLOOR,
            "fit_mass_fraction_min": P72_FIT_MASS_FRACTION_MIN,
            "log_normalizer_abs_bound": P72_LOG_NORMALIZER_ABS_BOUND,
            "low_level_condition_number_veto": P70_CONDITION_NUMBER_VETO,
        },
        "p72_policy_id": p72_policy["policy_id"],
        "nonclaims": (
            "not a source-faithful adaptive Zhao-Cui reproduction",
            "P73 renewal and audit gates are fixed-variant evidence safeguards",
            "cross-entropy values are explanatory unless downstream fresh-audit gates pass",
            "focused tests are not repaired lower-gate evidence",
            "no d18 validation claim",
            "no HMC readiness claim",
            "no rank promotion claim",
        ),
    }


def p73_renewal_role_record(
    *,
    point_id: str,
    cloud_hash: str,
    role: str,
    created_round: int,
    entered_training_round: int | None,
    audit_round: int | None,
    source_channel: str,
    parent_point_ids: tuple[str, ...] | list[str] = (),
    seed_or_constructor_label: str = "",
) -> Mapping[str, object]:
    """Build one P73 point-provenance record with renewal-role fields."""

    point = str(point_id)
    cloud = str(cloud_hash)
    role_value = str(role)
    channel = str(source_channel)
    if not point.strip():
        raise ValueError("point_id must be nonempty")
    if not cloud.strip():
        raise ValueError("cloud_hash must be nonempty")
    if role_value not in _P73_RENEWAL_ROLES:
        raise ValueError("role must be a P73 renewal role")
    if not channel.strip():
        raise ValueError("source_channel must be nonempty")
    if int(created_round) < 0:
        raise ValueError("created_round must be nonnegative")
    entered = None if entered_training_round is None else int(entered_training_round)
    if entered is not None and entered < 0:
        raise ValueError("entered_training_round must be nonnegative or None")
    audit = None if audit_round is None else int(audit_round)
    if audit is not None and audit < 0:
        raise ValueError("audit_round must be nonnegative or None")
    parents = tuple(str(parent) for parent in parent_point_ids)
    return {
        "point_id": point,
        "cloud_hash": cloud,
        "role": role_value,
        "created_round": int(created_round),
        "entered_training_round": entered,
        "audit_round": audit,
        "source_channel": channel,
        "parent_point_ids": parents,
        "seed_or_constructor_label": str(seed_or_constructor_label),
    }


def _p73_records_tuple(records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(dict(record) for record in records)


def _p73_record_cloud_hashes(
    records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
) -> set[str]:
    hashes: set[str] = set()
    for record in records:
        cloud_hash = record.get("cloud_hash")
        if cloud_hash not in (None, ""):
            hashes.add(str(cloud_hash))
    return hashes


def p73_no_audit_coefficient_selection(
    *,
    renewal_round: int,
    coefficient_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
    audit_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]] = (),
    audit_line_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]] = (),
    coefficient_cloud_hashes: tuple[str, ...] | list[str] = (),
    audit_cloud_hashes: tuple[str, ...] | list[str] = (),
    audit_line_cloud_hashes: tuple[str, ...] | list[str] = (),
    enrichment_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]] = (),
) -> Mapping[str, object]:
    """Evaluate the P73 ``NO_AUDIT_COEFFICIENT_SELECTION`` predicate."""

    round_index = int(renewal_round)
    coefficient = _p73_records_tuple(coefficient_records)
    audit = _p73_records_tuple(audit_records)
    audit_line = _p73_records_tuple(audit_line_records)
    enrichment = _p73_records_tuple(enrichment_records)
    reasons: list[str] = []
    if round_index < 0:
        reasons.append("negative_renewal_round")
    if not coefficient:
        reasons.append("missing_coefficient_records")
    for record in coefficient:
        missing = tuple(
            field for field in _P73_REQUIRED_PROVENANCE_FIELDS if field not in record
        )
        if missing:
            reasons.append("missing_coefficient_record_fields")
            continue
        role = str(record["role"])
        if role not in _P73_RENEWAL_ROLES:
            reasons.append("unknown_coefficient_role")
        if role in _P73_AUDIT_CHANNELS:
            reasons.append("audit_role_in_coefficient_selection")
        entered_round = record.get("entered_training_round")
        if entered_round is None:
            reasons.append("coefficient_without_training_round")
        else:
            try:
                entered = int(entered_round)
            except (TypeError, ValueError):
                reasons.append("invalid_training_round")
            else:
                if entered > round_index:
                    reasons.append("coefficient_enters_training_after_round")
        audit_round = record.get("audit_round")
        if audit_round is not None:
            try:
                audit_value = int(audit_round)
            except (TypeError, ValueError):
                reasons.append("invalid_audit_round")
            else:
                if audit_value <= round_index:
                    reasons.append("current_or_prior_audit_record_in_coefficients")
    coefficient_hashes = set(str(value) for value in coefficient_cloud_hashes)
    coefficient_hashes.update(_p73_record_cloud_hashes(coefficient))
    audit_hashes = set(str(value) for value in audit_cloud_hashes)
    audit_hashes.update(_p73_record_cloud_hashes(audit))
    audit_line_hashes = set(str(value) for value in audit_line_cloud_hashes)
    audit_line_hashes.update(_p73_record_cloud_hashes(audit_line))
    overlap = coefficient_hashes.intersection(audit_hashes.union(audit_line_hashes))
    if overlap:
        reasons.append("same_round_audit_hash_overlap")
    for record in enrichment:
        missing = tuple(
            field for field in _P73_REQUIRED_PROVENANCE_FIELDS if field not in record
        )
        if missing:
            reasons.append("missing_enrichment_record_fields")
            continue
        source = str(record["source_channel"])
        role = str(record["role"])
        if source not in _P73_GUARD_ENRICHMENT_CHANNELS:
            reasons.append("enrichment_from_non_guard_channel")
        if role in _P73_AUDIT_CHANNELS:
            reasons.append("audit_role_in_enrichment")
    unique_reasons = tuple(dict.fromkeys(reasons))
    passed = not unique_reasons
    return {
        "predicate_id": "NO_AUDIT_COEFFICIENT_SELECTION",
        "NO_AUDIT_COEFFICIENT_SELECTION": passed,
        "status": "pass" if passed else "block",
        "reasons": unique_reasons,
        "renewal_round": round_index,
        "coefficient_point_count": len(coefficient),
        "audit_point_count": len(audit),
        "audit_line_point_count": len(audit_line),
        "enrichment_point_count": len(enrichment),
        "coefficient_cloud_hashes": tuple(sorted(coefficient_hashes)),
        "same_round_audit_cloud_hashes": tuple(sorted(audit_hashes)),
        "same_round_audit_line_cloud_hashes": tuple(sorted(audit_line_hashes)),
        "overlap_hashes": tuple(sorted(overlap)),
        "nonclaims": (
            "predicate pass is provenance evidence only",
            "predicate pass is not a lower-gate repair claim",
        ),
    }


def p73_training_batch_from_renewed_fit(
    *,
    fit_points: tf.Tensor,
    fit_target_values: tf.Tensor,
    fit_weights: tf.Tensor,
    fit_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
    audit_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]] = (),
    audit_line_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]] = (),
    coefficient_cloud_hashes: tuple[str, ...] | list[str] = (),
    audit_cloud_hashes: tuple[str, ...] | list[str] = (),
    audit_line_cloud_hashes: tuple[str, ...] | list[str] = (),
    renewal_round: int = P73_RENEWAL_COUNT,
) -> tuple[FixedTTFitSampleBatch, Mapping[str, object]]:
    """Build a P73 coefficient-selection batch from the renewed fit cloud only."""

    fit = tf.convert_to_tensor(fit_points, dtype=tf.float64)
    targets = tf.convert_to_tensor(fit_target_values, dtype=tf.float64)
    if fit.shape.rank != 2:
        raise ValueError(f"fit_points: {HighDimStatus.INVALID_SHAPE.value}")
    if targets.shape != (int(fit.shape[1]),):
        raise ValueError(f"fit_target_values: {HighDimStatus.INVALID_SHAPE.value}")
    records = _p73_records_tuple(fit_records)
    if len(records) != int(fit.shape[1]):
        raise ValueError("fit_records length must equal fit point count")
    weights = _p72_normalize_set_weights(fit_weights)
    predicate = p73_no_audit_coefficient_selection(
        renewal_round=int(renewal_round),
        coefficient_records=records,
        audit_records=audit_records,
        audit_line_records=audit_line_records,
        coefficient_cloud_hashes=coefficient_cloud_hashes,
        audit_cloud_hashes=audit_cloud_hashes,
        audit_line_cloud_hashes=audit_line_cloud_hashes,
    )
    if predicate["status"] != "pass":
        raise ValueError("NO_AUDIT_COEFFICIENT_SELECTION failed")
    batch = FixedTTFitSampleBatch(tf.transpose(fit), targets, weights)
    manifest = {
        "policy_id": "p73_F1_only_training_batch.v1",
        "renewal_round": int(renewal_round),
        "fit_point_count": int(fit.shape[1]),
        "guard_point_count_used_for_training": 0,
        "audit_point_count_used_for_training": 0,
        "audit_line_point_count_used_for_training": 0,
        "training_rule": "coefficients_selected_from_F_1_only",
        "no_audit_coefficient_selection": predicate,
        "fit_weight_mass": float(tf.reduce_sum(weights).numpy()),
        "fit_point_hash": _p69_hash_tensor("p73_fit_points_hash.v1", fit),
        "training_point_hash": _p69_hash_tensor("p73_training_points_hash.v1", fit),
        "fit_target_hash": _p69_hash_tensor("p73_fit_targets_hash.v1", targets),
        "training_target_hash": _p69_hash_tensor("p73_training_targets_hash.v1", targets),
        "fit_weight_hash": _p69_hash_tensor("p73_fit_weights_hash.v1", weights),
        "nonclaims": (
            "P73 training batch excludes same-round guard and audit clouds",
            "batch construction is not lower-gate evidence",
        ),
    }
    return batch, manifest


def p73_validate_enrichment_boundary(
    *,
    enrichment_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
) -> Mapping[str, object]:
    """Check that a P73 enrichment set uses guard or guard-line sources only."""

    records = _p73_records_tuple(enrichment_records)
    reasons: list[str] = []
    for record in records:
        missing = tuple(
            field for field in _P73_REQUIRED_PROVENANCE_FIELDS if field not in record
        )
        if missing:
            reasons.append("missing_enrichment_record_fields")
            continue
        source = str(record["source_channel"])
        role = str(record["role"])
        if source not in _P73_GUARD_ENRICHMENT_CHANNELS:
            reasons.append("enrichment_from_non_guard_channel")
        if role in _P73_AUDIT_CHANNELS:
            reasons.append("audit_role_in_enrichment")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return {
        "status": "pass" if not unique_reasons else "block",
        "reasons": unique_reasons,
        "enrichment_point_count": len(records),
        "allowed_source_channels": _P73_GUARD_ENRICHMENT_CHANNELS,
        "forbidden_source_channels": _P73_AUDIT_CHANNELS,
        "rule": "E_r subset G_r union L_r_guard",
    }


def p73_density_aware_cross_entropy(
    *,
    density: SquaredTTDensity,
    support_points: tf.Tensor,
    target_values: tf.Tensor,
    support_weights: tf.Tensor,
    point_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
    audit_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]] = (),
    audit_line_records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]] = (),
    coefficient_cloud_hashes: tuple[str, ...] | list[str] = (),
    audit_cloud_hashes: tuple[str, ...] | list[str] = (),
    audit_line_cloud_hashes: tuple[str, ...] | list[str] = (),
    renewal_round: int = P73_RENEWAL_COUNT,
    lambda_ce: float = P73_LAMBDA_CE,
) -> Mapping[str, object]:
    """Evaluate the opt-in P73-B empirical cross-entropy term on ``F_r``."""

    if not isinstance(density, SquaredTTDensity):
        raise TypeError("density must be a SquaredTTDensity")
    points = tf.convert_to_tensor(support_points, dtype=tf.float64)
    targets = tf.convert_to_tensor(target_values, dtype=tf.float64)
    if points.shape.rank != 2:
        raise ValueError(f"support_points: {HighDimStatus.INVALID_SHAPE.value}")
    if targets.shape != (int(points.shape[1]),):
        raise ValueError(f"target_values: {HighDimStatus.INVALID_SHAPE.value}")
    records = _p73_records_tuple(point_records)
    if len(records) != int(points.shape[1]):
        raise ValueError("point_records length must equal support point count")
    weights = _p72_normalize_set_weights(support_weights)
    predicate = p73_no_audit_coefficient_selection(
        renewal_round=int(renewal_round),
        coefficient_records=records,
        audit_records=audit_records,
        audit_line_records=audit_line_records,
        coefficient_cloud_hashes=coefficient_cloud_hashes,
        audit_cloud_hashes=audit_cloud_hashes,
        audit_line_cloud_hashes=audit_line_cloud_hashes,
    )
    if predicate["status"] != "pass":
        return {
            "status": "block",
            "reasons": ("no_audit_coefficient_selection_failed",),
            "no_audit_coefficient_selection": predicate,
            "density_aware_objective_status": P73_DENSITY_AWARE_OBJECTIVE_STATUS,
            "p73_b_optimizer_status": P73_B_OPTIMIZER_BLOCKED,
        }
    lam = float(lambda_ce)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lambda_ce must be finite nonnegative")
    points_nxd = tf.transpose(points)
    try:
        defensive_values = tf.exp(density.defensive_density.log_density(points_nxd))
        alpha_raw = weights * (tf.square(targets) + density.tau * defensive_values)
        alpha_total = tf.reduce_sum(alpha_raw)
        if bool((alpha_total <= 0.0).numpy()) or not bool(tf.math.is_finite(alpha_total).numpy()):
            return {
                "status": "block",
                "reasons": ("nonpositive_cross_entropy_weight_mass",),
                "no_audit_coefficient_selection": predicate,
                "density_aware_objective_status": P73_DENSITY_AWARE_OBJECTIVE_STATUS,
                "p73_b_optimizer_status": P73_B_OPTIMIZER_BLOCKED,
            }
        alpha = alpha_raw / alpha_total
        log_density_values = density.log_density(points_nxd)
        if not bool(tf.reduce_all(tf.math.is_finite(log_density_values)).numpy()):
            return {
                "status": "block",
                "reasons": ("nonfinite_log_density_values",),
                "no_audit_coefficient_selection": predicate,
                "density_aware_objective_status": P73_DENSITY_AWARE_OBJECTIVE_STATUS,
                "p73_b_optimizer_status": P73_B_OPTIMIZER_BLOCKED,
            }
        cross_entropy = -tf.reduce_sum(alpha * log_density_values)
        weighted_contribution = tf.constant(lam, dtype=tf.float64) * cross_entropy
    except (ValueError, tf.errors.InvalidArgumentError) as exc:
        return {
            "status": "block",
            "reasons": ("density_evaluation_exception",),
            "density_exception": str(exc),
            "no_audit_coefficient_selection": predicate,
            "density_aware_objective_status": P73_DENSITY_AWARE_OBJECTIVE_STATUS,
            "p73_b_optimizer_status": P73_B_OPTIMIZER_BLOCKED,
        }
    return {
        "status": "pass",
        "reasons": (),
        "renewal_round": int(renewal_round),
        "density_aware_objective_status": P73_DENSITY_AWARE_OBJECTIVE_STATUS,
        "lambda_ce": lam,
        "cross_entropy": float(cross_entropy.numpy()),
        "weighted_cross_entropy": float(weighted_contribution.numpy()),
        "alpha_min": float(tf.reduce_min(alpha).numpy()),
        "alpha_max": float(tf.reduce_max(alpha).numpy()),
        "alpha_sum": float(tf.reduce_sum(alpha).numpy()),
        "log_density_min": float(tf.reduce_min(log_density_values).numpy()),
        "log_density_max": float(tf.reduce_max(log_density_values).numpy()),
        "support_point_count": int(points.shape[1]),
        "support_point_hash": _p69_hash_tensor("p73_density_aware_support_points_hash.v1", points),
        "target_hash": _p69_hash_tensor("p73_density_aware_target_hash.v1", targets),
        "weight_hash": _p69_hash_tensor("p73_density_aware_weights_hash.v1", weights),
        "no_audit_coefficient_selection": predicate,
        "p73_b_optimizer_status": P73_B_OPTIMIZER_BLOCKED,
        "nonclaims": (
            "cross-entropy evaluation is not nonlinear optimizer evidence",
            "training-support objective values are explanatory only",
        ),
    }


def p73_density_aware_optimizer_status() -> Mapping[str, object]:
    """Report the Phase-4 status of the opt-in P73-B optimizer surface."""

    return {
        "status": P73_B_OPTIMIZER_BLOCKED,
        "density_aware_objective_status": P73_DENSITY_AWARE_OBJECTIVE_STATUS,
        "reason": (
            "Phase 4 implements objective evaluation only; the nonlinear "
            "TensorFlow refinement is not yet implemented."
        ),
        "phase5_runnable": False,
        "nonclaims": (
            "P73-B is not runnable in Phase 5 while this status is active",
            "least-squares reweighting is not treated as cross-entropy optimization",
        ),
    }


def _p59_fixed_ttsirt_fit_quality_diagnostics(fit_result) -> Mapping[str, object]:
    records = tuple(dict(record) for record in fit_result.core_update_statuses)
    numeric_conditions: list[float] = []
    condition_warnings = []
    condition_vetoes = []
    for record in records:
        condition = record.get("condition_number")
        if isinstance(condition, (int, float)):
            numeric_conditions.append(float(condition))
        if bool(record.get("condition_warning", False)):
            condition_warnings.append(record.get("core_index"))
        if str(record.get("status")) == HighDimStatus.CONDITION_NUMBER_VETO.value:
            condition_vetoes.append(record.get("core_index"))
    diagnostics = {} if fit_result.diagnostics is None else dict(fit_result.diagnostics)
    fit_residual = diagnostics.get("fit_residual", fit_result.fit_residual)
    holdout_residual = diagnostics.get("holdout_residual", fit_result.holdout_residual)
    return {
        "status": fit_result.status.value,
        "termination_reason": fit_result.termination_reason,
        "stop_condition_triggered": fit_result.stop_condition_triggered,
        "fit_residual": fit_residual,
        "fit_residual_available": fit_residual is not None,
        "holdout_residual": holdout_residual,
        "holdout_available": holdout_residual is not None,
        "holdout_status": "available" if holdout_residual is not None else "not_supplied",
        "per_core_update_statuses": records,
        "condition_number_summary": {
            "available": bool(records),
            "finite_condition_count": len(numeric_conditions),
            "max_condition_number": (
                max(numeric_conditions) if numeric_conditions else None
            ),
            "condition_warning_core_indices": tuple(condition_warnings),
            "condition_veto_core_indices": tuple(condition_vetoes),
            "condition_number_warning": _first_record_value(
                records,
                "condition_number_warning",
            ),
            "condition_number_veto": _first_record_value(
                records,
                "condition_number_veto",
            ),
        },
        "nonclaims": (
            "fit residual is not filtering correctness",
            "condition diagnostics are not structural convergence proof",
            "holdout absence is not holdout validation",
        ),
    }


def _first_record_value(records: tuple[Mapping[str, object], ...], key: str) -> object | None:
    for record in records:
        if key in record:
            return record[key]
    return None


def _source_route_rank_tuple(target_dim: int, fit_rank: int) -> tuple[int, ...]:
    dim = int(target_dim)
    rank = int(fit_rank)
    if dim <= 0:
        raise ValueError("target_dim must be positive")
    if rank <= 0:
        raise ValueError("fit_rank must be positive")
    return tuple([1] + [rank] * (dim - 1) + [1])


def _source_route_constant_path_initial_cores(
    *,
    ranks: tuple[int, ...],
    basis_dim: int,
    constant_value: tf.Tensor,
) -> tuple[TTCore, ...]:
    values = tf.convert_to_tensor(constant_value, dtype=tf.float64)
    if values.shape.rank != 0:
        raise ValueError(f"constant_value: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.math.is_finite(values).numpy()):
        raise ValueError(f"constant_value: {HighDimStatus.NONFINITE_VALUE.value}")
    if bool((values <= 0.0).numpy()):
        raise ValueError(f"constant_value: {HighDimStatus.NONFINITE_VALUE.value}")
    cores = []
    for axis in range(len(ranks) - 1):
        core_values = tf.zeros(
            [int(ranks[axis]), int(basis_dim), int(ranks[axis + 1])],
            dtype=tf.float64,
        )
        entry = values if axis == 0 else tf.constant(1.0, dtype=tf.float64)
        indices = tf.constant([[0, 0, 0]], dtype=tf.int64)
        cores.append(TTCore(tf.tensor_scatter_nd_update(core_values, indices, [entry])))
    return tuple(cores)


def _source_route_seeded_channel_initial_cores(
    *,
    ranks: tuple[int, ...],
    basis_dim: int,
    constant_value: tf.Tensor,
    epsilon: float = P70_SEEDED_CHANNEL_EPSILON,
) -> tuple[TTCore, ...]:
    values = tf.convert_to_tensor(constant_value, dtype=tf.float64)
    if values.shape.rank != 0:
        raise ValueError(f"constant_value: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.math.is_finite(values).numpy()):
        raise ValueError(f"constant_value: {HighDimStatus.NONFINITE_VALUE.value}")
    if bool((values <= 0.0).numpy()):
        raise ValueError(f"constant_value: {HighDimStatus.NONFINITE_VALUE.value}")
    if float(epsilon) <= 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be positive and finite")
    dim = len(ranks) - 1
    if dim <= 0 or int(basis_dim) <= 0:
        raise ValueError(f"basis_dim: {HighDimStatus.INVALID_SHAPE.value}")
    max_rank = max(int(rank) for rank in ranks)
    extra_count = max(max_rank - 1, 0)
    seeded_scale = (
        values * tf.constant(float(epsilon) / float(extra_count), dtype=tf.float64)
        if extra_count > 0
        else tf.constant(0.0, dtype=tf.float64)
    )
    cores = []
    for axis in range(dim):
        left_rank = int(ranks[axis])
        right_rank = int(ranks[axis + 1])
        core_values = tf.zeros(
            [left_rank, int(basis_dim), right_rank],
            dtype=tf.float64,
        )
        updates = []
        indices = []
        if left_rank > 0 and right_rank > 0:
            indices.append([0, 0, 0])
            updates.append(values if axis == 0 else tf.constant(1.0, dtype=tf.float64))
        for channel in range(1, min(left_rank, right_rank)):
            basis_index = _p70_seeded_basis_index(axis=axis, channel=channel, basis_dim=int(basis_dim))
            indices.append([channel, basis_index, channel])
            updates.append(tf.constant(1.0, dtype=tf.float64))
        if axis == 0:
            for channel in range(1, right_rank):
                basis_index = _p70_seeded_basis_index(axis=axis, channel=channel, basis_dim=int(basis_dim))
                indices.append([0, basis_index, channel])
                updates.append(seeded_scale)
        if axis == dim - 1:
            for channel in range(1, left_rank):
                basis_index = _p70_seeded_basis_index(axis=axis, channel=channel, basis_dim=int(basis_dim))
                indices.append([channel, basis_index, 0])
                updates.append(tf.constant(1.0, dtype=tf.float64))
        if indices:
            core_values = tf.tensor_scatter_nd_update(
                core_values,
                tf.constant(indices, dtype=tf.int64),
                tf.stack(updates),
            )
        cores.append(TTCore(core_values))
    return tuple(cores)


def _p70_seeded_basis_index(*, axis: int, channel: int, basis_dim: int) -> int:
    if int(basis_dim) <= 1:
        return 0
    return 1 + ((int(axis) + int(channel) - 1) % (int(basis_dim) - 1))


def _p70_canonical_alternating_sweep_order(target_dim: int) -> tuple[int, ...]:
    dim = int(target_dim)
    if dim <= 0:
        raise ValueError("target_dim must be positive")
    forward = tuple(range(dim))
    return forward + tuple(reversed(forward))


def _p70_row_adequacy_diagnostics(
    *,
    row_count: int,
    target_dim: int,
    fit_degree: int,
    fit_rank: int,
) -> Mapping[str, object]:
    dim = int(target_dim)
    degree = int(fit_degree)
    rank = int(fit_rank)
    n_rows = int(row_count)
    max_core_columns = (degree + 1) * rank * rank
    hard = max(4, math.ceil(dim / 4), max_core_columns)
    preferred = max(dim, 2 * max_core_columns)
    if n_rows < hard:
        status = "branch_fit_row_adequacy_failed"
    elif n_rows < preferred:
        status = "diagnostic_only_below_preferred_rows"
    else:
        status = "ok"
    return {
        "status": status,
        "row_count": n_rows,
        "n_hard": hard,
        "n_preferred": preferred,
        "max_core_columns": max_core_columns,
        "threshold_role": "bayesfilter_fixed_hmc_engineering_safeguard",
        "nonclaim": "row adequacy is not sample coverage or convergence proof",
    }


def _p59_execution_only_row_adequacy_diagnostics(
    *,
    fit_sample_count: int,
    fit_degree: int = 0,
    fit_rank: int = 1,
) -> Mapping[str, object]:
    return _p70_row_adequacy_diagnostics(
        row_count=int(fit_sample_count),
        target_dim=P59_9A_AUTHOR_SIR_TARGET_DIMENSION,
        fit_degree=int(fit_degree),
        fit_rank=int(fit_rank),
    )


def _p70_channel_activity_diagnostics(
    *,
    cores: tuple[TTCore, ...],
    target_dim: int,
    fit_rank: int,
) -> Mapping[str, object]:
    dim = int(target_dim)
    rank = int(fit_rank)
    if len(cores) != dim:
        raise ValueError(f"cores: {HighDimStatus.INVALID_SHAPE.value}")
    scores_by_bond = []
    first_scores: list[float] = []
    for bond in range(max(dim - 1, 0)):
        left = cores[bond].values
        right = cores[bond + 1].values
        bond_scores = []
        channel_count = min(int(left.shape[2]), int(right.shape[0]))
        for channel in range(channel_count):
            left_norm = float(tf.norm(left[:, :, channel]).numpy())
            right_norm = float(tf.norm(right[channel, :, :]).numpy())
            score = left_norm * right_norm
            bond_scores.append(score)
            if channel == 0:
                first_scores.append(score)
        scores_by_bond.append(tuple(bond_scores))
    a_ref = max(first_scores) if first_scores else 0.0
    if not math.isfinite(a_ref) or a_ref <= 0.0:
        threshold = math.inf
        status = "rank_channel_activity_failed"
    else:
        threshold = max(
            P70_CHANNEL_ACTIVITY_ABS_TOL,
            P70_CHANNEL_ACTIVITY_REL_TOL * a_ref,
        )
        status = "ok"
    b_min = max(1, math.ceil(0.25 * max(dim - 1, 0)))
    active_counts = {}
    inactive_channels = []
    for channel in range(1, rank):
        active_count = 0
        for bond_scores in scores_by_bond:
            if channel < len(bond_scores) and bond_scores[channel] >= threshold:
                active_count += 1
        active_counts[channel] = active_count
        if active_count < b_min:
            inactive_channels.append(channel)
    if inactive_channels:
        status = "rank_channel_activity_failed"
    return {
        "status": status,
        "score_by_bond": tuple(scores_by_bond),
        "reference_channel_score": a_ref,
        "activity_threshold": threshold,
        "epsilon_channel_abs": P70_CHANNEL_ACTIVITY_ABS_TOL,
        "epsilon_channel_rel": P70_CHANNEL_ACTIVITY_REL_TOL,
        "minimum_active_bonds": b_min,
        "extra_channel_active_bond_counts": active_counts,
        "inactive_extra_channels": tuple(inactive_channels),
        "gauge_boundary": "stored_deterministic_gauge_only_not_gauge_invariant",
    }


def _p70_fixed_fitting_policy_payload(
    *,
    target_dim: int,
    fit_degree: int,
    fit_rank: int,
    ridge: float = P70_FIT_RIDGE,
    row_adequacy: Mapping[str, object],
    channel_activity: Mapping[str, object],
    stabilization_policy: Mapping[str, object] | None = None,
    stabilization_diagnostics_summary: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    stability = (
        None
        if stabilization_policy is None
        else dict(stabilization_policy)
    )
    stability_summary = (
        None
        if stabilization_diagnostics_summary is None
        else dict(stabilization_diagnostics_summary)
    )
    return {
        "policy_id": "p70_fixed_hmc_seeded_channel_fit_v1",
        "classification": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "initialization_rule": P70_FIXED_BRANCH_INITIALIZATION_RULE,
        "seeded_channel_epsilon": P70_SEEDED_CHANNEL_EPSILON,
        "sweep_order": _p70_canonical_alternating_sweep_order(int(target_dim)),
        "max_sweeps": P70_FIXED_BRANCH_MAX_SWEEPS,
        "ridge": float(ridge),
        "condition_number_warning": P70_CONDITION_NUMBER_WARNING,
        "condition_number_veto": P70_CONDITION_NUMBER_VETO,
        "stabilization_policy": stability,
        "stabilization_diagnostics_summary": stability_summary,
        "stabilization_policy_id": (
            None if stability is None else stability.get("stabilization_policy_id")
        ),
        "solver_backend": None if stability is None else stability.get("solver_backend"),
        "objective_preserving_column_scaling": (
            None if stability is None else stability.get("objective_preserving_column_scaling")
        ),
        "column_scale_floor": None if stability is None else stability.get("column_scale_floor"),
        "transformed_ridge_rule": (
            None if stability is None else stability.get("transformed_ridge_rule")
        ),
        "condition_number_gate_target": (
            None if stability is None else stability.get("condition_number_gate_target")
        ),
        "original_unscaled_normal_condition_role": (
            None
            if stability is None
            else stability.get("original_unscaled_normal_condition_role")
        ),
        "row_adequacy": dict(row_adequacy),
        "channel_activity": dict(channel_activity),
        "normalizer_thresholds": {
            "defensive_only_sqrt_normalizer_tol": P70_DEFENSIVE_ONLY_SQRT_NORMALIZER_TOL,
            "fit_mass_fraction_min": P70_FIT_MASS_FRACTION_MIN,
            "log_increment_abs_bound": P70_LOG_INCREMENT_ABS_BOUND,
        },
        "holdout_replay_thresholds": {
            "normalized_residual_veto": P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO,
        },
        "threshold_role": "bayesfilter_fixed_hmc_engineering_safeguards",
        "nonclaims": (
            "not Zhao-Cui source-faithful theory",
            "not validation evidence",
            "not HMC readiness",
        ),
        "fit_degree": int(fit_degree),
        "fit_rank": int(fit_rank),
    }


def _weighted_mean_target_value(target_values: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    targets = tf.convert_to_tensor(target_values, dtype=tf.float64)
    fit_weights = tf.convert_to_tensor(weights, dtype=tf.float64)
    if targets.shape.rank != 1 or fit_weights.shape.rank != 1:
        raise ValueError(f"target_values: {HighDimStatus.INVALID_SHAPE.value}")
    if int(targets.shape[0]) != int(fit_weights.shape[0]):
        raise ValueError(f"target_values: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(
        tf.reduce_all(tf.math.is_finite(targets)).numpy()
        and tf.reduce_all(tf.math.is_finite(fit_weights)).numpy()
    ):
        raise ValueError(f"target_values: {HighDimStatus.NONFINITE_VALUE.value}")
    if bool(
        tf.reduce_any(targets <= 0.0).numpy()
        or tf.reduce_any(fit_weights < 0.0).numpy()
        or (tf.reduce_sum(fit_weights) <= 0.0).numpy()
    ):
        raise ValueError(f"target_values: {HighDimStatus.NONFINITE_VALUE.value}")
    return tf.reduce_sum(fit_weights * targets) / tf.reduce_sum(fit_weights)


def _source_route_initial_core_values(
    *,
    left_rank: int,
    basis_dim: int,
    right_rank: int,
) -> tf.Tensor:
    values = tf.ones(
        [int(left_rank), int(basis_dim), int(right_rank)],
        dtype=tf.float64,
    )
    scale = tf.cast(max(values.shape.num_elements() or 1, 1), tf.float64)
    return values / scale


def _p59_author_sir_unit_reference_points(sample_count: int, target_dim: int) -> tf.Tensor:
    columns = []
    base = tf.linspace(
        tf.constant(0.15, dtype=tf.float64),
        tf.constant(0.85, dtype=tf.float64),
        int(target_dim),
    )
    for index in range(int(sample_count)):
        shift = tf.cast(index, tf.float64) / tf.cast(max(int(sample_count), 1), tf.float64)
        shifted = tf.math.floormod(base + 0.13 * shift, 0.8) + 0.1
        columns.append(shifted)
    return tf.stack(columns, axis=1)


def _p59_author_sir_defensive_tau_tensor() -> tf.Tensor:
    return tf.constant(P62_AUTHOR_TTSIRT_EXECUTABLE_DEFAULT_TAU, dtype=tf.float64)


def p60_author_sir_same_route_rank_comparator(
    *,
    sample_count: int = 1,
    fit_sample_count: int = P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT,
    low_fit_degree: int = 0,
    high_fit_degree: int = 1,
    low_fit_rank: int = 1,
    high_fit_rank: int = 2,
) -> P60AuthorSIRSameRouteRankComparatorResult:
    """Run the P60-2 same-route rank comparator with fail-closed gates."""

    if int(high_fit_rank) <= int(low_fit_rank):
        return _p60_rank_comparator_block(
            blockers=("high_fit_rank_must_exceed_low_fit_rank",),
            low_result=None,
            high_result=None,
            extra_manifest={
                "low_fit_rank": int(low_fit_rank),
                "high_fit_rank": int(high_fit_rank),
            },
        )

    low: P59AuthorSIRStepSpecAssemblyResult | None = None
    high: P59AuthorSIRStepSpecAssemblyResult | None = None
    blockers: list[str] = []
    try:
        low = p59_author_sir_step_spec_assembly(
            sample_count=int(sample_count),
            fit_sample_count=int(fit_sample_count),
            fit_degree=int(low_fit_degree),
            fit_rank=int(low_fit_rank),
        )
    except Exception as exc:  # pragma: no cover - preserved in manifest
        blockers.append("candidate_low_exception_" + type(exc).__name__ + "_" + str(exc))
    try:
        high = p59_author_sir_step_spec_assembly(
            sample_count=int(sample_count),
            fit_sample_count=int(fit_sample_count),
            fit_degree=int(high_fit_degree),
            fit_rank=int(high_fit_rank),
        )
    except Exception as exc:
        blockers.append("candidate_high_exception_" + type(exc).__name__ + "_" + str(exc))

    if low is not None and low.status != P59_9B_PASS_STATUS:
        blockers.append("candidate_low_failed_" + "_".join(low.blockers))
    if high is not None and high.status != P59_9B_PASS_STATUS:
        blockers.append("candidate_high_failed_" + "_".join(high.blockers))
    if low is None or high is None or blockers:
        return _p60_rank_comparator_block(
            blockers=tuple(blockers) if blockers else ("missing_candidate_row",),
            low_result=low,
            high_result=high,
            extra_manifest={
                "candidate_low": _p60_candidate_manifest_or_none(low),
                "candidate_high": _p60_candidate_manifest_or_none(high),
                "nonclaims": _p60_rank_nonclaims(),
            },
        )

    manifest = _p60_rank_comparator_manifest(low, high)
    veto_blockers = tuple(str(blocker) for blocker in manifest["veto_blockers"])
    if veto_blockers:
        return P60AuthorSIRSameRouteRankComparatorResult(
            status=P60_D18_RANK_CONVERGENCE_BLOCK_STATUS,
            blockers=veto_blockers,
            low_result=low,
            high_result=high,
            manifest=manifest,
        )
    return P60AuthorSIRSameRouteRankComparatorResult(
        status=P60_D18_RANK_CONVERGENCE_PASS_STATUS,
        blockers=(),
        low_result=low,
        high_result=high,
        manifest=manifest,
    )


def p66_fixed_branch_sample_adequacy(
    *,
    target_dim: int = P59_9A_AUTHOR_SIR_TARGET_DIMENSION,
    fit_degree: int,
    fit_rank: int,
    fit_sample_count: int,
    diagnostic_min_multiplier: int = 2,
    preferred_multiplier: int = 4,
) -> Mapping[str, object]:
    """Return the reviewed P66 fixed-branch sample-adequacy heuristic."""

    degree = int(fit_degree)
    rank = int(fit_rank)
    sample_count = int(fit_sample_count)
    if degree < 0:
        raise ValueError("fit_degree must be nonnegative")
    if rank <= 0:
        raise ValueError("fit_rank must be positive")
    if sample_count <= 0:
        raise ValueError("fit_sample_count must be positive")
    if int(diagnostic_min_multiplier) <= 0:
        raise ValueError("diagnostic_min_multiplier must be positive")
    if int(preferred_multiplier) <= 0:
        raise ValueError("preferred_multiplier must be positive")
    ranks = _source_route_rank_tuple(int(target_dim), rank)
    max_core_columns = max(
        int(ranks[axis]) * (degree + 1) * int(ranks[axis + 1])
        for axis in range(len(ranks) - 1)
    )
    diagnostic_minimum = int(diagnostic_min_multiplier) * max_core_columns
    preferred = int(preferred_multiplier) * max_core_columns
    status = (
        P66_SAMPLE_ADEQUATE_STATUS
        if sample_count >= diagnostic_minimum
        else P66_FIT_DESIGN_UNDERDETERMINED_BLOCK_STATUS
    )
    return {
        "fit_degree": degree,
        "fit_rank": rank,
        "rank_tuple": ranks,
        "fit_sample_count": sample_count,
        "max_core_columns": max_core_columns,
        "diagnostic_min_multiplier": int(diagnostic_min_multiplier),
        "preferred_multiplier": int(preferred_multiplier),
        "diagnostic_min_fit_samples": diagnostic_minimum,
        "preferred_fit_samples": preferred,
        "status": status,
        "heuristic_scope": "P66 fixed-branch comparator only",
        "nonclaim": "sample adequacy is permission to diagnose, not convergence",
    }


def p66_fixed_branch_fit_budget_resolution(
    *,
    candidate_fit_sample_count: int | None = None,
    candidate_fit_degree: int = 1,
    candidate_fit_rank: int = 2,
    diagnostic_min_multiplier: int = 2,
    preferred_multiplier: int = 4,
    rank_ladder_fit_sample_count: int | None = None,
    degree_ladder_fit_sample_count: int | None = None,
) -> Mapping[str, object]:
    """Resolve P66 candidate/rank/degree fit budgets without running the route."""

    candidate_degree = int(candidate_fit_degree)
    candidate_rank = int(candidate_fit_rank)
    candidate_min = _p66_diagnostic_minimum(
        fit_degree=candidate_degree,
        fit_rank=candidate_rank,
        diagnostic_min_multiplier=int(diagnostic_min_multiplier),
        preferred_multiplier=int(preferred_multiplier),
    )
    rank_min = _p66_diagnostic_minimum(
        fit_degree=candidate_degree,
        fit_rank=candidate_rank + 1,
        diagnostic_min_multiplier=int(diagnostic_min_multiplier),
        preferred_multiplier=int(preferred_multiplier),
    )
    degree_min = _p66_diagnostic_minimum(
        fit_degree=candidate_degree + 1,
        fit_rank=candidate_rank,
        diagnostic_min_multiplier=int(diagnostic_min_multiplier),
        preferred_multiplier=int(preferred_multiplier),
    )
    candidate_resolved = (
        candidate_min
        if candidate_fit_sample_count is None
        else int(candidate_fit_sample_count)
    )
    rank_resolved = (
        max(candidate_min, rank_min)
        if rank_ladder_fit_sample_count is None
        else int(rank_ladder_fit_sample_count)
    )
    degree_resolved = (
        max(candidate_min, degree_min)
        if degree_ladder_fit_sample_count is None
        else int(degree_ladder_fit_sample_count)
    )
    return _p66_fit_budget_resolution(
        candidate_user=candidate_fit_sample_count,
        candidate_resolved=candidate_resolved,
        rank_user=rank_ladder_fit_sample_count,
        rank_resolved=rank_resolved,
        degree_user=degree_ladder_fit_sample_count,
        degree_resolved=degree_resolved,
        candidate_min=candidate_min,
        rank_min=rank_min,
        degree_min=degree_min,
    )


def p66_author_sir_fixed_branch_validation_ladder(
    *,
    sample_count: int = 1,
    sentinel_fit_sample_count: int = 2,
    candidate_fit_sample_count: int | None = None,
    candidate_fit_degree: int = 1,
    candidate_fit_rank: int = 2,
    diagnostic_min_multiplier: int = 2,
    preferred_multiplier: int = 4,
    rank_ladder_fit_sample_count: int | None = None,
    degree_ladder_fit_sample_count: int | None = None,
    execute_adjacent_ladders: bool = False,
    candidate_result: P59AuthorSIRStepSpecAssemblyResult | None = None,
    sentinel_result: P60AuthorSIRSameRouteRankComparatorResult | None = None,
) -> P66AuthorSIRFixedBranchValidationLadderResult:
    """Build the P66 fixed-branch validation-ladder schema and gates."""

    if candidate_result is not None and not isinstance(
        candidate_result,
        P59AuthorSIRStepSpecAssemblyResult,
    ):
        raise TypeError("candidate_result must be P59AuthorSIRStepSpecAssemblyResult")
    if sentinel_result is not None and not isinstance(
        sentinel_result,
        P60AuthorSIRSameRouteRankComparatorResult,
    ):
        raise TypeError("sentinel_result must be P60AuthorSIRSameRouteRankComparatorResult")

    if int(sample_count) <= 0:
        return _p66_block_result(
            status=P66_VALIDATION_LADDER_SCOPE_BLOCK_STATUS,
            blockers=("sample_count_must_be_positive",),
            candidate_result=None,
            sentinel_result=None,
            manifest_extra={},
        )
    if execute_adjacent_ladders:
        return _p66_block_result(
            status=P66_VALIDATION_LADDER_SCOPE_BLOCK_STATUS,
            blockers=("execute_adjacent_ladders_requires_reviewed_experiment_plan",),
            candidate_result=None,
            sentinel_result=None,
            manifest_extra={
                "execute_adjacent_ladders": True,
                "schema_only_reason": (
                    "Phase 2 does not execute adjacent ladders without a reviewed "
                    "experiment plan."
                ),
            },
        )
    candidate_degree = int(candidate_fit_degree)
    candidate_rank = int(candidate_fit_rank)
    rank_ladder_degree = candidate_degree
    rank_ladder_rank = candidate_rank + 1
    degree_ladder_degree = candidate_degree + 1
    degree_ladder_rank = candidate_rank
    fit_budget_resolution = p66_fixed_branch_fit_budget_resolution(
        candidate_fit_sample_count=candidate_fit_sample_count,
        candidate_fit_degree=candidate_degree,
        candidate_fit_rank=candidate_rank,
        diagnostic_min_multiplier=int(diagnostic_min_multiplier),
        preferred_multiplier=int(preferred_multiplier),
        rank_ladder_fit_sample_count=rank_ladder_fit_sample_count,
        degree_ladder_fit_sample_count=degree_ladder_fit_sample_count,
    )
    resolved_candidate_count = int(
        fit_budget_resolution["candidate"]["resolved_fit_sample_count"]
    )
    resolved_rank_count = int(
        fit_budget_resolution["rank_ladder"]["resolved_fit_sample_count"]
    )
    resolved_degree_count = int(
        fit_budget_resolution["degree_ladder"]["resolved_fit_sample_count"]
    )
    if resolved_candidate_count <= 0 or resolved_rank_count <= 0 or resolved_degree_count <= 0:
        return _p66_block_result(
            status=P66_VALIDATION_LADDER_SCOPE_BLOCK_STATUS,
            blockers=("fit_sample_counts_must_be_positive",),
            candidate_result=None,
            sentinel_result=None,
            manifest_extra={
                "fit_budget_resolution": fit_budget_resolution,
            },
        )

    sentinel = sentinel_result or p60_author_sir_same_route_rank_comparator(
        sample_count=int(sample_count),
        fit_sample_count=int(sentinel_fit_sample_count),
        low_fit_degree=0,
        high_fit_degree=candidate_degree,
        low_fit_rank=1,
        high_fit_rank=candidate_rank,
    )
    candidate: P59AuthorSIRStepSpecAssemblyResult | None = candidate_result
    blockers: list[str] = []
    if candidate is None:
        try:
            candidate = p59_author_sir_step_spec_assembly(
                sample_count=int(sample_count),
                fit_sample_count=resolved_candidate_count,
                fit_degree=candidate_degree,
                fit_rank=candidate_rank,
            )
        except Exception as exc:  # pragma: no cover - preserved in manifest
            blockers.append("candidate_exception_" + type(exc).__name__ + "_" + str(exc))
    if candidate is not None and candidate.status != P59_9B_PASS_STATUS:
        blockers.append("candidate_failed_" + "_".join(candidate.blockers))
    if candidate is None or blockers:
        return _p66_block_result(
            status=P66_VALIDATION_LADDER_SCOPE_BLOCK_STATUS,
            blockers=tuple(blockers) if blockers else ("missing_candidate_row",),
            candidate_result=candidate,
            sentinel_result=sentinel,
            manifest_extra={
                "fit_budget_resolution": fit_budget_resolution,
                "old_p60_sentinel_payload": sentinel.manifest,
            },
        )

    assert candidate.sequential_result is not None
    candidate_payload = _p66_candidate_payload(candidate, resolved_candidate_count)
    candidate_admissibility_status = str(candidate_payload["admissibility_status"])
    source_invariant_check = _p66_check_expected_source_invariants(candidate)
    comparison_invariants = {
        "candidate_to_rank_ladder": _p66_comparison_invariants(
            candidate=candidate,
            stronger_result=None,
            authorized_field="fit_rank",
            authorized_reason="adjacent rank ladder compares fixed degree with one higher TT rank",
        ),
        "candidate_to_degree_ladder": _p66_comparison_invariants(
            candidate=candidate,
            stronger_result=None,
            authorized_field="fit_degree",
            authorized_reason="adjacent degree ladder compares fixed rank with one higher polynomial degree",
        ),
    }
    rank_adequacy = p66_fixed_branch_sample_adequacy(
        fit_degree=rank_ladder_degree,
        fit_rank=rank_ladder_rank,
        fit_sample_count=resolved_rank_count,
        diagnostic_min_multiplier=int(diagnostic_min_multiplier),
        preferred_multiplier=int(preferred_multiplier),
    )
    degree_adequacy = p66_fixed_branch_sample_adequacy(
        fit_degree=degree_ladder_degree,
        fit_rank=degree_ladder_rank,
        fit_sample_count=resolved_degree_count,
        diagnostic_min_multiplier=int(diagnostic_min_multiplier),
        preferred_multiplier=int(preferred_multiplier),
    )
    rank_ladder = _p66_schema_only_ladder_payload(
        ladder_kind="rank",
        candidate=candidate,
        candidate_fit_sample_count=resolved_candidate_count,
        stronger_degree=rank_ladder_degree,
        stronger_rank=rank_ladder_rank,
        stronger_fit_sample_count=resolved_rank_count,
        adequacy=rank_adequacy,
        status=P66_RANK_LADDER_SCHEMA_ONLY_STATUS,
    )
    degree_ladder = _p66_schema_only_ladder_payload(
        ladder_kind="degree",
        candidate=candidate,
        candidate_fit_sample_count=resolved_candidate_count,
        stronger_degree=degree_ladder_degree,
        stronger_rank=degree_ladder_rank,
        stronger_fit_sample_count=resolved_degree_count,
        adequacy=degree_adequacy,
        status=P66_DEGREE_LADDER_SCHEMA_ONLY_STATUS,
    )
    if not bool(source_invariant_check["passed"]):
        return _p66_result_from_manifest(
            status=P66_SOURCE_ROUTE_INVARIANT_DRIFT_BLOCK_STATUS,
            blockers=tuple(source_invariant_check["blockers"]),
            candidate=candidate,
            sentinel=sentinel,
            manifest=_p66_manifest(
                status=P66_SOURCE_ROUTE_INVARIANT_DRIFT_BLOCK_STATUS,
                blockers=tuple(source_invariant_check["blockers"]),
                sample_count=int(sample_count),
                sentinel_fit_sample_count=int(sentinel_fit_sample_count),
                candidate=candidate_payload,
                sentinel=sentinel,
                sample_adequacy=candidate_payload["sample_adequacy"],
                fit_budget_resolution=fit_budget_resolution,
                rank_ladder=rank_ladder,
                degree_ladder=degree_ladder,
                source_invariants=source_invariant_check,
                comparison_invariants=comparison_invariants,
            ),
        )
    if candidate_admissibility_status == P66_FIXED_BRANCH_DEFENSIVE_ONLY_BLOCK_STATUS:
        return _p66_result_from_manifest(
            status=P66_FIXED_BRANCH_DEFENSIVE_ONLY_BLOCK_STATUS,
            blockers=("candidate_fixed_branch_defensive_only",),
            candidate=candidate,
            sentinel=sentinel,
            manifest=_p66_manifest(
                status=P66_FIXED_BRANCH_DEFENSIVE_ONLY_BLOCK_STATUS,
                blockers=("candidate_fixed_branch_defensive_only",),
                sample_count=int(sample_count),
                sentinel_fit_sample_count=int(sentinel_fit_sample_count),
                candidate=candidate_payload,
                sentinel=sentinel,
                sample_adequacy=candidate_payload["sample_adequacy"],
                fit_budget_resolution=fit_budget_resolution,
                rank_ladder=rank_ladder,
                degree_ladder=degree_ladder,
                source_invariants=source_invariant_check,
                comparison_invariants=comparison_invariants,
            ),
        )

    candidate_adequacy = candidate_payload["sample_adequacy"]
    sample_adequacy_status = (
        P66_SAMPLE_ADEQUATE_STATUS
        if (
            candidate_adequacy["status"] == P66_SAMPLE_ADEQUATE_STATUS
            and rank_adequacy["status"] == P66_SAMPLE_ADEQUATE_STATUS
            and degree_adequacy["status"] == P66_SAMPLE_ADEQUATE_STATUS
        )
        else P66_FIT_DESIGN_UNDERDETERMINED_BLOCK_STATUS
    )
    status = (
        P66_FIXED_BRANCH_VALIDATION_LADDER_READY_STATUS
        if sample_adequacy_status == P66_SAMPLE_ADEQUATE_STATUS
        else P66_FIT_DESIGN_UNDERDETERMINED_BLOCK_STATUS
    )
    result_blockers: tuple[str, ...] = ()
    if status == P66_FIT_DESIGN_UNDERDETERMINED_BLOCK_STATUS:
        result_blockers = ("fit_design_underdetermined_for_adjacent_ladder_diagnostics",)

    manifest = _p66_manifest(
        status=status,
        blockers=result_blockers,
        sample_count=int(sample_count),
        sentinel_fit_sample_count=int(sentinel_fit_sample_count),
        candidate=candidate_payload,
        sentinel=sentinel,
        sample_adequacy={
            "candidate": candidate_adequacy,
            "rank_ladder": rank_adequacy,
            "degree_ladder": degree_adequacy,
            "status": sample_adequacy_status,
        },
        fit_budget_resolution=fit_budget_resolution,
        rank_ladder=rank_ladder,
        degree_ladder=degree_ladder,
        source_invariants=source_invariant_check,
        comparison_invariants=comparison_invariants,
    )
    return _p66_result_from_manifest(
        status=status,
        blockers=result_blockers,
        candidate=candidate,
        sentinel=sentinel,
        manifest=manifest,
    )


def _p60_rank_comparator_manifest(
    low: P59AuthorSIRStepSpecAssemblyResult,
    high: P59AuthorSIRStepSpecAssemblyResult,
) -> Mapping[str, object]:
    assert low.sequential_result is not None
    assert high.sequential_result is not None
    blockers: list[str] = []
    low_seq = low.sequential_result
    high_seq = high.sequential_result
    if low.manifest["target_dimension"] != high.manifest["target_dimension"]:
        blockers.append("target_dimension_mismatch")
    if low.manifest["source_target_order"] != high.manifest["source_target_order"]:
        blockers.append("source_target_order_mismatch")
    if low.manifest["previous_marginal_keep_axes"] != high.manifest["previous_marginal_keep_axes"]:
        blockers.append("previous_marginal_keep_axes_mismatch")
    if low.manifest["previous_marginal_input_axes"] != high.manifest["previous_marginal_input_axes"]:
        blockers.append("previous_marginal_input_axes_mismatch")
    low_rank = int(low.manifest.get("fit_rank", 1))
    high_rank = int(high.manifest.get("fit_rank", 1))
    if high_rank <= low_rank:
        blockers.append("high_rank_not_strictly_stronger")
    log_delta = abs(
        float(high_seq.log_marginal_likelihood.numpy())
        - float(low_seq.log_marginal_likelihood.numpy())
    )
    normalizer_deltas = tuple(
        abs(float(high_step.normalizer_increment.numpy()) - float(low_step.normalizer_increment.numpy()))
        for low_step, high_step in zip(low_seq.steps, high_seq.steps)
    )
    low_normalizer_terms = _p64_normalizer_terms_by_step(low_seq)
    high_normalizer_terms = _p64_normalizer_terms_by_step(high_seq)
    low_collapsed = _p64_defensive_only_steps(low_normalizer_terms)
    high_collapsed = _p64_defensive_only_steps(high_normalizer_terms)
    low_core_terms = _p65_sqrt_tt_core_diagnostics_by_step(low_seq)
    high_core_terms = _p65_sqrt_tt_core_diagnostics_by_step(high_seq)
    if low_collapsed:
        blockers.append("candidate_low_defensive_only_transport")
    if high_collapsed:
        blockers.append("candidate_high_defensive_only_transport")
    probe_delta = _p60_probe_log_density_delta(low, high)
    retained_delta = _p60_retained_log_density_delta(low, high)
    thresholds = {
        "log_marginal_abs_delta": 5.0,
        "normalizer_increment_abs_delta": 5.0,
        "probe_log_density_median_abs_delta": 10.0,
        "retained_log_density_median_abs_delta": 10.0,
    }
    if log_delta > thresholds["log_marginal_abs_delta"]:
        blockers.append("log_marginal_delta_threshold_exceeded")
    if any(delta > thresholds["normalizer_increment_abs_delta"] for delta in normalizer_deltas):
        blockers.append("normalizer_increment_delta_threshold_exceeded")
    if probe_delta > thresholds["probe_log_density_median_abs_delta"]:
        blockers.append("probe_log_density_delta_threshold_exceeded")
    if retained_delta is None:
        blockers.append("retained_density_delta_unavailable")
    elif retained_delta > thresholds["retained_log_density_median_abs_delta"]:
        blockers.append("retained_log_density_delta_threshold_exceeded")
    return {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P60-2",
        "artifact_role": "same_route_rank_comparator",
        "status": (
            P60_D18_RANK_CONVERGENCE_BLOCK_STATUS
            if blockers
            else P60_D18_RANK_CONVERGENCE_PASS_STATUS
        ),
        "candidate_low": _p60_candidate_manifest_or_none(low),
        "candidate_high": _p60_candidate_manifest_or_none(high),
        "log_marginal_likelihoods": (
            float(low_seq.log_marginal_likelihood.numpy()),
            float(high_seq.log_marginal_likelihood.numpy()),
        ),
        "log_marginal_abs_delta": log_delta,
        "normalizer_increment_abs_deltas": normalizer_deltas,
        "normalizer_decomposition": {
            "candidate_low": low_normalizer_terms,
            "candidate_high": high_normalizer_terms,
            "defensive_only_sqrt_normalizer_tol": P64_DEFENSIVE_ONLY_SQRT_NORMALIZER_TOL,
            "candidate_low_defensive_only_steps": low_collapsed,
            "candidate_high_defensive_only_steps": high_collapsed,
        },
        "sqrt_tt_core_diagnostics": {
            "candidate_low": low_core_terms,
            "candidate_high": high_core_terms,
            "zero_sqrt_tt_core_norm_tol": P65_ZERO_SQRT_TT_CORE_NORM_TOL,
        },
        "effective_sample_size_by_step": {
            "candidate_low": _p60_ess_by_step(low_seq),
            "candidate_high": _p60_ess_by_step(high_seq),
        },
        "correction_log_weight_ranges": {
            "candidate_low": _p60_correction_ranges(low_seq),
            "candidate_high": _p60_correction_ranges(high_seq),
        },
        "probe_log_density_median_abs_delta": probe_delta,
        "retained_log_density_median_abs_delta": retained_delta,
        "thresholds": thresholds,
        "veto_blockers": tuple(blockers),
        "source_invariants": {
            "route": "Zhao-Cui full_sol",
            "realized_target": "[x_t, x_{t-1}]",
            "target_dimension": low.manifest["target_dimension"],
            "previous_marginal_keep_axes": low.manifest["previous_marginal_keep_axes"],
            "previous_marginal_input_axes": low.manifest["previous_marginal_input_axes"],
        },
        "nonclaims": _p60_rank_nonclaims(),
    }


def _p60_probe_log_density_delta(
    low: P59AuthorSIRStepSpecAssemblyResult,
    high: P59AuthorSIRStepSpecAssemblyResult,
) -> float:
    deltas = []
    for low_spec, high_spec in zip(low.step_specs, high.step_specs):
        probes = _p59_author_sir_reference_points(
            3,
            low_spec.target.coordinate_frame.dimension,
        )
        low_values = low_spec.transport.eval_pdf(probes)
        high_values = high_spec.transport.eval_pdf(probes)
        deltas.append(tf.abs(tf.math.log(low_values) - tf.math.log(high_values)))
    joined = tf.concat(deltas, axis=0)
    return _p60_tensor_median_float(joined)


def _p60_retained_log_density_delta(
    low: P59AuthorSIRStepSpecAssemblyResult,
    high: P59AuthorSIRStepSpecAssemblyResult,
) -> float | None:
    assert low.sequential_result is not None
    assert high.sequential_result is not None
    deltas = []
    for low_step, high_step in zip(low.sequential_result.steps, high.sequential_result.steps):
        physical = low_step.retained_object.samples
        low_local = tf.transpose(
            tf.linalg.solve(
                low_step.retained_object.coordinate_frame.matrix,
                physical - low_step.retained_object.coordinate_frame.mu[:, tf.newaxis],
            )
        )
        high_local = tf.transpose(
            tf.linalg.solve(
                high_step.retained_object.coordinate_frame.matrix,
                physical - high_step.retained_object.coordinate_frame.mu[:, tf.newaxis],
            )
        )
        low_density = low_step.retained_object.transport_object.density.normalized_retained_density_values(
            tuple(range(low_step.retained_object.coordinate_frame.dimension)),
            low_local,
        )
        high_density = high_step.retained_object.transport_object.density.normalized_retained_density_values(
            tuple(range(high_step.retained_object.coordinate_frame.dimension)),
            high_local,
        )
        deltas.append(tf.abs(tf.math.log(low_density) - tf.math.log(high_density)))
    if not deltas:
        return None
    return _p60_tensor_median_float(tf.concat(deltas, axis=0))


def _p60_tensor_median_float(values: tf.Tensor) -> float:
    vector = tf.reshape(tf.convert_to_tensor(values, dtype=tf.float64), [-1])
    if int(vector.shape[0]) == 0:
        raise ValueError("median requires nonempty values")
    if not bool(tf.reduce_all(tf.math.is_finite(vector)).numpy()):
        return float("inf")
    ordered = tf.sort(vector)
    return float(ordered[int(vector.shape[0]) // 2].numpy())


def _p64_normalizer_terms_by_step(
    sequential: SourceRouteSequentialResult,
) -> tuple[Mapping[str, object], ...]:
    terms = []
    for step in sequential.steps:
        transport = step.retained_object.transport_object
        density = getattr(transport, "density", None)
        if density is None or not callable(getattr(density, "sqrt_square_normalizer", None)):
            terms.append(
                {
                    "time_index": int(step.time_index),
                    "available": False,
                    "normalizer_increment": float(step.normalizer_increment.numpy()),
                }
            )
            continue
        sqrt_square = tf.convert_to_tensor(
            density.sqrt_square_normalizer(),
            dtype=tf.float64,
        )
        defensive_z = tf.convert_to_tensor(
            density.defensive_density.normalizer(density.measure_convention.mass_measure),
            dtype=tf.float64,
        )
        tau = tf.convert_to_tensor(density.tau, dtype=tf.float64)
        mixture_z = tf.convert_to_tensor(density.normalizer(), dtype=tf.float64)
        terms.append(
            {
                "time_index": int(step.time_index),
                "available": True,
                "sqrt_square_normalizer": float(sqrt_square.numpy()),
                "defensive_normalizer": float(defensive_z.numpy()),
                "tau": float(tau.numpy()),
                "defensive_contribution": float((tau * defensive_z).numpy()),
                "mixture_normalizer": float(mixture_z.numpy()),
                "log_transport_normalizer": float(tf.math.log(mixture_z).numpy()),
                "shift_constant": float(step.target.shift_constant.numpy()),
                "coordinate_frame_log_abs_det": float(
                    step.target.coordinate_frame.log_abs_det().numpy()
                ),
                "normalizer_increment": float(step.normalizer_increment.numpy()),
            }
        )
    return tuple(terms)


def _p64_defensive_only_steps(
    terms_by_step: tuple[Mapping[str, object], ...],
) -> tuple[int, ...]:
    collapsed = []
    for row in terms_by_step:
        if not bool(row.get("available", False)):
            continue
        sqrt_square = float(row["sqrt_square_normalizer"])
        if sqrt_square <= P64_DEFENSIVE_ONLY_SQRT_NORMALIZER_TOL:
            collapsed.append(int(row["time_index"]))
    return tuple(collapsed)


def _p65_sqrt_tt_core_diagnostics_by_step(
    sequential: SourceRouteSequentialResult,
) -> tuple[Mapping[str, object], ...]:
    rows = []
    tol = float(P65_ZERO_SQRT_TT_CORE_NORM_TOL)
    for step in sequential.steps:
        transport = step.retained_object.transport_object
        density = getattr(transport, "density", None)
        sqrt_tt = getattr(density, "sqrt_tt", None)
        if sqrt_tt is None:
            rows.append(
                {
                    "time_index": int(step.time_index),
                    "available": False,
                }
            )
            continue
        core_norms = tuple(
            float(tf.linalg.norm(tf.reshape(core.values, [-1])).numpy())
            for core in sqrt_tt.cores
        )
        core_max_abs = tuple(
            float(tf.reduce_max(tf.abs(core.values)).numpy())
            for core in sqrt_tt.cores
        )
        nonzero_entries = tuple(
            int(tf.math.count_nonzero(core.values).numpy())
            for core in sqrt_tt.cores
        )
        branch_identity = getattr(sqrt_tt, "branch_identity", None)
        branch_hash = (
            None
            if branch_identity is None
            else branch_identity.hash.value
        )
        rows.append(
            {
                "time_index": int(step.time_index),
                "available": True,
                "core_count": len(core_norms),
                "core_norm_min": min(core_norms),
                "core_norm_max": max(core_norms),
                "core_max_abs_min": min(core_max_abs),
                "core_max_abs_max": max(core_max_abs),
                "zero_core_count": sum(1 for value in core_norms if value == 0.0),
                "near_zero_core_count": sum(1 for value in core_norms if value <= tol),
                "nonzero_entries_min": min(nonzero_entries),
                "nonzero_entries_max": max(nonzero_entries),
                "zero_core_norm_tol": tol,
                "sqrt_tt_branch_hash": branch_hash,
            }
        )
    return tuple(rows)


def _p60_ess_by_step(sequential: SourceRouteSequentialResult) -> tuple[float, ...]:
    return tuple(
        float(step.retained_samples.diagnostics.effective_sample_size.numpy())
        for step in sequential.steps
    )


def _p60_correction_ranges(sequential: SourceRouteSequentialResult) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            float(tf.reduce_min(step.retained_samples.correction_log_weights).numpy()),
            float(tf.reduce_max(step.retained_samples.correction_log_weights).numpy()),
        )
        for step in sequential.steps
    )


def _p60_candidate_manifest_or_none(
    result: P59AuthorSIRStepSpecAssemblyResult | None,
) -> Mapping[str, object] | None:
    if result is None:
        return None
    return {
        "status": result.status,
        "blockers": result.blockers,
        "fit_rank": result.manifest.get("fit_rank"),
        "fit_degree": result.manifest.get("fit_degree"),
        "rank_tuple": result.manifest.get("rank_tuple"),
        "fit_branch_hashes": result.manifest.get("fit_branch_hashes"),
        "density_branch_hashes": result.manifest.get("density_branch_hashes"),
        "fixed_branch_adaptation_class": result.manifest.get(
            "fixed_branch_adaptation_class"
        ),
        "fit_initialization_rule": result.manifest.get("fit_initialization_rule"),
        "fit_initialization_rule_source": result.manifest.get(
            "fit_initialization_rule_source"
        ),
        "defensive_tau": result.manifest.get("defensive_tau"),
        "defensive_tau_source": result.manifest.get("defensive_tau_source"),
        "target_dimension": result.manifest.get("target_dimension"),
        "source_target_order": result.manifest.get("source_target_order"),
    }


def _p60_rank_nonclaims() -> tuple[str, ...]:
    return (
        "same-route rank convergence is not exact correctness",
        "no d18 correctness candidate claim",
        "no d50 or d100 scaling claim",
        "no HMC production readiness claim",
        "no adaptive Zhao-Cui parity claim",
        "no UKF correctness comparator",
    )


def _p66_rank_nonclaims() -> tuple[str, ...]:
    return (
        "old P60 low-high comparison is sentinel evidence only",
        "sample adequacy is not convergence",
        "admissibility is not d18 correctness",
        "schema-only adjacent ladders are not stability evidence",
        "no d18 correctness claim",
        "no d50 or d100 scaling claim",
        "no HMC production readiness claim",
        "no adaptive Zhao-Cui parity claim",
    )


def _p66_diagnostic_minimum(
    *,
    fit_degree: int,
    fit_rank: int,
    diagnostic_min_multiplier: int,
    preferred_multiplier: int,
) -> int:
    return int(
        p66_fixed_branch_sample_adequacy(
            fit_degree=int(fit_degree),
            fit_rank=int(fit_rank),
            fit_sample_count=1,
            diagnostic_min_multiplier=int(diagnostic_min_multiplier),
            preferred_multiplier=int(preferred_multiplier),
        )["diagnostic_min_fit_samples"]
    )


def _p66_fit_budget_resolution(
    *,
    candidate_user: int | None,
    candidate_resolved: int,
    rank_user: int | None,
    rank_resolved: int,
    degree_user: int | None,
    degree_resolved: int,
    candidate_min: int,
    rank_min: int,
    degree_min: int,
) -> Mapping[str, object]:
    return {
        "candidate": {
            "user_supplied_fit_sample_count": candidate_user,
            "resolved_fit_sample_count": int(candidate_resolved),
            "diagnostic_min_fit_samples": int(candidate_min),
            "resolution_rule": (
                "user_supplied"
                if candidate_user is not None
                else "candidate diagnostic minimum"
            ),
        },
        "rank_ladder": {
            "user_supplied_fit_sample_count": rank_user,
            "resolved_fit_sample_count": int(rank_resolved),
            "diagnostic_min_fit_samples": int(rank_min),
            "resolution_rule": (
                "user_supplied"
                if rank_user is not None
                else "max(candidate diagnostic minimum, rank-ladder diagnostic minimum)"
            ),
        },
        "degree_ladder": {
            "user_supplied_fit_sample_count": degree_user,
            "resolved_fit_sample_count": int(degree_resolved),
            "diagnostic_min_fit_samples": int(degree_min),
            "resolution_rule": (
                "user_supplied"
                if degree_user is not None
                else "max(candidate diagnostic minimum, degree-ladder diagnostic minimum)"
            ),
        },
        "recorded_before_interpretation": True,
    }


def _p66_candidate_payload(
    result: P59AuthorSIRStepSpecAssemblyResult,
    fit_sample_count: int,
) -> Mapping[str, object]:
    assert result.sequential_result is not None
    normalizer_terms = _p64_normalizer_terms_by_step(result.sequential_result)
    defensive_only_steps = _p64_defensive_only_steps(normalizer_terms)
    core_terms = _p65_sqrt_tt_core_diagnostics_by_step(result.sequential_result)
    sample_adequacy = p66_fixed_branch_sample_adequacy(
        fit_degree=int(result.manifest["fit_degree"]),
        fit_rank=int(result.manifest["fit_rank"]),
        fit_sample_count=int(fit_sample_count),
    )
    admissibility_status = (
        P66_FIXED_BRANCH_DEFENSIVE_ONLY_BLOCK_STATUS
        if defensive_only_steps
        else P66_CANDIDATE_ADMISSIBLE_STATUS
    )
    return {
        "status": result.status,
        "blockers": result.blockers,
        "degree": result.manifest.get("fit_degree"),
        "rank": result.manifest.get("fit_rank"),
        "rank_tuple": result.manifest.get("rank_tuple"),
        "fit_sample_count": int(fit_sample_count),
        "fit_branch_hashes": result.manifest.get("fit_branch_hashes"),
        "density_branch_hashes": result.manifest.get("density_branch_hashes"),
        "square_root_normalizers_by_step": tuple(
            row.get("sqrt_square_normalizer") for row in normalizer_terms
        ),
        "normalizer_terms_by_step": normalizer_terms,
        "defensive_only_steps": defensive_only_steps,
        "core_diagnostics_by_step": core_terms,
        "near_zero_core_counts": tuple(
            row.get("near_zero_core_count") for row in core_terms if row.get("available")
        ),
        "effective_sample_size_by_step": _p60_ess_by_step(result.sequential_result),
        "correction_log_weight_ranges": _p60_correction_ranges(result.sequential_result),
        "fixed_branch_adaptation_class": result.manifest.get(
            "fixed_branch_adaptation_class"
        ),
        "fit_initialization_rule": result.manifest.get("fit_initialization_rule"),
        "fit_data_mode": result.manifest.get("fit_data_mode"),
        "defensive_tau": result.manifest.get("defensive_tau"),
        "sample_adequacy": sample_adequacy,
        "admissibility_status": admissibility_status,
    }


def _p66_expected_source_invariants(
    result: P59AuthorSIRStepSpecAssemblyResult,
) -> Mapping[str, object]:
    return {
        "route_family": "Zhao-Cui full_sol",
        "source_authority": "P59/P60 fixed-HMC adaptation of author source route",
        "realized_target": "[x_t, x_{t-1}]",
        "target_dimension": P59_9A_AUTHOR_SIR_TARGET_DIMENSION,
        "source_target_order": "[theta, x_t, x_{t-1}]",
        "previous_marginal_keep_axes": tuple(range(18)),
        "previous_marginal_input_axes": tuple(range(18, 36)),
        "fit_data_mode": P63_AUTHOR_SIR_SOURCE_FIT_DATA_MODE,
        "defensive_tau": P62_AUTHOR_TTSIRT_EXECUTABLE_DEFAULT_TAU,
        "initialization_rule": P65_FIXED_BRANCH_INITIALIZATION_RULE,
        "fixed_hmc_adaptation_class": P65_FIXED_BRANCH_ADAPTATION_CLASS,
        "fit_sample_budget_policy": "P66 reviewed diagnostic-minimum resolution",
        "sample_adequacy_rule": (
            "max_axis ranks[axis] * (degree + 1) * ranks[axis+1], "
            "diagnostic minimum multiplier 2 by default"
        ),
        "diagnostic_threshold_definitions": "P60 thresholds preserved for sentinel only",
        "source_anchors": result.manifest.get("source_anchors"),
    }


def _p66_check_expected_source_invariants(
    result: P59AuthorSIRStepSpecAssemblyResult,
) -> Mapping[str, object]:
    expected = _p66_expected_source_invariants(result)
    observed = {
        "route_family": "Zhao-Cui full_sol",
        "source_authority": "P59/P60 fixed-HMC adaptation of author source route",
        "realized_target": "[x_t, x_{t-1}]",
        "target_dimension": result.manifest.get("target_dimension"),
        "source_target_order": result.manifest.get("source_target_order"),
        "previous_marginal_keep_axes": result.manifest.get("previous_marginal_keep_axes"),
        "previous_marginal_input_axes": result.manifest.get("previous_marginal_input_axes"),
        "fit_data_mode": result.manifest.get("fit_data_mode"),
        "defensive_tau": result.manifest.get("defensive_tau"),
        "initialization_rule": result.manifest.get("fit_initialization_rule"),
        "fixed_hmc_adaptation_class": result.manifest.get(
            "fixed_branch_adaptation_class"
        ),
        "fit_sample_budget_policy": "P66 reviewed diagnostic-minimum resolution",
        "sample_adequacy_rule": expected["sample_adequacy_rule"],
        "diagnostic_threshold_definitions": expected["diagnostic_threshold_definitions"],
        "source_anchors": result.manifest.get("source_anchors"),
    }
    mismatches = tuple(
        key for key, expected_value in expected.items() if observed.get(key) != expected_value
    )
    return {
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
        "passed": not mismatches,
        "blockers": tuple(f"{key}_mismatch" for key in mismatches),
    }


def _p66_comparison_invariants(
    *,
    candidate: P59AuthorSIRStepSpecAssemblyResult,
    stronger_result: P59AuthorSIRStepSpecAssemblyResult | None,
    authorized_field: str,
    authorized_reason: str,
) -> Mapping[str, object]:
    candidate_fields = _p66_comparison_fields(candidate)
    stronger_fields = (
        {
            **candidate_fields,
            authorized_field: (
                int(candidate.manifest["fit_rank"]) + 1
                if authorized_field == "fit_rank"
                else int(candidate.manifest["fit_degree"]) + 1
            ),
        }
        if stronger_result is None
        else _p66_comparison_fields(stronger_result)
    )
    differences = tuple(
        key
        for key, value in candidate_fields.items()
        if stronger_fields.get(key) != value
    )
    unauthorized = tuple(key for key in differences if key != authorized_field)
    return {
        "candidate": candidate_fields,
        "stronger": stronger_fields,
        "differences": differences,
        "unauthorized_differences": unauthorized,
        "status": (
            P66_SOURCE_ROUTE_INVARIANT_DRIFT_BLOCK_STATUS
            if unauthorized
            else "PASS_COMPARISON_INVARIANTS"
        ),
        "authorized_comparison_difference": authorized_field in differences,
        "authorized_comparison_difference_field": authorized_field,
        "authorized_comparison_difference_reason": authorized_reason,
    }


def _p66_comparison_fields(
    result: P59AuthorSIRStepSpecAssemblyResult,
) -> Mapping[str, object]:
    return {
        "route_family": "Zhao-Cui full_sol",
        "target_dimension": result.manifest.get("target_dimension"),
        "source_target_order": result.manifest.get("source_target_order"),
        "previous_marginal_keep_axes": result.manifest.get("previous_marginal_keep_axes"),
        "previous_marginal_input_axes": result.manifest.get("previous_marginal_input_axes"),
        "fit_data_mode": result.manifest.get("fit_data_mode"),
        "defensive_tau": result.manifest.get("defensive_tau"),
        "initialization_rule": result.manifest.get("fit_initialization_rule"),
        "fixed_hmc_adaptation_class": result.manifest.get(
            "fixed_branch_adaptation_class"
        ),
        "fit_degree": result.manifest.get("fit_degree"),
        "fit_rank": result.manifest.get("fit_rank"),
    }


def _p66_schema_only_ladder_payload(
    *,
    ladder_kind: str,
    candidate: P59AuthorSIRStepSpecAssemblyResult,
    candidate_fit_sample_count: int,
    stronger_degree: int,
    stronger_rank: int,
    stronger_fit_sample_count: int,
    adequacy: Mapping[str, object],
    status: str,
) -> Mapping[str, object]:
    return {
        "candidate_branch": {
            "degree": candidate.manifest.get("fit_degree"),
            "rank": candidate.manifest.get("fit_rank"),
            "rank_tuple": candidate.manifest.get("rank_tuple"),
        },
        "stronger_branch": {
            "degree": int(stronger_degree),
            "rank": int(stronger_rank),
            "rank_tuple": _source_route_rank_tuple(
                P59_9A_AUTHOR_SIR_TARGET_DIMENSION,
                int(stronger_rank),
            ),
            "fit_sample_count": int(stronger_fit_sample_count),
        },
        "ladder_kind": str(ladder_kind),
        "executed": False,
        "schema_only_reason": (
            "Phase 2 implements schema and invariant gates; adjacent ladder "
            "execution requires a reviewed experiment plan."
        ),
        "sample_adequacy": {
            "candidate": p66_fixed_branch_sample_adequacy(
                fit_degree=int(candidate.manifest["fit_degree"]),
                fit_rank=int(candidate.manifest["fit_rank"]),
                fit_sample_count=int(candidate_fit_sample_count),
            ),
            "stronger": adequacy,
        },
        "log_marginal_delta": None,
        "normalizer_increment_deltas": None,
        "probe_log_density_median_delta": None,
        "retained_log_density_median_delta": None,
        "effective_sample_size": None,
        "correction_ranges": None,
        "status": status,
        "blockers": (),
    }


def _p66_manifest(
    *,
    status: str,
    blockers: tuple[str, ...],
    sample_count: int,
    sentinel_fit_sample_count: int,
    candidate: Mapping[str, object],
    sentinel: P60AuthorSIRSameRouteRankComparatorResult,
    sample_adequacy: Mapping[str, object],
    fit_budget_resolution: Mapping[str, object],
    rank_ladder: Mapping[str, object],
    degree_ladder: Mapping[str, object],
    source_invariants: Mapping[str, object],
    comparison_invariants: Mapping[str, object],
) -> Mapping[str, object]:
    sentinel_status = (
        P66_SENTINEL_PASS_STATUS
        if sentinel.status == P60_D18_RANK_CONVERGENCE_PASS_STATUS
        else P66_SENTINEL_WARN_STATUS
    )
    return {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P66",
        "artifact_role": "fixed_branch_validation_ladder",
        "status": status,
        "blockers": tuple(blockers),
        "sample_count": int(sample_count),
        "sentinel_fit_sample_count": int(sentinel_fit_sample_count),
        "candidate": candidate,
        "sentinel": {
            "old_p60_status": sentinel.status,
            "old_p60_blockers": sentinel.blockers,
            "old_p60_log_marginal_abs_delta": sentinel.manifest.get(
                "log_marginal_abs_delta"
            ),
            "old_p60_normalizer_increment_abs_deltas": sentinel.manifest.get(
                "normalizer_increment_abs_deltas"
            ),
            "old_p60_thresholds": sentinel.manifest.get("thresholds"),
            "status": sentinel_status,
            "interpretation": "explanatory_sentinel_not_primary_gate",
        },
        "sample_adequacy": sample_adequacy,
        "fit_budget_resolution": fit_budget_resolution,
        "rank_ladder": rank_ladder,
        "degree_ladder": degree_ladder,
        "source_invariants": source_invariants,
        "comparison_invariants": comparison_invariants,
        "nonclaims": _p66_rank_nonclaims(),
        "p65_baseline_reference": "P65 high branch noncollapsed fixed-HMC branch",
        "old_p60_sentinel_payload": sentinel.manifest,
    }


def _p66_result_from_manifest(
    *,
    status: str,
    blockers: tuple[str, ...],
    candidate: object | None,
    sentinel: P60AuthorSIRSameRouteRankComparatorResult | None,
    manifest: Mapping[str, object],
) -> P66AuthorSIRFixedBranchValidationLadderResult:
    return P66AuthorSIRFixedBranchValidationLadderResult(
        status=status,
        blockers=tuple(blockers),
        candidate_result=(
            candidate if isinstance(candidate, P59AuthorSIRStepSpecAssemblyResult) else None
        ),
        sentinel_result=sentinel,
        manifest=manifest,
    )


def _p66_block_result(
    *,
    status: str,
    blockers: tuple[str, ...],
    candidate_result: object | None,
    sentinel_result: P60AuthorSIRSameRouteRankComparatorResult | None,
    manifest_extra: Mapping[str, object],
) -> P66AuthorSIRFixedBranchValidationLadderResult:
    manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P66",
        "artifact_role": "blocked_fixed_branch_validation_ladder",
        "status": str(status),
        "blockers": tuple(blockers),
        "nonclaims": _p66_rank_nonclaims(),
    }
    manifest = {**manifest, **dict(manifest_extra)}
    return P66AuthorSIRFixedBranchValidationLadderResult(
        status=status,
        blockers=tuple(blockers),
        candidate_result=(
            candidate_result
            if isinstance(candidate_result, P59AuthorSIRStepSpecAssemblyResult)
            else None
        ),
        sentinel_result=sentinel_result,
        manifest=manifest,
    )


def _p60_rank_comparator_block(
    *,
    blockers: tuple[str, ...],
    low_result: P59AuthorSIRStepSpecAssemblyResult | None,
    high_result: P59AuthorSIRStepSpecAssemblyResult | None,
    extra_manifest: Mapping[str, object] | None = None,
) -> P60AuthorSIRSameRouteRankComparatorResult:
    manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P60-2",
        "artifact_role": "blocked_same_route_rank_comparator",
        "status": P60_D18_RANK_CONVERGENCE_BLOCK_STATUS,
        "veto_blockers": tuple(blockers),
        "nonclaims": _p60_rank_nonclaims(),
    }
    if extra_manifest is not None:
        manifest = {**manifest, **dict(extra_manifest)}
    return P60AuthorSIRSameRouteRankComparatorResult(
        status=P60_D18_RANK_CONVERGENCE_BLOCK_STATUS,
        blockers=tuple(blockers),
        low_result=low_result,
        high_result=high_result,
        manifest=manifest,
    )


def _p59_assembly_fit_sample_counts(
    assembly: P59AuthorSIRStepSpecAssemblyResult,
) -> tuple[int, ...]:
    fit_data_manifests = assembly.manifest.get("fit_data_manifests", ())
    counts: list[int] = []
    if isinstance(fit_data_manifests, tuple):
        for row in fit_data_manifests:
            if isinstance(row, Mapping) and "fit_sample_count" in row:
                counts.append(int(row["fit_sample_count"]))
    return tuple(counts)


def _p59_common_fit_sample_count(
    assembly: P59AuthorSIRStepSpecAssemblyResult,
) -> int | None:
    counts = _p59_assembly_fit_sample_counts(assembly)
    if not counts or len(set(counts)) != 1:
        return None
    return counts[0]


def _p59_assembly_row_adequacy_by_step(
    assembly: P59AuthorSIRStepSpecAssemblyResult,
) -> tuple[Mapping[str, object], ...]:
    fit_quality_rows = assembly.manifest.get("fit_quality_diagnostics_by_step", ())
    row_adequacy_rows: list[Mapping[str, object]] = []
    if isinstance(fit_quality_rows, tuple):
        for row in fit_quality_rows:
            policy = (
                row.get("p70_fixed_fitting_policy", {})
                if isinstance(row, Mapping)
                else {}
            )
            row_adequacy = (
                policy.get("row_adequacy", {})
                if isinstance(policy, Mapping)
                else {}
            )
            row_adequacy_rows.append(
                dict(row_adequacy) if isinstance(row_adequacy, Mapping) else {}
            )
    return tuple(row_adequacy_rows)


def _p59_execution_fit_sample_count_policy() -> Mapping[str, object]:
    return {
        "default_fit_sample_count": P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT,
        "default_fit_sample_count_source": P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT_SOURCE,
        "default_row_adequacy": _p59_execution_only_row_adequacy_diagnostics(
            fit_sample_count=P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT,
        ),
        "explicit_under_rowed_calls_fail_closed": True,
        "nonclaim": "row adequacy is not sample coverage or convergence proof",
    }


def p59_author_sir_runner_manifest_path(
    *,
    assembly_result: P59AuthorSIRStepSpecAssemblyResult | None = None,
    manifest_path: str | Path = P59_9D_DEFAULT_MANIFEST_PATH,
    comparator_tier: str = "d18_execution_only",
    sample_count: int = 1,
    fit_sample_count: int = P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT,
    write_manifest: bool = False,
) -> P59AuthorSIRRunnerManifestResult:
    """Build the P59-9d M9 runner/readiness manifest path.

    This is the bounded command surface that consumes the already gated
    P59-9a/9b/9c artifacts.  It does not run the P59-9e validation ladder.
    """

    if assembly_result is not None and not isinstance(
        assembly_result,
        P59AuthorSIRStepSpecAssemblyResult,
    ):
        raise TypeError("assembly_result must be P59AuthorSIRStepSpecAssemblyResult")
    assembly = assembly_result or p59_author_sir_step_spec_assembly(
        sample_count=int(sample_count),
        fit_sample_count=int(fit_sample_count),
    )
    blockers: list[str] = []
    if assembly.status != P59_9B_PASS_STATUS:
        blockers.append("missing_p59_9b_step_spec_assembly_pass")
    if assembly.route_decision.status != P59_9C_PASS_STATUS:
        blockers.append("missing_p59_9c_route_decision_pass")
    if assembly.route_decision.route_decision != P59_9C_FULL_ROUTE_SELECTED:
        blockers.append("p59_9c_route_not_full_route_selected")
    if assembly.prep_result is None or assembly.prep_result.status != P59_9A_PASS_STATUS:
        blockers.append("missing_p59_9a_target_fit_prep_pass")
    if str(comparator_tier) not in P58_M9_ALLOWED_COMPARATOR_TIERS:
        blockers.append("invalid_m9_comparator_tier")
    manifest_target = Path(manifest_path)
    if not str(manifest_target).strip():
        blockers.append("missing_manifest_path")

    p58_manifest = _p59_9d_p58_manifest(
        assembly=assembly,
        comparator_tier=str(comparator_tier),
        manifest_path=str(manifest_target),
        has_runner_manifest_path=not blockers,
    )
    readiness = p58_m9_source_route_pipeline_readiness(p58_manifest)
    if readiness.status != P58_M9_READY_STATUS:
        blockers.extend(
            blocker for blocker in readiness.blockers if blocker not in blockers
        )
    if blockers:
        return P59AuthorSIRRunnerManifestResult(
            status=P59_9D_BLOCK_STATUS,
            blockers=tuple(blockers),
            assembly_result=assembly,
            readiness=readiness,
            manifest_path=str(manifest_target),
            manifest={
                "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
                "pipeline_phase": "P59-9d",
                "artifact_role": "blocked_runner_manifest_path",
                "blockers": tuple(blockers),
                "p58_manifest": p58_manifest,
                "fail_closed_rule": "P59-9e validation ladder must not proceed.",
                "nonclaims": (
                    "no Phase-9 validation launch",
                    "no d18 filtering accuracy claim",
                ),
            },
        )

    fit_sample_counts = _p59_assembly_fit_sample_counts(assembly)
    row_adequacy_by_step = _p59_assembly_row_adequacy_by_step(assembly)
    manifest = {
        "schema_version": "bayesfilter.p59_9d.runner_manifest.v1",
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P59-9d",
        "artifact_role": "runner_manifest_path",
        "status": P59_9D_PASS_STATUS,
        "runner_command": (
            "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python "
            "scripts/p59_author_sir_m9_runner_manifest.py --output "
            + str(manifest_target)
        ),
        "consumed_artifacts": {
            "p59_9a_status": assembly.prep_result.status,
            "p59_9b_status": assembly.status,
            "p59_9c_status": assembly.route_decision.status,
            "p59_9c_route_decision": assembly.route_decision.route_decision,
        },
        "source_anchors": _P59_9D_AUTHOR_SIR_SOURCE_ANCHORS,
        "p58_manifest": p58_manifest,
        "p58_readiness": readiness.manifest_payload(),
        "step_spec_count": len(assembly.step_specs),
        "fit_sample_count": _p59_common_fit_sample_count(assembly),
        "fit_sample_count_by_step": fit_sample_counts,
        "fit_sample_count_policy": _p59_execution_fit_sample_count_policy(),
        "row_adequacy_by_step": row_adequacy_by_step,
        "sequential_status": (
            None
            if assembly.sequential_result is None
            else assembly.sequential_result.sequential_status
        ),
        "manifest_path": str(manifest_target),
        "ready_for_validation_ladder": True,
        "nonclaims": (
            "no P59-9e validation has been run",
            "no d18 filtering accuracy claim",
            "no same-route rank convergence claim",
            "no d50 or d100 scaling claim",
            "no HMC production readiness claim",
            "no adaptive Zhao-Cui parity claim",
        ),
    }
    if bool(write_manifest):
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        manifest_target.write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return P59AuthorSIRRunnerManifestResult(
        status=P59_9D_PASS_STATUS,
        blockers=(),
        assembly_result=assembly,
        readiness=readiness,
        manifest_path=str(manifest_target),
        manifest=manifest,
    )


def _p59_9d_p58_manifest(
    *,
    assembly: P59AuthorSIRStepSpecAssemblyResult,
    comparator_tier: str,
    manifest_path: str,
    has_runner_manifest_path: bool,
) -> Mapping[str, object]:
    return {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_kind": P58_M9_SOURCE_ROUTE_PIPELINE_KIND,
        "route_class": "fixed_ttsirt_source_route",
        "storage_kind": "source_transport_object",
        "transition_interface": "sample_propagation",
        "m9_comparator_tier": str(comparator_tier),
        "rank_policy_status": "PASS_P57_M7_SOURCE_FAITHFUL_RANK_UKF_CALIBRATION",
        "preconditioned_route_required": False,
        "preconditioned_route_status": assembly.route_decision.preconditioned_route_status,
        "uses_contract_test_double": False,
        "uses_ukf_as_comparator": False,
        "uses_rank_memory_proxy_as_comparator": False,
        "has_author_sir_callback": True,
        "has_fixed_ttsirt_fit_artifacts": assembly.status == P59_9B_PASS_STATUS,
        "has_fixed_ttsirt_transports": assembly.status == P59_9B_PASS_STATUS,
        "has_frozen_reference_samples": assembly.status == P59_9B_PASS_STATUS,
        "has_source_route_step_specs": assembly.status == P59_9B_PASS_STATUS,
        "has_sequential_retained_carry": (
            assembly.status == P59_9B_PASS_STATUS and assembly.sequential_result is not None
        ),
        "has_previous_marginal_evidence": bool(
            assembly.sequential_result is not None
            and len(assembly.sequential_result.steps) > 1
            and assembly.sequential_result.steps[1].previous_marginal_density is not None
        ),
        "has_m9_runner_manifest_path": bool(has_runner_manifest_path),
        "runner_manifest_path": str(manifest_path),
        "source_anchors": _P59_9D_AUTHOR_SIR_SOURCE_ANCHORS,
    }


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            raw = value.numpy()
            if hasattr(raw, "item"):
                return raw.item()
            return raw
        return value.numpy().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def p59_author_sir_validation_ladder(
    *,
    runner_result: P59AuthorSIRRunnerManifestResult | None = None,
    tier: str = "d18_execution_only",
    sample_count: int = 1,
    fit_sample_count: int = P59_D18_EXECUTION_ONLY_FIT_SAMPLE_COUNT,
) -> P59AuthorSIRValidationLadderResult:
    """Run the P59-9e validation ladder at the declared honest tier."""

    selected_tier = str(tier)
    if selected_tier not in P58_M9_ALLOWED_COMPARATOR_TIERS:
        return _p59_9e_block_result(
            blockers=("invalid_p59_9e_tier",),
            tier=selected_tier,
            runner_result=runner_result,
        )
    runner = runner_result or p59_author_sir_runner_manifest_path(
        sample_count=int(sample_count),
        fit_sample_count=int(fit_sample_count),
        comparator_tier=selected_tier,
    )
    blockers: list[str] = []
    if runner.status != P59_9D_PASS_STATUS:
        blockers.append("missing_p59_9d_runner_manifest_pass")
    if selected_tier == "d18_same_route_rank_convergence":
        blockers.append("missing_higher_rank_same_route_comparator")
    if selected_tier == "d18_correctness_candidate":
        blockers.append("missing_same_target_reference_or_bridge")
    if selected_tier != "d18_execution_only":
        return _p59_9e_block_result(
            blockers=tuple(blockers),
            tier=selected_tier,
            runner_result=runner,
        )
    if blockers:
        return _p59_9e_block_result(
            blockers=tuple(blockers),
            tier=selected_tier,
            runner_result=runner,
        )
    assembly = runner.assembly_result
    if assembly is None or assembly.sequential_result is None:
        return _p59_9e_block_result(
            blockers=("missing_p59_9b_sequential_result",),
            tier=selected_tier,
            runner_result=runner,
        )
    sequential = assembly.sequential_result
    log_marginal = sequential.log_marginal_likelihood
    if not bool(tf.math.is_finite(log_marginal).numpy()):
        return _p59_9e_block_result(
            blockers=("nonfinite_log_marginal_likelihood",),
            tier=selected_tier,
            runner_result=runner,
        )
    ess_values = tuple(
        float(step.retained_samples.diagnostics.effective_sample_size.numpy())
        for step in sequential.steps
    )
    normalizer_increments = tuple(
        float(step.normalizer_increment.numpy()) for step in sequential.steps
    )
    correction_ranges = tuple(
        (
            float(tf.reduce_min(step.retained_samples.correction_log_weights).numpy()),
            float(tf.reduce_max(step.retained_samples.correction_log_weights).numpy()),
        )
        for step in sequential.steps
    )
    fit_sample_counts = _p59_assembly_fit_sample_counts(assembly)
    row_adequacy_by_step = _p59_assembly_row_adequacy_by_step(assembly)
    manifest = {
        "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
        "pipeline_phase": "P59-9e",
        "artifact_role": "d18_execution_only_validation_ladder",
        "status": P59_9E_D18_EXECUTION_ONLY_PASS_STATUS,
        "tier": selected_tier,
        "prerequisite_tokens": {
            "p59_9a": assembly.prep_result.status if assembly.prep_result else None,
            "p59_9b": assembly.status,
            "p59_9c": assembly.route_decision.status,
            "p59_9d": runner.status,
            "p58_readiness": runner.readiness.status,
        },
        "step_count": len(sequential.steps),
        "target_dimension": assembly.manifest["target_dimension"],
        "state_dimension": assembly.manifest["state_dimension"],
        "parameter_dimension": assembly.manifest["parameter_dimension"],
        "comparator_tier": selected_tier,
        "fit_sample_count": _p59_common_fit_sample_count(assembly),
        "fit_sample_count_by_step": fit_sample_counts,
        "fit_sample_count_policy": _p59_execution_fit_sample_count_policy(),
        "row_adequacy_by_step": row_adequacy_by_step,
        "log_marginal_likelihood": float(log_marginal.numpy()),
        "normalizer_increments": normalizer_increments,
        "effective_sample_size_by_step": ess_values,
        "correction_log_weight_ranges": correction_ranges,
        "transport_source_contract_levels": assembly.manifest[
            "transport_source_contract_levels"
        ],
        "fit_branch_hashes": assembly.manifest["fit_branch_hashes"],
        "density_branch_hashes": assembly.manifest["density_branch_hashes"],
        "previous_marginal_keep_axes": assembly.manifest["previous_marginal_keep_axes"],
        "previous_marginal_input_axes": assembly.manifest["previous_marginal_input_axes"],
        "holdout_replay_diagnostics_by_step": assembly.manifest[
            "holdout_replay_diagnostics_by_step"
        ],
        "source_anchors": _P59_9D_AUTHOR_SIR_SOURCE_ANCHORS,
        "runner_manifest_path": runner.manifest_path,
        "promotion_boundary": (
            "execution-only tier permits finite values, ESS, replay, normalizer, "
            "rank, memory, and wall-time diagnostics; it cannot claim accuracy."
        ),
        "blocked_higher_tiers": {
            "d18_same_route_rank_convergence": "missing_higher_rank_same_route_comparator",
            "d18_correctness_candidate": "missing_same_target_reference_or_bridge",
            "d50": "requires d18_same_route_rank_convergence first",
            "d100": "requires d50 non-veto scaling evidence first",
        },
        "nonclaims": (
            "no d18 filtering accuracy claim",
            "no same-route rank convergence claim",
            "no d18 correctness candidate claim",
            "no d50 or d100 scaling claim",
            "no HMC production readiness claim",
            "no adaptive Zhao-Cui parity claim",
            "no UKF correctness comparator",
        ),
    }
    return P59AuthorSIRValidationLadderResult(
        status=P59_9E_D18_EXECUTION_ONLY_PASS_STATUS,
        blockers=(),
        tier=selected_tier,
        runner_result=runner,
        manifest=manifest,
    )


def _p59_9e_block_result(
    *,
    blockers: tuple[str, ...],
    tier: str,
    runner_result: P59AuthorSIRRunnerManifestResult | None,
) -> P59AuthorSIRValidationLadderResult:
    return P59AuthorSIRValidationLadderResult(
        status=P59_9E_BLOCK_STATUS,
        blockers=tuple(blockers) if blockers else ("p59_9e_validation_ladder_blocked",),
        tier=str(tier) if str(tier) in P58_M9_ALLOWED_COMPARATOR_TIERS else "d18_execution_only",
        runner_result=runner_result,
        manifest={
            "target_id": P58_M9_AUTHOR_SIR_TARGET_ID,
            "pipeline_phase": "P59-9e",
            "artifact_role": "blocked_validation_ladder",
            "requested_tier": str(tier),
            "blockers": tuple(blockers) if blockers else ("p59_9e_validation_ladder_blocked",),
            "fail_closed_rule": "Do not attempt d50/d100 or promote correctness.",
            "nonclaims": (
                "no d18 filtering accuracy claim",
                "no same-route rank convergence claim",
                "no d50 or d100 scaling claim",
            ),
        },
    )


@dataclass(frozen=True)
class SourceRoutePreviousMarginalDensityResult:
    """Previous retained-object marginal density in physical coordinates."""

    previous_retained_object: SourceRouteRetainedObject
    keep_axes: tuple[int, ...]
    marginal_transport: object
    physical_points: tf.Tensor
    local_points: tf.Tensor
    log_density: tf.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.previous_retained_object, SourceRouteRetainedObject):
            raise TypeError("previous_retained_object must be SourceRouteRetainedObject")
        keep_axes = tuple(int(axis) for axis in self.keep_axes)
        if not keep_axes:
            raise ValueError("keep_axes must be nonempty")
        if keep_axes != tuple(range(len(keep_axes))):
            raise ValueError("source previous marginalization requires prefix keep axes")
        if keep_axes[-1] >= self.previous_retained_object.coordinate_frame.dimension:
            raise ValueError(f"keep_axes: {HighDimStatus.INVALID_SHAPE.value}")
        physical = tf.convert_to_tensor(self.physical_points, dtype=tf.float64)
        local = tf.convert_to_tensor(self.local_points, dtype=tf.float64)
        log_density = _finite_vector("log_density", self.log_density)
        if physical.shape.rank != 2 or local.shape.rank != 2:
            raise ValueError(f"previous marginal points: {HighDimStatus.INVALID_SHAPE.value}")
        if (
            int(physical.shape[0]) != len(keep_axes)
            or local.shape != physical.shape
            or log_density.shape != (int(physical.shape[1]),)
        ):
            raise ValueError(f"previous marginal density: {HighDimStatus.INVALID_SHAPE.value}")
        _transport_manifest_payload(self.marginal_transport)
        object.__setattr__(self, "keep_axes", keep_axes)
        object.__setattr__(self, "physical_points", physical)
        object.__setattr__(self, "local_points", local)
        object.__setattr__(self, "log_density", log_density)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRoutePreviousMarginalDensityResult",
            "previous_retained_hash": (
                self.previous_retained_object.branch_identity.hash.value
            ),
            "keep_axes": self.keep_axes,
            "marginal_transport": _transport_manifest_payload(self.marginal_transport),
            "physical_points": self.physical_points,
            "local_points": self.local_points,
            "log_density": self.log_density,
        }


@dataclass(frozen=True)
class SourceRoutePreconditionerContract:
    """Source-route preconditioner/residual target decomposition metadata."""

    variant: str
    coefficient: tf.Tensor
    reference_density_label: str
    forward_map_label: str
    inverse_map_label: str
    route_label: str = SOURCE_FAITHFUL_ROUTE_LABEL

    def __post_init__(self) -> None:
        if str(self.variant) not in ("full", "preconditioner", "residual"):
            raise ValueError("variant must be full, preconditioner, or residual")
        coefficient = tf.convert_to_tensor(self.coefficient, dtype=tf.float64)
        if coefficient.shape.rank != 0:
            raise ValueError(f"coefficient: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.math.is_finite(coefficient).numpy()):
            raise ValueError(f"coefficient: {HighDimStatus.NONFINITE_VALUE.value}")
        for name, value in (
            ("reference_density_label", self.reference_density_label),
            ("forward_map_label", self.forward_map_label),
            ("inverse_map_label", self.inverse_map_label),
            ("route_label", self.route_label),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be nonempty")
        object.__setattr__(self, "variant", str(self.variant))
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "reference_density_label", str(self.reference_density_label))
        object.__setattr__(self, "forward_map_label", str(self.forward_map_label))
        object.__setattr__(self, "inverse_map_label", str(self.inverse_map_label))
        object.__setattr__(self, "route_label", str(self.route_label))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRoutePreconditionerContract",
            "variant": self.variant,
            "coefficient": self.coefficient,
            "reference_density_label": self.reference_density_label,
            "forward_map_label": self.forward_map_label,
            "inverse_map_label": self.inverse_map_label,
            "route_label": self.route_label,
        }


@dataclass(frozen=True)
class SourceRouteLinearPreconditionerResult:
    """Linear preconditioner output mirroring author ``precond.m``."""

    matrix: tf.Tensor
    sigma_k: tf.Tensor
    parameter_dim: int
    state_dim: int
    source_anchor: str = (
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/tensordot/"
        "precond.m:43-56"
    )

    def __post_init__(self) -> None:
        matrix = tf.convert_to_tensor(self.matrix, dtype=tf.float64)
        sigma_k = tf.convert_to_tensor(self.sigma_k, dtype=tf.float64)
        if matrix.shape.rank != 2 or sigma_k.shape != matrix.shape:
            raise ValueError(f"linear preconditioner: {HighDimStatus.INVALID_SHAPE.value}")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"linear preconditioner: {HighDimStatus.INVALID_SHAPE.value}")
        d = int(self.parameter_dim)
        m = int(self.state_dim)
        if d < 0 or m <= 0 or d + m != int(matrix.shape[0]):
            raise ValueError("parameter_dim + state_dim must match matrix shape")
        if not bool(
            tf.reduce_all(tf.math.is_finite(matrix)).numpy()
            and tf.reduce_all(tf.math.is_finite(sigma_k)).numpy()
        ):
            raise ValueError(f"linear preconditioner: {HighDimStatus.NONFINITE_VALUE.value}")
        if not str(self.source_anchor).strip():
            raise ValueError("source_anchor must be nonempty")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "sigma_k", sigma_k)
        object.__setattr__(self, "parameter_dim", d)
        object.__setattr__(self, "state_dim", m)
        object.__setattr__(self, "source_anchor", str(self.source_anchor))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteLinearPreconditionerResult",
            "matrix": self.matrix,
            "sigma_k": self.sigma_k,
            "parameter_dim": int(self.parameter_dim),
            "state_dim": int(self.state_dim),
            "source_anchor": self.source_anchor,
        }


def source_route_linear_preconditioner_from_covariances(
    *,
    sigma1: tf.Tensor,
    sigma2: tf.Tensor,
) -> SourceRouteLinearPreconditionerResult:
    """Construct the author linear preconditioner from covariance matrices.

    This is a TensorFlow transcription of ``precond.m:43-56``.  ``sigma1`` has
    dimension ``d + m`` and ``sigma2`` has dimension ``m``; the result contains
    the source matrix ``C`` and diagonal ``Sigmak`` used by the preconditioned
    route.
    """

    s1 = tf.convert_to_tensor(sigma1, dtype=tf.float64)
    s2 = tf.convert_to_tensor(sigma2, dtype=tf.float64)
    if s1.shape.rank != 2 or s2.shape.rank != 2:
        raise ValueError(f"preconditioner covariances: {HighDimStatus.INVALID_SHAPE.value}")
    if s1.shape[0] != s1.shape[1] or s2.shape[0] != s2.shape[1]:
        raise ValueError(f"preconditioner covariances: {HighDimStatus.INVALID_SHAPE.value}")
    if s1.shape[0] is None or s2.shape[0] is None:
        raise ValueError("preconditioner covariance dimensions must be statically known")
    n = int(s1.shape[0])
    m = int(s2.shape[0])
    d = n - m
    if d < 0 or n <= 0 or m <= 0:
        raise ValueError(f"preconditioner covariances: {HighDimStatus.INVALID_SHAPE.value}")
    assert_tf_float64("sigma1", s1)
    assert_tf_float64("sigma2", s2)
    if not bool(
        tf.reduce_all(tf.math.is_finite(s1)).numpy()
        and tf.reduce_all(tf.math.is_finite(s2)).numpy()
    ):
        raise ValueError(f"preconditioner covariances: {HighDimStatus.NONFINITE_VALUE.value}")

    sigma2_full = tf.pad(s2, paddings=[[d, 0], [d, 0]])
    sigma3 = tf.linalg.solve(s1, sigma2_full)
    _, eigvecs = tf.linalg.eig(tf.cast(sigma3, tf.complex128))
    c_complex = tf.linalg.adjoint(eigvecs)
    if d > 0:
        chol_param = tf.linalg.cholesky(s1[:d, :d])
        c11 = tf.cast(tf.linalg.inv(chol_param), tf.complex128)
        c_complex = tf.concat(
            [
                tf.concat([c11, c_complex[:d, d:]], axis=1),
                tf.concat([c_complex[d:, :d], c_complex[d:, d:]], axis=1),
            ],
            axis=0,
        )
    s1_complex = tf.cast(s1, tf.complex128)
    row_variance = tf.math.real(
        tf.linalg.diag_part(c_complex @ s1_complex @ tf.linalg.adjoint(c_complex))
    )
    if not bool(tf.reduce_all(row_variance > 0.0).numpy()):
        raise ValueError(f"preconditioner row variance: {HighDimStatus.NONFINITE_VALUE.value}")
    c_matrix = tf.math.real(
        c_complex / tf.cast(tf.sqrt(row_variance)[:, tf.newaxis], tf.complex128)
    )
    c22 = c_matrix[d:, d:]
    sigma_k_block = c22 @ s2 @ tf.transpose(c22)
    sigma_k_diag = tf.concat(
        [
            tf.zeros([d], dtype=tf.float64),
            tf.linalg.diag_part(sigma_k_block),
        ],
        axis=0,
    )
    sigma_k = tf.linalg.diag(sigma_k_diag)
    return SourceRouteLinearPreconditionerResult(
        matrix=c_matrix,
        sigma_k=sigma_k,
        parameter_dim=d,
        state_dim=m,
    )


@dataclass(frozen=True)
class SourceRoutePreconditionedMap:
    """Fixed-HMC version of the author preconditioner maps ``Tu2x``/``Tx2u``."""

    preconditioner_transport: SourceRouteTransportProtocol
    preconditioner_frame: SourceRouteCoordinateFrame
    reference_forward_fn: object
    reference_inverse_fn: object
    reference_log_density_fn: object
    source_anchor: str = (
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pre_sol.m:212-213"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.preconditioner_transport, SourceRouteTransportProtocol):
            raise TypeError("preconditioner_transport must be SourceRouteTransportProtocol")
        if not isinstance(self.preconditioner_frame, SourceRouteCoordinateFrame):
            raise TypeError("preconditioner_frame must be SourceRouteCoordinateFrame")
        for name in (
            "reference_forward_fn",
            "reference_inverse_fn",
            "reference_log_density_fn",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if not str(self.source_anchor).strip():
            raise ValueError("source_anchor must be nonempty")
        object.__setattr__(self, "source_anchor", str(self.source_anchor))

    @property
    def dimension(self) -> int:
        return self.preconditioner_frame.dimension

    def Tu2x(self, residual_points: tf.Tensor) -> tf.Tensor:
        residual = self._finite_points("residual_points", residual_points)
        reference = self._reference_forward(residual)
        local = self.preconditioner_transport.inverse_transport(reference)
        return self._frame_forward(local)

    def Tx2u(self, physical_points: tf.Tensor) -> tf.Tensor:
        physical = self._finite_points("physical_points", physical_points)
        local = self._frame_inverse(physical)
        reference = self.preconditioner_transport.forward_transport(local)
        return self._reference_inverse(reference)

    def reference_log_density(self, residual_points: tf.Tensor) -> tf.Tensor:
        residual = self._finite_points("residual_points", residual_points)
        values = _finite_vector("reference_log_density", self.reference_log_density_fn(residual))
        if values.shape != (int(residual.shape[1]),):
            raise ValueError(f"reference_log_density: {HighDimStatus.INVALID_SHAPE.value}")
        return values

    def preconditioner_log_density(self, physical_points: tf.Tensor) -> tf.Tensor:
        physical = self._finite_points("physical_points", physical_points)
        local = self._frame_inverse(physical)
        density = self.preconditioner_transport.eval_pdf(local)
        return tf.math.log(density) - self.preconditioner_frame.log_abs_det()

    def map_roundtrip_error(self, residual_points: tf.Tensor) -> tf.Tensor:
        residual = self._finite_points("residual_points", residual_points)
        return tf.reduce_max(tf.abs(self.Tx2u(self.Tu2x(residual)) - residual))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRoutePreconditionedMap",
            "dimension": self.dimension,
            "forward_map_label": "Tu2x",
            "inverse_map_label": "Tx2u",
            "source_anchor": self.source_anchor,
            "preconditioner_transport": self.preconditioner_transport.manifest_payload(),
            "preconditioner_frame": self.preconditioner_frame.manifest_payload(),
        }

    def _finite_points(self, name: str, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank != 2 or int(values.shape[0]) != self.dimension:
            raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64(name, values)
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
        return values

    def _reference_forward(self, residual_points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(
            self.reference_forward_fn(residual_points),
            dtype=tf.float64,
        )
        if values.shape != residual_points.shape:
            raise ValueError(f"reference_forward_fn: {HighDimStatus.INVALID_SHAPE.value}")
        return values

    def _reference_inverse(self, reference_points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(
            self.reference_inverse_fn(reference_points),
            dtype=tf.float64,
        )
        if values.shape != reference_points.shape:
            raise ValueError(f"reference_inverse_fn: {HighDimStatus.INVALID_SHAPE.value}")
        return values

    def _frame_inverse(self, physical_points: tf.Tensor) -> tf.Tensor:
        return tf.linalg.solve(
            self.preconditioner_frame.matrix,
            physical_points - self.preconditioner_frame.mu[:, tf.newaxis],
        )

    def _frame_forward(self, local_points: tf.Tensor) -> tf.Tensor:
        return (
            tf.linalg.matmul(self.preconditioner_frame.matrix, local_points)
            + self.preconditioner_frame.mu[:, tf.newaxis]
        )


@dataclass(frozen=True)
class SourceRoutePreconditionedProposalResult:
    """Proposal correction for the fixed preconditioned Algorithm-5 surface."""

    residual_local_samples: tf.Tensor
    physical_samples: tf.Tensor
    residual_log_density: tf.Tensor
    preconditioner_log_density: tf.Tensor
    reference_log_density: tf.Tensor
    full_negative_log_density: tf.Tensor
    proposal_log_density: tf.Tensor
    target_log_density: tf.Tensor
    correction_log_weights: tf.Tensor
    source_anchor: str = (
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pre_sol.m:245-255"
    )

    def __post_init__(self) -> None:
        residual = tf.convert_to_tensor(self.residual_local_samples, dtype=tf.float64)
        physical = tf.convert_to_tensor(self.physical_samples, dtype=tf.float64)
        if residual.shape.rank != 2 or physical.shape != residual.shape:
            raise ValueError(f"preconditioned samples: {HighDimStatus.INVALID_SHAPE.value}")
        residual_log = _finite_vector("residual_log_density", self.residual_log_density)
        pre_log = _finite_vector("preconditioner_log_density", self.preconditioner_log_density)
        ref_log = _finite_vector("reference_log_density", self.reference_log_density)
        full_neg = _finite_vector("full_negative_log_density", self.full_negative_log_density)
        proposal = _finite_vector("proposal_log_density", self.proposal_log_density)
        target = _finite_vector("target_log_density", self.target_log_density)
        correction = _finite_vector("correction_log_weights", self.correction_log_weights)
        expected_shape = (int(residual.shape[1]),)
        for name, values in (
            ("residual_log_density", residual_log),
            ("preconditioner_log_density", pre_log),
            ("reference_log_density", ref_log),
            ("full_negative_log_density", full_neg),
            ("proposal_log_density", proposal),
            ("target_log_density", target),
            ("correction_log_weights", correction),
        ):
            if values.shape != expected_shape:
                raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
        tf.debugging.assert_near(proposal, residual_log + pre_log - ref_log, atol=1e-10)
        tf.debugging.assert_near(target, -full_neg, atol=1e-10)
        tf.debugging.assert_near(correction, target - proposal, atol=1e-10)
        if not str(self.source_anchor).strip():
            raise ValueError("source_anchor must be nonempty")
        object.__setattr__(self, "residual_local_samples", residual)
        object.__setattr__(self, "physical_samples", physical)
        object.__setattr__(self, "residual_log_density", residual_log)
        object.__setattr__(self, "preconditioner_log_density", pre_log)
        object.__setattr__(self, "reference_log_density", ref_log)
        object.__setattr__(self, "full_negative_log_density", full_neg)
        object.__setattr__(self, "proposal_log_density", proposal)
        object.__setattr__(self, "target_log_density", target)
        object.__setattr__(self, "correction_log_weights", correction)
        object.__setattr__(self, "source_anchor", str(self.source_anchor))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRoutePreconditionedProposalResult",
            "residual_local_samples": self.residual_local_samples,
            "physical_samples": self.physical_samples,
            "residual_log_density": self.residual_log_density,
            "preconditioner_log_density": self.preconditioner_log_density,
            "reference_log_density": self.reference_log_density,
            "full_negative_log_density": self.full_negative_log_density,
            "proposal_log_density": self.proposal_log_density,
            "target_log_density": self.target_log_density,
            "correction_log_weights": self.correction_log_weights,
            "source_anchor": self.source_anchor,
        }


def source_route_preconditioned_proposal_correction(
    *,
    preconditioned_map: SourceRoutePreconditionedMap,
    residual_transport: SourceRouteTransportProtocol,
    reference_samples: tf.Tensor,
    full_negative_log_density_fn,
) -> SourceRoutePreconditionedProposalResult:
    """Generate fixed preconditioned-route proposal corrections.

    This mirrors the author-code density identity in ``pre_sol.m:245-255``:
    residual SIRT inverse samples produce ``p``; ``Tu2x`` maps the residual
    sample into physical space; the proposal log density is residual log
    density plus preconditioner log density minus the residual-reference log
    density; the correction is target minus proposal.
    """

    if not isinstance(preconditioned_map, SourceRoutePreconditionedMap):
        raise TypeError("preconditioned_map must be SourceRoutePreconditionedMap")
    if not isinstance(residual_transport, SourceRouteTransportProtocol):
        raise TypeError("residual_transport must be SourceRouteTransportProtocol")
    if not callable(full_negative_log_density_fn):
        raise TypeError("full_negative_log_density_fn must be callable")
    reference = tf.convert_to_tensor(reference_samples, dtype=tf.float64)
    if (
        reference.shape.rank != 2
        or int(reference.shape[0]) != preconditioned_map.dimension
    ):
        raise ValueError(f"reference_samples: {HighDimStatus.INVALID_SHAPE.value}")
    assert_tf_float64("reference_samples", reference)
    if not bool(tf.reduce_all(tf.math.is_finite(reference)).numpy()):
        raise ValueError(f"reference_samples: {HighDimStatus.NONFINITE_VALUE.value}")

    residual_local = residual_transport.inverse_transport(reference)
    physical_samples = preconditioned_map.Tu2x(residual_local)
    residual_log_density = residual_transport.proposal_log_density(
        local_points=residual_local,
        reference_points=reference,
    )
    preconditioner_log_density = preconditioned_map.preconditioner_log_density(
        physical_samples
    )
    reference_log_density = preconditioned_map.reference_log_density(residual_local)
    full_negative_log_density = _finite_vector(
        "full_negative_log_density",
        full_negative_log_density_fn(physical_samples),
    )
    if full_negative_log_density.shape != (int(reference.shape[1]),):
        raise ValueError(f"full_negative_log_density: {HighDimStatus.INVALID_SHAPE.value}")
    proposal_log_density = (
        residual_log_density + preconditioner_log_density - reference_log_density
    )
    target_log_density = -full_negative_log_density
    correction = source_route_proposal_log_weights(
        log_target_density=target_log_density,
        log_proposal_density=proposal_log_density,
    )
    return SourceRoutePreconditionedProposalResult(
        residual_local_samples=residual_local,
        physical_samples=physical_samples,
        residual_log_density=residual_log_density,
        preconditioner_log_density=preconditioner_log_density,
        reference_log_density=reference_log_density,
        full_negative_log_density=full_negative_log_density,
        proposal_log_density=proposal_log_density,
        target_log_density=target_log_density,
        correction_log_weights=correction,
    )


@dataclass(frozen=True)
class SourceRoutePredatorPreyLadderRow:
    """Route-separated M5 predator-prey ladder row."""

    row_id: str
    route_label: str
    horizon: int
    baseline: str
    primary_check: str
    decision_status: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("row_id", self.row_id),
            ("route_label", self.route_label),
            ("baseline", self.baseline),
            ("primary_check", self.primary_check),
            ("decision_status", self.decision_status),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be nonempty")
        if int(self.horizon) <= 0:
            raise ValueError("horizon must be positive")
        claims = tuple(str(item) for item in self.non_claims)
        if not claims:
            raise ValueError("non_claims must be nonempty")
        object.__setattr__(self, "row_id", str(self.row_id))
        object.__setattr__(self, "route_label", str(self.route_label))
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "baseline", str(self.baseline))
        object.__setattr__(self, "primary_check", str(self.primary_check))
        object.__setattr__(self, "decision_status", str(self.decision_status))
        object.__setattr__(self, "non_claims", claims)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "row_id": self.row_id,
            "route_label": self.route_label,
            "horizon": int(self.horizon),
            "baseline": self.baseline,
            "primary_check": self.primary_check,
            "decision_status": self.decision_status,
            "non_claims": self.non_claims,
        }


@dataclass(frozen=True)
class SourceRoutePredatorPreyLadderManifest:
    """M5 ladder manifest separating fixed-branch and source-route evidence."""

    rows: tuple[SourceRoutePredatorPreyLadderRow, ...]
    fixed_design_failure_interpretation: str
    source_route_claim_status: str
    production_token_emitted: bool

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("rows must be nonempty")
        if not all(isinstance(row, SourceRoutePredatorPreyLadderRow) for row in rows):
            raise TypeError("rows must contain SourceRoutePredatorPreyLadderRow")
        if not str(self.fixed_design_failure_interpretation).strip():
            raise ValueError("fixed_design_failure_interpretation must be nonempty")
        if not str(self.source_route_claim_status).strip():
            raise ValueError("source_route_claim_status must be nonempty")
        if bool(self.production_token_emitted):
            has_source_production_pass = any(
                row.route_label == SOURCE_FAITHFUL_ROUTE_LABEL
                and row.decision_status == "PASS_SOURCE_PRECONDITIONED_FILTERING"
                for row in rows
            )
            if not has_source_production_pass:
                raise ValueError(
                    "production token requires PASS_SOURCE_PRECONDITIONED_FILTERING"
                )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "fixed_design_failure_interpretation",
            str(self.fixed_design_failure_interpretation),
        )
        object.__setattr__(self, "source_route_claim_status", str(self.source_route_claim_status))
        object.__setattr__(self, "production_token_emitted", bool(self.production_token_emitted))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRoutePredatorPreyLadderManifest",
            "rows": [row.manifest_payload() for row in self.rows],
            "fixed_design_failure_interpretation": self.fixed_design_failure_interpretation,
            "source_route_claim_status": self.source_route_claim_status,
            "production_token_emitted": bool(self.production_token_emitted),
        }


@dataclass(frozen=True)
class SourceRouteSmoothingBoundary:
    """Boundary contract separating filtering evidence from smoothing evidence."""

    smoothing_status: str
    required_backward_fields: tuple[str, ...]
    filtering_tokens: tuple[str, ...]
    dedicated_smoothing_tokens: tuple[str, ...]
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        status = str(self.smoothing_status)
        if status not in ("deferred", "implemented"):
            raise ValueError("smoothing_status must be deferred or implemented")
        required = tuple(str(item) for item in self.required_backward_fields)
        if not required:
            raise ValueError("required_backward_fields must be nonempty")
        for required_name in (
            "backward_conditional_maps",
            "backward_weights",
            "smoothing_marginal_checks",
        ):
            if required_name not in required:
                raise ValueError(f"required_backward_fields missing {required_name}")
        filtering_tokens = tuple(str(item) for item in self.filtering_tokens)
        smoothing_tokens = tuple(str(item) for item in self.dedicated_smoothing_tokens)
        non_claims = tuple(str(item) for item in self.non_claims)
        if not non_claims:
            raise ValueError("non_claims must be nonempty")
        if status == "implemented" and not smoothing_tokens:
            raise ValueError("implemented smoothing requires a dedicated smoother token")
        if status == "deferred" and smoothing_tokens:
            raise ValueError("deferred smoothing cannot carry smoother pass tokens")
        object.__setattr__(self, "smoothing_status", status)
        object.__setattr__(self, "required_backward_fields", required)
        object.__setattr__(self, "filtering_tokens", filtering_tokens)
        object.__setattr__(self, "dedicated_smoothing_tokens", smoothing_tokens)
        object.__setattr__(self, "non_claims", non_claims)

    def filtering_tokens_are_smoothing_evidence(self) -> bool:
        return False

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "SourceRouteSmoothingBoundary",
            "smoothing_status": self.smoothing_status,
            "required_backward_fields": self.required_backward_fields,
            "filtering_tokens": self.filtering_tokens,
            "dedicated_smoothing_tokens": self.dedicated_smoothing_tokens,
            "filtering_tokens_are_smoothing_evidence": self.filtering_tokens_are_smoothing_evidence(),
            "non_claims": self.non_claims,
        }


@dataclass(frozen=True)
class SourceRouteRetainedObject:
    """Minimal retained object for the clean-room source-faithful filtering lane."""

    transport_object: object
    coordinate_frame: SourceRouteCoordinateFrame
    samples: tf.Tensor
    log_weights: tf.Tensor
    sample_diagnostics: SourceRouteSampleDiagnostics
    normalizer: SourceRouteNormalizerContribution
    measure_convention: MeasureConvention
    route_label: str
    storage_kind: str
    transition_interface: str
    branch_identity: BranchIdentity
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_frame, SourceRouteCoordinateFrame):
            raise TypeError("coordinate_frame must be a SourceRouteCoordinateFrame")
        if not isinstance(self.sample_diagnostics, SourceRouteSampleDiagnostics):
            raise TypeError("sample_diagnostics must be a SourceRouteSampleDiagnostics")
        if not isinstance(self.normalizer, SourceRouteNormalizerContribution):
            raise TypeError("normalizer must be a SourceRouteNormalizerContribution")
        assert_density_matches_mass(self.measure_convention)
        samples = tf.convert_to_tensor(self.samples, dtype=tf.float64)
        log_weights = tf.convert_to_tensor(self.log_weights, dtype=tf.float64)
        if samples.shape.rank != 2:
            raise ValueError(f"samples: {HighDimStatus.INVALID_SHAPE.value}")
        if samples.shape[0] != self.coordinate_frame.dimension:
            raise ValueError(f"samples: {HighDimStatus.INVALID_SHAPE.value}")
        if log_weights.shape != (int(samples.shape[1]),):
            raise ValueError(f"log_weights: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("samples", samples)
        assert_tf_float64("log_weights", log_weights)
        if not bool(
            tf.reduce_all(tf.math.is_finite(samples)).numpy()
            and tf.reduce_all(tf.math.is_finite(log_weights)).numpy()
        ):
            raise ValueError(f"SourceRouteRetainedObject: {HighDimStatus.NONFINITE_VALUE.value}")
        if int(samples.shape[1]) != self.sample_diagnostics.sample_count:
            raise ValueError("sample_diagnostics sample_count must match samples")
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be a BranchIdentity")
        _validate_route_storage_contract(
            route_label=str(self.route_label),
            storage_kind=str(self.storage_kind),
            transition_interface=str(self.transition_interface),
        )
        expected_manifest = source_route_retained_object_manifest(
            transport_object=self.transport_object,
            coordinate_frame=self.coordinate_frame,
            samples=samples,
            log_weights=log_weights,
            sample_diagnostics=self.sample_diagnostics,
            normalizer=self.normalizer,
            measure_convention=self.measure_convention,
            route_label=str(self.route_label),
            storage_kind=str(self.storage_kind),
            transition_interface=str(self.transition_interface),
            diagnostics=self.diagnostics,
        )
        expected_identity = BranchIdentity(
            manifest=expected_manifest,
            hash=expected_manifest.sha256(),
        )
        if self.branch_identity.hash.value != expected_identity.hash.value:
            raise ValueError(HighDimStatus.INVALID_BRANCH_MISMATCH.value)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "log_weights", log_weights)
        object.__setattr__(self, "route_label", str(self.route_label))
        object.__setattr__(self, "storage_kind", str(self.storage_kind))
        object.__setattr__(self, "transition_interface", str(self.transition_interface))
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


def normalize_log_weights(log_weights: tf.Tensor) -> tf.Tensor:
    """Return log weights normalized to sum to one."""

    values = _finite_vector("log_weights", log_weights)
    return values - tf.reduce_logsumexp(values)


def effective_sample_size_from_log_weights(log_weights: tf.Tensor) -> tf.Tensor:
    """Compute ESS from finite log weights."""

    normalized = tf.exp(normalize_log_weights(log_weights))
    return 1.0 / tf.reduce_sum(tf.square(normalized))


def source_route_needs_enhancement(
    *,
    effective_sample_size: tf.Tensor,
    sample_count: int,
    min_ess_fraction: float,
) -> bool:
    """Return whether source-route sampling should be enhanced by the ESS gate."""

    diagnostics = SourceRouteSampleDiagnostics(
        sample_count=int(sample_count),
        effective_sample_size=tf.convert_to_tensor(effective_sample_size, dtype=tf.float64),
    )
    threshold = float(min_ess_fraction)
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError("min_ess_fraction must be in (0, 1]")
    required = threshold * float(diagnostics.sample_count)
    return bool((diagnostics.effective_sample_size < required).numpy())


def source_route_proposal_log_weights(
    *,
    log_target_density: tf.Tensor,
    log_proposal_density: tf.Tensor,
) -> tf.Tensor:
    """Compute proposal-correction log weights in log-density convention.

    The source target is often written as a negative-log objective.  This helper
    deliberately does not accept negative-log values: callers must pass
    ``log_target_density``.  Use
    :func:`source_route_proposal_log_weights_from_negative_log_target` when the
    available target value is a negative log density.
    """

    log_target, log_proposal = _finite_same_shape_vectors(
        "log_target_density",
        log_target_density,
        "log_proposal_density",
        log_proposal_density,
    )
    return log_target - log_proposal


def source_route_proposal_log_weights_from_negative_log_target(
    *,
    negative_log_target: tf.Tensor,
    log_proposal_density: tf.Tensor,
) -> tf.Tensor:
    """Compute correction weights from a negative-log target value."""

    neg_log_target, log_proposal = _finite_same_shape_vectors(
        "negative_log_target",
        negative_log_target,
        "log_proposal_density",
        log_proposal_density,
    )
    return -neg_log_target - log_proposal


def source_route_discrete_log_normalizer_from_correction(
    *,
    log_proposal_density: tf.Tensor,
    correction_log_weights: tf.Tensor,
) -> tf.Tensor:
    """Exact discrete-support normalizer implied by proposal correction."""

    log_proposal, correction = _finite_same_shape_vectors(
        "log_proposal_density",
        log_proposal_density,
        "correction_log_weights",
        correction_log_weights,
    )
    return tf.reduce_logsumexp(log_proposal + correction)


def source_route_equal_weight_log_normalizer_estimate(
    correction_log_weights: tf.Tensor,
) -> tf.Tensor:
    """Monte Carlo log-normalizer estimate for equal-weight proposal samples."""

    correction = _finite_vector("correction_log_weights", correction_log_weights)
    sample_count = tf.cast(tf.shape(correction)[0], tf.float64)
    return tf.reduce_logsumexp(correction) - tf.math.log(sample_count)


def source_route_push_sample_batch(
    *,
    previous_batch: SourceRouteSampleBatch,
    propagated_samples: tf.Tensor,
    log_weights: tf.Tensor,
    time_index: int,
    sample_origin: str = "propagated",
) -> SourceRouteSampleBatch:
    """Create a propagated source-route sample batch with explicit metadata."""

    if not isinstance(previous_batch, SourceRouteSampleBatch):
        raise TypeError("previous_batch must be a SourceRouteSampleBatch")
    samples = tf.convert_to_tensor(propagated_samples, dtype=tf.float64)
    if samples.shape.rank != 2:
        raise ValueError(f"propagated_samples: {HighDimStatus.INVALID_SHAPE.value}")
    if int(samples.shape[1]) != previous_batch.sample_count:
        raise ValueError("propagated_samples must preserve sample count")
    if int(time_index) <= previous_batch.time_index:
        raise ValueError("time_index must advance")
    return SourceRouteSampleBatch(
        samples=samples,
        log_weights=log_weights,
        time_index=int(time_index),
        route_label=previous_batch.route_label,
        sample_origin=str(sample_origin),
    )


def source_route_push_and_augment_samples(
    *,
    previous_batch: SourceRouteSampleBatch,
    transition_fn,
    log_likelihood_fn,
    parameter_dim: int,
    state_dim: int,
    time_index: int,
) -> SourceRoutePushResult:
    """Mirror the source ``push_samples`` and full-solver augmentation step.

    ``previous_batch.samples`` must have shape ``[d + m, N]``.  The transition
    callable receives those samples and the integer time index, then returns
    the next state block with shape ``[m, N]``.  The likelihood callable receives
    the propagated ``[theta, x_t]`` samples and the same time index, then returns
    log likelihood values with shape ``[N]``.  The augmented batch has source
    shape ``[d + 2m, N] = [theta, x_t, x_{t-1}]``.
    """

    if not isinstance(previous_batch, SourceRouteSampleBatch):
        raise TypeError("previous_batch must be a SourceRouteSampleBatch")
    if previous_batch.route_label != SOURCE_FAITHFUL_ROUTE_LABEL:
        raise ValueError("source push requires source_faithful_filtering input")
    d = int(parameter_dim)
    m = int(state_dim)
    if d < 0 or m <= 0:
        raise ValueError("parameter_dim must be nonnegative and state_dim positive")
    expected_dim = d + m
    if previous_batch.dimension != expected_dim:
        raise ValueError("previous_batch dimension must equal parameter_dim + state_dim")
    if int(time_index) <= previous_batch.time_index:
        raise ValueError("time_index must advance")
    theta = previous_batch.samples[:d, :]
    previous_state = previous_batch.samples[d : d + m, :]
    next_state = tf.convert_to_tensor(
        transition_fn(previous_batch.samples, int(time_index)),
        dtype=tf.float64,
    )
    if next_state.shape != previous_state.shape:
        raise ValueError(f"transition output: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(next_state)).numpy()):
        raise ValueError(f"transition output: {HighDimStatus.NONFINITE_VALUE.value}")
    propagated_samples = tf.concat([theta, next_state], axis=0)
    log_likelihood = _finite_vector(
        "log_likelihood",
        log_likelihood_fn(propagated_samples, int(time_index)),
    )
    if log_likelihood.shape != previous_batch.log_weights.shape:
        raise ValueError(f"log_likelihood: {HighDimStatus.INVALID_SHAPE.value}")
    propagated_log_weights = normalize_log_weights(
        previous_batch.log_weights + log_likelihood
    )
    propagated = SourceRouteSampleBatch(
        samples=propagated_samples,
        log_weights=propagated_log_weights,
        time_index=int(time_index),
        route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
        sample_origin="propagated",
    )
    augmented = SourceRouteSampleBatch(
        samples=tf.concat([theta, next_state, previous_state], axis=0),
        log_weights=propagated_log_weights,
        time_index=int(time_index),
        route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
        sample_origin="augmented_propagated",
    )
    return SourceRoutePushResult(
        propagated_batch=propagated,
        augmented_batch=augmented,
        diagnostics=augmented.diagnostics(enhancement_attempts=0),
    )


def build_source_route_target(
    *,
    negative_log_physical_density_fn,
    coordinate_frame: SourceRouteCoordinateFrame,
    shift_constant: tf.Tensor,
    time_index: int,
    source_terms: tuple[str, ...] = ("prior_or_previous", "transition", "likelihood"),
    target_family: str = "full",
) -> SourceRouteTarget:
    """Build a clean-room source-route local target object."""

    return SourceRouteTarget(
        negative_log_physical_density_fn=negative_log_physical_density_fn,
        coordinate_frame=coordinate_frame,
        shift_constant=shift_constant,
        time_index=int(time_index),
        target_family=str(target_family),
        source_terms=tuple(source_terms),
    )


def source_route_generate_retained_samples(
    *,
    target: SourceRouteTarget,
    transport: SourceRouteTransportProtocol,
    reference_samples: tf.Tensor,
    time_index: int,
) -> SourceRouteRetainedSampleResult:
    """Generate retained samples and proposal-correction weights."""

    if not isinstance(target, SourceRouteTarget):
        raise TypeError("target must be SourceRouteTarget")
    if not isinstance(transport, SourceRouteTransportProtocol):
        raise TypeError("transport must be SourceRouteTransportProtocol")
    if int(time_index) != target.time_index:
        raise ValueError("time_index must match target")
    reference = tf.convert_to_tensor(reference_samples, dtype=tf.float64)
    if reference.shape.rank != 2:
        raise ValueError(f"reference_samples: {HighDimStatus.INVALID_SHAPE.value}")
    if int(reference.shape[0]) != target.coordinate_frame.dimension:
        raise ValueError(f"reference_samples: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(reference)).numpy()):
        raise ValueError(f"reference_samples: {HighDimStatus.NONFINITE_VALUE.value}")
    local_samples = transport.inverse_transport(reference)
    physical_samples = target.physical_points_from_reference(local_samples)
    proposal_log_density = transport.proposal_log_density(
        local_points=local_samples,
        reference_points=reference,
    )
    target_log_density = target.log_target_density(local_samples)
    correction = source_route_proposal_log_weights(
        log_target_density=target_log_density,
        log_proposal_density=proposal_log_density,
    )
    normalized_correction = normalize_log_weights(correction)
    retained_batch = SourceRouteSampleBatch(
        samples=physical_samples,
        log_weights=normalized_correction,
        time_index=int(time_index),
        route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
        sample_origin="retained_from_transport",
    )
    diagnostics = retained_batch.diagnostics()
    normalizer = SourceRouteNormalizerContribution(
        log_transport_normalizer=transport.log_normalizer(),
        shift_constant=target.shift_constant,
        log_abs_det_policy=target.log_abs_det_policy,
    )
    return SourceRouteRetainedSampleResult(
        retained_batch=retained_batch,
        proposal_log_density=proposal_log_density,
        target_log_density=target_log_density,
        correction_log_weights=correction,
        diagnostics=diagnostics,
        normalizer=normalizer,
    )


def source_route_previous_marginal_log_density(
    *,
    previous_retained_object: SourceRouteRetainedObject,
    physical_points: tf.Tensor,
    keep_axes: tuple[int, ...],
) -> SourceRoutePreviousMarginalDensityResult:
    """Evaluate the previous retained marginal density used at ``t > 1``.

    This mirrors the author-code prior term in ``full_sol.reapprox``:
    marginalize the previous SIRT to the retained ``[theta, x_{t-1}]`` prefix,
    invert the previous affine frame on that prefix, evaluate the marginal
    SIRT density, then divide by the prefix affine determinant.
    """

    if not isinstance(previous_retained_object, SourceRouteRetainedObject):
        raise TypeError("previous_retained_object must be SourceRouteRetainedObject")
    keep = tuple(int(axis) for axis in keep_axes)
    if not keep:
        raise ValueError("keep_axes must be nonempty")
    if keep != tuple(range(len(keep))):
        raise ValueError("source previous marginalization requires prefix keep axes")
    frame = previous_retained_object.coordinate_frame
    if keep[-1] >= frame.dimension:
        raise ValueError(f"keep_axes: {HighDimStatus.INVALID_SHAPE.value}")
    points = tf.convert_to_tensor(physical_points, dtype=tf.float64)
    if points.shape.rank != 2 or int(points.shape[0]) != len(keep):
        raise ValueError(f"physical_points: {HighDimStatus.INVALID_SHAPE.value}")
    assert_tf_float64("physical_points", points)
    if not bool(tf.reduce_all(tf.math.is_finite(points)).numpy()):
        raise ValueError(f"physical_points: {HighDimStatus.NONFINITE_VALUE.value}")
    transport = SourceRouteTransportProtocol(
        previous_retained_object.transport_object
    )
    marginal_transport = transport.marginalize(keep)
    mu_prefix = tf.gather(frame.mu, keep)
    matrix_prefix = tf.gather(tf.gather(frame.matrix, keep, axis=0), keep, axis=1)
    local_points = tf.linalg.solve(
        matrix_prefix,
        points - mu_prefix[:, tf.newaxis],
    )
    eval_pdf = _source_route_eval_marginal_pdf(marginal_transport, local_points)
    if not bool(tf.reduce_all(eval_pdf > 0.0).numpy()):
        raise ValueError(f"previous_marginal_eval_pdf: {HighDimStatus.NONFINITE_VALUE.value}")
    log_density = tf.math.log(eval_pdf) - tf.math.log(
        tf.abs(tf.linalg.det(matrix_prefix))
    )
    return SourceRoutePreviousMarginalDensityResult(
        previous_retained_object=previous_retained_object,
        keep_axes=keep,
        marginal_transport=marginal_transport,
        physical_points=points,
        local_points=local_points,
        log_density=log_density,
    )


def source_route_coordinate_frame_hash(
    coordinate_frame: SourceRouteCoordinateFrame,
) -> str:
    """Return the canonical manifest hash for a source-route affine frame."""

    if not isinstance(coordinate_frame, SourceRouteCoordinateFrame):
        raise TypeError("coordinate_frame must be SourceRouteCoordinateFrame")
    return BranchManifest(
        version="source_route_coordinate_frame.v1",
        payload=coordinate_frame.manifest_payload(),
    ).sha256().value


def source_route_callable_identity(fn) -> str:
    """Return the stable callable identity used by P90 bridge bindings."""

    if not callable(fn):
        raise TypeError("fn must be callable")
    module = str(getattr(fn, "__module__", type(fn).__module__))
    qualname = str(getattr(fn, "__qualname__", type(fn).__qualname__))
    return f"{module}.{qualname}"


def source_route_author_formula_negative_log_physical_density(
    *,
    physical_points: tf.Tensor,
    binding: SourceRouteValueBridgeBinding,
    transition_log_density_fn,
    likelihood_log_density_fn,
    prior_log_density_fn=None,
    previous_retained_object: SourceRouteRetainedObject | None = None,
    current_transport_branch_hash: str,
    current_coordinate_frame: SourceRouteCoordinateFrame,
) -> SourceRouteAuthorFormulaReplayResult:
    """Replay the author formula without calling the production scalar helper."""

    if not isinstance(binding, SourceRouteValueBridgeBinding):
        raise TypeError("binding must be SourceRouteValueBridgeBinding")
    points = tf.convert_to_tensor(physical_points, dtype=tf.float64)
    if points.shape.rank != 2:
        raise ValueError(f"physical_points: {HighDimStatus.INVALID_SHAPE.value}")
    if tuple(int(item) for item in points.shape) != binding.physical_shape:
        raise ValueError("P90 value bridge physical shape mismatch")
    assert_tf_float64("physical_points", points)
    if not bool(tf.reduce_all(tf.math.is_finite(points)).numpy()):
        raise ValueError(f"physical_points: {HighDimStatus.NONFINITE_VALUE.value}")
    if str(current_transport_branch_hash) != binding.transport_branch_hash:
        raise ValueError("P90 value bridge transport branch hash mismatch")
    if source_route_coordinate_frame_hash(current_coordinate_frame) != binding.coordinate_frame_hash:
        raise ValueError("P90 value bridge coordinate frame hash mismatch")
    if not callable(transition_log_density_fn):
        raise TypeError("transition_log_density_fn must be callable")
    if not callable(likelihood_log_density_fn):
        raise TypeError("likelihood_log_density_fn must be callable")
    if source_route_callable_identity(transition_log_density_fn) != binding.transition_log_density_id:
        raise ValueError("P90 value bridge transition function identity mismatch")
    if source_route_callable_identity(likelihood_log_density_fn) != binding.likelihood_log_density_id:
        raise ValueError("P90 value bridge likelihood function identity mismatch")
    d = binding.parameter_dim
    m = binding.state_dim
    previous_input_axes = tuple(range(d)) + tuple(range(d + m, d + 2 * m))
    prior_points = tf.gather(points, previous_input_axes, axis=0)
    previous_local_points = None
    if binding.time_index == 1:
        if prior_log_density_fn is None or not callable(prior_log_density_fn):
            raise TypeError("t=1 bridge replay requires callable prior_log_density_fn")
        if source_route_callable_identity(prior_log_density_fn) != binding.prior_log_density_id:
            raise ValueError("P90 value bridge prior function identity mismatch")
        if previous_retained_object is not None:
            raise ValueError("t=1 bridge replay cannot include previous retained object")
        prior_or_previous_log_density = _finite_vector(
            "prior_log_density",
            prior_log_density_fn(prior_points),
        )
    else:
        if prior_log_density_fn is not None:
            raise ValueError("t>1 bridge replay uses previous retained marginal")
        if previous_retained_object is None:
            raise TypeError("t>1 bridge replay requires previous_retained_object")
        if (
            previous_retained_object.branch_identity.hash.value
            != binding.previous_retained_hash
        ):
            raise ValueError("P90 value bridge previous retained hash mismatch")
        keep = binding.previous_marginal_keep_axes
        if keep is None or keep != tuple(range(d + m)):
            raise ValueError("P90 value bridge previous keep axes mismatch")
        transport = SourceRouteTransportProtocol(
            previous_retained_object.transport_object
        )
        marginal_transport = transport.marginalize(keep)
        frame = previous_retained_object.coordinate_frame
        mu_prefix = tf.gather(frame.mu, keep)
        matrix_prefix = tf.gather(tf.gather(frame.matrix, keep, axis=0), keep, axis=1)
        previous_local_points = tf.linalg.solve(
            matrix_prefix,
            prior_points - mu_prefix[:, tf.newaxis],
        )
        eval_pdf = _source_route_eval_marginal_pdf(
            marginal_transport,
            previous_local_points,
        )
        if not bool(tf.reduce_all(eval_pdf > 0.0).numpy()):
            raise ValueError(
                f"previous_marginal_eval_pdf: {HighDimStatus.NONFINITE_VALUE.value}"
            )
        prior_or_previous_log_density = tf.math.log(eval_pdf) - tf.math.log(
            tf.abs(tf.linalg.det(matrix_prefix))
        )
    transition_log_density = _finite_vector(
        "transition_log_density",
        transition_log_density_fn(points, binding.time_index),
    )
    likelihood_log_density = _finite_vector(
        "likelihood_log_density",
        likelihood_log_density_fn(points, binding.time_index),
    )
    if (
        prior_or_previous_log_density.shape != transition_log_density.shape
        or prior_or_previous_log_density.shape != likelihood_log_density.shape
    ):
        raise ValueError(f"source sequential density: {HighDimStatus.INVALID_SHAPE.value}")
    return SourceRouteAuthorFormulaReplayResult(
        binding=binding,
        physical_points=points,
        prior_or_previous_log_density=prior_or_previous_log_density,
        transition_log_density=transition_log_density,
        likelihood_log_density=likelihood_log_density,
        negative_log_density=(
            -prior_or_previous_log_density
            - transition_log_density
            - likelihood_log_density
        ),
        previous_marginal_local_points=previous_local_points,
    )


def _source_route_eval_marginal_pdf(marginal_transport: object, local_points: tf.Tensor) -> tf.Tensor:
    local = tf.convert_to_tensor(local_points, dtype=tf.float64)
    if local.shape.rank != 2:
        raise ValueError(f"local_points: {HighDimStatus.INVALID_SHAPE.value}")
    if callable(getattr(marginal_transport, "eval_pdf", None)):
        values = tf.convert_to_tensor(marginal_transport.eval_pdf(local), dtype=tf.float64)
    elif callable(getattr(marginal_transport, "normalized_retained_density_values", None)):
        values = tf.convert_to_tensor(
            marginal_transport.normalized_retained_density_values(tf.transpose(local)),
            dtype=tf.float64,
        )
    else:
        raise TypeError("marginal transport must provide eval_pdf or normalized_retained_density_values")
    if values.shape != (int(local.shape[1]),):
        raise ValueError(f"previous_marginal_eval_pdf: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
        raise ValueError(f"previous_marginal_eval_pdf: {HighDimStatus.NONFINITE_VALUE.value}")
    return values


def source_route_sequential_negative_log_physical_density(
    *,
    physical_points: tf.Tensor,
    time_index: int,
    parameter_dim: int,
    state_dim: int,
    transition_log_density_fn,
    likelihood_log_density_fn,
    prior_log_density_fn=None,
    previous_retained_object: SourceRouteRetainedObject | None = None,
) -> tf.Tensor:
    """Author-style sequential negative-log density in physical coordinates.

    Physical points use the source ordering ``[theta, x_t, x_{t-1}]``.  At
    ``t=1`` the prior term is supplied by ``prior_log_density_fn`` on
    ``[theta, x_0]``.  At ``t>1`` it is supplied by the previous retained SIRT
    marginal over its ``[theta, x_{t-1}]`` prefix.
    """

    points = tf.convert_to_tensor(physical_points, dtype=tf.float64)
    if points.shape.rank != 2:
        raise ValueError(f"physical_points: {HighDimStatus.INVALID_SHAPE.value}")
    d = int(parameter_dim)
    m = int(state_dim)
    if d < 0 or m <= 0:
        raise ValueError("parameter_dim must be nonnegative and state_dim positive")
    expected_dim = d + 2 * m
    if int(points.shape[0]) != expected_dim:
        raise ValueError(f"physical_points: {HighDimStatus.INVALID_SHAPE.value}")
    if int(time_index) < 1:
        raise ValueError("time_index must be positive")
    if not callable(transition_log_density_fn):
        raise TypeError("transition_log_density_fn must be callable")
    if not callable(likelihood_log_density_fn):
        raise TypeError("likelihood_log_density_fn must be callable")
    input_axes = tuple(range(d)) + tuple(range(d + m, d + 2 * m))
    prior_points = tf.gather(points, input_axes, axis=0)
    if int(time_index) == 1:
        if prior_log_density_fn is None or not callable(prior_log_density_fn):
            raise TypeError("t=1 requires callable prior_log_density_fn")
        prior_log_density = _finite_vector(
            "prior_log_density",
            prior_log_density_fn(prior_points),
        )
    else:
        if prior_log_density_fn is not None:
            raise ValueError("t>1 uses previous retained marginal, not prior_log_density_fn")
        if previous_retained_object is None:
            raise TypeError("t>1 requires previous_retained_object")
        keep_axes = tuple(range(d + m))
        previous_density = source_route_previous_marginal_log_density(
            previous_retained_object=previous_retained_object,
            physical_points=prior_points,
            keep_axes=keep_axes,
        )
        prior_log_density = previous_density.log_density
    transition_log_density = _finite_vector(
        "transition_log_density",
        transition_log_density_fn(points, int(time_index)),
    )
    likelihood_log_density = _finite_vector(
        "likelihood_log_density",
        likelihood_log_density_fn(points, int(time_index)),
    )
    if (
        prior_log_density.shape != transition_log_density.shape
        or prior_log_density.shape != likelihood_log_density.shape
    ):
        raise ValueError(f"source sequential density: {HighDimStatus.INVALID_SHAPE.value}")
    return -prior_log_density - transition_log_density - likelihood_log_density


def source_route_default_operation_audit(
    *,
    source_anchor: str = "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-130",
    notes: str = "P57-M6 fixed-HMC source-loop operation coverage",
) -> SourceRouteImplementationAudit:
    """Build the fixed-HMC source-loop operation audit used by M6."""

    return SourceRouteImplementationAudit(
        records=tuple(
            SourceRouteOperationRecord(
                operation_id=operation_id,
                source_anchor=str(source_anchor),
                implementation_status="implemented",
                route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
                notes=str(notes),
            )
            for operation_id in SOURCE_ROUTE_REQUIRED_OPERATION_IDS
        ),
    )


def _source_route_step_target_from_components(
    *,
    spec: SourceRouteSequentialStepSpec,
    previous_retained_object: SourceRouteRetainedObject | None,
) -> SourceRouteTarget:
    def negative_log_physical(points: tf.Tensor) -> tf.Tensor:
        return spec.density_components.negative_log_physical_density(
            physical_points=points,
            time_index=spec.time_index,
            previous_retained_object=previous_retained_object,
        )

    return SourceRouteTarget(
        negative_log_physical_density_fn=negative_log_physical,
        coordinate_frame=spec.target.coordinate_frame,
        shift_constant=spec.target.shift_constant,
        time_index=spec.time_index,
        target_family=spec.target.target_family,
        source_terms=spec.target.source_terms,
        log_abs_det_policy=spec.target.log_abs_det_policy,
    )


def source_route_run_sequential_fixed_hmc(
    *,
    step_specs: tuple[SourceRouteSequentialStepSpec, ...],
    branch_audit: SourceRouteImplementationAudit | None = None,
) -> SourceRouteSequentialResult:
    """Run the replayable sequential source-route skeleton for fixed HMC.

    The caller supplies frozen targets, transports, reference samples, and
    branch schedules.  This function enforces the retained-object carry and the
    previous retained marginalization gate; it does not fit TT/SIRT objects or
    choose ranks.
    """

    specs = tuple(step_specs)
    if len(specs) < 2:
        raise ValueError("sequential source loop requires at least two steps")
    if not all(isinstance(spec, SourceRouteSequentialStepSpec) for spec in specs):
        raise TypeError("step_specs must contain SourceRouteSequentialStepSpec")
    expected_times = tuple(range(1, len(specs) + 1))
    actual_times = tuple(spec.time_index for spec in specs)
    if actual_times != expected_times:
        raise ValueError("step_specs must be consecutive starting at t=1")
    audit = branch_audit or source_route_default_operation_audit()
    if audit.status != "PASS_SOURCE_ROUTE_OPERATION_COVERAGE":
        raise ValueError("branch_audit must pass source-route operation coverage")

    previous_retained: SourceRouteRetainedObject | None = None
    steps: list[SourceRouteSequentialStepResult] = []
    for spec in specs:
        step_target = _source_route_step_target_from_components(
            spec=spec,
            previous_retained_object=previous_retained,
        )
        current_reference = spec.transport.inverse_transport(spec.reference_samples)
        current_physical = step_target.physical_points_from_reference(
            current_reference
        )
        if spec.time_index == 1:
            previous_marginal_density = None
        else:
            if previous_retained is None:
                raise ValueError("t>1 requires previous retained object")
            if (
                spec.previous_marginal_keep_axes is None
                or spec.previous_marginal_input_axes is None
            ):
                raise ValueError("t>1 requires previous marginal axes")
            previous_physical = tf.gather(
                current_physical,
                spec.previous_marginal_input_axes,
                axis=0,
            )
            previous_marginal_density = source_route_previous_marginal_log_density(
                previous_retained_object=previous_retained,
                physical_points=previous_physical,
                keep_axes=spec.previous_marginal_keep_axes,
            )

        component_negative_log = spec.density_components.negative_log_physical_density(
            physical_points=current_physical,
            time_index=spec.time_index,
            previous_retained_object=previous_retained,
        )
        target_negative_log = _finite_vector(
            "target_negative_log_physical_density",
            step_target.negative_log_physical_density_fn(current_physical),
        )
        tf.debugging.assert_near(
            target_negative_log,
            component_negative_log,
            atol=1e-10,
        )

        retained_samples = source_route_generate_retained_samples(
            target=step_target,
            transport=spec.transport,
            reference_samples=spec.reference_samples,
            time_index=spec.time_index,
        )
        diagnostics = {
            "phase": "P57-M6",
            "time_index": int(spec.time_index),
            "sequential_status": "sequential_fixed_hmc_source_loop",
            "source_anchor": (
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-130"
            ),
            "previous_retained_hash": (
                None
                if previous_retained is None
                else previous_retained.branch_identity.hash.value
            ),
            "previous_marginal_keep_axes": spec.previous_marginal_keep_axes,
            "previous_marginal_input_axes": spec.previous_marginal_input_axes,
        }
        identity = source_route_retained_object_identity(
            transport_object=spec.transport.transport_object,
            coordinate_frame=step_target.coordinate_frame,
            samples=retained_samples.retained_batch.samples,
            log_weights=retained_samples.retained_batch.log_weights,
            sample_diagnostics=retained_samples.diagnostics,
            normalizer=retained_samples.normalizer,
            measure_convention=spec.measure_convention,
            route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
            storage_kind=spec.storage_kind,
            transition_interface=spec.transition_interface,
            diagnostics=diagnostics,
        )
        retained_object = SourceRouteRetainedObject(
            transport_object=spec.transport.transport_object,
            coordinate_frame=step_target.coordinate_frame,
            samples=retained_samples.retained_batch.samples,
            log_weights=retained_samples.retained_batch.log_weights,
            sample_diagnostics=retained_samples.diagnostics,
            normalizer=retained_samples.normalizer,
            measure_convention=spec.measure_convention,
            route_label=SOURCE_FAITHFUL_ROUTE_LABEL,
            storage_kind=spec.storage_kind,
            transition_interface=spec.transition_interface,
            branch_identity=identity,
            diagnostics=diagnostics,
        )
        step = SourceRouteSequentialStepResult(
            spec=spec,
            target=step_target,
            retained_samples=retained_samples,
            retained_object=retained_object,
            previous_retained_object=previous_retained,
            previous_marginal_density=previous_marginal_density,
        )
        steps.append(step)
        previous_retained = retained_object

    return SourceRouteSequentialResult(
        steps=tuple(steps),
        branch_audit=audit,
    )


def source_route_one_step_reapproximation(
    *,
    target: SourceRouteTarget,
    transport: SourceRouteTransportProtocol,
    reference_samples: tf.Tensor,
    time_index: int,
    previous_retained_object: object | None = None,
) -> SourceRouteOneStepResult:
    """Run the P55 one-step source reapproximation boundary.

    This is deliberately limited to ``t=1``.  A later phase must implement
    previous retained-object marginalization before any sequential source-route
    claim is allowed.
    """

    if int(time_index) != 1 or target.time_index != 1:
        raise ValueError(
            "P55 one-step source reapproximation only supports t=1; "
            "previous retained-object marginalization is not implemented"
        )
    if previous_retained_object is not None:
        raise ValueError(
            "previous retained-object marginalization is not implemented"
        )
    retained = source_route_generate_retained_samples(
        target=target,
        transport=transport,
        reference_samples=reference_samples,
        time_index=int(time_index),
    )
    return SourceRouteOneStepResult(
        target=target,
        transport=transport,
        retained_samples=retained,
        time_index=int(time_index),
    )


def source_route_recenter(
    *,
    samples: tf.Tensor,
    log_weights: tf.Tensor,
    expansion_factor: float,
    covariance_jitter: float = 1e-5,
    quantile_fraction: float = 0.01,
    min_ess_for_quantile_scale: float = 1000.0,
    use_quantile_scale: bool = True,
) -> SourceRouteCoordinateFrame:
    """Compute the source-route weighted affine recentering frame.

    This follows the author-code ``computeL`` structure: weighted mean,
    weighted covariance, Cholesky factor with jitter, and, when ESS is large
    enough, a diagonal quantile stretch in whitened coordinates.  The final
    expansion factor corresponds to the source ``epd`` multiplier.
    """

    sample_tensor = tf.convert_to_tensor(samples, dtype=tf.float64)
    if sample_tensor.shape.rank != 2:
        raise ValueError(f"samples: {HighDimStatus.INVALID_SHAPE.value}")
    log_weight_tensor = tf.convert_to_tensor(log_weights, dtype=tf.float64)
    if log_weight_tensor.shape != (int(sample_tensor.shape[1]),):
        raise ValueError(f"log_weights: {HighDimStatus.INVALID_SHAPE.value}")
    finite_columns = tf.reduce_all(tf.math.is_finite(sample_tensor), axis=0)
    finite_weights = tf.math.is_finite(log_weight_tensor)
    keep = tf.logical_and(finite_columns, finite_weights)
    if not bool(tf.reduce_any(keep).numpy()):
        raise ValueError(f"samples/log_weights: {HighDimStatus.NONFINITE_VALUE.value}")
    sample_tensor = tf.boolean_mask(sample_tensor, keep, axis=1)
    log_weight_tensor = tf.boolean_mask(log_weight_tensor, keep)
    weights = tf.exp(normalize_log_weights(log_weight_tensor))
    if float(expansion_factor) <= 0.0:
        raise ValueError("expansion_factor must be positive")
    jitter = float(covariance_jitter)
    if jitter < 0.0:
        raise ValueError("covariance_jitter must be nonnegative")
    q = float(quantile_fraction)
    if q <= 0.0 or q >= 0.5:
        raise ValueError("quantile_fraction must be in (0, 0.5)")
    min_ess = float(min_ess_for_quantile_scale)
    if min_ess < 0.0:
        raise ValueError("min_ess_for_quantile_scale must be nonnegative")
    assert_tf_float64("samples", sample_tensor)
    mu = tf.reduce_sum(sample_tensor * weights[tf.newaxis, :], axis=1)
    centered = sample_tensor - mu[:, tf.newaxis]
    covariance = tf.einsum("n,in,jn->ij", weights, centered, centered)
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    if jitter > 0.0:
        dim = int(sample_tensor.shape[0])
        covariance = covariance + tf.eye(dim, dtype=tf.float64) * tf.constant(
            jitter,
            dtype=tf.float64,
        )
    matrix = tf.linalg.cholesky(covariance)
    ess = effective_sample_size_from_log_weights(log_weight_tensor)
    if bool(use_quantile_scale) and bool((ess > min_ess).numpy()):
        standardized = tf.linalg.triangular_solve(matrix, centered, lower=True)
        scale_diag = _source_route_computeL_quantile_scale(
            standardized,
            weights,
            quantile_fraction=q,
        )
        matrix = tf.matmul(matrix, tf.linalg.diag(scale_diag))
    matrix = matrix * tf.constant(
        float(expansion_factor),
        dtype=tf.float64,
    )
    return SourceRouteCoordinateFrame(
        mu=mu,
        matrix=matrix,
        expansion_factor=float(expansion_factor),
    )


def source_route_reference_log_density_from_physical(
    *,
    log_physical_density: tf.Tensor,
    coordinate_frame: SourceRouteCoordinateFrame,
) -> tf.Tensor:
    """Convert physical log density to local-coordinate log density."""

    log_density = tf.convert_to_tensor(log_physical_density, dtype=tf.float64)
    if log_density.shape.rank not in (0, 1):
        raise ValueError(f"log_physical_density: {HighDimStatus.INVALID_SHAPE.value}")
    assert_tf_float64("log_physical_density", log_density)
    if not bool(tf.reduce_all(tf.math.is_finite(log_density)).numpy()):
        raise ValueError(f"log_physical_density: {HighDimStatus.NONFINITE_VALUE.value}")
    return log_density + coordinate_frame.log_abs_det()


def source_route_shifted_negative_log_target(
    *,
    negative_log_target: tf.Tensor,
    shift_constant: tf.Tensor,
) -> tf.Tensor:
    """Apply the source-style stability shift to a negative-log target."""

    target = tf.convert_to_tensor(negative_log_target, dtype=tf.float64)
    shift = tf.convert_to_tensor(shift_constant, dtype=tf.float64)
    if target.shape.rank not in (0, 1):
        raise ValueError(f"negative_log_target: {HighDimStatus.INVALID_SHAPE.value}")
    if shift.shape.rank != 0:
        raise ValueError(f"shift_constant: {HighDimStatus.INVALID_SHAPE.value}")
    assert_tf_float64("negative_log_target", target)
    assert_tf_float64("shift_constant", shift)
    if not bool(
        tf.reduce_all(tf.math.is_finite(target)).numpy()
        and tf.math.is_finite(shift).numpy()
    ):
        raise ValueError(f"source_route_shifted_negative_log_target: {HighDimStatus.NONFINITE_VALUE.value}")
    return target - shift


def source_route_log_normalizer_update(
    *,
    log_transport_normalizer: tf.Tensor,
    shift_constant: tf.Tensor,
) -> tf.Tensor:
    """Return the source-style log-likelihood increment `log(z) - const`."""

    normalizer = SourceRouteNormalizerContribution(
        log_transport_normalizer=log_transport_normalizer,
        shift_constant=shift_constant,
        log_abs_det_policy="included_in_target",
    )
    return normalizer.log_increment()


def source_route_residual_negative_log_target(
    *,
    full_negative_log_target: tf.Tensor,
    preconditioner_negative_log_target: tf.Tensor,
) -> tf.Tensor:
    """Return residual negative-log target so preconditioner plus residual is full."""

    full, preconditioner = _finite_same_shape_vectors(
        "full_negative_log_target",
        full_negative_log_target,
        "preconditioner_negative_log_target",
        preconditioner_negative_log_target,
    )
    return full - preconditioner


def source_route_preconditioned_target_identity_error(
    *,
    full_negative_log_target: tf.Tensor,
    preconditioner_negative_log_target: tf.Tensor,
    residual_negative_log_target: tf.Tensor,
) -> tf.Tensor:
    """Return max absolute error in the full = preconditioner + residual identity."""

    full, preconditioner = _finite_same_shape_vectors(
        "full_negative_log_target",
        full_negative_log_target,
        "preconditioner_negative_log_target",
        preconditioner_negative_log_target,
    )
    _, residual = _finite_same_shape_vectors(
        "full_negative_log_target",
        full,
        "residual_negative_log_target",
        residual_negative_log_target,
    )
    return tf.reduce_max(tf.abs(full - (preconditioner + residual)))


def source_route_retained_object_identity(
    *,
    transport_object: object,
    coordinate_frame: SourceRouteCoordinateFrame,
    samples: tf.Tensor,
    log_weights: tf.Tensor,
    sample_diagnostics: SourceRouteSampleDiagnostics,
    normalizer: SourceRouteNormalizerContribution,
    measure_convention: MeasureConvention,
    route_label: str,
    storage_kind: str,
    transition_interface: str,
    diagnostics: Mapping[str, object] | None = None,
) -> BranchIdentity:
    """Build the canonical identity for a source-route retained object."""

    manifest = source_route_retained_object_manifest(
        transport_object=transport_object,
        coordinate_frame=coordinate_frame,
        samples=tf.convert_to_tensor(samples, dtype=tf.float64),
        log_weights=tf.convert_to_tensor(log_weights, dtype=tf.float64),
        sample_diagnostics=sample_diagnostics,
        normalizer=normalizer,
        measure_convention=measure_convention,
        route_label=str(route_label),
        storage_kind=str(storage_kind),
        transition_interface=str(transition_interface),
        diagnostics=diagnostics,
    )
    return BranchIdentity(manifest=manifest, hash=manifest.sha256())


def source_route_retained_object_manifest(
    *,
    transport_object: object,
    coordinate_frame: SourceRouteCoordinateFrame,
    samples: tf.Tensor,
    log_weights: tf.Tensor,
    sample_diagnostics: SourceRouteSampleDiagnostics,
    normalizer: SourceRouteNormalizerContribution,
    measure_convention: MeasureConvention,
    route_label: str,
    storage_kind: str,
    transition_interface: str,
    diagnostics: Mapping[str, object] | None = None,
) -> BranchManifest:
    """Create a canonical retained-object manifest."""

    assert_density_matches_mass(measure_convention)
    _validate_route_storage_contract(
        route_label=str(route_label),
        storage_kind=str(storage_kind),
        transition_interface=str(transition_interface),
    )
    transport_payload = _transport_manifest_payload(transport_object)
    return BranchManifest(
        version="source_route_retained_object.v1",
        payload={
            "transport_object": transport_payload,
            "coordinate_frame": coordinate_frame.manifest_payload(),
            "samples": samples,
            "log_weights": log_weights,
            "sample_diagnostics": sample_diagnostics.manifest_payload(),
            "normalizer": normalizer.manifest_payload(),
            "measure_convention": {
                "density_measure": measure_convention.density_measure.value,
                "mass_measure": measure_convention.mass_measure.value,
                "reference_weight_name": measure_convention.reference_weight_name,
                "physical_coordinate_name": measure_convention.physical_coordinate_name,
                "reference_coordinate_name": measure_convention.reference_coordinate_name,
                "dtype_name": measure_convention.dtype_name,
            },
            "route_label": str(route_label),
            "storage_kind": str(storage_kind),
            "transition_interface": str(transition_interface),
            "diagnostics": freeze_mapping(diagnostics),
        },
    )


def _validate_route_storage_contract(
    *,
    route_label: str,
    storage_kind: str,
    transition_interface: str,
) -> None:
    if not route_label.strip():
        raise ValueError("route_label must be nonempty")
    if not storage_kind.strip():
        raise ValueError("storage_kind must be nonempty")
    if not transition_interface.strip():
        raise ValueError("transition_interface must be nonempty")
    if route_label == SOURCE_FAITHFUL_ROUTE_LABEL:
        if storage_kind in _FORBIDDEN_SOURCE_STORAGE_KINDS:
            raise ValueError(
                "source-faithful retained object cannot use all-grid retained storage"
            )
        if transition_interface in _FORBIDDEN_SOURCE_TRANSITION_INTERFACES:
            raise ValueError(
                "source-faithful retained object cannot use pairwise grid propagation"
            )


def _transport_manifest_payload(transport_object: object) -> Mapping[str, object]:
    manifest_payload = getattr(transport_object, "manifest_payload", None)
    if callable(manifest_payload):
        payload = manifest_payload()
    elif isinstance(transport_object, SquaredTTMarginal):
        payload = {
            "family": "SquaredTTMarginal",
            "source_contract_level": "fixed_ttsirt_marginal",
            "keep_axes": transport_object.keep_axes,
            "normalizer": transport_object.normalizer,
            "diagnostics": transport_object.diagnostics,
        }
    else:
        raise TypeError("transport_object must expose manifest_payload()")
    if not isinstance(payload, Mapping):
        raise TypeError("transport_object manifest_payload() must return a mapping")
    return payload


def _source_route_computeL_quantile_scale(
    standardized_samples: tf.Tensor,
    weights: tf.Tensor,
    *,
    quantile_fraction: float,
) -> tf.Tensor:
    samples = tf.convert_to_tensor(standardized_samples, dtype=tf.float64)
    normalized_weights = tf.convert_to_tensor(weights, dtype=tf.float64)
    if samples.shape.rank != 2 or normalized_weights.shape != (int(samples.shape[1]),):
        raise ValueError(f"quantile scale: {HighDimStatus.INVALID_SHAPE.value}")
    q = tf.constant(float(quantile_fraction), dtype=tf.float64)
    normal_q = tfp_normal_quantile(q)
    scales = []
    for axis in range(int(samples.shape[0])):
        values = samples[axis, :]
        order = tf.argsort(values, stable=True)
        sorted_values = tf.gather(values, order)
        sorted_weights = tf.gather(normalized_weights, order)
        cumulative = tf.cumsum(sorted_weights)
        left_index = tf.argmax(tf.cast(cumulative > q, tf.int32), output_type=tf.int32)
        right_index = tf.argmax(
            tf.cast(cumulative > (1.0 - q), tf.int32),
            output_type=tf.int32,
        )
        width = tf.gather(sorted_values, right_index) - tf.gather(sorted_values, left_index)
        scale = -width / normal_q / 2.0
        scales.append(tf.maximum(scale, tf.constant(1e-12, dtype=tf.float64)))
    return tf.stack(scales)


def tfp_normal_quantile(probability: tf.Tensor) -> tf.Tensor:
    """Return the standard-normal quantile using TensorFlow primitives."""

    p = tf.convert_to_tensor(probability, dtype=tf.float64)
    if p.shape.rank != 0:
        raise ValueError(f"probability: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.math.is_finite(p).numpy()) or not bool((p > 0.0).numpy() and (p < 1.0).numpy()):
        raise ValueError("probability must be in (0, 1)")
    return tf.sqrt(tf.constant(2.0, dtype=tf.float64)) * tf.math.erfinv(
        2.0 * p - 1.0
    )


def _finite_vector(name: str, value: tf.Tensor) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank != 1:
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
    assert_tf_float64(name, tensor)
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
    return tensor


def _finite_same_shape_vectors(
    left_name: str,
    left: tf.Tensor,
    right_name: str,
    right: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    left_tensor = _finite_vector(left_name, left)
    right_tensor = _finite_vector(right_name, right)
    if left_tensor.shape != right_tensor.shape:
        raise ValueError(f"{left_name}/{right_name}: {HighDimStatus.INVALID_SHAPE.value}")
    return left_tensor, right_tensor
