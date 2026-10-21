from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

EVIDENCE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVIDENCE_ROOT.parent.parent
ALLOWED_ARTIFACT_ROOT = EVIDENCE_ROOT

CANONICAL_DIAGNOSTIC_IDS = (
    "linear_gaussian_recovery",
    "synthetic_affine_flow",
    "pfpf_algebra_parity",
    "soft_resampling_bias",
    "sinkhorn_residual",
    "learned_map_residual",
    "hmc_value_gradient",
    "posterior_sensitivity_summary",
)

PHASE_DIAGNOSTIC_OWNERSHIP = {
    "IE3": ("linear_gaussian_recovery",),
    "IE4": ("synthetic_affine_flow", "pfpf_algebra_parity"),
    "IE5": ("soft_resampling_bias", "sinkhorn_residual"),
    "IE6": ("learned_map_residual",),
    "IE7": ("hmc_value_gradient",),
    "IE8": ("posterior_sensitivity_summary",),
}

SOURCE_SUPPORT_ORDER = {
    "source_gap": 0,
    "bibliography_spine_only": 1,
    "reviewed_local_summary": 2,
    "local_derivation_only": 3,
    "not_source_dependent": 4,
}

IMPORT_BOUNDARY_PREFIXES = (
    "bayesfilter",
    "experiments.student_dpf_baselines",
    "experiments.controlled_dpf_baseline",
)

FORBIDDEN_PATH_SNIPPETS = (
    "student_dpf_baselines",
    "controlled_dpf_baseline",
    "/bayesfilter/",
)

CANONICAL_CAP_KEYS = (
    "max_particles",
    "max_time_steps",
    "max_sinkhorn_iterations",
    "max_finite_difference_evaluations",
    "max_replications",
    "max_wall_clock_seconds",
)

REQUIRED_RUN_MANIFEST_KEYS = (
    "command",
    "branch",
    "commit",
    "dirty_state_summary",
    "python_version",
    "package_versions",
    "cpu_only",
    "cuda_visible_devices",
    "gpu_devices_visible",
    "gpu_hidden_before_import",
    "pre_import_cuda_visible_devices",
    "pre_import_gpu_hiding_assertion",
    "seed_policy",
    "wall_clock_cap_seconds",
    "started_at_utc",
    "ended_at_utc",
    "artifact_paths",
)

REQUIRED_RESULT_KEYS = (
    "phase_id",
    "diagnostic_id",
    "chapter_label",
    "diagnostic_role",
    "comparator_id",
    "source_family",
    "source_support_class",
    "row_level_source_support_class",
    "seed_policy",
    "status",
    "coverage",
    "tolerance",
    "finite_checks",
    "shape_checks",
    "runtime_seconds",
    "blocker_class",
    "non_implication",
    "promotion_criterion_status",
    "promotion_veto_status",
    "continuation_veto_status",
    "repair_trigger",
    "explanatory_only_diagnostics",
    "environment",
    "command",
    "wall_time_seconds",
    "wall_clock_cap_seconds",
    "artifact_paths",
    "cpu_gpu_mode",
    "uncertainty_status",
    "replication_count",
    "mcse_or_interval",
    "post_run_red_team_note",
    "cap_non_applicability_reasons",
    "cap_values",
    "run_manifest",
)


class CoverageStatus(str, Enum):
    MISSING = "missing"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    PASSED = "passed"


class ResultStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class SourceSupportClass(str, Enum):
    SOURCE_GAP = "source_gap"
    BIBLIOGRAPHY_SPINE_ONLY = "bibliography_spine_only"
    REVIEWED_LOCAL_SUMMARY = "reviewed_local_summary"
    LOCAL_DERIVATION_ONLY = "local_derivation_only"
    NOT_SOURCE_DEPENDENT = "not_source_dependent"


class DiagnosticRole(str, Enum):
    PROMOTION_CRITERION = "promotion_criterion"
    PROMOTION_VETO = "promotion_veto"
    CONTINUATION_VETO = "continuation_veto"
    REPAIR_TRIGGER = "repair_trigger"
    EXPLANATORY = "explanatory"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_TRIGGERED = "not_triggered"
    N_A = "not_applicable"


class BlockerClass(str, Enum):
    NONE = "none"
    SOURCE = "source"
    IMPORT_BOUNDARY = "import_boundary"
    ENVIRONMENT = "environment"
    SCHEMA = "schema"
    EXECUTION = "execution"


@dataclass(frozen=True)
class FixtureContract:
    fixture_id: str
    description: str
    expected_diagnostic_ids: tuple[str, ...]


FIXTURE_CONTRACTS = (
    FixtureContract(
        fixture_id="linear_gaussian_recovery_fixture",
        description="Small linear-Gaussian state-space fixture for PF and EDH recovery checks.",
        expected_diagnostic_ids=("linear_gaussian_recovery",),
    ),
    FixtureContract(
        fixture_id="synthetic_affine_flow_fixture",
        description="Synthetic affine-flow fixture for PF-PF density and log-det parity checks.",
        expected_diagnostic_ids=("synthetic_affine_flow", "pfpf_algebra_parity"),
    ),
    FixtureContract(
        fixture_id="soft_resampling_bias_fixture",
        description="Two-particle soft-resampling fixture for bias diagnostics.",
        expected_diagnostic_ids=("soft_resampling_bias",),
    ),
    FixtureContract(
        fixture_id="sinkhorn_residual_fixture",
        description="Small Sinkhorn transport fixture with bounded iteration caps.",
        expected_diagnostic_ids=("sinkhorn_residual",),
    ),
    FixtureContract(
        fixture_id="learned_map_residual_fixture",
        description="Teacher/student residual fixture for learned transport diagnostics.",
        expected_diagnostic_ids=("learned_map_residual",),
    ),
    FixtureContract(
        fixture_id="fixed_scalar_hmc_target_fixture",
        description="Fixed scalar target fixture for value-gradient and compiled-repeatability checks.",
        expected_diagnostic_ids=("hmc_value_gradient", "posterior_sensitivity_summary"),
    ),
)


def _ensure_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _ensure_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _ensure_enum(value: str, enum_cls: type[Enum], field_name: str) -> None:
    try:
        enum_cls(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_cls)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _validate_artifact_paths(paths: list[Any], field_name: str) -> None:
    for path_value in paths:
        if not isinstance(path_value, str):
            raise ValueError(f"{field_name} entries must be strings")
        resolved = (REPO_ROOT / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value).resolve()
        try:
            resolved.relative_to(ALLOWED_ARTIFACT_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} entry {path_value!r} is outside experiments/dpf_monograph_evidence"
            ) from exc
        for snippet in FORBIDDEN_PATH_SNIPPETS:
            if snippet in str(resolved):
                raise ValueError(f"{field_name} entry {path_value!r} points into a forbidden tree")


def _validate_coverage(coverage: dict[str, Any]) -> None:
    missing_ids = [diagnostic_id for diagnostic_id in CANONICAL_DIAGNOSTIC_IDS if diagnostic_id not in coverage]
    extra_ids = [diagnostic_id for diagnostic_id in coverage if diagnostic_id not in CANONICAL_DIAGNOSTIC_IDS]
    if missing_ids:
        raise ValueError(f"coverage is missing canonical diagnostic ids: {missing_ids}")
    if extra_ids:
        raise ValueError(f"coverage has unknown diagnostic ids: {extra_ids}")
    for diagnostic_id, status in coverage.items():
        _ensure_enum(status, CoverageStatus, f"coverage[{diagnostic_id}]")


def _validate_cap_values(cap_values: dict[str, Any], reasons: dict[str, Any]) -> None:
    extra_keys = [key for key in cap_values if key not in CANONICAL_CAP_KEYS]
    missing_keys = [key for key in CANONICAL_CAP_KEYS if key not in cap_values]
    if missing_keys:
        raise ValueError(f"cap_values is missing canonical keys: {missing_keys}")
    if extra_keys:
        raise ValueError(f"cap_values has unknown keys: {extra_keys}")
    if "max_wall_clock_seconds" not in cap_values or cap_values["max_wall_clock_seconds"] is None:
        raise ValueError("cap_values.max_wall_clock_seconds is required")
    for key, value in cap_values.items():
        if value == "N/A":
            raise ValueError(f"cap_values.{key} must use null instead of 'N/A'")
        if key == "max_wall_clock_seconds" and not isinstance(value, int):
            raise ValueError("cap_values.max_wall_clock_seconds must be an integer")
        if key != "max_wall_clock_seconds" and value is None and key not in reasons:
            raise ValueError(
                f"cap_non_applicability_reasons must explain why {key} is null"
            )


def _validate_source_support(result: dict[str, Any]) -> None:
    row_level = result["row_level_source_support_class"]
    source_level = result["source_support_class"]
    _ensure_enum(row_level, SourceSupportClass, "row_level_source_support_class")
    _ensure_enum(source_level, SourceSupportClass, "source_support_class")
    if SOURCE_SUPPORT_ORDER[row_level] != SOURCE_SUPPORT_ORDER[source_level]:
        raise ValueError(
            "row_level_source_support_class must match source_support_class for row-level reporting"
        )


def _validate_phase_ownership(phase_id: str, diagnostic_id: str) -> None:
    allowed = PHASE_DIAGNOSTIC_OWNERSHIP.get(phase_id)
    if allowed is None:
        raise ValueError(f"phase_id must be one of: {', '.join(PHASE_DIAGNOSTIC_OWNERSHIP)}")
    if diagnostic_id not in allowed:
        raise ValueError(f"{diagnostic_id} is not owned by phase {phase_id}")


def _validate_run_manifest(manifest: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_RUN_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"run_manifest is missing required keys: {missing}")
    if manifest["cpu_only"] is not True:
        raise ValueError("run_manifest.cpu_only must be true for IE2 clean-room runs")
    if manifest["cuda_visible_devices"] != "-1":
        raise ValueError("run_manifest.cuda_visible_devices must be '-1'")
    if manifest["gpu_hidden_before_import"] is not True:
        raise ValueError("run_manifest.gpu_hidden_before_import must be true")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise ValueError("run_manifest.pre_import_cuda_visible_devices must be '-1'")
    if manifest["pre_import_gpu_hiding_assertion"] is not True:
        raise ValueError("run_manifest.pre_import_gpu_hiding_assertion must be true")
    package_versions = _ensure_mapping(manifest["package_versions"], "run_manifest.package_versions")
    if not package_versions:
        raise ValueError("run_manifest.package_versions must not be empty")
    artifact_paths = _ensure_list(manifest["artifact_paths"], "run_manifest.artifact_paths")
    _validate_artifact_paths(artifact_paths, "run_manifest.artifact_paths")


def _validate_import_boundary_manifest(environment: dict[str, Any]) -> None:
    scanned_modules = _ensure_list(environment.get("import_boundary_checked_modules", []), "environment.import_boundary_checked_modules")
    for module_name in scanned_modules:
        if not isinstance(module_name, str):
            raise ValueError("environment.import_boundary_checked_modules entries must be strings")
        if module_name.startswith(IMPORT_BOUNDARY_PREFIXES):
            raise ValueError(f"forbidden import recorded in boundary scan: {module_name}")


def validate_result_record(result: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise ValueError(f"result is missing required keys: {missing}")

    diagnostic_id = result["diagnostic_id"]
    if diagnostic_id not in CANONICAL_DIAGNOSTIC_IDS:
        raise ValueError(f"diagnostic_id must be one of: {list(CANONICAL_DIAGNOSTIC_IDS)}")
    _validate_phase_ownership(result["phase_id"], diagnostic_id)
    if result["source_family"] != "not_source_dependent":
        _ensure_enum(result["source_support_class"], SourceSupportClass, "source_support_class")
        if result["source_support_class"] == SourceSupportClass.REVIEWED_LOCAL_SUMMARY.value:
            raise ValueError(
                "reviewed_local_summary requires an IE1 successor artifact and is not allowed in the IE2 clean-room skeleton"
            )
    _ensure_enum(result["status"], ResultStatus, "status")
    _ensure_enum(result["diagnostic_role"], DiagnosticRole, "diagnostic_role")
    _ensure_enum(result["promotion_criterion_status"], GateStatus, "promotion_criterion_status")
    _ensure_enum(result["promotion_veto_status"], GateStatus, "promotion_veto_status")
    _ensure_enum(result["continuation_veto_status"], GateStatus, "continuation_veto_status")
    _ensure_enum(result["blocker_class"], BlockerClass, "blocker_class")
    _ensure_enum(result["uncertainty_status"], GateStatus, "uncertainty_status")

    if not str(result["non_implication"]).strip():
        raise ValueError("non_implication must be non-empty")

    coverage = _ensure_mapping(result["coverage"], "coverage")
    _validate_coverage(coverage)

    cap_values = _ensure_mapping(result["cap_values"], "cap_values")
    cap_reasons = _ensure_mapping(result["cap_non_applicability_reasons"], "cap_non_applicability_reasons")
    _validate_cap_values(cap_values, cap_reasons)

    environment = _ensure_mapping(result["environment"], "environment")
    _validate_import_boundary_manifest(environment)
    if environment.get("cpu_only") is not True:
        raise ValueError("environment.cpu_only must be true")
    if environment.get("gpu_hidden_before_import") is not True:
        raise ValueError("environment.gpu_hidden_before_import must be true")

    artifact_paths = _ensure_list(result["artifact_paths"], "artifact_paths")
    _validate_artifact_paths(artifact_paths, "artifact_paths")

    cpu_gpu_mode = _ensure_mapping(result["cpu_gpu_mode"], "cpu_gpu_mode")
    if cpu_gpu_mode.get("cpu_only") is not True:
        raise ValueError("cpu_gpu_mode.cpu_only must be true")
    if cpu_gpu_mode.get("cuda_visible_devices") != "-1":
        raise ValueError("cpu_gpu_mode.cuda_visible_devices must be '-1'")
    if cpu_gpu_mode.get("gpu_hidden_before_import") is not True:
        raise ValueError("cpu_gpu_mode.gpu_hidden_before_import must be true")

    _validate_source_support(result)

    run_manifest = _ensure_mapping(result["run_manifest"], "run_manifest")
    _validate_run_manifest(run_manifest)

    explanatory = _ensure_list(result["explanatory_only_diagnostics"], "explanatory_only_diagnostics")
    for item in explanatory:
        if not isinstance(item, str):
            raise ValueError("explanatory_only_diagnostics entries must be strings")

    finite_checks = _ensure_mapping(result["finite_checks"], "finite_checks")
    shape_checks = _ensure_mapping(result["shape_checks"], "shape_checks")
    tolerance = _ensure_mapping(result["tolerance"], "tolerance")
    mcse_or_interval = _ensure_mapping(result["mcse_or_interval"], "mcse_or_interval")
    for field_name, field_value in {
        "finite_checks": finite_checks,
        "shape_checks": shape_checks,
        "tolerance": tolerance,
        "mcse_or_interval": mcse_or_interval,
    }.items():
        if not field_value:
            raise ValueError(f"{field_name} must not be empty")


def load_result_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return data


def validate_result_file(path: Path) -> dict[str, Any]:
    result = load_result_file(path)
    validate_result_record(result)
    return result
