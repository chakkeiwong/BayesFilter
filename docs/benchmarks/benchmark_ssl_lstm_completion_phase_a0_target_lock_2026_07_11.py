"""Build and strictly verify the Phase A0 scalar SSL-LSTM target lock.

This is a CPU-hidden, non-XLA reference/artifact harness. It does not run HMC,
train NeuTra, forecast, benchmark performance, or provide scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md"
)
TARGET_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a0_target_lock.v1"
MANIFEST_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a0_dependency_manifest.v1"
CHECKPOINT_STAGE = "opening_warmup_complete_closing_and_handoff_rehash_match"
EXPECTED_INTERPRETER = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
EXPECTED_PYTHON = "3.13.13"
EXPECTED_PACKAGES = {
    "tensorflow": "2.20.0",
    "tensorflow_probability_distribution": "0.25.0",
    "numpy": "2.1.3",
}
ARTIFACT_ROOT = ROOT / "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0"
EXPECTED_MANIFEST_PATH = ARTIFACT_ROOT / "dependency-manifest.json"
EXPECTED_DISCOVERY_LOG_PATH = ARTIFACT_ROOT / "dependency-discovery.log"
EXPECTED_LOCK_PATH = ARTIFACT_ROOT / "target-lock.json"
EXPECTED_LOCK_LOG_PATH = ARTIFACT_ROOT / "target-lock.log"
EXPECTED_GENERATION_COMMAND = (
    "CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 "
    "TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 "
    "TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 "
    f"{EXPECTED_INTERPRETER} {HARNESS_PATH} --dependency-manifest "
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json "
    "--output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json "
    "--log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.log"
)
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "TF_DETERMINISTIC_OPS",
    "TF_ENABLE_ONEDNN_OPTS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "OMP_NUM_THREADS",
    "TF_CPP_MIN_LOG_LEVEL",
)
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "PYTHONHASHSEED": "0",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "1",
}
FREE_NAMES = (
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
)
FREE_INDICES = (12, 13, 14, 15)
TRUTH_FREE = (0.35, -0.08, 0.65, 0.05)
FULL_FIXTURE = (
    0.09, -0.07, 0.05, 0.04, 0.03, -0.02, 0.06, -0.05,
    0.01, 0.04, -0.03, 0.02, 0.35, -0.08, 0.65, 0.05,
    0.15, -0.10, 0.20, -0.35, 0.15, 0.55, 0.35, -0.15,
)
PARAMETER_CHART = (
    "lstm_input.input.0.0",
    "lstm_input.forget.0.0",
    "lstm_input.output.0.0",
    "lstm_input.candidate.0.0",
    "lstm_recurrent.input.0.0",
    "lstm_recurrent.forget.0.0",
    "lstm_recurrent.output.0.0",
    "lstm_recurrent.candidate.0.0",
    "lstm_bias.input.0",
    "lstm_bias.forget.0",
    "lstm_bias.output.0",
    "lstm_bias.candidate.0",
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
    "initial_mean.0",
    "initial_mean.1",
    "initial_mean.2",
    "initial_std_unconstrained.0",
    "initial_std_unconstrained.1",
    "initial_std_unconstrained.2",
    "process_std_unconstrained.0",
    "observation_std_unconstrained.0",
)
STATIC_CONFIG = {
    "horizon": 30,
    "latent_dim": 1,
    "hidden_dim": 1,
    "observation_dim": 1,
    "augmented_state_dim": 3,
    "covariance_mode": "diagonal",
    "full_parameter_dim": 24,
}
LIKELIHOOD_CONFIG = {
    "name": "svd_ukf_filtering_log_likelihood",
    "score_helper": "tf_ssl_lstm_svd_ukf_score",
    "std_floor": 1.0e-4,
    "alpha": 1.0,
    "beta": 2.0,
    "kappa": 0.0,
    "placement_floor": 0.0,
    "innovation_floor": 1.0e-12,
    "rank_tolerance": 1.0e-12,
    "spectral_gap_tolerance": 1.0e-10,
    "fixed_null_tolerance": 1.0e-10,
    "jitter": 0.0,
    "allow_fixed_null_support": False,
}
HISTORICAL_EXECUTION_MODE = {
    "device": "CPU",
    "cpu_hidden": True,
    "jit_compile": False,
    "xla": False,
    "dtype": "float64",
    "role": "cpu_hidden_non_xla_reference_replay_only",
}
EXPECTED_NONCLAIMS = [
    "not a production implementation",
    "not posterior correctness evidence",
    "not HMC or NeuTra readiness evidence",
    "not predictive equivalence or calibration evidence",
    "not GPU/XLA or default readiness evidence",
    "not a sampler ranking or scientific claim",
]
CRITICAL_ROOTS = {
    "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py":
        "historical_target_constructor",
    "bayesfilter/nonlinear/ssl_lstm_protocol.py": "parameter_chart_and_static_config",
    "bayesfilter/nonlinear/ssl_lstm_zhaocui_hmc_minimal.py": "scalar_fixture",
    "bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py": "ssl_lstm_svd_ukf_adapter",
    "bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py": "analytic_svd_ukf_score",
    "bayesfilter/nonlinear/sigma_points_tf.py": "svd_ukf_filter",
    "bayesfilter/structural.py": "structural_model_protocol",
    "bayesfilter/structural_tf.py": "tensorflow_structural_model",
    "bayesfilter/results_tf.py": "tensorflow_result_contract",
}
HISTORICAL_INPUTS = {
    "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py":
        "historical_target_source",
    "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py":
        "historical_sampler_geometry_source",
    "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json":
        "historical_target_artifact",
    "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json":
        "historical_sampler_geometry_artifact",
    "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json":
        "historical_hmc_diagnostic_context",
    "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json":
        "historical_reference_context",
    "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json":
        "historical_reference_context",
    "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json":
        "historical_reference_context",
    "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json":
        "historical_reference_context",
}
GOVERNANCE_INPUTS = {
    "docs/chapters/ch28a_neural_network_state_space_model_applications.tex":
        "model_and_validation_design",
    "docs/plans/bayesfilter-scalar-filtering-hmc-validation-reset-memo-2026-07-10.md":
        "reset_memo",
    "docs/plans/bayesfilter-scalar-ssl-lstm-predictive-equivalence-master-program-2026-07-11.md":
        "predictive_equivalence_program",
    PLAN_PATH: "phase_a0_contract",
    "docs/plans/bayesfilter-ssl-lstm-completion-roadmap-2026-07-11.md":
        "completion_roadmap",
}
DSGE_GOVERNANCE = {
    "/home/ubuntu/python/dsge_hmc/AGENTS.md": "external_neutra_governance",
    "/home/ubuntu/python/dsge_hmc/CLAUDE.md": "external_neutra_governance",
}
DSGE_CONTEXT = {
    "/home/ubuntu/python/dsge_hmc/docs/plans/neutra-gate1-real-linux-gpu-german-budget-result-2026-05-06.md":
        "external_gate1_closure_context",
    "/home/ubuntu/python/dsge_hmc/docs/plans/neutra-gate3-surrogate-hmc-reset-memo-clean-2026-05-16.md":
        "external_neutra_current_context",
}
TOP_KEYS = {
    "schema_version", "created_at_utc", "artifact_role", "classification",
    "run_manifest", "immutable_attempt_fingerprint", "source_provenance",
    "target_semantics", "implementation_execution", "sampler_geometry",
    "forecast_design", "historical_artifact_disposition", "probe_results",
    "signatures", "nonclaims",
}


class ContractError(RuntimeError):
    """Raised when an A0 artifact violates the reviewed contract."""


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ContractError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(actual == expected, f"{label} keys mismatch: {sorted(actual ^ expected)}")


def parse_utc_timestamp(value: Any, label: str) -> datetime:
    require(isinstance(value, str), f"{label} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ContractError(f"invalid {label}: {value}") from error
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{label} lacks timezone")
    return parsed.astimezone(UTC)


def repo_path(relative: str) -> Path:
    require(relative != "" and not Path(relative).is_absolute(), f"invalid repo path: {relative}")
    resolved = (ROOT / relative).resolve()
    require(resolved.is_relative_to(ROOT), f"path escapes repository: {relative}")
    require(resolved.is_file(), f"required file is missing: {relative}")
    return resolved


def git_status(relative: str) -> str:
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", relative],
        cwd=ROOT,
        text=True,
    ).splitlines()
    require(len(output) <= 1, f"ambiguous Git status for {relative}")
    if not output:
        return "  "
    row = output[0]
    require(len(row) >= 4, f"invalid Git status row for {relative}")
    require(row[3:] == relative, f"Git status path mismatch for {relative}: {row}")
    status = row[:2]
    allowed = set(" MADRCU?!T")
    require(status == "??" or all(char in allowed for char in status), f"invalid XY status: {status}")
    return status


def descriptor(relative: str, role: str) -> dict[str, Any]:
    path = repo_path(relative)
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "git_status": git_status(relative),
        "role": role,
    }


def descriptors(items: Mapping[str, str]) -> list[dict[str, Any]]:
    return [descriptor(path, items[path]) for path in sorted(items)]


def external_descriptors(items: Mapping[str, str]) -> list[dict[str, Any]]:
    result = []
    for absolute in sorted(items):
        path = Path(absolute)
        require(path.is_absolute() and path.is_file(), f"missing external file: {absolute}")
        result.append({"absolute_path": absolute, "sha256": file_sha256(path), "role": items[absolute]})
    return result


def current_environment() -> dict[str, str]:
    result = {key: os.environ.get(key, "<unset>") for key in ENVIRONMENT_KEYS}
    require(result == EXPECTED_ENVIRONMENT, f"environment mismatch: {result}")
    return result


def package_versions() -> dict[str, str]:
    values = {
        "tensorflow": importlib.metadata.version("tensorflow"),
        "tensorflow_probability_distribution": importlib.metadata.version("tfp-nightly"),
        "numpy": importlib.metadata.version("numpy"),
    }
    require(values == EXPECTED_PACKAGES, f"package version mismatch: {values}")
    return values


def runtime_contract() -> tuple[str, str, dict[str, str], dict[str, str]]:
    interpreter = os.path.abspath(sys.executable)
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    require(interpreter == EXPECTED_INTERPRETER, f"interpreter mismatch: {interpreter}")
    require(python_version == EXPECTED_PYTHON, f"Python mismatch: {python_version}")
    return interpreter, python_version, package_versions(), current_environment()


def module_paths() -> list[str]:
    paths: set[str] = set()
    for module in tuple(sys.modules.values()):
        file_name = getattr(module, "__file__", None)
        if not file_name:
            continue
        try:
            resolved = Path(file_name).resolve()
        except (OSError, RuntimeError):
            continue
        if resolved.suffix in {".pyc", ".pyo"}:
            source = Path(importlib.util.source_from_cache(str(resolved)))
            if source.is_file():
                resolved = source.resolve()
        if resolved.is_file() and resolved.is_relative_to(ROOT):
            paths.add(resolved.relative_to(ROOT).as_posix())
    paths.discard(HARNESS_PATH)
    return sorted(paths)


def tensor_descriptor(name: str, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value, dtype="<f8")
    if array.ndim > 0:
        array = np.ascontiguousarray(array)
    require(array.ndim <= 2, f"tensor rank above two: {name}")
    require(bool(np.all(np.isfinite(array))), f"non-finite tensor: {name}")
    if array.ndim == 0:
        values: Any = float(array)
    else:
        values = array.tolist()
    return {
        "name": name,
        "dtype": "float64",
        "shape": list(array.shape),
        "order": "C",
        "byte_order": "little",
        "values": values,
        "raw_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def load_historical_module() -> ModuleType:
    path = repo_path("docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py")
    spec = importlib.util.spec_from_file_location("_ssl_lstm_phase_a0_historical", path)
    require(spec is not None and spec.loader is not None, "cannot load historical target module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_probe(target: Any, free_value: Any, name: str) -> dict[str, Any]:
    import numpy as np
    import tensorflow as tf

    free = tf.reshape(tf.convert_to_tensor(free_value, dtype=tf.float64), [-1])
    full = target.full_theta(free)
    score_helper = target._value_and_score_impl.__func__.__globals__["tf_ssl_lstm_svd_ukf_score"]
    result, _ = score_helper(
        target.observations,
        full,
        target.config,
        evidence_path=PLAN_PATH,
        std_floor=1.0e-4,
        alpha=1.0,
        beta=2.0,
        kappa=0.0,
        spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
    )
    likelihood_value = np.asarray(result.log_likelihood.numpy(), dtype=np.float64)
    full_score = np.asarray(tf.reshape(result.score, [-1]).numpy(), dtype=np.float64)
    likelihood_score = full_score[list(FREE_INDICES)]
    free_np = np.asarray(free.numpy(), dtype=np.float64)
    truth = np.asarray(TRUTH_FREE, dtype=np.float64)
    prior_score = -(free_np - truth) / 16.0
    prior_value = np.asarray(-0.5 * np.sum(np.square(free_np - truth) / 16.0), dtype=np.float64)
    total_value = np.asarray(likelihood_value + prior_value, dtype=np.float64)
    total_score = np.asarray(likelihood_score + prior_score, dtype=np.float64)
    value_residual = abs(float(total_value) - (float(likelihood_value) + float(prior_value)))
    score_residual = float(np.max(np.abs(total_score - (likelihood_score + prior_score))))
    eps64 = 2.0**-52
    value_tolerance = 8.0 * eps64 * max(
        1.0, abs(float(total_value)), abs(float(likelihood_value)) + abs(float(prior_value))
    )
    score_tolerance = 8.0 * eps64 * max(
        1.0,
        float(np.max(np.abs(total_score))),
        float(np.max(np.abs(likelihood_score))) + float(np.max(np.abs(prior_score))),
    )
    passed = bool(
        np.all(np.isfinite(likelihood_score))
        and np.all(np.isfinite(total_score))
        and math.isfinite(float(total_value))
        and value_residual <= value_tolerance
        and score_residual <= score_tolerance
    )
    require(passed, f"probe decomposition failed: {name}")
    return {
        "name": name,
        "free_position": tensor_descriptor(f"{name}.free_position", free_np),
        "likelihood_value": tensor_descriptor(f"{name}.likelihood_value", likelihood_value),
        "likelihood_score": tensor_descriptor(f"{name}.likelihood_score", likelihood_score),
        "prior_value": tensor_descriptor(f"{name}.prior_value", prior_value),
        "prior_score": tensor_descriptor(f"{name}.prior_score", prior_score),
        "total_value": tensor_descriptor(f"{name}.total_value", total_value),
        "total_score": tensor_descriptor(f"{name}.total_score", total_score),
        "value_residual": tensor_descriptor(f"{name}.value_residual", value_residual),
        "score_residual_inf": tensor_descriptor(f"{name}.score_residual_inf", score_residual),
        "value_tolerance": tensor_descriptor(f"{name}.value_tolerance", value_tolerance),
        "score_tolerance": tensor_descriptor(f"{name}.score_tolerance", score_tolerance),
        "passed": passed,
    }


def geometry_payload(phase2s: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    handoff = phase2s["map_local_handoff"]
    coordinate_names = phase2s["precondition"]["coordinate_contract"]["free_parameter_names"]
    require(tuple(coordinate_names) == FREE_NAMES, "Phase 2S coordinate names mismatch")
    center = np.asarray(handoff["center_free_parameter_values"], dtype=np.float64)
    scale = np.asarray(handoff["scale"], dtype=np.float64)
    factor = np.asarray(handoff["factor_z"], dtype=np.float64)
    covariance_z = np.asarray(handoff["covariance_z"], dtype=np.float64)
    precision_z = np.asarray(handoff["precision_z"], dtype=np.float64)
    covariance_theta = np.asarray(handoff["covariance_theta"], dtype=np.float64)
    precision_theta = np.asarray(handoff["precision_theta"], dtype=np.float64)
    for name, value, shape in (
        ("center_free", center, (4,)), ("scale", scale, (4,)),
        ("factor_z", factor, (4, 4)), ("covariance_z", covariance_z, (4, 4)),
        ("precision_z", precision_z, (4, 4)),
        ("covariance_theta", covariance_theta, (4, 4)),
        ("precision_theta", precision_theta, (4, 4)),
    ):
        require(value.shape == shape and np.all(np.isfinite(value)), f"invalid geometry {name}")
    eps64 = 2.0**-52
    norm_inf = lambda value: float(np.linalg.norm(value, ord=np.inf))
    factor_residual = norm_inf(factor @ factor.T - covariance_z)
    factor_tolerance = 64.0 * eps64 * max(
        1.0, norm_inf(factor) * norm_inf(factor.T), norm_inf(covariance_z)
    )
    diagonal = np.diag(scale)
    inverse_diagonal = np.diag(1.0 / scale)
    raw_precision_theta = (inverse_diagonal @ precision_z) @ inverse_diagonal
    raw_precision_theta = 0.5 * (raw_precision_theta + raw_precision_theta.T)
    raw_covariance_theta = (diagonal @ covariance_z) @ diagonal
    expected_raw_covariance_theta = np.linalg.inv(raw_precision_theta)
    expected_raw_covariance_theta = 0.5 * (
        expected_raw_covariance_theta + expected_raw_covariance_theta.T
    )
    raw_covariance_residual = norm_inf(
        raw_covariance_theta - expected_raw_covariance_theta
    )
    raw_covariance_tolerance = 64.0 * eps64 * max(
        1.0, norm_inf(raw_covariance_theta), norm_inf(expected_raw_covariance_theta)
    )
    mass = phase2s["settings"]["mass"]
    require(
        mass == {"dense": True, "eigenvalue_floor": 1.0e-4, "jitter": 1.0e-9,
                 "max_condition_number": 1.0e5},
        "Phase 2S mass metadata mismatch",
    )
    report = phase2s["initializer"]["mass_matrix"]["regularization_report"]
    jittered = raw_precision_theta + float(mass["jitter"]) * np.eye(4)
    eigenvalues, eigenvectors = np.linalg.eigh(jittered)
    effective_floor = max(
        float(mass["eigenvalue_floor"]),
        float(np.max(eigenvalues)) / float(mass["max_condition_number"]),
    )
    regularized_eigenvalues = np.maximum(eigenvalues, effective_floor)
    expected_precision_theta = (
        eigenvectors * regularized_eigenvalues
    ) @ eigenvectors.T
    expected_precision_theta = 0.5 * (
        expected_precision_theta + expected_precision_theta.T
    )
    clipped_count = int(np.sum(regularized_eigenvalues > eigenvalues))
    require(report["method"] == "symmetric_eigendecomposition_floor", "regularization method mismatch")
    require(report["jitter"] == float(mass["jitter"]), "regularization jitter mismatch")
    require(report["requested_eigenvalue_floor"] == float(mass["eigenvalue_floor"]), "requested floor mismatch")
    require(report["max_condition_number"] == float(mass["max_condition_number"]), "condition cap mismatch")
    require(report["effective_eigenvalue_floor"] == effective_floor, "effective floor mismatch")
    require(report["clipped_eigenvalue_count"] == clipped_count, "clipped count mismatch")
    require(report["input_asymmetric"] is False, "unexpected asymmetric precision input")
    require(report["diagonal_fallback_used"] is False, "unexpected diagonal fallback")
    regularized_precision_residual = norm_inf(
        expected_precision_theta - precision_theta
    )
    regularized_precision_tolerance = 64.0 * eps64 * max(
        1.0, norm_inf(expected_precision_theta), norm_inf(precision_theta)
    )
    expected_covariance_theta = np.linalg.inv(expected_precision_theta)
    expected_covariance_theta = 0.5 * (
        expected_covariance_theta + expected_covariance_theta.T
    )
    regularized_covariance_residual = norm_inf(
        expected_covariance_theta - covariance_theta
    )
    regularized_covariance_tolerance = 64.0 * eps64 * max(
        1.0, norm_inf(expected_covariance_theta), norm_inf(covariance_theta)
    )
    raw_to_stored_covariance_residual = norm_inf(
        raw_covariance_theta - covariance_theta
    )
    raw_to_stored_precision_residual = norm_inf(
        raw_precision_theta - precision_theta
    )
    z_inverse_residual = norm_inf(precision_z @ covariance_z - np.eye(4))
    z_inverse_tolerance = 64.0 * eps64 * max(
        1.0, norm_inf(precision_z) * norm_inf(covariance_z)
    )
    theta_inverse_residual = norm_inf(precision_theta @ covariance_theta - np.eye(4))
    theta_inverse_tolerance = 64.0 * eps64 * max(
        1.0, norm_inf(precision_theta) * norm_inf(covariance_theta)
    )
    lower = bool(np.max(np.abs(np.triu(factor, 1))) == 0.0)
    positive = bool(np.all(np.diag(factor) > 0.0))
    passed = bool(
        lower and positive
        and factor_residual <= factor_tolerance
        and raw_covariance_residual <= raw_covariance_tolerance
        and regularized_precision_residual <= regularized_precision_tolerance
        and regularized_covariance_residual <= regularized_covariance_tolerance
        and z_inverse_residual <= z_inverse_tolerance
        and theta_inverse_residual <= theta_inverse_tolerance
    )
    require(passed, "Phase 2S geometry identities failed")
    source = "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
    source_script = "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py"
    return {
        "role": "historical_sampler_initialization_and_tuning_context_only",
        "source_path": source,
        "source_sha256": file_sha256(repo_path(source)),
        "source_script_path": source_script,
        "source_script_sha256": file_sha256(repo_path(source_script)),
        "coordinate_names": list(coordinate_names),
        "coordinate_formula": "free = center_free + scale * (factor_z @ u)",
        "center_free": tensor_descriptor("sampler_geometry.center_free", center),
        "scale": tensor_descriptor("sampler_geometry.scale", scale),
        "factor_z": tensor_descriptor("sampler_geometry.factor_z", factor),
        "covariance_z": tensor_descriptor("sampler_geometry.covariance_z", covariance_z),
        "precision_z": tensor_descriptor("sampler_geometry.precision_z", precision_z),
        "covariance_theta": tensor_descriptor("sampler_geometry.covariance_theta", covariance_theta),
        "precision_theta": tensor_descriptor("sampler_geometry.precision_theta", precision_theta),
        "reconstruction_tolerance_formula":
            "eps64=2**-52;norm_inf=max_abs_row_sum;matmul=left_to_right_numpy_float64;source_regularize_precision=symmetrize_plus_jitter_then_eigh_floor_cap;tol=64*eps64*max(1,lhs_norm,rhs_norm)",
        "regularization": {
            "dense_mass": True,
            "jitter": 1.0e-9,
            "eigenvalue_floor": 1.0e-4,
            "max_condition_number": 1.0e5,
        },
        "checks": {
            "factor_lower_triangular": lower,
            "factor_positive_diagonal": positive,
            "factor_covariance_residual_inf": factor_residual,
            "factor_covariance_tolerance": factor_tolerance,
            "raw_theta_covariance_residual_inf": raw_covariance_residual,
            "raw_theta_covariance_tolerance": raw_covariance_tolerance,
            "regularized_theta_precision_residual_inf": regularized_precision_residual,
            "regularized_theta_precision_tolerance": regularized_precision_tolerance,
            "regularized_theta_covariance_residual_inf": regularized_covariance_residual,
            "regularized_theta_covariance_tolerance": regularized_covariance_tolerance,
            "raw_to_stored_theta_covariance_residual_inf": raw_to_stored_covariance_residual,
            "raw_to_stored_theta_precision_residual_inf": raw_to_stored_precision_residual,
            "z_inverse_residual_inf": z_inverse_residual,
            "z_inverse_tolerance": z_inverse_tolerance,
            "theta_inverse_residual_inf": theta_inverse_residual,
            "theta_inverse_tolerance": theta_inverse_tolerance,
            "regularization_effective_eigenvalue_floor": effective_floor,
            "regularization_clipped_eigenvalue_count": clipped_count,
            "passed": passed,
        },
        "nonclaims": [
            "not a certified global MAP",
            "not posterior covariance correctness evidence",
            "not an alternate target definition",
        ],
    }


def warmup_cycle(module: ModuleType) -> dict[str, Any]:
    import numpy as np

    target_a = module.build_filtering_geometry_target()
    target_b = module.build_filtering_geometry_target()
    obs_a = np.asarray(target_a.observations.numpy(), dtype="<f8")
    obs_b = np.asarray(target_b.observations.numpy(), dtype="<f8")
    require(obs_a.shape == (30, 1) and obs_a.tobytes() == obs_b.tobytes(), "observation replay drift")
    phase2s = strict_load(repo_path(
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
    ))
    geometry = geometry_payload(phase2s)
    probes = {
        "truth_free": evaluate_probe(target_a, TRUTH_FREE, "truth_free"),
        "phase2s_center": evaluate_probe(
            target_a,
            phase2s["map_local_handoff"]["center_free_parameter_values"],
            "phase2s_center",
        ),
    }
    return {"target_a": target_a, "target_b": target_b, "geometry": geometry, "probes": probes}


def validate_historical_target(target: Any) -> None:
    import numpy as np

    config = target.config
    require(
        (config.horizon, config.latent_dim, config.hidden_dim, config.observation_dim,
         config.augmented_state_dim, config.parameter_dim) == (30, 1, 1, 1, 3, 24),
        "historical target dimensions mismatch",
    )
    require(tuple(config.parameter_names) == PARAMETER_CHART, "parameter chart mismatch")
    require(tuple(target.free_indices) == FREE_INDICES, "free indices mismatch")
    require(tuple(target.free_parameter_names) == FREE_NAMES, "free names mismatch")
    require(np.array_equal(np.asarray(target.base_theta.numpy()), np.asarray(FULL_FIXTURE)), "fixture mismatch")
    require(np.array_equal(np.asarray(target.truth_free.numpy()), np.asarray(TRUTH_FREE)), "truth mismatch")
    historical = strict_load(repo_path(
        "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json"
    ))
    require(historical["schema_version"] == "scalar_ssl_lstm.filtering_geometry.v1", "historical schema mismatch")
    require(historical["target"]["free_parameter_indices"] == list(FREE_INDICES), "historical free indices mismatch")
    require(historical["target"]["free_parameter_names"] == list(FREE_NAMES), "historical free names mismatch")
    require(historical["target"]["truth_free_parameters"] == list(TRUTH_FREE), "historical truth mismatch")
    settings = historical["settings"]
    require(settings["horizon"] == 30 and settings["seed"] == [20260708, 2301], "historical data identity mismatch")
    require(settings["simulation_noise_scale"] == 1.0 and settings["prior_scale"] == 4.0, "historical settings mismatch")
    require(settings["spectral_gap_tolerance"] == 1.0e-10, "historical spectral tolerance mismatch")


def manifest_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": value["schema_version"],
        "harness": value["harness"],
        "critical_roots": value["critical_roots"],
        "runtime_loaded_local_dependencies": value["runtime_loaded_local_dependencies"],
        "historical_inputs": value["historical_inputs"],
        "environment": value["environment"],
        "probe_names": value["probe_names"],
        "discovery_only": value["discovery_only"],
    }


def verify_descriptor(item: Mapping[str, Any], *, role_required: bool = True) -> None:
    keys = {"path", "sha256", "git_status", "role"} if role_required else {"path", "sha256"}
    exact_keys(item, keys, "path descriptor")
    path = repo_path(str(item["path"]))
    require(item["sha256"] == file_sha256(path), f"file hash drift: {item['path']}")
    if role_required:
        require(item["git_status"] == git_status(str(item["path"])), f"Git status drift: {item['path']}")
        require(isinstance(item["role"], str) and item["role"], f"invalid role: {item['path']}")


def verify_manifest(value: Mapping[str, Any]) -> None:
    exact_keys(value, {
        "schema_version", "created_at_utc", "harness", "critical_roots",
        "runtime_loaded_local_dependencies", "historical_inputs", "environment",
        "probe_names", "discovery_only", "aggregate_sha256",
    }, "dependency manifest")
    require(value["schema_version"] == MANIFEST_SCHEMA, "dependency manifest schema mismatch")
    exact_keys(value["harness"], {"path", "sha256"}, "manifest harness")
    verify_descriptor(value["harness"], role_required=False)
    require(value["harness"]["path"] == HARNESS_PATH, "manifest harness path mismatch")
    for name in ("critical_roots", "runtime_loaded_local_dependencies", "historical_inputs"):
        rows = value[name]
        require(isinstance(rows, list), f"manifest {name} must be a list")
        require([row["path"] for row in rows] == sorted(row["path"] for row in rows), f"manifest {name} ordering")
        require(len({row["path"] for row in rows}) == len(rows), f"manifest {name} duplicates")
        for row in rows:
            verify_descriptor(row)
    require(value["critical_roots"] == descriptors(CRITICAL_ROOTS), "critical-root manifest mismatch")
    require(value["historical_inputs"] == descriptors(HISTORICAL_INPUTS), "historical-input manifest mismatch")
    require(
        all(row["role"] == "runtime_loaded_local_dependency" for row in value["runtime_loaded_local_dependencies"]),
        "runtime dependency role mismatch",
    )
    exact_keys(value["environment"], set(ENVIRONMENT_KEYS), "manifest environment")
    require(value["environment"] == current_environment(), "manifest environment drift")
    require(value["probe_names"] == ["truth_free", "phase2s_center"], "manifest probes mismatch")
    require(value["discovery_only"] is True, "manifest must be discovery only")
    require(value["aggregate_sha256"] == canonical_sha256(manifest_projection(value)), "manifest aggregate mismatch")


def discover(output: Path, log_path: Path) -> None:
    runtime_contract()
    module = load_historical_module()
    previous: list[str] | None = None
    stable: list[str] | None = None
    cycles = 0
    for cycles in range(1, 6):
        payload = warmup_cycle(module)
        validate_historical_target(payload["target_a"])
        current = module_paths()
        if previous == current:
            stable = current
            break
        previous = current
    require(stable is not None, "runtime dependency closure did not stabilize in five cycles")
    runtime_items = {path: "runtime_loaded_local_dependency" for path in stable}
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "harness": {"path": HARNESS_PATH, "sha256": file_sha256(repo_path(HARNESS_PATH))},
        "critical_roots": descriptors(CRITICAL_ROOTS),
        "runtime_loaded_local_dependencies": descriptors(runtime_items),
        "historical_inputs": descriptors(HISTORICAL_INPUTS),
        "environment": current_environment(),
        "probe_names": ["truth_free", "phase2s_center"],
        "discovery_only": True,
        "aggregate_sha256": "",
    }
    manifest["aggregate_sha256"] = canonical_sha256(manifest_projection(manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    verify_manifest(strict_load(output))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"status=dependency_discovery_passed\ncycles={cycles}\nmodules={len(stable)}\nmanifest={output}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "dependency_discovery_passed", "cycles": cycles, "module_count": len(stable), "artifact": str(output)}))


def fingerprint_members(manifest: Mapping[str, Any], manifest_path: str) -> list[dict[str, Any]]:
    roles: dict[str, set[str]] = {}
    for collection in (manifest["critical_roots"], manifest["runtime_loaded_local_dependencies"], manifest["historical_inputs"]):
        for item in collection:
            roles.setdefault(item["path"], set()).add(item["role"])
    roles.setdefault(HARNESS_PATH, set()).add("a0_lock_harness")
    roles.setdefault(manifest_path, set()).add("dependency_manifest_exact_bytes")
    return [descriptor(path, "+".join(sorted(roles[path]))) for path in sorted(roles)]


def fingerprint_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_stage": value["checkpoint_stage"],
        "dependency_manifest_path": value["dependency_manifest_path"],
        "dependency_manifest_file_sha256": value["dependency_manifest_file_sha256"],
        "dependency_manifest_aggregate_sha256": value["dependency_manifest_aggregate_sha256"],
        "members": value["members"],
    }


def make_fingerprint(manifest: Mapping[str, Any], manifest_relative: str) -> dict[str, Any]:
    value = {
        "checkpoint_stage": CHECKPOINT_STAGE,
        "dependency_manifest_path": manifest_relative,
        "dependency_manifest_file_sha256": file_sha256(repo_path(manifest_relative)),
        "dependency_manifest_aggregate_sha256": manifest["aggregate_sha256"],
        "members": fingerprint_members(manifest, manifest_relative),
        "aggregate_sha256": "",
    }
    value["aggregate_sha256"] = canonical_sha256(fingerprint_projection(value))
    return value


def forecast_design() -> dict[str, Any]:
    return {
        "status": "prospective_not_frozen_until_a4",
        "horizon": 10,
        "terminal_state": "final_svd_ukf_filtered_gaussian_per_parameter_draw",
        "transition": "structural_ssl_lstm_transition",
        "process_noise": "stochastic_latent_coordinate_only",
        "hidden_cell_completion": "deterministic",
        "observation_noise": "ssl_lstm_observation_law",
        "path_clustering": "entire_length_10_path_is_one_clustered_observation",
        "innovation_banks": [
            "shared_for_paired_mean_log_variance",
            "independent_arm_specific_for_primary_mmd_and_robustness",
        ],
        "unfrozen_fields": [
            "equivalence_margins", "feature_scales", "forecast_replication_count",
            "mmd_bandwidths", "mmd_mixture_weights", "mmd_tolerance",
            "bootstrap_type", "bootstrap_count", "bootstrap_seed", "block_length",
            "confidence_level", "covariance_ridge", "condition_number_cap",
            "sampler_seeds", "forecast_seeds",
        ],
    }


def source_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    dsge_root = Path("/home/ubuntu/python/dsge_hmc")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dsge_root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=dsge_root, text=True).strip())
    return {
        "critical_roots": descriptors(CRITICAL_ROOTS),
        "runtime_loaded_local_dependencies": manifest["runtime_loaded_local_dependencies"],
        "historical_inputs": descriptors(HISTORICAL_INPUTS),
        "governance_inputs": descriptors(GOVERNANCE_INPUTS),
        "dsge_hmc": {
            "repository_path": str(dsge_root),
            "git_commit": commit,
            "git_dirty": dirty,
            "governance_files": external_descriptors(DSGE_GOVERNANCE),
            "gate1_context_files": external_descriptors(DSGE_CONTEXT),
            "role": "external_design_provenance_only_not_bayesfilter_evidence",
        },
    }


def target_semantics(target: Any) -> dict[str, Any]:
    import numpy as np

    chart = list(target.config.parameter_names)
    fixture = np.asarray(target.base_theta.numpy(), dtype=np.float64)
    fixed = [
        {"name": chart[index], "index": index, "value": float(fixture[index])}
        for index in range(len(chart)) if index not in FREE_INDICES
    ]
    return {
        "static_config": dict(STATIC_CONFIG),
        "parameter_chart": chart,
        "full_fixture": tensor_descriptor("target_semantics.full_fixture", fixture),
        "free_mask": {
            "names": list(FREE_NAMES), "indices": list(FREE_INDICES),
            "truth_free": tensor_descriptor("target_semantics.free_mask.truth_free", TRUTH_FREE),
        },
        "fixed_parameters": fixed,
        "observations": tensor_descriptor("target_semantics.observations", target.observations.numpy()),
        "likelihood": dict(LIKELIHOOD_CONFIG),
        "prior": {
            "family": "unnormalized_isotropic_gaussian_log_kernel",
            "center": tensor_descriptor("target_semantics.prior.center", TRUTH_FREE),
            "standard_deviation": 4.0,
            "normalized": False,
            "log_kernel_formula": "-0.5 * sum((free - truth_free)^2 / 4.0^2)",
        },
        "dtype": "float64",
    }


def dispositions() -> list[dict[str, Any]]:
    mapping = {
        "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py":
            "target_construction_source_context_not_production",
        "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py":
            "untracked_sampler_geometry_context_only",
        "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json":
            "historical_target_context_replayed_not_promoted",
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json":
            "untracked_sampler_geometry_context_only",
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json":
            "diagnostic_context_forbidden_as_confirmatory_baseline",
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json":
            "failed_or_exploratory_context_not_posterior_evidence",
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json":
            "failed_or_exploratory_context_not_posterior_evidence",
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json":
            "failed_or_exploratory_context_not_posterior_evidence",
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json":
            "failed_or_exploratory_context_not_posterior_evidence",
    }
    return [
        {
            "path": path, "sha256": file_sha256(repo_path(path)),
            "git_status": git_status(path), "role": HISTORICAL_INPUTS[path],
            "disposition": mapping[path], "promoting": False,
        }
        for path in sorted(mapping)
    ]


def signatures(value: Mapping[str, Any]) -> dict[str, str]:
    schema = value["schema_version"]
    result = {
        "target_semantic_sha256": canonical_sha256({
            "schema_version": schema, "target_semantics": value["target_semantics"],
            "probe_results": value["probe_results"],
        }),
        "implementation_execution_sha256": canonical_sha256({
            "schema_version": schema, "implementation_execution": value["implementation_execution"],
        }),
        "sampler_geometry_sha256": canonical_sha256({
            "schema_version": schema, "sampler_geometry": value["sampler_geometry"],
        }),
        "forecast_design_sha256": canonical_sha256({
            "schema_version": schema, "forecast_design": value["forecast_design"],
        }),
        "aggregate_sha256": "",
    }
    result["aggregate_sha256"] = canonical_sha256({
        "schema_version": schema,
        "target_semantic_sha256": result["target_semantic_sha256"],
        "implementation_execution_sha256": result["implementation_execution_sha256"],
        "sampler_geometry_sha256": result["sampler_geometry_sha256"],
        "forecast_design_sha256": result["forecast_design_sha256"],
    })
    return result


def generate(manifest_path: Path, output: Path, log_path: Path) -> None:
    start_clock = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    interpreter, python_version, packages, environment = runtime_contract()
    manifest = strict_load(manifest_path)
    verify_manifest(manifest)
    manifest_relative = manifest_path.resolve().relative_to(ROOT).as_posix()
    module = load_historical_module()
    expected_modules = [row["path"] for row in manifest["runtime_loaded_local_dependencies"]]
    matching_cycles = 0
    for _ in range(5):
        warmup_cycle(module)
        current_modules = module_paths()
        if current_modules == expected_modules:
            matching_cycles += 1
            if matching_cycles == 2:
                break
        else:
            matching_cycles = 0
    require(matching_cycles == 2, "opening dependency closure did not match twice in five cycles")
    opening = make_fingerprint(manifest, manifest_relative)
    evidence = warmup_cycle(module)
    validate_historical_target(evidence["target_a"])
    target = evidence["target_a"]
    critical = descriptors(CRITICAL_ROOTS)
    runtime = manifest["runtime_loaded_local_dependencies"]
    harness = {"path": HARNESS_PATH, "sha256": file_sha256(repo_path(HARNESS_PATH))}
    implementation = {
        "interpreter": interpreter,
        "python_version": python_version,
        "packages": packages,
        "environment": environment,
        "critical_roots": critical,
        "runtime_loaded_local_dependencies": runtime,
        "dependency_manifest_file_sha256": file_sha256(manifest_path),
        "dependency_manifest_aggregate_sha256": manifest["aggregate_sha256"],
        "harness": harness,
        "historical_execution_mode": dict(HISTORICAL_EXECUTION_MODE),
    }
    output_relative = output.resolve().relative_to(ROOT).as_posix()
    log_relative = log_path.resolve().relative_to(ROOT).as_posix()
    payload: dict[str, Any] = {
        "schema_version": TARGET_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_role": "phase_a0_historical_scalar_ssl_lstm_target_lock",
        "classification": "extension_or_invention",
        "run_manifest": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "cwd": str(ROOT), "command": EXPECTED_GENERATION_COMMAND,
            "interpreter": interpreter,
            "python_version": python_version, "packages": packages,
            "environment": environment, "execution_mode": "cpu_hidden_non_xla_reference_replay",
            "cpu_gpu_status": "CPU hidden by CUDA_VISIBLE_DEVICES=-1",
            "trust_basis": "cpu_hidden_reference_exception_not_gpu_evidence",
            "started_at_utc": started, "completed_at_utc": "pending",
            "wall_time_seconds": 0.0, "output_path": output_relative,
            "log_path": log_relative, "plan_path": PLAN_PATH, "result_path": RESULT_PATH,
        },
        "immutable_attempt_fingerprint": opening,
        "source_provenance": source_provenance(manifest),
        "target_semantics": target_semantics(target),
        "implementation_execution": implementation,
        "sampler_geometry": evidence["geometry"],
        "forecast_design": forecast_design(),
        "historical_artifact_disposition": dispositions(),
        "probe_results": evidence["probes"],
        "signatures": {},
        "nonclaims": list(EXPECTED_NONCLAIMS),
    }
    payload["run_manifest"]["completed_at_utc"] = datetime.now(UTC).isoformat()
    payload["run_manifest"]["wall_time_seconds"] = float(time.perf_counter() - start_clock)
    payload["signatures"] = signatures(payload)
    closing = make_fingerprint(manifest, manifest_relative)
    require(
        canonical_bytes(fingerprint_projection(opening))
        == canonical_bytes(fingerprint_projection(closing)),
        "opening/closing fingerprint drift",
    )
    require(module_paths() == expected_modules, "closing dependency closure drift")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"status=target_lock_generated\noutput={output_relative}\nimmutable_aggregate={opening['aggregate_sha256']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "target_lock_generated", "artifact": output_relative, "immutable_aggregate": opening["aggregate_sha256"]}))


def verify_tensor(value: Mapping[str, Any], label: str, expected_name: str) -> None:
    import numpy as np

    exact_keys(value, {"name", "dtype", "shape", "order", "byte_order", "values", "raw_sha256"}, label)
    require(value["name"] == expected_name, f"tensor name mismatch: {label}")
    require(value["dtype"] == "float64" and value["order"] == "C" and value["byte_order"] == "little", f"tensor metadata mismatch: {label}")
    shape = value["shape"]
    require(
        isinstance(shape, list)
        and len(shape) <= 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape),
        f"invalid tensor shape: {label}",
    )
    values = value["values"]
    numeric = lambda item: isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
    if len(shape) == 0:
        require(numeric(values), f"invalid scalar tensor values: {label}")
    elif len(shape) == 1:
        require(isinstance(values, list) and len(values) == shape[0], f"invalid vector nesting: {label}")
        require(all(numeric(item) for item in values), f"invalid vector values: {label}")
    else:
        require(isinstance(values, list) and len(values) == shape[0], f"invalid matrix rows: {label}")
        require(
            all(
                isinstance(row, list)
                and len(row) == shape[1]
                and all(numeric(item) for item in row)
                for row in values
            ),
            f"invalid matrix values: {label}",
        )
    array = np.asarray(values, dtype="<f8")
    require(list(array.shape) == shape, f"tensor shape/value mismatch: {label}")
    require(bool(np.all(np.isfinite(array))), f"non-finite tensor: {label}")
    array = np.ascontiguousarray(array)
    require(value["raw_sha256"] == hashlib.sha256(array.tobytes(order="C")).hexdigest(), f"tensor byte hash mismatch: {label}")


def tensor_array(value: Mapping[str, Any], label: str, expected_name: str) -> Any:
    import numpy as np

    verify_tensor(value, label, expected_name)
    array = np.asarray(value["values"], dtype="<f8")
    return np.ascontiguousarray(array) if array.ndim > 0 else array


def verify_target_lock(path: Path) -> None:
    import numpy as np

    require(path == EXPECTED_LOCK_PATH.resolve(), f"unexpected target-lock path: {path}")
    runtime_contract()
    value = strict_load(path)
    exact_keys(value, TOP_KEYS, "target lock")
    require(value["schema_version"] == TARGET_SCHEMA, "target-lock schema mismatch")
    require(value["artifact_role"] == "phase_a0_historical_scalar_ssl_lstm_target_lock", "artifact role mismatch")
    require(value["classification"] == "extension_or_invention", "classification mismatch")
    manifest_relative = value["immutable_attempt_fingerprint"]["dependency_manifest_path"]
    manifest_path = repo_path(manifest_relative)
    manifest = strict_load(manifest_path)
    verify_manifest(manifest)
    fingerprint = value["immutable_attempt_fingerprint"]
    exact_keys(fingerprint, {
        "checkpoint_stage", "dependency_manifest_path", "dependency_manifest_file_sha256",
        "dependency_manifest_aggregate_sha256", "members", "aggregate_sha256",
    }, "immutable fingerprint")
    require(fingerprint["checkpoint_stage"] == CHECKPOINT_STAGE, "checkpoint stage mismatch")
    require(fingerprint["dependency_manifest_file_sha256"] == file_sha256(manifest_path), "manifest file hash mismatch")
    require(fingerprint["dependency_manifest_aggregate_sha256"] == manifest["aggregate_sha256"], "manifest semantic hash mismatch")
    expected_fingerprint = make_fingerprint(manifest, manifest_relative)
    require(fingerprint == expected_fingerprint, "immutable fingerprint drift")
    exact_keys(value["run_manifest"], {
        "git_commit", "git_dirty", "cwd", "command", "interpreter", "python_version",
        "packages", "environment", "execution_mode", "cpu_gpu_status", "trust_basis",
        "started_at_utc", "completed_at_utc", "wall_time_seconds", "output_path",
        "log_path", "plan_path", "result_path",
    }, "run manifest")
    exact_keys(value["run_manifest"]["packages"], set(EXPECTED_PACKAGES), "run packages")
    exact_keys(value["run_manifest"]["environment"], set(ENVIRONMENT_KEYS), "run environment")
    require(value["run_manifest"]["interpreter"] == EXPECTED_INTERPRETER, "recorded interpreter mismatch")
    require(value["run_manifest"]["python_version"] == EXPECTED_PYTHON, "recorded Python mismatch")
    require(value["run_manifest"]["packages"] == EXPECTED_PACKAGES, "recorded package mismatch")
    require(value["run_manifest"]["environment"] == EXPECTED_ENVIRONMENT, "recorded environment mismatch")
    require(value["run_manifest"]["git_commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "Git commit drift")
    require(value["run_manifest"]["git_dirty"] == bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()), "Git dirty-state drift")
    require(value["run_manifest"]["cwd"] == str(ROOT), "recorded cwd mismatch")
    require(value["run_manifest"]["execution_mode"] == "cpu_hidden_non_xla_reference_replay", "execution mode mismatch")
    require(value["run_manifest"]["cpu_gpu_status"] == "CPU hidden by CUDA_VISIBLE_DEVICES=-1", "CPU/GPU status mismatch")
    require(value["run_manifest"]["trust_basis"] == "cpu_hidden_reference_exception_not_gpu_evidence", "trust basis mismatch")
    require(value["run_manifest"]["command"] == EXPECTED_GENERATION_COMMAND, "recorded command mismatch")
    require(value["run_manifest"]["output_path"] == EXPECTED_LOCK_PATH.relative_to(ROOT).as_posix(), "output path mismatch")
    require(value["run_manifest"]["log_path"] == EXPECTED_LOCK_LOG_PATH.relative_to(ROOT).as_posix(), "log path mismatch")
    require(value["run_manifest"]["plan_path"] == PLAN_PATH and value["run_manifest"]["result_path"] == RESULT_PATH, "plan/result path mismatch")
    require(isinstance(value["run_manifest"]["wall_time_seconds"], (int, float)) and not isinstance(value["run_manifest"]["wall_time_seconds"], bool) and math.isfinite(value["run_manifest"]["wall_time_seconds"]) and value["run_manifest"]["wall_time_seconds"] >= 0.0, "invalid wall time")
    started = parse_utc_timestamp(value["run_manifest"]["started_at_utc"], "run start")
    created = parse_utc_timestamp(value["created_at_utc"], "artifact creation")
    completed = parse_utc_timestamp(value["run_manifest"]["completed_at_utc"], "run completion")
    require(started <= created <= completed, "run/artifact timestamps are out of order")
    source = value["source_provenance"]
    exact_keys(source, {"critical_roots", "runtime_loaded_local_dependencies", "historical_inputs", "governance_inputs", "dsge_hmc"}, "source provenance")
    require(source["critical_roots"] == descriptors(CRITICAL_ROOTS), "critical provenance drift")
    require(source["runtime_loaded_local_dependencies"] == manifest["runtime_loaded_local_dependencies"], "runtime provenance drift")
    require(source["historical_inputs"] == descriptors(HISTORICAL_INPUTS), "historical provenance drift")
    require(source["governance_inputs"] == descriptors(GOVERNANCE_INPUTS), "governance provenance drift")
    dsge = source["dsge_hmc"]
    exact_keys(dsge, {"repository_path", "git_commit", "git_dirty", "governance_files", "gate1_context_files", "role"}, "dsge provenance")
    require(dsge["governance_files"] == external_descriptors(DSGE_GOVERNANCE), "dsge governance drift")
    require(dsge["gate1_context_files"] == external_descriptors(DSGE_CONTEXT), "dsge context drift")
    dsge_root = Path("/home/ubuntu/python/dsge_hmc")
    require(dsge["repository_path"] == str(dsge_root), "dsge repository path mismatch")
    require(dsge["git_commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dsge_root, text=True).strip(), "dsge commit drift")
    require(dsge["git_dirty"] == bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=dsge_root, text=True).strip()), "dsge dirty-state drift")
    require(dsge["role"] == "external_design_provenance_only_not_bayesfilter_evidence", "dsge role mismatch")
    target = value["target_semantics"]
    exact_keys(target, {"static_config", "parameter_chart", "full_fixture", "free_mask", "fixed_parameters", "observations", "likelihood", "prior", "dtype"}, "target semantics")
    exact_keys(target["static_config"], {"horizon", "latent_dim", "hidden_dim", "observation_dim", "augmented_state_dim", "covariance_mode", "full_parameter_dim"}, "static config")
    exact_keys(target["free_mask"], {"names", "indices", "truth_free"}, "free mask")
    exact_keys(target["likelihood"], {"name", "score_helper", "std_floor", "alpha", "beta", "kappa", "placement_floor", "innovation_floor", "rank_tolerance", "spectral_gap_tolerance", "fixed_null_tolerance", "jitter", "allow_fixed_null_support"}, "likelihood")
    exact_keys(target["prior"], {"family", "center", "standard_deviation", "normalized", "log_kernel_formula"}, "prior")
    require(target["static_config"] == STATIC_CONFIG, "static target config mismatch")
    require(target["parameter_chart"] == list(PARAMETER_CHART), "parameter chart mismatch")
    require(target["free_mask"]["names"] == list(FREE_NAMES) and target["free_mask"]["indices"] == list(FREE_INDICES), "free mask mismatch")
    fixture = tensor_array(target["full_fixture"], "target.full_fixture", "target_semantics.full_fixture")
    observations = tensor_array(target["observations"], "target.observations", "target_semantics.observations")
    truth = tensor_array(target["free_mask"]["truth_free"], "target.truth_free", "target_semantics.free_mask.truth_free")
    prior_center = tensor_array(target["prior"]["center"], "target.prior.center", "target_semantics.prior.center")
    require(np.array_equal(fixture, np.asarray(FULL_FIXTURE, dtype="<f8")), "fixture values mismatch")
    require(observations.shape == (30, 1), "observation shape mismatch")
    require(np.array_equal(truth, np.asarray(TRUTH_FREE, dtype="<f8")), "truth values mismatch")
    require(np.array_equal(prior_center, truth), "prior center mismatch")
    require(target["likelihood"] == LIKELIHOOD_CONFIG, "likelihood config mismatch")
    require(target["prior"] == {
        "family": "unnormalized_isotropic_gaussian_log_kernel",
        "center": target["prior"]["center"],
        "standard_deviation": 4.0,
        "normalized": False,
        "log_kernel_formula": "-0.5 * sum((free - truth_free)^2 / 4.0^2)",
    }, "prior contract mismatch")
    require(target["dtype"] == "float64", "target dtype mismatch")
    fixed = target["fixed_parameters"]
    require(len(fixed) == 20, "fixed-parameter count mismatch")
    for row in fixed:
        exact_keys(row, {"name", "index", "value"}, "fixed parameter")
        require(math.isfinite(row["value"]), "non-finite fixed parameter")
    expected_fixed = [
        {"name": PARAMETER_CHART[index], "index": index, "value": FULL_FIXTURE[index]}
        for index in range(24) if index not in FREE_INDICES
    ]
    require(fixed == expected_fixed, "fixed-parameter values mismatch")
    implementation = value["implementation_execution"]
    exact_keys(implementation, {"interpreter", "python_version", "packages", "environment", "critical_roots", "runtime_loaded_local_dependencies", "dependency_manifest_file_sha256", "dependency_manifest_aggregate_sha256", "harness", "historical_execution_mode"}, "implementation execution")
    exact_keys(implementation["harness"], {"path", "sha256"}, "implementation harness")
    exact_keys(implementation["historical_execution_mode"], {"device", "cpu_hidden", "jit_compile", "xla", "dtype", "role"}, "historical execution mode")
    require(implementation["packages"] == value["run_manifest"]["packages"], "package objects differ")
    require(implementation["environment"] == value["run_manifest"]["environment"], "environment objects differ")
    require(implementation["interpreter"] == EXPECTED_INTERPRETER, "implementation interpreter mismatch")
    require(implementation["python_version"] == EXPECTED_PYTHON, "implementation Python mismatch")
    require(implementation["critical_roots"] == descriptors(CRITICAL_ROOTS), "implementation critical roots drift")
    require(implementation["runtime_loaded_local_dependencies"] == manifest["runtime_loaded_local_dependencies"], "implementation runtime closure drift")
    require(implementation["dependency_manifest_file_sha256"] == file_sha256(manifest_path), "implementation manifest file hash mismatch")
    require(implementation["dependency_manifest_aggregate_sha256"] == manifest["aggregate_sha256"], "implementation manifest aggregate mismatch")
    require(implementation["harness"] == {"path": HARNESS_PATH, "sha256": file_sha256(repo_path(HARNESS_PATH))}, "implementation harness mismatch")
    require(implementation["historical_execution_mode"] == HISTORICAL_EXECUTION_MODE, "historical execution-mode mismatch")
    geometry = value["sampler_geometry"]
    exact_keys(geometry, {"role", "source_path", "source_sha256", "source_script_path", "source_script_sha256", "coordinate_names", "coordinate_formula", "center_free", "scale", "factor_z", "covariance_z", "precision_z", "covariance_theta", "precision_theta", "reconstruction_tolerance_formula", "regularization", "checks", "nonclaims"}, "sampler geometry")
    exact_keys(geometry["regularization"], {"dense_mass", "jitter", "eigenvalue_floor", "max_condition_number"}, "geometry regularization")
    exact_keys(geometry["checks"], {"factor_lower_triangular", "factor_positive_diagonal", "factor_covariance_residual_inf", "factor_covariance_tolerance", "raw_theta_covariance_residual_inf", "raw_theta_covariance_tolerance", "regularized_theta_precision_residual_inf", "regularized_theta_precision_tolerance", "regularized_theta_covariance_residual_inf", "regularized_theta_covariance_tolerance", "raw_to_stored_theta_covariance_residual_inf", "raw_to_stored_theta_precision_residual_inf", "z_inverse_residual_inf", "z_inverse_tolerance", "theta_inverse_residual_inf", "theta_inverse_tolerance", "regularization_effective_eigenvalue_floor", "regularization_clipped_eigenvalue_count", "passed"}, "geometry checks")
    for label in ("center_free", "scale", "factor_z", "covariance_z", "precision_z", "covariance_theta", "precision_theta"):
        verify_tensor(geometry[label], f"geometry.{label}", f"sampler_geometry.{label}")
    phase2s = strict_load(repo_path(geometry["source_path"]))
    expected_geometry = geometry_payload(phase2s)
    require(geometry == expected_geometry, "geometry semantic drift")
    forecast = value["forecast_design"]
    exact_keys(forecast, {"status", "horizon", "terminal_state", "transition", "process_noise", "hidden_cell_completion", "observation_noise", "path_clustering", "innovation_banks", "unfrozen_fields"}, "forecast design")
    require(forecast == forecast_design(), "forecast design mismatch")
    probes = value["probe_results"]
    exact_keys(probes, {"truth_free", "phase2s_center"}, "probe results")
    probe_keys = {"name", "free_position", "likelihood_value", "likelihood_score", "prior_value", "prior_score", "total_value", "total_score", "value_residual", "score_residual_inf", "value_tolerance", "score_tolerance", "passed"}
    historical_phase1 = strict_load(repo_path(
        "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json"
    ))
    historical_phase2s = strict_load(repo_path(
        "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
    ))
    expected_probe_totals = {
        "truth_free": {
            "value": historical_phase1["center"]["log_prob"],
            "score": historical_phase1["center"]["score"],
        },
        "phase2s_center": {
            "value": historical_phase2s["target_replay"]["values"]["map_candidate"]["value"],
            "score": historical_phase2s["target_replay"]["values"]["map_candidate"]["score"],
        },
    }
    for name in ("truth_free", "phase2s_center"):
        probe = probes[name]
        exact_keys(probe, probe_keys, f"probe {name}")
        require(probe["name"] == name and probe["passed"] is True, f"probe status mismatch: {name}")
        arrays = {
            field: tensor_array(probe[field], f"probe.{name}.{field}", f"{name}.{field}")
            for field in probe_keys - {"name", "passed"}
        }
        require(arrays["free_position"].shape == (4,), f"probe free-position shape: {name}")
        for field in ("likelihood_value", "prior_value", "total_value", "value_residual", "score_residual_inf", "value_tolerance", "score_tolerance"):
            require(arrays[field].shape == (), f"probe scalar shape mismatch: {name}.{field}")
        for field in ("likelihood_score", "prior_score", "total_score"):
            require(arrays[field].shape == (4,), f"probe score shape mismatch: {name}.{field}")
        expected_position = truth if name == "truth_free" else tensor_array(
            geometry["center_free"],
            "geometry.center_free.probe",
            "sampler_geometry.center_free",
        )
        require(np.array_equal(arrays["free_position"], expected_position), f"probe position mismatch: {name}")
        expected_prior_score = -(arrays["free_position"] - truth) / 16.0
        expected_prior_value = -0.5 * np.sum(np.square(arrays["free_position"] - truth) / 16.0)
        require(np.array_equal(arrays["prior_score"], expected_prior_score), f"probe prior score mismatch: {name}")
        require(float(arrays["prior_value"]) == float(expected_prior_value), f"probe prior value mismatch: {name}")
        expected_total_value = float(arrays["likelihood_value"]) + float(arrays["prior_value"])
        expected_total_score = arrays["likelihood_score"] + arrays["prior_score"]
        require(float(arrays["total_value"]) == expected_total_value, f"probe total value mismatch: {name}")
        require(np.array_equal(arrays["total_score"], expected_total_score), f"probe total score mismatch: {name}")
        historical_value = float(expected_probe_totals[name]["value"])
        historical_score = np.asarray(expected_probe_totals[name]["score"], dtype="<f8")
        eps64 = 2.0**-52
        historical_value_tolerance = 8.0 * eps64 * max(
            1.0, abs(float(arrays["total_value"])), abs(historical_value)
        )
        historical_score_tolerance = 8.0 * eps64 * max(
            1.0,
            float(np.max(np.abs(arrays["total_score"]))),
            float(np.max(np.abs(historical_score))),
        )
        require(
            abs(float(arrays["total_value"]) - historical_value)
            <= historical_value_tolerance,
            f"historical probe value mismatch: {name}",
        )
        require(
            float(np.max(np.abs(arrays["total_score"] - historical_score)))
            <= historical_score_tolerance,
            f"historical probe score mismatch: {name}",
        )
        value_residual = abs(float(arrays["total_value"]) - expected_total_value)
        score_residual = float(np.max(np.abs(arrays["total_score"] - expected_total_score)))
        value_tolerance = 8.0 * eps64 * max(1.0, abs(float(arrays["total_value"])), abs(float(arrays["likelihood_value"])) + abs(float(arrays["prior_value"])))
        score_tolerance = 8.0 * eps64 * max(1.0, float(np.max(np.abs(arrays["total_score"]))), float(np.max(np.abs(arrays["likelihood_score"]))) + float(np.max(np.abs(arrays["prior_score"]))))
        require(float(arrays["value_residual"]) == value_residual and float(arrays["value_tolerance"]) == value_tolerance, f"probe value diagnostics mismatch: {name}")
        require(float(arrays["score_residual_inf"]) == score_residual and float(arrays["score_tolerance"]) == score_tolerance, f"probe score diagnostics mismatch: {name}")
        require(value_residual <= value_tolerance and score_residual <= score_tolerance, f"probe decomposition failed: {name}")
    replay_module = load_historical_module()
    replay = warmup_cycle(replay_module)
    validate_historical_target(replay["target_a"])
    replay_observations = np.asarray(replay["target_a"].observations.numpy(), dtype="<f8")
    require(np.array_equal(observations, replay_observations), "fresh-process observation replay mismatch")
    require(probes == replay["probes"], "fresh-process target-probe replay mismatch")
    require(
        module_paths() == [row["path"] for row in manifest["runtime_loaded_local_dependencies"]],
        "fresh-process verifier dependency closure drift",
    )
    dispositions_value = value["historical_artifact_disposition"]
    require(dispositions_value == dispositions(), "historical disposition drift")
    exact_keys(value["signatures"], {"target_semantic_sha256", "implementation_execution_sha256", "sampler_geometry_sha256", "forecast_design_sha256", "aggregate_sha256"}, "signatures")
    require(value["signatures"] == signatures(value), "component/signature aggregate mismatch")
    require(value["nonclaims"] == EXPECTED_NONCLAIMS, "nonclaims mismatch")
    print(json.dumps({"status": "target_lock_verified", "artifact": str(path), "immutable_aggregate": fingerprint["aggregate_sha256"], "signature_aggregate": value["signatures"]["aggregate_sha256"]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--discover-dependencies", type=Path)
    group.add_argument("--verify", type=Path)
    group.add_argument("--dependency-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()
    if args.discover_dependencies is not None:
        require(args.log_path is not None and args.output is None, "discovery requires --log-path only")
        require(args.discover_dependencies.resolve() == EXPECTED_MANIFEST_PATH.resolve(), "unexpected discovery manifest path")
        require(args.log_path.resolve() == EXPECTED_DISCOVERY_LOG_PATH.resolve(), "unexpected discovery log path")
    elif args.dependency_manifest is not None:
        require(args.output is not None and args.log_path is not None, "generation requires --output and --log-path")
        require(args.dependency_manifest.resolve() == EXPECTED_MANIFEST_PATH.resolve(), "unexpected dependency manifest path")
        require(args.output.resolve() == EXPECTED_LOCK_PATH.resolve(), "unexpected target-lock output path")
        require(args.log_path.resolve() == EXPECTED_LOCK_LOG_PATH.resolve(), "unexpected target-lock log path")
        require(len({args.dependency_manifest.resolve(), args.output.resolve(), args.log_path.resolve()}) == 3, "generation paths must not alias")
    else:
        require(args.output is None and args.log_path is None, "verification accepts no output/log arguments")
        require(args.verify.resolve() == EXPECTED_LOCK_PATH.resolve(), "unexpected verification path")
    return args


def main() -> None:
    require(ROOT == Path.cwd().resolve(), f"run from repository root: {ROOT}")
    args = parse_args()
    if args.discover_dependencies is not None:
        discover(args.discover_dependencies.resolve(), args.log_path.resolve())
    elif args.dependency_manifest is not None:
        generate(args.dependency_manifest.resolve(), args.output.resolve(), args.log_path.resolve())
    else:
        verify_target_lock(args.verify.resolve())


if __name__ == "__main__":
    main()
