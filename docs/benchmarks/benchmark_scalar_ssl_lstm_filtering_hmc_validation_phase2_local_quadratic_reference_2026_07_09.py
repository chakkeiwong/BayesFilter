"""Phase 2 local quadratic reference agreement for scalar filtering HMC.

This artifact compares Phase 1R HMC marginal summaries against the local
quadratic Gaussian reference implied by the accepted geometry/mass handoff in
the HMC execution coordinate ``u``.  The reference is not an exact posterior
oracle and this screen does not claim posterior correctness, convergence, HMC
readiness, zero divergences, GPU/XLA readiness, or default readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2_local_quadratic_reference.v1"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
)
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-result-2026-07-09.md"
)
DEFAULT_GEOMETRY_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json"
)
DEFAULT_MASS_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_mass_handoff_cpu_hidden_2026-07-08.json"
)
DEFAULT_PHASE1R_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md"
)
NONCLAIMS = (
    "local quadratic reference agreement screen only",
    "not an exact posterior reference",
    "not HMC readiness evidence",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not a zero-divergence claim when native divergence is unavailable",
    "not sampler superiority evidence",
    "not statistically supported ranking evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
    "not Zhao-Cui source-faithfulness evidence",
)


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2_reference_agreement(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase1r_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    precondition = validate_inputs(geometry_payload, mass_payload, phase1r_payload)
    vetoes = list(precondition["vetoes"])
    reference: Mapping[str, Any] | None = None
    hmc_summary: Mapping[str, Any] | None = None
    agreement: Mapping[str, Any] | None = None

    if not vetoes:
        reference = build_local_quadratic_reference(geometry_payload, mass_payload)
        vetoes.extend(reference.get("vetoes", ()))
    if reference is not None and not vetoes:
        hmc_summary = summarize_phase1r_hmc(phase1r_payload)
        vetoes.extend(hmc_summary.get("vetoes", ()))
    if reference is not None and hmc_summary is not None and not vetoes:
        agreement = evaluate_agreement(reference, hmc_summary)
        vetoes.extend(agreement.get("vetoes", ()))

    unique_vetoes = tuple(dict.fromkeys(vetoes))
    passed = bool(not unique_vetoes and agreement and agreement.get("passed") is True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2_local_quadratic_reference",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": phase1r_payload.get("target_scope"),
        "settings": {
            "mean_abs_error_threshold": 0.5,
            "std_ratio_lower": 0.5,
            "std_ratio_upper": 2.0,
            "reference_type": "local_quadratic_gaussian_in_hmc_u_coordinate",
            "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        },
        "source_artifacts": {
            "geometry_json": str(DEFAULT_GEOMETRY_PATH.relative_to(ROOT)),
            "mass_json": str(DEFAULT_MASS_PATH.relative_to(ROOT)),
            "phase1r_json": str(DEFAULT_PHASE1R_PATH.relative_to(ROOT)),
        },
        "environment": environment_payload(),
        "git": git_payload(),
        "precondition": precondition,
        "reference": reference,
        "hmc_summary": hmc_summary,
        "agreement": agreement,
        "telemetry_policy": telemetry_policy_payload(phase1r_payload),
        "decision": {
            "phase2_local_quadratic_reference_agreement_passed": passed,
            "vetoes": unique_vetoes,
            "zero_divergence_claim_made": False,
            "viable_for_phase3_gpu_xla_subplan": passed,
            "next_justified_action": (
                "write Phase 2 result and refresh/review Phase 3 GPU/XLA subplan"
                if passed
                else "write Phase 2 result and draft reference/localization repair before GPU/XLA"
            ),
        },
        "metric_roles": {
            "phase2_local_quadratic_reference_agreement_passed": "primary_phase2_pass_fail",
            "reference_spd": "hard_veto_evidence",
            "input_artifacts_valid": "hard_veto_evidence",
            "mean_abs_error_max": "promotion_veto_screen",
            "std_ratio_range": "promotion_veto_screen",
            "native_divergence_unavailable": "telemetry_availability_not_zero_divergences",
            "acceptance_rates": "explanatory_only",
            "log_accept_tails": "explanatory_only",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "statistically_supported_ranking": "none; no method comparison and no uncertainty interval",
            "descriptive_only_differences": "mean errors, standard-deviation ratios, acceptance, and log-accept tails",
            "posterior_correctness": "not assessed; local quadratic reference only",
            "default_readiness": "not assessed",
            "gpu_xla_readiness": "not assessed; CPU-hidden artifact analysis",
            "hmc_readiness": "not assessed",
            "zero_divergence_claim": "not made",
            "next_evidence_needed": (
                "Phase 3 GPU/XLA only if this local-reference screen passes"
                if passed
                else "localize geometry/transform/short-chain mismatch before GPU/XLA"
            ),
        },
        "decision_table": {
            "decision": "Phase 2 local quadratic reference agreement",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "The comparator is a local quadratic Gaussian, not an exact posterior. "
                "Short-chain HMC summary errors may reflect local geometry mismatch, "
                "short-chain bias, transform issues, or target nonquadraticity."
            ),
            "next_justified_action": (
                "refresh/review Phase 3 GPU/XLA" if passed else "draft a localization repair before Phase 3"
            ),
            "what_is_not_being_concluded": (
                "No posterior correctness, HMC convergence/readiness, zero-divergence "
                "claim, sampler superiority, GPU/XLA readiness, default readiness, or "
                "Zhao-Cui source-faithfulness."
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 180 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden artifact analysis",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": phase1r_payload.get("settings", {}).get("seeds"),
            "wall_time_seconds": float(time.perf_counter() - start),
            "output_artifacts": (
                str(DEFAULT_JSON_PATH.relative_to(ROOT)),
                str(DEFAULT_MARKDOWN_PATH.relative_to(ROOT)),
            ),
            "plan_file": PLAN_PATH,
            "subplan_file": SUBPLAN_PATH,
            "result_file": RESULT_PATH,
        },
        "nonclaims": NONCLAIMS,
    }
    return json_ready(payload)


def validate_inputs(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase1r_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    if geometry_payload.get("schema_version") != "scalar_ssl_lstm.filtering_geometry.v1":
        vetoes.append("geometry_schema_mismatch")
    if geometry_payload.get("decision", {}).get("geometry_sanity_passed") is not True:
        vetoes.append("geometry_not_passed")
    if mass_payload.get("schema_version") != "scalar_ssl_lstm.filtering_mass_handoff.v1":
        vetoes.append("mass_schema_mismatch")
    if mass_payload.get("decision", {}).get("mass_handoff_passed") is not True:
        vetoes.append("mass_handoff_not_passed")
    if (
        phase1r_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase1r.v1"
    ):
        vetoes.append("phase1r_schema_mismatch")
    if (
        phase1r_payload.get("decision", {}).get("phase1r_acceptance_repair_screen_passed")
        is not True
    ):
        vetoes.append("phase1r_screen_not_passed")
    if phase1r_payload.get("decision", {}).get("vetoes"):
        vetoes.append("phase1r_vetoes_present")
    statuses = phase1r_payload.get("telemetry_policy", {}).get("native_divergence_statuses", ())
    if any(str(status) == "available" for status in statuses):
        # Available native divergence is allowed only if no seed reports a positive count.
        for row in phase1r_payload.get("seed_rows", ()):
            native = row.get("trace_summary", {}).get("native_divergence", {})
            if isinstance(native, Mapping) and native.get("available") and int(native.get("count", 0)) > 0:
                vetoes.append("phase1r_native_divergence_detected")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "geometry_schema": geometry_payload.get("schema_version"),
        "mass_schema": mass_payload.get("schema_version"),
        "phase1r_schema": phase1r_payload.get("schema_version"),
        "coordinate_contract": mass_payload.get("coordinate_contract", {}),
        "phase1r_decision": phase1r_payload.get("decision", {}),
    }


def build_local_quadratic_reference(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    low_rank = geometry_payload.get("low_rank_geometry", {})
    mass = mass_payload.get("mass_handoff", {})
    linear_z = np.asarray(low_rank.get("linear_term"), dtype=float)
    k_z = np.asarray(mass.get("regularized_precision_K_z"), dtype=float)
    factor = np.asarray(mass.get("factor"), dtype=float)
    if linear_z.shape != (4,):
        vetoes.append("linear_term_shape_mismatch")
    if k_z.shape != (4, 4) or not np.all(np.isfinite(k_z)):
        vetoes.append("precision_shape_or_finiteness_mismatch")
    if factor.shape != (4, 4) or not np.all(np.isfinite(factor)):
        vetoes.append("factor_shape_or_finiteness_mismatch")
    if vetoes:
        return {"passed": False, "vetoes": tuple(dict.fromkeys(vetoes))}
    k_u = factor.T @ k_z @ factor
    k_u = 0.5 * (k_u + k_u.T)
    try:
        c_u = np.linalg.inv(k_u)
    except np.linalg.LinAlgError:
        vetoes.append("u_precision_not_invertible")
        c_u = np.full_like(k_u, np.nan)
    c_u = 0.5 * (c_u + c_u.T)
    linear_u = factor.T @ linear_z
    mean_u = c_u @ linear_u
    eig_k = np.linalg.eigvalsh(k_u)
    eig_c = np.linalg.eigvalsh(c_u)
    if not np.all(np.isfinite(eig_k)) or np.any(eig_k <= 0.0):
        vetoes.append("u_precision_not_spd")
    if not np.all(np.isfinite(eig_c)) or np.any(eig_c <= 0.0):
        vetoes.append("u_covariance_not_spd")
    reference_std = np.sqrt(np.diag(c_u))
    if not np.all(np.isfinite(reference_std)) or np.any(reference_std <= 0.0):
        vetoes.append("reference_std_invalid")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "formula": {
            "local_log_density": "c + l_z^T z - 0.5 z^T K_z z",
            "coordinate_map": "z = F u, F = chol(M_z)",
            "precision_u": "K_u = F.T @ K_z @ F",
            "covariance_u": "C_u = inv(K_u)",
            "mean_u": "m_u = C_u @ F.T @ l_z",
        },
        "linear_z": linear_z,
        "linear_u": linear_u,
        "mean_u": mean_u,
        "precision_u": k_u,
        "covariance_u": c_u,
        "std_u": reference_std,
        "precision_u_identity_max_abs_error": float(np.max(np.abs(k_u - np.eye(4)))),
        "covariance_u_identity_max_abs_error": float(np.max(np.abs(c_u - np.eye(4)))),
        "precision_u_eigenvalues": eig_k,
        "covariance_u_eigenvalues": eig_c,
        "nonclaims": (
            "local quadratic Gaussian reference only",
            "not an exact posterior reference",
        ),
    }


def summarize_phase1r_hmc(phase1r_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    vetoes: list[str] = []
    rows = phase1r_payload.get("seed_rows", ())
    means = []
    variances = []
    counts = []
    acceptance_rates = []
    log_accept_max_abs = []
    native_statuses = []
    for index, row in enumerate(rows):
        samples = row.get("samples_summary", {})
        mean = np.asarray(samples.get("mean_u"), dtype=float)
        std = np.asarray(samples.get("std_u"), dtype=float)
        count = int(samples.get("finite_sample_count", 0))
        if mean.shape != (4,) or std.shape != (4,) or count <= 0:
            vetoes.append(f"seed_{index}_sample_summary_invalid")
            continue
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            vetoes.append(f"seed_{index}_sample_summary_nonfinite")
        means.append(mean)
        variances.append(std**2)
        counts.append(count)
        trace = row.get("trace_summary", {})
        acceptance_rates.append(trace.get("acceptance_rate"))
        log_accept = trace.get("log_accept_ratio", {})
        if isinstance(log_accept, Mapping):
            log_accept_max_abs.append(log_accept.get("max_abs_finite"))
        native = trace.get("native_divergence", {})
        if isinstance(native, Mapping):
            native_statuses.append(native.get("status", "available" if native.get("available") else "unknown"))
    if not means:
        vetoes.append("no_valid_seed_summaries")
        return {"passed": False, "vetoes": tuple(dict.fromkeys(vetoes))}
    means_array = np.vstack(means)
    variances_array = np.vstack(variances)
    counts_array = np.asarray(counts, dtype=float)
    total = float(np.sum(counts_array))
    pooled_mean = np.sum(means_array * counts_array[:, np.newaxis], axis=0) / total
    pooled_second = np.sum(
        (variances_array + means_array**2) * counts_array[:, np.newaxis],
        axis=0,
    ) / total
    pooled_var = pooled_second - pooled_mean**2
    if not np.all(np.isfinite(pooled_var)) or np.any(pooled_var < 0.0):
        vetoes.append("pooled_variance_invalid")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "seed_count": len(rows),
        "total_retained_samples": int(total),
        "pooled_mean_u": pooled_mean,
        "pooled_std_u": np.sqrt(np.maximum(pooled_var, 0.0)),
        "seed_mean_u": means_array,
        "seed_std_u": np.sqrt(variances_array),
        "acceptance_rates": acceptance_rates,
        "log_accept_max_abs_by_seed": log_accept_max_abs,
        "native_divergence_statuses": native_statuses,
    }


def evaluate_agreement(
    reference: Mapping[str, Any],
    hmc_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    ref_mean = np.asarray(reference.get("mean_u"), dtype=float)
    ref_std = np.asarray(reference.get("std_u"), dtype=float)
    hmc_mean = np.asarray(hmc_summary.get("pooled_mean_u"), dtype=float)
    hmc_std = np.asarray(hmc_summary.get("pooled_std_u"), dtype=float)
    mean_error = hmc_mean - ref_mean
    mean_abs = np.abs(mean_error)
    std_ratio = hmc_std / ref_std
    vetoes: list[str] = []
    if not np.all(np.isfinite(mean_abs)):
        vetoes.append("mean_error_nonfinite")
    if not np.all(np.isfinite(std_ratio)):
        vetoes.append("std_ratio_nonfinite")
    if float(np.max(mean_abs)) > 0.5:
        vetoes.append("mean_abs_error_above_0p5")
    if np.any(std_ratio < 0.5) or np.any(std_ratio > 2.0):
        vetoes.append("std_ratio_outside_0p5_2p0")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "thresholds": {
            "mean_abs_error_max": 0.5,
            "std_ratio_lower": 0.5,
            "std_ratio_upper": 2.0,
            "role": "loose_engineering_screen_not_statistical_proof",
        },
        "hmc_mean_u": hmc_mean,
        "reference_mean_u": ref_mean,
        "mean_error": mean_error,
        "mean_abs_error": mean_abs,
        "mean_abs_error_max": float(np.max(mean_abs)),
        "hmc_std_u": hmc_std,
        "reference_std_u": ref_std,
        "std_ratio": std_ratio,
        "std_ratio_min": float(np.min(std_ratio)),
        "std_ratio_max": float(np.max(std_ratio)),
        "interpretation": (
            "local quadratic reference screen passed"
            if not vetoes
            else "local quadratic reference screen failed; localize geometry, transform, or short-chain behavior before GPU/XLA"
        ),
    }


def telemetry_policy_payload(phase1r_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    telemetry = phase1r_payload.get("telemetry_policy", {})
    return {
        "native_divergence_statuses": telemetry.get("native_divergence_statuses", ()),
        "native_divergence_interpretation": telemetry.get("native_divergence_interpretation"),
        "zero_divergence_claim_made": False,
        "unavailable_native_divergence_is_zero_divergence": False,
        "log_accept_threshold_used_as_native_divergence": False,
    }


def environment_payload() -> Mapping[str, Any]:
    return {
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
    }


def git_payload() -> Mapping[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(lines),
        "dirty_line_count": len(lines),
        "dirty_preview": lines[:20],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    agreement = payload.get("agreement") or {}
    reference = payload.get("reference") or {}
    hmc = payload.get("hmc_summary") or {}
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2 - Local Quadratic Reference",
        "",
        "## Decision",
        "",
        f"- phase2_local_quadratic_reference_agreement_passed: `{decision['phase2_local_quadratic_reference_agreement_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Reference",
        "",
        f"- formula: `{reference.get('formula')}`",
        f"- reference mean u: `{reference.get('mean_u')}`",
        f"- reference std u: `{reference.get('std_u')}`",
        f"- precision-u identity max abs error: `{reference.get('precision_u_identity_max_abs_error')}`",
        "",
        "## HMC Summary",
        "",
        f"- pooled mean u: `{hmc.get('pooled_mean_u')}`",
        f"- pooled std u: `{hmc.get('pooled_std_u')}`",
        f"- acceptance rates: `{hmc.get('acceptance_rates')}`",
        f"- native divergence statuses: `{hmc.get('native_divergence_statuses')}`",
        "",
        "## Agreement",
        "",
        f"- mean abs error: `{agreement.get('mean_abs_error')}`",
        f"- mean abs error max: `{agreement.get('mean_abs_error_max')}`",
        f"- std ratio: `{agreement.get('std_ratio')}`",
        f"- std ratio range: `{agreement.get('std_ratio_min')}` to `{agreement.get('std_ratio_max')}`",
        f"- interpretation: {agreement.get('interpretation')}",
        "",
        "## Inference Status",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key, value in payload["inference_status"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines) + "\n"


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument("--mass-json", type=Path, default=DEFAULT_MASS_PATH)
    parser.add_argument("--phase1r-json", type=Path, default=DEFAULT_PHASE1R_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase2_reference_agreement(
        load_json(args.geometry_json),
        load_json(args.mass_json),
        load_json(args.phase1r_json),
    )
    payload["source_artifacts"] = {
        "geometry_json": str(args.geometry_json),
        "mass_json": str(args.mass_json),
        "phase1r_json": str(args.phase1r_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
