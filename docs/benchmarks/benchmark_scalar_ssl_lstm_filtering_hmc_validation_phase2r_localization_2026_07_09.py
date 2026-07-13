"""Phase 2R localization for scalar filtering HMC local-reference mismatch."""

from __future__ import annotations

import argparse
import importlib.util
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_NAME = "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py"
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2r_localization.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-result-2026-07-09.md"
)
DEFAULT_GEOMETRY_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json"
)
DEFAULT_MASS_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_mass_handoff_cpu_hidden_2026-07-08.json"
)
DEFAULT_PHASE2_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md"
)
GEOMETRY_MODULE_PATH = ROOT / "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py"
NONCLAIMS = (
    "Phase 2R localization diagnostic only",
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


def run_phase2r_localization(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase2_payload: Mapping[str, Any],
    *,
    replay_target: bool = True,
) -> Mapping[str, Any]:
    start = time.perf_counter()
    precondition = validate_inputs(geometry_payload, mass_payload, phase2_payload)
    vetoes = list(precondition["vetoes"])
    transform = transform_checks(phase2_payload)
    vetoes.extend(transform["vetoes"])
    diagnostics = localization_diagnostics(geometry_payload, phase2_payload)
    vetoes.extend(diagnostics["vetoes"])
    replay = target_replay_diagnostics(phase2_payload) if replay_target and not vetoes else {
        "computed": False,
        "reason": "skipped_due_to_precondition_or_disabled",
        "vetoes": (),
    }
    vetoes.extend(replay.get("vetoes", ()))
    outcome = select_outcome(transform, diagnostics, replay)
    unique_vetoes = tuple(dict.fromkeys(vetoes))
    passed = bool(not unique_vetoes and outcome["selected_outcome"] is not None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2r_localization",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": phase2_payload.get("target_scope"),
        "settings": {
            "transform_identity_threshold": 1.0e-8,
            "trust_radius": float(geometry_payload.get("settings", {}).get("low_rank_trust_radius", 0.30)),
            "outside_trust_warning_threshold": 0.60,
            "large_quadratic_drop_threshold": 10.0,
            "replay_target": bool(replay_target),
        },
        "source_artifacts": {
            "geometry_json": str(DEFAULT_GEOMETRY_PATH.relative_to(ROOT)),
            "mass_json": str(DEFAULT_MASS_PATH.relative_to(ROOT)),
            "phase2_json": str(DEFAULT_PHASE2_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "transform_checks": transform,
        "localization_diagnostics": diagnostics,
        "target_replay": replay,
        "outcome": outcome,
        "telemetry_policy": phase2_payload.get("telemetry_policy", {}),
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": {
            "phase2r_localization_passed": passed,
            "vetoes": unique_vetoes,
            "selected_outcome": outcome["selected_outcome"],
            "zero_divergence_claim_made": False,
            "next_justified_action": outcome["next_subplan"],
        },
        "metric_roles": {
            "phase2r_localization_passed": "primary_phase2r_pass_fail",
            "transform_identity": "hard_veto_evidence",
            "outside_trust_region": "localization_outcome_evidence",
            "quadratic_drop": "localization_outcome_evidence",
            "target_replay": "explanatory_localization_only",
            "native_divergence_unavailable": "telemetry_availability_not_zero_divergences",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "selected_localization_outcome": outcome["selected_outcome"],
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed",
            "gpu_xla_readiness": "blocked until targeted repair",
            "default_readiness": "not assessed",
            "zero_divergence_claim": "not made",
            "next_evidence_needed": outcome["next_subplan"],
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 240 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden artifact analysis with optional cheap target replay",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
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
    phase2_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    if geometry_payload.get("schema_version") != "scalar_ssl_lstm.filtering_geometry.v1":
        vetoes.append("geometry_schema_mismatch")
    if mass_payload.get("schema_version") != "scalar_ssl_lstm.filtering_mass_handoff.v1":
        vetoes.append("mass_schema_mismatch")
    if (
        phase2_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2_local_quadratic_reference.v1"
    ):
        vetoes.append("phase2_schema_mismatch")
    if phase2_payload.get("decision", {}).get("phase2_local_quadratic_reference_agreement_passed") is not False:
        vetoes.append("phase2_expected_failure_missing")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2_vetoes": phase2_payload.get("decision", {}).get("vetoes", ()),
    }


def transform_checks(phase2_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = phase2_payload.get("reference", {})
    precision_error = float(reference.get("precision_u_identity_max_abs_error", np.inf))
    covariance_error = float(reference.get("covariance_u_identity_max_abs_error", np.inf))
    max_error = max(precision_error, covariance_error)
    vetoes = []
    if not np.isfinite(max_error) or max_error > 1.0e-8:
        vetoes.append("transform_identity_check_failed")
    return {
        "passed": not vetoes,
        "vetoes": tuple(vetoes),
        "precision_u_identity_max_abs_error": precision_error,
        "covariance_u_identity_max_abs_error": covariance_error,
        "transform_identity_max_abs_error": max_error,
        "threshold": 1.0e-8,
    }


def localization_diagnostics(
    geometry_payload: Mapping[str, Any],
    phase2_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    trust_radius = float(geometry_payload.get("settings", {}).get("low_rank_trust_radius", 0.30))
    outside_threshold = 2.0 * trust_radius
    reference = phase2_payload.get("reference", {})
    hmc = phase2_payload.get("hmc_summary", {})
    precision = np.asarray(reference.get("precision_u"), dtype=float)
    linear = np.asarray(reference.get("linear_u"), dtype=float)
    ref_mean = np.asarray(reference.get("mean_u"), dtype=float)
    pooled = np.asarray(hmc.get("pooled_mean_u"), dtype=float)
    seed_means = np.asarray(hmc.get("seed_mean_u"), dtype=float)
    points = {"reference_mean": ref_mean, "pooled_hmc_mean": pooled}
    for index, row in enumerate(seed_means):
        points[f"seed_{index}_mean"] = row
    norms = {name: float(np.linalg.norm(value)) for name, value in points.items()}
    drops = {
        name: local_quadratic_drop(value, precision, linear)
        for name, value in points.items()
    }
    outside = {
        name: value
        for name, value in norms.items()
        if name != "reference_mean" and value > outside_threshold
    }
    large_drop = {
        name: value
        for name, value in drops.items()
        if name != "reference_mean" and value > 10.0
    }
    return {
        "passed": True,
        "vetoes": (),
        "trust_radius": trust_radius,
        "outside_trust_warning_threshold": outside_threshold,
        "large_quadratic_drop_threshold": 10.0,
        "point_norms_u": norms,
        "local_quadratic_drop": drops,
        "outside_trust_region_points": outside,
        "large_quadratic_drop_points": large_drop,
        "pooled_hmc_mean_norm": norms["pooled_hmc_mean"],
    }


def local_quadratic_drop(point: np.ndarray, precision: np.ndarray, linear: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    return float(0.5 * point @ precision @ point - linear @ point)


def target_replay_diagnostics(phase2_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        module = load_geometry_module()
        settings = module.default_settings()
        target = module.build_filtering_geometry_target(settings)
        mass_contract = phase2_payload.get("precondition", {}).get("coordinate_contract", {})
        scale = np.asarray(mass_contract.get("scale"), dtype=float)
        center = np.asarray(
            load_json(DEFAULT_GEOMETRY_PATH).get("center", {}).get("free_parameter_values"),
            dtype=float,
        )
        factor = np.asarray(load_json(DEFAULT_MASS_PATH).get("mass_handoff", {}).get("factor"), dtype=float)
        points_u = {
            "center_u_zero": np.zeros(4),
            "reference_mean_u": np.asarray(phase2_payload.get("reference", {}).get("mean_u"), dtype=float),
            "pooled_hmc_mean_u": np.asarray(phase2_payload.get("hmc_summary", {}).get("pooled_mean_u"), dtype=float),
        }
        values = {}
        for name, u in points_u.items():
            z = factor @ u
            free = center + scale * z
            value, score, status = module.safe_value_and_score(
                target,
                module.tf.constant(free, dtype=module.tf.float64),
            )
            values[name] = {
                "u": u,
                "free": free,
                "value": value,
                "score_norm": None if score is None else float(np.linalg.norm(np.asarray(score, dtype=float))),
                "status": status,
            }
        center_value = values["center_u_zero"]["value"]
        pooled_value = values["pooled_hmc_mean_u"]["value"]
        return {
            "computed": True,
            "vetoes": (),
            "values": values,
            "pooled_minus_center_value": (
                None
                if center_value is None or pooled_value is None
                else float(pooled_value - center_value)
            ),
            "role": "explanatory_only",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "computed": False,
            "vetoes": (),
            "error": f"{type(exc).__name__}: {exc}",
            "role": "explanatory_only_failed_closed",
        }


def load_geometry_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_geometry_phase2r_replay",
        GEOMETRY_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load geometry module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_outcome(
    transform: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not transform.get("passed"):
        selected = "transform_bookkeeping_mismatch"
        next_subplan = "code/test repair before any new sampling"
    elif diagnostics.get("outside_trust_region_points") and diagnostics.get("large_quadratic_drop_points"):
        selected = "outside_geometry_trust_region"
        next_subplan = "draft geometry/centering repair or MAP-local reference subplan"
    elif (
        replay.get("computed")
        and replay.get("pooled_minus_center_value") is not None
        and float(replay["pooled_minus_center_value"]) > 0.0
    ):
        selected = "local_quadratic_reference_center_weak"
        next_subplan = "draft MAP/centering repair subplan"
    elif diagnostics.get("outside_trust_region_points"):
        selected = "short_chain_transient_or_multimodality_possible"
        next_subplan = "draft longer CPU-hidden chain/replication subplan"
    else:
        selected = "inconclusive_needs_longer_cpu_chain"
        next_subplan = "draft longer CPU-hidden chain/replication subplan"
    return {
        "selected_outcome": selected,
        "next_subplan": next_subplan,
        "allowed_outcomes": (
            "transform_bookkeeping_mismatch",
            "outside_geometry_trust_region",
            "local_quadratic_reference_center_weak",
            "short_chain_transient_or_multimodality_possible",
            "inconclusive_needs_longer_cpu_chain",
        ),
        "nonclaims": (
            "localization outcome only",
            "not posterior correctness evidence",
            "not HMC readiness evidence",
        ),
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
    diagnostics = payload.get("localization_diagnostics", {})
    replay = payload.get("target_replay", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2R - Localization",
        "",
        "## Decision",
        "",
        f"- phase2r_localization_passed: `{decision['phase2r_localization_passed']}`",
        f"- selected_outcome: `{decision['selected_outcome']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Diagnostics",
        "",
        f"- transform identity max abs error: `{payload.get('transform_checks', {}).get('transform_identity_max_abs_error')}`",
        f"- point norms in u: `{diagnostics.get('point_norms_u')}`",
        f"- outside trust region points: `{diagnostics.get('outside_trust_region_points')}`",
        f"- local quadratic drops: `{diagnostics.get('local_quadratic_drop')}`",
        f"- large quadratic drop points: `{diagnostics.get('large_quadratic_drop_points')}`",
        "",
        "## Target Replay",
        "",
        f"- computed: `{replay.get('computed')}`",
        f"- pooled minus center value: `{replay.get('pooled_minus_center_value')}`",
        f"- role: {replay.get('role')}",
        "",
        "## Nonclaims",
        "",
    ]
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
    parser.add_argument("--phase2-json", type=Path, default=DEFAULT_PHASE2_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--no-replay-target", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase2r_localization(
        load_json(args.geometry_json),
        load_json(args.mass_json),
        load_json(args.phase2_json),
        replay_target=not args.no_replay_target,
    )
    payload["source_artifacts"] = {
        "geometry_json": str(args.geometry_json),
        "mass_json": str(args.mass_json),
        "phase2_json": str(args.phase2_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
