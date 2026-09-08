"""Adjudicate the v3.4 six-bank paired support/geometry diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Phase 52 report requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 52 report requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
BOUNDARY_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase52_2026_08_26.py"
FIXTURE_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase52_fixture_2026_08_26.py"
PHASE50_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase50-defensive-proposal-support/report/result.json"
PHASE51_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase51-mode-aware-proposal-geometry/report/result.json"
PHASE52_ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/"
    "phase52-fresh-paired-uncertainty-replication"
)
PHASE52_ATTEMPT_ROOT = PHASE52_ARTIFACT_ROOT / "attempt-02"

EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_VERSION = "v3.4-fresh-paired-uncertainty-replication"
EXPECTED_BOUNDARY_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_fresh_paired_boundary.v1"
EXPECTED_BOUNDARY_STATUS = "PASS_V3_4_FRESH_PAIRED_BOUNDARY"
EXPECTED_FIXTURE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_fresh_paired_fixture.v1"
EXPECTED_FIXTURE_STATUS = "PASS_V3_4_FRESH_PAIRED_FIXTURE"
EXPECTED_ARM_STATUS = "PASS_V3_4_MUTATION_ARM"
EXPECTED_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
EXPECTED_PILOT_RUNNER_RECEIPT_SHA256 = "c0b793ab10bd8d69cec22347c3beba00b5dd15e77e129f61b25d8dc585b9b703"
EXPECTED_PILOT_RUNNER_CURRENT_SHA256 = "e06845ee3f16773f181380c35297beaa2c4a489561c4b7d642c89853bb8ace1b"
EXPECTED_PILOT_RUNNER_EQUIVALENCE = "one_trailing_blank_line_only_verified_2026_08_28"
EXPECTED_STEPS = 8
EXPECTED_SUPPORT_RHO = 0.50
EXPECTED_SUPPORT_STD = 4.0
EXPECTED_GEOMETRY_RHO = 0.50
EXPECTED_GEOMETRY_SCALE = 2.0
REPLICATE_COUNT = 6
EXPECTED_FRESH_SEEDS = tuple((20260826, 5201 + index) for index in range(REPLICATE_COUNT))
EXPECTED_PILOT_ROOT_SEEDS = tuple((20260826, 5101 + index) for index in range(REPLICATE_COUNT))
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = (20260826, 52052)
PHASE50_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_defensive_support_report.v1"
PHASE50_STATUS = "PASS_V3_2_DEFENSIVE_SUPPORT_REPORT"
PHASE50_VERSION = "v3.2-defensive-proposal-support"
PHASE50_BRANCH = "support_broadened_does_not_reduce_variability"
PHASE51_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_mode_aware_geometry_report.v1"
PHASE51_STATUS = "PASS_V3_3_MODE_AWARE_GEOMETRY_REPORT"
PHASE51_VERSION = "v3.3-mode-aware-proposal-geometry"
PHASE51_BRANCH = "mode_aware_geometry_reduces_between_bank_variability_descriptive"
METRICS = (
    "theta_mean_0",
    "covariance_offdiag_max_abs",
    "negative_mode_fraction",
    "root_count",
    "weighted_ess_fraction",
)
PRIMARY_METRICS = (
    "theta_mean_0",
    "negative_mode_fraction",
    "covariance_offdiag_max_abs",
)


class Phase52ReportError(RuntimeError):
    """Raised when a v3.4 report input violates the evidence contract."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(item) for item in value.as_list()]
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase52ReportError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(root: Path, name: str = "result.json") -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase52ReportError(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase52ReportError(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _summary_fields(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        raise Phase52ReportError("summary is not a mapping")
    for key in METRICS:
        if key not in summary or not _finite(summary[key]):
            raise Phase52ReportError(f"non-finite or missing summary field: {key}")
    return {key: summary[key] for key in METRICS}


def _validate_tensor_receipt(receipt: Mapping[str, Any], label: str) -> None:
    if not isinstance(receipt, Mapping):
        raise Phase52ReportError(f"{label} tensor receipt is not a mapping")
    path_value = receipt.get("path")
    if not isinstance(path_value, str):
        raise Phase52ReportError(f"{label} tensor receipt has no path")
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise Phase52ReportError(f"{label} tensor path escapes the repository") from error
    if (
        not path.is_file()
        or receipt.get("sha256") != _sha(path)
        or int(receipt.get("bytes", -1)) != path.stat().st_size
    ):
        raise Phase52ReportError(f"{label} tensor receipt does not match its file")


def _load_frozen_reports() -> tuple[Path, Mapping[str, Any], Path, Mapping[str, Any]]:
    if not PHASE50_REPORT.is_file() or not PHASE51_REPORT.is_file():
        raise Phase52ReportError("a frozen Phase 50/51 report is missing")
    phase50 = json.loads(PHASE50_REPORT.read_text(encoding="utf-8"))
    phase51 = json.loads(PHASE51_REPORT.read_text(encoding="utf-8"))
    if (
        phase50.get("schema") != PHASE50_SCHEMA
        or phase50.get("status") != PHASE50_STATUS
        or phase50.get("plan_version") != PHASE50_VERSION
        or phase50.get("target_signature") != EXPECTED_TARGET
        or phase50.get("branch") != PHASE50_BRANCH
        or not isinstance(phase50.get("support_spreads"), Mapping)
    ):
        raise Phase52ReportError("frozen Phase 50 report is stale or incomplete")
    if (
        phase51.get("schema") != PHASE51_SCHEMA
        or phase51.get("status") != PHASE51_STATUS
        or phase51.get("plan_version") != PHASE51_VERSION
        or phase51.get("target_signature") != EXPECTED_TARGET
        or phase51.get("branch") != PHASE51_BRANCH
        or not isinstance(phase51.get("geometry_spreads"), Mapping)
    ):
        raise Phase52ReportError("frozen Phase 51 report is stale or incomplete")
    return PHASE50_REPORT, phase50, PHASE51_REPORT, phase51


def _arm_summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = arm.get("gates") if isinstance(arm, Mapping) else None
    if not isinstance(gates, Mapping) or not gates or not all(gates.values()):
        raise Phase52ReportError("arm gates are absent or not passing")
    return _summary_fields(arm.get("final_summary", {}))


def _spread(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def _mutation_receipt(arm: Mapping[str, Any], expected_kind: str) -> Mapping[str, Any]:
    stages = arm.get("stages")
    if not isinstance(stages, list) or len(stages) != 5:
        raise Phase52ReportError("expected five annealing stage rows")
    active_rows: list[Mapping[str, Any]] = []
    for stage in stages:
        if not _finite(stage):
            raise Phase52ReportError("non-finite stage diagnostic")
        mutation = stage.get("mutation")
        if not isinstance(mutation, Mapping):
            raise Phase52ReportError("stage has no mutation receipt")
        if int(mutation.get("accepted_invalid_count", 0)) != 0:
            raise Phase52ReportError("an invalid proposal was accepted")
        steps = int(mutation.get("steps", 0))
        if expected_kind == "none":
            if steps != 0 or mutation.get("proposal_kind") != "none":
                raise Phase52ReportError("identity arm unexpectedly mutates")
        elif steps > 0:
            if (
                mutation.get("kernel") != "independent_mh"
                or mutation.get("proposal_kind") != expected_kind
                or steps != EXPECTED_STEPS
            ):
                raise Phase52ReportError(f"{expected_kind} arm has the wrong mutation contract")
            fraction = mutation.get("candidate_component_fraction")
            if not _finite(fraction) or not 0.0 <= float(fraction) <= 1.0:
                raise Phase52ReportError(f"{expected_kind} arm has an invalid component fraction")
            active_rows.append(mutation)
        elif mutation.get("proposal_kind") != expected_kind:
            raise Phase52ReportError(f"{expected_kind} terminal mutation label is stale")
    if expected_kind != "none" and len(active_rows) != 4:
        raise Phase52ReportError(f"{expected_kind} arm does not have four active stages")
    denominator = max(1, len(active_rows))
    return {
        "proposal_kind": expected_kind,
        "active_stage_count": len(active_rows),
        "acceptance_rate_mean": sum(float(row["acceptance_rate"]) for row in active_rows) / denominator,
        "move_fraction_mean": sum(float(row["move_fraction"]) for row in active_rows) / denominator,
        "candidate_component_fraction_mean": sum(
            float(row.get("candidate_component_fraction", 0.0)) for row in active_rows
        )
        / denominator,
        "candidate_safe_fraction_mean": sum(
            float(row.get("candidate_safe_fraction", 0.0)) for row in active_rows
        )
        / denominator,
        "mean_displacement_mean": sum(float(row["mean_displacement"]) for row in active_rows) / denominator,
        "invalid_candidate_count": sum(int(row["invalid_candidate_count"]) for row in active_rows),
        "accepted_invalid_count": sum(int(row["accepted_invalid_count"]) for row in active_rows),
        "stage_rows": active_rows,
    }


def _bootstrap_spread_difference(
    geometry: list[float],
    support: list[float],
    indices: tf.Tensor,
) -> Mapping[str, Any]:
    geometry_tensor = tf.constant(geometry, tf.float64)
    support_tensor = tf.constant(support, tf.float64)
    geometry_draws = tf.gather(geometry_tensor, indices)
    support_draws = tf.gather(support_tensor, indices)
    differences = (
        tf.reduce_max(geometry_draws, axis=1)
        - tf.reduce_min(geometry_draws, axis=1)
        - tf.reduce_max(support_draws, axis=1)
        + tf.reduce_min(support_draws, axis=1)
    )
    ordered = tf.sort(differences)
    lower_index = int(math.floor(0.025 * (BOOTSTRAP_REPS - 1)))
    upper_index = int(math.ceil(0.975 * BOOTSTRAP_REPS) - 1)
    point = (max(geometry) - min(geometry)) - (max(support) - min(support))
    lower = float(ordered[lower_index].numpy())
    upper = float(ordered[upper_index].numpy())
    return {
        "estimand": "range_geometry_minus_range_support",
        "point_estimate": point,
        "bootstrap_mean": float(tf.reduce_mean(differences).numpy()),
        "bootstrap_standard_deviation": float(tf.math.reduce_std(differences).numpy()),
        "percentile_95_lower": lower,
        "percentile_95_upper": upper,
        "upper_bound_nonpositive": upper <= 0.0,
        "bootstrap_nonpositive_fraction": float(
            tf.reduce_mean(tf.cast(differences <= 0.0, tf.float64)).numpy()
        ),
    }


def _sign_counts(values: list[float]) -> Mapping[str, int]:
    return {
        "geometry_minus_support_negative": sum(value < 0.0 for value in values),
        "geometry_minus_support_zero": sum(value == 0.0 for value in values),
        "geometry_minus_support_positive": sum(value > 0.0 for value in values),
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# v3.4 Six-Bank Paired Proposal Report",
        "",
        f"Status: `{result['status']}`",
        f"Branch: `{result['branch']}`",
        "",
        "| Replicate | Support mean0 | Geometry mean0 | Support negative mass | Geometry negative mass | Support ESS | Geometry ESS |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["replicate_rows"]:
        support = row["isotropic_support_mh"]
        geometry = row["mode_aware_geometry_mh"]
        lines.append(
            f"| {row['replicate']} | {float(support['theta_mean_0']):.6f} | "
            f"{float(geometry['theta_mean_0']):.6f} | "
            f"{float(support['negative_mode_fraction']):.6f} | "
            f"{float(geometry['negative_mode_fraction']):.6f} | "
            f"{float(support['weighted_ess_fraction']):.6f} | "
            f"{float(geometry['weighted_ess_fraction']):.6f} |"
        )
    lines.extend(
        [
            "",
            "| Metric | Support spread | Geometry spread | Geometry - support | 95% lower | 95% upper | Upper <= 0? |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for metric in METRICS:
        interval = result["bootstrap_intervals"][metric]
        lines.append(
            f"| `{metric}` | {float(result['support_spreads'][metric]):.6f} | "
            f"{float(result['geometry_spreads'][metric]):.6f} | "
            f"{float(interval['point_estimate']):.6f} | "
            f"{float(interval['percentile_95_lower']):.6f} | "
            f"{float(interval['percentile_95_upper']):.6f} | "
            f"{interval['upper_bound_nonpositive']} |"
        )
    lines.extend(
        [
            "",
            "The intervals describe this six-bank replication only. They do not establish a population ranking, posterior correctness, IID whitening, HMC readiness, or default readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--boundary-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = (args.fixture_root, args.boundary_root, args.output_root)
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise Phase52ReportError("all paths must be repository-relative")
    expected_paths = (
        PHASE52_ARTIFACT_ROOT / "fixture",
        PHASE52_ATTEMPT_ROOT / "q20-paired",
        PHASE52_ATTEMPT_ROOT / "report",
    )
    if paths != expected_paths:
        raise Phase52ReportError("report paths do not match the predeclared Phase 52 artifact roots")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase52ReportError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    fixture_path, fixture = _load(args.fixture_root)
    boundary_path, boundary = _load(args.boundary_root)
    phase50_path, phase50, phase51_path, phase51 = _load_frozen_reports()
    fixture_manifest = fixture.get("run_manifest", {})
    if (
        fixture.get("schema") != EXPECTED_FIXTURE_SCHEMA
        or fixture.get("status") != EXPECTED_FIXTURE_STATUS
        or fixture.get("plan_version") != EXPECTED_VERSION
        or fixture.get("seed") != [20260826, 5200]
        or fixture.get("depth_steps") != EXPECTED_STEPS
        or float(fixture.get("support_rho", -1.0)) != EXPECTED_SUPPORT_RHO
        or float(fixture.get("support_std", -1.0)) != EXPECTED_SUPPORT_STD
        or float(fixture.get("geometry_rho", -1.0)) != EXPECTED_GEOMETRY_RHO
        or float(fixture.get("geometry_scale", -1.0)) != EXPECTED_GEOMETRY_SCALE
        or not all(fixture.get("gates", {}).values())
        or fixture_manifest.get("source_sha256", {}).get("runner") != _sha(FIXTURE_RUNNER)
        or fixture_manifest.get("source_sha256", {}).get("plan") != _sha(PLAN)
        or fixture_manifest.get("cuda_visible_devices") != "-1"
        or fixture_manifest.get("jit_compile") is not True
    ):
        raise Phase52ReportError("support/geometry algebra fixture is not passing")
    device = boundary.get("device", {})
    run_manifest = boundary.get("run_manifest", {})
    if (
        boundary.get("schema") != EXPECTED_BOUNDARY_SCHEMA
        or boundary.get("status") != EXPECTED_BOUNDARY_STATUS
        or boundary.get("plan_version") != EXPECTED_VERSION
        or boundary.get("target_signature") != EXPECTED_TARGET
        or boundary.get("measure") != "theta_R4"
        or int(boundary.get("replicate_count", -1)) != REPLICATE_COUNT
        or int(boundary.get("mh_steps", -1)) != EXPECTED_STEPS
        or float(boundary.get("support_rho", -1.0)) != EXPECTED_SUPPORT_RHO
        or float(boundary.get("support_std", -1.0)) != EXPECTED_SUPPORT_STD
        or float(boundary.get("geometry_rho", -1.0)) != EXPECTED_GEOMETRY_RHO
        or float(boundary.get("geometry_scale", -1.0)) != EXPECTED_GEOMETRY_SCALE
        or boundary.get("phase50_report_sha256") != _sha(phase50_path)
        or boundary.get("phase51_report_sha256") != _sha(phase51_path)
        or boundary.get("pilot_receipts_distinct") is not True
        or boundary.get("pilot_roots_match_fresh_namespace") is not True
        or boundary.get("pilot_seeds_match_fresh_ledger") is not True
        or boundary.get("pilot_runner_sha256") != EXPECTED_PILOT_RUNNER_RECEIPT_SHA256
        or boundary.get("pilot_runner_current_sha256") != EXPECTED_PILOT_RUNNER_CURRENT_SHA256
        or boundary.get("pilot_runner_equivalence") != EXPECTED_PILOT_RUNNER_EQUIVALENCE
        or boundary.get("fresh_rows_used_for_training") is not False
        or boundary.get("fresh_rows_used_for_selection") is not False
        or boundary.get("hmc_launched") is not False
        or not device.get("physical_devices")
        or not device.get("logical_devices")
        or device.get("tf32_enabled") is not True
        or device.get("jit_compile_target") is not True
        or device.get("jit_compile_mutation") is not True
        or device.get("trust_basis") != EXPECTED_TRUST_BASIS
        or run_manifest.get("gpu_memory_growth_verified") is not True
        or run_manifest.get("tf_force_gpu_allow_growth") != "true"
        or run_manifest.get("jit_compile") is not True
        or run_manifest.get("trust_basis") != EXPECTED_TRUST_BASIS
        or run_manifest.get("source_sha256", {}).get("runner") != _sha(BOUNDARY_RUNNER)
        or run_manifest.get("source_sha256", {}).get("plan") != _sha(PLAN)
        or run_manifest.get("source_sha256", {}).get("corrected_pilot_runner")
        != EXPECTED_PILOT_RUNNER_CURRENT_SHA256
        or run_manifest.get("source_sha256", {}).get("corrected_pilot_runner_receipt")
        != EXPECTED_PILOT_RUNNER_RECEIPT_SHA256
        or run_manifest.get("source_sha256", {}).get("fixture") != _sha(fixture_path)
        or run_manifest.get("source_sha256", {}).get("phase50_report") != _sha(phase50_path)
        or run_manifest.get("source_sha256", {}).get("phase51_report") != _sha(phase51_path)
    ):
        raise Phase52ReportError("q=20 boundary is not passing or its provenance is stale")
    replicates = boundary.get("replicates")
    if not isinstance(replicates, list) or len(replicates) != REPLICATE_COUNT:
        raise Phase52ReportError("expected exactly six paired boundary replicates")
    rows: list[Mapping[str, Any]] = []
    pilot_hashes: set[str] = set()
    for index, replicate in enumerate(replicates, start=1):
        expected_root = PHASE52_ATTEMPT_ROOT / f"pilot-{index:02d}"
        if (
            replicate.get("replicate") != index
            or replicate.get("pilot_root") != expected_root.as_posix()
            or tuple(replicate.get("pilot_m0_seed", ())) != EXPECTED_FRESH_SEEDS[index - 1]
            or tuple(replicate.get("pilot_root_seed", ())) != EXPECTED_PILOT_ROOT_SEEDS[index - 1]
        ):
            raise Phase52ReportError(f"replicate {index} root or seed is not the predeclared fresh input")
        pilot_hash = replicate.get("pilot_sha256")
        if not isinstance(pilot_hash, str) or len(pilot_hash) != 64:
            raise Phase52ReportError(f"replicate {index} pilot hash is absent")
        actual_pilot_path = ROOT / expected_root / "pilot.json"
        if not actual_pilot_path.is_file() or _sha(actual_pilot_path) != pilot_hash:
            raise Phase52ReportError(f"replicate {index} pilot hash does not match its fresh receipt")
        pilot_hashes.add(pilot_hash)
        paired = replicate.get("paired", {})
        initial_tensors = replicate.get("initial_tensors", {})
        initial_hash = initial_tensors.get("theta", {}).get("sha256")
        required_initial_tensors = (
            "theta",
            "proposal_q",
            "proposal_support",
            "proposal_geometry",
            "target",
            "roots",
            "components",
        )
        if any(not initial_tensors.get(key, {}).get("sha256") for key in required_initial_tensors):
            raise Phase52ReportError(f"replicate {index} lacks initial tensor provenance")
        for key in required_initial_tensors:
            _validate_tensor_receipt(initial_tensors[key], f"replicate {index} initial {key}")
        if (
            paired.get("same_initial_cloud") is not True
            or paired.get("same_resampling_seeds") is not True
            or paired.get("initial_tensor_hash") != initial_hash
            or paired.get("identity_initial_tensor_hash") != initial_hash
            or paired.get("isotropic_support_initial_tensor_hash") != initial_hash
            or paired.get("mode_aware_geometry_initial_tensor_hash") != initial_hash
        ):
            raise Phase52ReportError(f"replicate {index} pairing gate failed")
        identity = replicate.get("identity", {})
        support = replicate.get("isotropic_support_mh", {})
        geometry = replicate.get("mode_aware_geometry_mh", {})
        for arm, kind in ((identity, "none"), (support, "support"), (geometry, "geometry")):
            if (
                arm.get("status") != EXPECTED_ARM_STATUS
                or arm.get("proposal_kind") != kind
                or arm.get("initial_tensor_hash") != initial_hash
                or not _finite(arm)
            ):
                raise Phase52ReportError(f"replicate {index} {kind} arm is invalid")
            final_tensors = arm.get("final_tensors", {})
            for key in ("final_theta", "final_roots", "final_weights"):
                _validate_tensor_receipt(final_tensors.get(key, {}), f"replicate {index} {kind} {key}")
        if not (identity.get("resampling_seeds") == support.get("resampling_seeds") == geometry.get("resampling_seeds")):
            raise Phase52ReportError(f"replicate {index} resampling seed streams differ")
        rows.append(
            {
                "replicate": index,
                "pilot_root": replicate["pilot_root"],
                "pilot_sha256": pilot_hash,
                "pilot_m0_seed": list(EXPECTED_FRESH_SEEDS[index - 1]),
                "pilot_root_seed": list(EXPECTED_PILOT_ROOT_SEEDS[index - 1]),
                "identity": _arm_summary(identity),
                "isotropic_support_mh": _arm_summary(support),
                "mode_aware_geometry_mh": _arm_summary(geometry),
                "identity_mutation": _mutation_receipt(identity, "none"),
                "isotropic_support_mh_mutation": _mutation_receipt(support, "support"),
                "mode_aware_geometry_mh_mutation": _mutation_receipt(geometry, "geometry"),
            }
        )
    if len(pilot_hashes) != REPLICATE_COUNT:
        raise Phase52ReportError("fresh pilot receipt hashes are not distinct")
    identity_spreads = {key: _spread([row["identity"] for row in rows], key) for key in METRICS}
    support_spreads = {key: _spread([row["isotropic_support_mh"] for row in rows], key) for key in METRICS}
    geometry_spreads = {key: _spread([row["mode_aware_geometry_mh"] for row in rows], key) for key in METRICS}
    bootstrap_indices = tf.random.stateless_uniform(
        (BOOTSTRAP_REPS, REPLICATE_COUNT),
        seed=tf.constant(BOOTSTRAP_SEED, tf.int32),
        minval=0,
        maxval=REPLICATE_COUNT,
        dtype=tf.int32,
    )
    bootstrap_intervals: dict[str, Mapping[str, Any]] = {}
    paired_differences: dict[str, Mapping[str, Any]] = {}
    for metric in METRICS:
        support_values = [float(row["isotropic_support_mh"][metric]) for row in rows]
        geometry_values = [float(row["mode_aware_geometry_mh"][metric]) for row in rows]
        differences = [geometry - support for geometry, support in zip(geometry_values, support_values)]
        bootstrap_intervals[metric] = _bootstrap_spread_difference(
            geometry_values, support_values, bootstrap_indices
        )
        paired_differences[metric] = {
            "geometry_minus_support": differences,
            "sign_counts": _sign_counts(differences),
        }
    branch = (
        "fresh_geometry_uncertainty_compatible"
        if all(bootstrap_intervals[key]["upper_bound_nonpositive"] for key in PRIMARY_METRICS)
        else "fresh_geometry_uncertainty_incompatible"
    )
    candidate_decision = (
        "retain_role_limited_geometry_nominee_for_larger_validation"
        if branch == "fresh_geometry_uncertainty_compatible"
        else "do_not_retain_the_frozen_geometry_as_a_spread_reduction_nominee"
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_fresh_paired_report.v1",
        "status": "PASS_V3_4_FRESH_PAIRED_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_six_bank_paired_uncertainty_adjudication",
        "target_signature": EXPECTED_TARGET,
        "fixture_status": fixture["status"],
        "boundary_status": boundary["status"],
        "branch": branch,
        "branch_is_population_or_superiority_ranking": False,
        "replicate_count": REPLICATE_COUNT,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": list(BOOTSTRAP_SEED),
        "bootstrap_scope": "paired percentile diagnostic for the observed six-bank design",
        "replicate_rows": rows,
        "identity_spreads": identity_spreads,
        "support_spreads": support_spreads,
        "geometry_spreads": geometry_spreads,
        "bootstrap_intervals": bootstrap_intervals,
        "paired_differences": paired_differences,
        "primary_metrics": list(PRIMARY_METRICS),
        "historical_context_not_pooled": {
            "phase50_status": phase50["status"],
            "phase50_branch": phase50["branch"],
            "phase50_support_spreads": phase50["support_spreads"],
            "phase51_status": phase51["status"],
            "phase51_branch": phase51["branch"],
            "phase51_geometry_spreads": phase51["geometry_spreads"],
        },
        "decision_table": [
            {
                "decision": "retain_theta_target_and_three_arm_harness",
                "primary_criterion_status": "pass",
                "veto_diagnostic_status": "none fired",
                "main_uncertainty": "finite target/harness evidence does not establish posterior correctness",
                "next_justified_action": "retain as a role-limited diagnostic harness",
                "not_concluded": "posterior correctness",
            },
            {
                "decision": "retain_frozen_geometry_candidate",
                "primary_criterion_status": branch,
                "veto_diagnostic_status": "no engineering veto; scientific promotion remains vetoed",
                "main_uncertainty": "six-bank percentile bootstrap is finite-design evidence only",
                "next_justified_action": candidate_decision,
                "not_concluded": "population ranking, superiority, or default readiness",
            },
            {
                "decision": "promote_IID_whitening_or_HMC",
                "primary_criterion_status": "veto",
                "veto_diagnostic_status": "no Gaussian-law, posterior-agreement, or HMC convergence evidence",
                "main_uncertainty": "proposal-cloud moments are not a whitening theorem",
                "next_justified_action": "keep NeuTra whitening and HMC closed",
                "not_concluded": "IID Gaussian law or HMC readiness",
            },
        ],
        "inference_status": {
            "hard_veto_screen": "passed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "six paired raw rows, ranges, sign counts, and finite-design bootstrap intervals",
            "default_readiness": "not_ready",
            "next_evidence_needed": candidate_decision,
        },
        "red_team": {
            "strongest_alternative_explanation": "any favorable range relation may be driven by six sampled banks, inherited local mode representatives, or an ESS tradeoff rather than globally faithful geometry",
            "overturning_evidence": "larger fresh paired replication plus downstream target/posterior agreement under a separately reviewed protocol",
            "weakest_evidence": "the range estimand is tail-sensitive and six banks cannot support a stable population claim",
        },
        "nonclaims": [
            "No population ranking or method superiority is inferred from the paired bootstrap.",
            "No finite-run convergence, posterior correctness, IID whitening, exhaustive mode discovery, HMC, canonical LEDH, or default claim.",
            "Acceptance, ESS, roots, mode mass, and range differences remain diagnostic quantities.",
            "No Phase 50/51 row was pooled into the fresh six-bank estimand.",
            "No fresh row was used to train NeuTra or tune the frozen proposal laws.",
        ],
        "sources": {
            "fixture_root": args.fixture_root,
            "boundary_root": args.boundary_root,
            "fixture_sha256": _sha(fixture_path),
            "boundary_sha256": _sha(boundary_path),
            "phase50_report": phase50_path,
            "phase50_report_sha256": _sha(phase50_path),
            "phase51_report": phase51_path,
            "phase51_report_sha256": _sha(phase51_path),
        },
        "run_manifest": {
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "gpu_hidden_intentionally": True,
            "jit_compile": False,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": list(BOOTSTRAP_SEED),
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "phase50_report": _sha(phase50_path),
                "phase51_report": _sha(phase51_path),
            },
        },
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(
        json.dumps(
            {"status": result["status"], "branch": branch, "output_root": args.output_root.as_posix()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
