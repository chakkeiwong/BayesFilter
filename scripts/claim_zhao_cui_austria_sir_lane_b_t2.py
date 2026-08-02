#!/usr/bin/env python3
"""Evaluate the one-shot untouched T2 value gate without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    COMPAT_DECODER_ID,
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tail_tf import (  # noqa: E402
    estimate_tail_log_normalizer,
    load_tail_cloud,
    tail_source_closure,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (  # noqa: E402
    load_lane_b_t2_artifact,
)
from scripts.select_zhao_cui_austria_sir_lane_b_t2 import (  # noqa: E402
    build_selection,
)


PLAN = Path("docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t2-plan-2026-07-31.md")
MEMORY_CAP_BYTES = 6 * 1024**3
UNTOUCHED_SCOPE = (16384, 73804, 73814)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy().item() if value.shape.rank == 0 else value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _resolve_selection(selection_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    selection = json.loads(selection_path.read_text())
    if selection.get("schema_version") != "bayesfilter.zhao_cui_austria_sir_lane_b_t2_selection.v1":
        raise ValueError("T2 claim selection schema mismatch")
    if selection.get("status") != "SELECTED_VIABLE_T2_PILOT_ARM":
        raise ValueError("T2 claim requires a viable frozen selection")
    recomputed = build_selection(selection_path.parent)
    if json.dumps(selection, sort_keys=True) != json.dumps(recomputed, sort_keys=True):
        raise ValueError("T2 claim selection is not the deterministic recomputation")
    result_path = ROOT / str(selection["selected_result_path"])
    if _sha256(result_path) != selection.get("selected_result_sha256"):
        raise ValueError("T2 selected pilot result hash mismatch")
    result = json.loads(result_path.read_text())
    if result.get("status") != "VIABLE_T2_PILOT_ARM":
        raise ValueError("T2 selected pilot is not viable")
    if result.get("artifact_identity") != selection.get("selected_artifact_identity"):
        raise ValueError("T2 selected artifact identity mismatch")
    if result.get("artifact_manifest") != selection.get("selected_artifact_manifest"):
        raise ValueError("T2 selected artifact path mismatch")
    artifact_dir = (ROOT / str(result["artifact_manifest"])).parent
    return selection, result, artifact_dir


def run_claim(
    *,
    selection_path: Path,
    parent_t1_dir: Path,
    untouched_dir: Path,
) -> Mapping[str, Any]:
    started = time.monotonic()
    selection, selected_result, artifact_dir = _resolve_selection(selection_path)
    parent = load_lane_b_t1_artifact_v1_compat(parent_t1_dir)
    artifact = load_lane_b_t2_artifact(artifact_dir, parent_artifact=parent)
    if artifact.identity.hash.value != selection["selected_artifact_identity"]:
        raise ValueError("T2 fresh reload identity differs from frozen selection")
    cloud, untouched_payload = load_tail_cloud(untouched_dir)
    expected_count, expected_reference, expected_transition = UNTOUCHED_SCOPE
    if (
        cloud.role != "untouched"
        or cloud.sample_count != expected_count
        or cloud.reference_seed != expected_reference
        or cloud.transition_seed != expected_transition
    ):
        raise ValueError("T2 untouched prepared scope mismatch")

    untouched = estimate_tail_log_normalizer(
        cloud, artifact.shift_constant
    )
    density = artifact.density()
    eager_increment = artifact.increment()
    eager_value = artifact.value()
    direct_log_mass = tf.math.log(density.normalizer())

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_scalars(
        log_weight: tf.Tensor,
        fixed_log_mass: tf.Tensor,
        shift: tf.Tensor,
        parent_value: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        maximum = tf.reduce_max(log_weight)
        audit_increment = maximum + tf.math.log(
            tf.reduce_mean(tf.exp(log_weight - maximum))
        )
        increment = fixed_log_mass - shift
        value = parent_value + increment
        return increment, value, audit_increment

    xla_increment, xla_value, xla_audit_increment = compiled_scalars(
        cloud.log_importance_weight,
        direct_log_mass,
        artifact.shift_constant,
        parent.value(),
    )
    xla_increment_residual = tf.abs(xla_increment - eager_increment)
    xla_value_residual = tf.abs(xla_value - eager_value)
    xla_audit_residual = tf.abs(xla_audit_increment - untouched.log_increment)

    value_difference = tf.abs(eager_increment - untouched.log_increment)
    value_tolerance = 3.0 * tf.sqrt(
        tf.square(artifact.calibration_estimate.log_standard_error)
        + tf.square(untouched.log_standard_error)
    ) + tf.constant(1e-6, tf.float64)
    direct_mass_residual = tf.abs(
        direct_log_mass - artifact.calibration_estimate.log_shifted_normalizer
    )
    direct_mass_tolerance = tf.constant(1e-9, tf.float64) * (
        1.0 + tf.abs(eager_increment)
    )
    cumulative_residual = tf.abs(
        eager_value - (parent.value() + eager_increment)
    )
    cumulative_tolerance = direct_mass_tolerance
    value_gate = bool((value_difference <= value_tolerance).numpy())
    mass_gate = bool((direct_mass_residual <= direct_mass_tolerance).numpy())
    cumulative_gate = bool((cumulative_residual <= cumulative_tolerance).numpy())
    xla_gate = bool(
        (
            tf.maximum(
                tf.maximum(xla_increment_residual, xla_value_residual),
                xla_audit_residual,
            )
            <= tf.constant(1e-10, tf.float64)
        ).numpy()
    )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    memory_gate = int(memory["peak"]) <= MEMORY_CAP_BYTES
    passed = value_gate and mass_gate and cumulative_gate and xla_gate and memory_gate

    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not logical:
        raise RuntimeError("T2 claim requires a logical GPU")
    source_hashes = dict(tail_source_closure())
    for path in (
        Path(__file__).resolve().relative_to(ROOT),
        Path("scripts/select_zhao_cui_austria_sir_lane_b_t2.py"),
        PLAN,
    ):
        source_hashes[path.as_posix()] = _sha256(ROOT / path)
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_claim.v1",
        "status": (
            "PASS_NEW_FIXED_VARIANT_T1_T2_VALUE_BASELINE"
            if passed
            else "BLOCK_T2_UNTOUCHED_VALUE_GATE"
        ),
        "selected_arm_id": selection["selected_arm_id"],
        "artifact_identity": artifact.identity.hash.value,
        "parent_t1_identity": parent.identity.hash.value,
        "artifact_reload_decoder_id": COMPAT_DECODER_ID,
        "artifact_increment": eager_increment,
        "artifact_cumulative_value": eager_value,
        "parent_t1_value": parent.value(),
        "untouched_log_increment": untouched.log_increment,
        "untouched_log_standard_error": untouched.log_standard_error,
        "value_difference": value_difference,
        "value_tolerance": value_tolerance,
        "direct_tt_log_mass": direct_log_mass,
        "serialized_calibration_log_mass": artifact.calibration_estimate.log_shifted_normalizer,
        "direct_mass_log_residual": direct_mass_residual,
        "direct_mass_log_tolerance": direct_mass_tolerance,
        "cumulative_value_residual": cumulative_residual,
        "cumulative_value_tolerance": cumulative_tolerance,
        "xla_tie_out": {
            "increment": xla_increment,
            "value": xla_value,
            "untouched_increment": xla_audit_increment,
            "increment_residual": xla_increment_residual,
            "value_residual": xla_value_residual,
            "untouched_increment_residual": xla_audit_residual,
        },
        "gates": {
            "fresh_t1_t2_reload_identity": True,
            "untouched_same_scalar_value": value_gate,
            "direct_tt_mass": mass_gate,
            "cumulative_value_identity": cumulative_gate,
            "gpu_xla_tie_out": xla_gate,
            "memory_under_6_gib": memory_gate,
            "passed": passed,
        },
        "untouched_manifest": cloud.manifest_payload(),
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": tuple(sys.argv),
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "device": tuple(device.name for device in logical),
            "dtype": "float64",
            "jit_compile": True,
            "gpu_memory_policy": dict(MEMORY_POLICY),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "selection_sha256": _sha256(selection_path),
            "selected_result_sha256": _sha256(ROOT / str(selection["selected_result_path"])),
            "untouched_result_sha256": _sha256(untouched_dir / "result.json"),
            "source_sha256": {path: source_hashes[path] for path in sorted(source_hashes)},
            "wall_time_seconds": time.monotonic() - started,
        },
        "decision_table": {
            "decision": "admit_T1_T2_value_and_open_score_slice" if passed else "stop_T2_value_claim",
            "primary_criterion_status": "passed" if value_gate else "failed",
            "veto_diagnostic_status": "passed" if passed else "one_or_more_vetoes_failed",
            "main_uncertainty": "independent Monte Carlo uncertainty for the untouched T2 increment",
            "next_justified_action": "three_parameter_exact_zero_slice" if passed else "do_not_reuse_untouched_data_for_tuning",
            "not_concluded": "no analytical score, T5/T10/T20, HMC, production KR, posterior, or scientific validity",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "analytical total score and then horizon ladder",
        },
        "nonclaims": (
            "no statistically supported arm ranking",
            "no score correctness",
            "no T5/T10/T20, HMC, production KR, posterior, or scientific claim",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--parent-t1-dir", type=Path, required=True)
    parser.add_argument("--untouched-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = run_claim(
            selection_path=args.selection.resolve(),
            parent_t1_dir=args.parent_t1_dir.resolve(),
            untouched_dir=args.untouched_dir.resolve(),
        )
        _write_json(output / "result.json", result)
    except Exception as exc:
        _write_json(
            output / "result.json",
            {
                "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_claim_failure.v1",
                "status": "INFRASTRUCTURE_OR_IMPLEMENTATION_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "command": tuple(sys.argv),
                "gpu_memory_policy": dict(MEMORY_POLICY),
            },
        )
        raise


if __name__ == "__main__":
    main()
