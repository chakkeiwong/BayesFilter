#!/usr/bin/env python3
"""Run or independently replay the bounded Lane-B B2 sampler admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    COMPAT_DECODER_ID,
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_sampler_tf import (  # noqa: E402
    LaneBRetainedGridSampler,
    retained_sampler_workspace_estimate_bytes,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (  # noqa: E402
    tensor_sha256,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-zhao-cui-austria-sir-lane-b-b2-sampler-admission-plan-2026-07-31.md"
)
CALIBRATION_SEED = 73702
REFERENCE_SEED = 73703
SAMPLE_COUNT = 64
ROUNDTRIP_TOLERANCE = 2e-12
MASS_RESIDUAL_TOLERANCE = 5e-10
MEMORY_CAP_BYTES = 512 * 1024**2
CPU_PROCESS_CAP_BYTES = 12 * 1024**3
EXPECTED_ARTIFACT_IDENTITY = (
    "e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            return _jsonable(value.numpy().item())
        return _jsonable(value.numpy().tolist())
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> Mapping[str, str]:
    paths = (
        "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_sampler_tf.py",
        "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t2_boundary_tf.py",
        "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_artifact_compat.py",
        "scripts/run_zhao_cui_austria_sir_lane_b_b2_sampler.py",
        PLAN.as_posix(),
    )
    return {path: _sha256(ROOT / path) for path in paths}


def _serialize_tensor(path: Path, value: tf.Tensor) -> Mapping[str, object]:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    tf.io.write_file(path.as_posix(), serialized)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
        "dtype": value.dtype.name,
        "shape": value.shape.as_list(),
    }


def _read_tensor(path: Path, row: Mapping[str, object]) -> tf.Tensor:
    serialized = tf.io.read_file(path.as_posix())
    digest = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
    if digest != row.get("sha256"):
        raise ValueError("B2 sealed reference tensor hash mismatch")
    value = tf.io.parse_tensor(
        serialized, out_type=tf.dtypes.as_dtype(str(row["dtype"]))
    )
    return tf.ensure_shape(value, row["shape"])


def _run_manifest(mode: str, started: float) -> Mapping[str, object]:
    logical = tf.config.list_logical_devices("GPU")
    if logical:
        raise RuntimeError("B2 FP64 CPU reference must hide GPU devices")
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": tuple(sys.argv),
        "mode": mode,
        "environment": sys.prefix,
        "host": platform.node(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device": "explicit_cpu_hidden_fp64_reference",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "float64",
        "tf32_enabled": False,
        "jit_compile": False,
        "jit_exception": "bounded eager transport diagnostic_only",
        "gpu_memory_policy": "N/A: CUDA_VISIBLE_DEVICES=-1 before TensorFlow import",
        "data_version": "selected_T1_artifact_and_sealed_stateless_reference_v1",
        "random_seeds": {
            "calibration": CALIBRATION_SEED,
            "sealed_untouched_reference": REFERENCE_SEED,
        },
        "plan": PLAN.as_posix(),
        "result_file": "result.json",
        "source_sha256": _source_hashes(),
        "wall_time_seconds": time.monotonic() - started,
    }


def _compute(artifact_dir: Path, reference: tf.Tensor) -> Mapping[str, object]:
    artifact = load_lane_b_t1_artifact_v1_compat(artifact_dir)
    if artifact.identity.hash.value != EXPECTED_ARTIFACT_IDENTITY:
        raise ValueError("B2 source artifact identity mismatch")
    sampler = LaneBRetainedGridSampler(artifact)
    estimate = retained_sampler_workspace_estimate_bytes(
        sample_count=SAMPLE_COUNT,
        grid_size=sampler.grid_size,
        max_rank=artifact.settings.rank,
    )
    sample = sampler.inverse(reference)
    replay_reference, replay_proposal, replay_mass_residuals = (
        sampler.forward_and_log_proposal(sample.local_points)
    )
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    roundtrip_residual = tf.reduce_max(tf.abs(replay_reference - reference))
    proposal_residual = tf.reduce_max(
        tf.abs(replay_proposal - sample.proposal_log_density)
    )
    mass_replay_residual = tf.reduce_max(
        tf.abs(replay_mass_residuals - sample.raw_conditional_mass_residuals)
    )
    maximum_mass_residual = tf.reduce_max(
        sample.raw_conditional_mass_residuals
    )
    finite = tf.reduce_all(
        tf.math.is_finite(
            tf.concat(
                [
                    sample.proposal_log_density,
                    sample.target_log_density,
                    sample.correction_log_weights,
                ],
                axis=0,
            )
        )
    )
    gates = {
        "source_identity": True,
        "frame_prefix_exact": True,
        "inverse_forward_roundtrip": bool(
            (roundtrip_residual <= ROUNDTRIP_TOLERANCE).numpy()
        ),
        "proposal_equals_inversion_jacobian": bool(
            (proposal_residual <= ROUNDTRIP_TOLERANCE).numpy()
        ),
        "mass_diagnostics_replay": bool(
            (mass_replay_residual <= ROUNDTRIP_TOLERANCE).numpy()
        ),
        "raw_conditional_mass": bool(
            (maximum_mass_residual <= MASS_RESIDUAL_TOLERANCE).numpy()
        ),
        "finite_proposal_target_and_correction": bool(finite.numpy()),
        "static_workspace_under_cap": estimate <= MEMORY_CAP_BYTES,
        "cpu_process_peak_under_cap": peak_rss_bytes <= CPU_PROCESS_CAP_BYTES,
    }
    gates["passed"] = all(gates.values())
    return {
        "source_artifact_identity": artifact.identity.hash.value,
        "artifact_reload_decoder_id": COMPAT_DECODER_ID,
        "sampler": sampler.manifest_payload(),
        "sample": sample.manifest_payload(),
        "diagnostics": {
            "maximum_inverse_forward_residual": roundtrip_residual,
            "maximum_proposal_jacobian_residual": proposal_residual,
            "maximum_mass_diagnostic_replay_residual": mass_replay_residual,
            "maximum_raw_conditional_mass_residual": maximum_mass_residual,
            "roundtrip_tolerance": ROUNDTRIP_TOLERANCE,
            "mass_residual_tolerance": MASS_RESIDUAL_TOLERANCE,
            "static_workspace_estimate_bytes": estimate,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "cpu_process_cap_bytes": CPU_PROCESS_CAP_BYTES,
            "cpu_process_peak_rss_bytes": peak_rss_bytes,
        },
        "gates": gates,
    }


def _localize(artifact_dir: Path, output_dir: Path) -> Mapping[str, object]:
    """Find the first GPU axis/batch extent that produces a non-finite table."""

    started = time.monotonic()
    artifact = load_lane_b_t1_artifact_v1_compat(artifact_dir)
    sampler = LaneBRetainedGridSampler(artifact)
    rows = []
    for sample_count in (1, 2, 4, 8, 16, 32, 64):
        reference = tf.random.stateless_uniform(
            [18, sample_count],
            seed=tf.constant([REFERENCE_SEED, sample_count], tf.int32),
            minval=tf.constant(1e-6, tf.float64),
            maxval=tf.constant(1.0 - 1e-6, tf.float64),
            dtype=tf.float64,
        )
        prefixes = tf.zeros([sample_count, 0], tf.float64)
        for axis in range(18):
            try:
                local, log_q, residual = sampler._invert_axis(
                    axis, prefixes, reference[axis]
                )
                row = {
                    "sample_count": sample_count,
                    "axis": axis,
                    "status": "finite",
                    "prefix_max_abs": tf.reduce_max(tf.abs(prefixes))
                    if axis
                    else tf.constant(0.0, tf.float64),
                    "local_max_abs": tf.reduce_max(tf.abs(local)),
                    "log_q_min": tf.reduce_min(log_q),
                    "log_q_max": tf.reduce_max(log_q),
                    "mass_residual_max": tf.reduce_max(residual),
                }
                rows.append(row)
                prefixes = tf.concat([prefixes, local[:, tf.newaxis]], axis=1)
            except Exception as error:  # Diagnostic artifact records the exact exception.
                rows.append(
                    {
                        "sample_count": sample_count,
                        "axis": axis,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "prefix_sha256": tensor_sha256(prefixes),
                        "prefix_max_abs": tf.reduce_max(tf.abs(prefixes))
                        if axis
                        else tf.constant(0.0, tf.float64),
                        "reference_sha256": tensor_sha256(reference),
                    }
                )
                break
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_b2_gpu_localization.v1",
        "status": "DIAGNOSTIC_ONLY_B2_GPU_NONFINITE_LOCALIZATION",
        "source_artifact_identity": artifact.identity.hash.value,
        "rows": rows,
        "run_manifest": _run_manifest("gpu_nonfinite_localization", started),
        "nonclaims": (
            "diagnostic only",
            "no sampler admission",
            "no T2, score, T20, HMC, or production claim",
        ),
    }


def _claim(artifact_dir: Path, output_dir: Path) -> Mapping[str, object]:
    started = time.monotonic()
    generator = tf.random.Generator.from_seed(REFERENCE_SEED)
    reference = generator.uniform(
        [18, SAMPLE_COUNT],
        minval=tf.constant(1e-6, tf.float64),
        maxval=tf.constant(1.0 - 1e-6, tf.float64),
        dtype=tf.float64,
    )
    reference_row = _serialize_tensor(output_dir / "reference_uniforms.tensor", reference)
    computed = _compute(artifact_dir, reference)
    passed = bool(computed["gates"]["passed"])
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_b2_sampler.v1",
        "status": "PASS_B2_SAMPLER_CLAIM_PENDING_FRESH_REPLAY" if passed else "BLOCK_B2_SAMPLER_CLAIM",
        "mode": "claim",
        "sealed_reference_tensor": reference_row,
        **computed,
        "run_manifest": _run_manifest("claim", started),
        "decision_table": {
            "decision": "require_fresh_process_replay" if passed else "repair_B2_sampler",
            "primary_criterion_status": "pending_fresh_replay" if passed else "failed",
            "veto_diagnostic_status": "passed_current_process" if passed else "failed",
            "main_uncertainty": "finite grid conditional differs from the fitted TT density and is explicitly corrected",
            "next_justified_action": "fresh process replay of sealed reference" if passed else "localize failed sampler gate",
            "not_concluded": "no production KR, T2 value, score, T20, HMC, or scientific claim",
        },
    }


def _calibrate(artifact_dir: Path) -> Mapping[str, object]:
    """Engineering calibration on a seed disjoint from the untouched claim."""

    started = time.monotonic()
    reference = tf.random.stateless_uniform(
        [18, SAMPLE_COUNT],
        seed=tf.constant([CALIBRATION_SEED, 1], tf.int32),
        minval=tf.constant(1e-6, tf.float64),
        maxval=tf.constant(1.0 - 1e-6, tf.float64),
        dtype=tf.float64,
    )
    computed = _compute(artifact_dir, reference)
    passed = bool(computed["gates"]["passed"])
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_b2_sampler_calibration.v1",
        "status": "PASS_B2_SAMPLER_CALIBRATION" if passed else "BLOCK_B2_SAMPLER_CALIBRATION",
        "mode": "calibration",
        "calibration_reference_sha256": tensor_sha256(reference),
        **computed,
        "run_manifest": _run_manifest("calibration", started),
        "nonclaims": (
            "calibration data cannot admit the sampler",
            "no production KR, T2, score, T20, HMC, or scientific claim",
        ),
    }


def _replay(
    artifact_dir: Path, claim_dir: Path, output_dir: Path
) -> Mapping[str, object]:
    started = time.monotonic()
    claim = json.loads((claim_dir / "result.json").read_text())
    if claim.get("status") != "PASS_B2_SAMPLER_CLAIM_PENDING_FRESH_REPLAY":
        raise ValueError("B2 replay requires a passing claim pending replay")
    row = claim.get("sealed_reference_tensor")
    if not isinstance(row, Mapping):
        raise ValueError("B2 claim reference ledger missing")
    reference = _read_tensor(claim_dir / str(row["path"]), row)
    computed = _compute(artifact_dir, reference)
    claim_sample = claim.get("sample")
    replay_sample = computed["sample"]
    if not isinstance(claim_sample, Mapping):
        raise ValueError("B2 claim sample ledger missing")
    hash_fields = (
        "reference_uniforms_sha256",
        "local_points_sha256",
        "physical_points_sha256",
        "proposal_log_density_sha256",
        "target_log_density_sha256",
        "correction_log_weights_sha256",
    )
    hash_matches = {
        field: claim_sample.get(field) == replay_sample.get(field)
        for field in hash_fields
    }
    replay_passed = bool(computed["gates"]["passed"] and all(hash_matches.values()))
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_b2_sampler_replay.v1",
        "status": "PASS_B2_RETAINED_SAMPLER_ADMISSION" if replay_passed else "BLOCK_B2_RETAINED_SAMPLER_REPLAY",
        "mode": "fresh_process_replay",
        "claim_result_path": (claim_dir / "result.json").relative_to(ROOT).as_posix(),
        "claim_result_sha256": _sha256(claim_dir / "result.json"),
        "sealed_reference_tensor": row,
        **computed,
        "fresh_replay_hash_matches": hash_matches,
        "fresh_replay_passed": replay_passed,
        "run_manifest": _run_manifest("fresh_process_replay", started),
        "decision_table": {
            "decision": "open_scope_specific_T2_plan" if replay_passed else "repair_B2_sampler_replay",
            "primary_criterion_status": "passed" if replay_passed else "failed",
            "veto_diagnostic_status": "passed" if replay_passed else "failed",
            "main_uncertainty": "T1 retained shape is a finite TT approximation; B2 proves only its correctly scored finite sampler",
            "next_justified_action": "refresh and execute B4 T2 tuning plan" if replay_passed else "localize replay mismatch",
            "not_concluded": "no production KR, T2 value, score, T20, HMC, or scientific claim",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if replay_passed else "failed",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "scope-specific T2 tuning and untouched same-scalar value gate",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("calibrate", "claim", "replay", "localize"),
        required=True,
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--claim-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    if args.mode == "calibrate":
        result = _calibrate(args.artifact_dir.resolve())
    elif args.mode == "claim":
        result = _claim(args.artifact_dir.resolve(), output)
    elif args.mode == "localize":
        result = _localize(args.artifact_dir.resolve(), output)
    else:
        if args.claim_dir is None:
            raise ValueError("--claim-dir is required in replay mode")
        result = _replay(
            args.artifact_dir.resolve(), args.claim_dir.resolve(), output
        )
    _write_json(output / "result.json", result)
    if args.mode != "localize" and (
        not bool(result.get("gates", {}).get("passed")) or (
        args.mode == "replay" and not bool(result.get("fresh_replay_passed"))
        )
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
