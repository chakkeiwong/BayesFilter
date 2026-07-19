#!/usr/bin/env python3
"""Exact 250-draw continuation of the SSL-LSTM A4 repair-02 kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_HARNESS_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py"
)
HARNESS_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_continuation_01_2026_07_14.py"
)
TEST_PATH = Path("tests/test_ssl_lstm_a4_hmc_repair_02_continuation_01.py")
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-repair-02-"
    "continuation-01-plan-2026-07-14.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-repair-02-"
    "continuation-01-result-2026-07-14.md"
)
REPAIR02_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
    "hmc-acquisition/repair-02"
)
REPAIR02_PRIVATE = REPAIR02_ROOT / "private"
ADAPTATION_RECEIPT = REPAIR02_ROOT / "adaptation.json"
SEGMENT0_RECEIPT = REPAIR02_ROOT / "segment-0.json"
SEGMENT0_LABEL = "repair_02_segment_0"
CONTINUATION_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
    "hmc-acquisition/repair-02-continuation-01"
)
CONTINUATION_PRIVATE = CONTINUATION_ROOT / "private"
OUTPUT_PATH = CONTINUATION_ROOT / "segment-1.json"
CONTINUATION_LABEL = "repair_02_continuation_01_segment_1"
CONTINUATION_SEED = (20260714, 1640)
CONTINUATION_DRAWS = 250
CONTINUATION_BURNIN = 0
FROZEN_STEP_SIZE = 0.37613058552609946
NUM_LEAPFROG_STEPS = 4
TRAJECTORY_LENGTH = FROZEN_STEP_SIZE * NUM_LEAPFROG_STEPS
PRIOR_GPU_SECONDS = 2040.799946242012
PROJECTED_SECONDS = 1800.0

INPUT_HASHES = (
    (
        ADAPTATION_RECEIPT,
        "97df6a564171deaeb101d20e5d81f93139d3294982519ce3781114bbbfbc2d7d",
    ),
    (
        SEGMENT0_RECEIPT,
        "58e3d9c19ae82450539ce4a16f98e63bb409a630beae0e8a4da5c16703d4c9e3",
    ),
    (
        REPAIR02_PRIVATE / "repair_02_segment_0_private_manifest.json",
        "0a48a13852046ad5ae888ec369e0c107678a61f2ff600ba434a6af45a659d673",
    ),
    (
        REPAIR02_PRIVATE / "repair_02_segment_0_retained_samples.tftensor",
        "14255bb3f15897eadccd84a1f295d69d6bbebee74689d0e46c0ecf499e76d43e",
    ),
    (
        REPAIR02_PRIVATE / "repair_02_segment_0_final_state.tftensor",
        "b0df9f30dee43e4b0fe7e545226e5c5a36a4c2a6e387b1d9b7cb01d45d7e38bf",
    ),
)


def _load_base() -> ModuleType:
    path = ROOT / BASE_HARNESS_PATH
    spec = importlib.util.spec_from_file_location("ssl_lstm_a4_cont01_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load A4 acquisition harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PLAN_PATH = PLAN_PATH
    module.RESULT_PATH = RESULT_PATH
    return module


base = _load_base()
ContinuationError = base.AcquisitionError


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _file_row(path: Path, role: str) -> dict[str, Any]:
    absolute = ROOT / path
    return {
        "path": path.as_posix(),
        "role": role,
        "bytes": absolute.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_bindings() -> list[dict[str, Any]]:
    return [
        _file_row(PLAN_PATH, "prospective_exact_continuation_contract"),
        _file_row(HARNESS_PATH, "continuation_harness"),
        _file_row(TEST_PATH, "focused_continuation_tests"),
        _file_row(BASE_HARNESS_PATH, "reviewed_acquisition_authority"),
        _file_row(base.A0_LOCK_PATH, "locked_sampler_geometry"),
        _file_row(
            Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
            "locked_a1_target",
        ),
        _file_row(Path("bayesfilter/inference/hmc.py"), "retained_hmc_runtime"),
        _file_row(
            Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
            "rank_normalized_admission_diagnostics",
        ),
    ]


def _assert_recorded_sources_current(payload: Mapping[str, Any], path: Path) -> None:
    rows = payload.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise ContinuationError(f"missing source bindings: {path}")
    for row in rows:
        source_path = Path(str(row.get("path", "")))
        absolute = ROOT / source_path
        if not absolute.is_file():
            raise ContinuationError(f"recorded source missing: {source_path}")
        if (
            int(row.get("bytes", -1)) != absolute.stat().st_size
            or row.get("sha256") != _sha256(source_path)
        ):
            raise ContinuationError(f"recorded source binding drift: {source_path}")


def _trusted_wall_time(payload: Mapping[str, Any], path: Path) -> float:
    manifest = payload.get("run_manifest", {})
    if (
        manifest.get("trust_basis")
        != "owner_designated_managed_session_visible_gpu_trusted"
        or manifest.get("cpu_gpu_status") != "trusted_gpu_xla"
        or manifest.get("data_version") != base.TARGET_SEMANTIC_SHA256
    ):
        raise ContinuationError(f"input is not trusted locked-target GPU evidence: {path}")
    wall_time = float(manifest.get("wall_time_seconds", math.nan))
    if not math.isfinite(wall_time) or wall_time < 0.0:
        raise ContinuationError(f"invalid trusted GPU wall time: {path}")
    return wall_time


def _budget_lineage_paths(segment: Mapping[str, Any]) -> tuple[Path, ...]:
    return tuple(Path(str(path)) for path in segment.get("budget_lineage_artifacts", ()))


def validate_handoff() -> tuple[dict[str, Any], Any, Any, dict[str, Any], float]:
    import tensorflow as tf

    for path, expected_hash in INPUT_HASHES:
        if not (ROOT / path).is_file():
            raise ContinuationError(f"missing continuation input: {path}")
        if _sha256(path) != expected_hash:
            raise ContinuationError(f"continuation input SHA-256 drift: {path}")

    adaptation = base._strict_load(ADAPTATION_RECEIPT)
    segment = base._strict_load(SEGMENT0_RECEIPT)
    if adaptation.get("status") != "SELECTED":
        raise ContinuationError("repair-02 adaptation is no longer selected")
    if (
        segment.get("status") != "NOT_ADMITTED"
        or segment.get("admission_diagnostics", {}).get("decision")
        != "PROMOTION_VETO_EXTEND_IF_BUDGET_ALLOWS"
        or segment.get("admission_diagnostics", {}).get("hard_vetoes")
    ):
        raise ContinuationError("repair-02 segment 0 is not eligible for exact extension")
    frozen = segment.get("frozen_kernel", {})
    if (
        float(frozen.get("step_size", math.nan)) != FROZEN_STEP_SIZE
        or int(frozen.get("num_leapfrog_steps", -1)) != NUM_LEAPFROG_STEPS
        or float(frozen.get("trajectory_length", math.nan)) != TRAJECTORY_LENGTH
        or frozen.get("source") != ADAPTATION_RECEIPT.as_posix()
    ):
        raise ContinuationError("repair-02 frozen-kernel identity drift")
    if segment.get("initial_state_policy") != "exact_repair_02_adaptation_screen_final_state":
        raise ContinuationError("repair-02 segment-0 initial-state lineage drift")
    _assert_recorded_sources_current(adaptation, ADAPTATION_RECEIPT)
    _assert_recorded_sources_current(segment, SEGMENT0_RECEIPT)

    lineage = _budget_lineage_paths(segment)
    if not lineage or lineage[-1] != ADAPTATION_RECEIPT:
        raise ContinuationError("repair-02 segment budget lineage drift")
    total = 0.0
    for path in (*lineage, SEGMENT0_RECEIPT):
        total += _trusted_wall_time(base._strict_load(path), path)
    if not math.isclose(total, PRIOR_GPU_SECONDS, rel_tol=0.0, abs_tol=1.0e-9):
        raise ContinuationError(
            f"prior GPU budget mismatch: observed {total}, expected {PRIOR_GPU_SECONDS}"
        )

    old_samples, final_state, private_manifest = base._read_archive(
        REPAIR02_PRIVATE, SEGMENT0_LABEL
    )
    if tuple(old_samples.shape) != (250, 4, 4) or tuple(final_state.shape) != (4, 4):
        raise ContinuationError("repair-02 private handoff shape drift")
    if not bool(
        tf.reduce_all(tf.math.is_finite(old_samples)).numpy()
        and tf.reduce_all(tf.math.is_finite(final_state)).numpy()
    ):
        raise ContinuationError("repair-02 private handoff is nonfinite")
    sample_row = private_manifest.get("sample_shards", [{}])[0]
    state_row = private_manifest.get("sidecars", {}).get("final_state", {})
    if (
        sample_row.get("sha256") != INPUT_HASHES[3][1]
        or state_row.get("sha256") != INPUT_HASHES[4][1]
    ):
        raise ContinuationError("repair-02 private shard manifest binding drift")
    return segment, old_samples, final_state, private_manifest, total


def _private_output_paths() -> tuple[Path, ...]:
    return tuple(
        CONTINUATION_PRIVATE / f"{CONTINUATION_LABEL}_{suffix}"
        for suffix in (
            "retained_samples.tftensor",
            "final_state.tftensor",
            "final_target_log_prob.tftensor",
            "private_manifest.json",
        )
    )


def _require_fresh() -> None:
    collisions = [
        path for path in (OUTPUT_PATH, *_private_output_paths()) if (ROOT / path).exists()
    ]
    if collisions:
        raise ContinuationError(
            "continuation artifact collision; refusing overwrite: "
            + ", ".join(path.as_posix() for path in collisions)
        )


def run_continuation() -> dict[str, Any]:
    import tensorflow as tf

    base._require_gpu()
    _require_fresh()
    segment0, old_samples, current_state, old_manifest, prior_seconds = validate_handoff()
    if prior_seconds + PROJECTED_SECONDS > base.GPU_BUDGET_SECONDS:
        raise ContinuationError("continuation projection exceeds shared GPU budget")

    adapter = base.A4CalibrationHMCAdapter()
    started_at = base._now()
    started = time.perf_counter()
    (new_samples, _state, new_manifest, call_s), diagnostics, metadata, _ = (
        base._run_archive(
            adapter=adapter,
            archive_dir=CONTINUATION_PRIVATE,
            label=CONTINUATION_LABEL,
            current_state=current_state,
            num_results=CONTINUATION_DRAWS,
            num_burnin_steps=CONTINUATION_BURNIN,
            step_size=FROZEN_STEP_SIZE,
            leapfrog_steps=NUM_LEAPFROG_STEPS,
            seed=CONTINUATION_SEED,
            role="trusted_gpu_xla_a4_hmc_repair_02_exact_continuation_01",
        )
    )
    cumulative = tf.concat((old_samples, new_samples), axis=0)
    admission = base._admission_diagnostics(
        latent_draw_major=cumulative,
        adapter=adapter,
        segment_manifests=(old_manifest, new_manifest),
    )
    completed_at = base._now()
    wall_time = time.perf_counter() - started
    status = "ADMITTED" if admission["admitted"] else (
        "HARD_VETO" if admission["hard_vetoes"] else "NOT_ADMITTED"
    )
    lineage = [
        *(path.as_posix() for path in _budget_lineage_paths(segment0)),
        SEGMENT0_RECEIPT.as_posix(),
    ]
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_repair_02_continuation_01.v1",
        "status": status,
        "continuation": {
            "index": 1,
            "new_draw_count": CONTINUATION_DRAWS,
            "new_burnin_count": CONTINUATION_BURNIN,
            "cumulative_draw_count": int(cumulative.shape[0]),
            "seed": CONTINUATION_SEED,
            "label": CONTINUATION_LABEL,
        },
        "initial_state_policy": "exact_repair_02_segment_0_final_state",
        "frozen_kernel": {
            "step_size": FROZEN_STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "trajectory_length": TRAJECTORY_LENGTH,
            "source": ADAPTATION_RECEIPT.as_posix(),
        },
        "prior_segment_artifact": SEGMENT0_RECEIPT.as_posix(),
        "admission_diagnostics": admission,
        "runner_diagnostics": diagnostics,
        "runner_metadata": metadata,
        "private_manifest_sha256": _sha256(
            CONTINUATION_PRIVATE / f"{CONTINUATION_LABEL}_private_manifest.json"
        ),
        "cumulative_private_sample_sha256": hashlib.sha256(
            bytes(tf.io.serialize_tensor(cumulative).numpy())
        ).hexdigest(),
        "budget_lineage_artifacts": lineage,
        "source_files": _source_bindings(),
        "run_manifest": base._environment_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output_paths=(OUTPUT_PATH, CONTINUATION_PRIVATE),
            random_seeds=(base.ROOT_SEED, CONTINUATION_SEED),
        ),
        "gpu_budget": {
            "cap_seconds": base.GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_seconds,
            "projected_seconds_before_run": PROJECTED_SECONDS,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": call_s,
            "remaining_seconds": base.GPU_BUDGET_SECONDS - prior_seconds - wall_time,
        },
        "nonclaims": admission["nonclaims"],
    }
    base._write_json(OUTPUT_PATH, payload)
    if admission["hard_vetoes"]:
        raise ContinuationError(
            f"continuation hard vetoes: {admission['hard_vetoes']}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-handoff", "run"))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "validate-handoff":
        _segment, samples, state, _manifest, prior = validate_handoff()
        payload = {
            "status": "PASSED",
            "sample_shape": [int(dim) for dim in samples.shape],
            "state_shape": [int(dim) for dim in state.shape],
            "prior_gpu_seconds": prior,
            "remaining_gpu_seconds": base.GPU_BUDGET_SECONDS - prior,
        }
        output = None
    else:
        payload = run_continuation()
        output = OUTPUT_PATH.as_posix()
    print(
        base._canonical_bytes(
            {"command": args.command, "status": payload["status"], "output": output}
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContinuationError as exc:
        print(f"A4_HMC_CONTINUATION_01_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
