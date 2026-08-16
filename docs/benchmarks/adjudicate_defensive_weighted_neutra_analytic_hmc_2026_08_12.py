#!/usr/bin/env python3
"""Re-adjudicate immutable run-v5 without an uncalibrated joint moment veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-plan-2026-08-12.md"
SOURCE = ROOT / "docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5"
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-"
    "run-v5-adjudication-v1"
)
CHECKPOINT = ROOT / (
    "docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/"
    "r1-two-mode/capacity-depth6-width128-updates10000-confirmation-1-v1/"
    "trainer_states.json"
)
EXPECTED_SOURCE_RESULT_SHA256 = "db4ed848e0da72b591796acd7ee8018cfdbf6acf3130caf6e38acb6249289988"
EXPECTED_SOURCE_MANIFEST_SHA256 = "d85b21dfb2f55baed07d7edd1f0ef7eb4feee009b90d7bc13b68c82614ec16e3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_archive_receipts(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    count = 0
    for phase in ("warmup_chunks", "retained_chunks"):
        for row in manifest.get(phase, ()):
            receipts = (
                row["sample_receipt"],
                *row["trace_receipts"].values(),
                row["receipt"],
            )
            for receipt in receipts:
                path = Path(receipt["path"])
                if not path.is_file() or _sha256(path) != receipt["sha256"]:
                    raise RuntimeError(f"archive receipt mismatch: {path}")
                count += 1
    return {"all_passed": True, "verified_receipt_count": count}


def _retained_samples(tf: Any, manifest: Mapping[str, Any]) -> Any:
    tensors = []
    for row in manifest.get("retained_chunks", ()):
        path = Path(row["sample_receipt"]["path"])
        tensors.append(tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64))
    if not tensors:
        raise RuntimeError("source archive has no retained samples")
    return tf.concat(tensors, axis=0)


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    output.mkdir(parents=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import tensorflow as tf

    from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import (
        load_weighted_neutra_transport,
        retained_analytic_diagnostics,
    )

    source_result_path = SOURCE / "result.json"
    source_manifest_path = SOURCE / "archive/weighted-analytic-manifest.json"
    if _sha256(source_result_path) != EXPECTED_SOURCE_RESULT_SHA256:
        raise RuntimeError("source result SHA-256 mismatch")
    if _sha256(source_manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source archive manifest SHA-256 mismatch")
    source = _read(source_result_path)
    manifest = _read(source_manifest_path)
    receipts = _verify_archive_receipts(manifest)
    latent = _retained_samples(tf, manifest)
    loaded = load_weighted_neutra_transport(CHECKPOINT)
    physical = tf.reshape(
        loaded.transport.forward_batch(tf.reshape(latent, (-1, 4))), tf.shape(latent)
    )
    diagnostics = retained_analytic_diagnostics(physical)

    sequential = source.get("sequential", {})
    sequential_passed = bool(sequential.get("passed"))
    hard_analytic_passed = bool(diagnostics["passed_primary_screens"])
    compatible = sequential_passed and hard_analytic_passed and receipts["all_passed"]
    payload = {
        "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_adjudication.v1",
        "plan": PLAN.as_posix(),
        "source": {
            "root": SOURCE.as_posix(),
            "result_path": source_result_path.as_posix(),
            "result_sha256": EXPECTED_SOURCE_RESULT_SHA256,
            "archive_manifest_path": source_manifest_path.as_posix(),
            "archive_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "original_decision": source.get("decision"),
            "original_binary_label_status": "preserved_but_overly_stringent",
        },
        "archive_receipts": receipts,
        "warmup_excluded_from_posterior": bool(
            sequential.get("metadata", {}).get("warmup_excluded_from_posterior")
        ),
        "sequential_passed": sequential_passed,
        "retained_analytic_diagnostics": diagnostics,
        "adjudication": {
            "status": (
                "statistically_compatible_under_declared_marginal_diagnostics"
                if compatible
                else "not_compatible_under_declared_marginal_diagnostics"
            ),
            "hard_primary_screens_passed": compatible,
            "joint_moment_test_performed": False,
            "mean_interval_pass_count": diagnostics["moment_diagnostics"][
                "mean_interval_pass_count"
            ],
            "mean_interval_total_count": diagnostics["moment_diagnostics"][
                "mean_interval_total_count"
            ],
            "covariance_interval_pass_count": diagnostics["moment_diagnostics"][
                "covariance_interval_pass_count"
            ],
            "covariance_interval_total_count": diagnostics["moment_diagnostics"][
                "covariance_interval_total_count"
            ],
            "claim": (
                "This one frozen-transport HMC run is statistically compatible with "
                "the analytic target under the declared diagnostics."
            ),
            "nonclaims": (
                "no proof of distributional equality or stationarity",
                "no statistically supported sampler or kernel ranking",
                "no cross-transport-seed robustness claim",
                "no general NeuTra, SSL-LSTM, or default-readiness claim",
                "marginal moment interval misses remain reported and unexplained",
            ),
        },
        "execution": {
            "backend": "tensorflow_diagnostic_only",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_intentionally_hidden": True,
            "new_hmc_transitions_run": 0,
        },
    }
    result_path = output / "adjudication.json"
    _write(result_path, payload)
    git_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    terminal_manifest_path = output / "terminal_manifest.json"
    _write(
        terminal_manifest_path,
        {
            "schema": "bayesfilter.defensive_weighted_neutra_hmc_terminal_manifest.v1",
            "git_commit": git_commit,
            "git_worktree_dirty": bool(git_status),
            "git_status_line_count": len(git_status),
            "command": " ".join(sys.argv),
            "environment": "tfgpu",
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": __import__("tensorflow_probability").__version__,
            "cpu_gpu_status": {
                "execution_device": "CPU",
                "gpu_intentionally_hidden": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source_gpu_run": {
                    "host_device_request": "CUDA_VISIBLE_DEVICES=1",
                    "framework_logical_device": "/device:GPU:0",
                    "memory_policy": "memory_growth_verified",
                    "xla": True,
                    "tf32": False,
                },
            },
            "data_version": "analytic_target_separated_two_mode_unequal_weight_d4_v1",
            "random_seeds": {
                "tuning": ((20260812, 92001), (20260812, 93001), (20260812, 94001)),
                "sequential_root": (20260812, 91001),
                "adjudication": "none",
            },
            "wall_time_seconds": {
                "source_campaign": float(source.get("wall_seconds")),
                "source_sequential": float(
                    source.get("sequential", {}).get("metadata", {}).get("wall_seconds")
                ),
                "adjudication": time.perf_counter() - started,
            },
            "output_artifacts": {
                "source_result": source_result_path.as_posix(),
                "source_archive_manifest": source_manifest_path.as_posix(),
                "adjudication": result_path.as_posix(),
                "terminal_manifest": terminal_manifest_path.as_posix(),
            },
            "plan_file": PLAN.as_posix(),
            "result_file": (
                ROOT
                / "docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-result-2026-08-12.md"
            ).as_posix(),
            "source_hashes": {
                "result": EXPECTED_SOURCE_RESULT_SHA256,
                "archive_manifest": EXPECTED_SOURCE_MANIFEST_SHA256,
                "checkpoint": loaded.checkpoint_sha256,
            },
            "nonclaims": payload["adjudication"]["nonclaims"],
        },
    )
    _write(
        output / "artifact_hashes.json",
        {
            "schema": "bayesfilter.defensive_weighted_neutra_hmc_adjudication_hashes.v1",
            "artifacts": {
                "adjudication.json": _sha256(result_path),
                "terminal_manifest.json": _sha256(terminal_manifest_path),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
