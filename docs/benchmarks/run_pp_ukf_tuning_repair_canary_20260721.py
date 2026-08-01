#!/usr/bin/env python3
"""Run a bounded PP-UKF fixed-identity tuning mechanics canary.

The canary uses the real frozen transport and public tuner with diagnostic
budgets, timeout, heartbeat, GPU/XLA, and no sequential sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-transport", type=Path, required=True)
    parser.add_argument("--frozen-transport-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1500.0)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"canary output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.hmc_kernel_tuning import HMCKernelTuningConfig, tune_hmc_kernel
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    transport_sha = _sha256(args.frozen_transport)
    if transport_sha != str(args.frozen_transport_sha256).lower():
        raise ValueError(f"frozen transport SHA mismatch: {transport_sha}")
    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == "PP-UKF")
    loaded = load_frozen_neutra_artifact(
        json.loads(args.frozen_transport.read_text(encoding="utf-8")),
        expected_target_signature=spec.target_signature,
    )
    bound = BatchNativeBoundAdapter(
        spec.adapter_factory(), target_signature=spec.target_signature
    )
    scope = "PP-UKF:ten_phase_tuning_repair_canary"
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bound,
        transport=loaded.transport,
        target_scope=scope,
        evidence_path=str(Path(__file__).relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    started = time.monotonic()
    tuning_dir = args.output_root / "tuning"
    result = tune_hmc_kernel(
        adapter=adapter,
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        config=HMCKernelTuningConfig.diagnostic(
            max_attempts=1,
            bootstrap_max_repairs=2,
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
            mass_policy="fixed_identity",
            chain_execution_mode="tf_function",
            use_xla=True,
            target_scope=scope,
            target_status_trace_policy="per_chain_step",
            public_timeout_budget_s=float(args.timeout_seconds),
            incall_progress_heartbeat_s=30.0,
            source="bayesfilter.pp_ukf.ten_phase_tuning_repair_canary",
        ),
        output_dir=tuning_dir,
    )
    manifest = {
        "schema": "bayesfilter.pp_ukf_tuning_repair_canary_manifest.v1",
        "role": "bounded_tuning_mechanics_canary_only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "device_list": [str(item) for item in tf.config.list_logical_devices()],
        "memory_policy": memory_policy,
        "jit_compile": True,
        "tf32_execution_enabled": True,
        "dtype": "float64",
        "target_signature": spec.target_signature,
        "adapter_signature": bound.adapter_signature(),
        "transport_sha256": transport_sha,
        "random_seed": result.config.seed,
        "timeout_seconds": args.timeout_seconds,
        "heartbeat_seconds": result.config.incall_progress_heartbeat_s,
        "wall_time_seconds": time.monotonic() - started,
        "tuning_result_path": result.artifact_path,
        "tuning_final_status": result.final_status,
        "tuning_passed": result.passed,
        "sampling_launched": False,
        "plan_path": "docs/plans/bayesfilter-pp-ukf-ten-phase-tuning-repair-plan-2026-07-21.md",
        "nonclaims": [
            "diagnostic-budget canary only",
            "no serious tuning admission",
            "no posterior sampling or convergence claim",
            "no scientific or default-readiness claim",
        ],
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result.final_status, "passed": result.passed, "manifest": str(args.output_root / "run_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
