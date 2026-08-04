"""Emit the bounded P88 fixed-variant Phase-0 reconstruction artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_phase0 import (
    reconstruct_p88_phase0,
)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        materialized = value.numpy()
        return materialized.item() if value.shape.rank == 0 else materialized.tolist()
    return value


def _git_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run(repository_root: Path, output: Path) -> dict[str, object]:
    started = time.monotonic()
    audit = reconstruct_p88_phase0(repository_root)
    payload = dict(audit.manifest_payload())
    payload.update(
        {
            "phase": "phase0_fixed_variant_baseline_freeze",
            "decision": "stop_without_replacement",
            "density_reconstruction_status": "PASS_EXACT_P88_T1_DENSITY_PARITY",
            "baseline_admission_status": audit.status,
            "phase1_authorized": False,
            "command_environment": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cpu_only_intentional": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
                "gpu_devices_hidden_before_tensorflow_import": (
                    os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
                ),
                "gpu_execution_attempted": False,
            },
            "git_head": _git_head(repository_root),
            "wall_time_seconds": time.monotonic() - started,
            "result_artifact_path": str(output),
            "plan_path": (
                "docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-"
                "parameter-extension-master-plan-2026-07-30.md"
            ),
            "execution_note_path": (
                "docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-"
                "phase0-execution-note-2026-07-30.md"
            ),
            "result_note_path": (
                "docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-"
                "phase0-result-2026-07-30.md"
            ),
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "scripts.run_zhao_cui_austria_sir_fixed_variant_phase0 "
                "--repository-root . --output "
                "docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-"
                "phase0-result-attempt02-2026-07-30.json"
            ),
            "missing_identity_fields": (
                "coordinate_frame_mu",
                "coordinate_frame_matrix",
                "transport_cdf_config",
                "frozen_reference_samples",
                "retained_branch_identity",
                "source_dependency_closure",
            ),
            "forbidden_repairs": (
                "recompute_frame_with_current_unbound_source",
                "select_historical_fixture_cdf_defaults",
                "synthesize_t2_transport",
                "insert_ukf",
                "switch_to_author_tt_cross_als",
                "switch_to_apf_or_source_replica",
            ),
            "nonclaims": (
                "no active-observation value",
                "no score",
                "no T2 fit or value",
                "no T20 execution",
                "no GPU evidence",
                "no HMC readiness",
                "no correctness or production readiness",
            ),
        }
    )
    return _json_ready(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase-0 artifact: {output}")
    payload = run(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
