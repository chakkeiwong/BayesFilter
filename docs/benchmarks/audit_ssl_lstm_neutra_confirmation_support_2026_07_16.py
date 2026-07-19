#!/usr/bin/env python3
"""Audit support across existing Fresh C NeuTra checkpoints without training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.neutra_training import NeuTraReverseKLTrainer  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    locked_ssl_lstm_posterior_target,
)


CONFIRMATION_PATH = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_plateau_confirmation_2026_07_15.py"
)
CONFIRMATION_SPEC = importlib.util.spec_from_file_location(
    "confirmation_support_audit_runner", CONFIRMATION_PATH
)
if CONFIRMATION_SPEC is None or CONFIRMATION_SPEC.loader is None:
    raise RuntimeError("unable to load confirmation runner")
CONFIRMATION = importlib.util.module_from_spec(CONFIRMATION_SPEC)
sys.modules[CONFIRMATION_SPEC.name] = CONFIRMATION
CONFIRMATION_SPEC.loader.exec_module(CONFIRMATION)

SCHEMA = "bayesfilter.ssl_lstm_neutra.confirmation_support_audit.v1"
DEFAULT_POLICY = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "optuna-plateau-repair-study-2026-07-15/frozen-tuning-policy.json"
)
DEFAULT_STREAM_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "plateau-confirmation-2026-07-16/fresh-c"
)
DEFAULT_OUTPUT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "plateau-confirmation-2026-07-16/fresh-c-support-audit.json"
)
STEPS = (100, 300, 600, 800, 1100)


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"output already exists: {path}")
    path.write_bytes(canonical(payload))


def configure_gpu() -> list[Any]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("support audit requires a visible GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    return gpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--stream-root", type=Path, default=DEFAULT_STREAM_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-cap-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    policy_path = ROOT / args.policy
    stream_root = ROOT / args.stream_root
    output = ROOT / args.output
    policy = CONFIRMATION.load_policy(policy_path)
    result = json.loads((stream_root / "result.json").read_text(encoding="utf-8"))
    expected_hashes = {
        int(row["step"]): str(row["checkpoint_hash"])
        for row in result["checkpoints"]
    }
    gpus = configure_gpu()
    started = time.perf_counter()
    target = locked_ssl_lstm_posterior_target()
    config = CONFIRMATION.trainer_config(
        target, CONFIRMATION.FRESH_STREAMS[0], policy
    )
    rows = []
    for step in STEPS:
        if time.perf_counter() - started + 60.0 >= float(args.gpu_cap_seconds):
            raise RuntimeError("support audit GPU cap exhausted")
        checkpoint_path = stream_root / f"checkpoint-{step:04d}.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("checkpoint_hash") != expected_hashes[step]:
            raise RuntimeError(f"checkpoint hash mismatch at step {step}")
        trainer = NeuTraReverseKLTrainer(target, config)
        trainer.restore_state(checkpoint["trainer_state"])
        frozen = trainer.frozen_transport_payload(
            transport_id=f"ssl-lstm-fresh-c-support-audit-{step}",
            target_signature=target.target_signature(),
        )
        loaded = load_frozen_neutra_artifact(
            frozen, expected_target_signature=target.target_signature()
        )
        probes = CONFIRMATION.HELPERS._probe_diagnostics(  # noqa: SLF001
            target, loaded.transport
        )
        history = next(row for row in result["history"] if int(row["step"]) == step)
        rows.append(
            {
                "step": step,
                "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha256(checkpoint_path),
                "checkpoint_hash": checkpoint["checkpoint_hash"],
                "mean_loss": history["mean_loss"],
                "saturation_fraction": history["saturation_fraction"],
                "controller_action": history["controller_action"],
                "all_finite": probes["all_finite"],
                "roundtrip_max_abs": probes["roundtrip_max_abs"],
                "original_neighborhood_max_inverse_radius": probes[
                    "original_neighborhood_max_inverse_radius"
                ],
                "moderate_shell_max_inverse_radius": probes[
                    "moderate_shell_max_inverse_radius"
                ],
                "support_screen_passed": bool(
                    probes["all_finite"]
                    and probes["roundtrip_max_abs"] <= 1.0e-9
                    and probes["moderate_shell_max_inverse_radius"] <= 4.30
                ),
            }
        )
    any_later_pass = any(
        row["support_screen_passed"] and row["step"] > int(result["best_step"])
        for row in rows
    )
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "question": (
            "Did Fresh C support become admissible after the loss-selected best "
            "checkpoint without changing training or thresholds?"
        ),
        "decision": (
            "CHECKPOINT_SELECTION_MISALIGNMENT_SUPPORTED"
            if any_later_pass
            else "SELECTED_CANDIDATE_SUPPORT_INSTABILITY_SUPPORTED"
        ),
        "rows": rows,
        "loss_selected_best_step": result["best_step"],
        "any_later_support_pass": any_later_pass,
        "policy_path": args.policy.as_posix(),
        "policy_sha256": sha256(policy_path),
        "fresh_c_result_path": (args.stream_root / "result.json").as_posix(),
        "fresh_c_result_sha256": sha256(stream_root / "result.json"),
        "run_manifest": {
            "command": " ".join(sys.argv),
            "charged_gpu_seconds": time.perf_counter() - started,
            "gpu_cap_seconds": float(args.gpu_cap_seconds),
            "physical_gpus": [gpu.name for gpu in gpus],
            "jit_compile": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "dtype": "float64",
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": [
            "read-only checkpoint audit; no retraining or reselection authority",
            "no posterior-correctness, HMC, superiority, or default-readiness claim",
        ],
    }
    write_json(output, payload)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "any_later_support_pass": any_later_pass,
                "charged_gpu_seconds": payload["run_manifest"]["charged_gpu_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
