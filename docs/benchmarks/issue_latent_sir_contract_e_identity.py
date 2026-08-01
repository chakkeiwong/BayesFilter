"""Issue the repository-owned Contract E identity for frozen latent-SIR inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.ledh_contract_e_identity import (  # noqa: E402
    _require_factory_identity,
    issue_latent_sir_contract_e_route_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor(record):
    value = tf.convert_to_tensor(
        record["values"], tf.dtypes.as_dtype(record["dtype"])
    )
    value = tf.reshape(value, record["shape"])
    serialized = tf.io.serialize_tensor(value).numpy()
    if hashlib.sha256(serialized).hexdigest() != record["serialized_tensor_sha256"]:
        raise ValueError("prepared tensor serialized hash mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--gpu-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("identity issuance must be CPU-only with CUDA_VISIBLE_DEVICES=-1")
    args.output.parent.mkdir(parents=True, exist_ok=False)

    prepared_payload = json.loads(args.prepared_input.read_text(encoding="utf-8"))
    gpu_payload = json.loads(args.gpu_result.read_text(encoding="utf-8"))
    prepared = {
        name: _tensor(record)
        for name, record in prepared_payload["prepared"].items()
    }
    expected_symbol = (
        "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:"
        "latent_sir_contract_e_canonical_value_and_score_tf"
    )
    if gpu_payload.get("route_execution_symbol") != expected_symbol:
        raise ValueError("GPU result did not execute the registered SIR route symbol")
    if gpu_payload.get("status") != "PASS_BOUNDED_DIAGNOSTIC":
        raise ValueError("GPU result did not pass its engineering diagnostic")
    if not gpu_payload.get("configuration", {}).get("canonical_registered_route"):
        raise ValueError("GPU result is not marked as the canonical registered route")
    if not gpu_payload.get("configuration", {}).get("jit_compile"):
        raise ValueError("GPU result did not use XLA JIT")

    identity = issue_latent_sir_contract_e_route_identity(
        prepared_inputs=prepared
    )
    _require_factory_identity(identity)
    payload = {
        "schema": "bayesfilter.latent_sir_contract_e_identity_certificate.v1",
        "status": "PASS_FACTORY_ISSUED_IDENTITY_NOT_SCIENTIFICALLY_ADMITTED",
        "identity": identity.to_dict(),
        "identity_sha256": identity.identity_sha256,
        "prepared_input": {
            "path": str(args.prepared_input),
            "sha256": _sha256(args.prepared_input),
        },
        "gpu_result": {
            "path": str(args.gpu_result),
            "sha256": _sha256(args.gpu_result),
            "route_execution_symbol": expected_symbol,
        },
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "command": " ".join(sys.argv),
        "nonclaims": [
            "factory identity is not scientific admission",
            "not proof of target accuracy or teacher agreement beyond recorded artifacts",
            "not HMC or leaderboard readiness",
            "not fixed-TTSIRT source-route closure",
        ],
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

