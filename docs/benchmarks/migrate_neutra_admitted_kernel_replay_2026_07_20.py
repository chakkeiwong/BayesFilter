#!/usr/bin/env python3
"""Migrate preserved private LGSSM tuning evidence into replay mechanics.

This command is intentionally migration-only: it does not tune, sample, or
modify any historical attempt root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tuning-result", type=Path, required=True)
    parser.add_argument("--frozen-transport", type=Path, required=True)
    parser.add_argument("--frozen-transport-sha256", required=True)
    parser.add_argument("--cell", default="LGSSM-EXACT")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"migration output must be fresh: {args.output_root}")
    if not args.source_tuning_result.is_file():
        raise FileNotFoundError(args.source_tuning_result)
    if not args.frozen_transport.is_file():
        raise FileNotFoundError(args.frozen_transport)
    observed_transport_sha256 = _sha256(args.frozen_transport)
    expected_transport_sha256 = str(args.frozen_transport_sha256).lower()
    if observed_transport_sha256 != expected_transport_sha256:
        raise ValueError(
            "frozen transport SHA-256 mismatch: "
            f"{observed_transport_sha256} != {expected_transport_sha256}"
        )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_end_to_end import (
        BatchNativeBoundAdapter,
        _fixed_transport_adapter,
        _target_signature,
    )
    from bayesfilter.inference.hmc_kernel_tuning import (
        admitted_kernel_mechanics_payload_from_serialized_tuning_payload,
        build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
    )
    from bayesfilter.runtime import atomic_write_json
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == args.cell)
    source_payload = _read_mapping(args.source_tuning_result)
    if source_payload.get("schema") == "bayesfilter.neutra.all_models.cell_result.v1":
        tuning = source_payload.get("tuning")
        if not isinstance(tuning, Mapping):
            raise ValueError("cell result is missing its tuning payload")
    else:
        tuning = source_payload
    adapter = spec.adapter_factory()
    if _target_signature(adapter) != spec.target_signature:
        raise ValueError("current adapter target signature mismatch")
    bound = BatchNativeBoundAdapter(adapter, target_signature=spec.target_signature)
    loaded = load_frozen_neutra_artifact(
        _read_mapping(args.frozen_transport),
        expected_target_signature=spec.target_signature,
    )
    target_scope = f"{spec.cell_id}:fixed_neutra_native_tuning"
    tuned_adapter = _fixed_transport_adapter(bound, loaded.transport, target_scope)
    execution = {
        "dtype": "float64",
        "backend": "tensorflow_probability",
        "jit_compile": True,
        "tf32_execution_enabled": True,
        "mass_policy": "fixed_identity",
    }
    artifact = admitted_kernel_mechanics_payload_from_serialized_tuning_payload(
        adapter=tuned_adapter,
        tuning_payload=tuning,
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        target_signature=spec.target_signature,
        target_scope=target_scope,
        execution=execution,
        source_artifact_path=str(args.source_tuning_result),
        source_artifact_sha256=_sha256(args.source_tuning_result),
    )
    replay = build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload(
        adapter=tuned_adapter,
        mechanics_payload=artifact,
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        target_signature=spec.target_signature,
        target_scope=target_scope,
        execution=execution,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
    )
    args.output_root.mkdir(parents=True)
    output_path = args.output_root / "admitted_kernel_mechanics.json"
    atomic_write_json(output_path, artifact)
    manifest = {
        "schema": "bayesfilter.neutra.admitted_kernel_replay_migration_manifest.v1",
        "cell_id": spec.cell_id,
        "target_signature": spec.target_signature,
        "frozen_transport_path": str(args.frozen_transport),
        "frozen_transport_sha256": observed_transport_sha256,
        "source_tuning_result": str(args.source_tuning_result),
        "source_tuning_result_sha256": _sha256(args.source_tuning_result),
        "output_path": str(output_path),
        "mechanics_sha256": artifact["mechanics_sha256"],
        "target_scope": target_scope,
        "step_size": replay.step_size,
        "num_leapfrog_steps": replay.num_leapfrog_steps,
        "tuning_invoked": False,
        "sampling_invoked": False,
        "gpu_hidden_for_migration": True,
        "nonclaims": (
            "migration and replay reconstruction only",
            "no posterior or second-seed sampling evidence",
        ),
    }
    atomic_write_json(args.output_root / "migration_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
