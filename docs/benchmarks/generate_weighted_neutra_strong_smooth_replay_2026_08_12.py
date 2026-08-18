#!/usr/bin/env python3
"""Generate CPU-only local-coordinate replay tensors for the strong-smooth canary."""

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

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-strong-smooth-reflection-proposal-repair-plan-2026-08-12.md"
)
DEFAULT_PROPOSAL = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "varying-hessian/strong-smooth-proposal-r7-reflected/selected_proposal.json"
)
CONSTANTS = Path(
    "/home/ubuntu/python/dsge_hmc/results/neutra/gate3/"
    "nk_strong_smooth_bridge_20260604/frozen_constants/"
    "strong_smooth_from_seed42_affine_lift.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-name", choices=("nk_like_mild_smooth", "nk_like_strong_smooth"), default="nk_like_strong_smooth")
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--training-size", type=int, default=1_048_576)
    parser.add_argument("--selection-size", type=int, default=16_384)
    parser.add_argument("--audit-size", type=int, default=16_384)
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _proposal(path: Path, target_name: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("frozen proposal object is required")
    expected = str(payload.get("proposal_hash", ""))
    checked = {key: value for key, value in payload.items() if key != "proposal_hash"}
    if len(expected) != 64 or _stable_hash(checked) != expected:
        raise RuntimeError("frozen replay proposal hash mismatch")
    if payload.get("target_name") != target_name:
        raise RuntimeError("frozen replay proposal target mismatch")
    proposal = payload.get("proposal")
    if not isinstance(proposal, Mapping):
        raise RuntimeError("frozen proposal payload is missing")
    return payload


def _rows(spec: Any, proposal: Mapping[str, Any], count: int, seed: tuple[int, int]) -> tuple[Any, Any]:
    from bayesfilter.inference.neutra_varying_hessian_target import (
        physical_to_affine_local,
        varying_hessian_log_prob_and_score_batch,
    )
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob,
        sample_gaussian_mixture,
    )

    physical, _labels = sample_gaussian_mixture(
        count,
        proposal["probabilities"],
        proposal["means"],
        proposal["covariances"],
        seed=seed,
    )
    target, score = varying_hessian_log_prob_and_score_batch(spec, physical)
    del score
    return physical_to_affine_local(spec, physical), target - gaussian_mixture_log_prob(
        physical,
        proposal["probabilities"],
        proposal["means"],
        proposal["covariances"],
    )


def main() -> int:
    args = _parse_args()
    if any(
        int(value) <= 1
        for value in (args.training_size, args.selection_size, args.audit_size)
    ):
        raise ValueError("replay sizes must exceed one")
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"output root must be fresh: {root}")
    proposal_path = args.proposal.resolve()
    constants_path = args.constants.resolve()
    if not PLAN.is_file() or not proposal_path.is_file() or not constants_path.is_file():
        raise FileNotFoundError("plan, frozen proposal, or constants missing")
    root.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    from bayesfilter.inference.neutra_varying_hessian_target import (
        load_varying_hessian_target_spec,
    )
    from bayesfilter.testing.importance_sampling_tf import validate_gaussian_mixture

    proposal_payload = _proposal(proposal_path, args.target_name)
    spec = load_varying_hessian_target_spec(
        constants_path, expected_name=args.target_name
    )
    if proposal_payload.get("target_constants_sha256") != spec.constants_sha256:
        raise RuntimeError("frozen replay proposal constants hash mismatch")
    frozen = proposal_payload["proposal"]
    # Re-establish float64 before validation; otherwise TensorFlow defaults
    # JSON lists to float32 and the strict normalization check rounds the
    # seven-component probabilities away from one.
    raw_probabilities = tf.constant(frozen["probabilities"], tf.float64)
    raw_means = tf.constant(frozen["means"], tf.float64)
    raw_covariances = tf.constant(frozen["covariances"], tf.float64)
    probabilities, means, covariances, _ = validate_gaussian_mixture(
        raw_probabilities, raw_means, raw_covariances
    )
    proposal = {
        # JSON has no dtype. Re-establish the source target's float64 contract
        # before the batch-native target sees proposal-generated rows.
        "probabilities": tf.cast(probabilities, tf.float64),
        "means": tf.cast(means, tf.float64),
        "covariances": tf.cast(covariances, tf.float64),
    }
    training_rows, training_log_weights = _rows(
        spec, proposal, int(args.training_size), (20260812, 16000)
    )
    selection_rows, selection_log_weights = _rows(
        spec, proposal, int(args.selection_size), (20260812, 16001)
    )
    audit_rows, audit_log_weights = _rows(
        spec, proposal, int(args.audit_size), (20260812, 16002)
    )
    tensors = {
        "training_local_rows": training_rows,
        "training_log_weights": training_log_weights,
        "selection_local_rows": selection_rows,
        "selection_log_weights": selection_log_weights,
        "audit_local_rows": audit_rows,
        "audit_log_weights": audit_log_weights,
    }
    receipts = {}
    for name, value in tensors.items():
        path = root / f"{name}.tftensor"
        path.write_bytes(tf.io.serialize_tensor(value).numpy())
        receipts[name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "shape": tuple(int(item) for item in value.shape),
            "dtype": "float64",
        }
    manifest = {
        "schema": "bayesfilter.weighted_neutra_strong_smooth_cpu_replay.v1",
        "plan": PLAN.as_posix(),
        "proposal_path": proposal_path.as_posix(),
        "proposal_sha256": _sha256(proposal_path),
        "proposal_hash": proposal_payload["proposal_hash"],
        "source_constants_path": constants_path.as_posix(),
        "source_constants_sha256": spec.constants_sha256,
        "target": spec.manifest_payload(),
        "cpu_only": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "training_coordinates": "frozen_affine_lift_local_x",
        "replay_seeds": {
            "training": [20260812, 16000],
            "selection": [20260812, 16001],
            "audit": [20260812, 16002],
        },
        "partitions_disjoint_by_stateless_seed": True,
        "receipts": receipts,
    }
    _write(root / "replay_manifest.json", manifest)
    _write(
        root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.weighted_neutra_strong_smooth_cpu_replay_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in root.iterdir()
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"output_root": root.as_posix(), "cpu_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
