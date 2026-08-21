#!/usr/bin/env python3
"""Diagnose a frozen reverse-NeuTra defensive German proposal on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-weighted-forward-kl-german-credit-plan-2026-08-13.md"
STATE_SCHEMA = "bayesfilter.weighted_neutra_german_reverse_state.v1"
ESS_FRACTION_MIN = 0.0625
BATCH_SIZE = 4096


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=65536)
    parser.add_argument(
        "--proposal-kind",
        choices=("reverse_scale_mixture", "reference_augmented"),
        default="reverse_scale_mixture",
    )
    parser.add_argument("--scales", type=float, nargs="+", default=(1.0, 1.25, 1.5))
    parser.add_argument(
        "--probabilities", type=float, nargs="+", default=(0.90, 0.08, 0.02)
    )
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
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


def _ess(log_weights: Any) -> tuple[Any, Any]:
    import tensorflow as tf

    normalized = tf.nn.softmax(log_weights)
    return tf.math.reciprocal(tf.reduce_sum(tf.square(normalized))), tf.reduce_max(normalized)


def main() -> int:
    args = _parse_args()
    root = args.output_root.resolve()
    training_root = args.training_root.resolve()
    if root.exists():
        raise FileExistsError(f"output root must be fresh: {root}")
    if int(args.sample_count) <= 1 or int(args.sample_count) % BATCH_SIZE != 0:
        raise ValueError("sample-count must exceed one and be divisible by 4096")
    required = (
        PLAN,
        args.data.resolve(),
        args.reference.resolve(),
        training_root / "trainer_state.json",
        training_root / "artifact_hashes.json",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("German proposal inputs are missing")
    root.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    from bayesfilter.inference.neutra_german_credit_proposal import (
        load_frozen_german_transport,
        sample_reference_augmented_proposal,
        sample_defensive_pushed_proposal,
    )
    from bayesfilter.inference.neutra_german_credit_target import (
        german_credit_log_prob_batch,
        load_german_credit_target_spec,
    )

    started = time.perf_counter()
    spec = load_german_credit_target_spec(args.data, args.reference)
    frozen = load_frozen_german_transport(
        training_root / "trainer_state.json",
        training_root / "artifact_hashes.json",
        expected_schema=STATE_SCHEMA,
    )
    if frozen.target_name != spec.name:
        raise RuntimeError("German proposal checkpoint target mismatch")
    if frozen.target_data_sha256 != spec.data_sha256:
        raise RuntimeError("German proposal checkpoint data hash mismatch")
    if frozen.target_reference_sha256 != spec.reference_sha256:
        raise RuntimeError("German proposal checkpoint reference hash mismatch")
    if args.proposal_kind == "reference_augmented":
        physical, latent, log_proposal, labels = sample_reference_augmented_proposal(
            frozen.transport,
            spec,
            int(args.sample_count),
            seed=(20260813, 44002),
        )
        proposal = {
            "identity": "reference_marginals_plus_frozen_reverse_pushforward_v1",
            "component_probabilities": [0.85, 0.10, 0.05],
            "reference_marginal_scales": [1.0, 1.5],
            "full_support": True,
            "reference_informed_representation_test": True,
        }
        seed = (20260813, 44002)
        component_count = 3
    else:
        physical, latent, log_proposal, labels = sample_defensive_pushed_proposal(
            frozen.transport,
            int(args.sample_count),
            tuple(args.scales),
            tuple(args.probabilities),
            seed=(20260813, 44001),
        )
        proposal = {
            "identity": "frozen_reverse_iaf_pushed_isotropic_scale_mixture_v1",
            "base_scales": list(args.scales),
            "base_probabilities": list(args.probabilities),
            "full_support": True,
        }
        seed = (20260813, 44001)
        component_count = len(args.scales)
    target_chunks = []
    batch_ess = []
    batch_maximum_weight = []
    for start in range(0, int(args.sample_count), BATCH_SIZE):
        stop = start + BATCH_SIZE
        target = german_credit_log_prob_batch(spec, physical[start:stop])
        target_chunks.append(target)
        ess, maximum = _ess(target - log_proposal[start:stop])
        batch_ess.append(ess)
        batch_maximum_weight.append(maximum)
    target_log_prob = tf.concat(target_chunks, axis=0)
    log_weights = target_log_prob - log_proposal
    global_ess, global_maximum = _ess(log_weights)
    batch_ess_tensor = tf.stack(batch_ess)
    batch_fraction = batch_ess_tensor / tf.constant(float(BATCH_SIZE), tf.float64)
    sorted_batch_fraction = tf.sort(batch_fraction)
    median_batch_fraction = tfp_median(sorted_batch_fraction)
    passed = bool(
        (global_ess / tf.cast(args.sample_count, tf.float64) >= ESS_FRACTION_MIN).numpy()
        and (median_batch_fraction >= ESS_FRACTION_MIN).numpy()
    )
    proposal_payload = {
        "schema": "bayesfilter.weighted_neutra_german_defensive_proposal.v1",
        "target": spec.manifest_payload(),
        "training_state": (training_root / "trainer_state.json").as_posix(),
        "training_state_sha256": frozen.state_sha256,
        "training_state_hash": frozen.state_hash,
        "selected_reverse_update": frozen.selected_update,
        "proposal": proposal,
    }
    proposal_payload["proposal_hash"] = _stable_hash(proposal_payload)
    result = {
        "schema": "bayesfilter.weighted_neutra_german_proposal_diagnostic.v1",
        "passed": passed,
        "primary_criterion": {
            "global_ess_fraction_min": ESS_FRACTION_MIN,
            "median_batch_ess_fraction_min": ESS_FRACTION_MIN,
        },
        "sample_count": int(args.sample_count),
        "batch_size": BATCH_SIZE,
        "global_ess": global_ess,
        "global_ess_fraction": global_ess / tf.cast(args.sample_count, tf.float64),
        "global_maximum_normalized_weight": global_maximum,
        "batch_ess": batch_ess_tensor,
        "batch_ess_fraction": batch_fraction,
        "median_batch_ess_fraction": median_batch_fraction,
        "minimum_batch_ess_fraction": tf.reduce_min(batch_fraction),
        "maximum_batch_normalized_weight": tf.reduce_max(tf.stack(batch_maximum_weight)),
        "component_counts": tf.math.bincount(
            labels, minlength=component_count, maxlength=component_count
        ),
        "all_finite": bool(
            tf.reduce_all(
                tf.math.is_finite(target_log_prob)
                & tf.math.is_finite(log_proposal)
                & tf.reduce_all(tf.math.is_finite(physical), axis=1)
                & tf.reduce_all(tf.math.is_finite(latent), axis=1)
            ).numpy()
        ),
        "proposal": proposal_payload,
        "execution": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tensorflow_version": tf.__version__,
            "batch_native_target": True,
            "sample_wise_loop_or_scalar_fallback": False,
            "seed": list(seed),
            "wall_seconds": time.perf_counter() - started,
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "command": " ".join(sys.argv),
        },
        "nonclaims": (
            "proposal support diagnostic only",
            "importance rows are not unweighted posterior draws",
            "proposal ESS does not establish transport or HMC validity",
        ),
    }
    if not result["all_finite"]:
        raise RuntimeError("German proposal diagnostic produced nonfinite output")
    _write(root / "selected_proposal.json", proposal_payload)
    _write(root / "result.json", result)
    _write(
        root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.weighted_neutra_german_proposal_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(root.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"passed": passed, "output_root": root.as_posix()}))
    return 0


def tfp_median(sorted_values: Any) -> Any:
    """Return the median of a static even-length sorted TensorFlow vector."""

    import tensorflow as tf

    count = sorted_values.shape[0]
    if count is None or int(count) < 2:
        raise ValueError("median input must have a static length of at least two")
    middle = int(count) // 2
    if int(count) % 2:
        return sorted_values[middle]
    return tf.constant(0.5, sorted_values.dtype) * (
        sorted_values[middle - 1] + sorted_values[middle]
    )


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
