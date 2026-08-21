#!/usr/bin/env python3
"""Tiny CPU/XLA replay canary for the non-circular NeuTra plan.

This is an engineering smoke only.  It generates proposal rows, evaluates the
exact q=20 target, and performs a weighted forward-KL update.  It does not
create posterior samples or establish transport/HMC quality.
"""

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
GEOMETRY = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
)
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-plan-2026-08-19.md"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-2026-08-19/replay-canary-r1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "dirty": bool(
            subprocess.run(
                ("git", "status", "--short"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.rows < 2 or args.updates < 1:
        raise SystemExit("rows must be >=2 and updates must be positive")

    # This script is explicitly a CPU reference/smoke lane.  Hide GPUs before
    # importing TensorFlow, as required for CPU-only diagnostic artifacts.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob,
        sample_gaussian_mixture,
    )

    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    means = tf.constant(
        [
            geometry["representatives"][label]["position"]
            for label in ("plus", "minus")
        ],
        tf.float64,
    )
    precisions = tf.constant(
        [
            geometry["source_curvature"][label]["records"][-1]["precision"]
            for label in ("plus", "minus")
        ],
        tf.float64,
    )
    covariances = tf.linalg.inv(precisions)
    probabilities = tf.constant((0.5, 0.5), tf.float64)
    rows, labels = sample_gaussian_mixture(
        int(args.rows), probabilities, means, covariances, seed=(20260819, 1)
    )
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    target_value, _score, status = target.neutra_batch_log_prob_and_grad_status(rows)
    proposal_value = gaussian_mixture_log_prob(
        rows, probabilities, means, covariances
    )
    log_weights = target_value - proposal_value
    valid = tf.logical_and(
        tf.equal(tf.cast(status["status_code"], tf.int32), 0),
        tf.cast(status["valid_pre_regularized_score"], tf.bool),
    )
    trainer = WeightedForwardKLNeuTraTrainer(
        WeightedNeuTraConfig(
            dimension=4,
            hidden_layers=(16, 16),
            stages=2,
            initialization_seed=(20260819, 2),
            jit_compile=True,
        )
    )
    started = time.perf_counter()
    steps = []
    for _ in range(int(args.updates)):
        step = trainer.train_step(rows, log_weights)
        steps.append(
            {
                "step": step.step,
                "loss": step.loss,
                "effective_sample_size_fraction": step.effective_sample_size_fraction,
                "gradient_norm": step.gradient_norm,
                "finite": True,
            }
        )
    validation = trainer.validation_batch(rows, log_weights)
    state = trainer.state_payload()
    state_identity = str(state["state_hash"])
    trainer.transport.bind_frozen_identity(
        {
            "checkpoint_sha256": state_identity,
            "training_state_hash": state_identity,
            "transport_tensor_hash": state_identity,
        }
    )
    bound = BatchNativeBoundAdapter(
        target, target_signature=target.target_signature()
    )
    transformed = FixedTransportValueScoreAdapter(
        base_adapter=bound,
        transport=trainer.transport,
        target_scope="ssl_lstm_q20_neutra_global_mixing_cpu_canary",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
        require_batch_native=True,
    )
    latent_representatives, representative_logdet = (
        trainer.transport.inverse_and_forward_logdet(means)
    )
    transformed_value, transformed_score = transformed.log_prob_and_grad(
        latent_representatives
    )
    physical_value, _physical_score, physical_status = (
        target.neutra_batch_log_prob_and_grad_status(means)
    )
    transformed_value_residual = tf.reduce_max(
        tf.abs(transformed_value - physical_value - representative_logdet)
    )
    transformed_status_valid = tf.reduce_all(
        tf.logical_and(
            tf.cast(physical_status["status_code"], tf.int32) == 0,
            tf.cast(physical_status["valid_pre_regularized_score"], tf.bool),
        )
    )
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_replay_canary.v1",
        "status": "REPLAY_TRAINING_CANARY_COMPLETED",
        "role": "cpu_xla_engineering_smoke_not_posterior_evidence",
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha256(PLAN),
        "geometry": GEOMETRY.as_posix(),
        "geometry_sha256": _sha256(GEOMETRY),
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
        "rows": int(args.rows),
        "updates": int(args.updates),
        "valid_rows": tf.reduce_sum(tf.cast(valid, tf.int32)),
        "proposal_labels": tf.reduce_sum(tf.cast(labels == 0, tf.int32)),
        "proposal_log_weight_ess_fraction": tf.math.reciprocal(
            tf.reduce_sum(tf.square(tf.nn.softmax(log_weights)))
        ) / tf.cast(tf.size(log_weights), tf.float64),
        "target_values_finite": tf.reduce_all(tf.math.is_finite(target_value)),
        "transport_validation_finite": tf.reduce_all(
            tf.math.is_finite(validation.latent)
        ),
        "exact_pullback_adapter": {
            "adapter_signature": transformed.adapter_signature(),
            "values_finite": tf.reduce_all(tf.math.is_finite(transformed_value)),
            "scores_finite": tf.reduce_all(tf.math.is_finite(transformed_score)),
            "target_status_valid": transformed_status_valid,
            "value_identity_maximum_absolute_residual": transformed_value_residual,
            "value_identity_tolerance": 1.0e-10,
            "value_identity_passed": transformed_value_residual
            <= tf.constant(1.0e-10, tf.float64),
        },
        "training_steps": steps,
        "heldout_weighted_nll": validation.loss,
        "jit_compile": True,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wall_seconds": time.perf_counter() - started,
        "git": _git_manifest(),
        "nonclaims": [
            "not a posterior archive",
            "not a global mode-discovery result",
            "not HMC evidence",
            "not a predictive-equivalence result",
        ],
    }
    _write(output / "result.json", result)
    _write(
        output / "manifest.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_replay_manifest.v1",
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "result": (output / "result.json").as_posix(),
            "cpu_only": True,
            "jit_compile": True,
        },
    )
    print(json.dumps({"status": result["status"], "output": output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
