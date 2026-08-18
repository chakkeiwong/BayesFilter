#!/usr/bin/env python3
"""Train a matched reverse-KL German-credit proposal/comparator transport."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--updates", type=int, choices=(200, 1000, 3000), required=True)
    parser.add_argument("--hidden-width", type=int, default=51)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-size", type=int, default=8192)
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "as_list"):
        return _ready(value.as_list())
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


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _heldout_loss(trainer: Any, target_value: Any, latent: Any) -> float:
    import tensorflow as tf

    physical, logdet = trainer.forward_and_logdet(latent)
    values = target_value(physical)
    loss = tf.reduce_mean(-values - logdet)
    tf.debugging.assert_all_finite(loss, "German reverse-KL heldout loss")
    return float(loss.numpy())


def main() -> int:
    args = _parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"output root must be fresh: {root}")
    if int(args.batch_size) <= 1 or int(args.validation_size) <= 1:
        raise ValueError("batch and validation sizes must exceed one")
    if not PLAN.is_file() or not args.data.is_file() or not args.reference.is_file():
        raise FileNotFoundError("plan, data, or reference is missing")
    root.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import tensorflow as tf

    from bayesfilter.inference.neutra_german_credit_target import (
        GermanCreditValueScoreAdapter,
        german_credit_log_prob_batch,
        load_german_credit_target_spec,
    )
    from bayesfilter.inference.neutra_weighted_training import (
        MatchedReverseKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    started = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical_gpus}")
    spec = load_german_credit_target_spec(args.data, args.reference)
    adapter = GermanCreditValueScoreAdapter(spec)
    width = int(args.hidden_width)
    stages = int(args.stages)
    if width < 1 or stages < 1 or float(args.learning_rate) <= 0.0:
        raise ValueError("German repair architecture and learning rate must be positive")
    config = WeightedNeuTraConfig(
        dimension=spec.dimension,
        hidden_layers=(width, width),
        stages=stages,
        activation="elu",
        s_max=1.0,
        initialization_scale=0.02,
        initialization_seed=(20260813, 43001),
        learning_rate=float(args.learning_rate),
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    target_value = lambda rows: german_credit_log_prob_batch(spec, rows)
    trainer = MatchedReverseKLNeuTraTrainer(config, target_value)
    selection_latent = tf.random.stateless_normal(
        (int(args.validation_size), spec.dimension),
        seed=(20260813, 43002),
        dtype=tf.float64,
    )
    audit_latent = tf.random.stateless_normal(
        (int(args.validation_size), spec.dimension),
        seed=(20260813, 43003),
        dtype=tf.float64,
    )
    checkpoint_every = 50 if int(args.updates) == 200 else 100
    checkpoints = []
    selected_variables: list[Any] | None = None
    selected_update = 0
    selected_loss = float("inf")
    clipped_updates = 0
    last_step: Mapping[str, Any] = {}
    for update in range(1, int(args.updates) + 1):
        latent = tf.random.stateless_normal(
            (int(args.batch_size), spec.dimension),
            seed=tf.random.experimental.stateless_fold_in(
                tf.constant((20260813, 43004), tf.int32), update
            ),
            dtype=tf.float64,
        )
        step = trainer.train_step(latent)
        clipped_updates += int(bool(step.clipping_applied.numpy()))
        last_step = _ready(step.__dict__)
        if update % checkpoint_every == 0:
            selection_loss = _heldout_loss(trainer, target_value, selection_latent)
            checkpoints.append(
                {
                    "update": update,
                    "selection_reverse_kl": selection_loss,
                    "training_loss": float(step.loss.numpy()),
                    "gradient_norm": float(step.gradient_norm.numpy()),
                    "clipping_applied": bool(step.clipping_applied.numpy()),
                }
            )
            if selection_loss < selected_loss:
                selected_loss = selection_loss
                selected_update = update
                selected_variables = [variable.numpy().tolist() for variable in trainer.variables]
    if selected_variables is None or selected_update <= 0:
        raise RuntimeError("German reverse-KL training produced no finite checkpoint")
    for variable, raw in zip(trainer.variables, selected_variables, strict=True):
        variable.assign(tf.constant(raw, tf.float64))
    audit_loss = _heldout_loss(trainer, target_value, audit_latent)
    state_payload = {
        "schema": "bayesfilter.weighted_neutra_german_reverse_state.v1",
        "selected_update": selected_update,
        "config": config.manifest_payload(),
        "target_name": spec.name,
        "target_data_sha256": spec.data_sha256,
        "target_reference_sha256": spec.reference_sha256,
        "variables": selected_variables,
    }
    state_payload["state_hash"] = _stable_hash(state_payload)
    manifest = {
        "schema": "bayesfilter.weighted_neutra_german_reverse_manifest.v1",
        "plan": PLAN.as_posix(),
        "target": spec.manifest_payload(),
        "adapter_signature": adapter.adapter_signature(),
        "config": config.manifest_payload(),
        "training_objective": "matched_reverse_kl",
        "training_batch_size": int(args.batch_size),
        "updates": int(args.updates),
        "selection_size": int(args.validation_size),
        "audit_size": int(args.validation_size),
        "selection_audit_disjoint": True,
        "seeds": {
            "initialization": [20260813, 43001],
            "selection": [20260813, 43002],
            "audit": [20260813, 43003],
            "training_root": [20260813, 43004],
        },
        "batch_native_target_backend": "tensorflow_exact_german_credit_sparse_logistic",
        "sample_wise_loop_or_scalar_fallback": False,
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_policy": memory_policy,
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "tensorflow_version": tf.__version__,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "command": " ".join(sys.argv),
        "wall_seconds": time.perf_counter() - started,
    }
    result = {
        "schema": "bayesfilter.weighted_neutra_german_reverse_result.v1",
        "candidate_passed": True,
        "candidate_status": "proposal_and_comparator_candidate_only",
        "selected_update": selected_update,
        "selection_reverse_kl": selected_loss,
        "audit_reverse_kl": audit_loss,
        "checkpoints": checkpoints,
        "clipped_updates": clipped_updates,
        "last_step": last_step,
        "manifest": manifest,
        "nonclaims": (
            "reverse-KL transport proposal/comparator only",
            "training loss does not establish posterior quality",
            "no HMC, posterior, weighted-objective, or ranking claim",
        ),
    }
    _write(root / "trainer_state.json", state_payload)
    _write(root / "run_manifest.json", manifest)
    _write(root / "result.json", result)
    _write(
        root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.weighted_neutra_german_reverse_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(root.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "selected_update": selected_update,
                "output_root": root.as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
