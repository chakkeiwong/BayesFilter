#!/usr/bin/env python3
"""Train one fresh reverse- or exact-replay forward-KL paper d100 transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-paper-d100-fresh-baseline-plan-2026-08-13.md"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--gaussian-constants", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("paper_funnel", "paper_ill_cond_gaussian"), required=True
    )
    parser.add_argument("--objective", choices=("reverse_kl", "forward_kl"), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--hidden-width", type=int, default=100)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "paper_piecewise"),
        required=True,
    )
    parser.add_argument("--initialization-seed-offset", type=int, default=0)
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


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _load_tensor(tf: Any, root: Path, receipt: Mapping[str, Any]) -> Any:
    path = root / str(receipt.get("path", ""))
    if not path.is_file() or _sha256(path) != str(receipt.get("sha256", "")):
        raise RuntimeError(f"replay tensor hash mismatch: {path.name}")
    with tf.device("/CPU:0"):
        tensor = tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)
    expected_shape = tuple(int(value) for value in receipt.get("shape", ()))
    if tensor.shape != expected_shape or tensor.shape.rank != 2 or tensor.shape[1] != 100:
        raise RuntimeError(f"replay tensor shape mismatch: {path.name}")
    tf.debugging.assert_all_finite(tensor, f"replay tensor {path.name}")
    return tensor


def _learning_rate(args: argparse.Namespace, update: int) -> float:
    base = float(args.learning_rate)
    if args.learning_rate_schedule == "constant":
        return base
    if int(update) > 4000:
        return base * 0.01
    if int(update) > 1000:
        return base * 0.1
    return base


def _assign_learning_rate(trainer: Any, value: float) -> None:
    learning_rate = trainer.optimizer.learning_rate
    if not hasattr(learning_rate, "assign"):
        raise RuntimeError("optimizer learning rate is not assignable")
    learning_rate.assign(float(value))


def _reverse_loss(tf: Any, trainer: Any, target_value: Any, latent: Any) -> float:
    physical, logdet = trainer.forward_and_logdet(latent)
    row_loss = -target_value(physical) - logdet
    tf.debugging.assert_all_finite(row_loss, "paper d100 reverse heldout rows")
    return float(tf.reduce_mean(row_loss).numpy())


def _forward_nll(tf: Any, trainer: Any, physical: Any) -> float:
    row_loss = -trainer.log_prob(physical)
    tf.debugging.assert_all_finite(row_loss, "paper d100 exact heldout NLL rows")
    return float(tf.reduce_mean(row_loss).numpy())


def _training_batch(rows: Any, update: int, batch_size: int) -> Any:
    row_count = int(rows.shape[0])
    if row_count % int(batch_size) != 0:
        raise RuntimeError("training replay row count must be divisible by batch size")
    start = ((int(update) - 1) * int(batch_size)) % row_count
    batch = rows[start : start + int(batch_size)]
    if batch.shape != (int(batch_size), 100):
        raise RuntimeError("training replay batch shape mismatch")
    return batch


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("updates", "batch_size", "checkpoint_every", "hidden_width", "stages"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.batch_size) <= 1:
        raise ValueError("batch size must exceed one")
    if int(args.checkpoint_every) > int(args.updates):
        raise ValueError("checkpoint interval exceeds update count")
    if not math.isfinite(float(args.learning_rate)) or float(args.learning_rate) <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if int(args.updates) > 10_000:
        raise ValueError("paper d100 training is capped at 10000 updates")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    output_root = args.output_root.resolve()
    replay_root = args.replay_root.resolve()
    constants_path = args.gaussian_constants.resolve()
    replay_manifest_path = replay_root / "replay_manifest.json"
    replay_hashes_path = replay_root / "artifact_hashes.json"
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    required = (PLAN, constants_path, replay_manifest_path, replay_hashes_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("plan, constants, or replay evidence is missing")
    output_root.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    import tensorflow as tf

    from bayesfilter.inference.neutra_paper_d100_target import (
        PaperD100ValueScoreAdapter,
        load_paper_gaussian_spec,
        make_paper_funnel_spec,
        paper_d100_log_prob_batch,
    )
    from bayesfilter.inference.neutra_weighted_training import (
        MatchedReverseKLNeuTraTrainer,
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical_gpus}")
    spec = (
        make_paper_funnel_spec()
        if args.target == "paper_funnel"
        else load_paper_gaussian_spec(constants_path)
    )
    adapter = PaperD100ValueScoreAdapter(spec)
    replay_manifest = _load_json(replay_manifest_path)
    replay_hashes = _load_json(replay_hashes_path)
    if replay_manifest.get("schema") != "bayesfilter.neutra.paper_d100_exact_replay.v1":
        raise RuntimeError("replay manifest schema mismatch")
    if replay_manifest.get("target", {}).get("name") != spec.name:
        raise RuntimeError("replay target mismatch")
    if replay_hashes.get("artifacts", {}).get(replay_manifest_path.name) != _sha256(
        replay_manifest_path
    ):
        raise RuntimeError("replay manifest artifact hash mismatch")
    receipts = replay_manifest.get("receipts")
    if not isinstance(receipts, Mapping):
        raise RuntimeError("replay receipts are missing")
    training_rows = _load_tensor(tf, replay_root, receipts["training_rows"])
    selection_rows = _load_tensor(tf, replay_root, receipts["selection_rows"])
    selection_count = int(selection_rows.shape[0])
    seed_offset = int(args.initialization_seed_offset)
    config = WeightedNeuTraConfig(
        dimension=100,
        hidden_layers=(int(args.hidden_width), int(args.hidden_width)),
        stages=int(args.stages),
        activation="elu",
        s_max=1.0,
        initialization_scale=0.02,
        initialization_seed=(20260813, 53001 + seed_offset),
        learning_rate=float(args.learning_rate),
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    target_value = lambda rows: paper_d100_log_prob_batch(spec, rows)
    trainer: Any
    if args.objective == "reverse_kl":
        trainer = MatchedReverseKLNeuTraTrainer(config, target_value)
        selection_latent = tf.random.stateless_normal(
            (selection_count, 100),
            seed=(20260813, 53002 + seed_offset),
            dtype=tf.float64,
        )
    else:
        trainer = WeightedForwardKLNeuTraTrainer(config)
        selection_latent = None
    checkpoints = []
    selected_variables: list[Any] | None = None
    selected_update = 0
    selected_loss = float("inf")
    clipped_updates = 0
    last_step: Mapping[str, Any] = {}
    for update in range(1, int(args.updates) + 1):
        current_lr = _learning_rate(args, update)
        _assign_learning_rate(trainer, current_lr)
        if args.objective == "reverse_kl":
            latent = tf.random.stateless_normal(
                (int(args.batch_size), 100),
                seed=tf.random.experimental.stateless_fold_in(
                    tf.constant((20260813, 53003 + seed_offset), tf.int32), update
                ),
                dtype=tf.float64,
            )
            step = trainer.train_step(latent)
        else:
            batch = _training_batch(training_rows, update, int(args.batch_size))
            step = trainer.train_step(
                batch, tf.zeros((int(args.batch_size),), tf.float64)
            )
        clipped_updates += int(bool(step.clipping_applied.numpy()))
        last_step = _ready(step.__dict__)
        if update % int(args.checkpoint_every) == 0 or update == int(args.updates):
            if args.objective == "reverse_kl":
                selection_loss = _reverse_loss(
                    tf, trainer, target_value, selection_latent
                )
                selection_metric = "heldout_reverse_kl"
            else:
                selection_loss = _forward_nll(tf, trainer, selection_rows)
                selection_metric = "heldout_exact_forward_nll"
            checkpoint = {
                "update": update,
                "learning_rate": current_lr,
                "selection_metric": selection_metric,
                "selection_loss": selection_loss,
                "training_loss": float(step.loss.numpy()),
                "gradient_norm": float(step.gradient_norm.numpy()),
                "clipping_applied": bool(step.clipping_applied.numpy()),
            }
            checkpoints.append(checkpoint)
            if selection_loss < selected_loss:
                selected_loss = selection_loss
                selected_update = update
                selected_variables = [
                    variable.numpy().tolist() for variable in trainer.variables
                ]
    if selected_variables is None or selected_update <= 0:
        raise RuntimeError("paper d100 training produced no finite checkpoint")
    for variable, raw in zip(trainer.variables, selected_variables, strict=True):
        variable.assign(tf.constant(raw, tf.float64))

    # The untouched audit partition is not parsed until checkpoint selection is frozen.
    audit_rows = _load_tensor(tf, replay_root, receipts["audit_rows"])
    audit_exact_forward_nll = _forward_nll(tf, trainer, audit_rows)
    audit_reverse_kl = None
    if args.objective == "reverse_kl":
        audit_latent = tf.random.stateless_normal(
            (int(audit_rows.shape[0]), 100),
            seed=(20260813, 53004 + seed_offset),
            dtype=tf.float64,
        )
        audit_reverse_kl = _reverse_loss(tf, trainer, target_value, audit_latent)
    state_payload = {
        "schema": "bayesfilter.neutra.paper_d100_training_state.v1",
        "objective": args.objective,
        "target": spec.manifest_payload(),
        "replay_manifest_sha256": _sha256(replay_manifest_path),
        "selected_update": selected_update,
        "config": config.manifest_payload(),
        "variables": selected_variables,
    }
    state_payload["state_hash"] = _stable_hash(state_payload)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    manifest = {
        "schema": "bayesfilter.neutra.paper_d100_training_manifest.v1",
        "plan": PLAN.as_posix(),
        "target": spec.manifest_payload(),
        "adapter_signature": adapter.adapter_signature(),
        "objective": args.objective,
        "config": config.manifest_payload(),
        "updates": int(args.updates),
        "training_batch_size": int(args.batch_size),
        "checkpoint_every": int(args.checkpoint_every),
        "learning_rate_schedule": args.learning_rate_schedule,
        "learning_rate_schedule_semantics": (
            "base through update 1000; 0.1*base updates 1001-4000; "
            "0.01*base after update 4000"
            if args.learning_rate_schedule == "paper_piecewise"
            else "constant"
        ),
        "selection_size": selection_count,
        "audit_size": int(audit_rows.shape[0]),
        "selection_audit_disjoint": True,
        "audit_opened_after_checkpoint_freeze": True,
        "replay_root": replay_root.as_posix(),
        "replay_manifest": replay_manifest_path.as_posix(),
        "replay_manifest_sha256": _sha256(replay_manifest_path),
        "seeds": {
            "initialization": [20260813, 53001 + seed_offset],
            "reverse_selection": [20260813, 53002 + seed_offset],
            "reverse_training_root": [20260813, 53003 + seed_offset],
            "reverse_audit": [20260813, 53004 + seed_offset],
            "exact_replay": replay_manifest.get("seeds"),
        },
        "batch_native_target_backend": "tensorflow_exact_paper_d100",
        "sample_wise_loop_or_scalar_fallback": False,
        "external_exact_sample_generation_cpu_only": True,
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_policy": _ready(memory_policy),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "allocator_bytes": _ready(allocator),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "tensorflow_version": tf.__version__,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "command": " ".join(sys.argv),
        "wall_seconds": time.perf_counter() - started,
    }
    result = {
        "schema": "bayesfilter.neutra.paper_d100_training_result.v1",
        "candidate_passed": True,
        "candidate_status": "training_candidate_only",
        "target": spec.name,
        "objective": args.objective,
        "selected_update": selected_update,
        "selection_metric": checkpoints[-1]["selection_metric"],
        "selection_loss": selected_loss,
        "audit_exact_forward_nll": audit_exact_forward_nll,
        "audit_reverse_kl": audit_reverse_kl,
        "checkpoints": checkpoints,
        "clipped_updates": clipped_updates,
        "last_step": last_step,
        "manifest": manifest,
        "nonclaims": (
            "finite selected training checkpoint only",
            "training and audit loss do not establish posterior quality",
            "no HMC, objective ranking, paper replication, or default promotion",
        ),
    }
    _write(output_root / "trainer_state.json", state_payload)
    _write(output_root / "run_manifest.json", manifest)
    _write(output_root / "result.json", result)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.paper_d100_training_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(output_root.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "target": spec.name,
                "objective": args.objective,
                "selected_update": selected_update,
                "selection_loss": selected_loss,
                "output_root": output_root.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
