#!/usr/bin/env python3
"""GPU/XLA local-affine weighted NeuTra canary for ``nk_like_strong_smooth``.

The replay proposal is frozen by the preceding CPU-only source-bound diagnostic.
The learned IAF operates in the frozen affine-lift coordinates.  This script is
only a training-route canary: no HMC, posterior, or cross-objective claim is
made here.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-strong-smooth-reflection-proposal-repair-plan-2026-08-12.md"
)
PROPOSAL_ROOT = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "varying-hessian/strong-smooth-proposal-r7-reflected"
)
PROPOSAL = PROPOSAL_ROOT / "selected_proposal.json"
CONSTANTS = Path(
    "/home/ubuntu/python/dsge_hmc/results/neutra/gate3/"
    "nk_strong_smooth_bridge_20260604/frozen_constants/"
    "strong_smooth_from_seed42_affine_lift.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--target-name", choices=("nk_like_mild_smooth", "nk_like_strong_smooth"), default="nk_like_strong_smooth")
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=16_384)
    parser.add_argument("--audit-size", type=int, default=16_384)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
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


def _proposal_payload(path: Path, target_name: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(payload.get("proposal_hash", ""))
    hashed = {key: value for key, value in payload.items() if key != "proposal_hash"}
    if len(expected_hash) != 64 or _stable_hash(hashed) != expected_hash:
        raise RuntimeError("frozen replay proposal hash mismatch")
    if payload.get("target_name") != target_name:
        raise RuntimeError("frozen replay proposal target mismatch")
    proposal = payload.get("proposal")
    if not isinstance(proposal, Mapping):
        raise RuntimeError("frozen replay proposal payload is missing")
    return payload


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "updates",
        "batch_size",
        "selection_size",
        "audit_size",
        "checkpoint_every",
        "hidden_width",
        "stages",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.batch_size) <= 1:
        raise ValueError("batch_size must exceed one")
    if int(args.checkpoint_every) > int(args.updates):
        raise ValueError("checkpoint_every must not exceed updates")
    if not float(args.learning_rate) > 0.0:
        raise ValueError("learning_rate must be positive")
    if int(args.updates) >= 10000 and (
        int(args.hidden_width) != 128
        or int(args.stages) != 6
        or float(args.learning_rate) != 1.0e-3
    ):
        raise ValueError("serious strong-smooth rung is frozen to (128,128), six stages, learning rate 1e-3")


def _cpu_replay_rows(spec: Any, proposal: Mapping[str, Any], count: int, seed: tuple[int, int]) -> tuple[Any, Any]:
    from bayesfilter.inference.neutra_varying_hessian_target import (
        physical_to_affine_local,
        varying_hessian_log_prob_and_score_batch,
    )
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob,
        sample_gaussian_mixture,
    )

    physical, _labels = sample_gaussian_mixture(
        int(count),
        proposal["probabilities"],
        proposal["means"],
        proposal["covariances"],
        seed=seed,
    )
    target, score = varying_hessian_log_prob_and_score_batch(spec, physical)
    del score
    log_proposal = gaussian_mixture_log_prob(
        physical,
        proposal["probabilities"],
        proposal["means"],
        proposal["covariances"],
    )
    return physical_to_affine_local(spec, physical), target - log_proposal


def _checkpoint(
    tf: Any, trainer: Any, rows: Any, log_weights: Any, update: int
) -> Mapping[str, Any]:
    validation = trainer.validation_batch(rows, log_weights)
    return {
        "update": int(update),
        "heldout_weighted_nll": validation.loss,
        "heldout_effective_sample_size_fraction": validation.effective_sample_size_fraction,
        "heldout_maximum_normalized_weight": validation.maximum_normalized_weight,
        "latent_weighted_mean_l2": tf.linalg.norm(validation.latent_weighted_mean),
        "latent_weighted_covariance_trace": tf.linalg.trace(validation.latent_weighted_covariance),
    }


def _load_replay_tensor(path: Path, expected_sha256: str, tf: Any) -> Any:
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"replay tensor SHA-256 mismatch: {path.name}")
    return tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)


def _load_replay(replay_root: Path, tf: Any, proposal_path: Path, target_name: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest_path = replay_root / "replay_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "bayesfilter.weighted_neutra_strong_smooth_cpu_replay.v1"
    ):
        raise RuntimeError("replay manifest schema mismatch")
    proposal_payload = _proposal_payload(proposal_path, target_name)
    if payload.get("proposal_sha256") != _sha256(proposal_path):
        raise RuntimeError("replay proposal file hash mismatch")
    if payload.get("proposal_hash") != proposal_payload.get("proposal_hash"):
        raise RuntimeError("replay proposal semantic hash mismatch")
    receipts = payload.get("receipts")
    if not isinstance(receipts, Mapping):
        raise RuntimeError("replay receipts are missing")
    tensors = {}
    for name in (
        "training_local_rows",
        "training_log_weights",
        "selection_local_rows",
        "selection_log_weights",
        "audit_local_rows",
        "audit_log_weights",
    ):
        receipt = receipts.get(name)
        if not isinstance(receipt, Mapping):
            raise RuntimeError(f"replay receipt missing: {name}")
        tensors[name] = _load_replay_tensor(
            replay_root / str(receipt.get("path", "")),
            str(receipt.get("sha256", "")),
            tf,
        )
    selection_rows = tensors["selection_local_rows"]
    audit_rows = tensors["audit_local_rows"]
    training_rows = tensors["training_local_rows"]
    if (
        training_rows.shape.rank != 2
        or selection_rows.shape.rank != 2
        or audit_rows.shape.rank != 2
    ):
        raise RuntimeError("replay local rows must be rank two")
    if (
        tensors["training_log_weights"].shape != (training_rows.shape[0],)
        or tensors["selection_log_weights"].shape != (selection_rows.shape[0],)
        or tensors["audit_log_weights"].shape != (audit_rows.shape[0],)
    ):
        raise RuntimeError("replay row and log-weight shapes disagree")
    return payload, tensors


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    output_root = args.output_root.resolve()
    replay_root = args.replay_root.resolve()
    proposal_path = args.proposal.resolve()
    constants_path = args.constants.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    if not replay_root.is_dir():
        raise FileNotFoundError("CPU-only replay root is missing")
    if not PLAN.is_file() or not proposal_path.is_file() or not constants_path.is_file():
        raise FileNotFoundError("plan, frozen proposal, or frozen source constants missing")
    output_root.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.monotonic()
    import tensorflow as tf

    from bayesfilter.inference.neutra_weighted_training import (
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
        raise RuntimeError(f"expected exactly one visible logical GPU, found {logical_gpus}")
    replay_manifest, replay = _load_replay(replay_root, tf, proposal_path, args.target_name)
    training_rows = tf.identity(replay["training_local_rows"])
    training_weights = tf.identity(replay["training_log_weights"])
    selection_rows = tf.identity(replay["selection_local_rows"])
    selection_weights = tf.identity(replay["selection_log_weights"])
    audit_rows = tf.identity(replay["audit_local_rows"])
    audit_weights = tf.identity(replay["audit_log_weights"])
    if int(selection_rows.shape[1]) != 9:
        raise RuntimeError("replay local dimension must be nine")
    config = WeightedNeuTraConfig(
        dimension=int(selection_rows.shape[1]),
        hidden_layers=(int(args.hidden_width), int(args.hidden_width)),
        stages=int(args.stages),
        activation="tanh",
        initialization_scale=0.02,
        initialization_seed=(20260812, 16011),
        learning_rate=float(args.learning_rate),
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    trainer = WeightedForwardKLNeuTraTrainer(config)
    checkpoints = [_checkpoint(tf, trainer, selection_rows, selection_weights, 0)]
    best_state = [variable.numpy().copy() for variable in trainer.variables]
    best_nll = float(checkpoints[0]["heldout_weighted_nll"].numpy())
    best_update = 0
    clipped_updates = 0
    last_step: Mapping[str, Any] | None = None
    for update in range(1, int(args.updates) + 1):
        offset = ((update - 1) * int(args.batch_size)) % int(training_rows.shape[0])
        indices = tf.math.floormod(
            tf.range(int(args.batch_size), dtype=tf.int32) + int(offset),
            int(training_rows.shape[0]),
        )
        rows = tf.gather(training_rows, indices)
        weights = tf.gather(training_weights, indices)
        step = trainer.train_step(rows, weights)
        clipped_updates += int(bool(step.clipping_applied.numpy()))
        last_step = {
            "loss": step.loss,
            "effective_sample_size_fraction": step.effective_sample_size_fraction,
            "maximum_normalized_weight": step.maximum_normalized_weight,
            "gradient_norm": step.gradient_norm,
            "clipped_gradient_norm": step.clipped_gradient_norm,
        }
        if update % int(args.checkpoint_every) == 0 or update == int(args.updates):
            checkpoint = _checkpoint(tf, trainer, selection_rows, selection_weights, update)
            checkpoints.append(checkpoint)
            nll = float(checkpoint["heldout_weighted_nll"].numpy())
            if nll < best_nll:
                best_nll = nll
                best_update = update
                best_state = [variable.numpy().copy() for variable in trainer.variables]
    for variable, value in zip(trainer.variables, best_state):
        variable.assign(value)
    audit = _checkpoint(tf, trainer, audit_rows, audit_weights, best_update)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    finite = bool(
        tf.reduce_all(tf.math.is_finite(audit["heldout_weighted_nll"])).numpy()
        and all(bool(tf.reduce_all(tf.math.is_finite(variable)).numpy()) for variable in trainer.variables)
    )
    selected = bool(best_update > 0 and finite)
    manifest = {
        "schema": "bayesfilter.weighted_neutra_strong_smooth_canary_manifest.v1",
        "plan": PLAN.as_posix(),
        "proposal_path": proposal_path.as_posix(),
        "proposal_sha256": _sha256(proposal_path),
        "proposal_hash": replay_manifest["proposal_hash"],
        "source_constants_path": constants_path.as_posix(),
        "source_constants_sha256": replay_manifest["source_constants_sha256"],
        "target": replay_manifest["target"],
        "replay_manifest_path": (replay_root / "replay_manifest.json").as_posix(),
        "replay_manifest_sha256": _sha256(replay_root / "replay_manifest.json"),
        "command": " ".join(sys.argv),
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_policy": memory_policy,
        "allocator_bytes": {key: int(value) for key, value in allocator.items()},
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "training_coordinates": "frozen_affine_lift_local_x",
        "composed_physical_map": "theta = mu + L @ IAF(z)",
        "batch_native_target_backend": "tensorflow_precomputed_batch_native_replay_rows",
        "sample_wise_loop_or_scalar_fallback": False,
        "training_batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "training_replay_size": int(training_rows.shape[0]),
        "updates": int(args.updates),
        "selection_size": int(selection_rows.shape[0]),
        "audit_size": int(audit_rows.shape[0]),
        "cpu_replay_seeds": replay_manifest["replay_seeds"],
        "training_selection_audit_disjoint": bool(
            replay_manifest.get("partitions_disjoint_by_stateless_seed")
        ),
        "wall_seconds": time.monotonic() - started,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }
    result = {
        "schema": "bayesfilter.weighted_neutra_strong_smooth_canary_result.v1",
        "research_question": "Does source-bound local-coordinate weighted forward-KL execute batch-native GPU/XLA updates with a finite frozen candidate?",
        "candidate_passed": selected,
        "candidate_status": "nominated_for_separate_hmc_review" if selected else "training_canary_rejected",
        "config": config.manifest_payload(),
        "checkpoints": checkpoints,
        "selection": {"criterion": "minimum disjoint heldout weighted NLL", "update": best_update, "nll": best_nll},
        "audit": audit,
        "clipped_updates": clipped_updates,
        "last_step": last_step,
        "manifest": manifest,
        "nonclaims": (
            "training canary only",
            "no reverse-KL comparator in this canary",
            "no HMC, posterior, or historical-regression success claim",
        ),
    }
    _write(output_root / "result.json", result)
    _write(output_root / "run_manifest.json", manifest)
    _write(
        output_root / "trainer_state.json",
        {
            "schema": "bayesfilter.weighted_neutra_strong_smooth_local_state.v1",
            "selected_update": best_update,
            "config": config.manifest_payload(),
            "variables": [variable.numpy().tolist() for variable in trainer.variables],
            "state_hash": _stable_hash({
                "config": config.manifest_payload(),
                "selected_update": best_update,
                "variables": [variable.numpy().tolist() for variable in trainer.variables],
            }),
        },
    )
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.weighted_neutra_strong_smooth_canary_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(output_root.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"passed": selected, "output_root": output_root.as_posix(), "best_update": best_update}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
