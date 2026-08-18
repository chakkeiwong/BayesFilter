#!/usr/bin/env python3
"""Profile one frozen reverse-KL paper-d100 funnel update and evaluation."""

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

PLAN = ROOT / "docs/plans/bayesfilter-weighted-forward-kl-paper-d100-reverse-funnel-profile-plan-2026-08-14.md"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--device", default="1")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    training_root = args.training_root.resolve()
    if output.exists():
        raise FileExistsError(f"profile output must be fresh: {output}")
    required = (
        PLAN,
        training_root / "trainer_state.json",
        training_root / "run_manifest.json",
        training_root / "artifact_hashes.json",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("reverse-funnel profile input is missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()

    import tensorflow as tf

    from bayesfilter.inference.neutra_paper_d100_target import (
        PaperD100ValueScoreAdapter,
        make_paper_funnel_spec,
    )
    from bayesfilter.inference.neutra_weighted_training import (
        MatchedReverseKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(gpus) != 1:
        raise RuntimeError(f"expected one visible GPU, got {gpus}")

    state_path = training_root / "trainer_state.json"
    state = _load_json(state_path)
    manifest = _load_json(training_root / "run_manifest.json")
    hashes = _load_json(training_root / "artifact_hashes.json")
    if hashes.get("artifacts", {}).get(state_path.name) != _sha256(state_path):
        raise RuntimeError("reverse-funnel training-state hash mismatch")
    if state.get("objective") != "reverse_kl" or manifest.get("objective") != "reverse_kl":
        raise RuntimeError("profile requires a reverse-KL training state")
    target = state.get("target")
    if not isinstance(target, Mapping) or target.get("name") != "paper_funnel":
        raise RuntimeError("profile requires a paper funnel training state")

    config_payload = dict(state["config"])
    config_payload.pop("schema", None)
    config_payload["hidden_layers"] = tuple(config_payload["hidden_layers"])
    config_payload["initialization_seed"] = tuple(config_payload["initialization_seed"])
    config_payload["stage_s_max"] = tuple(config_payload.get("stage_s_max", ()))
    config = WeightedNeuTraConfig(**config_payload)
    spec = make_paper_funnel_spec()
    base = PaperD100ValueScoreAdapter(spec)
    target_value = lambda rows: base.log_prob(rows)
    trainer = MatchedReverseKLNeuTraTrainer(config, target_value)
    for variable, raw in zip(trainer.variables, state["variables"], strict=True):
        variable.assign(tf.constant(raw, tf.float64))

    selection_seed = (20260813, 53002 + 311)
    selection = tf.random.stateless_normal((65536, 100), seed=selection_seed, dtype=tf.float64)
    final_training_seed = tf.random.experimental.stateless_fold_in(
        tf.constant((20260813, 53003 + 311), tf.int32),
        int(state["selected_update"]),
    )
    batch = tf.random.stateless_normal((4096, 100), seed=final_training_seed, dtype=tf.float64)

    # Compile on the exact shapes, then restore both model and optimizer state.
    # The training artifact did not preserve Adam moments, so the measured
    # update is a zero-slot timing diagnostic at the frozen transport, not a
    # continuation claim.
    effective_learning_rate = float(config.learning_rate) * 0.01
    trainer.optimizer.learning_rate.assign(effective_learning_rate)
    frozen_variables = [variable.read_value() for variable in trainer.variables]
    frozen_optimizer = [tf.identity(variable) for variable in trainer.optimizer.variables]
    trainer.train_step(batch)
    physical, logdet = trainer.forward_and_logdet(selection)
    target_rows = target_value(physical)
    tf.debugging.assert_all_finite(target_rows, "reverse-funnel profile target rows")
    tf.debugging.assert_all_finite(logdet, "reverse-funnel profile logdet")
    for variable, frozen in zip(trainer.variables, frozen_variables, strict=True):
        variable.assign(frozen)
    for variable, frozen in zip(trainer.optimizer.variables, frozen_optimizer, strict=True):
        variable.assign(frozen)

    frozen_physical, frozen_logdet = trainer.forward_and_logdet(selection)
    frozen_row_loss = -target_value(frozen_physical) - frozen_logdet
    frozen_evaluation = tf.reduce_mean(frozen_row_loss)
    tf.debugging.assert_all_finite(frozen_row_loss, "reverse-funnel frozen heldout rows")

    profile_dir = output / "tensorflow_profile"
    tf.profiler.experimental.start(str(profile_dir))
    update_start = time.perf_counter()
    update = trainer.train_step(batch)
    update_wall = time.perf_counter() - update_start
    eval_start = time.perf_counter()
    physical, logdet = trainer.forward_and_logdet(selection)
    row_loss = -target_value(physical) - logdet
    evaluation = tf.reduce_mean(row_loss)
    tf.debugging.assert_all_finite(row_loss, "reverse-funnel profile heldout rows")
    eval_wall = time.perf_counter() - eval_start
    tf.profiler.experimental.stop()

    payload = {
        "schema": "bayesfilter.neutra.paper_d100_reverse_funnel_profile_result.v1",
        "plan": PLAN.as_posix(),
        "training_root": training_root.as_posix(),
        "training_state_sha256": _sha256(state_path),
        "target": "paper_funnel",
        "objective": "reverse_kl",
        "selection_seed": selection_seed,
        "final_training_seed": _ready(final_training_seed),
        "selected_update": int(state["selected_update"]),
        "effective_learning_rate": effective_learning_rate,
        "optimizer_state_provenance": "fresh_zero_slots_restored_after_compile_warmup",
        "batch_size": 4096,
        "heldout_rows_profiled": int(selection.shape[0]),
        "update_wall_seconds": update_wall,
        "heldout_eval_wall_seconds": eval_wall,
        "update_loss": float(update.loss.numpy()),
        "heldout_reverse_kl_before_profiled_update": float(frozen_evaluation.numpy()),
        "heldout_reverse_kl_after_profiled_update": float(evaluation.numpy()),
        "gradient_norm": float(update.gradient_norm.numpy()),
        "clipping_applied": bool(update.clipping_applied.numpy()),
        "jit_compile": True,
        "dtype": "float64",
        "tf32_enabled": False,
        "memory_policy": _ready(policy),
        "gpu": str(gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "profile_directory": profile_dir.as_posix(),
        "profile_directory_exists": profile_dir.is_dir(),
        "execution_target": "gpu_xla_diagnostic_only",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "command": " ".join(sys.argv),
        "wall_seconds": time.perf_counter() - started,
        "nonclaims": [
            "no training objective change",
            "no HMC or scientific validity claim",
            "runtime profile only",
            "optimizer moments were not stored by the training artifact; update timing uses zero Adam slots",
        ],
    }
    _write(output / "result.json", payload)
    _write(output / "run_manifest.json", payload)
    _write(
        output / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.paper_d100_reverse_funnel_profile_hashes.v1",
            "artifacts": {
                path.relative_to(output).as_posix(): _sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"output_root": output.as_posix(), "update_wall_seconds": update_wall, "heldout_eval_wall_seconds": eval_wall}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
