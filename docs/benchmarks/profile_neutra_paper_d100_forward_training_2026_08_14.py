#!/usr/bin/env python3
"""Profile one forward-KL d100 update and one heldout likelihood evaluation."""

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
PLAN = ROOT / "docs/plans/bayesfilter-weighted-forward-kl-paper-d100-repair-plan-2026-08-14.md"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--gaussian-constants", type=Path, required=True)
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
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(v) for v in value]
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
    path.write_text(json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_tensor(tf: Any, root: Path, receipt: Mapping[str, Any]) -> Any:
    path = root / str(receipt["path"])
    if _sha256(path) != str(receipt["sha256"]):
        raise RuntimeError(f"replay hash mismatch: {path}")
    return tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    training_root = args.training_root.resolve()
    replay_root = args.replay_root.resolve()
    constants = args.gaussian_constants.resolve()
    required = (PLAN, constants, training_root / "trainer_state.json", training_root / "run_manifest.json", training_root / "artifact_hashes.json", replay_root / "replay_manifest.json", replay_root / "artifact_hashes.json")
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("profile input is missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    import tensorflow as tf
    from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer, WeightedNeuTraConfig
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(gpus) != 1:
        raise RuntimeError(f"expected one visible GPU, got {gpus}")
    state = _load_json(training_root / "trainer_state.json")
    manifest = _load_json(training_root / "run_manifest.json")
    if state.get("objective") != "forward_kl" or manifest.get("objective") != "forward_kl":
        raise RuntimeError("profile requires a forward-KL training state")
    config_payload = dict(state["config"])
    config_payload.pop("schema", None)
    config_payload["hidden_layers"] = tuple(config_payload["hidden_layers"])
    config_payload["initialization_seed"] = tuple(config_payload["initialization_seed"])
    config_payload["stage_s_max"] = tuple(config_payload.get("stage_s_max", ()))
    config = WeightedNeuTraConfig(**config_payload)
    trainer = WeightedForwardKLNeuTraTrainer(config)
    for variable, raw in zip(trainer.variables, state["variables"], strict=True):
        variable.assign(tf.constant(raw, tf.float64))
    replay = _load_json(replay_root / "replay_manifest.json")
    receipts = replay["receipts"]
    training = _load_tensor(tf, replay_root, receipts["training_rows"])
    selection = _load_tensor(tf, replay_root, receipts["selection_rows"])
    batch = training[:4096]
    selection_profile = selection[:65536]
    zeros_batch = tf.zeros((4096,), tf.float64)
    zeros_selection = tf.zeros((int(selection_profile.shape[0]),), tf.float64)
    trainer.train_step(batch, zeros_batch)
    trainer.validation_batch(selection_profile, zeros_selection)
    profile_dir = output / "tensorflow_profile"
    tf.profiler.experimental.start(str(profile_dir))
    update_start = time.perf_counter()
    update = trainer.train_step(batch, zeros_batch)
    update_wall = time.perf_counter() - update_start
    eval_start = time.perf_counter()
    validation = trainer.validation_batch(selection_profile, zeros_selection)
    eval_wall = time.perf_counter() - eval_start
    tf.profiler.experimental.stop()
    payload = {
        "schema": "bayesfilter.neutra.paper_d100_forward_profile_result.v1",
        "plan": PLAN.as_posix(), "training_root": training_root.as_posix(),
        "training_state_sha256": _sha256(training_root / "trainer_state.json"),
        "replay_root": replay_root.as_posix(), "target": "paper_ill_cond_gaussian", "objective": "forward_kl",
        "batch_size": 4096, "heldout_rows_profiled": int(selection_profile.shape[0]),
        "update_wall_seconds": update_wall, "heldout_eval_wall_seconds": eval_wall,
        "selected_update_loss": float(update.loss.numpy()), "selected_eval_loss": float(validation.loss.numpy()),
        "clipping_applied": bool(update.clipping_applied.numpy()), "jit_compile": True, "dtype": "float64", "tf32_enabled": False,
        "memory_policy": _ready(policy), "gpu": str(gpus[0]), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "profile_directory": profile_dir.as_posix(), "profile_directory_exists": profile_dir.is_dir(),
        "execution_target": "gpu_xla_diagnostic_only",
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "command": " ".join(sys.argv), "wall_seconds": time.perf_counter() - started,
        "nonclaims": ["no training objective change", "no HMC or scientific validity claim", "runtime profile only"],
    }
    _write(output / "result.json", payload)
    _write(output / "run_manifest.json", payload)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.paper_d100_forward_profile_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"update_wall_seconds": update_wall, "heldout_eval_wall_seconds": eval_wall, "output_root": output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
