#!/usr/bin/env python3
"""Run one reviewed adaptive five-stage repair cell on a known-law target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

PLAN = ROOT / "docs/plans/bayesfilter-neutra-generic-adaptive-five-stage-repair-plan-2026-08-15.md"
BASE_RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py"


def _base_runner() -> Any:
    spec = importlib.util.spec_from_file_location("generic_five_stage_base_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generic five-stage base runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", choices=("gaussian", "banana"), required=True)
    parser.add_argument(
        "--route", choices=("adaptive_reset", "adaptive_carry", "cold"), required=True
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rates", default="2e-4,5e-4,1e-3")
    parser.add_argument("--affine-updates", type=int, default=100)
    parser.add_argument("--simple-updates", type=int, default=300)
    parser.add_argument("--progressive-updates", type=int, default=100)
    parser.add_argument("--joint-updates", type=int, default=2300)
    parser.add_argument("--cold-updates", type=int, default=3000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--adaptive-minimum-updates", type=int, default=400)
    parser.add_argument("--adaptive-patience-checkpoints", type=int, default=4)
    parser.add_argument("--adaptive-minimum-improvement", type=float, default=1.0e-5)
    parser.add_argument("--adaptive-lr-reduction-factor", type=float, default=0.5)
    parser.add_argument("--adaptive-maximum-lr-reductions", type=int, default=3)
    parser.add_argument("--proposal-audit-count", type=int, default=131072)
    return parser.parse_args()


def _rates(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in str(raw).split(","))
    except ValueError as error:
        raise ValueError("learning-rates must be comma-separated numbers") from error
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("learning-rates must be finite and positive")
    return values


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _stage_summary(result: Any) -> list[Mapping[str, Any]]:
    return [
        {
            "name": stage.name,
            "stage": stage.stage,
            "active_groups": stage.active_groups,
            "trainable_variables": stage.trainable_variables,
            "incoming_loss": stage.incoming_loss,
            "selected_learning_rate": stage.selected_learning_rate,
            "selected_update": stage.selected_update,
            "selected_loss": stage.selected_loss,
            "optimizer_state_policy": stage.optimizer_state_policy,
            "incoming_optimizer_iterations": stage.incoming_optimizer_iterations,
            "selected_optimizer_iterations": stage.selected_optimizer_iterations,
            "candidates": [
                {
                    "learning_rate": candidate.learning_rate,
                    "selected_update": candidate.selected_update,
                    "selected_loss": candidate.selected_loss,
                    "terminal_loss": candidate.terminal_loss,
                    "clipped_updates": candidate.clipped_updates,
                    "gradient_norm": candidate.gradient_norm,
                    "executed_updates": candidate.executed_updates,
                    "learning_rate_reductions": candidate.learning_rate_reductions,
                    "stop_reason": candidate.stop_reason,
                    "checkpoint_history": candidate.checkpoint_history,
                }
                for candidate in stage.candidates
            ],
        }
        for stage in result.stages
    ]


def main() -> int:
    args = _args()
    rates = _rates(args.learning_rates)
    if int(args.batch_size) <= 1 or int(args.proposal_audit_count) < 32768:
        raise ValueError("batch size must exceed one and audit count must be at least 32768")
    selected_ceiling = (
        int(args.affine_updates)
        + int(args.simple_updates)
        + 3 * int(args.progressive_updates)
        + int(args.joint_updates)
    )
    if selected_ceiling != int(args.cold_updates):
        raise ValueError("staged and cold selected-path ceilings must match")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if not PLAN.is_file() or not BASE_RUNNER.is_file():
        raise FileNotFoundError("reviewed plan or base runner is missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical_gpus}")

    base = _base_runner()
    model = base._model(tf, str(args.target))
    selection_latent = tf.random.stateless_normal(
        (65536, int(model["dimension"])), seed=(20260815, 52001), dtype=tf.float64
    )
    if args.route in {"adaptive_reset", "adaptive_carry"}:
        from bayesfilter.inference.neutra_staged_training import (
            NeuTraAdaptiveStagePolicy,
            dense_iaf_five_stage_spec,
            dense_iaf_five_stage_variable_groups,
            train_neutra_five_stage,
        )

        transport = base._transport(tf, model, int(args.seed_index))
        adaptive_policy = NeuTraAdaptiveStagePolicy(
            minimum_updates=int(args.adaptive_minimum_updates),
            patience_checkpoints=int(args.adaptive_patience_checkpoints),
            minimum_improvement=float(args.adaptive_minimum_improvement),
            learning_rate_reduction_factor=float(args.adaptive_lr_reduction_factor),
            maximum_learning_rate_reductions=int(args.adaptive_maximum_lr_reductions),
        )
        stage_spec = dense_iaf_five_stage_spec(
            stages=int(model["stages"]),
            learning_rates=rates,
            affine_updates=int(args.affine_updates),
            simple_updates=int(args.simple_updates),
            progressive_updates=int(args.progressive_updates),
            joint_updates=int(args.joint_updates),
            checkpoint_every=int(args.checkpoint_every),
            joint_adaptive_policy=adaptive_policy,
        )
        phase_index = {
            phase.name: index for index, phase in enumerate(stage_spec.optimizer_phases())
        }

        def latent_batch(phase: str, update: int, _candidate: int) -> Any:
            return tf.random.stateless_normal(
                (int(args.batch_size), int(model["dimension"])),
                seed=(
                    20260815 + int(args.seed_index),
                    53000 + 10000 * phase_index[phase] + update,
                ),
                dtype=tf.float64,
            )

        optimizer_state_policy = (
            "carry_selected" if args.route == "adaptive_carry" else "phase_reset"
        )
        staged = train_neutra_five_stage(
            transport=transport,
            target_log_prob_fn=model["target_log_prob"],
            variable_groups=dense_iaf_five_stage_variable_groups(transport),
            spec=stage_spec,
            latent_batch_fn=latent_batch,
            selection_loss_fn=lambda active: base._selection_loss(
                tf, active, model["target_log_prob"], selection_latent
            ),
            validation_fn=lambda active: base._proposal_audit(
                tf,
                active,
                model,
                sample_count=int(args.proposal_audit_count),
                seed=(20260815, 54001 + int(args.seed_index)),
            ),
            gradient_clip_norm=10.0,
            jit_compile=True,
            optimizer_state_policy=optimizer_state_policy,
        )
        validation = staged.validation
        training = {
            "route": str(args.route),
            "optimizer_state_policy": staged.optimizer_state_policy,
            "stages": _stage_summary(staged),
            "selected_path_updates": staged.selected_path_updates,
            "tuning_optimizer_updates": staged.tuning_optimizer_updates,
            "nonclaims": staged.nonclaims,
        }
    else:
        transport, training = base._cold_train(
            tf,
            model,
            rates,
            updates=int(args.cold_updates),
            checkpoint_every=int(args.checkpoint_every),
            batch_size=int(args.batch_size),
            seed_index=int(args.seed_index),
            selection_latent=selection_latent,
        )
        training = {"route": "cold", **training}
        validation = base._proposal_audit(
            tf,
            transport,
            model,
            sample_count=int(args.proposal_audit_count),
            seed=(20260815, 54001 + int(args.seed_index)),
        )

    state = {
        "schema": "bayesfilter.neutra.generic_adaptive_five_stage_state.v1",
        "target": model["manifest"],
        "route": str(args.route),
        "config": transport.config.manifest_payload(),
        "variables": [variable.numpy().tolist() for variable in transport.trainable_variables],
    }
    state["state_hash"] = _stable_hash(state)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    adaptive_configuration = {
        "minimum_updates": int(args.adaptive_minimum_updates),
        "patience_checkpoints": int(args.adaptive_patience_checkpoints),
        "minimum_improvement": float(args.adaptive_minimum_improvement),
        "learning_rate_reduction_factor": float(args.adaptive_lr_reduction_factor),
        "maximum_learning_rate_reductions": int(args.adaptive_maximum_lr_reductions),
    }
    manifest = {
        "schema": "bayesfilter.neutra.generic_adaptive_five_stage_manifest.v1",
        "plan": PLAN.as_posix(),
        "target": model["manifest"],
        "target_name": str(args.target),
        "route": str(args.route),
        "config": transport.config.manifest_payload(),
        "learning_rates": rates,
        "batch_size": int(args.batch_size),
        "seed_index": int(args.seed_index),
        "updates": {
            "affine": int(args.affine_updates),
            "simple": int(args.simple_updates),
            "progressive_each": int(args.progressive_updates),
            "joint_ceiling": int(args.joint_updates),
            "cold_ceiling": int(args.cold_updates),
            "matched_selected_path_ceiling": selected_ceiling,
        },
        "adaptive_configuration": adaptive_configuration,
        "proposal_audit_count": int(args.proposal_audit_count),
        "jit_compile": True,
        "dtype": "float64",
        "tf32_enabled": False,
        "sample_wise_loop_or_scalar_fallback": False,
        "memory_policy": memory_policy,
        "allocator_bytes": allocator,
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    passed = bool(tf.convert_to_tensor(validation["passed"]).numpy())
    result = {
        "schema": "bayesfilter.neutra.generic_adaptive_five_stage_result.v1",
        "manifest": manifest,
        "training": training,
        "validation": validation,
        "decision": {
            "status": "known_law_gate_passed" if passed else "known_law_gate_failed",
            "known_law_gate_passed": passed,
            "promotion": False,
            "no_hmc": True,
            "nonclaims": [
                "no universal training-procedure claim",
                "no multimodal or SSL-LSTM transfer claim",
                "no HMC or posterior-correctness claim",
                "no runtime-efficiency claim from unequal tuning work",
            ],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "trainer_state.json", state)
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    _write(
        output / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.generic_adaptive_five_stage_hashes.v1",
            "artifacts": {
                path.relative_to(output).as_posix(): _sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "output_root": output.as_posix(),
                "target": str(args.target),
                "route": str(args.route),
                "seed_index": int(args.seed_index),
                "passed": passed,
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
