#!/usr/bin/env python3
"""Train and audit one matched reverse-KL paper-funnel capacity arm."""

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

DEFAULT_PLAN = ROOT / "docs/plans/bayesfilter-neutra-reverse-funnel-capacity-repair-plan-2026-08-14.md"
INTERVAL_LEVEL = 0.999
CRITICAL_VALUE = 3.2905267314919255
EXACT_TAIL_PROBABILITY = 0.02275013194817921


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--architecture-id", default="legacy_capacity_arm")
    parser.add_argument("--run-mode", choices=("calibration", "confirmation"), default="confirmation")
    parser.add_argument("--device", default="1")
    parser.add_argument("--stage-s-max", required=True)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=100)
    parser.add_argument(
        "--permutation-policy",
        choices=("full_reverse", "root_preserving_reverse"),
        default="full_reverse",
    )
    parser.add_argument(
        "--first-stage-scale-linear-skip",
        action="store_true",
        help="enable the zero-initialized masked linear scale skip in stage zero",
    )
    parser.add_argument(
        "--first-stage-unbounded-scale-linear",
        action="store_true",
        help="enable the zero-initialized additive linear scale path in stage zero",
    )
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "piecewise_60_85"),
        default="constant",
    )
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--selection-count", type=int, default=65536)
    parser.add_argument(
        "--trainable-mode", choices=("joint", "root_scale_only"), default="joint"
    )
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--include-initial-checkpoint", action="store_true")
    parser.add_argument("--proposal-audit-count", type=int, default=131072)
    return parser.parse_args()


def _parse_stage_caps(value: str, stages: int | None = None) -> tuple[float, ...]:
    try:
        caps = tuple(float(item.strip()) for item in str(value).split(","))
    except ValueError as error:
        raise ValueError("stage-s-max must be comma-separated numbers") from error
    if (
        (stages is not None and len(caps) != int(stages))
        or not caps
        or any(not math.isfinite(cap) or cap <= 0.0 for cap in caps)
    ):
        expected = "the configured stage count" if stages is not None else "at least one value"
        raise ValueError(f"stage-s-max must contain {expected} of finite positive values")
    return caps


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _assign_learning_rate(trainer: Any, value: float) -> None:
    learning_rate = trainer.optimizer.learning_rate
    if not hasattr(learning_rate, "assign"):
        raise RuntimeError("optimizer learning rate is not assignable")
    learning_rate.assign(float(value))


def _scheduled_learning_rate(
    peak: float, schedule: str, update: int, total_updates: int
) -> float:
    if schedule == "constant":
        return float(peak)
    fraction = float(update) / float(total_updates)
    if fraction < 0.60:
        multiplier = 1.0
    elif fraction < 0.85:
        multiplier = 0.1
    else:
        multiplier = 0.01
    return float(peak) * multiplier


def _compatible_state_config(current: Mapping[str, Any], stored: Mapping[str, Any]) -> bool:
    keys = (
        "dimension",
        "hidden_layers",
        "stages",
        "activation",
        "stage_s_max",
        "stage_scale_linear_skip",
        "stage_unbounded_scale_linear",
        "permutation_policy",
    )
    return all(current.get(key) == stored.get(key) for key in keys)


def _load_initial_state(tf: Any, trainer: Any, path: Path) -> Mapping[str, Any]:
    payload = _load_json(path)
    stored_hash = str(payload.get("state_hash", ""))
    hash_payload = dict(payload)
    hash_payload.pop("state_hash", None)
    if not stored_hash or _stable_hash(hash_payload) != stored_hash:
        raise RuntimeError("initial-state hash mismatch")
    stored_config = payload.get("config")
    if not isinstance(stored_config, Mapping) or not _compatible_state_config(
        trainer.config.manifest_payload(), stored_config
    ):
        raise RuntimeError("initial-state architecture mismatch")
    variables = payload.get("variables")
    if not isinstance(variables, list) or len(variables) != len(trainer.variables):
        raise RuntimeError("initial-state variable count mismatch")
    for variable, raw in zip(trainer.variables, variables, strict=True):
        value = tf.convert_to_tensor(raw, tf.float64)
        if value.shape != variable.shape:
            raise RuntimeError("initial-state variable shape mismatch")
        variable.assign(value)
    return {"path": path.as_posix(), "state_hash": stored_hash, "sha256": _sha256(path)}


def _reverse_rows(tf: Any, trainer: Any, target_value: Any, latent: Any) -> Any:
    physical, logdet = trainer.forward_and_logdet(latent)
    row_loss = -target_value(physical) - logdet
    tf.debugging.assert_all_finite(row_loss, "reverse-funnel objective rows")
    return row_loss


def _mean_interval(tf: Any, row_values: Any, exact: float) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(row_values, tf.float64)
    count = tf.cast(tf.size(values), tf.float64)
    estimate = tf.reduce_mean(values)
    centered = values - estimate
    variance = tf.reduce_sum(tf.square(centered)) / (count - tf.constant(1.0, tf.float64))
    standard_error = tf.sqrt(variance / count)
    radius = tf.constant(CRITICAL_VALUE, tf.float64) * standard_error
    lower = estimate - radius
    upper = estimate + radius
    return {
        "estimate": estimate,
        "exact": float(exact),
        "standard_error": standard_error,
        "lower": lower,
        "upper": upper,
        "passed": tf.logical_and(
            tf.constant(float(exact), tf.float64) >= lower,
            tf.constant(float(exact), tf.float64) <= upper,
        ),
    }


def _wilson_interval(tf: Any, successes: Any, count: int, exact: float) -> Mapping[str, Any]:
    n = tf.constant(float(count), tf.float64)
    estimate = tf.cast(successes, tf.float64) / n
    z = tf.constant(CRITICAL_VALUE, tf.float64)
    z2 = tf.square(z)
    denominator = tf.constant(1.0, tf.float64) + z2 / n
    center = (estimate + z2 / (tf.constant(2.0, tf.float64) * n)) / denominator
    radius = (
        z
        * tf.sqrt(
            estimate * (tf.constant(1.0, tf.float64) - estimate) / n
            + z2 / (tf.constant(4.0, tf.float64) * tf.square(n))
        )
        / denominator
    )
    lower = center - radius
    upper = center + radius
    exact_tensor = tf.constant(float(exact), tf.float64)
    return {
        "successes": tf.cast(successes, tf.int64),
        "count": int(count),
        "estimate": estimate,
        "exact": float(exact),
        "lower": lower,
        "upper": upper,
        "passed": tf.logical_and(exact_tensor >= lower, exact_tensor <= upper),
    }


def _scale_diagnostics(tf: Any, transport: Any, latent: Any) -> Mapping[str, Any]:
    current = latent
    stages = []
    for index, stage in enumerate(transport.stages):
        scale, _shift = stage._network(current)
        unbounded_linear = stage._unbounded_scale_linear(current)
        bounded_residual = scale - unbounded_linear
        cap = tf.constant(float(stage.s_max), tf.float64)
        stages.append(
            {
                "stage": index,
                "cap": float(stage.s_max),
                "minimum": tf.reduce_min(scale),
                "maximum": tf.reduce_max(scale),
                "mean": tf.reduce_mean(scale),
                "bounded_residual_minimum": tf.reduce_min(bounded_residual),
                "bounded_residual_maximum": tf.reduce_max(bounded_residual),
                "unbounded_linear_minimum": tf.reduce_min(unbounded_linear),
                "unbounded_linear_maximum": tf.reduce_max(unbounded_linear),
                "unbounded_linear_mean": tf.reduce_mean(unbounded_linear),
                "fraction_abs_ge_0.99_cap": tf.reduce_mean(
                    tf.cast(
                        tf.abs(bounded_residual)
                        >= tf.constant(0.99, tf.float64) * cap,
                        tf.float64,
                    )
                ),
                "fraction_abs_ge_0.999_cap": tf.reduce_mean(
                    tf.cast(
                        tf.abs(bounded_residual)
                        >= tf.constant(0.999, tf.float64) * cap,
                        tf.float64,
                    )
                ),
            }
        )
        current, _increment = stage.forward_and_logdet(current)
        if index + 1 < len(transport.stages):
            current = transport._between_stage_permutation(current)
    return {"stages": stages}


def _proposal_audit(
    tf: Any,
    trainer: Any,
    target_value: Any,
    count: int,
    seed: tuple[int, int] = (20260814, 71001),
    authority: str = "exact_iid_funnel_proposal_law",
) -> Mapping[str, Any]:
    latent = tf.random.stateless_normal(
        (int(count), 100), seed=seed, dtype=tf.float64
    )
    physical, logdet = trainer.forward_and_logdet(latent)
    y = physical[:, 0]
    residual = physical[:, 1:] * tf.exp(-y[:, tf.newaxis])
    residual_mean_by_row = tf.reduce_mean(residual, axis=1)
    residual_square_by_row = tf.reduce_mean(tf.square(residual), axis=1)
    mean_screens = {
        "y_mean": _mean_interval(tf, y, 0.0),
        "y_second_moment": _mean_interval(tf, tf.square(y), 1.0),
        "residual_mean": _mean_interval(tf, residual_mean_by_row, 0.0),
        "residual_second_moment": _mean_interval(tf, residual_square_by_row, 1.0),
    }
    tail_screens = {
        "probability_below_minus2": _wilson_interval(
            tf, tf.reduce_sum(tf.cast(y < -2.0, tf.int64)), int(count), EXACT_TAIL_PROBABILITY
        ),
        "probability_above_plus2": _wilson_interval(
            tf, tf.reduce_sum(tf.cast(y > 2.0, tf.int64)), int(count), EXACT_TAIL_PROBABILITY
        ),
    }
    log_base = -tf.constant(0.5, tf.float64) * (
        tf.reduce_sum(tf.square(latent), axis=1)
        + tf.constant(100.0 * math.log(2.0 * math.pi), tf.float64)
    )
    log_ratio = target_value(physical) + logdet - log_base
    normalized = tf.nn.softmax(log_ratio)
    all_passed = tf.reduce_all(
        tf.stack(
            [screen["passed"] for screen in (*mean_screens.values(), *tail_screens.values())]
        )
    )
    return {
        "authority": authority,
        "sample_count": int(count),
        "interval_level": INTERVAL_LEVEL,
        "mean_screens": mean_screens,
        "tail_screens": tail_screens,
        "all_individual_intervals_passed": all_passed,
        "importance_ess_fraction": tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
        / tf.cast(count, tf.float64),
        "maximum_normalized_importance_weight": tf.reduce_max(normalized),
        "log_target_to_proposal_ratio_stddev": tf.math.reduce_std(log_ratio),
        "scale_diagnostics": _scale_diagnostics(tf, trainer.transport, latent),
        "nonclaims": (
            "proposal draws are not posterior draws",
            "separate 99.9% diagnostics; no omnibus p-value",
            "proposal-law pass does not establish HMC validity",
        ),
    }


def main() -> int:
    args = _args()
    caps = _parse_stage_caps(args.stage_s_max, args.stages)
    if bool(args.first_stage_scale_linear_skip) and bool(
        args.first_stage_unbounded_scale_linear
    ):
        raise ValueError("pre-cap and unbounded scale-linear paths are mutually exclusive")
    if int(args.updates) <= 0 or int(args.updates) > 5000:
        raise ValueError("updates must lie in [1,5000]")
    if int(args.batch_size) <= 1:
        raise ValueError("batch size must exceed one")
    if int(args.stages) <= 0 or int(args.hidden_width) <= 0:
        raise ValueError("stages and hidden-width must be positive")
    if int(args.seed_index) < 0:
        raise ValueError("seed-index must be nonnegative")
    if int(args.selection_count) < 8192:
        raise ValueError("selection-count must be at least 8192")
    if int(args.checkpoint_every) <= 0 or int(args.checkpoint_every) > int(args.updates):
        raise ValueError("checkpoint interval is invalid")
    if int(args.proposal_audit_count) < 32768:
        raise ValueError("proposal audit count must be at least 32768")
    if not math.isfinite(float(args.learning_rate)) or float(args.learning_rate) <= 0.0:
        raise ValueError("learning rate must be finite and positive")

    output = args.output_root.resolve()
    plan = args.plan.resolve()
    replay_root = args.replay_root.resolve()
    replay_manifest_path = replay_root / "replay_manifest.json"
    replay_hashes_path = replay_root / "artifact_hashes.json"
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if any(not path.is_file() for path in (plan, replay_manifest_path, replay_hashes_path)):
        raise FileNotFoundError("capacity plan or replay evidence is missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()

    import tensorflow as tf

    from bayesfilter.inference.neutra_paper_d100_target import (
        PaperD100ValueScoreAdapter,
        make_paper_funnel_spec,
        paper_d100_log_prob_batch,
    )
    from bayesfilter.inference.neutra_weighted_training import (
        MatchedReverseKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical_gpus}")

    replay_manifest = _load_json(replay_manifest_path)
    replay_hashes = _load_json(replay_hashes_path)
    if replay_manifest.get("target", {}).get("name") != "paper_funnel":
        raise RuntimeError("capacity runner requires paper-funnel replay")
    if replay_hashes.get("artifacts", {}).get(replay_manifest_path.name) != _sha256(
        replay_manifest_path
    ):
        raise RuntimeError("replay manifest artifact hash mismatch")
    receipts = replay_manifest.get("receipts")
    if not isinstance(receipts, Mapping):
        raise RuntimeError("replay receipts are missing")
    audit_rows_receipt = receipts["audit_rows"]

    spec = make_paper_funnel_spec()
    base_adapter = PaperD100ValueScoreAdapter(spec)
    target_value = lambda rows: paper_d100_log_prob_batch(spec, rows)
    config = WeightedNeuTraConfig(
        dimension=100,
        hidden_layers=(int(args.hidden_width), int(args.hidden_width)),
        stages=int(args.stages),
        activation="elu",
        s_max=1.0,
        stage_s_max=caps,
        stage_scale_linear_skip=(
            (True,) + (False,) * (int(args.stages) - 1)
            if bool(args.first_stage_scale_linear_skip)
            else (False,) * int(args.stages)
        ),
        stage_unbounded_scale_linear=(
            (True,) + (False,) * (int(args.stages) - 1)
            if bool(args.first_stage_unbounded_scale_linear)
            else (False,) * int(args.stages)
        ),
        permutation_policy=str(args.permutation_policy),
        initialization_scale=0.02,
        initialization_seed=(20260815, 20001 + 100 * int(args.seed_index)),
        learning_rate=float(args.learning_rate),
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    trainer = MatchedReverseKLNeuTraTrainer(config, target_value)
    initial_state_receipt = None
    if args.initial_state is not None:
        initial_state_receipt = _load_initial_state(
            tf, trainer, args.initial_state.resolve()
        )
    if args.trainable_mode == "root_scale_only":
        root_scale = trainer.transport.stages[0].unbounded_scale_linear_weight
        if root_scale is None:
            raise RuntimeError("root_scale_only requires first-stage unbounded scale")
        trainer.variables = (root_scale,)
        trainer.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(args.learning_rate),
            beta_1=float(config.beta1),
            beta_2=float(config.beta2),
            epsilon=float(config.epsilon),
        )
        trainer.optimizer.build(trainer.variables)
        trainer._compiled_train_step = tf.function(
            trainer._train_step_impl,
            jit_compile=True,
            reduce_retracing=True,
        )
    selection_latent = tf.random.stateless_normal(
        (int(args.selection_count), 100), seed=(20260815, 20002), dtype=tf.float64
    )
    checkpoints = []
    selected_variables: list[Any] | None = None
    selected_update = 0
    selected_loss = float("inf")
    clipped_updates = 0
    last_step: Mapping[str, Any] = {}
    if bool(args.include_initial_checkpoint):
        selected_loss = float(
            tf.reduce_mean(
                _reverse_rows(tf, trainer, target_value, selection_latent)
            ).numpy()
        )
        selected_variables = [
            variable.numpy().tolist() for variable in trainer.transport.trainable_variables
        ]
        checkpoints.append(
            {
                "update": 0,
                "selection_loss": selected_loss,
                "training_loss": None,
                "learning_rate": 0.0,
                "gradient_norm": None,
                "clipping_applied": False,
            }
        )
    for update in range(1, int(args.updates) + 1):
        current_learning_rate = _scheduled_learning_rate(
            float(args.learning_rate),
            str(args.learning_rate_schedule),
            update,
            int(args.updates),
        )
        _assign_learning_rate(trainer, current_learning_rate)
        latent = tf.random.stateless_normal(
            (int(args.batch_size), 100),
            seed=tf.random.experimental.stateless_fold_in(
                tf.constant(
                    (20260815, 20003 + 100 * int(args.seed_index)), tf.int32
                ),
                update,
            ),
            dtype=tf.float64,
        )
        step = trainer.train_step(latent)
        clipped_updates += int(bool(step.clipping_applied.numpy()))
        last_step = _ready(step.__dict__)
        if update % int(args.checkpoint_every) == 0 or update == int(args.updates):
            selection_loss = float(
                tf.reduce_mean(_reverse_rows(tf, trainer, target_value, selection_latent)).numpy()
            )
            checkpoint = {
                "update": update,
                "selection_loss": selection_loss,
                "training_loss": float(step.loss.numpy()),
                "learning_rate": current_learning_rate,
                "gradient_norm": float(step.gradient_norm.numpy()),
                "clipping_applied": bool(step.clipping_applied.numpy()),
            }
            checkpoints.append(checkpoint)
            if selection_loss < selected_loss:
                selected_loss = selection_loss
                selected_update = update
                selected_variables = [
                    variable.numpy().tolist()
                    for variable in trainer.transport.trainable_variables
                ]
    if selected_variables is None:
        raise RuntimeError("capacity arm produced no finite checkpoint")
    for variable, raw in zip(
        trainer.transport.trainable_variables, selected_variables, strict=True
    ):
        variable.assign(tf.constant(raw, tf.float64))

    audit_latent = tf.random.stateless_normal(
        (int(args.selection_count), 100), seed=(20260815, 20004), dtype=tf.float64
    )
    audit_reverse_kl = float(
        tf.reduce_mean(_reverse_rows(tf, trainer, target_value, audit_latent)).numpy()
    )
    audit_exact_forward_nll = None
    if args.run_mode == "confirmation":
        exact_audit_rows = _load_tensor(tf, replay_root, audit_rows_receipt)
        audit_exact_forward_nll = float(
            tf.reduce_mean(-trainer.log_prob(exact_audit_rows)).numpy()
        )
    proposal_seed = (
        (20260815, 22001 + int(args.seed_index))
        if args.run_mode == "confirmation"
        else (20260815, 21001)
    )
    proposal = _proposal_audit(
        tf,
        trainer,
        target_value,
        int(args.proposal_audit_count),
        seed=proposal_seed,
        authority=(
            "untouched_exact_iid_funnel_proposal_law"
            if args.run_mode == "confirmation"
            else "calibration_explanatory_exact_iid_funnel_proposal_law"
        ),
    )
    proposal_passed = bool(proposal["all_individual_intervals_passed"].numpy())

    state = {
        "schema": "bayesfilter.neutra.paper_d100_training_state.v1",
        "objective": "reverse_kl",
        "target": spec.manifest_payload(),
        "replay_manifest_sha256": _sha256(replay_manifest_path),
        "selected_update": selected_update,
        "config": config.manifest_payload(),
        "variables": selected_variables,
    }
    state["state_hash"] = _stable_hash(state)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    manifest = {
        "schema": "bayesfilter.neutra.reverse_funnel_capacity_manifest.v1",
        "plan": plan.as_posix(),
        "architecture_id": str(args.architecture_id),
        "run_mode": str(args.run_mode),
        "target": spec.manifest_payload(),
        "adapter_signature": base_adapter.adapter_signature(),
        "objective": "reverse_kl",
        "config": config.manifest_payload(),
        "resolved_stage_s_max": list(caps),
        "first_stage_scale_linear_skip": bool(args.first_stage_scale_linear_skip),
        "resolved_stage_scale_linear_skip": [
            stage.scale_linear_skip_enabled for stage in trainer.transport.stages
        ],
        "first_stage_unbounded_scale_linear": bool(
            args.first_stage_unbounded_scale_linear
        ),
        "resolved_stage_unbounded_scale_linear": [
            stage.unbounded_scale_linear_enabled for stage in trainer.transport.stages
        ],
        "updates": int(args.updates),
        "training_batch_size": int(args.batch_size),
        "checkpoint_every": int(args.checkpoint_every),
        "learning_rate": float(args.learning_rate),
        "learning_rate_schedule": str(args.learning_rate_schedule),
        "trainable_mode": str(args.trainable_mode),
        "initial_state": initial_state_receipt,
        "include_initial_checkpoint": bool(args.include_initial_checkpoint),
        "selection_size": int(args.selection_count),
        "audit_size": int(args.selection_count),
        "proposal_audit_count": int(args.proposal_audit_count),
        "seeds": {
            "initialization": [20260815, 20001 + 100 * int(args.seed_index)],
            "selection": [20260815, 20002],
            "training_root": [20260815, 20003 + 100 * int(args.seed_index)],
            "audit": [20260815, 20004],
            "proposal_audit": list(proposal_seed),
        },
        "memory_policy": _ready(memory_policy),
        "allocator_bytes": _ready(allocator),
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "sample_wise_loop_or_scalar_fallback": False,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    result = {
        "schema": "bayesfilter.neutra.reverse_funnel_capacity_result.v1",
        "manifest": manifest,
        "checkpoints": checkpoints,
        "selected_update": selected_update,
        "selected_loss": selected_loss,
        "audit_reverse_kl": audit_reverse_kl,
        "audit_exact_forward_nll": audit_exact_forward_nll,
        "clipped_updates": clipped_updates,
        "last_step": last_step,
        "proposal_audit": proposal,
        "decision": {
            "status": (
                "proposal_gate_passed"
                if proposal_passed and args.run_mode == "confirmation"
                else "proposal_gate_failed"
                if args.run_mode == "confirmation"
                else "calibration_complete"
            ),
            "proposal_gate_passed": proposal_passed,
            "eligible_for_hmc_nomination": (
                proposal_passed
                and args.run_mode == "confirmation"
                and int(args.updates) == 5000
            ),
            "promotion": False,
            "diagnostic_roles": {
                "proposal_exact_law": "promotion_criterion",
                "nonfinite_training": "hard_veto",
                "training_loss": "explanatory_only",
                "scale_saturation": "explanatory_only",
            },
            "nonclaims": [
                "no HMC validity claim",
                "no objective ranking",
                "no SSL-LSTM transfer claim",
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
            "schema": "bayesfilter.neutra.reverse_funnel_capacity_hashes.v1",
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
                "stage_s_max": caps,
                "first_stage_scale_linear_skip": bool(
                    args.first_stage_scale_linear_skip
                ),
                "first_stage_unbounded_scale_linear": bool(
                    args.first_stage_unbounded_scale_linear
                ),
                "selected_update": selected_update,
                "proposal_gate_passed": proposal_passed,
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
