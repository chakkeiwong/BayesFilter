#!/usr/bin/env python3
"""Run the frozen SSL-LSTM q=20 weighted-NeuTra training screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-2026-08-19.md"
)
CANARY_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py"
)
PREFLIGHT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/"
    "gpu-preflight.json"
)
CANARY_RESULT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/"
    "gpu-canary-retry-01/result.json"
)
CANARY_MANIFEST = CANARY_RESULT.with_name("manifest.json")
GEOMETRY = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/"
    "training-screen"
)

PREFLIGHT_SHA256 = "28be07fcc83be539b9b643f2127094f025706d0a92b733873a85ddeb56b50a45"
CANARY_RESULT_SHA256 = "27685b3f22936659b5b5b34bc07c675eb16c47ddb86a465bfd83fb31c31d7bfa"
CANARY_MANIFEST_SHA256 = "79b7bb1d29784c701cebb4f4886ecf740094785186cc98a92efa2e24a3597395"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
PRE_CANARY_PLAN_SHA256 = "9c2e03bfd5cb121421c8e50b5a2434131efbf65e719c77f5967836be06e0df84"

PRIMARY_CAPACITIES = ((64, 3), (128, 6))
PRIMARY_LEARNING_RATES = (1.0e-3, 3.0e-4)
TRAINING_SEEDS = (2, 3)
UPDATE_LADDER = (250, 2000, 8000)
SELECTION_INTERVAL = 250
REPAIR_CAPACITY = (128, 6)
REPAIR_LEARNING_RATE = 1.0e-4
MAX_PHASE_WALL_SECONDS = 7200.0
INTERNAL_DEFAULT_WALL_SECONDS = 7050.0


class TrainingBudgetExhausted(RuntimeError):
    """Raised before the external timeout so partial artifacts can be closed."""


class ArmNumericalFailure(RuntimeError):
    """Classify a finite-contract failure as an arm result, not a harness result."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite training artifact: {path}")
    temporary.write_text(
        json.dumps(
            _json_ready(payload), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _reserve_output_root(path: Path) -> Path:
    absolute = path if path.is_absolute() else ROOT / path
    if absolute.exists():
        raise RuntimeError(f"refusing to reuse training output root: {absolute}")
    absolute.mkdir(parents=True, exist_ok=False)
    return absolute


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty)}


def _parse_capacities(raw: str) -> tuple[tuple[int, int], ...]:
    try:
        parsed = tuple(
            tuple(int(value) for value in cell.lower().split("x"))
            for cell in raw.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("capacities must use WIDTHxSTAGES") from error
    if any(len(cell) != 2 for cell in parsed):
        raise argparse.ArgumentTypeError("capacities must use WIDTHxSTAGES")
    return parsed


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="1")
    parser.add_argument("--capacities", type=_parse_capacities, default=PRIMARY_CAPACITIES)
    parser.add_argument(
        "--learning-rates", type=_parse_float_tuple, default=PRIMARY_LEARNING_RATES
    )
    parser.add_argument("--seeds", type=_parse_int_tuple, default=TRAINING_SEEDS)
    parser.add_argument("--update-ladder", type=_parse_int_tuple, default=UPDATE_LADDER)
    parser.add_argument(
        "--time-cap-seconds", type=float, default=INTERNAL_DEFAULT_WALL_SECONDS
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if tuple(args.capacities) != PRIMARY_CAPACITIES:
        raise SystemExit(f"capacities are frozen to {PRIMARY_CAPACITIES}")
    if tuple(args.learning_rates) != PRIMARY_LEARNING_RATES:
        raise SystemExit(f"learning rates are frozen to {PRIMARY_LEARNING_RATES}")
    if tuple(args.seeds) != TRAINING_SEEDS:
        raise SystemExit(f"seeds are frozen to {TRAINING_SEEDS}")
    if tuple(args.update_ladder) != UPDATE_LADDER:
        raise SystemExit(f"update ladder is frozen to {UPDATE_LADDER}")
    if not str(args.device).isdigit():
        raise SystemExit("device must be one nonnegative physical GPU index")
    if not math.isfinite(args.time_cap_seconds) or not (
        0.0 < args.time_cap_seconds <= MAX_PHASE_WALL_SECONDS
    ):
        raise SystemExit("time cap must be finite, positive, and at most 7200 seconds")


def _load_canary_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ssl_lstm_q20_weighted_replay_canary", CANARY_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the receipt-bound replay helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_prior_artifacts() -> Mapping[str, Any]:
    expected = (
        (PREFLIGHT, PREFLIGHT_SHA256),
        (CANARY_RESULT, CANARY_RESULT_SHA256),
        (CANARY_MANIFEST, CANARY_MANIFEST_SHA256),
        (GEOMETRY, GEOMETRY_SHA256),
    )
    for path, digest in expected:
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"prior artifact SHA-256 mismatch: {path}")
    preflight = _read_json(PREFLIGHT)
    canary = _read_json(CANARY_RESULT)
    if preflight.get("status") != "GPU_PREFLIGHT_PASSED":
        raise RuntimeError("GPU preflight did not pass")
    if preflight.get("plan", {}).get("sha256") != PRE_CANARY_PLAN_SHA256:
        raise RuntimeError("GPU preflight plan identity mismatch")
    if canary.get("status") != "GPU_REPLAY_TRAINING_AND_PULLBACK_HMC_COMPLETED":
        raise RuntimeError("mechanics canary did not complete")
    if canary.get("plan_sha256") != PRE_CANARY_PLAN_SHA256:
        raise RuntimeError("mechanics canary plan identity mismatch")
    if canary.get("exact_pullback_parity", {}).get("passed") is not True:
        raise RuntimeError("mechanics canary exact pullback did not pass")
    mixing = canary.get("global_mixing_report", {})
    if mixing.get("passed") is not False:
        raise RuntimeError("training repair requires the preserved mode-locked canary")
    if mixing.get("chain_transition_counts") != [0, 0, 0, 0]:
        raise RuntimeError("mechanics canary trigger has drifted")
    if int(canary.get("audit_rows_loaded_or_evaluated", -1)) != 0:
        raise RuntimeError("mechanics canary touched the reserved audit bank")
    return {
        "preflight": {"path": PREFLIGHT.as_posix(), "sha256": PREFLIGHT_SHA256},
        "canary_result": {
            "path": CANARY_RESULT.as_posix(),
            "sha256": CANARY_RESULT_SHA256,
            "mode_locked": True,
        },
        "canary_manifest": {
            "path": CANARY_MANIFEST.as_posix(),
            "sha256": CANARY_MANIFEST_SHA256,
        },
        "geometry": {"path": GEOMETRY.as_posix(), "sha256": GEOMETRY_SHA256},
    }


def _state_without_hash(state: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(state)
    supplied = payload.pop("state_hash", None)
    if not isinstance(supplied, str) or supplied != _stable_hash(payload):
        raise RuntimeError("weighted NeuTra state semantic hash mismatch")
    return payload


def _restore_trainer_state(tf: Any, trainer: Any, state: Mapping[str, Any]) -> None:
    payload = _state_without_hash(state)
    if payload.get("config") != trainer.config.manifest_payload():
        raise RuntimeError("weighted NeuTra state config mismatch")
    raw_variables = payload.get("variables")
    raw_optimizer = payload.get("optimizer_variables")
    if not isinstance(raw_variables, list) or len(raw_variables) != len(trainer.variables):
        raise RuntimeError("weighted NeuTra variable count mismatch")
    if not isinstance(raw_optimizer, list) or len(raw_optimizer) != len(
        trainer.optimizer.variables
    ):
        raise RuntimeError("weighted NeuTra optimizer variable count mismatch")

    def assign(variable: Any, raw: Any, label: str) -> None:
        variable_dtype = tf.dtypes.as_dtype(variable.dtype)
        tensor = tf.convert_to_tensor(raw, dtype=variable_dtype)
        if tensor.shape != variable.shape:
            raise RuntimeError(f"weighted NeuTra {label} shape mismatch")
        if variable_dtype.is_floating:
            tf.debugging.assert_all_finite(tensor, f"weighted NeuTra {label}")
        variable.assign(tensor)

    for variable, raw in zip(trainer.variables, raw_variables, strict=True):
        assign(variable, raw, "transport variable")
    for variable, raw in zip(trainer.optimizer.variables, raw_optimizer, strict=True):
        assign(variable, raw, "optimizer variable")
    trainer.step.assign(tf.cast(int(payload["step"]), trainer.step.dtype))
    restored = trainer.state_payload()
    if restored.get("state_hash") != state.get("state_hash"):
        raise RuntimeError("weighted NeuTra state failed exact restore roundtrip")


def _config_from_state(state: Mapping[str, Any], config_type: Any) -> Any:
    raw = dict(state.get("config", {}))
    if raw.pop("schema", None) != "bayesfilter.neutra.weighted_forward_kl_config.v1":
        raise RuntimeError("weighted NeuTra config schema mismatch")
    for name in (
        "hidden_layers",
        "initialization_seed",
        "stage_s_max",
        "stage_scale_linear_skip",
        "stage_unbounded_scale_linear",
    ):
        raw[name] = tuple(raw[name])
    return config_type(**raw)


def _arm_id(width: int, stages: int, rate: float, seed: int, role: str) -> str:
    return (
        f"{role}-seed-{seed}-width-{width}-stages-{stages}-"
        f"lr-{rate:.0e}"
    )


def _remaining_or_raise(started: float, cap: float, context: str) -> float:
    remaining = cap - (time.perf_counter() - started)
    if remaining <= 0.0:
        raise TrainingBudgetExhausted(f"training wall budget exhausted during {context}")
    return remaining


def _run_arm(
    tf: Any,
    trainer_type: Any,
    config_type: Any,
    train_rows: Any,
    train_weights: Any,
    selection_rows: Any,
    selection_weights: Any,
    *,
    width: int,
    stages: int,
    rate: float,
    seed: int,
    role: str,
    campaign_started: float,
    time_cap_seconds: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    arm_name = _arm_id(width, stages, rate, seed, role)
    config = config_type(
        dimension=4,
        hidden_layers=(int(width), int(width)),
        stages=int(stages),
        activation="tanh",
        s_max=2.0,
        permutation_policy="full_reverse",
        initialization_scale=0.02,
        initialization_seed=(20260819, 1000 + int(seed)),
        learning_rate=float(rate),
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    arm_started = time.perf_counter()
    trainer = trainer_type(config)
    best_state: Mapping[str, Any] | None = None
    best_selection_loss = math.inf
    best_update: int | None = None
    checkpoints = []
    last_step: Any = None
    for update in range(1, UPDATE_LADDER[-1] + 1):
        if update == 1 or update % 25 == 0:
            _remaining_or_raise(campaign_started, time_cap_seconds, arm_name)
        last_step = trainer.train_step(train_rows, train_weights)
        if update % SELECTION_INTERVAL != 0:
            continue
        selection = trainer.validation_batch(selection_rows, selection_weights)
        selection_loss = float(selection.loss.numpy())
        gradient_norm = float(last_step.gradient_norm.numpy())
        clipped_norm = float(last_step.clipped_gradient_norm.numpy())
        if not all(math.isfinite(value) for value in (selection_loss, gradient_norm, clipped_norm)):
            raise ArmNumericalFailure(f"nonfinite selection checkpoint in {arm_name}")
        checkpoint = {
            "update": update,
            "ladder_milestone": update in UPDATE_LADDER,
            "training_loss": last_step.loss,
            "gradient_norm": last_step.gradient_norm,
            "clipped_gradient_norm": last_step.clipped_gradient_norm,
            "clipping_applied": last_step.clipping_applied,
            "selection_loss": selection.loss,
            "selection_effective_sample_size": selection.effective_sample_size,
            "selection_effective_sample_size_fraction": (
                selection.effective_sample_size_fraction
            ),
            "selection_maximum_normalized_weight": selection.maximum_normalized_weight,
        }
        checkpoints.append(_json_ready(checkpoint))
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_update = update
            best_state = trainer.state_payload()
    if best_state is None or best_update is None:
        raise RuntimeError(f"arm did not produce a selectable checkpoint: {arm_name}")
    _restore_trainer_state(tf, trainer, best_state)
    frozen_state = trainer.state_payload()
    summary = {
        "arm_id": arm_name,
        "role": role,
        "status": "TRAINING_COMPLETED",
        "seed": int(seed),
        "width": int(width),
        "hidden_layers": [int(width), int(width)],
        "stages": int(stages),
        "learning_rate": float(rate),
        "updates_completed": UPDATE_LADDER[-1],
        "best_update": best_update,
        "best_selection_loss": best_selection_loss,
        "training_state_hash": frozen_state["state_hash"],
        "transport_tensor_hash": _stable_hash(
            {"variables": frozen_state["variables"]}
        ),
        "checkpoints": checkpoints,
        "wall_seconds": time.perf_counter() - arm_started,
    }
    del trainer
    gc.collect()
    return summary, frozen_state


def _load_audit(tf: Any, canary_module: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    verified, metadata = canary_module._verified_replay_sources(
        (7,), allow_reserved_audit=True
    )
    if len(verified) != 1:
        raise RuntimeError("audit loader must return exactly central-07")
    theta_path, weight_path, source = verified[0]
    rows = tf.ensure_shape(
        tf.io.parse_tensor(tf.io.read_file(str(theta_path)), tf.float64), (100, 4)
    )
    normalized = tf.ensure_shape(
        tf.io.parse_tensor(tf.io.read_file(str(weight_path)), tf.float64), (100,)
    )
    tf.debugging.assert_all_finite(rows, "central-07 audit rows")
    tf.debugging.assert_all_finite(normalized, "central-07 audit weights")
    tf.debugging.assert_positive(normalized, "central-07 audit weights")
    tf.debugging.assert_near(
        tf.reduce_sum(normalized), tf.constant(1.0, tf.float64), atol=1.0e-10
    )
    return rows, tf.math.log(normalized), {**metadata, "source": source}


def _parity_report(
    tf: Any,
    trainer: Any,
    target: Any,
    transformed: Any,
    representatives: Any,
) -> Mapping[str, Any]:
    latent, inverse_logdet = trainer.transport.inverse_and_forward_logdet(
        representatives
    )
    recovered = trainer.transport.forward_batch(latent)
    direct_logdet = trainer.transport.log_abs_det_jacobian_batch(latent)
    transformed_value, transformed_score, transformed_status = (
        transformed.log_prob_and_grad_status(latent)
    )
    physical_value, physical_score, physical_status = (
        target.neutra_batch_log_prob_and_grad_status(representatives)
    )
    expected_score = trainer.transport.pullback_score_batch(
        latent, tf.stop_gradient(physical_score)
    ) + trainer.transport.log_abs_det_jacobian_score_batch(latent)
    inverse_residual = tf.reduce_max(tf.abs(recovered - representatives))
    logdet_residual = tf.reduce_max(tf.abs(inverse_logdet - direct_logdet))
    value_residual = tf.reduce_max(
        tf.abs(transformed_value - physical_value - direct_logdet)
    )
    score_residual = tf.reduce_max(tf.abs(transformed_score - expected_score))
    tolerance = tf.constant(1.0e-10, tf.float64)
    status_valid = tf.reduce_all(
        (tf.cast(physical_status["status_code"], tf.int32) == 0)
        & tf.cast(physical_status["valid_pre_regularized_score"], tf.bool)
        & (tf.cast(transformed_status["status_code"], tf.int32) == 0)
        & tf.cast(transformed_status["valid_pre_regularized_score"], tf.bool)
    )
    finite = tf.reduce_all(
        tf.stack(
            (
                tf.reduce_all(tf.math.is_finite(recovered)),
                tf.reduce_all(tf.math.is_finite(direct_logdet)),
                tf.reduce_all(tf.math.is_finite(transformed_value)),
                tf.reduce_all(tf.math.is_finite(transformed_score)),
            )
        )
    )
    passed = (
        finite
        & status_valid
        & (inverse_residual <= tolerance)
        & (logdet_residual <= tolerance)
        & (value_residual <= tolerance)
        & (score_residual <= tolerance)
    )
    return {
        "inverse_forward_maximum_absolute_residual": inverse_residual,
        "logdet_maximum_absolute_residual": logdet_residual,
        "value_identity_maximum_absolute_residual": value_residual,
        "score_composition_maximum_absolute_residual": score_residual,
        "tolerance": tolerance,
        "all_finite": finite,
        "target_status_valid": status_valid,
        "passed": passed,
    }


def _audit_candidate(
    tf: Any,
    trainer_type: Any,
    config_type: Any,
    adapter_type: Any,
    bound_adapter_type: Any,
    target: Any,
    representatives: Any,
    audit_rows: Any,
    audit_weights: Any,
    nomination: Mapping[str, Any],
) -> Mapping[str, Any]:
    state_path = Path(str(nomination["state_path"]))
    if _sha256(state_path) != nomination.get("state_sha256"):
        raise RuntimeError("nominated training state artifact hash mismatch")
    wrapper = _read_json(state_path)
    if wrapper.get("schema") != "bayesfilter.ssl_lstm.q20_neutra_training_state.v1":
        raise RuntimeError("nominated training state schema mismatch")
    state = wrapper.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("nominated training state is missing")
    config = _config_from_state(state, config_type)
    trainer = trainer_type(config)
    _restore_trainer_state(tf, trainer, state)
    audit = trainer.validation_batch(audit_rows, audit_weights)
    audit_finite = tf.reduce_all(
        tf.stack(
            (
                tf.reduce_all(tf.math.is_finite(audit.loss)),
                tf.reduce_all(tf.math.is_finite(audit.per_sample_negative_log_prob)),
                tf.reduce_all(tf.math.is_finite(audit.normalized_weights)),
                tf.reduce_all(tf.math.is_finite(audit.latent)),
                tf.reduce_all(tf.math.is_finite(audit.latent_weighted_mean)),
                tf.reduce_all(tf.math.is_finite(audit.latent_weighted_covariance)),
                tf.reduce_all(tf.math.is_finite(audit.effective_sample_size)),
                tf.reduce_all(
                    tf.math.is_finite(audit.effective_sample_size_fraction)
                ),
                tf.reduce_all(
                    tf.math.is_finite(audit.maximum_normalized_weight)
                ),
            )
        )
    )
    tensor_hash = _stable_hash({"variables": state["variables"]})
    if tensor_hash != nomination.get("transport_tensor_hash"):
        raise RuntimeError("nominated transport tensor hash mismatch")
    trainer.transport.bind_frozen_identity(
        {
            "checkpoint_sha256": str(nomination["state_sha256"]),
            "training_state_hash": str(state["state_hash"]),
            "transport_tensor_hash": tensor_hash,
        }
    )
    base = bound_adapter_type(target, target_signature=target.target_signature())
    transformed = adapter_type(
        base_adapter=base,
        transport=trainer.transport,
        target_scope=(
            "ssl_lstm_q20_neutra_global_mixing_training_audit:"
            + str(nomination["arm_id"])
        ),
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
        require_batch_native=True,
        nonclaims=(
            "frozen training audit only",
            "audit loss does not rank candidates",
            "no HMC, posterior, predictive, or default-readiness claim",
        ),
    )
    parity = _parity_report(tf, trainer, target, transformed, representatives)
    passed = audit_finite & tf.cast(parity["passed"], tf.bool)
    return {
        "arm_id": nomination["arm_id"],
        "seed": nomination["seed"],
        "state_path": state_path.as_posix(),
        "state_sha256": nomination["state_sha256"],
        "training_state_hash": state["state_hash"],
        "transport_tensor_hash": tensor_hash,
        "transport_manifest": trainer.transport.manifest_payload(),
        "transformed_adapter_signature": transformed.adapter_signature(),
        "audit_validation_call_count": 1,
        "audit_loss": audit.loss,
        "audit_effective_sample_size": audit.effective_sample_size,
        "audit_effective_sample_size_fraction": audit.effective_sample_size_fraction,
        "audit_maximum_normalized_weight": audit.maximum_normalized_weight,
        "audit_finite": audit_finite,
        "exact_pullback_parity": parity,
        "passed": passed,
    }


def _execute(args: argparse.Namespace, output: Path, started: float) -> int:
    prior = _validate_prior_artifacts()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    import tensorflow as tf

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical}")

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
        WeightedNeuTraTrainingError,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )

    canary_module = _load_canary_module()
    physical, replay_log_weights, replay_meta = canary_module._load_replay(tf)
    train_rows = tf.ensure_shape(physical[:600], (600, 4))
    train_weights = tf.ensure_shape(replay_log_weights[:600], (600,))
    selection_rows = tf.ensure_shape(physical[600:700], (100, 4))
    selection_weights = tf.ensure_shape(replay_log_weights[600:700], (100,))
    _write(
        output / "launch.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_training_launch.v1",
            "status": "TRAINING_SCREEN_STARTED",
            "timestamp_utc": _utc_now(),
            "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
            "runner": {"path": Path(__file__).as_posix(), "sha256": _sha256(Path(__file__))},
            "prior_artifacts": prior,
            "memory_policy": memory_policy,
            "requested_physical_device_selector": str(args.device),
            "visible_logical_gpus": [str(device) for device in logical],
            "training_batch_size": 600,
            "selection_row_count": 100,
            "audit_rows_loaded_or_evaluated": 0,
            "batch_native_target_backend": "ssl_lstm_q20_batch_native_tensorflow_xla",
            "sample_wise_loop_used": False,
            "scalar_target_fallback_used": False,
            "jit_compile": True,
            "dtype": "float64",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "managed_session_trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
            ),
            "replay": replay_meta,
        },
    )

    arm_summaries: list[Mapping[str, Any]] = []

    def run_and_preserve(
        width: int, stages: int, rate: float, seed: int, role: str
    ) -> None:
        try:
            summary, state = _run_arm(
                tf,
                WeightedForwardKLNeuTraTrainer,
                WeightedNeuTraConfig,
                train_rows,
                train_weights,
                selection_rows,
                selection_weights,
                width=width,
                stages=stages,
                rate=rate,
                seed=seed,
                role=role,
                campaign_started=started,
                time_cap_seconds=float(args.time_cap_seconds),
            )
            state_path = output / "arms" / str(summary["arm_id"]) / "best-state.json"
            _write(
                state_path,
                {
                    "schema": "bayesfilter.ssl_lstm.q20_neutra_training_state.v1",
                    "arm": summary,
                    "state": state,
                },
            )
            completed = {
                **summary,
                "state_path": state_path.as_posix(),
                "state_sha256": _sha256(state_path),
            }
            _write(state_path.with_name("result.json"), completed)
            arm_summaries.append(completed)
        except (WeightedNeuTraTrainingError, ArmNumericalFailure) as error:
            failed = {
                "arm_id": _arm_id(width, stages, rate, seed, role),
                "role": role,
                "status": "NUMERICAL_TRAINING_FAILURE",
                "seed": seed,
                "width": width,
                "stages": stages,
                "learning_rate": rate,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write(output / "arms" / str(failed["arm_id"]) / "result.json", failed)
            arm_summaries.append(failed)
        _write(
            output / f"progress-{len(arm_summaries):02d}.json",
            {
                "schema": "bayesfilter.ssl_lstm.q20_neutra_training_progress.v1",
                "completed_arm_count": len(arm_summaries),
                "arms": arm_summaries,
                "elapsed_wall_seconds": time.perf_counter() - started,
                "remaining_internal_wall_seconds": _remaining_or_raise(
                    started, float(args.time_cap_seconds), "progress checkpoint"
                ),
            },
        )

    for seed in TRAINING_SEEDS:
        for width, stages in PRIMARY_CAPACITIES:
            for rate in PRIMARY_LEARNING_RATES:
                run_and_preserve(width, stages, rate, seed, "primary")

    for seed in TRAINING_SEEDS:
        finite_primary = any(
            arm.get("status") == "TRAINING_COMPLETED"
            and arm.get("role") == "primary"
            and int(arm.get("seed", -1)) == seed
            for arm in arm_summaries
        )
        if not finite_primary:
            run_and_preserve(
                REPAIR_CAPACITY[0],
                REPAIR_CAPACITY[1],
                REPAIR_LEARNING_RATE,
                seed,
                "repair",
            )

    nominations = []
    for seed in TRAINING_SEEDS:
        candidates = [
            arm
            for arm in arm_summaries
            if arm.get("status") == "TRAINING_COMPLETED"
            and int(arm.get("seed", -1)) == seed
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda arm: (
                float(arm["best_selection_loss"]),
                int(arm["width"]),
                float(arm["learning_rate"]),
            ),
        )
        nominations.append(
            {
                "arm_id": selected["arm_id"],
                "seed": seed,
                "selection_rule": (
                    "minimum finite central-06 weighted NLL; exact tie uses smaller "
                    "width then lower learning rate"
                ),
                "selection_loss": selected["best_selection_loss"],
                "best_update": selected["best_update"],
                "state_path": selected["state_path"],
                "state_sha256": selected["state_sha256"],
                "training_state_hash": selected["training_state_hash"],
                "transport_tensor_hash": selected["transport_tensor_hash"],
                "audit_opened": False,
            }
        )
    _write(
        output / "nominations-before-audit.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_nominations_before_audit.v1",
            "status": (
                "NOMINATIONS_FROZEN" if len(nominations) == 2 else "NOMINATION_FAILED"
            ),
            "nomination_count": len(nominations),
            "nominations": nominations,
            "reserved_audit_bank_opened": False,
            "audit_used_for_ranking": False,
        },
    )
    if len(nominations) != 2:
        result = {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_training_screen.v1",
            "status": "TRAINING_PROTOCOL_NEGATIVE_RESULT",
            "reason": "one or more seeds produced no finite complete candidate",
            "arms": arm_summaries,
            "nominations": nominations,
            "audit_rows_loaded_or_evaluated": 0,
            "wall_seconds": time.perf_counter() - started,
        }
        _write(output / "result.json", result)
        return 0

    # This is the first operation permitted to open central-07. The nomination
    # receipt above is immutable and already on disk.
    audit_rows, audit_weights, audit_meta = _load_audit(tf, canary_module)
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    if target.target_signature() != TARGET_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 target signature drift")
    if target.adapter_signature() != ADAPTER_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 adapter signature drift")
    geometry = _read_json(GEOMETRY)
    representatives = tf.constant(
        [
            geometry["representatives"][label]["position"]
            for label in ("plus", "minus")
        ],
        tf.float64,
    )
    audits = []
    for nomination in nominations:
        _remaining_or_raise(started, float(args.time_cap_seconds), "frozen audit")
        audit = _audit_candidate(
            tf,
            WeightedForwardKLNeuTraTrainer,
            WeightedNeuTraConfig,
            FixedTransportValueScoreAdapter,
            BatchNativeBoundAdapter,
            target,
            representatives,
            audit_rows,
            audit_weights,
            nomination,
        )
        audits.append(_json_ready(audit))
        _write(output / "audits" / f"{nomination['arm_id']}.json", audit)

    admitted = [audit for audit in audits if audit.get("passed") is True]
    status = (
        "TRAINING_SCREEN_AND_FROZEN_AUDIT_COMPLETED"
        if admitted
        else "TRAINING_PROTOCOL_NEGATIVE_RESULT"
    )
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    result = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_training_screen.v1",
        "status": status,
        "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
        "runner": {"path": Path(__file__).as_posix(), "sha256": _sha256(Path(__file__))},
        "prior_artifacts": prior,
        "target_signature": TARGET_SIGNATURE,
        "adapter_signature": ADAPTER_SIGNATURE,
        "grid": {
            "capacities": [list(item) for item in PRIMARY_CAPACITIES],
            "learning_rates": list(PRIMARY_LEARNING_RATES),
            "seeds": list(TRAINING_SEEDS),
            "update_ladder": list(UPDATE_LADDER),
            "selection_interval": SELECTION_INTERVAL,
            "repair_capacity": list(REPAIR_CAPACITY),
            "repair_learning_rate": REPAIR_LEARNING_RATE,
        },
        "training_batch_size": 600,
        "selection_row_count": 100,
        "audit_row_count": 100,
        "batch_native_target_backend": "ssl_lstm_q20_batch_native_tensorflow_xla",
        "sample_wise_loop_used": False,
        "scalar_target_fallback_used": False,
        "audit": {
            "bank": "central-07",
            "loaded_after_nomination_receipt": True,
            "used_for_ranking": False,
            "candidate_evaluation_count": len(audits),
            "metadata": audit_meta,
        },
        "arms": arm_summaries,
        "nominations": nominations,
        "frozen_audits": audits,
        "audit_passing_candidate_arm_ids": [audit["arm_id"] for audit in admitted],
        "memory_policy": memory_policy,
        "allocator_bytes": {
            "current": int(allocator["current"]),
            "peak": int(allocator["peak"]),
        },
        "requested_physical_device_selector": str(args.device),
        "visible_logical_gpus": [str(device) for device in logical],
        "managed_session_trust_basis": (
            "owner_designated_managed_session_visible_gpu_trusted"
        ),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "wall_seconds": time.perf_counter() - started,
        "git": _git_manifest(),
        "nonclaims": [
            "selection and audit losses do not establish sampler mixing",
            "the two training seeds are not statistically ranked",
            "SMC replay rows are not posterior draws or HMC mode weights",
            "no posterior, predictive, scientific-validity, or default-readiness claim",
        ],
    }
    _write(output / "result.json", result)
    manifest = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_training_manifest.v1",
        "status": status,
        "command": list(sys.argv),
        "cwd": str(Path.cwd()),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": sys.version.split()[0],
        "git": result["git"],
        "plan": result["plan"],
        "result": (output / "result.json").as_posix(),
        "output_root": output.as_posix(),
        "wall_seconds": result["wall_seconds"],
        "random_seeds": list(TRAINING_SEEDS),
        "gpu_memory_growth_required": True,
        "memory_policy": memory_policy,
        "requested_physical_device_selector": str(args.device),
        "visible_logical_gpus": [str(device) for device in logical],
        "managed_session_trust_basis": (
            "owner_designated_managed_session_visible_gpu_trusted"
        ),
        "training_batch_size": 600,
        "batch_native_target_backend": "ssl_lstm_q20_batch_native_tensorflow_xla",
        "sample_wise_loop_used": False,
        "scalar_target_fallback_used": False,
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": result["tf32_enabled"],
        "data_version": prior,
        "timestamp_completed_utc": _utc_now(),
    }
    _write(output / "manifest.json", manifest)
    artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*.json"))
    }
    _write(
        output / "artifact-hashes.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_training_hashes.v1",
            "artifacts": artifacts,
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "audit_passing_candidates": result["audit_passing_candidate_arm_ids"],
                "output": output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _args()
    _validate_args(args)
    output = _reserve_output_root(args.output_root)
    started = time.perf_counter()
    try:
        return _execute(args, output, started)
    except TrainingBudgetExhausted as error:
        _write(
            output / "terminal-under-budgeted.json",
            {
                "schema": "bayesfilter.ssl_lstm.q20_neutra_training_terminal.v1",
                "status": "UNDER_BUDGETED_TRAINING_SCREEN",
                "reason": str(error),
                "wall_seconds": time.perf_counter() - started,
                "time_cap_seconds": float(args.time_cap_seconds),
                "timestamp_utc": _utc_now(),
                "nonclaim": "an incomplete factorial screen cannot nominate a candidate",
            },
        )
        return 3
    except Exception as error:
        _write(
            output / "failure.json",
            {
                "schema": "bayesfilter.ssl_lstm.q20_neutra_training_failure.v1",
                "status": "TRAINING_HARNESS_FAILURE",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
                "timestamp_utc": _utc_now(),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
