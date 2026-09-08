#!/usr/bin/env python3
"""Run the bounded SSL-LSTM q=20 adaptive-replay A/B screen.

Candidate A uses a fixed known-density proposal and content-independent
whole-block refresh. Candidate B freezes the current transport before drawing
from a known-density mixture containing that transport and uses a deterministic
decaying stale-replay coefficient. The runner is a training screen only: it
does not run HMC and does not make a posterior or whitening claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-ab-comparison-plan-2026-08-24.md"
)
GEOMETRY = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-adaptive-replay-neutra-ab-2026-08-24/r1"
)

DIMENSION = 4
Q = 20
FIXED_MIXTURE_PROBABILITIES = (0.5, 0.5)
ADAPTIVE_TRANSPORT_MIXTURE_FRACTION = 0.5
# A owns four fixed block slots. B has one fresh slot plus three stale slots,
# so both arms present the same static batch shape to the XLA trainer.
BUFFER_CAPACITY = 4
STALE_CAPACITY = 3
REFRESH_PROBABILITY = 0.5
LAMBDA_ZERO = 0.5
LAMBDA_POWER = 1.25
DEFAULT_ROWS_PER_BLOCK = 64
DEFAULT_UPDATES = 24
DEFAULT_WIDTH = 16
DEFAULT_STAGES = 2
DEFAULT_LEARNING_RATE = 3.0e-4
DEFAULT_SEEDS = (17, 29)


class RouteVeto(RuntimeError):
    """Raised when a declared A/B route contract fails closed."""


@dataclass(frozen=True)
class Block:
    rows: Any
    log_weights: Any
    proposal_kind: str
    proposal_metadata: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    generation_step: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
        raise RouteVeto(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _reserve_phase_root(root: Path, phase: str) -> Path:
    absolute = root if root.is_absolute() else ROOT / root
    phase_root = absolute / phase
    if phase_root.exists():
        raise RouteVeto(f"refusing to reuse phase output root: {phase_root}")
    phase_root.mkdir(parents=True, exist_ok=False)
    return phase_root


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


def _state_hash(trainer: Any) -> str:
    digest = hashlib.sha256()
    for variable in trainer.variables:
        digest.update(variable.numpy().tobytes())
    return digest.hexdigest()


def _seed_tensor(tf: Any, seed: tuple[int, int]) -> Any:
    return tf.constant((int(seed[0]), int(seed[1])), tf.int32)


def _split_seed(tf: Any, seed: tuple[int, int], count: int) -> Any:
    return tf.random.experimental.stateless_split(_seed_tensor(tf, seed), count)


def _load_proposal(tf: Any) -> Mapping[str, Any]:
    geometry = json.loads(GEOMETRY.read_text(encoding="ascii"))
    representatives = geometry["representatives"]
    curvature = geometry["source_curvature"]
    means = tf.constant(
        [representatives[label]["position"] for label in ("plus", "minus")],
        tf.float64,
    )
    precisions = tf.constant(
        [curvature[label]["records"][-1]["precision"] for label in ("plus", "minus")],
        tf.float64,
    )
    covariances = tf.linalg.inv(precisions)
    probabilities = tf.constant(FIXED_MIXTURE_PROBABILITIES, tf.float64)
    factors = tf.linalg.cholesky(covariances)
    tf.debugging.assert_all_finite(means, "proposal means")
    tf.debugging.assert_all_finite(covariances, "proposal covariances")
    tf.debugging.assert_all_finite(factors, "proposal Cholesky factors")
    return {
        "probabilities": probabilities,
        "means": means,
        "covariances": covariances,
        "factors": factors,
        "geometry_sha256": _sha256(GEOMETRY),
    }


def _sample_fixed(tf: Any, proposal: Mapping[str, Any], count: int, seed: Any) -> Any:
    split = tf.random.experimental.stateless_split(seed, 2)
    probabilities = proposal["probabilities"]
    means = proposal["means"]
    factors = proposal["factors"]
    labels = tf.reshape(
        tf.random.stateless_categorical(
            tf.math.log(probabilities)[tf.newaxis, :], int(count), seed=split[0]
        ),
        (-1,),
    )
    noise = tf.random.stateless_normal(
        (int(count), DIMENSION), seed=split[1], dtype=tf.float64
    )
    selected_means = tf.gather(means, labels)
    selected_factors = tf.gather(factors, labels)
    rows = selected_means + tf.linalg.matvec(selected_factors, noise)
    return tf.ensure_shape(rows, (int(count), DIMENSION))


def _fixed_log_prob(tf: Any, proposal: Mapping[str, Any], rows: Any) -> Any:
    # This is the same TensorFlow reference density used by the existing
    # importance-sampling diagnostics; no NumPy computation is involved.
    centered = rows[:, tf.newaxis, :] - proposal["means"][tf.newaxis, :, :]
    solved = tf.linalg.triangular_solve(
        proposal["factors"][tf.newaxis, :, :, :],
        centered[:, :, :, tf.newaxis],
        lower=True,
    )[..., 0]
    quadratic = tf.reduce_sum(tf.square(solved), axis=2)
    log_det = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(proposal["factors"])), axis=1
    )
    normalizer = tf.cast(DIMENSION, tf.float64) * tf.math.log(
        tf.constant(2.0 * math.pi, tf.float64)
    ) + log_det[tf.newaxis, :]
    component = -0.5 * (quadratic + normalizer)
    return tf.reduce_logsumexp(
        tf.math.log(proposal["probabilities"])[tf.newaxis, :] + component, axis=1
    )


def _screen_log_weights(tf: Any, log_weights: Any) -> Any:
    """Normalize one block before combining blocks in the empirical screen."""

    values = tf.convert_to_tensor(log_weights, tf.float64)
    centered = values - tf.reduce_logsumexp(values)
    tf.debugging.assert_all_finite(centered, "screen log weights")
    return centered


def _block_diagnostics(tf: Any, log_weights: Any) -> Mapping[str, Any]:
    normalized = tf.nn.softmax(log_weights)
    count = tf.cast(tf.size(normalized), tf.float64)
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
    return {
        "count": int(tf.size(normalized).numpy()),
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / count,
        "maximum_normalized_weight": tf.reduce_max(normalized),
        "log_weight_mean": tf.reduce_mean(log_weights),
        "log_weight_stddev": tf.math.reduce_std(log_weights),
        "log_weight_q01": tfp_quantile(tf, log_weights, 0.01),
        "log_weight_q99": tfp_quantile(tf, log_weights, 0.99),
    }


def tfp_quantile(tf: Any, values: Any, probability: float) -> Any:
    """Small TensorFlow-only linear quantile for artifact diagnostics."""

    ordered = tf.sort(tf.convert_to_tensor(values, tf.float64))
    n = tf.cast(tf.size(ordered) - 1, tf.float64)
    position = tf.cast(probability, tf.float64) * n
    lower = tf.cast(tf.floor(position), tf.int32)
    upper = tf.cast(tf.math.ceil(position), tf.int32)
    fraction = position - tf.floor(position)
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _target_block(
    tf: Any,
    target: Any,
    proposal: Mapping[str, Any],
    rows: Any,
    proposal_log_prob: Any,
    *,
    proposal_kind: str,
    metadata: Mapping[str, Any],
    generation_step: int,
) -> Block:
    target_log_prob, _score, status = target.neutra_batch_log_prob_and_grad_status(rows)
    status_code = tf.cast(status["status_code"], tf.int32)
    valid_status = tf.cast(status["valid_pre_regularized_score"], tf.bool)
    finite = (
        tf.reduce_all(tf.math.is_finite(rows))
        & tf.reduce_all(tf.math.is_finite(proposal_log_prob))
        & tf.reduce_all(tf.math.is_finite(target_log_prob))
    )
    status_ok = tf.reduce_all(tf.logical_and(tf.equal(status_code, 0), valid_status))
    if not bool(finite.numpy()):
        raise RouteVeto(f"{proposal_kind} block contains non-finite target/proposal values")
    if not bool(status_ok.numpy()):
        raise RouteVeto(f"{proposal_kind} block contains an invalid target status")
    log_weights = target_log_prob - proposal_log_prob
    return Block(
        rows=rows,
        log_weights=log_weights,
        proposal_kind=proposal_kind,
        proposal_metadata=dict(metadata),
        diagnostics={
            **_block_diagnostics(tf, log_weights),
            "target_log_prob_min": tf.reduce_min(target_log_prob),
            "target_log_prob_max": tf.reduce_max(target_log_prob),
            "proposal_log_prob_min": tf.reduce_min(proposal_log_prob),
            "proposal_log_prob_max": tf.reduce_max(proposal_log_prob),
            "finite": finite,
            "target_status_valid": status_ok,
        },
        generation_step=int(generation_step),
    )


def _fixed_block(
    tf: Any,
    target: Any,
    proposal: Mapping[str, Any],
    count: int,
    seed: tuple[int, int],
    generation_step: int,
) -> Block:
    rows = _sample_fixed(tf, proposal, count, _seed_tensor(tf, seed))
    proposal_log_prob = _fixed_log_prob(tf, proposal, rows)
    return _target_block(
        tf,
        target,
        proposal,
        rows,
        proposal_log_prob,
        proposal_kind="fixed_known_density",
        metadata={
            "density": "r0_gaussian_mixture",
            "seed": list(seed),
            "full_support": True,
        },
        generation_step=generation_step,
    )


def _adaptive_block(
    tf: Any,
    target: Any,
    proposal: Mapping[str, Any],
    trainer: Any,
    count: int,
    seed: tuple[int, int],
    generation_step: int,
) -> Block:
    before = _state_hash(trainer)
    split = _split_seed(tf, seed, 4)
    alpha = tf.constant(ADAPTIVE_TRANSPORT_MIXTURE_FRACTION, tf.float64)
    labels = tf.reshape(
        tf.random.stateless_categorical(
            tf.math.log(tf.stack((1.0 - alpha, alpha)))[tf.newaxis, :],
            int(count),
            seed=split[0],
        ),
        (-1,),
    )
    fixed_rows = _sample_fixed(tf, proposal, count, split[1])
    base_rows = tf.random.stateless_normal(
        (int(count), DIMENSION), seed=split[2], dtype=tf.float64
    )
    transport_rows = trainer.transport.forward_batch(base_rows)
    rows = tf.where(labels[:, tf.newaxis] == 1, transport_rows, fixed_rows)
    rows = tf.ensure_shape(rows, (int(count), DIMENSION))
    fixed_log = _fixed_log_prob(tf, proposal, rows)
    transport_log = trainer.transport.log_prob(rows)
    proposal_log_prob = tf.reduce_logsumexp(
        tf.stack(
            (
                tf.math.log(1.0 - alpha) + fixed_log,
                tf.math.log(alpha) + transport_log,
            ),
            axis=0,
        ),
        axis=0,
    )
    after = _state_hash(trainer)
    if before != after:
        raise RouteVeto("adaptive proposal state changed during frozen-before-draw block")
    return _target_block(
        tf,
        target,
        proposal,
        rows,
        proposal_log_prob,
        proposal_kind="adaptive_known_density",
        metadata={
            "density": "(1-alpha)*r0+alpha*q_phi",
            "alpha": ADAPTIVE_TRANSPORT_MIXTURE_FRACTION,
            "seed": list(seed),
            "transport_state_hash_before": before,
            "transport_state_hash_after": after,
            "transport_component_count": int(
                tf.reduce_sum(tf.cast(labels == 1, tf.int32)).numpy()
            ),
            "fixed_component_count": int(
                tf.reduce_sum(tf.cast(labels == 0, tf.int32)).numpy()
            ),
            "full_support_from_fixed_component": True,
        },
        generation_step=generation_step,
    )


def _combine_blocks(tf: Any, blocks: list[Block], coefficients: list[float]) -> tuple[Any, Any]:
    if not blocks or len(blocks) != len(coefficients):
        raise RouteVeto("cannot combine an empty or mismatched block list")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in coefficients):
        raise RouteVeto("block coefficients must be finite and positive")
    total_count = sum(int(block.rows.shape[0]) for block in blocks)
    rows = tf.ensure_shape(
        tf.concat([block.rows for block in blocks], axis=0),
        (total_count, DIMENSION),
    )
    weighted_logs = [
        _screen_log_weights(tf, block.log_weights)
        + tf.math.log(tf.constant(float(coefficient), tf.float64))
        for block, coefficient in zip(blocks, coefficients, strict=True)
    ]
    return rows, tf.ensure_shape(tf.concat(weighted_logs, axis=0), (total_count,))


def _validation_summary(tf: Any, validation: Any) -> Mapping[str, Any]:
    return {
        "loss": validation.loss,
        "effective_sample_size": validation.effective_sample_size,
        "effective_sample_size_fraction": validation.effective_sample_size_fraction,
        "maximum_normalized_weight": validation.maximum_normalized_weight,
        "latent_weighted_mean": validation.latent_weighted_mean,
        "latent_weighted_covariance": validation.latent_weighted_covariance,
        "latent_finite": tf.reduce_all(tf.math.is_finite(validation.latent)),
    }


def _config(config_type: Any, seed: int, args: argparse.Namespace) -> Any:
    return config_type(
        dimension=DIMENSION,
        hidden_layers=(int(args.hidden_width), int(args.hidden_width)),
        stages=int(args.stages),
        activation="tanh",
        s_max=2.0,
        permutation_policy="full_reverse",
        initialization_scale=0.02,
        initialization_seed=(20260824, int(seed)),
        learning_rate=float(args.learning_rate),
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )


def _run_arm(
    tf: Any,
    trainer_type: Any,
    config_type: Any,
    target: Any,
    proposal: Mapping[str, Any],
    validation: Block,
    *,
    arm: str,
    seed: int,
    args: argparse.Namespace,
    started: float,
) -> Mapping[str, Any]:
    trainer = trainer_type(_config(config_type, seed, args))
    stale: list[Block] = []
    fixed_buffer: list[Block | None] = [None] * BUFFER_CAPACITY
    updates = []
    refresh_history = []
    block_records = []
    arm_seed = int(seed) + (100000 if arm == "B" else 0)
    if arm == "B":
        # Pre-fill the fixed stale capacity while phi_0 is frozen. This keeps
        # every subsequent B update at the same static batch shape as A.
        for slot in range(STALE_CAPACITY):
            initial = _adaptive_block(
                tf,
                target,
                proposal,
                trainer,
                int(args.rows_per_block),
                (20260824, arm_seed * 100000 + slot),
                0,
            )
            stale.append(initial)
            block_records.append(
                {
                    "update": 0,
                    "slot": slot,
                    "fresh": initial.diagnostics,
                    "fresh_metadata": initial.proposal_metadata,
                    "role": "initial_stale_prefill",
                }
            )
    for update in range(1, int(args.updates) + 1):
        if time.perf_counter() - started > float(args.max_seconds):
            raise RouteVeto(f"campaign cap reached during arm {arm} seed {seed}")
        if arm == "A":
            refreshed = []
            for slot in range(BUFFER_CAPACITY):
                coin_seed = (20260824, arm_seed * 10000 + update * 100 + slot)
                coin = tf.random.stateless_uniform((), seed=_seed_tensor(tf, coin_seed))
                replace = update == 1 or bool((coin < REFRESH_PROBABILITY).numpy())
                if replace:
                    block = _fixed_block(
                        tf,
                        target,
                        proposal,
                        int(args.rows_per_block),
                        (20260824, arm_seed * 100000 + update * 100 + slot),
                        update,
                    )
                    fixed_buffer[slot] = block
                refreshed.append(bool(replace))
            if any(block is None for block in fixed_buffer):
                raise RouteVeto("Candidate A buffer was not fully initialized")
            blocks = [block for block in fixed_buffer if block is not None]
            rows, logs = _combine_blocks(tf, blocks, [1.0 / BUFFER_CAPACITY] * BUFFER_CAPACITY)
            lambda_value = 0.0
            stale_count = 0
            refresh_history.append({"update": update, "refreshed": refreshed})
        else:
            fresh = _adaptive_block(
                tf,
                target,
                proposal,
                trainer,
                int(args.rows_per_block),
                (20260824, arm_seed * 100000 + update),
                update,
            )
            active_blocks = [fresh]
            coefficients = [1.0]
            lambda_value = LAMBDA_ZERO / (float(update) ** LAMBDA_POWER)
            stale_weight = lambda_value / float(len(stale))
            active_blocks.extend(stale)
            coefficients.extend([stale_weight] * len(stale))
            rows, logs = _combine_blocks(tf, active_blocks, coefficients)
            stale_count = len(stale)
            stale.append(fresh)
            if len(stale) > STALE_CAPACITY:
                stale.pop(0)
            block_records.append(
                {
                    "update": update,
                    "fresh": fresh.diagnostics,
                    "fresh_metadata": fresh.proposal_metadata,
                }
            )
        expected_batch = int(args.rows_per_block) * BUFFER_CAPACITY
        rows = tf.ensure_shape(rows, (expected_batch, DIMENSION))
        logs = tf.ensure_shape(logs, (expected_batch,))
        step = trainer.train_step(rows, logs)
        validation_summary = trainer.validation_batch(
            validation.rows, _screen_log_weights(tf, validation.log_weights)
        )
        updates.append(
            {
                "update": update,
                "training_loss": step.loss,
                "gradient_norm": step.gradient_norm,
                "clipped_gradient_norm": step.clipped_gradient_norm,
                "effective_sample_size": step.effective_sample_size,
                "effective_sample_size_fraction": step.effective_sample_size_fraction,
                "maximum_normalized_weight": step.maximum_normalized_weight,
                "validation": _validation_summary(tf, validation_summary),
                "lambda_t": lambda_value,
                "stale_block_count": stale_count,
                "batch_size": int(rows.shape[0]),
            }
        )
    final_validation = trainer.validation_batch(
        validation.rows, _screen_log_weights(tf, validation.log_weights)
    )
    return {
        "schema": "bayesfilter.ssl_lstm.q20_adaptive_replay_arm_result.v1",
        "status": "SCREEN_COMPLETED",
        "arm": arm,
        "seed": int(seed),
        "config": trainer.config.manifest_payload(),
        "updates": updates,
        "refresh_history": refresh_history,
        "adaptive_block_records": block_records,
        "final_validation": _validation_summary(tf, final_validation),
        "transport_state_hash": _state_hash(trainer),
        "nonclaims": [
            "finite self-normalized screen estimator is not Theorem-1 unbiased",
            "training loss and latent moments do not prove whitening",
            "no mode-discovery, HMC, posterior, predictive, or superiority claim",
        ],
    }


def _preflight(args: argparse.Namespace, phase_root: Path) -> Mapping[str, Any]:
    import tensorflow as tf

    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )

    proposal = _load_proposal(tf)
    target = batch_native_complexity_posterior_target(
        Q, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    records = []
    for seed in DEFAULT_SEEDS:
        fixed = _fixed_block(tf, target, proposal, max(8, int(args.rows_per_block)), (20260824, seed), 0)
        trainer = WeightedForwardKLNeuTraTrainer(_config(WeightedNeuTraConfig, seed, args))
        adaptive = _adaptive_block(
            tf,
            target,
            proposal,
            trainer,
            max(8, int(args.rows_per_block)),
            (20260824, 500000 + seed),
            0,
        )
        fixed_log_recomputed = _fixed_log_prob(tf, proposal, fixed.rows)
        # The expression above is intentionally not a density identity: log
        # weights include the target. Record the correct proposal recomputation
        # separately below and use it as the actual gate.
        target_log, _score, _status = target.neutra_batch_log_prob_and_grad_status(fixed.rows)
        proposal_recomputed = target_log - fixed.log_weights
        proposal_gate = tf.reduce_max(tf.abs(proposal_recomputed - fixed_log_recomputed))
        lambda_sum_bound = LAMBDA_ZERO * (1.0 + 1.0 / (LAMBDA_POWER - 1.0))
        records.append(
            {
                "seed": seed,
                "fixed": fixed.diagnostics,
                "adaptive": adaptive.diagnostics,
                "adaptive_metadata": adaptive.proposal_metadata,
                "fixed_density_recomputation_max_abs_residual": proposal_gate,
                "summability": {
                    "lambda_zero": LAMBDA_ZERO,
                    "power": LAMBDA_POWER,
                    "infinite_series_upper_bound": lambda_sum_bound,
                    "deterministic": True,
                },
            }
        )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20_adaptive_replay_preflight.v1",
        "status": "PREFLIGHT_PASSED",
        "phase": "cpu_preflight" if args.cpu_only else "gpu_preflight",
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha256(PLAN),
        "geometry": GEOMETRY.as_posix(),
        "geometry_sha256": proposal["geometry_sha256"],
        "target_signature": target.target_signature(),
        "rows_per_block": int(args.rows_per_block),
        "records": records,
        "batch_native_target_backend": "ssl_lstm_q20_batch_native_tensorflow_xla",
        "scalar_target_fallback_used": False,
        "jit_compile": True,
        "nonclaims": [
            "preflight does not establish proposal completeness",
            "preflight does not establish boundedness assumptions",
            "preflight does not establish whitening, HMC, or posterior correctness",
        ],
    }
    _write_json(phase_root / "preflight.json", result)
    return result


def _screen(args: argparse.Namespace, phase_root: Path, preflight: Mapping[str, Any]) -> Mapping[str, Any]:
    import tensorflow as tf

    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )

    proposal = _load_proposal(tf)
    target = batch_native_complexity_posterior_target(
        Q, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    started = time.perf_counter()
    arms = []
    for seed in DEFAULT_SEEDS:
        validation = _fixed_block(
            tf,
            target,
            proposal,
            int(args.rows_per_block),
            (20260824, 900000 + seed),
            -1,
        )
        for arm in ("A", "B"):
            arm_root = phase_root / f"arm-{arm}" / f"seed-{seed}"
            arm_root.mkdir(parents=True, exist_ok=False)
            try:
                result = _run_arm(
                    tf,
                    WeightedForwardKLNeuTraTrainer,
                    WeightedNeuTraConfig,
                    target,
                    proposal,
                    validation,
                    arm=arm,
                    seed=seed,
                    args=args,
                    started=started,
                )
            except Exception as exc:
                failure = {
                    "schema": "bayesfilter.ssl_lstm.q20_adaptive_replay_arm_result.v1",
                    "status": "SCREEN_FAILED",
                    "arm": arm,
                    "seed": seed,
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                    "nonclaims": ["failed arm is not evidence against the replay research direction"],
                }
                _write_json(arm_root / "result.json", failure)
                raise
            _write_json(arm_root / "result.json", result)
            arms.append({"arm": arm, "seed": seed, "status": result["status"], "path": (arm_root / "result.json").as_posix()})
    result = {
        "schema": "bayesfilter.ssl_lstm_q20_adaptive_replay_ab_screen.v1",
        "status": "SCREEN_COMPLETED",
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha256(PLAN),
        "preflight": preflight,
        "arms": arms,
        "target_signature": target.target_signature(),
        "geometry_sha256": proposal["geometry_sha256"],
        "training_batch_size": int(args.rows_per_block) * BUFFER_CAPACITY,
        "batch_native_target_backend": "ssl_lstm_q20_batch_native_tensorflow_xla",
        "sample_wise_loop_used": False,
        "scalar_target_fallback_used": False,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "wall_seconds": time.perf_counter() - started,
        "git": _git_manifest(),
        "nonclaims": [
            "A/B descriptive diagnostics do not establish statistical superiority",
            "screen estimator is self-normalized and finite-block",
            "no HMC, posterior, predictive, mode-discovery, or default-readiness claim",
        ],
    }
    _write_json(phase_root / "result.json", result)
    return result


def _configure_runtime(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if not str(args.device).isdigit():
            raise RouteVeto("GPU device must be a nonnegative physical index")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.cpu_only:
        return {"cpu_only": True, "cuda_visible_devices": "-1"}
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    import tensorflow as tf

    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RouteVeto(f"expected one visible logical GPU, found {logical}")
    return {
        "cpu_only": False,
        "requested_physical_device_selector": str(args.device),
        "visible_logical_gpus": [str(device) for device in logical],
        "memory_policy": policy,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "screen"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device", default="1")
    parser.add_argument("--rows-per-block", type=int, default=DEFAULT_ROWS_PER_BLOCK)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--hidden-width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--stages", type=int, default=DEFAULT_STAGES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-seconds", type=float, default=61200.0)
    args = parser.parse_args()
    if args.rows_per_block < 2 or args.updates < 1:
        raise SystemExit("rows-per-block must be >=2 and updates must be positive")
    if args.hidden_width < 1 or args.stages < 1 or args.learning_rate <= 0.0:
        raise SystemExit("invalid transport configuration")
    runtime = _configure_runtime(args)
    phase_root = _reserve_phase_root(args.output_root, args.phase)
    _write_json(
        phase_root / "launch.json",
        {
            "schema": "bayesfilter.ssl_lstm_q20_adaptive_replay_launch.v1",
            "status": "LAUNCH_RECORDED",
            "timestamp_utc": _utc_now(),
            "phase": args.phase,
            "command": list(sys.argv),
            "cwd": str(Path.cwd()),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "runtime": runtime,
            "git": _git_manifest(),
            "plan": PLAN.as_posix(),
            "plan_sha256": _sha256(PLAN),
            "geometry": GEOMETRY.as_posix(),
            "geometry_sha256": _sha256(GEOMETRY),
            "batch_size": int(args.rows_per_block) * BUFFER_CAPACITY,
            "jit_compile": True,
            "dtype": "float64",
        },
    )
    if args.phase == "preflight":
        result = _preflight(args, phase_root)
    else:
        preflight_path = (args.output_root if args.output_root.is_absolute() else ROOT / args.output_root) / "preflight" / "preflight.json"
        if not preflight_path.is_file():
            raise RouteVeto(f"screen requires completed preflight: {preflight_path}")
        preflight = json.loads(preflight_path.read_text(encoding="ascii"))
        if preflight.get("status") != "PREFLIGHT_PASSED":
            raise RouteVeto("screen refuses a failed preflight")
        result = _screen(args, phase_root, preflight)
    print(json.dumps({"status": result["status"], "phase_root": phase_root.as_posix()}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RouteVeto as exc:
        print(f"ROUTE_VETO: {exc}", file=sys.stderr)
        raise SystemExit(2)
