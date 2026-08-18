#!/usr/bin/env python3
"""Run bounded analytic GPU/XLA checks for defensive weighted NeuTra training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = "docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md"
POSITIVE_CONTROL_PLAN = (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md"
)
DEFAULT_ROOT = Path(
    "docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11"
)
DTYPE_NAME = "float64"
SEED_ROOT = 20260811


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("gaussian-canary", "two-mode-canary", "three-mode-canary"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--audit-size", type=int, default=65_536)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--plan-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    seed_root = SEED_ROOT + 100_000 * int(args.replication)
    active_plan = (
        args.plan_file.resolve()
        if args.plan_file is not None
        else (ROOT / (POSITIVE_CONTROL_PLAN if args.mode == "three-mode-canary" else PLAN))
    )
    if not active_plan.is_file():
        raise FileNotFoundError(f"active plan is missing: {active_plan}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)

    # This must precede the TensorFlow import and any device discovery.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.inference.neutra_weighted_training import (
        MatchedReverseKLNeuTraTrainer,
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob,
        gaussian_mixture_log_prob_responsibilities_score,
        sample_gaussian_mixture,
        self_normalized_importance_diagnostics,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected exactly one visible logical GPU, found {logical_gpus}")

    dtype = tf.float64
    target = _target(args.mode, tf, dtype)
    dimension = int(target["means"].shape[1])
    config = WeightedNeuTraConfig(
        dimension=dimension,
        hidden_layers=(int(args.hidden_width), int(args.hidden_width)),
        stages=int(args.stages),
        activation="tanh",
        initialization_scale=0.02,
        initialization_seed=(seed_root, 11_001),
        learning_rate=1.0e-3,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )

    target_log_prob = lambda rows: gaussian_mixture_log_prob(
        rows,
        target["target_probabilities"],
        target["means"],
        target["covariances"],
    )
    proposal_log_prob = lambda rows: gaussian_mixture_log_prob(
        rows,
        target["proposal_probabilities"],
        target["proposal_means"],
        target["proposal_covariances"],
    )

    weighted = WeightedForwardKLNeuTraTrainer(config)
    reverse = MatchedReverseKLNeuTraTrainer(config, target_log_prob)
    initial_variable_parity = all(
        bool(tf.reduce_all(left == right).numpy())
        for left, right in zip(weighted.variables, reverse.variables)
    )
    if not initial_variable_parity:
        raise RuntimeError("matched trainers did not start from identical variables")

    selection_rows, selection_labels = sample_gaussian_mixture(
        int(args.audit_size),
        target["proposal_probabilities"],
        target["proposal_means"],
        target["proposal_covariances"],
        seed=(seed_root, 12_001),
    )
    selection_target = target_log_prob(selection_rows)
    selection_proposal = proposal_log_prob(selection_rows)
    selection_log_weights = selection_target - selection_proposal

    checkpoints: list[Mapping[str, Any]] = []
    checkpoints.append(
        _checkpoint(
            tf,
            update=0,
            weighted=weighted,
            reverse=reverse,
            rows=selection_rows,
            log_weights=selection_log_weights,
        )
    )
    best_weighted = _snapshot(weighted)
    best_weighted_nll = float(checkpoints[-1]["weighted_heldout_nll"])
    best_weighted_update = 0
    best_reverse = _snapshot(reverse)
    best_reverse_nll = float(checkpoints[-1]["reverse_heldout_nll"])
    best_reverse_update = 0
    clipping = {"weighted": 0, "reverse_kl": 0}
    last_steps: Mapping[str, Any] = {}

    for update in range(1, int(args.updates) + 1):
        train_rows, _train_labels = sample_gaussian_mixture(
            int(args.batch_size),
            target["proposal_probabilities"],
            target["proposal_means"],
            target["proposal_covariances"],
            seed=(seed_root, 20_000 + update),
        )
        train_log_weights = target_log_prob(train_rows) - proposal_log_prob(train_rows)
        weighted_step = weighted.train_step(train_rows, train_log_weights)

        latent = tf.random.stateless_normal(
            (int(args.batch_size), dimension),
            seed=(seed_root, 30_000 + update),
            dtype=dtype,
        )
        reverse_step = reverse.train_step(latent)
        clipping["weighted"] += int(bool(weighted_step.clipping_applied.numpy()))
        clipping["reverse_kl"] += int(bool(reverse_step.clipping_applied.numpy()))
        last_steps = {
            "weighted": _step_payload(weighted_step),
            "reverse_kl": _step_payload(reverse_step),
        }

        if update % int(args.checkpoint_every) == 0 or update == int(args.updates):
            row = _checkpoint(
                tf,
                update=update,
                weighted=weighted,
                reverse=reverse,
                rows=selection_rows,
                log_weights=selection_log_weights,
            )
            checkpoints.append(row)
            weighted_nll = float(row["weighted_heldout_nll"])
            reverse_nll = float(row["reverse_heldout_nll"])
            if weighted_nll < best_weighted_nll:
                best_weighted_nll = weighted_nll
                best_weighted_update = update
                best_weighted = _snapshot(weighted)
            if reverse_nll < best_reverse_nll:
                best_reverse_nll = reverse_nll
                best_reverse_update = update
                best_reverse = _snapshot(reverse)

    _restore(weighted, best_weighted)
    _restore(reverse, best_reverse)

    audit_rows, audit_labels = sample_gaussian_mixture(
        int(args.audit_size),
        target["proposal_probabilities"],
        target["proposal_means"],
        target["proposal_covariances"],
        seed=(seed_root, 40_001),
    )
    audit_target = target_log_prob(audit_rows)
    audit_proposal = proposal_log_prob(audit_rows)
    audit_log_weights = audit_target - audit_proposal
    initial = WeightedForwardKLNeuTraTrainer(config)
    importance = self_normalized_importance_diagnostics(
        audit_target,
        audit_proposal,
        audit_labels == 0,
    )
    audit = {
        "initial": _transport_audit(
            tf,
            initial,
            audit_rows,
            audit_log_weights,
            target,
            base_seed=(seed_root, 50_000),
        ),
        "weighted": _transport_audit(
            tf,
            weighted,
            audit_rows,
            audit_log_weights,
            target,
            base_seed=(seed_root, 50_001),
        ),
        "reverse_kl": _transport_audit(
            tf,
            reverse,
            audit_rows,
            audit_log_weights,
            target,
            base_seed=(seed_root, 50_002),
        ),
    }

    if args.mode == "gaussian-canary":
        decision = _gaussian_canary_decision(audit)
    else:
        decision = _mixture_canary_decision(
            tf,
            weighted,
            reverse,
            target,
            int(args.audit_size),
            seed_root=seed_root,
        )
        audit["base_component_coverage"] = decision.pop("base_component_coverage")

    allocator = tf.config.experimental.get_memory_info("GPU:0")
    wall_time = time.monotonic() - started
    manifest = {
        "schema": "bayesfilter.defensive_weighted_neutra_analytic_manifest.v1",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_status_short": subprocess.run(
            ("git", "status", "--short"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": sys.version,
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "device": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_memory_policy": memory_policy,
        "allocator_bytes": {key: int(value) for key, value in allocator.items()},
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "dtype": DTYPE_NAME,
        "batch_native_target_backend": "tensorflow_full_batch_gaussian_mixture",
        "sample_wise_loop_or_scalar_fallback": False,
        "training_batch_size": int(args.batch_size),
        "updates_per_arm": int(args.updates),
        "target_density_evaluations_per_arm": int(args.batch_size) * int(args.updates),
        "selection_size": int(args.audit_size),
        "audit_size": int(args.audit_size),
        "seed_domains": {
            "initialization": [seed_root, 11_001],
            "selection": [seed_root, 12_001],
            "training_weighted": f"[{seed_root}, 20001:{20_000 + int(args.updates)}]",
            "training_reverse": f"[{seed_root}, 30001:{30_000 + int(args.updates)}]",
            "audit": [seed_root, 40_001],
            "base_audit": [[seed_root, 50_001], [seed_root, 50_002]],
        },
        "started_at_utc": started_at.isoformat(),
        "wall_time_seconds": wall_time,
        "plan_file": PLAN,
        "active_plan_file": active_plan.as_posix(),
        "result_file": str(output_root / "result.json"),
        "output_root": str(output_root),
        "mode": args.mode,
        "replication": int(args.replication),
        "capacity_arm": {
            "hidden_width": int(args.hidden_width),
            "hidden_layers": 2,
            "stages": int(args.stages),
        },
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }
    result = {
        "schema": "bayesfilter.defensive_weighted_neutra_analytic_result.v1",
        "mode": args.mode,
        "replication": int(args.replication),
        "capacity_arm": {
            "hidden_width": int(args.hidden_width),
            "hidden_layers": 2,
            "stages": int(args.stages),
        },
        "research_question": (
            "Does defensive weighted forward-KL learn global target geometry more "
            "faithfully than matched reverse KL on an analytic target?"
        ),
        "target": _ready(target),
        "config": config.manifest_payload(),
        "initial_variable_parity": initial_variable_parity,
        "checkpoint_selection": {
            "criterion": "minimum disjoint heldout weighted negative log likelihood",
            "weighted_update": best_weighted_update,
            "weighted_nll": best_weighted_nll,
            "reverse_kl_update": best_reverse_update,
            "reverse_kl_nll": best_reverse_nll,
        },
        "checkpoints": checkpoints,
        "clipping_update_counts": clipping,
        "last_steps": last_steps,
        "importance_audit": _importance_summary(importance),
        "audit": audit,
        "decision": decision,
        "nonclaims": [
            "canary thresholds are calibration hypotheses, not promoted defaults",
            "training loss or heldout NLL alone does not establish posterior correctness",
            "analytic-target success does not establish SSL-LSTM validity",
            "no HMC or default-readiness claim",
        ],
        "run_manifest": manifest,
    }
    _write(output_root / "result.json", result)
    _write(output_root / "run_manifest.json", manifest)
    _write(
        output_root / "trainer_states.json",
        {
            "weighted": weighted.state_payload(),
            "reverse_kl": _reverse_state_payload(reverse),
        },
    )
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.defensive_weighted_neutra_analytic_hashes.v1",
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
                "completed": True,
                "mode": args.mode,
                "candidate_passed": bool(decision["candidate_passed"]),
                "wall_time_seconds": wall_time,
                "output_root": str(output_root),
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "updates",
        "batch_size",
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
    if int(args.replication) < 0:
        raise ValueError("replication must be nonnegative")


def _target(mode: str, tf: Any, dtype: Any) -> Mapping[str, Any]:
    if mode == "gaussian-canary":
        factor = tf.constant(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.8, 0.3, 0.0, 0.0),
                (0.2, -0.15, 0.12, 0.0),
                (-0.1, 0.05, 0.08, 0.04),
            ),
            dtype,
        )
        covariance = tf.matmul(factor, factor, transpose_b=True)
        mean = tf.constant(((1.0, -1.0, 0.5, -0.5),), dtype)
        return {
            "identity": "correlated_ill_conditioned_gaussian_d4_v1",
            "target_probabilities": tf.constant((1.0,), dtype),
            "means": mean,
            "covariances": covariance[tf.newaxis, :, :],
            "proposal_probabilities": tf.constant((1.0,), dtype),
            "proposal_means": mean,
            "proposal_covariances": (1.5 * covariance)[tf.newaxis, :, :],
            "proposal_scale_hypothesis": 1.5,
            "true_mean": mean[0],
            "true_covariance": covariance,
        }
    if mode == "two-mode-canary":
        means = tf.constant(
            ((-4.0, -0.5, 0.75, -0.25), (4.0, 0.5, -0.75, 0.25)), dtype
        )
        factors = tf.constant(
            (
                ((0.8, 0.0, 0.0, 0.0), (0.2, 0.6, 0.0, 0.0), (0.0, 0.1, 0.5, 0.0), (0.1, 0.0, 0.15, 0.4)),
                ((0.5, 0.0, 0.0, 0.0), (-0.1, 0.9, 0.0, 0.0), (0.05, -0.2, 0.7, 0.0), (0.0, 0.1, -0.1, 0.55)),
            ),
            dtype,
        )
        target_probabilities = tf.constant((0.8, 0.2), dtype)
        proposal_probabilities = tf.constant((0.5, 0.5), dtype)
        identity = "separated_two_mode_unequal_weight_d4_v1"
        proposal_scale = 1.0
    elif mode == "three-mode-canary":
        means = tf.constant(
            (
                (-4.5, -1.0, 0.8, -0.4),
                (4.0, -1.8, -0.7, 0.5),
                (0.5, 4.8, 0.2, -0.6),
            ),
            dtype,
        )
        factors = tf.constant(
            (
                ((0.75, 0.0, 0.0, 0.0), (0.18, 0.55, 0.0, 0.0), (0.05, 0.10, 0.45, 0.0), (0.08, -0.03, 0.12, 0.38)),
                ((0.48, 0.0, 0.0, 0.0), (-0.16, 0.88, 0.0, 0.0), (0.08, -0.22, 0.68, 0.0), (0.02, 0.14, -0.09, 0.52)),
                ((0.62, 0.0, 0.0, 0.0), (0.28, 0.58, 0.0, 0.0), (-0.12, 0.16, 0.82, 0.0), (0.10, 0.04, 0.20, 0.44)),
            ),
            dtype,
        )
        target_probabilities = tf.constant((0.5, 0.3, 0.2), dtype)
        proposal_probabilities = tf.constant((1.0 / 3.0,) * 3, dtype)
        identity = "separated_three_mode_unequal_weight_d4_v1"
        proposal_scale = 1.5
    else:
        raise ValueError(f"unsupported mode: {mode}")
    covariances = tf.matmul(factors, factors, transpose_b=True)
    true_mean = tf.reduce_sum(target_probabilities[:, tf.newaxis] * means, axis=0)
    centered = means - true_mean
    true_covariance = tf.reduce_sum(
        target_probabilities[:, tf.newaxis, tf.newaxis]
        * (covariances + centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :]),
        axis=0,
    )
    return {
        "identity": identity,
        "target_probabilities": target_probabilities,
        "means": means,
        "covariances": covariances,
        "proposal_probabilities": proposal_probabilities,
        "proposal_means": means,
        "proposal_covariances": proposal_scale * covariances,
        "proposal_scale_hypothesis": proposal_scale,
        "true_mean": true_mean,
        "true_covariance": true_covariance,
    }


def _checkpoint(
    tf: Any,
    *,
    update: int,
    weighted: Any,
    reverse: Any,
    rows: Any,
    log_weights: Any,
) -> Mapping[str, Any]:
    normalized = tf.nn.softmax(log_weights)

    def evaluate(trainer: Any) -> tuple[Any, Any, Any]:
        nll = -trainer.log_prob(rows)
        latent, _ = trainer.transport.inverse_and_forward_logdet(rows)
        mean = tf.reduce_sum(normalized[:, tf.newaxis] * latent, axis=0)
        centered = latent - mean
        covariance = tf.matmul(
            centered,
            normalized[:, tf.newaxis] * centered,
            transpose_a=True,
        )
        return tf.reduce_sum(normalized * nll), mean, covariance

    weighted_nll, weighted_mean, weighted_covariance = evaluate(weighted)
    reverse_nll, reverse_mean, reverse_covariance = evaluate(reverse)
    return {
        "update": int(update),
        "weighted_heldout_nll": float(weighted_nll.numpy()),
        "weighted_latent_mean_norm": float(tf.linalg.norm(weighted_mean).numpy()),
        "weighted_latent_covariance_error": float(
            tf.linalg.norm(
                weighted_covariance - tf.eye(int(rows.shape[1]), dtype=tf.float64)
            ).numpy()
        ),
        "reverse_heldout_nll": float(reverse_nll.numpy()),
        "reverse_latent_mean_norm": float(tf.linalg.norm(reverse_mean).numpy()),
        "reverse_latent_covariance_error": float(
            tf.linalg.norm(
                reverse_covariance - tf.eye(int(rows.shape[1]), dtype=tf.float64)
            ).numpy()
        ),
    }


def _transport_audit(
    tf: Any,
    trainer: Any,
    rows: Any,
    log_weights: Any,
    target: Mapping[str, Any],
    *,
    base_seed: tuple[int, int],
) -> Mapping[str, Any]:
    normalized = tf.nn.softmax(log_weights)
    dimension = int(rows.shape[1])
    nll = -trainer.log_prob(rows)
    latent, _ = trainer.transport.inverse_and_forward_logdet(rows)
    latent_mean = tf.reduce_sum(normalized[:, tf.newaxis] * latent, axis=0)
    latent_centered = latent - latent_mean
    latent_covariance = tf.matmul(
        latent_centered,
        normalized[:, tf.newaxis] * latent_centered,
        transpose_a=True,
    )
    base = tf.random.stateless_normal(
        (int(rows.shape[0]), dimension), seed=base_seed, dtype=tf.float64
    )
    physical, _ = trainer.forward_and_logdet(base)
    physical_mean = tf.reduce_mean(physical, axis=0)
    centered = physical - physical_mean
    physical_covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(physical)[0], tf.float64
    )
    true_covariance = target["true_covariance"]
    covariance_scale = tf.linalg.norm(true_covariance)
    return {
        "weighted_nll": float(tf.reduce_sum(normalized * nll).numpy()),
        "latent_weighted_mean": latent_mean.numpy().tolist(),
        "latent_weighted_mean_norm": float(tf.linalg.norm(latent_mean).numpy()),
        "latent_weighted_covariance": latent_covariance.numpy().tolist(),
        "latent_covariance_error_frobenius": float(
            tf.linalg.norm(
                latent_covariance - tf.eye(dimension, dtype=tf.float64)
            ).numpy()
        ),
        "base_pushforward_mean": physical_mean.numpy().tolist(),
        "base_pushforward_mean_error": float(
            tf.linalg.norm(physical_mean - target["true_mean"]).numpy()
        ),
        "base_pushforward_covariance": physical_covariance.numpy().tolist(),
        "base_pushforward_relative_covariance_error": float(
            (tf.linalg.norm(physical_covariance - true_covariance) / covariance_scale).numpy()
        ),
        "all_finite": bool(
            tf.reduce_all(
                tf.stack(
                    (
                        tf.reduce_all(tf.math.is_finite(nll)),
                        tf.reduce_all(tf.math.is_finite(latent)),
                        tf.reduce_all(tf.math.is_finite(physical)),
                    )
                )
            ).numpy()
        ),
    }


def _gaussian_canary_decision(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    initial = audit["initial"]
    weighted = audit["weighted"]
    reverse = audit["reverse_kl"]
    gates = {
        "finite": bool(weighted["all_finite"]),
        "heldout_nll_improved_from_initial": (
            float(weighted["weighted_nll"]) < float(initial["weighted_nll"])
        ),
        # Calibration hypotheses for the bounded canary, not promoted defaults.
        "latent_mean_norm_below_0p10": float(weighted["latent_weighted_mean_norm"]) < 0.10,
        "latent_covariance_error_below_0p20": (
            float(weighted["latent_covariance_error_frobenius"]) < 0.20
        ),
        "pushforward_relative_covariance_error_below_0p10": (
            float(weighted["base_pushforward_relative_covariance_error"]) < 0.10
        ),
    }
    return {
        "candidate_passed": all(gates.values()),
        "gates": gates,
        "threshold_status": "canary_calibration_hypotheses_only",
        "weighted_vs_reverse_nll_difference": float(weighted["weighted_nll"])
        - float(reverse["weighted_nll"]),
        "interpretation": (
            "A pass permits the predeclared unequal-weight canary; it does not promote "
            "a default or establish posterior correctness."
        ),
    }


def _mixture_canary_decision(
    tf: Any,
    weighted: Any,
    reverse: Any,
    target: Mapping[str, Any],
    audit_size: int,
    *,
    seed_root: int,
) -> Mapping[str, Any]:
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob_responsibilities_score,
    )

    component_count = int(target["means"].shape[0])
    dimension = int(target["means"].shape[1])
    coverage = {}
    gates = {}
    for index, (name, trainer) in enumerate((('weighted', weighted), ('reverse_kl', reverse))):
        base = tf.random.stateless_normal(
            (int(audit_size), dimension),
            seed=(seed_root, 60_001 + index),
            dtype=tf.float64,
        )
        physical, _ = trainer.forward_and_logdet(base)
        _value, responsibilities, _score = gaussian_mixture_log_prob_responsibilities_score(
            physical,
            target["target_probabilities"],
            target["means"],
            target["covariances"],
        )
        assigned = tf.argmax(responsibilities, axis=1, output_type=tf.int32)
        probabilities = tf.stack(
            [
                tf.reduce_mean(tf.cast(assigned == component, tf.float64))
                for component in range(component_count)
            ]
        )
        all_components_observed = bool(tf.reduce_all(probabilities > 0.0).numpy())
        coverage[name] = {
            "hard_assignment_component_probabilities": probabilities.numpy().tolist(),
            "soft_responsibility_component_probabilities": tf.reduce_mean(
                responsibilities, axis=0
            ).numpy().tolist(),
            "all_components_observed": all_components_observed,
            "all_finite": bool(
                tf.reduce_all(tf.math.is_finite(responsibilities)).numpy()
            ),
        }
        if component_count == 2:
            coverage[name]["both_components_observed"] = all_components_observed
    weighted_probabilities = coverage["weighted"]["soft_responsibility_component_probabilities"]
    gates["finite"] = bool(
        coverage["weighted"]["all_finite"] and coverage["reverse_kl"]["all_finite"]
    )
    gates["all_components_observed"] = bool(
        coverage["weighted"]["all_components_observed"]
    )
    if component_count == 2:
        gates["both_components_observed"] = gates["all_components_observed"]
    maximum_error = max(
        abs(float(weighted_probabilities[i]) - float(target["target_probabilities"][i].numpy()))
        for i in range(component_count)
    )
    gates["component_probability_max_error_below_0p05"] = maximum_error < 0.05
    return {
        "candidate_passed": all(gates.values()),
        "gates": gates,
        "threshold_status": "canary_calibration_hypotheses_only",
        "component_probability_maximum_absolute_error": maximum_error,
        "base_component_coverage": coverage,
        "interpretation": (
            "A pass nominates replication and interval calibration; it does not establish "
            "exact target-weight recovery or posterior correctness."
        ),
    }


def _snapshot(trainer: Any) -> Mapping[str, list[Any]]:
    return {
        "transport": [_frozen_value(variable) for variable in trainer.variables],
        "optimizer": [
            _frozen_value(variable) for variable in trainer.optimizer.variables
        ],
    }


def _restore(trainer: Any, values: Mapping[str, list[Any]]) -> None:
    for variable, value in zip(trainer.variables, values["transport"]):
        variable.assign(value)
    for variable, value in zip(trainer.optimizer.variables, values["optimizer"]):
        variable.assign(value)
    trainer.step.assign(trainer.optimizer.iterations.value)


def _frozen_value(variable: Any) -> Any:
    value = getattr(variable, "value", None)
    if callable(value):
        return value()
    if value is not None and hasattr(value, "read_value"):
        return value.read_value()
    if hasattr(variable, "read_value"):
        return variable.read_value()
    raise TypeError(f"cannot snapshot variable of type {type(variable)!r}")


def _step_payload(step: Any) -> Mapping[str, Any]:
    return {
        "loss": float(step.loss.numpy()),
        "effective_sample_size": float(step.effective_sample_size.numpy()),
        "effective_sample_size_fraction": float(
            step.effective_sample_size_fraction.numpy()
        ),
        "maximum_normalized_weight": float(step.maximum_normalized_weight.numpy()),
        "gradient_norm": float(step.gradient_norm.numpy()),
        "clipped_gradient_norm": float(step.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(step.clipping_applied.numpy()),
        "step": int(step.step.numpy()),
    }


def _importance_summary(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: _ready(value)
        for key, value in values.items()
        if key not in {"log_weights", "normalized_weights"}
    }


def _reverse_state_payload(trainer: Any) -> Mapping[str, Any]:
    payload = {
        "schema": "bayesfilter.neutra.matched_reverse_kl_state.v1",
        "config": trainer.config.manifest_payload(),
        "step": int(trainer.step.numpy()),
        "variables": [variable.numpy().tolist() for variable in trainer.variables],
        "optimizer_variables": [value.numpy().tolist() for value in trainer.optimizer.variables],
    }
    return {**payload, "state_hash": _semantic_hash(payload)}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_ready(payload), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    return value


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
