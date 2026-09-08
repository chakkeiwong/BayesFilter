"""Run a bounded GPU/XLA batch-native NeuTra screen on fresh theta rows.

The screen is deliberately downstream of the corrected parameter authority.
It trains a weighted transport on theta in R^4 and evaluates the q=20 target
only on an untouched theta audit partition.  It does not launch HMC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    raise RuntimeError("Phase 31 requires a visible trusted GPU")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 31 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
PHYSICAL_GPUS = tuple(tf.config.list_physical_devices("GPU"))
LOGICAL_GPUS = tuple(tf.config.list_logical_devices("GPU"))
if not PHYSICAL_GPUS or not LOGICAL_GPUS:
    raise RuntimeError("Phase 31 GPU memory policy produced no logical GPU")
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except (AttributeError, RuntimeError):
    pass

from bayesfilter.inference.neutra_weighted_training import (
    WeightedForwardKLNeuTraTrainer,
    WeightedNeuTraConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
TRAINING_MODULE = ROOT / "bayesfilter/inference/neutra_weighted_training.py"
TARGET_MODULE = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
GPU_MODULE = ROOT / "bayesfilter/runtime/gpu_memory_policy.py"


class Phase31Error(RuntimeError):
    """Raised when the GPU/XLA boundary cannot be audited."""


LEGACY_SPLIT_POLICY = "legacy_ordered_v2"
STRATIFIED_SPLIT_POLICY = "stratified_hash_v1"
ROOT_GROUP_SPLIT_POLICY = "root_group_stratified_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(item) for item in value.as_list()]
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase31Error(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise Phase31Error(f"refusing to overwrite output: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    path.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase31Error(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _status_valid(status: Mapping[str, Any]) -> tf.Tensor:
    return tf.logical_and(
        tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
        tf.cast(status["valid_pre_regularized_score"], tf.bool),
    )


def _split_indices(
    theta: tf.Tensor,
    *,
    policy: str,
    roots: tf.Tensor | None = None,
) -> tuple[list[int], list[int], list[int], Mapping[str, Any]]:
    """Construct a frozen sign-balanced split without changing particle weights."""
    if policy not in {
        LEGACY_SPLIT_POLICY,
        STRATIFIED_SPLIT_POLICY,
        ROOT_GROUP_SPLIT_POLICY,
    }:
        raise Phase31Error(f"unknown split policy: {policy}")
    sign = theta[:, 2] < 0.0
    negative = [int(index) for index in tf.reshape(tf.where(sign), [-1]).numpy().tolist()]
    positive = [int(index) for index in tf.reshape(tf.where(tf.logical_not(sign)), [-1]).numpy().tolist()]
    if len(negative) < 12 or len(positive) < 12:
        raise Phase31Error("fresh bank cannot form two sign-balanced audit/validation partitions")
    if policy == LEGACY_SPLIT_POLICY:
        audit_indices = negative[:6] + positive[:6]
        selected = set(audit_indices)
        remaining = [index for index in range(int(theta.shape[0])) if index not in selected]
        validation_count = min(12, len(remaining) // 3)
        validation_indices = remaining[:validation_count]
        train_indices = remaining[validation_count:]
        construction = "audit=negative[:6]+positive[:6]; validation=remaining[:12]; train=remaining[12:]"
    elif policy == STRATIFIED_SPLIT_POLICY:
        # A fixed hash permutation breaks storage-order/ancestry correlation while
        # preserving equal negative/positive counts in validation and audit.
        def keyed(indices: list[int]) -> list[int]:
            return sorted(
                indices,
                key=lambda index: hashlib.sha256(
                    f"{STRATIFIED_SPLIT_POLICY}:{index}".encode("ascii")
                ).digest(),
            )

        negative = keyed(negative)
        positive = keyed(positive)
        audit_indices = negative[:6] + positive[:6]
        validation_indices = negative[6:12] + positive[6:12]
        train_indices = negative[12:] + positive[12:]
        construction = (
            "within-sign SHA256 index permutation with fixed salt; "
            "audit=6+6; validation=6+6; train=remaining"
        )
    else:
        if roots is None or roots.shape.rank != 1 or int(roots.shape[0]) != int(theta.shape[0]):
            raise Phase31Error("root-group split requires one root id per theta row")
        root_values = [int(value) for value in roots.numpy().tolist()]
        groups: dict[int, list[int]] = {}
        for index, root in enumerate(root_values):
            groups.setdefault(root, []).append(index)
        groups_by_sign: dict[bool, list[tuple[int, list[int]]]] = {False: [], True: []}
        for root, indices in groups.items():
            group_signs = {bool(sign.numpy()[index]) for index in indices}
            if len(group_signs) != 1:
                raise Phase31Error(f"root {root} crosses the declared sign boundary")
            groups_by_sign[next(iter(group_signs))].append((root, indices))

        def choose_group_subset(
            candidates: list[tuple[int, list[int]]], target: int
        ) -> tuple[list[tuple[int, list[int]]], list[tuple[int, list[int]]]]:
            ordered = sorted(
                candidates,
                key=lambda item: hashlib.sha256(
                    f"{ROOT_GROUP_SPLIT_POLICY}:{item[0]}".encode("ascii")
                ).digest(),
            )
            # Deterministic bounded subset-sum. The target is six rows per sign;
            # whole-root allocation prevents ancestry leakage across partitions.
            states: dict[int, tuple[int, ...]] = {0: ()}
            for position, (_, indices) in enumerate(ordered):
                size = len(indices)
                next_states = dict(states)
                for total, selected_positions in states.items():
                    candidate_total = total + size
                    candidate = selected_positions + (position,)
                    prior = next_states.get(candidate_total)
                    if prior is None or candidate < prior:
                        next_states[candidate_total] = candidate
                states = next_states
            best_total = min(states, key=lambda total: (abs(total - target), total))
            selected_positions = set(states[best_total])
            selected = [item for position, item in enumerate(ordered) if position in selected_positions]
            remainder = [item for position, item in enumerate(ordered) if position not in selected_positions]
            return selected, remainder

        audit_groups: list[tuple[int, list[int]]] = []
        validation_groups: list[tuple[int, list[int]]] = []
        train_groups: list[tuple[int, list[int]]] = []
        group_counts: dict[str, Any] = {}
        for sign_value in (False, True):
            audit_selected, remaining_groups = choose_group_subset(groups_by_sign[sign_value], 6)
            validation_selected, train_remaining = choose_group_subset(remaining_groups, 6)
            audit_groups.extend(audit_selected)
            validation_groups.extend(validation_selected)
            train_groups.extend(train_remaining)
            group_counts["negative_axis2" if sign_value else "positive_axis2"] = {
                "audit_rows": sum(len(indices) for _, indices in audit_selected),
                "validation_rows": sum(len(indices) for _, indices in validation_selected),
                "train_rows": sum(len(indices) for _, indices in train_remaining),
                "audit_roots": len(audit_selected),
                "validation_roots": len(validation_selected),
                "train_roots": len(train_remaining),
            }
        audit_indices = [index for _, indices in audit_groups for index in indices]
        validation_indices = [index for _, indices in validation_groups for index in indices]
        train_indices = [index for _, indices in train_groups for index in indices]
        construction = (
            "whole-root deterministic SHA256 group allocation; subset-sum target "
            "six rows per sign for audit and validation; no root overlaps"
        )
    if len(train_indices) <= 1 or len(validation_indices) <= 1:
        raise Phase31Error("frozen split leaves no batch-sized training/validation set")
    split = {
        "policy": policy,
        "construction": construction,
        "train": len(train_indices),
        "validation": len(validation_indices),
        "audit": len(audit_indices),
        "audit_sign_counts": {
            "negative_axis2": sum(index in negative for index in audit_indices),
            "positive_axis2": sum(index in positive for index in audit_indices),
        },
        "validation_sign_counts": {
            "negative_axis2": sum(index in negative for index in validation_indices),
            "positive_axis2": sum(index in positive for index in validation_indices),
        },
        "root_group_counts": group_counts if policy == ROOT_GROUP_SPLIT_POLICY else None,
        "root_disjoint": None,
        "indices": {
            "train": train_indices,
            "validation": validation_indices,
            "audit": audit_indices,
        },
    }
    if roots is not None:
        root_values = [int(value) for value in roots.numpy().tolist()]
        partition_sets = {
            name: set(root_values[index] for index in indices)
            for name, indices in (
                ("train", train_indices),
                ("validation", validation_indices),
                ("audit", audit_indices),
            )
        }
        split["root_disjoint"] = all(
            partition_sets[left].isdisjoint(partition_sets[right])
            for left, right in (
                ("train", "validation"),
                ("train", "audit"),
                ("validation", "audit"),
            )
        )
    row_sets = {
        tuple(train_indices),
        tuple(validation_indices),
        tuple(audit_indices),
    }
    all_rows = set(train_indices) | set(validation_indices) | set(audit_indices)
    if len(all_rows) != int(theta.shape[0]) or sum(len(indices) for indices in (train_indices, validation_indices, audit_indices)) != len(all_rows):
        raise Phase31Error("split rows do not form a disjoint complete partition")
    split["row_partition_disjoint"] = len(row_sets) == 3
    split["row_partition_complete"] = len(all_rows) == int(theta.shape[0])
    return train_indices, validation_indices, audit_indices, split


def _affine_forward(physical: tf.Tensor, affine: Mapping[str, tf.Tensor] | None) -> tf.Tensor:
    if affine is None:
        return physical
    centered = physical - affine["mean"][tf.newaxis, :]
    transformed = tf.transpose(
        tf.linalg.triangular_solve(affine["chol"], tf.transpose(centered), lower=True)
    )
    return tf.ensure_shape(transformed, physical.shape)


def _affine_inverse(chart: tf.Tensor, affine: Mapping[str, tf.Tensor] | None) -> tf.Tensor:
    if affine is None:
        return chart
    physical = affine["mean"][tf.newaxis, :] + tf.matmul(
        chart, affine["chol"], transpose_b=True
    )
    return tf.ensure_shape(physical, chart.shape)


def _config(name: str, seed: tuple[int, int]) -> WeightedNeuTraConfig:
    if name == "compact":
        return WeightedNeuTraConfig(
            dimension=4,
            hidden_layers=(16, 16),
            stages=2,
            activation="tanh",
            initialization_scale=0.02,
            initialization_seed=seed,
            learning_rate=1.0e-3,
            jit_compile=True,
        )
    if name == "wide_low_lr":
        return WeightedNeuTraConfig(
            dimension=4,
            hidden_layers=(32, 32),
            stages=2,
            activation="tanh",
            initialization_scale=0.02,
            initialization_seed=seed,
            learning_rate=5.0e-4,
            jit_compile=True,
        )
    raise Phase31Error(f"unknown arm {name}")


def _step_payload(step: Any) -> Mapping[str, Any]:
    return {
        "loss": step.loss,
        "effective_sample_size_fraction": step.effective_sample_size_fraction,
        "maximum_normalized_weight": step.maximum_normalized_weight,
        "gradient_norm": step.gradient_norm,
        "clipped_gradient_norm": step.clipped_gradient_norm,
        "clipping_applied": step.clipping_applied,
        "step": step.step,
        "loss_device": str(step.loss.device),
        "gradient_device": str(step.gradient_norm.device),
    }


def _transport_moment_diagnostic(
    trainer: WeightedForwardKLNeuTraTrainer,
    rows: tf.Tensor,
    log_weights: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """Compute moments without retracing the static-shape validation graph."""
    normalized_weights = tf.exp(tf.nn.log_softmax(log_weights))
    latent, forward_logdet = trainer.transport.inverse_and_forward_logdet(rows)
    dimension = tf.cast(trainer.config.dimension, tf.float64)
    negative_log_prob = (
        tf.constant(0.5, tf.float64) * tf.reduce_sum(tf.square(latent), axis=-1)
        + tf.constant(0.5, tf.float64)
        * dimension
        * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))
        + forward_logdet
    )
    mean = tf.reduce_sum(normalized_weights[:, tf.newaxis] * latent, axis=0)
    centered = latent - mean
    covariance = tf.matmul(
        centered,
        normalized_weights[:, tf.newaxis] * centered,
        transpose_a=True,
    )
    return {
        "loss": tf.reduce_sum(normalized_weights * negative_log_prob),
        "effective_sample_size_fraction": tf.math.reciprocal(
            tf.cast(tf.size(normalized_weights), tf.float64)
            * tf.reduce_sum(tf.square(normalized_weights))
        ),
        "maximum_normalized_weight": tf.reduce_max(normalized_weights),
        "latent_weighted_mean": mean,
        "latent_weighted_covariance": covariance,
    }


def _run_arm(
    *,
    name: str,
    seed: tuple[int, int],
    train_rows: tf.Tensor,
    train_log_weights: tf.Tensor,
    validation_rows: tf.Tensor,
    validation_log_weights: tf.Tensor,
    audit_rows: tf.Tensor,
    target: Any,
    steps: int,
    affine: Mapping[str, tf.Tensor] | None,
    affine_round_trip_residual: tf.Tensor,
    affine_training_oracle_gate: bool,
    audit_log_weights: tf.Tensor | None = None,
    checkpoint_steps: tuple[int, ...] = (),
) -> Mapping[str, Any]:
    config = _config(name, seed)
    if int(train_rows.shape[0]) <= 1:
        raise Phase31Error("NeuTra optimizer batch must contain more than one row")
    with tf.device("/GPU:0"):
        trainer = WeightedForwardKLNeuTraTrainer(config)
        trace = []
        for _ in range(int(steps)):
            update = trainer.train_step(train_rows, train_log_weights)
            train_diagnostic = _transport_moment_diagnostic(
                trainer, train_rows, train_log_weights
            )
            validation = trainer.validation_batch(validation_rows, validation_log_weights)
            step_number = len(trace) + 1
            trace_item: dict[str, Any] = {
                "training": _step_payload(update),
                "training_latent_mean_max_abs": tf.reduce_max(
                    tf.abs(train_diagnostic["latent_weighted_mean"])
                ),
                "training_latent_covariance_max_abs_offdiag": tf.reduce_max(
                    tf.abs(
                        train_diagnostic["latent_weighted_covariance"]
                        - tf.linalg.diag(
                            tf.linalg.diag_part(train_diagnostic["latent_weighted_covariance"])
                        )
                    )
                ),
                "validation_loss": validation.loss,
                "validation_latent_mean_max_abs": tf.reduce_max(tf.abs(validation.latent_weighted_mean)),
                "validation_latent_covariance_max_abs_offdiag": tf.reduce_max(
                    tf.abs(
                        validation.latent_weighted_covariance
                        - tf.linalg.diag(tf.linalg.diag_part(validation.latent_weighted_covariance))
                    )
                ),
            }
            if audit_log_weights is not None and step_number in checkpoint_steps:
                audit_checkpoint = trainer.validation_batch(audit_rows, audit_log_weights)
                trace_item["audit_checkpoint"] = {
                    "loss": audit_checkpoint.loss,
                    "effective_sample_size_fraction": audit_checkpoint.effective_sample_size_fraction,
                    "maximum_normalized_weight": audit_checkpoint.maximum_normalized_weight,
                    "latent_weighted_mean": audit_checkpoint.latent_weighted_mean,
                    "latent_weighted_covariance": audit_checkpoint.latent_weighted_covariance,
                    "latent_mean_max_abs": tf.reduce_max(tf.abs(audit_checkpoint.latent_weighted_mean)),
                    "latent_covariance_max_abs_offdiag": tf.reduce_max(
                        tf.abs(
                            audit_checkpoint.latent_weighted_covariance
                            - tf.linalg.diag(
                                tf.linalg.diag_part(audit_checkpoint.latent_weighted_covariance)
                            )
                        )
                    ),
                }
            trace.append(trace_item)
        probe = tf.random.stateless_normal((12, 4), seed=(20260825, 7311), dtype=tf.float64)
        chart_physical, forward_logdet = trainer.transport.forward_and_logdet(probe)
        recovered, inverse_logdet = trainer.transport.inverse_and_forward_logdet(chart_physical)
        physical = _affine_inverse(chart_physical, affine)
        physical_audit_rows = _affine_inverse(audit_rows, affine)
        audit_value, audit_score, audit_status = target.neutra_batch_log_prob_and_grad_status(physical_audit_rows)
        transformed_value, transformed_score, transformed_status = target.neutra_batch_log_prob_and_grad_status(physical)
        train_diagnostic = _transport_moment_diagnostic(
            trainer, train_rows, train_log_weights
        )
        validation = trainer.validation_batch(validation_rows, validation_log_weights)
    audit_valid = _status_valid(audit_status)
    transformed_valid = _status_valid(transformed_status)
    parity = {
        "roundtrip_max_abs": tf.reduce_max(tf.abs(recovered - probe)),
        "logdet_roundtrip_max_abs": tf.reduce_max(tf.abs(inverse_logdet - forward_logdet)),
        "finite": tf.reduce_all(
            tf.stack(
                (
                    tf.reduce_all(tf.math.is_finite(physical)),
                    tf.reduce_all(tf.math.is_finite(forward_logdet)),
                    tf.reduce_all(tf.math.is_finite(recovered)),
                    tf.reduce_all(tf.math.is_finite(inverse_logdet)),
                )
            )
        ),
        "composed_log_q_theta_finite": tf.reduce_all(
            tf.math.is_finite(
                -0.5 * tf.reduce_sum(tf.square(probe), axis=1)
                - 2.0 * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))
                - forward_logdet
                - (affine["logdet"] if affine is not None else 0.0)
            )
        ),
    }
    gates = {
        "batch_size_gt_one": int(train_rows.shape[0]) > 1,
        "batch_shape_N_by_4": train_rows.shape[1] == 4,
        "affine_round_trip": affine is None or float(affine_round_trip_residual.numpy()) <= 1.0e-10,
        "affine_training_measure_oracle": affine is None or bool(affine_training_oracle_gate),
        "xla_configured": bool(config.jit_compile),
        "training_trace_nonempty": len(trace) > 0,
        "finite_training_and_validation": bool(
            all(
                bool(tf.reduce_all(tf.math.is_finite(item["training"][key])).numpy())
                for item in trace
                for key in ("loss", "gradient_norm", "clipped_gradient_norm")
            )
            and all(
                bool(tf.reduce_all(tf.math.is_finite(item["validation_loss"])).numpy())
                for item in trace
            )
        ),
        "transport_parity_finite": bool(parity["finite"].numpy()),
        "composed_log_q_theta_finite": bool(parity["composed_log_q_theta_finite"].numpy()),
        "transport_roundtrip": float(parity["roundtrip_max_abs"].numpy()) <= 1.0e-8,
        "transport_logdet_roundtrip": float(parity["logdet_roundtrip_max_abs"].numpy()) <= 1.0e-8,
        "audit_target_finite": bool(tf.reduce_all(tf.math.is_finite(audit_value)).numpy()),
        "audit_score_finite": bool(tf.reduce_all(tf.math.is_finite(audit_score)).numpy()),
        "audit_target_status_valid": bool(tf.reduce_all(audit_valid).numpy()),
        "transformed_target_finite": bool(tf.reduce_all(tf.math.is_finite(transformed_value)).numpy()),
        "transformed_score_finite": bool(tf.reduce_all(tf.math.is_finite(transformed_score)).numpy()),
        "transformed_target_status_valid": bool(tf.reduce_all(transformed_valid).numpy()),
    }
    return {
        "status": "PASS_NEUTRA_BOUNDARY_CANDIDATE" if all(gates.values()) else "NEUTRA_CANDIDATE_FAIL_REPAIR",
        "config": config.manifest_payload(),
        "precondition": "affine_weighted_theta_moments" if affine is not None else "identity",
        "gates": gates,
        "training_trace": trace,
        "training_measure": {
            "loss": train_diagnostic["loss"],
            "effective_sample_size_fraction": train_diagnostic["effective_sample_size_fraction"],
            "maximum_normalized_weight": train_diagnostic["maximum_normalized_weight"],
            "latent_weighted_mean": train_diagnostic["latent_weighted_mean"],
            "latent_weighted_covariance": train_diagnostic["latent_weighted_covariance"],
        },
        "validation": {
            "loss": validation.loss,
            "effective_sample_size_fraction": validation.effective_sample_size_fraction,
            "maximum_normalized_weight": validation.maximum_normalized_weight,
            "latent_weighted_mean": validation.latent_weighted_mean,
            "latent_weighted_covariance": validation.latent_weighted_covariance,
        },
        "parity": parity,
        "audit": {
            "row_count": int(audit_rows.shape[0]),
            "status_code": audit_status["status_code"],
            "valid": audit_valid,
            "transformed_status_code": transformed_status["status_code"],
            "transformed_valid": transformed_valid,
        },
        "state_hash": trainer.state_payload()["state_hash"],
        "nonclaims": [
            "The fixed normalized M0 bank is not an unnormalized SMC-U ledger.",
            "Short one-seed training and latent moments are descriptive, not whitening or posterior proofs.",
            "No HMC, convergence, predictive, mode-discovery, or default-readiness claim.",
        ],
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Corrected q=20 GPU/XLA NeuTra Boundary",
        "",
        f"Status: `{result['status']}`",
        "",
        "The screen trains only on theta in R^4 rows with a batch-native weighted transport. It does not launch HMC.",
        "",
        "| Arm | Status | Batch | Audit status |",
        "|---|---|---:|---|",
    ]
    for name, arm in result["arms"].items():
        lines.append(
            f"| {name} | `{arm['status']}` | `{result['split']['train']}` | `{all(arm['gates'].get(key, False) for key in ('audit_target_status_valid','transformed_target_status_valid'))}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is GPU/XLA batch-native candidate evidence. Loss and latent covariance remain explanatory; no whitening or posterior claim is made.",
            "",
            "## Nonclaims",
            "",
            "- no SMC-U authority or posterior correctness claim",
            "- no IID Gaussian whitening theorem",
            "- no HMC or default promotion",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", nargs=2, type=int, default=(20260825, 3101))
    parser.add_argument("--precondition", choices=("identity", "affine"), default="identity")
    parser.add_argument(
        "--split-policy",
        choices=(LEGACY_SPLIT_POLICY, STRATIFIED_SPLIT_POLICY, ROOT_GROUP_SPLIT_POLICY),
        default=LEGACY_SPLIT_POLICY,
        help="frozen train/validation/audit partition policy; legacy is retained for historical receipts",
    )
    parser.add_argument(
        "--checkpoint-steps",
        nargs="*",
        type=int,
        default=(),
        help="optional training steps at which to record untouched-audit moments",
    )
    args = parser.parse_args()
    for path in (args.authority_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase31Error("paths must be repository-relative")
    if int(args.steps) <= 0:
        raise Phase31Error("steps must be positive")
    checkpoint_steps = tuple(sorted(set(int(step) for step in args.checkpoint_steps)))
    if any(step <= 0 or step > int(args.steps) for step in checkpoint_steps):
        raise Phase31Error("checkpoint steps must lie in [1, steps]")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase31Error(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    authority = ROOT / args.authority_root
    pilot_path = authority / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase31Error("Phase 28 pilot is not a passing theta-measure receipt")
    m0 = pilot["arms"]["M0"]
    if m0.get("protocol", {}).get("measure") != "theta_R4":
        raise Phase31Error("M0 measure binding is not theta_R4")
    theta = _load_tensor(m0["receipts"]["final_theta"])
    weights = _load_tensor(m0["receipts"]["final_normalized_weights"])
    roots = _load_tensor(m0["receipts"]["final_roots"])
    if theta.shape.rank != 2 or theta.shape[1] != 4:
        raise Phase31Error(f"theta shape is not [N,4]: {theta.shape}")
    particle_count = int(theta.shape[0])
    if weights.shape != (particle_count,):
        raise Phase31Error("weight shape mismatch")
    weights = weights / tf.reduce_sum(weights)
    train_indices, validation_indices, audit_indices, split = _split_indices(
        theta,
        policy=args.split_policy,
        roots=roots,
    )
    train_idx = tf.constant(train_indices, tf.int32)
    validation_idx = tf.constant(validation_indices, tf.int32)
    audit_idx = tf.constant(audit_indices, tf.int32)
    train_weights = tf.maximum(tf.gather(weights, train_idx), tf.constant(1.0e-300, tf.float64))
    validation_weights = tf.maximum(tf.gather(weights, validation_idx), tf.constant(1.0e-300, tf.float64))
    audit_weights = tf.maximum(tf.gather(weights, audit_idx), tf.constant(1.0e-300, tf.float64))
    # The affine chart is a conditioning map for the exact empirical measure
    # consumed by the optimizer, not for the held-out or audit rows.
    train_measure_weights = train_weights / tf.reduce_sum(train_weights)
    affine = None
    if args.precondition == "affine":
        train_theta = tf.gather(theta, train_idx)
        affine_mean = tf.reduce_sum(train_measure_weights[:, tf.newaxis] * train_theta, axis=0)
        centered = train_theta - affine_mean[tf.newaxis, :]
        covariance = tf.einsum(
            "n,ni,nj->ij", train_measure_weights, centered, centered
        )
        covariance = 0.5 * (covariance + tf.transpose(covariance))
        eigenvalues = tf.linalg.eigvalsh(covariance)
        tf.debugging.assert_positive(eigenvalues, "affine training-measure covariance eigenvalues")
        chol = tf.linalg.cholesky(covariance)
        affine = {
            "mean": affine_mean,
            "covariance": covariance,
            "eigenvalues": eigenvalues,
            "chol": chol,
            "logdet": tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol))),
            "condition_estimate": tf.reduce_max(eigenvalues) / tf.reduce_min(eigenvalues),
            "moment_measure": "training_split_normalized_floored_weights",
        }
    train_rows = _affine_forward(tf.gather(theta, train_idx), affine)
    validation_rows = _affine_forward(tf.gather(theta, validation_idx), affine)
    audit_rows = _affine_forward(tf.gather(theta, audit_idx), affine)
    affine_round_trip_residual = tf.constant(0.0, tf.float64)
    affine_training_oracle = None
    if affine is not None:
        affine_round_trip_residual = tf.reduce_max(
            tf.abs(_affine_inverse(_affine_forward(theta, affine), affine) - theta)
        )
        oracle_mean = tf.reduce_sum(
            train_measure_weights[:, tf.newaxis] * train_rows, axis=0
        )
        oracle_centered = train_rows - oracle_mean[tf.newaxis, :]
        oracle_covariance = tf.einsum(
            "n,ni,nj->ij", train_measure_weights, oracle_centered, oracle_centered
        )
        affine_training_oracle = {
            "weighted_mean": oracle_mean,
            "weighted_covariance": oracle_covariance,
            "max_abs_mean": tf.reduce_max(tf.abs(oracle_mean)),
            "max_abs_covariance_residual": tf.reduce_max(
                tf.abs(oracle_covariance - tf.eye(4, dtype=tf.float64))
            ),
        }
    train_log_weights = tf.math.log(train_weights)
    validation_log_weights = tf.math.log(validation_weights)
    audit_log_weights = tf.math.log(audit_weights)
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    arms: dict[str, Any] = {}
    for offset, name in enumerate(("compact", "wide_low_lr")):
        arms[name] = _run_arm(
            name=name,
            seed=(int(args.seed[0]), int(args.seed[1]) + offset),
            train_rows=train_rows,
            train_log_weights=train_log_weights,
            validation_rows=validation_rows,
            validation_log_weights=validation_log_weights,
            audit_rows=audit_rows,
            target=target,
            steps=int(args.steps),
            affine=affine,
            affine_round_trip_residual=affine_round_trip_residual,
            affine_training_oracle_gate=(
                affine is None
                or (
                    affine_training_oracle is not None
                    and float(affine_training_oracle["max_abs_mean"].numpy()) <= 1.0e-10
                    and float(affine_training_oracle["max_abs_covariance_residual"].numpy()) <= 1.0e-10
                )
            ),
            audit_log_weights=audit_log_weights if checkpoint_steps else None,
            checkpoint_steps=checkpoint_steps,
        )
    candidate_pass = all(arm["status"] == "PASS_NEUTRA_BOUNDARY_CANDIDATE" for arm in arms.values())
    active_v2 = args.split_policy == ROOT_GROUP_SPLIT_POLICY
    active_v2_hash = args.split_policy == STRATIFIED_SPLIT_POLICY
    result = {
        "schema": (
            "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v3_root_group_stratified_split"
            if active_v2
            else "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v3_stratified_split"
            if active_v2_hash
            else "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v2_training_measure_bound"
        ),
        "plan_version": (
            "v2.2-root-group-stratified"
            if active_v2
            else "v2.1-stratified-split"
            if active_v2_hash
            else "v2.1-training-measure-bound"
        ),
        "status": "PASS_NEUTRA_BOUNDARY_ROLE_LIMITED" if candidate_pass else "NEUTRA_BOUNDARY_CANDIDATE_FAIL",
        "role": (
            "gpu_xla_batch_native_theta_transport_candidate_screen_training_measure_bound_root_group_stratified"
            if active_v2
            else "gpu_xla_batch_native_theta_transport_candidate_screen_training_measure_bound_stratified_split"
            if active_v2_hash
            else "gpu_xla_batch_native_theta_transport_candidate_screen_training_measure_bound"
        ),
        "precondition": args.precondition,
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(pilot_path),
            "m0_protocol_hash": m0["configuration"]["protocol_hash"],
            "target_signature": m0["target_signature"],
            "measure": "theta_R4",
            "parameter_dim": 4,
        },
        "arms": arms,
        "split": {**split, "selection_frozen_before_audit": True},
        "device": {
            "gpu_memory_policy": GPU_POLICY,
            "physical_devices": [device.name for device in PHYSICAL_GPUS],
            "logical_devices": [device.name for device in LOGICAL_GPUS],
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "global_jit_setting": str(tf.config.optimizer.get_jit()),
            "jit_compile_per_function": True,
        },
        "dtype": "float64",
        "batch_size": len(train_indices),
        "checkpoint_steps": list(checkpoint_steps),
        "training_measure": "normalized_floored_theta_weights_on_train_split",
        "affine_round_trip_residual": affine_round_trip_residual,
        "affine_round_trip_gate": bool(
            affine is None or float(affine_round_trip_residual.numpy()) <= 1.0e-10
        ),
        "affine": (
            {
                "mean": affine["mean"],
                "covariance": affine["covariance"],
                "eigenvalues": affine["eigenvalues"],
                "logdet": affine["logdet"],
                "condition_estimate": affine["condition_estimate"],
                "density_composition": "log_q_theta=log_q_chart-log_abs_det_chol",
                "moment_measure": affine["moment_measure"],
            }
            if affine is not None
            else None
        ),
        "affine_training_oracle": affine_training_oracle,
        "hmc_launched": False,
        "run_manifest": {
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "physical_gpus": [device.name for device in PHYSICAL_GPUS],
            "logical_gpus": [device.name for device in LOGICAL_GPUS],
            "gpu_memory_growth_verified": True,
            "jit_compile": True,
            "seed": list(args.seed),
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "training_module": _sha(TRAINING_MODULE),
                "target_module": _sha(TARGET_MODULE),
                "gpu_module": _sha(GPU_MODULE),
                "authority_pilot": _sha(pilot_path),
            },
        },
        "nonclaims": [
            "The normalized M0 bank is a fixed empirical training measure, not an unnormalized SMC-U ledger.",
            "Short one-seed loss and latent moments do not establish IID whitening or posterior correctness.",
            "No HMC, convergence, predictive, mode-discovery, or default-readiness claim.",
        ],
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
