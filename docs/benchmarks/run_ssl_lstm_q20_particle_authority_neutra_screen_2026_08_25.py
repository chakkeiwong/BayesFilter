"""Run the reviewed Phase 4 GPU/XLA NeuTra candidate screen.

This runner consumes a frozen M0 pilot bank as a fixed empirical weighted
cloud.  It performs two small, target-specific weighted forward-KL training
arms, selects one on a disjoint validation partition, and evaluates the frozen
selection on an untouched audit partition.  The terminal M0 weights are
normalized after resampling, so this is candidate/engineering evidence rather
than an unbiased SMC-U or posterior claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
M0_ROOT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-particle-authority-master-2026-08-25/"
    "phase2-attempt2-n100"
)
PLAN = ROOT / (
    "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase4-neutra-"
    "subplan-2026-08-25.md"
)
RUNNER = Path(__file__).resolve()
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-particle-authority-master-2026-08-25/phase4"
)

TRAIN_COUNT = 60
VALIDATION_COUNT = 20
AUDIT_COUNT = 20
AUDIT_PER_MODE = 10
MODE_AXIS = 2
DEFAULT_STEPS = 3
DEFAULT_SEED = (20260825, 4601)


class ScreenError(RuntimeError):
    """Raised when the Phase 4 evidence contract cannot be preserved."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ScreenError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    return {
        "commit": commit,
        "dirty": bool(
            subprocess.check_output(("git", "status", "--short"), cwd=ROOT, text=True).strip()
        ),
    }


def _load_tensor(tf: Any, path: Path, dtype: Any) -> Any:
    if not path.is_file():
        raise ScreenError(f"missing tensor receipt: {path}")
    encoded = path.read_bytes()
    actual = hashlib.sha256(encoded).hexdigest()
    # The pilot receipt is the authority for the expected digest.
    tensor = tf.io.parse_tensor(tf.convert_to_tensor(encoded), out_type=dtype)
    return tensor, actual


def _finite_bool(tf: Any, value: Any) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def _affine_forward(tf: Any, physical: Any, affine: Mapping[str, Any] | None) -> Any:
    if affine is None:
        return physical
    centered = physical - affine["mean"]
    transformed = tf.transpose(
        tf.linalg.triangular_solve(affine["chol"], tf.transpose(centered))
    )
    transformed.set_shape(physical.shape)
    return transformed


def _affine_inverse(tf: Any, whitened: Any, affine: Mapping[str, Any] | None) -> Any:
    if affine is None:
        return whitened
    transformed = affine["mean"] + tf.matmul(whitened, affine["chol"], transpose_b=True)
    transformed.set_shape(whitened.shape)
    return transformed


def _status_valid(tf: Any, status: Mapping[str, Any]) -> bool:
    required = ("status_code", "valid_pre_regularized_score")
    if any(name not in status for name in required):
        return False
    return bool(
        tf.reduce_all(
            tf.logical_and(
                tf.equal(tf.cast(status["status_code"], tf.int32), 0),
                tf.cast(status["valid_pre_regularized_score"], tf.bool),
            )
        ).numpy()
    )


def _config(WeightedNeuTraConfig: Any, *, arm: str, seed: tuple[int, int]) -> Any:
    # These are target-specific screen hypotheses, not promoted defaults.
    if arm == "compact":
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
    if arm == "wide_low_lr":
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
    if arm == "compact_low_lr":
        return WeightedNeuTraConfig(
            dimension=4,
            hidden_layers=(16, 16),
            stages=2,
            activation="tanh",
            initialization_scale=0.02,
            initialization_seed=seed,
            learning_rate=2.0e-4,
            jit_compile=True,
        )
    if arm == "wider_mid_lr":
        return WeightedNeuTraConfig(
            dimension=4,
            hidden_layers=(64, 32),
            stages=2,
            activation="tanh",
            initialization_scale=0.02,
            initialization_seed=seed,
            learning_rate=5.0e-4,
            jit_compile=True,
        )
    if arm == "high_capacity":
        return WeightedNeuTraConfig(
            dimension=4,
            hidden_layers=(128, 64, 32),
            stages=3,
            activation="tanh",
            initialization_scale=0.02,
            initialization_seed=seed,
            learning_rate=1.0e-3,
            jit_compile=True,
        )
    raise ScreenError(f"unknown arm: {arm}")


def _step_payload(step: Any) -> Mapping[str, Any]:
    return {
        "step": step.step,
        "loss": step.loss,
        "effective_sample_size_fraction": step.effective_sample_size_fraction,
        "maximum_normalized_weight": step.maximum_normalized_weight,
        "gradient_norm": step.gradient_norm,
        "clipped_gradient_norm": step.clipped_gradient_norm,
        "clipping_applied": step.clipping_applied,
    }


def _run_arm(
    *,
    tf: Any,
    trainer_type: Any,
    config: Any,
    train_rows: Any,
    train_log_weights: Any,
    validation_rows: Any,
    validation_log_weights: Any,
    full_rows: Any,
    full_log_weights: Any,
    audit_rows: Any,
    target: Any,
    steps: int,
    audit_count: int,
    affine: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    trainer = trainer_type(config)
    trace = []
    for _ in range(int(steps)):
        update = trainer.train_step(train_rows, train_log_weights)
        validation = trainer.validation_batch(validation_rows, validation_log_weights)
        trace.append(
            {
                "training": _step_payload(update),
                "validation_loss": validation.loss,
                "validation_latent_mean_max_abs": tf.reduce_max(
                    tf.abs(validation.latent_weighted_mean)
                ),
                "validation_latent_covariance_max_abs_offdiag": tf.reduce_max(
                    tf.abs(
                        validation.latent_weighted_covariance
                        - tf.linalg.diag(tf.linalg.diag_part(validation.latent_weighted_covariance))
                    )
                ),
            }
        )

    latent_probe = tf.random.stateless_normal(
        (int(audit_count), 4), seed=(20260825, 7701), dtype=tf.float64
    )
    preconditioned_probe, forward_logdet = trainer.transport.forward_and_logdet(latent_probe)
    recovered, inverse_logdet = trainer.transport.inverse_and_forward_logdet(preconditioned_probe)
    physical_probe = _affine_inverse(tf, preconditioned_probe, affine)
    parity = {
        "roundtrip_max_abs": tf.reduce_max(tf.abs(recovered - latent_probe)),
        "logdet_roundtrip_max_abs": tf.reduce_max(tf.abs(inverse_logdet - forward_logdet)),
        "finite": _finite_bool(tf, physical_probe)
        and _finite_bool(tf, forward_logdet)
        and _finite_bool(tf, recovered)
        and _finite_bool(tf, inverse_logdet),
    }

    physical_audit_rows = _affine_inverse(tf, audit_rows, affine)
    audit_target_value, audit_score, audit_status = (
        target.neutra_batch_log_prob_and_grad_status(physical_audit_rows)
    )
    transformed_target_value, transformed_score, transformed_status = (
        target.neutra_batch_log_prob_and_grad_status(physical_probe)
    )
    audit_valid = _status_valid(tf, audit_status)
    transformed_valid = _status_valid(tf, transformed_status)
    audit = {
        "target_value_finite": _finite_bool(tf, audit_target_value),
        "target_score_finite": _finite_bool(tf, audit_score),
        "target_status_valid": audit_valid,
        "transformed_target_value_finite": _finite_bool(tf, transformed_target_value),
        "transformed_target_score_finite": _finite_bool(tf, transformed_score),
        "transformed_target_status_valid": transformed_valid,
        "transformed_target_log_density_finite": _finite_bool(
            tf,
            transformed_target_value
            + forward_logdet
            + (affine["logdet"] if affine is not None else 0.0),
        ),
        "audit_rows": int(audit_rows.shape[0]),
    }
    validation = trainer.validation_batch(validation_rows, validation_log_weights)
    full_normalized_weights = tf.nn.softmax(full_log_weights)
    full_latent, full_forward_logdet = trainer.transport.inverse_and_forward_logdet(full_rows)
    full_dimension = tf.cast(config.dimension, tf.float64)
    full_negative_log_prob = (
        tf.constant(0.5, tf.float64) * tf.reduce_sum(tf.square(full_latent), axis=-1)
        + tf.constant(0.5, tf.float64)
        * full_dimension
        * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
        + full_forward_logdet
    )
    full_mean = tf.reduce_sum(full_normalized_weights[:, tf.newaxis] * full_latent, axis=0)
    full_centered = full_latent - full_mean
    full_covariance = tf.matmul(
        full_centered,
        full_normalized_weights[:, tf.newaxis] * full_centered,
        transpose_a=True,
    )
    full_bank = {
        "loss": tf.reduce_sum(full_normalized_weights * full_negative_log_prob),
        "effective_sample_size_fraction": tf.math.reciprocal(
            tf.cast(tf.size(full_normalized_weights), tf.float64)
            * tf.reduce_sum(tf.square(full_normalized_weights))
        ),
        "maximum_normalized_weight": tf.reduce_max(full_normalized_weights),
        "latent_weighted_mean": full_mean,
        "latent_weighted_covariance": full_covariance,
    }
    state = trainer.state_payload()
    return {
        "status": "PASS_CANDIDATE" if all(
            (
                bool(parity["finite"]),
                float(parity["roundtrip_max_abs"].numpy()) <= 1.0e-9,
                float(parity["logdet_roundtrip_max_abs"].numpy()) <= 1.0e-9,
                audit_valid,
                transformed_valid,
                bool(audit["transformed_target_log_density_finite"]),
                _finite_bool(tf, validation.loss),
            )
        ) else "CANDIDATE_FAIL",
        "config": config.manifest_payload(),
        "precondition": "affine_weighted_moment_oracle" if affine is not None else "identity",
        "training_trace": trace,
        "validation": {
            "loss": validation.loss,
            "effective_sample_size_fraction": validation.effective_sample_size_fraction,
            "maximum_normalized_weight": validation.maximum_normalized_weight,
            "latent_weighted_mean": validation.latent_weighted_mean,
            "latent_weighted_covariance": validation.latent_weighted_covariance,
        },
        "full_bank": {
            "loss": full_bank["loss"],
            "effective_sample_size_fraction": full_bank["effective_sample_size_fraction"],
            "maximum_normalized_weight": full_bank["maximum_normalized_weight"],
            "latent_weighted_mean": full_bank["latent_weighted_mean"],
            "latent_weighted_covariance": full_bank["latent_weighted_covariance"],
        },
        "parity": parity,
        "audit": audit,
        "training_state_hash": state["state_hash"],
        "nonclaims": [
            "terminal normalized M0 weights are a fixed empirical measure, not an unnormalized SMC-U ledger",
            "short one-seed training is not a ranking or posterior-correctness result",
            "no HMC, convergence, predictive, or default-readiness claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--m0-root", type=Path, default=M0_ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--profile", choices=("screen", "tuning", "capacity"), default="screen")
    parser.add_argument("--precondition", choices=("identity", "affine"), default="identity")
    parser.add_argument("--seed", nargs=2, type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise ScreenError("output root must be repository-relative")
    if args.output_root.exists():
        raise ScreenError(f"refusing to overwrite existing output root: {args.output_root}")
    m0_root = args.m0_root if args.m0_root.is_absolute() else ROOT / args.m0_root
    if ".." in m0_root.relative_to(ROOT).parts:
        raise ScreenError("M0 root must remain inside the repository")
    if not m0_root.is_dir():
        raise ScreenError(f"M0 root does not exist: {m0_root}")
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    if ".." in plan_path.relative_to(ROOT).parts:
        raise ScreenError("plan must remain inside the repository")
    if not plan_path.is_file():
        raise ScreenError(f"plan does not exist: {plan_path}")
    if int(args.steps) <= 0:
        raise ScreenError("steps must be positive")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise ScreenError("TF_FORCE_GPU_ALLOW_GROWTH=true is required before TensorFlow import")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise ScreenError("Phase 4 GPU screen cannot run with CUDA_VISIBLE_DEVICES=-1")

    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    launch = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.neutra_screen.launch.v1",
        "status": "STARTED",
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "git": _git_manifest(),
        "plan": plan_path.as_posix(),
        "plan_sha256": _sha256(plan_path),
        "runner_sha256": _sha256(RUNNER),
        "m0_root": m0_root.as_posix(),
        "m0_pilot_sha256": _sha256(m0_root / "pilot.json"),
        "seed": list(args.seed),
        "steps": int(args.steps),
        "precondition": args.precondition,
    }
    _write_json(args.output_root / "launch.json", launch)

    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        # TensorFlow must be imported only after the environment guard; memory
        # growth is configured before querying logical devices or allocating data.
        import tensorflow as tf

        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        gpu_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        physical = tuple(tf.config.list_physical_devices("GPU"))
        logical = tuple(tf.config.list_logical_devices("GPU"))
        if not physical or not logical:
            raise ScreenError("GPU memory policy produced no visible GPU")

        from bayesfilter.inference.neutra_weighted_training import (
            WeightedForwardKLNeuTraTrainer,
            WeightedNeuTraConfig,
        )
        from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
            batch_native_complexity_posterior_target,
        )
        pilot = json.loads((m0_root / "pilot.json").read_text(encoding="utf-8"))
        m0 = pilot["arms"]["M0"]
        if m0.get("status") != "PASS_GATE":
            raise ScreenError("M0 pilot status is not PASS_GATE")
        receipt_map = m0["receipts"]
        paths = {
            name: ROOT / str(receipt_map[name]["path"])
            for name in ("final_theta", "final_normalized_weights", "final_roots", "proposal_components")
        }
        for name, path in paths.items():
            expected = str(receipt_map[name]["sha256"])
            if _sha256(path) != expected:
                raise ScreenError(f"M0 tensor hash mismatch: {name}")
        theta, _ = _load_tensor(tf, paths["final_theta"], tf.float64)
        normalized_weights, _ = _load_tensor(
            tf, paths["final_normalized_weights"], tf.float64
        )
        roots, _ = _load_tensor(tf, paths["final_roots"], tf.int32)
        components, _ = _load_tensor(tf, paths["proposal_components"], tf.int32)
        particle_count = int(theta.shape[0])
        theta.set_shape((particle_count, 4))
        normalized_weights.set_shape((particle_count,))
        if theta.shape[1] != 4 or normalized_weights.shape != (particle_count,):
            raise ScreenError("unexpected M0 tensor shapes")
        if not _finite_bool(tf, theta) or not _finite_bool(tf, normalized_weights):
            raise ScreenError("M0 tensors contain non-finite values")

        affine = None
        if args.precondition == "affine":
            weights_for_moments = normalized_weights / tf.reduce_sum(normalized_weights)
            affine_mean = tf.reduce_sum(theta * weights_for_moments[:, tf.newaxis], axis=0)
            affine_centered = theta - affine_mean
            affine_covariance = tf.einsum(
                "n,ni,nj->ij", weights_for_moments, affine_centered, affine_centered
            )
            affine_chol = tf.linalg.cholesky(affine_covariance)
            if not _finite_bool(tf, affine_chol):
                raise ScreenError("affine preconditioner is non-finite")
            affine = {
                "mean": affine_mean,
                "chol": affine_chol,
                "logdet": tf.reduce_sum(tf.math.log(tf.linalg.diag_part(affine_chol))),
            }

        # The split is deterministic and frozen before any target evaluation.
        # The pilot's retained proposal-component field is all zero after
        # resampling, so the canary uses the explicit signed mode-axis
        # partition. This is a mode/sign diagnostic, not a mode-discovery claim.
        sign_labels = [
            int(value < 0.0) for value in theta[:, MODE_AXIS].numpy().tolist()
        ]
        labels = sorted(set(sign_labels))
        if len(labels) < 2:
            raise ScreenError("M0 proposal metadata has fewer than two components")
        audit_count = particle_count // 5
        validation_count = particle_count // 5
        train_count = particle_count - audit_count - validation_count
        audit_per_mode = audit_count // 2
        if audit_count < 2 or audit_count % 2 or train_count <= 0 or validation_count <= 0:
            raise ScreenError("M0 particle count cannot form the frozen split")
        audit_indices = []
        remaining_indices = []
        selected = set()
        for label in labels[:2]:
            candidates = [
                index for index, value in enumerate(sign_labels) if value == label
            ]
            if len(candidates) < audit_per_mode:
                raise ScreenError("M0 proposal component lacks the audit quota")
            chosen = candidates[:audit_per_mode]
            audit_indices.extend(chosen)
            selected.update(chosen)
        remaining_indices = [
            index for index in range(int(theta.shape[0])) if index not in selected
        ]
        if len(remaining_indices) < train_count + validation_count:
            raise ScreenError("not enough M0 rows remain after audit stratification")
        validation_indices = remaining_indices[:validation_count]
        train_indices = remaining_indices[validation_count : validation_count + train_count]
        train_rows = _affine_forward(tf, tf.gather(theta, train_indices), affine)
        validation_rows = _affine_forward(tf, tf.gather(theta, validation_indices), affine)
        audit_rows = _affine_forward(tf, tf.gather(theta, audit_indices), affine)
        full_rows = _affine_forward(tf, theta, affine)
        full_rows.set_shape((particle_count, 4))
        # Weight rows must follow the exact frozen partition indices.  Positional
        # slicing would silently pair a selected theta with another row's weight.
        train_weights = tf.maximum(
            tf.gather(normalized_weights, train_indices),
            tf.constant(1.0e-300, tf.float64),
        )
        validation_weights = tf.maximum(
            tf.gather(normalized_weights, validation_indices),
            tf.constant(1.0e-300, tf.float64),
        )
        audit_components = tf.gather(
            tf.cast(tf.convert_to_tensor(sign_labels), tf.int32), audit_indices
        )
        if int(tf.size(tf.unique(audit_components).y).numpy()) < 2:
            raise ScreenError("untouched audit partition does not contain both mode labels")
        train_log_weights = tf.math.log(train_weights)
        validation_log_weights = tf.math.log(validation_weights)
        full_log_weights = tf.math.log(
            tf.maximum(normalized_weights, tf.constant(1.0e-300, tf.float64))
        )
        target = batch_native_complexity_posterior_target(
            20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
        )

        arms = {}
        arm_specs = (
            (("compact", 0), ("wide_low_lr", 100))
            if args.profile == "screen"
            else (
                (("compact", 0), ("compact_low_lr", 100), ("wider_mid_lr", 200))
                if args.profile == "tuning"
                else (("compact", 0), ("high_capacity", 300))
            )
        )
        for index, (name, seed_offset) in enumerate(arm_specs):
            config = _config(
                WeightedNeuTraConfig,
                arm=name,
                seed=(int(args.seed[0]), int(args.seed[1]) + seed_offset),
            )
            arms[name] = _run_arm(
                tf=tf,
                trainer_type=WeightedForwardKLNeuTraTrainer,
                config=config,
                train_rows=train_rows,
                train_log_weights=train_log_weights,
                validation_rows=validation_rows,
                validation_log_weights=validation_log_weights,
                full_rows=full_rows,
                full_log_weights=full_log_weights,
                audit_rows=audit_rows,
                target=target,
                steps=int(args.steps),
                audit_count=audit_count,
                affine=affine,
            )

        viable = [name for name, payload in arms.items() if payload["status"] == "PASS_CANDIDATE"]
        if not viable:
            selected = None
            status = "CANDIDATE_FAIL"
        else:
            selected = min(
                viable,
                key=lambda name: float(arms[name]["validation"]["loss"].numpy()),
            )
            status = "PASS_CANDIDATE_ROLE_LIMITED"
        result = {
            "schema": "bayesfilter.ssl_lstm.q20.particle_authority.neutra_screen.v1",
            "status": status,
            "role": "gpu_xla_batch_native_candidate_screen",
            "profile": args.profile,
            "precondition": args.precondition,
            "selected_arm": selected,
            "arms": arms,
            "split": {
                "train": train_count,
                "validation": validation_count,
                "audit": audit_count,
                "mode_axis": MODE_AXIS,
                "selection_frozen_before_audit": True,
            },
            "m0_protocol_hash": m0["configuration"]["protocol_hash"],
            "m0_target_signature": m0["target_signature"],
            "device": {
                "gpu_memory_policy": gpu_policy,
                "physical_devices": [device.name for device in physical],
                "logical_devices": [device.name for device in logical],
                "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
                "global_jit_setting": str(tf.config.optimizer.get_jit()),
                "jit_compile_per_function": True,
            },
            "dtype": "float64",
            "batch_size": train_count,
            "hmc_launched": False,
            "run_manifest": {
                **launch,
                "tensorflow": tf.__version__,
                "physical_gpus": [device.name for device in physical],
                "logical_gpus": [device.name for device in logical],
                "gpu_memory_growth_verified": True,
                "wall_seconds": time.perf_counter() - started,
            },
            "nonclaims": [
                "normalized terminal M0 weights are a fixed empirical training measure, not an unnormalized SMC-U ledger",
                "two short arms and one seed cannot establish ranking or superiority",
                "no IID whitening theorem, posterior correctness, mode-discovery guarantee, HMC convergence, or default promotion",
            ],
        }
        _write_json(args.output_root / "result.json", result)
        (args.output_root / "result.md").write_text(
            "# Phase 4 NeuTra Screen Result\n\n"
            f"Status: `{status}`\n\n"
            f"Selected arm: `{selected}`\n\n"
            "This is GPU/XLA batch-native candidate evidence only. The terminal M0 weights were normalized after resampling; no posterior or unbiased SMC-U claim is made.\n",
            encoding="ascii",
        )
        print(json.dumps({"status": status, "output_root": args.output_root.as_posix()}, sort_keys=True))
        return 0 if status != "CANDIDATE_FAIL" else 2
    except Exception as exc:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20.particle_authority.neutra_screen.failure.v1",
            "status": "PHASE4_ATTEMPT_FAILED_REPAIR_TRIGGER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "run_manifest": {
                **launch,
                "wall_seconds": time.perf_counter() - started,
                "failure_artifact_preserved": True,
            },
            "nonclaims": ["failure is not evidence against NeuTra or the particle-authority direction"],
        }
        _write_json(args.output_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
