#!/usr/bin/env python3
"""Run the bounded q=20 K=4 joint-mixture fresh-root replication."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4b-joint-replication-subplan-2026-08-31.md"
C4A_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c4a-joint-feasibility/attempt-01/run_manifest.json"
C4A_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4a-joint-feasibility-result-2026-08-31.md"
C3B_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/attempt-02/run_manifest.json"
C3B_PROVENANCE = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/provenance-repair-2026-08-31/attempt-01/provenance_manifest.json"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_BACKEND = "tensorflow_eigh_strict"
SCHEMA = "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_c4b_joint_replication.v1"
DEFAULT_GPU_ID = "0"
COMPONENT_COUNT = 4
BATCH_SIZE = 32
BETA = 0.5
PILOT_UPDATES = 8
FORECAST_UPDATES = 16
VALIDATION_SIZE = 128
RELIABILITY_SIZE = 64
DIVERSITY_SIZE = 256
ALLOCATOR_CAP_BYTES = 4 * 1024**3
FORECAST_CAP_SECONDS = 3600.0
MATERIAL_CAP_SECONDS = 3600.0
ROW_SPECS = (
    {
        "row_id": "compact-high-root-1",
        "architecture": {"name": "compact-high", "hidden_layers": (16, 16), "activation": "tanh", "learning_rate": 1.0e-3},
        "initialization_root": (20260831, 62001),
        "preflight_root": (20260831, 62101),
        "training_root": (20260831, 62201),
        "validation_root": (20260831, 62301),
        "reliability_root": (20260831, 62401),
        "diversity_root": (20260831, 62501),
    },
    {
        "row_id": "compact-low-root-0",
        "architecture": {"name": "compact-low", "hidden_layers": (16, 16), "activation": "tanh", "learning_rate": 5.0e-4},
        "initialization_root": (20260831, 62002),
        "preflight_root": (20260831, 62102),
        "training_root": (20260831, 62202),
        "validation_root": (20260831, 62302),
        "reliability_root": (20260831, 62402),
        "diversity_root": (20260831, 62502),
    },
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

# This module performs no TensorFlow import at module load.  It supplies the
# already-audited standard-library artifact and replay helpers.
c3 = importlib.import_module("run_ssl_lstm_q20_phase8_c3_lineage_overlap_2026_08_30")


class C4BError(RuntimeError):
    pass


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _seed(tf: Any, root: tuple[int, int], *folds: int) -> tuple[int, int]:
    return c3._seed(tf, root, *folds)


def _scope(*, spec: Mapping[str, Any], component_id: str, arm: str, stage: str) -> Mapping[str, Any]:
    return {
        "data_identity": f"ssl-lstm-q20:{EXPECTED_TARGET_SIGNATURE}",
        "dtype": "float64",
        "backend": "tensorflow_tfp_gpu",
        "jit_compile": True,
        "principal_sqrt_backend": EXPECTED_BACKEND,
        "tf32_execution_enabled": True,
        "beta": BETA,
        "component_count": COMPONENT_COUNT,
        "component_id": component_id,
        "arm": arm,
        "stage": stage,
        "training_seed_derivation": {
            "initialization_root": list(spec["initialization_root"]),
            "preflight_root": list(spec["preflight_root"]),
            "training_root": list(spec["training_root"]),
            "validation_root": list(spec["validation_root"]),
        },
        "validation_bank_ids": [f"phase8-c4b-{spec['row_id']}-{arm}-{stage}-n{VALIDATION_SIZE}"],
    }


def _check_prerequisites() -> Mapping[str, Any]:
    for path in (C3B_MANIFEST, C3B_PROVENANCE, C4A_MANIFEST, C4A_RESULT, PLAN):
        if not path.is_file():
            raise C4BError(f"missing prerequisite: {path}")
    manifest = json.loads(C3B_MANIFEST.read_text(encoding="utf-8"))
    provenance = json.loads(C3B_PROVENANCE.read_text(encoding="utf-8"))
    c4a = json.loads(C4A_MANIFEST.read_text(encoding="utf-8"))
    c4a_result = C4A_RESULT.read_text(encoding="utf-8")
    checks = (
        manifest.get("status") == "PASS_PHASE8_C3B_L5_LADDER",
        manifest.get("target_signature") == EXPECTED_TARGET_SIGNATURE,
        manifest.get("principal_sqrt_backend") == EXPECTED_BACKEND,
        provenance.get("status") == "PASS_C3B_PROVENANCE_REPAIR",
        provenance.get("original_manifest", {}).get("manifest_hash") == manifest.get("manifest_hash"),
        c4a.get("status") == "PASS_PHASE8_C4A_JOINT_FEASIBILITY",
        c4a.get("target_signature") == EXPECTED_TARGET_SIGNATURE,
        c4a.get("principal_sqrt_backend") == EXPECTED_BACKEND,
        "Status: `PASS_C4A_JOINT_FEASIBILITY_NO_PROMOTION`" in c4a_result,
    )
    if not all(checks):
        raise C4BError("C3B/C4A prerequisite or provenance receipt failed")
    return {
        "c3b_manifest": {"path": str(C3B_MANIFEST.relative_to(ROOT)), "sha256": c3._sha256(C3B_MANIFEST), "status": manifest["status"]},
        "c3b_provenance": {"path": str(C3B_PROVENANCE.relative_to(ROOT)), "sha256": c3._sha256(C3B_PROVENANCE), "status": provenance["status"]},
        "c4a_manifest": {"path": str(C4A_MANIFEST.relative_to(ROOT)), "sha256": c3._sha256(C4A_MANIFEST), "status": c4a["status"]},
        "c4a_result": {"path": str(C4A_RESULT.relative_to(ROOT)), "sha256": c3._sha256(C4A_RESULT), "status": "PASS_C4A_JOINT_FEASIBILITY_NO_PROMOTION"},
    }


def _summary(tf: Any, physical: Any, logdet: Any, *, seed: tuple[int, int], component_index: int) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(physical, tf.float64)
    determinants = tf.convert_to_tensor(logdet, tf.float64)
    if values.shape != (DIVERSITY_SIZE, 4) or determinants.shape != (DIVERSITY_SIZE,):
        raise C4BError("diversity bank shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
        raise C4BError("nonfinite diversity map values")
    mean = tf.reduce_mean(values, axis=0)
    centered = values - mean[tf.newaxis, :]
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.constant(float(DIVERSITY_SIZE - 1), tf.float64)
    sign = values[:, 2]
    fractions = tf.stack((tf.reduce_mean(tf.cast(sign > 0.0, tf.float64)), tf.reduce_mean(tf.cast(sign < 0.0, tf.float64)), tf.reduce_mean(tf.cast(sign == 0.0, tf.float64))))
    return {
        "component_index": component_index,
        "seed": list(seed),
        "bank_size": DIVERSITY_SIZE,
        "mean": mean,
        "diagonal_variance": tf.linalg.diag_part(covariance),
        "covariance_trace": tf.linalg.trace(covariance),
        "covariance_frobenius_norm": tf.linalg.norm(covariance),
        "covariance": covariance,
        "sign_fraction_coordinate_2": fractions,
        "logdet_mean": tf.reduce_mean(determinants),
    }


def _update_payload(update: Any, elapsed: float) -> Mapping[str, Any]:
    return {
        "elapsed_seconds": elapsed,
        "loss": update.loss,
        "gradient_norm": update.gradient_norm,
        "clipped_gradient_norm": update.clipped_gradient_norm,
        "clipping_applied": update.clipping_applied,
        "step": update.step,
        "target_call_count": update.target_call_count,
        "cross_density_work": update.cross_density_work,
        "valid": update.valid,
        "target_finite": update.target_finite,
    }


def _run_arm(
    tf: Any,
    bridge: Any,
    *,
    spec: Mapping[str, Any],
    arm: str,
    start_checkpoints: Sequence[Mapping[str, Any]],
    configs: Sequence[Any],
    output_dir: Path,
    declared_points: Any,
    reference_points: Any,
    device_name: str,
) -> Mapping[str, Any]:
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        IndependentTemperedReverseKLTrainer,
        JointTemperedMixtureReverseKLTrainer,
        PreparedTransportInitialization,
        TransportBank,
        capture_trainable_transport_checkpoint,
        mixture_reverse_kl_terms,
        prepare_transport_initialization,
        pullback_gaussianization_diagnostic,
        restore_trainable_transport_checkpoint,
        transport_preflight_state_hash,
    )
    from bayesfilter.inference.tempered_transitions_tf import screen_transport_reliability

    if len(start_checkpoints) != COMPONENT_COUNT:
        raise C4BError("four start checkpoints are required")
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    independent_transports = []
    joint_transports = []
    independent_preflights = []
    joint_preflights = []
    component_ids = tuple(str(checkpoint["component_id"]) for checkpoint in start_checkpoints)
    for index, (checkpoint, config, component_id) in enumerate(zip(start_checkpoints, configs, component_ids, strict=True)):
        independent = restore_trainable_transport_checkpoint(checkpoint)
        joint = restore_trainable_transport_checkpoint(checkpoint)
        if transport_preflight_state_hash(independent) != transport_preflight_state_hash(joint):
            raise C4BError(f"initial copy mismatch for component {component_id}")
        independent_prepared = prepare_transport_initialization(
            independent, bridge, component_id=component_id, seed=_seed(tf, spec["preflight_root"], 0, index), batch_size=BATCH_SIZE, repair_scales=(1.0,), beta=BETA
        )
        joint_prepared = prepare_transport_initialization(
            joint, bridge, component_id=component_id, seed=_seed(tf, spec["preflight_root"], 1, index), batch_size=BATCH_SIZE, repair_scales=(1.0,), beta=BETA
        )
        if not independent_prepared.receipt.valid or not joint_prepared.receipt.valid:
            raise C4BError(f"beta=.5 preflight failed for component {component_id}")
        independent_transports.append(independent_prepared.transport)
        joint_transports.append(joint_prepared.transport)
        independent_preflights.append(independent_prepared.receipt)
        joint_preflights.append(joint_prepared.receipt)

    independent_bank = TransportBank(independent_transports, component_ids=component_ids)
    joint_bank = TransportBank(joint_transports, component_ids=component_ids)
    independent_trainers = tuple(
        IndependentTemperedReverseKLTrainer(
            config,
            bridge,
            beta=BETA,
            component_id=component_id,
            batch_size=BATCH_SIZE,
            prepared_initialization=PreparedTransportInitialization(transport, receipt),
        )
        for config, component_id, transport, receipt in zip(configs, component_ids, independent_transports, independent_preflights, strict=True)
    )
    joint_trainer = JointTemperedMixtureReverseKLTrainer(
        joint_bank,
        bridge,
        beta=BETA,
        batch_size=BATCH_SIZE,
        preflight_receipts=tuple(joint_preflights),
        learning_rate=float(spec["architecture"]["learning_rate"]),
        gradient_clip_norm=10.0,
        jit_compile=True,
        train_alpha=True,
    )

    independent_updates: list[list[Mapping[str, Any]]] = [[] for _ in range(COMPONENT_COUNT)]
    independent_times: list[list[float]] = [[] for _ in range(COMPONENT_COUNT)]
    joint_updates: list[Mapping[str, Any]] = []
    joint_times: list[float] = []
    for update_index in range(PILOT_UPDATES):
        base_seed = _seed(tf, spec["training_root"], update_index)
        for component_index, trainer in enumerate(independent_trainers):
            # Match the joint trainer's stateless fold-in construction.
            component_seed_tensor = tf.random.experimental.stateless_fold_in(tf.constant(base_seed, tf.int32), component_index)
            component_seed = tuple(int(value) for value in component_seed_tensor.numpy().tolist())
            started = time.monotonic()
            result = trainer.train_step(component_seed)
            elapsed = time.monotonic() - started
            if not bool(result.valid.numpy()):
                raise C4BError(f"independent update invalid for component {component_index}")
            if int(result.target_call_count.numpy()) != 1 or int(result.cross_density_work.numpy()) != 0:
                raise C4BError("independent work telemetry does not match its contract")
            independent_times[component_index].append(elapsed)
            independent_updates[component_index].append({"update": update_index + 1, "seed": list(component_seed), **_update_payload(result, elapsed)})
        started = time.monotonic()
        result = joint_trainer.train_step(base_seed)
        elapsed = time.monotonic() - started
        if not bool(result.valid.numpy()):
            raise C4BError("joint update invalid")
        if int(result.target_call_count.numpy()) != 1 or int(result.cross_density_work.numpy()) != COMPONENT_COUNT * COMPONENT_COUNT * BATCH_SIZE:
            raise C4BError("joint work telemetry does not match K^2 B contract")
        joint_times.append(elapsed)
        joint_updates.append({"update": update_index + 1, "seed": list(base_seed), **_update_payload(result, elapsed)})

    endpoint_checkpoints = {"independent": [], "joint": []}
    for arm_name, transports in (("independent", independent_transports), ("joint", joint_transports)):
        for index, (transport, component_id, start) in enumerate(zip(transports, component_ids, start_checkpoints, strict=True)):
            checkpoint = capture_trainable_transport_checkpoint(
                transport,
                component_id=component_id,
                beta=BETA,
                bridge_signature=str(bridge.signature),
                target_signature=EXPECTED_TARGET_SIGNATURE,
                parent_checkpoint_hash=str(start["checkpoint_hash"]),
                update_count=PILOT_UPDATES,
                checkpoint_scope=_scope(spec=spec, component_id=component_id, arm=arm_name, stage="final"),
            )
            c3._write_json(
                arm_dir / f"{arm_name}-component-{index}-beta05-final.json",
                c3._json_safe(checkpoint, tf),
            )
            restored = restore_trainable_transport_checkpoint(checkpoint)
            latent = tf.random.stateless_normal([VALIDATION_SIZE, 4], tf.constant(_seed(tf, spec["validation_root"], 0 if arm_name == "independent" else 1, index), tf.int32), dtype=tf.float64)
            replay = c3._checkpoint_replay_error(tf, transport, restored, latent)
            if any(float(value.numpy()) > 1.0e-10 for value in replay.values()):
                raise C4BError(f"{arm_name} checkpoint replay failed for component {index}")
            endpoint_checkpoints[arm_name].append({"checkpoint": checkpoint, "replay": replay, "transport": restored})

    def score_fn(physical: Any) -> Any:
        _value, score, status = bridge.value_score_status(physical, tf.constant(BETA, tf.float64))
        valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
        if not bool(tf.reduce_all(valid).numpy()):
            raise C4BError("reliability score bank contains an invalid row")
        return score

    arm_results = {}
    for arm_name, bank, endpoint in (("independent", independent_bank, endpoint_checkpoints["independent"]), ("joint", joint_bank, endpoint_checkpoints["joint"])):
        latent_bank = tf.stack([tf.random.stateless_normal([VALIDATION_SIZE, 4], tf.constant(_seed(tf, spec["validation_root"], 10 if arm_name == "independent" else 11, index), tf.int32), dtype=tf.float64) for index in range(COMPONENT_COUNT)], axis=0)
        physical_bank, logdet_bank = bank.forward_bank(latent_bank)
        flattened = tf.reshape(physical_bank, [COMPONENT_COUNT * VALIDATION_SIZE, 4])
        target, score, status = bridge.value_score_status(flattened, tf.constant(BETA, tf.float64))
        target_valid = bool(tf.reduce_all(tf.convert_to_tensor(status["bridge_valid"], tf.bool)).numpy())
        if not target_valid:
            raise C4BError(f"{arm_name} held-out target status invalid")
        target_bank = tf.reshape(target, [COMPONENT_COUNT, VALIDATION_SIZE])
        loss, _per_sample, _mixture, cross = mixture_reverse_kl_terms(bank, physical_bank, target_bank, logdet_bank)
        if not bool(tf.reduce_all(tf.math.is_finite(cross)).numpy()) or not bool(tf.math.is_finite(loss).numpy()):
            raise C4BError(f"{arm_name} held-out mixture objective nonfinite")
        reliability_latent = tf.stack([tf.random.stateless_normal([RELIABILITY_SIZE, 4], tf.constant(_seed(tf, spec["reliability_root"], 10 if arm_name == "independent" else 11, index), tf.int32), dtype=tf.float64) for index in range(COMPONENT_COUNT)], axis=0)
        reliability_physical = tf.stack([bank.transports[index].forward_and_logdet(reliability_latent[index])[0] for index in range(COMPONENT_COUNT)], axis=0)
        reliability = screen_transport_reliability(bank.transports, component_ids=component_ids, self_latent_bank=reliability_latent, cross_physical_bank=reliability_physical, reference_points=reference_points, declared_points=declared_points, physical_score_fn=score_fn, maximum_condition_number=1.0e8)
        if not reliability.passed:
            raise C4BError(f"{arm_name} learned-map reliability failed")
        summaries = []
        for index, transport in enumerate(bank.transports):
            latent = tf.random.stateless_normal([DIVERSITY_SIZE, 4], tf.constant(_seed(tf, spec["diversity_root"], 10 if arm_name == "independent" else 11, index), tf.int32), dtype=tf.float64)
            physical, logdet = transport.forward_and_logdet(latent)
            summaries.append(_summary(tf, physical, logdet, seed=_seed(tf, spec["diversity_root"], 10 if arm_name == "independent" else 11, index), component_index=index))
        hashes = [transport_preflight_state_hash(transport) for transport in bank.transports]
        diagnostics = []
        for index, transport in enumerate(bank.transports):
            latent = tf.random.stateless_normal([VALIDATION_SIZE, 4], tf.constant(_seed(tf, spec["validation_root"], 20 if arm_name == "independent" else 21, index), tf.int32), dtype=tf.float64)
            diagnostic = pullback_gaussianization_diagnostic(transport, bridge, beta=BETA, latent=latent)
            diagnostics.append({"component_index": index, "finite": diagnostic.finite, "centered_log_density_rms": diagnostic.centered_log_density_rms, "pullback_score_rms_per_coordinate": diagnostic.pullback_score_rms_per_coordinate, "pullback_score_maximum_row_norm": diagnostic.pullback_score_maximum_row_norm})
        if len(set(hashes)) != COMPONENT_COUNT:
            raise C4BError(f"{arm_name} produced exact duplicate final chart states")
        alpha_values = tf.convert_to_tensor(bank.alpha, tf.float64)
        if not bool(tf.reduce_all(tf.math.is_finite(alpha_values)).numpy()) or not bool(tf.reduce_all(alpha_values > 0.0).numpy()):
            raise C4BError(f"{arm_name} alpha is not finite and strictly positive")
        arm_results[arm_name] = {
            "loss_on_fresh_bank": loss,
            "target_rows": COMPONENT_COUNT * VALIDATION_SIZE,
            "target_call_count": 1,
            "cross_density_shape": cross.shape,
            "cross_density_work": COMPONENT_COUNT * COMPONENT_COUNT * VALIDATION_SIZE,
            "reliability": reliability.payload(),
            "diagnostics": diagnostics,
            "diversity": summaries,
            "state_hashes": hashes,
            "exact_duplicate_count": COMPONENT_COUNT - len(set(hashes)),
            "alpha": bank.alpha,
            "alpha_entropy": -tf.reduce_sum(bank.alpha * tf.math.log(bank.alpha)),
            "endpoint_checkpoints": [
                {"checkpoint": item["checkpoint"], "replay": item["replay"]}
                for item in endpoint
            ],
        }

    joint_forecast = float(joint_times[0]) + (FORECAST_UPDATES - 1) * float(statistics.median(joint_times[1:]))
    allocator = c3._memory_info(tf, device_name)
    if int(allocator.get("peak", ALLOCATOR_CAP_BYTES + 1)) > ALLOCATOR_CAP_BYTES:
        raise C4BError("C4B allocator cap exceeded")
    return {
        "status": "PASS_C4B_ROW" if joint_forecast <= FORECAST_CAP_SECONDS else "C4B_ROW_RESOURCE_FORECAST_EXCEEDED",
        "architecture": {"name": spec["architecture"]["name"], "hidden_layers": list(spec["architecture"]["hidden_layers"]), "activation": spec["architecture"]["activation"], "learning_rate": spec["architecture"]["learning_rate"]},
        "component_ids": list(component_ids),
        "component_count": COMPONENT_COUNT,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "pilot_updates": PILOT_UPDATES,
        "forecast_updates": FORECAST_UPDATES,
        "independent_updates": independent_updates,
        "joint_updates": joint_updates,
        "independent_update_times": independent_times,
        "joint_update_times": joint_times,
        "joint_resource_forecast_seconds": joint_forecast,
        "allocator": allocator,
        "arms": arm_results,
        "nonclaims": ["replication diagnostic only", "no whitening, mode discovery, posterior, HMC, ranking, superiority, or scaling claim"],
    }


def _protocol() -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "c4b-k4-joint-replication",
        "component_count": COMPONENT_COUNT,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "pilot_updates": PILOT_UPDATES,
        "forecast_updates": FORECAST_UPDATES,
        "validation_size": VALIDATION_SIZE,
        "reliability_size": RELIABILITY_SIZE,
        "diversity_size": DIVERSITY_SIZE,
        "rows": [
            {
                "row_id": spec["row_id"],
                "architecture": {
                    "name": spec["architecture"]["name"],
                    "hidden_layers": list(spec["architecture"]["hidden_layers"]),
                    "activation": spec["architecture"]["activation"],
                    "learning_rate": spec["architecture"]["learning_rate"],
                },
                "roots": {
                    key: list(spec[key])
                    for key in ("initialization_root", "preflight_root", "training_root", "validation_root", "reliability_root", "diversity_root")
                },
            }
            for spec in ROW_SPECS
        ],
        "role": "optional_joint_arm_replication_only",
    }


def _run(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        raise C4BError("--output-dir is required")
    if not _truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
        raise C4BError("C4B requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise C4BError("C4B requires one explicitly visible GPU")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise C4BError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    prerequisites = _check_prerequisites()
    import tensorflow as tf
    from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.inference.tempered_transport_ensemble_tf import capture_trainable_transport_checkpoint, prepare_transport_initialization, restore_trainable_transport_checkpoint

    gpu_snapshot_before = c3._nvidia_snapshot()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise C4BError("C4B requires exactly one visible logical GPU")
    device_name = str(logical_gpus[0].name)
    bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend=EXPECTED_BACKEND)
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise C4BError("q=20 target signature changed")
    declared_points, map_receipt = c3._map_representatives(tf, EXPECTED_TARGET_SIGNATURE)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    reference_points = tf.concat((center[tf.newaxis, :], center[tf.newaxis, :] + 4.0 * tf.eye(4, dtype=tf.float64), center[tf.newaxis, :] - 4.0 * tf.eye(4, dtype=tf.float64)), axis=0)
    route_paths = (ROOT / "bayesfilter/inference/tempered_target_tf.py", ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py", ROOT / "bayesfilter/inference/tempered_transitions_tf.py", ROOT / "bayesfilter/inference/tempered_lineage_tf.py")
    route_scan = c3._static_scan((*route_paths, Path(__file__).resolve()))
    if not route_scan["passed"]:
        raise C4BError(f"forbidden runtime route token: {route_scan}")

    rows = []
    failures = []
    for spec in ROW_SPECS:
        if time.monotonic() - started + 30.0 >= MATERIAL_CAP_SECONDS:
            raise C4BError("C4B material cap exhausted before all rows")
        c3._reset_memory(tf, device_name)
        architecture = spec["architecture"]
        configs = tuple(
            WeightedNeuTraConfig(
                dimension=4,
                hidden_layers=tuple(architecture["hidden_layers"]),
                stages=2,
                activation=str(architecture["activation"]),
                initialization_scale=0.02,
                initialization_seed=_seed(tf, spec["initialization_root"], index),
                learning_rate=float(architecture["learning_rate"]),
                jit_compile=True,
            )
            for index in range(COMPONENT_COUNT)
        )
        start_checkpoints = []
        for index, config in enumerate(configs):
            component_id = f"c4b-{spec['row_id']}-c{index}"
            raw = WeightedDenseIAFTransport(config)
            prepared = prepare_transport_initialization(
                raw,
                bridge,
                component_id=component_id,
                seed=_seed(tf, spec["preflight_root"], index),
                batch_size=BATCH_SIZE,
                repair_scales=(1.0,),
                beta=0.0,
                reference_center=center,
                reference_scale=tf.fill([4], tf.sqrt(tf.constant(float(bridge.prior_variance), tf.float64))),
            )
            if not prepared.receipt.valid:
                raise C4BError(f"beta=0 preflight failed for {component_id}")
            checkpoint = capture_trainable_transport_checkpoint(
                prepared.transport,
                component_id=component_id,
                beta=0.0,
                bridge_signature=str(bridge.signature),
                target_signature=EXPECTED_TARGET_SIGNATURE,
                parent_checkpoint_hash=None,
                update_count=0,
                checkpoint_scope=_scope(spec=spec, component_id=component_id, arm="shared-start", stage="beta0"),
            )
            beta0 = restore_trainable_transport_checkpoint(checkpoint)
            beta05 = prepare_transport_initialization(
                beta0,
                bridge,
                component_id=component_id,
                seed=_seed(tf, spec["preflight_root"], 20, index),
                batch_size=BATCH_SIZE,
                repair_scales=(1.0,),
                beta=BETA,
            )
            if not beta05.receipt.valid:
                raise C4BError(f"beta=.5 start preflight failed for {component_id}")
            start = capture_trainable_transport_checkpoint(
                beta05.transport,
                component_id=component_id,
                beta=BETA,
                bridge_signature=str(bridge.signature),
                target_signature=EXPECTED_TARGET_SIGNATURE,
                parent_checkpoint_hash=str(checkpoint["checkpoint_hash"]),
                update_count=0,
                checkpoint_scope=_scope(spec=spec, component_id=component_id, arm="shared-start", stage="beta05"),
            )
            restore_trainable_transport_checkpoint(start)
            start_checkpoints.append(start)
        row_dir = output_dir / spec["row_id"]
        row_dir.mkdir(parents=True, exist_ok=True)
        for index, checkpoint in enumerate(start_checkpoints):
            c3._write_json(row_dir / f"shared-component-{index}-beta05-start.json", c3._json_safe(checkpoint, tf))
        try:
            row = _run_arm(tf, bridge, spec=spec, arm="matched", start_checkpoints=tuple(start_checkpoints), configs=configs, output_dir=row_dir, declared_points=declared_points, reference_points=reference_points, device_name=device_name)
            rows.append(row)
        except Exception as exc:
            failures.append({"row_id": spec["row_id"], "error_type": type(exc).__name__, "error": str(exc)})
    elapsed = time.monotonic() - started
    passed = len(rows) == len(ROW_SPECS) and not failures and all(row["status"] == "PASS_C4B_ROW" and float(row["joint_resource_forecast_seconds"]) <= FORECAST_CAP_SECONDS for row in rows)
    source_paths = (*route_paths, Path(__file__).resolve(), Path(c3.__file__).resolve(), PLAN, C3B_MANIFEST, C3B_PROVENANCE, C4A_MANIFEST, C4A_RESULT)
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_PHASE8_C4B_JOINT_REPLICATION" if passed else "FAIL_PHASE8_C4B_JOINT_REPLICATION",
        "role": "optional_joint_arm_replication_only",
        "protocol": _protocol(),
        "command": sys.argv,
        "output_dir": str(output_dir),
        "git_commit": c3._git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": c3._git(("git", "status", "--porcelain")),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": __import__("tensorflow_probability").__version__,
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "bridge_signature": str(bridge.signature),
        "properness_receipt": bridge.properness_receipt.payload(),
        "principal_sqrt_backend": EXPECTED_BACKEND,
        "jit_compile": True,
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "logical_gpus": [str(item.name) for item in logical_gpus],
        "memory_policy": memory_policy,
        "gpu_environment": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""), "selection_policy": "repository_default_single_gpu_no_idle_probe"},
        "gpu_snapshot_before": gpu_snapshot_before,
        "gpu_snapshot_after": c3._nvidia_snapshot(),
        "prerequisites": prerequisites,
        "map_representatives": map_receipt,
        "route_scan": route_scan,
        "rows": rows,
        "failures": failures,
        "hard_screen": {"all_rows_complete": len(rows) == len(ROW_SPECS), "failure_count": len(failures), "target_status_valid": not failures, "joint_work_contract": not failures, "checkpoint_replay": not failures, "reliability": not failures, "alpha_valid": not failures, "allocator_cap": all(int(row["allocator"]["peak"]) <= ALLOCATOR_CAP_BYTES for row in rows), "forecast_cap": all(float(row["joint_resource_forecast_seconds"]) <= FORECAST_CAP_SECONDS for row in rows)},
        "budget": {"material_cap_seconds": MATERIAL_CAP_SECONDS, "elapsed_seconds": elapsed, "forecast_cap_seconds": FORECAST_CAP_SECONDS},
        "source_hashes": {str(path.relative_to(ROOT)): c3._sha256(path) for path in source_paths},
        "wall_time_seconds": elapsed,
        "nonclaims": ["replication diagnostic only", "no whitening, mode discovery, posterior, HMC, ranking, superiority, or scaling claim"],
    }
    safe = c3._json_safe(manifest, tf)
    safe["manifest_hash"] = c3._stable_hash(safe)
    c3._write_json(output_dir / "run_manifest.json", safe)
    print(json.dumps({"status": safe["status"], "output_dir": str(output_dir), "wall_time_seconds": elapsed, "row_count": len(rows), "failed_rows": len(failures)}, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("--output-dir", type=Path)
    parsed.add_argument("--print-protocol", action="store_true")
    args = parsed.parse_args()
    if args.print_protocol:
        print(json.dumps(_protocol(), sort_keys=True, indent=2))
        return 0
    try:
        return _run(args)
    except Exception as exc:
        if isinstance(args.output_dir, Path):
            output_dir = args.output_dir.expanduser().resolve()
            if output_dir.is_dir():
                try:
                    c3._write_json(output_dir / "failure.json", {"status": "FAIL_PHASE8_C4B_JOINT_REPLICATION", "error_type": type(exc).__name__, "error": str(exc), "command": sys.argv})
                except Exception:
                    pass
        print(json.dumps({"status": "FAIL_PHASE8_C4B_JOINT_REPLICATION", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
