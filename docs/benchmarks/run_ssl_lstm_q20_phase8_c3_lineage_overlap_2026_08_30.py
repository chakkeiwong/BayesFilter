#!/usr/bin/env python3
"""Run the bounded q=20 C3 lineage and temperature-overlap pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-subplan-2026-08-30.md"
MASTER_PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md"
MAP_ARTIFACT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r3/map-progress.json"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
PRINCIPAL_SQRT_BACKEND = "tensorflow_eigh_strict"
C2_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-result-2026-08-30.md"
C2_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/screen/attempt-02-eight-rows/run_manifest.json"
C2_PARITY_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/backend-parity/attempt-02-b8-graph-custom/run_manifest.json"
SCHEMA = "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_c3_lineage_overlap.v1"
DEFAULT_GPU_ID = "0"
MATERIAL_CAP_SECONDS = 6600.0
BATCH_SIZE = 32
TRAIN_UPDATES = 16
OVERLAP_CHAINS = 64
RELIABILITY_CHAINS = 64
STRESS_SIZE = 64
BETAS = (0.0, 0.5, 1.0)
ROOTS = ((20260830, 13001), (20260830, 13002))
TRAINING_ROOT = (20260830, 23001)
OVERLAP_ROOTS = ((20260830, 43001), (20260830, 43002))
ARCHITECTURES = (
    {
        "name": "compact-high",
        "hidden_layers": (16, 16),
        "activation": "tanh",
        "learning_rate": 1.0e-3,
    },
    {
        "name": "compact-low",
        "hidden_layers": (16, 16),
        "activation": "tanh",
        "learning_rate": 5.0e-4,
    },
)
ARMS = (
    {"name": "pure-continuation", "discovery_arm": "pure_continuation", "restart_indices": ()},
    {"name": "positive-branching", "discovery_arm": "positive_temperature_branching", "restart_indices": (1,)},
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class C3Error(RuntimeError):
    pass


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise C3Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any, tf: Any) -> Any:
    if tf.is_tensor(value):
        return _json_safe(value.numpy(), tf)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, tf) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, tf) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), tf)
    if hasattr(value, "item"):
        return _json_safe(value.item(), tf)
    return str(value)


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _nvidia_snapshot() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": type(exc).__name__}


def _seed(tf: Any, root: tuple[int, int], *folds: int) -> tuple[int, int]:
    value = tf.constant(root, tf.int32)
    for fold in folds:
        value = tf.random.experimental.stateless_fold_in(value, int(fold))
    return tuple(int(item) for item in value.numpy())


def _memory_info(tf: Any, device_name: str) -> Mapping[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info(device_name))
    except (AttributeError, RuntimeError, ValueError) as exc:
        return {"unavailable": type(exc).__name__}


def _reset_memory(tf: Any, device_name: str) -> None:
    try:
        tf.config.experimental.reset_memory_stats(device_name)
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise C3Error("could not reset GPU allocator telemetry") from exc


def _checkpoint_replay_error(tf: Any, original: Any, restored: Any, latent: Any) -> Mapping[str, Any]:
    original_physical, original_logdet = original.forward_and_logdet(latent)
    restored_physical, restored_logdet = restored.forward_and_logdet(latent)
    recovered, recovered_logdet = restored.inverse_and_forward_logdet(original_physical)
    return {
        "forward_max_abs": tf.reduce_max(tf.abs(restored_physical - original_physical)),
        "forward_logdet_max_abs": tf.reduce_max(tf.abs(restored_logdet - original_logdet)),
        "roundtrip_max_abs": tf.reduce_max(tf.abs(recovered - latent)),
        "inverse_logdet_max_abs": tf.reduce_max(tf.abs(recovered_logdet - original_logdet)),
    }


def _scope(
    *,
    target_signature: str,
    architecture: str,
    arm: str,
    root_index: int,
    beta: float,
) -> Mapping[str, Any]:
    label = str(beta).replace(".", "p")
    return {
        "data_identity": f"ssl-lstm-q20:{target_signature}",
        "dtype": "float64",
        "backend": "tensorflow_tfp_gpu",
        "jit_compile": True,
        "principal_sqrt_backend": PRINCIPAL_SQRT_BACKEND,
        "tf32_execution_enabled": True,
        "training_seed_derivation": {
            "initialization_root": list(ROOTS[root_index]),
            "training_root": list(TRAINING_ROOT),
            "overlap_root": list(OVERLAP_ROOTS[root_index]),
            "folds": {
                "architecture": architecture,
                "arm": arm,
                "root_index": int(root_index),
                "beta": float(beta),
            },
        },
        "validation_bank_ids": [
            f"phase8-c3-overlap-{architecture}-{arm}-r{root_index}-beta{label}-n{OVERLAP_CHAINS}"
        ],
    }


def _map_representatives(tf: Any, target_signature: str) -> tuple[Any, Mapping[str, Any]]:
    if target_signature != EXPECTED_TARGET_SIGNATURE:
        raise C3Error("MAP representatives are stale for the frozen target")
    payload = json.loads(MAP_ARTIFACT.read_text(encoding="utf-8"))
    starts = payload.get("starts")
    if not isinstance(starts, list):
        raise C3Error("MAP artifact does not contain starts")
    eligible: dict[str, list[Mapping[str, Any]]] = {"plus": [], "minus": []}
    for row in starts:
        if not isinstance(row, Mapping):
            continue
        try:
            position = [float(item) for item in row.get("position", ())]
            score = float(row.get("score_inf_norm"))
            log_prob = float(row.get("log_prob"))
        except (TypeError, ValueError):
            continue
        if len(position) != 4 or not all(math.isfinite(item) for item in position):
            continue
        if not math.isfinite(score) or score > 1.0e-5 or not math.isfinite(log_prob):
            continue
        if position[2] == 0.0:
            continue
        label = "plus" if position[2] > 0.0 else "minus"
        eligible[label].append({"position": position, "score_inf_norm": score, "log_prob": log_prob})
    if not eligible["plus"] or not eligible["minus"]:
        raise C3Error("MAP artifact lacks both sign regions")
    selected = {label: max(rows, key=lambda item: float(item["log_prob"])) for label, rows in eligible.items()}
    points = tf.stack(
        tuple(tf.convert_to_tensor(selected[label]["position"], tf.float64) for label in ("plus", "minus")),
        axis=0,
    )
    return points, {
        "path": str(MAP_ARTIFACT.relative_to(ROOT)),
        "sha256": _sha256(MAP_ARTIFACT),
        "selection": "highest_log_prob_finite_stationary_row_per_sign",
        "bound_target_signature": target_signature,
        "selected": selected,
    }


def _check_c2_prerequisites() -> Mapping[str, Any]:
    """Require the recorded strict-backend calibration and parity receipts."""
    required = (C2_RESULT, C2_MANIFEST, C2_PARITY_MANIFEST)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise C3Error(f"required C2 prerequisites are missing: {missing}")
    result_text = C2_RESULT.read_text(encoding="utf-8")
    if "Status: `PASS_C2_STRICT_CALIBRATION_WITHOUT_WHITENING_PROMOTION`" not in result_text:
        raise C3Error("C2 result note does not carry the required pass status")
    manifest = json.loads(C2_MANIFEST.read_text(encoding="utf-8"))
    parity = json.loads(C2_PARITY_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS_PHASE8_C2_STRICT_CALIBRATION"
        or manifest.get("target_signature") != EXPECTED_TARGET_SIGNATURE
        or manifest.get("principal_sqrt_backend") != PRINCIPAL_SQRT_BACKEND
        or not bool(manifest.get("hard_screen", {}).get("all_rows_pass"))
        or not bool(manifest.get("hard_screen", {}).get("all_reliability_pass"))
    ):
        raise C3Error("C2 strict-calibration manifest does not pass its hard screen")
    if (
        parity.get("status") != "PASS_Q20_BACKEND_PARITY_BATCH"
        or parity.get("target_signature") != EXPECTED_TARGET_SIGNATURE
        or int(parity.get("batch_size", -1)) != 8
        or parity.get("strict_backend") != PRINCIPAL_SQRT_BACKEND
        or not bool(parity.get("parity", {}).get("passed"))
    ):
        raise C3Error("C2 B=8 strict-backend parity receipt does not pass")
    return {
        "result_note": {"path": str(C2_RESULT.relative_to(ROOT)), "sha256": _sha256(C2_RESULT)},
        "screen_manifest": {"path": str(C2_MANIFEST.relative_to(ROOT)), "sha256": _sha256(C2_MANIFEST), "status": manifest["status"]},
        "parity_manifest": {"path": str(C2_PARITY_MANIFEST.relative_to(ROOT)), "sha256": _sha256(C2_PARITY_MANIFEST), "status": parity["status"]},
    }


def _diagnostic_payload(diagnostic: Any, tf: Any) -> Mapping[str, Any]:
    return {
        "finite": diagnostic.finite,
        "valid_row_count": diagnostic.valid_row_count,
        "batch_size": diagnostic.batch_size,
        "reverse_kl_mean": tf.reduce_mean(diagnostic.reverse_kl_per_sample),
        "centered_log_density_rms": diagnostic.centered_log_density_rms,
        "pullback_score_rms_per_coordinate": diagnostic.pullback_score_rms_per_coordinate,
        "pullback_score_maximum_row_norm": diagnostic.pullback_score_maximum_row_norm,
    }


def _static_scan(paths: Sequence[Path]) -> Mapping[str, Any]:
    forbidden = ("tf.map_fn", "tf.vectorized_map", "GradientTape.jacobian", "GradientTape.batch_jacobian", "pfor")
    hits = {token: [] for token in forbidden}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                hits[token].append(str(path.relative_to(ROOT)))
    return {"paths": [str(path.relative_to(ROOT)) for path in paths], "hits": hits, "passed": not any(hits.values())}


def _train_chart(
    tf: Any,
    bridge: Any,
    *,
    transport: Any,
    config: Any,
    component_id: str,
    beta: float,
    preflight_seed: tuple[int, int],
    training_seed: tuple[int, int],
    reference_center: Any | None,
    reference_scale: Any | None,
    parent_checkpoint_hash: str | None,
    checkpoint_scope: Mapping[str, Any],
    row_dir: Path,
    target_signature: str,
    bridge_signature: str,
    capture: Any,
    restore: Any,
    prepare: Any,
    trainer_class: Any,
    prepared_class: Any,
) -> Mapping[str, Any]:
    prepared = prepare(
        transport,
        bridge,
        component_id=component_id,
        seed=preflight_seed,
        batch_size=BATCH_SIZE,
        repair_scales=(1.0, 0.5, 0.25) if float(beta) == 0.0 else (1.0,),
        beta=float(beta),
        reference_center=reference_center,
        reference_scale=reference_scale,
    )
    if not prepared.receipt.valid:
        raise C3Error(f"preflight failed for {component_id} beta={beta}")
    checkpoint = capture(
        prepared.transport,
        component_id=component_id,
        beta=float(beta),
        bridge_signature=bridge_signature,
        target_signature=target_signature,
        parent_checkpoint_hash=parent_checkpoint_hash,
        update_count=0,
        checkpoint_scope=checkpoint_scope,
    )
    _write_json(row_dir / f"beta-{str(beta).replace('.', 'p')}-start.json", _json_safe(checkpoint, tf))
    restored = restore(
        checkpoint,
        expected_context={
            "component_id": component_id,
            "beta": float(beta),
            "bridge_signature": bridge_signature,
            "target_signature": target_signature,
            "checkpoint_scope": checkpoint["checkpoint_scope"],
        },
    )
    latent = tf.random.stateless_normal(
        [OVERLAP_CHAINS, int(config.dimension)], tf.constant(training_seed, tf.int32), dtype=tf.float64
    )
    replay = _checkpoint_replay_error(tf, prepared.transport, restored, latent)
    tolerance = 1.0e-10
    if any(float(value.numpy()) > tolerance for value in replay.values()):
        raise C3Error(f"checkpoint replay failed for {component_id} beta={beta}")
    if float(beta) == 0.0:
        return {"transport": restored, "checkpoint": checkpoint, "preflight": prepared.receipt.payload(), "replay": replay, "updates": []}
    trainer = trainer_class(
        config,
        bridge,
        beta=float(beta),
        component_id=component_id,
        batch_size=BATCH_SIZE,
        prepared_initialization=prepared_class(prepared.transport, prepared.receipt),
    )
    updates = []
    for update_index in range(TRAIN_UPDATES):
        update = trainer.train_step(_seed(tf, training_seed, update_index))
        if not bool(update.valid.numpy()):
            raise C3Error(f"invalid update for {component_id} beta={beta}")
        updates.append(
            {
                "update": update_index + 1,
                "loss": update.loss,
                "gradient_norm": update.gradient_norm,
                "clipped_gradient_norm": update.clipped_gradient_norm,
                "clipping_applied": update.clipping_applied,
                "step": update.step,
                "target_call_count": update.target_call_count,
                "cross_density_work": update.cross_density_work,
                "valid": update.valid,
            }
        )
    final_checkpoint = capture(
        trainer.transport,
        component_id=component_id,
        beta=float(beta),
        bridge_signature=bridge_signature,
        target_signature=target_signature,
        parent_checkpoint_hash=str(checkpoint["checkpoint_hash"]),
        update_count=TRAIN_UPDATES,
        checkpoint_scope=checkpoint_scope,
    )
    _write_json(row_dir / f"beta-{str(beta).replace('.', 'p')}-final.json", _json_safe(final_checkpoint, tf))
    final_restored = restore(
        final_checkpoint,
        expected_context={
            "component_id": component_id,
            "beta": float(beta),
            "bridge_signature": bridge_signature,
            "target_signature": target_signature,
            "checkpoint_scope": final_checkpoint["checkpoint_scope"],
        },
    )
    final_replay = _checkpoint_replay_error(tf, trainer.transport, final_restored, latent)
    if any(float(value.numpy()) > tolerance for value in final_replay.values()):
        raise C3Error(f"final checkpoint replay failed for {component_id} beta={beta}")
    return {
        "transport": final_restored,
        "checkpoint": final_checkpoint,
        "preflight": prepared.receipt.payload(),
        "replay": {"start": replay, "final": final_replay},
        "updates": updates,
    }


def _run_row(tf: Any, bridge: Any, *, architecture: Mapping[str, Any], architecture_index: int, arm: Mapping[str, Any], arm_index: int, root_index: int, output_dir: Path, device_name: str, declared_points: Any, reference_points: Any, started: float) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig
    from bayesfilter.inference.tempered_lineage_tf import TemperedLineageConfig, TemperedLineageController
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        IndependentTemperedReverseKLTrainer,
        PreparedTransportInitialization,
        capture_trainable_transport_checkpoint,
        prepare_transport_initialization,
        restore_trainable_transport_checkpoint,
        pullback_gaussianization_diagnostic,
    )
    from bayesfilter.inference.tempered_transitions_tf import screen_transport_reliability

    architecture_name = str(architecture["name"])
    arm_name = str(arm["name"])
    lineage = TemperedLineageController(
        TemperedLineageConfig(
            betas=BETAS,
            component_ids=(f"c3-{architecture_name}-{arm_name}-c0", f"c3-{architecture_name}-{arm_name}-c1"),
            root_seed=ROOTS[root_index],
            discovery_arm=str(arm["discovery_arm"]),
            positive_branch_betas=(0.5,) if arm["discovery_arm"] == "positive_temperature_branching" else (),
            restart_component_indices=tuple(int(item) for item in arm["restart_indices"]),
            preflight_batch_size=BATCH_SIZE,
        ),
        bridge,
    )
    checkpoints = [lineage.checkpoint(index) for index in range(len(BETAS))]
    config_by_component = []
    for component_index in range(2):
        config_by_component.append(
            WeightedNeuTraConfig(
                dimension=int(bridge.parameter_dim),
                hidden_layers=tuple(int(value) for value in architecture["hidden_layers"]),
                stages=2,
                activation=str(architecture["activation"]),
                initialization_scale=0.02,
                initialization_seed=lineage.component_seed(0, component_index),
                learning_rate=float(architecture["learning_rate"]),
                jit_compile=True,
            )
        )
    raw_beta0 = tuple(WeightedDenseIAFTransport(config) for config in config_by_component)
    beta0_preflight = lineage.preflight_components(raw_beta0, beta_index=0, batch_size=BATCH_SIZE)
    if not all(receipt.valid for receipt in beta0_preflight):
        raise C3Error(f"beta=0 lineage preflight failed for {architecture_name}/{arm_name}/r{root_index}")
    beta0_transports = lineage.admitted_transports(0)
    row_root = output_dir / "rows" / f"{architecture_name}-{arm_name}-root-{root_index}"
    row_root.mkdir(parents=True, exist_ok=True)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    scale = math.sqrt(float(bridge.prior_variance))
    charts_by_beta: list[list[Any]] = [list(beta0_transports)]
    chart_records: list[Mapping[str, Any]] = []
    beta0_records = []
    for component_index, (transport, config) in enumerate(zip(beta0_transports, config_by_component, strict=True)):
        component_id = lineage.config.component_ids[component_index]
        checkpoint_scope = _scope(target_signature=str(bridge.target_signature), architecture=architecture_name, arm=arm_name, root_index=root_index, beta=0.0)
        checkpoint = capture_trainable_transport_checkpoint(
            transport,
            component_id=component_id,
            beta=0.0,
            bridge_signature=str(bridge.signature),
            target_signature=str(bridge.target_signature),
            parent_checkpoint_hash=None,
            update_count=0,
            checkpoint_scope=checkpoint_scope,
        )
        _write_json(row_root / f"component-{component_index}-beta-0-start.json", _json_safe(checkpoint, tf))
        restored = restore_trainable_transport_checkpoint(
            checkpoint,
            expected_context={
                "component_id": component_id,
                "beta": 0.0,
                "bridge_signature": str(bridge.signature),
                "target_signature": str(bridge.target_signature),
                "checkpoint_scope": checkpoint["checkpoint_scope"],
            },
        )
        beta0_records.append({"component_id": component_id, "checkpoint": checkpoint, "transport": restored, "preflight": beta0_preflight[component_index].payload()})
    beta0_diagnostic_latent = tf.random.stateless_normal([OVERLAP_CHAINS, int(bridge.parameter_dim)], tf.constant(_seed(tf, OVERLAP_ROOTS[root_index], architecture_index, arm_index, 0), tf.int32), dtype=tf.float64)
    for item in beta0_records:
        diagnostic = pullback_gaussianization_diagnostic(item["transport"], bridge, beta=0.0, latent=beta0_diagnostic_latent)
        item["diagnostic"] = _diagnostic_payload(diagnostic, tf)

    beta0_by_component = {item["component_id"]: item for item in beta0_records}
    for beta_index, beta in enumerate((0.5, 1.0), start=1):
        beta_transports = []
        for component_index, config in enumerate(config_by_component):
            component_id = lineage.config.component_ids[component_index]
            parents = lineage.branch_parent_indices(beta_index)
            parent_index = parents[component_index]
            if beta_index == 1 and parent_index == -1:
                raw = WeightedDenseIAFTransport(
                    WeightedNeuTraConfig(
                        dimension=int(bridge.parameter_dim),
                        hidden_layers=config.hidden_layers,
                        stages=config.stages,
                        activation=config.activation,
                        initialization_scale=config.initialization_scale,
                        initialization_seed=lineage.component_seed(beta_index, component_index, role=1),
                        learning_rate=config.learning_rate,
                        jit_compile=True,
                    )
                )
                parent_hash = None
                ref_center, ref_scale = center, scale
            else:
                parent_record = chart_records[-1][component_index] if chart_records else beta0_records[component_index]
                raw = restore_trainable_transport_checkpoint(parent_record["checkpoint"])
                parent_hash = str(parent_record["checkpoint"]["checkpoint_hash"])
                ref_center, ref_scale = None, None
            checkpoint_scope = _scope(target_signature=str(bridge.target_signature), architecture=architecture_name, arm=arm_name, root_index=root_index, beta=beta)
            seed_root = ROOTS[root_index]
            preflight_seed = _seed(tf, seed_root, beta_index, component_index, 99 if parent_index != -1 else 199)
            training_seed = _seed(tf, TRAINING_ROOT, architecture_index, arm_index, root_index, beta_index, component_index)
            result = _train_chart(
                tf,
                bridge,
                transport=raw,
                config=config,
                component_id=component_id,
                beta=beta,
                preflight_seed=preflight_seed,
                training_seed=training_seed,
                reference_center=ref_center,
                reference_scale=ref_scale,
                parent_checkpoint_hash=parent_hash,
                checkpoint_scope=checkpoint_scope,
                row_dir=row_root / f"component-{component_index}",
                target_signature=str(bridge.target_signature),
                bridge_signature=str(bridge.signature),
                capture=capture_trainable_transport_checkpoint,
                restore=restore_trainable_transport_checkpoint,
                prepare=prepare_transport_initialization,
                trainer_class=IndependentTemperedReverseKLTrainer,
                prepared_class=PreparedTransportInitialization,
            )
            beta_transports.append(result)
        chart_records.append(beta_transports)
        charts_by_beta.append([item["transport"] for item in beta_transports])
    beta1_latent = tf.random.stateless_normal([OVERLAP_CHAINS, int(bridge.parameter_dim)], tf.constant(_seed(tf, OVERLAP_ROOTS[root_index], architecture_index, arm_index, 11), tf.int32), dtype=tf.float64)
    state_levels = []
    for level_index, charts in enumerate(charts_by_beta):
        half = OVERLAP_CHAINS // 2
        latent_a = beta1_latent[:half]
        latent_b = beta1_latent[half:]
        physical_a = charts[0].forward_and_logdet(latent_a)[0]
        physical_b = charts[1].forward_and_logdet(latent_b)[0]
        state_levels.append(tf.concat((physical_a, physical_b), axis=0))
    state = tf.stack(state_levels, axis=0)
    from bayesfilter.inference.tempered_transitions_tf import ProperBridgeReplicaExchange, proper_swap_log_ratio
    exchange = ProperBridgeReplicaExchange(bridge, BETAS)
    evaluated = exchange.evaluate(state)
    ratios = tuple(proper_swap_log_ratio(evaluated["cross_values"], index, index + 1) for index in range(len(BETAS) - 1))
    acceptance = tuple(tf.reduce_mean(tf.minimum(tf.ones_like(ratio), tf.exp(tf.minimum(ratio, tf.zeros_like(ratio))))) for ratio in ratios)
    finite = bool(tf.reduce_all(tf.math.is_finite(evaluated["cross_values"])).numpy()) and bool(tf.reduce_all(evaluated["valid_at_temperature"]).numpy())
    if not finite:
        raise C3Error(f"overlap bridge evaluation invalid for {architecture_name}/{arm_name}/r{root_index}")
    self_latent = tf.stack(
        [tf.random.stateless_normal([RELIABILITY_CHAINS, int(bridge.parameter_dim)], tf.constant(_seed(tf, OVERLAP_ROOTS[root_index], architecture_index, arm_index, 30 + component_index), tf.int32), dtype=tf.float64) for component_index in range(2)],
        axis=0,
    )
    cross_physical = tf.stack([charts_by_beta[-1][index].forward_and_logdet(self_latent[index])[0] for index in range(2)], axis=0)
    def score_fn(physical: Any) -> Any:
        _value, score, status = bridge.value_score_status(
            physical, tf.constant(1.0, tf.float64)
        )
        valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
        if not bool(tf.reduce_all(valid).numpy()):
            raise C3Error("reliability score bank contains an invalid target row")
        return score
    reliability = screen_transport_reliability(
        charts_by_beta[-1],
        component_ids=lineage.config.component_ids,
        self_latent_bank=self_latent,
        cross_physical_bank=cross_physical,
        reference_points=reference_points,
        declared_points=declared_points,
        physical_score_fn=score_fn,
        maximum_condition_number=1.0e8,
    )
    if not reliability.passed:
        raise C3Error(f"learned-map reliability failed for {architecture_name}/{arm_name}/r{root_index}")
    allocator = _memory_info(tf, device_name)
    peak = int(allocator.get("peak", 4 * 1024**3 + 1))
    if peak > 4 * 1024**3:
        raise C3Error(f"allocator cap exceeded for {architecture_name}/{arm_name}/r{root_index}")
    beta1_physical = cross_physical
    distance = tf.linalg.norm(tf.reduce_mean(beta1_physical[0], axis=0) - tf.reduce_mean(beta1_physical[1], axis=0))
    record = {
        "status": "PASS_C3_ROW",
        "architecture": {"name": architecture_name, "hidden_layers": list(architecture["hidden_layers"]), "activation": architecture["activation"], "learning_rate": architecture["learning_rate"]},
        "arm": dict(arm),
        "root_index": int(root_index),
        "batch_size": BATCH_SIZE,
        "updates_per_positive_beta": TRAIN_UPDATES,
        "lineage": lineage.manifest_payload(),
        "beta0": beta0_records,
        "beta05": chart_records[0],
        "beta1": chart_records[1],
        "overlap": {"finite": finite, "swap_log_ratio_means": [tf.reduce_mean(ratio) for ratio in ratios], "swap_acceptance_means": acceptance, "cross_values_shape": evaluated["cross_values"].shape.as_list()},
        "reliability": reliability.payload(),
        "allocator": allocator,
        "beta1_chart_mean_distance": distance,
        "target_signature": str(bridge.target_signature),
        "bridge_signature": str(bridge.signature),
        "principal_sqrt_backend": PRINCIPAL_SQRT_BACKEND,
        "jit_compile": True,
    }
    safe = _json_safe(record, tf)
    safe["row_hash"] = _stable_hash(safe)
    _write_json(row_root / "row-result.json", safe)
    return safe


def _prepare_gpu_environment() -> Mapping[str, Any]:
    visible_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_before is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("BAYESFILTER_GPU_ID", DEFAULT_GPU_ID)
    growth_before = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")
    if growth_before is None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""), "selection_policy": "repository_default_single_gpu_no_idle_probe"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-protocol", action="store_true")
    return parser.parse_args()


def _protocol() -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "c3-lineage-overlap",
        "betas": list(BETAS),
        "batch_size": BATCH_SIZE,
        "updates_per_positive_beta": TRAIN_UPDATES,
        "overlap_chains": OVERLAP_CHAINS,
        "reliability_chains": RELIABILITY_CHAINS,
        "architectures": [dict(row, hidden_layers=list(row["hidden_layers"])) for row in ARCHITECTURES],
        "arms": [dict(row, restart_indices=list(row["restart_indices"])) for row in ARMS],
        "roots": [list(root) for root in ROOTS],
        "principal_sqrt_backend": PRINCIPAL_SQRT_BACKEND,
        "role": "calibration_overlap_diagnostic_only",
    }


def main() -> int:
    args = _parse_args()
    if args.print_protocol:
        print(json.dumps(_protocol(), sort_keys=True, indent=2))
        return 0
    if args.output_dir is None:
        raise C3Error("--output-dir is required")
    gpu_environment = _prepare_gpu_environment()
    if not _truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
        raise C3Error("C3 requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise C3Error("C3 requires one explicitly visible GPU")
    c2_prerequisites = _check_c2_prerequisites()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise C3Error(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise C3Error("C3 requires exactly one visible logical GPU")
    device_name = str(logical_gpus[0].name)
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend=PRINCIPAL_SQRT_BACKEND)
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise C3Error("q=20 target signature changed after C3 freeze")
    declared_points, map_receipt = _map_representatives(tf, str(bridge.target_signature))
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    dimension = int(bridge.parameter_dim)
    reference_points = tf.concat((center[tf.newaxis, :], center[tf.newaxis, :] + 4.0 * tf.eye(dimension, dtype=tf.float64), center[tf.newaxis, :] - 4.0 * tf.eye(dimension, dtype=tf.float64)), axis=0)
    route_paths = (ROOT / "bayesfilter/inference/tempered_target_tf.py", ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py", ROOT / "bayesfilter/inference/tempered_lineage_tf.py", ROOT / "bayesfilter/inference/tempered_transitions_tf.py")
    route_scan = _static_scan(route_paths)
    if not route_scan["passed"]:
        raise C3Error(f"forbidden row-mapping/pfor token: {route_scan}")
    row_records = []
    failures = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for arm_index, arm in enumerate(ARMS):
            for root_index in range(len(ROOTS)):
                if time.monotonic() - started + 120.0 >= MATERIAL_CAP_SECONDS:
                    raise C3Error("C3 material cap exhausted")
                _reset_memory(tf, device_name)
                try:
                    row_records.append(_run_row(tf, bridge, architecture=architecture, architecture_index=architecture_index, arm=arm, arm_index=arm_index, root_index=root_index, output_dir=output_dir, device_name=device_name, declared_points=declared_points, reference_points=reference_points, started=started))
                except Exception as exc:
                    failures.append({"architecture": architecture["name"], "arm": arm["name"], "root_index": root_index, "error_type": type(exc).__name__, "error": str(exc)})
    summary = []
    for architecture in ARCHITECTURES:
        for arm in ARMS:
            rows = [row for row in row_records if row["architecture"]["name"] == architecture["name"] and row["arm"]["name"] == arm["name"]]
            summary.append({"architecture": architecture["name"], "arm": arm["name"], "successful_roots": len(rows), "hard_valid_on_both_roots": len(rows) == len(ROOTS), "overlap_acceptance_means": [[float(value) for value in row["overlap"]["swap_acceptance_means"]] for row in rows]})
    passed = len(failures) == 0 and len(row_records) == len(ARCHITECTURES) * len(ARMS) * len(ROOTS)
    manifest: Mapping[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_PHASE8_C3_LINEAGE_OVERLAP" if passed else "FAIL_PHASE8_C3_LINEAGE_OVERLAP",
        "role": "calibration_overlap_diagnostic_only",
        "protocol": _protocol(),
        "command": sys.argv,
        "output_dir": str(output_dir),
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _git(("git", "status", "--porcelain")),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": __import__("tensorflow_probability").__version__,
        "target_signature": str(bridge.target_signature),
        "bridge_signature": str(bridge.signature),
        "properness_receipt": bridge.properness_receipt.payload(),
        "principal_sqrt_backend": PRINCIPAL_SQRT_BACKEND,
        "jit_compile": True,
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "logical_gpus": [str(item.name) for item in logical_gpus],
        "memory_policy": memory_policy,
        "gpu_environment": gpu_environment,
        "gpu_snapshot_before": _nvidia_snapshot(),
        "gpu_snapshot_after": _nvidia_snapshot(),
        "map_representatives": map_receipt,
        "c2_prerequisites": c2_prerequisites,
        "route_scan": route_scan,
        "rows": row_records,
        "failures": failures,
        "summary": summary,
        "hard_screen": {"all_rows_completed": passed, "failure_count": len(failures)},
        "budget": {"material_cap_seconds": MATERIAL_CAP_SECONDS, "wall_time_seconds": time.monotonic() - started},
        "source_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in (*route_paths, Path(__file__).resolve(), PLAN, MASTER_PLAN)},
        "wall_time_seconds": time.monotonic() - started,
        "nonclaims": ["overlap and lineage calibration only", "no mode-discovery, whitening, posterior, HMC, superiority, or scaling claim"],
    }
    safe = _json_safe(manifest, tf)
    safe["manifest_hash"] = _stable_hash(safe)
    _write_json(output_dir / "run_manifest.json", safe)
    print(json.dumps({"status": safe["status"], "successful_rows": len(row_records), "failed_rows": len(failures), "wall_time_seconds": safe["wall_time_seconds"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
