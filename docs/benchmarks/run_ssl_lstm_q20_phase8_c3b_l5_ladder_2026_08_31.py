#!/usr/bin/env python3
"""Run the bounded C3B L5 pure-versus-single-restart calibration."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-subplan-2026-08-31.md"
C2_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/screen/attempt-02-eight-rows/run_manifest.json"
C3_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/attempt-01/run_manifest.json"
C3_REPAIR_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/diversity-repair-2026-08-31/attempt-01/run_manifest.json"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_BACKEND = "tensorflow_eigh_strict"
SCHEMA = "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_c3b_l5_ladder.v1"
DEFAULT_GPU_ID = "0"
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BATCH_SIZE = 32
TRAIN_UPDATES = 16
OVERLAP_CHAINS = 64
DIVERSITY_CHAINS = 256
ALLOCATOR_CAP_BYTES = 4 * 1024**3
MATERIAL_CAP_SECONDS = 5000.0
ROOTS = ((20260831, 15001), (20260831, 15002))
TRAINING_ROOT = (20260831, 25001)
OVERLAP_ROOTS = ((20260831, 45001), (20260831, 45002))
ARCHITECTURES = (
    {"name": "compact-high", "hidden_layers": (16, 16), "activation": "tanh", "learning_rate": 1.0e-3},
    {"name": "compact-low", "hidden_layers": (16, 16), "activation": "tanh", "learning_rate": 5.0e-4},
)
ARMS = (
    {"name": "pure-continuation", "discovery_arm": "pure_continuation", "restart_indices": ()},
    {"name": "positive-branching", "discovery_arm": "positive_temperature_branching", "restart_indices": (1,)},
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

# The C3 runner contains only standard-library definitions at import time.  Its
# checkpoint trainer and artifact helpers are reused so C3B cannot silently
# diverge from the already-audited training semantics.
c3 = importlib.import_module("run_ssl_lstm_q20_phase8_c3_lineage_overlap_2026_08_30")


class C3BError(RuntimeError):
    pass


def _check_prerequisites() -> Mapping[str, Any]:
    required = (C2_MANIFEST, C3_MANIFEST, C3_REPAIR_MANIFEST)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise C3BError(f"required preceding manifests are missing: {missing}")
    payloads = {path: __import__("json").loads(path.read_text(encoding="utf-8")) for path in required}
    c2, c3_manifest, repair = (payloads[path] for path in required)
    checks = (
        c2.get("status") == "PASS_PHASE8_C2_STRICT_CALIBRATION",
        c3_manifest.get("status") == "PASS_PHASE8_C3_LINEAGE_OVERLAP",
        repair.get("status") == "PASS_PHASE8_C3_DIVERSITY_REPAIR",
        c2.get("target_signature") == EXPECTED_TARGET_SIGNATURE,
        c3_manifest.get("target_signature") == EXPECTED_TARGET_SIGNATURE,
        repair.get("target_signature") == EXPECTED_TARGET_SIGNATURE,
        c2.get("principal_sqrt_backend") == EXPECTED_BACKEND,
        c3_manifest.get("principal_sqrt_backend") == EXPECTED_BACKEND,
        repair.get("principal_sqrt_backend") == EXPECTED_BACKEND,
    )
    if not all(checks):
        raise C3BError("preceding C2/C3 manifests do not satisfy the frozen prerequisites")
    return {
        "c2": {"path": str(C2_MANIFEST.relative_to(ROOT)), "sha256": c3._sha256(C2_MANIFEST), "status": c2["status"]},
        "c3": {"path": str(C3_MANIFEST.relative_to(ROOT)), "sha256": c3._sha256(C3_MANIFEST), "status": c3_manifest["status"]},
        "c3_diversity_repair": {"path": str(C3_REPAIR_MANIFEST.relative_to(ROOT)), "sha256": c3._sha256(C3_REPAIR_MANIFEST), "status": repair["status"]},
    }


def _scope(*, architecture: str, arm: str, root_index: int, beta: float) -> Mapping[str, Any]:
    return {
        "data_identity": f"ssl-lstm-q20:{EXPECTED_TARGET_SIGNATURE}",
        "dtype": "float64",
        "backend": "tensorflow_tfp_gpu",
        "jit_compile": True,
        "principal_sqrt_backend": EXPECTED_BACKEND,
        "tf32_execution_enabled": True,
        "ladder": list(BETAS),
        "training_seed_derivation": {
            "initialization_root": list(ROOTS[root_index]),
            "training_root": list(TRAINING_ROOT),
            "overlap_root": list(OVERLAP_ROOTS[root_index]),
            "folds": {"architecture": architecture, "arm": arm, "root_index": root_index, "beta": beta},
        },
        "validation_bank_ids": [f"phase8-c3b-l5-{architecture}-{arm}-r{root_index}-beta{str(beta).replace('.', 'p')}-n{OVERLAP_CHAINS}"],
    }


def _summary(tf: Any, physical: Any, logdet: Any, *, seed: tuple[int, int], component_index: int, dimension: int) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(physical, tf.float64)
    determinants = tf.convert_to_tensor(logdet, tf.float64)
    expected_shape = (DIVERSITY_CHAINS, dimension)
    if values.shape != expected_shape or determinants.shape != (DIVERSITY_CHAINS,):
        raise C3BError("diversity bank shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()) or not bool(tf.reduce_all(tf.math.is_finite(determinants)).numpy()):
        raise C3BError("diversity bank contains a nonfinite map value")
    mean = tf.reduce_mean(values, axis=0)
    centered = values - mean[tf.newaxis, :]
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(DIVERSITY_CHAINS - 1, tf.float64)
    sign = values[:, 2]
    sign_fraction = tf.stack((tf.reduce_mean(tf.cast(sign > 0.0, tf.float64)), tf.reduce_mean(tf.cast(sign < 0.0, tf.float64)), tf.reduce_mean(tf.cast(sign == 0.0, tf.float64))))
    if abs(float(tf.reduce_sum(sign_fraction).numpy()) - 1.0) > 1.0e-12:
        raise C3BError("sign fractions do not partition the diversity bank")
    return {
        "component_index": component_index,
        "seed": list(seed),
        "bank_size": DIVERSITY_CHAINS,
        "mean": mean,
        "diagonal_variance": tf.linalg.diag_part(covariance),
        "covariance_trace": tf.linalg.trace(covariance),
        "covariance_frobenius_norm": tf.linalg.norm(covariance),
        "covariance": covariance,
        "sign_fraction_coordinate_2": sign_fraction,
        "logdet_mean": tf.reduce_mean(determinants),
        "logdet_rms": tf.sqrt(tf.reduce_mean(tf.square(determinants))),
    }


def _public_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in result.items() if key != "transport"}


def _run_row(tf: Any, bridge: Any, *, architecture: Mapping[str, Any], architecture_index: int, arm: Mapping[str, Any], arm_index: int, root_index: int, output_dir: Path, device_name: str, declared_points: Any, reference_points: Any, started: float) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig
    from bayesfilter.inference.tempered_lineage_tf import TemperedLineageConfig, TemperedLineageController
    from bayesfilter.inference.tempered_transport_ensemble_tf import PreparedTransportInitialization, IndependentTemperedReverseKLTrainer, capture_trainable_transport_checkpoint, prepare_transport_initialization, restore_trainable_transport_checkpoint, pullback_gaussianization_diagnostic
    from bayesfilter.inference.tempered_transitions_tf import ProperBridgeReplicaExchange, proper_swap_log_ratio, screen_transport_reliability

    architecture_name = str(architecture["name"])
    arm_name = str(arm["name"])
    lineage = TemperedLineageController(
        TemperedLineageConfig(
            betas=BETAS,
            component_ids=(f"c3b-{architecture_name}-{arm_name}-c0", f"c3b-{architecture_name}-{arm_name}-c1"),
            root_seed=ROOTS[root_index],
            discovery_arm=str(arm["discovery_arm"]),
            positive_branch_betas=(0.5,) if arm["discovery_arm"] == "positive_temperature_branching" else (),
            restart_component_indices=tuple(int(item) for item in arm["restart_indices"]),
            preflight_batch_size=BATCH_SIZE,
        ),
        bridge,
    )
    for beta_index in range(len(BETAS)):
        lineage.checkpoint(beta_index)
    base_configs = [
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
        for component_index in range(2)
    ]
    raw_beta0 = tuple(WeightedDenseIAFTransport(config) for config in base_configs)
    beta0_preflight = lineage.preflight_components(raw_beta0, beta_index=0, batch_size=BATCH_SIZE)
    if not all(receipt.valid for receipt in beta0_preflight):
        raise C3BError(f"beta=0 preflight failed for {architecture_name}/{arm_name}/r{root_index}")
    beta0_transports = lineage.admitted_transports(0)
    row_root = output_dir / "rows" / f"{architecture_name}-{arm_name}-root-{root_index}"
    row_root.mkdir(parents=True, exist_ok=True)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    scale = math.sqrt(float(bridge.prior_variance))
    beta_records: list[list[Mapping[str, Any]]] = []
    beta_transports: list[list[Any]] = [list(beta0_transports)]
    beta0_records: list[Mapping[str, Any]] = []
    for component_index, transport in enumerate(beta0_transports):
        component_id = lineage.config.component_ids[component_index]
        checkpoint = capture_trainable_transport_checkpoint(
            transport,
            component_id=component_id,
            beta=0.0,
            bridge_signature=str(bridge.signature),
            target_signature=EXPECTED_TARGET_SIGNATURE,
            parent_checkpoint_hash=None,
            update_count=0,
            checkpoint_scope=_scope(architecture=architecture_name, arm=arm_name, root_index=root_index, beta=0.0),
        )
        c3._write_json(row_root / f"component-{component_index}-beta-0-start.json", c3._json_safe(checkpoint, tf))
        restored = restore_trainable_transport_checkpoint(checkpoint, expected_context={"component_id": component_id, "beta": 0.0, "bridge_signature": str(bridge.signature), "target_signature": EXPECTED_TARGET_SIGNATURE})
        diagnostic = pullback_gaussianization_diagnostic(restored, bridge, beta=0.0, latent=tf.random.stateless_normal([OVERLAP_CHAINS, int(bridge.parameter_dim)], tf.constant(c3._seed(tf, OVERLAP_ROOTS[root_index], architecture_index, component_index), tf.int32), dtype=tf.float64))
        beta0_records.append({"component_id": component_id, "checkpoint": checkpoint, "preflight": beta0_preflight[component_index].payload(), "diagnostic": c3._diagnostic_payload(diagnostic, tf), "transport": restored})
    beta_records.append(beta0_records)

    for beta_index in range(1, len(BETAS)):
        beta = BETAS[beta_index]
        current_records: list[Mapping[str, Any]] = []
        current_transports: list[Any] = []
        parents = lineage.branch_parent_indices(beta_index)
        for component_index, base_config in enumerate(base_configs):
            component_id = lineage.config.component_ids[component_index]
            parent_index = parents[component_index]
            if parent_index == -1:
                config = WeightedNeuTraConfig(
                    dimension=base_config.dimension,
                    hidden_layers=base_config.hidden_layers,
                    stages=base_config.stages,
                    activation=base_config.activation,
                    initialization_scale=base_config.initialization_scale,
                    initialization_seed=lineage.component_seed(beta_index, component_index, role=1),
                    learning_rate=base_config.learning_rate,
                    jit_compile=True,
                )
                raw = WeightedDenseIAFTransport(config)
                parent_hash = None
                reference_center, reference_scale = center, scale
            else:
                # ``parent_index`` identifies the source component within the
                # immediately preceding beta slice; it is not a beta-list
                # index.  Continuation therefore always reads the last slice.
                parent_record = beta_records[-1][component_index]
                raw = restore_trainable_transport_checkpoint(parent_record["checkpoint"])
                config = base_config
                parent_hash = str(parent_record["checkpoint"]["checkpoint_hash"])
                reference_center, reference_scale = None, None
            result = c3._train_chart(
                tf,
                bridge,
                transport=raw,
                config=config,
                component_id=component_id,
                beta=beta,
                preflight_seed=c3._seed(tf, ROOTS[root_index], beta_index, component_index, 99 if parent_index != -1 else 199),
                # Keep stochastic training inputs paired across arms.  The arm
                # may change only the declared restart/parent policy.
                training_seed=c3._seed(tf, TRAINING_ROOT, architecture_index, root_index, beta_index, component_index),
                reference_center=reference_center,
                reference_scale=reference_scale,
                parent_checkpoint_hash=parent_hash,
                checkpoint_scope=_scope(architecture=architecture_name, arm=arm_name, root_index=root_index, beta=beta),
                row_dir=row_root / f"component-{component_index}",
                target_signature=EXPECTED_TARGET_SIGNATURE,
                bridge_signature=str(bridge.signature),
                capture=capture_trainable_transport_checkpoint,
                restore=restore_trainable_transport_checkpoint,
                prepare=prepare_transport_initialization,
                trainer_class=IndependentTemperedReverseKLTrainer,
                prepared_class=PreparedTransportInitialization,
            )
            current_records.append(result)
            current_transports.append(result["transport"])
        beta_records.append(current_records)
        beta_transports.append(current_transports)

    # Build a common physical state bank at each L5 level and evaluate the
    # complete bridge at every destination temperature.
    state_levels = []
    for beta_index, charts in enumerate(beta_transports):
        latent = tf.random.stateless_normal([OVERLAP_CHAINS, int(bridge.parameter_dim)], tf.constant(c3._seed(tf, OVERLAP_ROOTS[root_index], architecture_index, beta_index), tf.int32), dtype=tf.float64)
        half = OVERLAP_CHAINS // 2
        physical0 = charts[0].forward_and_logdet(latent[:half])[0]
        physical1 = charts[1].forward_and_logdet(latent[half:])[0]
        state_levels.append(tf.concat((physical0, physical1), axis=0))
    state = tf.stack(state_levels, axis=0)
    exchange = ProperBridgeReplicaExchange(bridge, BETAS)
    evaluated = exchange.evaluate(state)
    ratios = tuple(proper_swap_log_ratio(evaluated["cross_values"], index, index + 1) for index in range(len(BETAS) - 1))
    acceptance = tuple(tf.reduce_mean(tf.minimum(tf.ones_like(ratio), tf.exp(tf.minimum(ratio, tf.zeros_like(ratio))))) for ratio in ratios)
    finite = bool(tf.reduce_all(tf.math.is_finite(evaluated["cross_values"])).numpy()) and bool(tf.reduce_all(evaluated["valid_at_temperature"]).numpy())
    if not finite:
        raise C3BError(f"proper-bridge overlap failed for {architecture_name}/{arm_name}/r{root_index}")
    if not all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in (*ratios, *acceptance)):
        raise C3BError(f"nonfinite adjacent overlap summary for {architecture_name}/{arm_name}/r{root_index}")

    diversity_summaries = []
    for component_index, chart in enumerate(beta_transports[-1]):
        seed = c3._seed(tf, OVERLAP_ROOTS[root_index], architecture_index, component_index, 700)
        latent = tf.random.stateless_normal([DIVERSITY_CHAINS, int(bridge.parameter_dim)], tf.constant(seed, tf.int32), dtype=tf.float64)
        physical, logdet = chart.forward_and_logdet(latent)
        diversity_summaries.append(_summary(tf, physical, logdet, seed=seed, component_index=component_index, dimension=int(bridge.parameter_dim)))
    mean_distance = tf.linalg.norm(diversity_summaries[0]["mean"] - diversity_summaries[1]["mean"])
    covariance_distance = tf.linalg.norm(diversity_summaries[0]["covariance"] - diversity_summaries[1]["covariance"])
    occupancy_distance = tf.linalg.norm(diversity_summaries[0]["sign_fraction_coordinate_2"] - diversity_summaries[1]["sign_fraction_coordinate_2"])

    def score_fn(physical: Any) -> Any:
        _value, score, status = bridge.value_score_status(physical, tf.constant(1.0, tf.float64))
        valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
        if not bool(tf.reduce_all(valid).numpy()):
            raise C3BError("beta-one reliability score bank is invalid")
        return score

    self_latent = tf.stack([tf.random.stateless_normal([OVERLAP_CHAINS, int(bridge.parameter_dim)], tf.constant(c3._seed(tf, OVERLAP_ROOTS[root_index], architecture_index, 800, component_index), tf.int32), dtype=tf.float64) for component_index in range(2)], axis=0)
    cross_physical = tf.stack([beta_transports[-1][index].forward_and_logdet(self_latent[index])[0] for index in range(2)], axis=0)
    reliability = screen_transport_reliability(beta_transports[-1], component_ids=lineage.config.component_ids, self_latent_bank=self_latent, cross_physical_bank=cross_physical, reference_points=reference_points, declared_points=declared_points, physical_score_fn=score_fn, maximum_condition_number=1.0e8)
    if not reliability.passed:
        raise C3BError(f"beta-one learned-map reliability failed for {architecture_name}/{arm_name}/r{root_index}")
    allocator = c3._memory_info(tf, device_name)
    if int(allocator.get("peak", ALLOCATOR_CAP_BYTES + 1)) > ALLOCATOR_CAP_BYTES:
        raise C3BError(f"allocator cap exceeded for {architecture_name}/{arm_name}/r{root_index}")
    record = {
        "status": "PASS_C3B_ROW",
        "architecture": {"name": architecture_name, "hidden_layers": list(architecture["hidden_layers"]), "activation": architecture["activation"], "learning_rate": architecture["learning_rate"]},
        "arm": dict(arm),
        "root_index": root_index,
        "batch_size": BATCH_SIZE,
        "updates_per_positive_beta": TRAIN_UPDATES,
        "betas": list(BETAS),
        "lineage": lineage.manifest_payload(),
        "beta_records": [[_public_result(item) for item in records] for records in beta_records],
        "overlap": {"finite": finite, "cross_values_shape": evaluated["cross_values"].shape.as_list(), "swap_log_ratio_means": [tf.reduce_mean(ratio) for ratio in ratios], "swap_acceptance_means": acceptance},
        "diversity": {"summaries": diversity_summaries, "mean_distance": mean_distance, "covariance_frobenius_distance": covariance_distance, "sign_occupancy_l2_distance": occupancy_distance},
        "reliability": reliability.payload(),
        "allocator": allocator,
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "bridge_signature": str(bridge.signature),
        "principal_sqrt_backend": EXPECTED_BACKEND,
        "jit_compile": True,
        "nonclaims": ["L5 overlap and lineage calibration only", "no mode-discovery, whitening, posterior, HMC, superiority, ranking, or scaling claim"],
    }
    safe = c3._json_safe(record, tf)
    safe["row_hash"] = c3._stable_hash(safe)
    c3._write_json(row_root / "row-result.json", safe)
    return safe


def _protocol() -> Mapping[str, Any]:
    return {"schema": SCHEMA, "mode": "c3b-l5-ladder", "betas": list(BETAS), "batch_size": BATCH_SIZE, "updates_per_positive_beta": TRAIN_UPDATES, "overlap_chains": OVERLAP_CHAINS, "diversity_chains": DIVERSITY_CHAINS, "architectures": [dict(item, hidden_layers=list(item["hidden_layers"])) for item in ARCHITECTURES], "arms": [dict(item, restart_indices=list(item["restart_indices"])) for item in ARMS], "roots": [list(root) for root in ROOTS], "principal_sqrt_backend": EXPECTED_BACKEND, "role": "calibration_overlap_diagnostic_only"}


def _run(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        raise C3BError("--output-dir is required")
    if not c3._truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
        raise C3BError("C3B requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise C3BError("C3B requires one explicitly visible GPU")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise C3BError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    prerequisites = _check_prerequisites()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise C3BError("C3B requires exactly one visible logical GPU")
    device_name = str(logical_gpus[0].name)
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend=EXPECTED_BACKEND)
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise C3BError("q=20 target signature changed")
    declared_points, map_receipt = c3._map_representatives(tf, EXPECTED_TARGET_SIGNATURE)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    dimension = int(bridge.parameter_dim)
    reference_points = tf.concat((center[tf.newaxis, :], center[tf.newaxis, :] + 4.0 * tf.eye(dimension, dtype=tf.float64), center[tf.newaxis, :] - 4.0 * tf.eye(dimension, dtype=tf.float64)), axis=0)
    route_paths = (ROOT / "bayesfilter/inference/tempered_target_tf.py", ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py", ROOT / "bayesfilter/inference/tempered_lineage_tf.py", ROOT / "bayesfilter/inference/tempered_transitions_tf.py")
    route_scan = c3._static_scan(route_paths)
    if not route_scan["passed"]:
        raise C3BError(f"forbidden runtime route token: {route_scan}")
    rows = []
    failures = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for arm_index, arm in enumerate(ARMS):
            for root_index in range(len(ROOTS)):
                if time.monotonic() - started + 120.0 >= MATERIAL_CAP_SECONDS:
                    raise C3BError("C3B material cap exhausted")
                c3._reset_memory(tf, device_name)
                try:
                    rows.append(_run_row(tf, bridge, architecture=architecture, architecture_index=architecture_index, arm=arm, arm_index=arm_index, root_index=root_index, output_dir=output_dir, device_name=device_name, declared_points=declared_points, reference_points=reference_points, started=started))
                except Exception as exc:
                    failures.append({"architecture": architecture["name"], "arm": arm["name"], "root_index": root_index, "error_type": type(exc).__name__, "error": str(exc)})
    passed = len(rows) == len(ARCHITECTURES) * len(ARMS) * len(ROOTS) and not failures
    summary = []
    for architecture in ARCHITECTURES:
        for arm in ARMS:
            selected = [row for row in rows if row["architecture"]["name"] == architecture["name"] and row["arm"]["name"] == arm["name"]]
            summary.append({"architecture": architecture["name"], "arm": arm["name"], "successful_roots": len(selected), "hard_valid_on_both_roots": len(selected) == len(ROOTS), "swap_acceptance_means": [[float(value) for value in row["overlap"]["swap_acceptance_means"]] for row in selected], "diversity_mean_distances": [float(row["diversity"]["mean_distance"]) for row in selected]})
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_PHASE8_C3B_L5_LADDER" if passed else "FAIL_PHASE8_C3B_L5_LADDER",
        "role": "calibration_overlap_diagnostic_only",
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
        "gpu_snapshot_before": c3._nvidia_snapshot(),
        "gpu_snapshot_after": c3._nvidia_snapshot(),
        "prerequisites": prerequisites,
        "map_representatives": map_receipt,
        "route_scan": route_scan,
        "rows": rows,
        "failures": failures,
        "summary": summary,
        "hard_screen": {"all_rows_completed": passed, "failure_count": len(failures)},
        "budget": {"material_cap_seconds": MATERIAL_CAP_SECONDS, "wall_time_seconds": time.monotonic() - started},
        "source_hashes": {str(path.relative_to(ROOT)): c3._sha256(path) for path in (*route_paths, Path(__file__).resolve(), PLAN, C2_MANIFEST, C3_MANIFEST, C3_REPAIR_MANIFEST)},
        "wall_time_seconds": time.monotonic() - started,
        "nonclaims": ["L5 overlap and lineage calibration only", "no mode-discovery, whitening, posterior, HMC, superiority, ranking, or scaling claim"],
    }
    safe = c3._json_safe(manifest, tf)
    safe["manifest_hash"] = c3._stable_hash(safe)
    c3._write_json(output_dir / "run_manifest.json", safe)
    print(__import__("json").dumps({"status": safe["status"], "successful_rows": len(rows), "failed_rows": len(failures), "wall_time_seconds": safe["wall_time_seconds"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parsed = None
    try:
        args = argparse.ArgumentParser(description=__doc__)
        args.add_argument("--output-dir", type=Path)
        args.add_argument("--print-protocol", action="store_true")
        parsed = args.parse_args()
        if parsed.print_protocol:
            print(__import__("json").dumps(_protocol(), sort_keys=True, indent=2))
            return 0
        return _run(parsed)
    except Exception as exc:
        if parsed is not None and isinstance(parsed.output_dir, Path):
            output_dir = parsed.output_dir.expanduser().resolve()
            if output_dir.is_dir():
                try:
                    c3._write_json(output_dir / "failure.json", {"status": "FAIL_PHASE8_C3B_L5_LADDER", "error_type": type(exc).__name__, "error": str(exc), "command": sys.argv})
                except Exception:
                    pass
        print(__import__("json").dumps({"status": "FAIL_PHASE8_C3B_L5_LADDER", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
