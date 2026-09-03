#!/usr/bin/env python3
"""Replicate viable NeuTra controls and validate them with sequential HMC."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-neutra-replication-hmc-plan-2026-08-16.md"
BASE_RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-replication-hmc-2026-08-16"
TRAIN_UPDATES = 3000
BATCH_SIZE = 4096
SELECTION_COUNT = 65536
AUDIT_COUNT = 131072
TRAIN_SEEDS = (10, 11, 12)
HMC_CHAINS = 4


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--time-cap", type=float, default=7200.0)
    return parser.parse_args()


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("neutra_replication_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base target runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class AnalyticControlAdapter:
    """Graph-native exact-law score adapter for one Gaussian/banana target."""

    def __init__(self, tf_module: Any, model: Mapping[str, Any], scope: str) -> None:
        self.tf = tf_module
        self.model = model
        self.target_scope = str(scope)
        self.parameter_dim = int(model["dimension"])
        self.supports_retained_draw_batch = False
        self.supports_retained_flat_batch = False

    def adapter_signature(self) -> str:
        return _payload_hash({"schema": "bayesfilter.neutra.analytic_control_adapter.v1", "target": self.model["manifest"], "scope": self.target_scope})

    def value_score_capability(self) -> Any:
        from bayesfilter.inference.posterior_adapter import ValueScoreCapability
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="tensorflow_graph_native_analytic_control",
            evidence_path="docs/benchmarks/run_neutra_replication_hmc_campaign_2026_08_16.py",
            target_scope=self.target_scope,
            nonclaims=("exact-law control adapter", "no general posterior authority"),
        )

    def _value_score(self, theta: Any) -> tuple[Any, Any]:
        tfm = self.tf
        rows = tfm.cast(tfm.convert_to_tensor(theta), tfm.float64)
        rank = rows.shape.rank
        if rank == 1:
            rows = rows[tfm.newaxis, :]
            scalar = True
        elif rank == 2:
            scalar = False
        else:
            raise ValueError("analytic control theta must be rank 1 or 2")
        if self.model["name"] == "gaussian":
            mean = tfm.convert_to_tensor(self.model["manifest"]["mean"], tfm.float64)
            factor = tfm.convert_to_tensor(self.model["manifest"]["factor"], tfm.float64)
            latent = tfm.transpose(tfm.linalg.triangular_solve(factor, tfm.transpose(rows - mean), lower=True))
            value = -0.5 * tfm.reduce_sum(tfm.square(latent), axis=1)
            score = -tfm.transpose(tfm.linalg.triangular_solve(tfm.transpose(factor), tfm.transpose(latent), lower=False))
        else:
            curvature = tfm.cast(self.model["manifest"]["curvature"], tfm.float64)
            latent = tfm.concat((rows[:, :1], rows[:, 1:2] - curvature * (tfm.square(rows[:, :1]) - 1.0), rows[:, 2:]), axis=1)
            value = -0.5 * tfm.reduce_sum(tfm.square(latent), axis=1)
            score = -latent
            score = tfm.concat((score[:, :1] + 2.0 * curvature * rows[:, :1] * latent[:, 1:2], score[:, 1:]), axis=1)
        if scalar:
            return value[0], score[0]
        return value, score

    def log_prob_and_grad(self, theta: Any) -> tuple[Any, Any]:
        return self._value_score(theta)

    def target_status_telemetry(self, theta: Any) -> Mapping[str, Any]:
        tfm = self.tf
        rows = tfm.convert_to_tensor(theta, tfm.float64)
        batch = tfm.shape(rows)[0] if rows.shape.rank == 2 else tfm.constant(1, tfm.int32)
        return {
            "status_code": tfm.zeros((batch,), tfm.int32),
            "valid_pre_regularized_score": tfm.ones((batch,), tfm.bool),
        }


def _state_hash(tf_module: Any, transport: Any) -> str:
    digest = hashlib.sha256()
    for variable in transport.trainable_variables:
        digest.update(bytes(tf_module.io.serialize_tensor(variable.read_value()).numpy()))
    return digest.hexdigest()


def _config(model: Mapping[str, Any], *, seed: int, permutation: str, hidden: int = 32) -> Any:
    from bayesfilter.inference.neutra_weighted_training import WeightedNeuTraConfig
    stages = int(model["stages"])
    return WeightedNeuTraConfig(
        dimension=int(model["dimension"]), hidden_layers=(int(hidden), int(hidden)), stages=stages,
        activation="elu", s_max=1.0, stage_s_max=tuple(model["stage_caps"]),
        stage_unbounded_scale_linear=(True,) + (False,) * (stages - 1),
        permutation_policy=str(permutation), initialization_scale=0.02,
        initialization_seed=(20260816, 50000 + int(seed)), learning_rate=1.0e-3,
        gradient_clip_norm=10.0, jit_compile=True,
    )


def _train(tf_module: Any, model: Mapping[str, Any], *, seed: int, rate: float, permutation: str) -> tuple[Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_weighted_training import MatchedReverseKLNeuTraTrainer
    config = _config(model, seed=seed, permutation=permutation)
    trainer = MatchedReverseKLNeuTraTrainer(config, model["target_log_prob"])
    selection = tf_module.random.stateless_normal((SELECTION_COUNT, int(model["dimension"])), seed=(20260816 + seed, 51001), dtype=tf_module.float64)
    best_loss = float(_selection_loss(tf_module, trainer.transport, model, selection).numpy())
    best_state = tuple(tf_module.identity(v) for v in trainer.variables)
    best_update = 0
    clipped = 0
    terminal = best_loss
    for update in range(1, TRAIN_UPDATES + 1):
        frac = float(update) / TRAIN_UPDATES
        multiplier = 1.0 if frac < 0.60 else 0.1 if frac < 0.85 else 0.01
        trainer.optimizer.learning_rate.assign(float(rate) * multiplier)
        latent = tf_module.random.stateless_normal((BATCH_SIZE, int(model["dimension"])), seed=(20260816 + seed, 52000 + update), dtype=tf_module.float64)
        step = trainer.train_step(latent)
        clipped += int(bool(step.clipping_applied.numpy()))
        if update % 250 == 0 or update == TRAIN_UPDATES:
            terminal = float(_selection_loss(tf_module, trainer.transport, model, selection).numpy())
            if terminal < best_loss:
                best_loss = terminal
                best_update = update
                best_state = tuple(tf_module.identity(v) for v in trainer.variables)
    for variable, value in zip(trainer.variables, best_state, strict=True):
        variable.assign(value)
    return trainer.transport, {"selected_loss": best_loss, "terminal_loss": terminal, "selected_update": best_update, "clipped_updates": clipped, "executed_updates": TRAIN_UPDATES}


def _selection_loss(tf_module: Any, transport: Any, model: Mapping[str, Any], latent: Any) -> Any:
    physical, logdet = transport.forward_and_logdet(latent)
    return tf_module.reduce_mean(-model["target_log_prob"](physical) - logdet)


def _audit(tf_module: Any, transport: Any, model: Mapping[str, Any], *, count: int, seed: tuple[int, int]) -> Mapping[str, Any]:
    latent = tf_module.random.stateless_normal((int(count), int(model["dimension"])), seed=seed, dtype=tf_module.float64)
    physical, logdet = transport.forward_and_logdet(latent)
    exact = model["exact_latent"](physical)
    log_ratio = model["target_log_prob"](physical) + logdet + 0.5 * tf_module.reduce_sum(tf_module.square(latent), axis=1)
    normalized = tf_module.nn.softmax(log_ratio)
    ess_fraction = tf_module.math.reciprocal(tf_module.reduce_sum(tf_module.square(normalized))) / tf_module.cast(count, tf_module.float64)
    n = tf_module.cast(tf_module.shape(exact)[0], tf_module.float64)
    z = tf_module.constant(3.2905267314919255, tf_module.float64)
    def interval(values: Any, truth: Any) -> Mapping[str, Any]:
        values = tf_module.convert_to_tensor(values, tf_module.float64)
        truth = tf_module.convert_to_tensor(truth, tf_module.float64)
        mean = tf_module.reduce_mean(values, axis=0)
        se = tf_module.math.reduce_std(values, axis=0) / tf_module.sqrt(n)
        passed = tf_module.logical_and(truth >= mean - z * se, truth <= mean + z * se)
        return {"estimate": mean, "exact": truth, "standard_error": se, "passed": passed, "all_passed": tf_module.reduce_all(passed)}
    screens = {
        "coordinate_mean": interval(exact, tf_module.zeros((int(model["dimension"]),), tf_module.float64)),
        "coordinate_second_moment": interval(tf_module.square(exact), tf_module.ones((int(model["dimension"]),), tf_module.float64)),
        "adjacent_cross_moment": interval(exact[:, :-1] * exact[:, 1:], tf_module.zeros((int(model["dimension"]) - 1,), tf_module.float64)),
    }
    return {"sample_count": int(count), "screens": screens, "passed": tf_module.reduce_all(tf_module.stack([v["all_passed"] for v in screens.values()])), "importance_ess_fraction": ess_fraction, "maximum_normalized_importance_weight": tf_module.reduce_max(normalized), "log_target_to_proposal_ratio_stddev": tf_module.math.reduce_std(log_ratio)}


def _bind_frozen(transport: Any, tf_module: Any, model: Mapping[str, Any], state_hash: str) -> None:
    transport.bind_frozen_identity({"checkpoint_sha256": state_hash, "training_state_hash": state_hash, "transport_tensor_hash": state_hash})


def _initial_bank(tf_module: Any, dimension: int) -> Any:
    rows = tf_module.random.stateless_normal((HMC_CHAINS, int(dimension)), seed=(20260816, 54001), dtype=tf_module.float64)
    return rows


def _tune_hmc(base_adapter: Any, transport: Any, initial: Any, out: Path, scope: str) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )
    cfg = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.05, leapfrog_grid=(3, 5, 10, 15, 20, 25), chain_count=HMC_CHAINS,
        initial_state_bank=tuple(tuple(float(x) for x in row) for row in initial.numpy().tolist()),
        target_accept_prob=0.70, acceptance_band=(0.55, 0.90), repair_band=(0.40, 0.95),
        selection_policy="acceptance_target_distance", selection_replications=1, fixed_grid_fallback_acceptance_max=0.95,
        budget_schedule=(32, 64, 128), tune_num_results=16, screen_num_results=64, screen_num_burnin_steps=16,
        verification_num_results=2000, verification_num_burnin_steps=64, require_modern_rank_normalized_verification=True,
        verification_coordinate_system="hmc_coordinates", verification_min_retained_results_per_chain=2000,
        tune_seed_base=(20260816, 55001), screen_seed_base=(20260816, 56001), verification_seed_base=(20260816, 57001),
        chain_execution_mode="tf_function", use_xla=True, target_scope=scope,
        tuning_policy=FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY, output_filename="tuning_result.json",
    )
    return tune_fixed_transport_hmc_kernel(base_adapter=base_adapter, fixed_transport=transport, initial_position=initial[0], config=cfg, output_dir=out).payload()


def _archive_callback(root: Path, tf_module: Any, label: str):
    def archive(*, stage: str, chunk_index: Any, latent_samples: Any, model_samples: Any, seed: Any, cumulative: bool) -> Mapping[str, Any]:
        suffix = "cumulative" if cumulative else f"chunk-{int(chunk_index):03d}"
        stage_root = root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        latent_path = stage_root / f"{label}-{suffix}-latent.tftensor"
        model_path = stage_root / f"{label}-{suffix}-model.tftensor"
        latent_bytes = bytes(tf_module.io.serialize_tensor(latent_samples).numpy())
        model_bytes = bytes(tf_module.io.serialize_tensor(model_samples).numpy())
        latent_path.write_bytes(latent_bytes)
        model_path.write_bytes(model_bytes)
        return {
            "stage": stage,
            "chunk_index": chunk_index,
            "cumulative": bool(cumulative),
            "seed": None if seed is None else list(seed),
            "latent_path": latent_path.as_posix(),
            "latent_sha256": hashlib.sha256(latent_bytes).hexdigest(),
            "model_path": model_path.as_posix(),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "sample_shape": [int(x) for x in latent_samples.shape],
        }
    return archive


def _hmc_summary(tf_module: Any, result: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if not str(key).startswith("private_")
    }


def _run_hmc(tf_module: Any, model: Mapping[str, Any], transport: Any, base_adapter: Any, initial: Any, tuning: Mapping[str, Any], out: Path, scope: str, cap: float) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, run_sequential_neutra_hmc
    kernel = tuning.get("final_kernel_payload")
    if not isinstance(kernel, Mapping) or tuning.get("passed") is not True:
        return {"status": "tuning_failed", "tuning": tuning, "passed": False}
    leapfrog = int(kernel["num_leapfrog_steps"])
    if leapfrog < 2:
        raise RuntimeError("L=1 is forbidden")
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_value_score_adapter

    adapter = build_fixed_transport_value_score_adapter(
        base_adapter=base_adapter,
        fixed_transport=transport,
        target_scope=f"{scope}:frozen",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
    )
    started = time.perf_counter()
    cfg = SequentialNeuTraHMCConfig(
        step_size=float(kernel["step_size"]),
        num_leapfrog_steps=leapfrog,
        warmup_seed=(20260816, 58001),
        retained_seed=(20260816, 58002),
        warmup_chunk_results=500,
        warmup_min_results=2000,
        warmup_check_window_results=1000,
        warmup_max_results=10000,
        retained_chunk_results=500,
        retained_min_results=2000,
        retained_max_results=10000,
        retained_rhat_max=1.01,
        minimum_chain_count=HMC_CHAINS,
        jit_compile=True,
    )

    def transform(samples: Any) -> Any:
        shape = tf_module.shape(samples)
        flat = tf_module.reshape(samples, (-1, int(model["dimension"])))
        physical = transport.forward_batch(flat)
        return tf_module.reshape(physical, shape)

    def retained_diagnostic(samples: Any) -> Mapping[str, Any]:
        from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
        from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_bulk_tail_ess
        rhat = rank_normalized_split_rhat_summary(samples, rhat_max=1.01)
        ess = rank_normalized_bulk_tail_ess(samples)
        min_bulk = float(tf_module.reduce_min(tf_module.convert_to_tensor(ess["bulk"], tf_module.float64)).numpy())
        min_tail = float(tf_module.reduce_min(tf_module.convert_to_tensor(ess["tail"], tf_module.float64)).numpy())
        passed = bool(rhat["passed"] and min_bulk >= 400.0 and min_tail >= 400.0)
        return {"passed": passed, "rhat": rhat, "ess": ess, "min_bulk_ess": min_bulk, "min_tail_ess": min_tail}

    result = run_sequential_neutra_hmc(
        adapter=adapter,
        initial_state=initial,
        model_transform=transform,
        parameter_names=tuple(f"z_{i}" for i in range(int(model["dimension"]))),
        config=cfg,
        retained_diagnostic_fn=retained_diagnostic,
        archive_callback=_archive_callback(out / "archive", tf_module, scope.replace(":", "-")),
    )
    retained_z = result["private_retained_z"]
    retained_model = result["private_retained_raw"]
    exact = model["exact_latent"](tf_module.reshape(retained_model, (-1, int(model["dimension"]))))
    n = tf_module.cast(tf_module.shape(exact)[0], tf_module.float64)
    zcrit = tf_module.constant(3.2905267314919255, tf_module.float64)
    def interval(values: Any, truth: Any) -> Mapping[str, Any]:
        values = tf_module.convert_to_tensor(values, tf_module.float64)
        truth = tf_module.convert_to_tensor(truth, tf_module.float64)
        mean = tf_module.reduce_mean(values, axis=0)
        se = tf_module.math.reduce_std(values, axis=0) / tf_module.sqrt(n)
        passed = tf_module.logical_and(truth >= mean - zcrit * se, truth <= mean + zcrit * se)
        return {"estimate": mean, "exact": truth, "standard_error": se, "passed": passed, "all_passed": tf_module.reduce_all(passed)}
    screens = {
        "coordinate_mean": interval(exact, tf_module.zeros((int(model["dimension"]),), tf_module.float64)),
        "coordinate_second_moment": interval(tf_module.square(exact), tf_module.ones((int(model["dimension"]),), tf_module.float64)),
        "adjacent_cross_moment": interval(exact[:, :-1] * exact[:, 1:], tf_module.zeros((int(model["dimension"]) - 1,), tf_module.float64)),
    }
    post_passed = bool(tf_module.reduce_all(tf_module.stack([v["all_passed"] for v in screens.values()])).numpy())
    payload = {"schema": "bayesfilter.neutra.sequential_hmc_result.v1", **_hmc_summary(tf_module, result), "post_hmc_exact_law": {"sample_count": int(exact.shape[0]), "screens": screens, "passed": post_passed}}
    _write(out / "sequential_result.json", payload)
    payload["post_hmc_exact_law"] = {**payload["post_hmc_exact_law"], "passed": post_passed}
    return payload


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if not PLAN.is_file() or not BASE_RUNNER.is_file():
        raise FileNotFoundError("plan or base runner missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical}")
    base = _load_base()
    targets = (("gaussian", "baseline", 1.0e-3, "full_reverse"), ("banana", "root_preserving", 5.0e-4, "root_preserving_reverse"))
    progress = {"schema": "bayesfilter.neutra.replication_hmc_progress.v1", "phase": "started", "completed_training_cells": 0}
    _write(output / "progress.json", progress)
    result_targets = {}
    for target_name, arm, rate, permutation in targets:
        model = base._model(tf, target_name)
        base_adapter = AnalyticControlAdapter(tf, model, f"analytic_control:{target_name}:base_v1")
        replications = []
        for seed in TRAIN_SEEDS:
            if time.perf_counter() - started > float(args.time_cap):
                raise TimeoutError("replication/HMC campaign time cap exhausted during training")
            transport, training = _train(tf, model, seed=seed, rate=rate, permutation=permutation)
            audit = _audit(tf, transport, model, count=AUDIT_COUNT, seed=(20260816, 59000 + seed))
            passed = bool(tf.convert_to_tensor(audit["passed"]).numpy())
            state_hash = _state_hash(tf, transport)
            replications.append({"seed": seed, "training": training, "audit": audit, "passed": passed, "state_hash": state_hash})
            progress.update({"phase": f"{target_name}_replication", "completed_training_cells": sum(len(v.get("replications", [])) for v in result_targets.values()) + len(replications)})
            _write(output / "progress.json", progress)
        target_result = {"replications": replications, "replication_passed": all(row["passed"] for row in replications)}
        if target_result["replication_passed"]:
            seed = TRAIN_SEEDS[-1]
            transport, training = _train(tf, model, seed=seed, rate=rate, permutation=permutation)
            state_hash = _state_hash(tf, transport)
            _bind_frozen(transport, tf, model, state_hash)
            adapter = __import__("bayesfilter.inference.fixed_transport_hmc_mechanics_tf", fromlist=["build_fixed_transport_value_score_adapter"]).build_fixed_transport_value_score_adapter(base_adapter=base_adapter, fixed_transport=transport, target_scope=f"analytic_control:{target_name}:frozen:{state_hash}", evidence_path=PLAN.as_posix(), xla_hmc_ready=True, full_chain_xla_diagnostic_ready=False)
            initial = _initial_bank(tf, int(model["dimension"]))
            hmc_root = output / target_name / "hmc"
            tuning = _tune_hmc(base_adapter, transport, initial, hmc_root / "tuning", f"analytic_control:{target_name}:hmc_tuning_v1")
            hmc_root.mkdir(parents=True, exist_ok=True)
            hmc = _run_hmc(tf, model, transport, base_adapter, initial, tuning, hmc_root, target_name, max(1.0, float(args.time_cap) - (time.perf_counter() - started)))
            target_result.update({"hmc_training_seed": seed, "hmc_training": training, "hmc_state_hash": state_hash, "hmc_tuning": tuning, "hmc": hmc, "hmc_passed": bool(hmc.get("passed", False) and hmc.get("post_hmc_exact_law", {}).get("passed", False))})
        else:
            target_result["hmc"] = {"status": "blocked_by_replication_veto", "passed": False}
        result_targets[target_name] = target_result
        progress.update({"phase": f"{target_name}_complete", "completed_training_cells": sum(len(v.get("replications", [])) for v in result_targets.values())})
        _write(output / "progress.json", progress)
    manifest = {"schema": "bayesfilter.neutra.replication_hmc_manifest.v1", "plan": PLAN.as_posix(), "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(), "gpu": str(logical[0]), "memory_policy": memory_policy, "dtype": "float64", "jit_compile": True, "tf32_enabled": False, "batch_size": BATCH_SIZE, "training_updates": TRAIN_UPDATES, "training_seeds": TRAIN_SEEDS, "chain_count": HMC_CHAINS, "trust_basis": "owner_designated_managed_session_visible_gpu_trusted", "wall_seconds": time.perf_counter() - started}
    result = {"schema": "bayesfilter.neutra.replication_hmc_result.v1", "manifest": manifest, "targets": result_targets, "decision": {"promotion": False, "no_nuts": True, "no_ssl_lstm_transfer": True, "status": "replication_hmc_campaign_complete", "nonclaims": ["no universal training default", "no statistical superiority", "no multimodal coverage", "no production HMC default"]}, "wall_seconds": time.perf_counter() - started}
    progress.update({"phase": "complete"})
    _write(output / "progress.json", progress); _write(output / "run_manifest.json", manifest); _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.replication_hmc_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"], "targets": {k: {"replication_passed": v["replication_passed"], "hmc_passed": bool(v.get("hmc", {}).get("passed", False))} for k, v in result_targets.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
