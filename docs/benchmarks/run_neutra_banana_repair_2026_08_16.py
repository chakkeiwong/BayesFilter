#!/usr/bin/env python3
"""Test a target-specific longer-budget repair for the banana NeuTra control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

PLAN = ROOT / "docs/plans/bayesfilter-neutra-banana-repair-plan-2026-08-16.md"
BASE_RUNNER = ROOT / "docs/benchmarks/run_neutra_replication_hmc_campaign_2026_08_16.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-banana-repair-2026-08-16"
SEEDS = (13, 14, 15)
BATCH_SIZE = 4096
AUDIT_COUNT = 131072
HMC_CHAINS = 4
SCHEDULE_HORIZON = 3000


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--time-cap", type=float, default=3600.0)
    return parser.parse_args()


def _load_replication() -> Any:
    spec = importlib.util.spec_from_file_location("neutra_replication_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load replication runner")
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


def _state_hash(tf: Any, transport: Any) -> str:
    digest = hashlib.sha256()
    for variable in transport.trainable_variables:
        digest.update(bytes(tf.io.serialize_tensor(variable.read_value()).numpy()))
    return digest.hexdigest()


def _config(model: Mapping[str, Any], *, seed: int) -> Any:
    from bayesfilter.inference.neutra_weighted_training import WeightedNeuTraConfig

    stages = int(model["stages"])
    return WeightedNeuTraConfig(
        dimension=int(model["dimension"]),
        hidden_layers=(32, 32),
        stages=stages,
        activation="elu",
        s_max=1.0,
        stage_s_max=tuple(model["stage_caps"]),
        stage_unbounded_scale_linear=(True,) + (False,) * (stages - 1),
        permutation_policy="root_preserving_reverse",
        initialization_scale=0.02,
        initialization_seed=(20260816, 50000 + int(seed)),
        learning_rate=5.0e-4,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )


def _train(tf: Any, model: Mapping[str, Any], *, seed: int, updates: int) -> tuple[Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_weighted_training import MatchedReverseKLNeuTraTrainer

    config = _config(model, seed=seed)
    trainer = MatchedReverseKLNeuTraTrainer(config, model["target_log_prob"])
    selection = tf.random.stateless_normal((65536, int(model["dimension"])), seed=(20260816 + seed, 51001), dtype=tf.float64)
    best_loss = float(_selection_loss(tf, trainer.transport, model, selection).numpy())
    best_state = tuple(tf.identity(v) for v in trainer.variables)
    best_update = 0
    clipped = 0
    terminal = best_loss
    for update in range(1, int(updates) + 1):
        # Preserve the original 3,000-update phase boundaries while testing
        # whether extra low-rate updates repair the banana basin.
        fraction = float(update) / float(SCHEDULE_HORIZON)
        multiplier = 1.0 if fraction < 0.60 else 0.1 if fraction < 0.85 else 0.01
        trainer.optimizer.learning_rate.assign(5.0e-4 * multiplier)
        latent = tf.random.stateless_normal((BATCH_SIZE, int(model["dimension"])), seed=(20260816 + seed, 52000 + update), dtype=tf.float64)
        step = trainer.train_step(latent)
        clipped += int(bool(step.clipping_applied.numpy()))
        if update % 250 == 0 or update == int(updates):
            terminal = float(_selection_loss(tf, trainer.transport, model, selection).numpy())
            if terminal < best_loss:
                best_loss = terminal
                best_update = update
                best_state = tuple(tf.identity(v) for v in trainer.variables)
    for variable, value in zip(trainer.variables, best_state, strict=True):
        variable.assign(value)
    if not math.isfinite(best_loss):
        raise RuntimeError("nonfinite training loss")
    return trainer.transport, {
        "selected_loss": best_loss,
        "terminal_loss": terminal,
        "selected_update": best_update,
        "clipped_updates": clipped,
        "executed_updates": int(updates),
        "schedule_horizon": SCHEDULE_HORIZON,
        "state_hash": _state_hash(tf, trainer.transport),
    }


def _selection_loss(tf: Any, transport: Any, model: Mapping[str, Any], latent: Any) -> Any:
    physical, logdet = transport.forward_and_logdet(latent)
    return tf.reduce_mean(-model["target_log_prob"](physical) - logdet)


def _cell(tf: Any, base: Any, model: Mapping[str, Any], *, seed: int, updates: int, out: Path) -> Mapping[str, Any]:
    transport, training = _train(tf, model, seed=seed, updates=updates)
    audit = base._audit(tf, transport, model, count=AUDIT_COUNT, seed=(20260816, 59000 + seed))
    passed = bool(tf.convert_to_tensor(audit["passed"]).numpy())
    row = {
        "seed": int(seed),
        "updates": int(updates),
        "training": training,
        "audit": audit,
        "passed": passed,
        "state_hash": _state_hash(tf, transport),
        "config": _safe(transport.config.manifest_payload()),
    }
    _write(out / f"seed-{seed}.json", row)
    return row


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if not PLAN.is_file() or not BASE_RUNNER.is_file():
        raise FileNotFoundError("plan or replication runner missing")
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
    replication = _load_replication()
    base = replication._load_base()
    model = base._model(tf, "banana")
    progress = {"schema": "bayesfilter.neutra.banana_repair_progress.v1", "phase": "started", "completed_cells": 0}
    _write(output / "progress.json", progress)
    arms = (("baseline_3000", 3000), ("extended_6000", 6000))
    cells: dict[str, list[Mapping[str, Any]]] = {name: [] for name, _ in arms}
    final_transports: dict[str, Any] = {}
    for name, updates in arms:
        for seed in SEEDS:
            if time.perf_counter() - started > float(args.time_cap):
                raise TimeoutError("banana repair time cap exhausted during training")
            out = output / name
            transport, training = _train(tf, model, seed=seed, updates=updates)
            audit = replication._audit(tf, transport, model, count=AUDIT_COUNT, seed=(20260816, 59000 + seed))
            row = {"seed": seed, "updates": updates, "training": training, "audit": audit, "passed": bool(tf.convert_to_tensor(audit["passed"]).numpy()), "state_hash": _state_hash(tf, transport), "config": _safe(transport.config.manifest_payload())}
            _write(out / f"seed-{seed}.json", row)
            cells[name].append(row)
            if name == "extended_6000" and seed == SEEDS[-1]:
                final_transports[name] = transport
            progress.update({"phase": name, "completed_cells": sum(len(rows) for rows in cells.values()), "latest": {"arm": name, "seed": seed, "passed": row["passed"]}})
            _write(output / "progress.json", progress)

    extended_passed = all(row["passed"] for row in cells["extended_6000"])
    hmc: Mapping[str, Any]
    if extended_passed:
        transport = final_transports["extended_6000"]
        state_hash = _state_hash(tf, transport)
        base_adapter = replication.AnalyticControlAdapter(tf, model, "analytic_control:banana:base_v1")
        replication._bind_frozen(transport, tf, model, state_hash)
        adapter = __import__("bayesfilter.inference.fixed_transport_hmc_mechanics_tf", fromlist=["build_fixed_transport_value_score_adapter"]).build_fixed_transport_value_score_adapter(base_adapter=base_adapter, fixed_transport=transport, target_scope=f"analytic_control:banana:frozen:{state_hash}", evidence_path=PLAN.as_posix(), xla_hmc_ready=True, full_chain_xla_diagnostic_ready=False)
        initial = replication._initial_bank(tf, int(model["dimension"]))
        hmc_root = output / "hmc"
        tuning = replication._tune_hmc(base_adapter, transport, initial, hmc_root / "tuning", "analytic_control:banana:hmc_tuning_v2")
        hmc = replication._run_hmc(tf, model, transport, base_adapter, initial, tuning, hmc_root, "banana_repair", max(1.0, float(args.time_cap) - (time.perf_counter() - started)))
        hmc = {"status": "executed", "tuning": tuning, **hmc}
    else:
        hmc = {"status": "blocked_by_extended_replication_veto", "passed": False}
    manifest = {
        "schema": "bayesfilter.neutra.banana_repair_manifest.v1",
        "plan": PLAN.as_posix(),
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(),
        "target": model["manifest"], "device": str(logical[0]), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_policy": memory_policy, "dtype": "float64", "jit_compile": True, "tf32_enabled": False,
        "batch_size": BATCH_SIZE, "audit_count": AUDIT_COUNT, "seeds": SEEDS, "arms": arms, "schedule_horizon": SCHEDULE_HORIZON, "trust_basis": "owner_designated_managed_session_visible_gpu_trusted", "wall_seconds": time.perf_counter() - started,
    }
    result = {"schema": "bayesfilter.neutra.banana_repair_result.v1", "manifest": manifest, "cells": cells, "extended_replication_passed": extended_passed, "hmc": hmc, "decision": {"promotion": False, "no_nuts": True, "no_ssl_lstm_transfer": True, "status": "banana_budget_repair_complete", "nonclaims": ["no universal budget", "no superiority", "no multimodal coverage", "no production HMC default"]}, "wall_seconds": time.perf_counter() - started}
    progress.update({"phase": "complete", "extended_replication_passed": extended_passed})
    _write(output / "progress.json", progress); _write(output / "run_manifest.json", manifest); _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.banana_repair_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"], "extended_replication_passed": extended_passed, "hmc_passed": bool(hmc.get("passed", False))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
