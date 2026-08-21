#!/usr/bin/env python3
"""Confirm the frozen banana learned-transport L=10 kernel on two banks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DISCOVERY = ROOT / "docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3"
PLAN = ROOT / "docs/plans/bayesfilter-neutra-banana-hmc-repair-plan-2026-08-16.md"
REPLICATION = ROOT / "docs/benchmarks/run_neutra_replication_hmc_campaign_2026_08_16.py"
REPAIR = ROOT / "docs/benchmarks/run_neutra_banana_repair_2026_08_16.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-banana-hmc-l10-confirmation-2026-08-16-r1"
DIMENSION = 16
CHAINS = 4
RETAINED = 5000


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="0")
    p.add_argument("--time-cap", type=float, default=1800.0)
    return p.parse_args()


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
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


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON object required: {path}")
    return value


def _central_bank(tf: Any) -> Any:
    rows = tf.zeros((CHAINS, DIMENSION), tf.float64)
    offsets = tf.constant((0.0, 0.25, -0.25, 0.25), tf.float64)
    axes = tf.constant((0, 0, 0, 1), tf.int32)
    return tf.tensor_scatter_nd_update(rows, tf.stack((tf.range(CHAINS), axes), axis=1), offsets)


def _archive(root: Path, tf: Any, label: str):
    def callback(*, stage: str, chunk_index: Any, latent_samples: Any, model_samples: Any, seed: Any, cumulative: bool) -> Mapping[str, Any]:
        suffix = "cumulative" if cumulative else f"chunk-{int(chunk_index):03d}"
        stage_root = root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        lp = stage_root / f"{label}-{suffix}-latent.tftensor"
        mp = stage_root / f"{label}-{suffix}-model.tftensor"
        lb = bytes(tf.io.serialize_tensor(latent_samples).numpy())
        mb = bytes(tf.io.serialize_tensor(model_samples).numpy())
        lp.write_bytes(lb); mp.write_bytes(mb)
        return {"stage": stage, "chunk_index": chunk_index, "cumulative": bool(cumulative), "seed": None if seed is None else list(seed), "latent_path": lp.as_posix(), "latent_sha256": hashlib.sha256(lb).hexdigest(), "model_path": mp.as_posix(), "model_sha256": hashlib.sha256(mb).hexdigest(), "sample_shape": [int(x) for x in latent_samples.shape]}
    return callback


def _run(replication: Any, tf: Any, model: Mapping[str, Any], transport: Any, base_adapter: Any, initial: Any, kernel: Mapping[str, Any], root: Path, label: str) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, run_sequential_neutra_hmc
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_bulk_tail_ess
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_value_score_adapter

    adapter = build_fixed_transport_value_score_adapter(base_adapter=base_adapter, fixed_transport=transport, target_scope=f"analytic_control:banana:l10_confirmation:{label}", evidence_path=PLAN.as_posix(), xla_hmc_ready=True, full_chain_xla_diagnostic_ready=False)
    cfg = SequentialNeuTraHMCConfig(step_size=float(kernel["step_size"]), num_leapfrog_steps=int(kernel["num_leapfrog_steps"]), warmup_seed=(20260816, 88001), retained_seed=(20260816, 88002), warmup_chunk_results=500, warmup_min_results=2000, warmup_check_window_results=1000, warmup_max_results=10000, retained_chunk_results=500, retained_min_results=RETAINED, retained_max_results=RETAINED, retained_rhat_max=1.01, minimum_chain_count=CHAINS, jit_compile=True)

    def transform(samples: Any) -> Any:
        shape = tf.shape(samples); flat = tf.reshape(samples, (-1, DIMENSION)); return tf.reshape(transport.forward_batch(flat), shape)

    def diagnostic(samples: Any) -> Mapping[str, Any]:
        rhat = rank_normalized_split_rhat_summary(samples, rhat_max=1.01)
        ess = rank_normalized_bulk_tail_ess(samples)
        bulk = float(tf.reduce_min(tf.convert_to_tensor(ess["bulk"], tf.float64)).numpy())
        tail = float(tf.reduce_min(tf.convert_to_tensor(ess["tail"], tf.float64)).numpy())
        return {"passed": bool(rhat["passed"] and bulk >= 400.0 and tail >= 400.0), "rhat": rhat, "ess": ess, "min_bulk_ess": bulk, "min_tail_ess": tail}

    result = run_sequential_neutra_hmc(adapter=adapter, initial_state=initial, model_transform=transform, parameter_names=tuple(f"z_{i}" for i in range(DIMENSION)), config=cfg, retained_diagnostic_fn=diagnostic, archive_callback=_archive(root / "archive", tf, label))
    retained = tf.reshape(result["private_retained_raw"], (-1, DIMENSION))
    exact = model["exact_latent"](retained)
    n = tf.cast(tf.shape(exact)[0], tf.float64); z = tf.constant(3.2905267314919255, tf.float64)
    def interval(values: Any, truth: Any) -> Mapping[str, Any]:
        values = tf.cast(values, tf.float64); truth = tf.cast(truth, tf.float64); mean = tf.reduce_mean(values, axis=0); se = tf.math.reduce_std(values, axis=0) / tf.sqrt(n); passed = tf.logical_and(truth >= mean - z * se, truth <= mean + z * se); return {"estimate": mean, "exact": truth, "standard_error": se, "passed": passed, "all_passed": tf.reduce_all(passed)}
    screens = {"coordinate_mean": interval(exact, tf.zeros((DIMENSION,), tf.float64)), "coordinate_second_moment": interval(tf.square(exact), tf.ones((DIMENSION,), tf.float64)), "adjacent_cross_moment": interval(exact[:, :-1] * exact[:, 1:], tf.zeros((DIMENSION - 1,), tf.float64))}
    post = bool(tf.reduce_all(tf.stack([v["all_passed"] for v in screens.values()])).numpy())
    payload = {k: v for k, v in result.items() if not str(k).startswith("private_")}; payload["post_hmc_exact_law"] = {"sample_count": int(retained.shape[0]), "screens": screens, "passed": post}; payload["hmc_passed"] = bool(result.get("passed", False) and post)
    _write(root / "sequential_result.json", payload)
    return payload


def main() -> int:
    args = _args(); output = args.output_root.resolve()
    if output.exists(): raise FileExistsError(f"output root must be fresh: {output}")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"; os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter(); import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True); tf.config.set_soft_device_placement(False); tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"));
    if len(logical) != 1: raise RuntimeError(f"expected one visible GPU, found {logical}")
    replication = _load(REPLICATION, "banana_l10_replication"); repair = _load(REPAIR, "banana_l10_repair"); base = replication._load_base(); model = base._model(tf, "banana"); base_adapter = replication.AnalyticControlAdapter(tf, model, "analytic_control:banana:base_v1")
    learned, training = repair._train(tf, model, seed=15, updates=6000); state_hash = replication._state_hash(tf, learned); replication._bind_frozen(learned, tf, model, state_hash)
    discovery = _read(DISCOVERY / "learned_training.json");
    if state_hash != str(discovery["state_hash"]): raise RuntimeError("transport replay hash mismatch")
    audit = replication._audit(tf, learned, model, count=131072, seed=(20260816, 59015));
    if not bool(tf.convert_to_tensor(audit["passed"]).numpy()): raise RuntimeError("proposal audit failed")
    tuning = _read(DISCOVERY / "learned_central_bank/tuning/tuning_result.json"); kernel = tuning["final_kernel_payload"]
    if tuning.get("passed") is not True or int(kernel["num_leapfrog_steps"]) != 10: raise RuntimeError("frozen L=10 kernel is not admissible")
    banks = {"original_bank": replication._initial_bank(tf, DIMENSION), "central_bank": _central_bank(tf)}; rows = {}
    _write(output / "progress.json", {"schema": "bayesfilter.neutra.banana_l10_confirmation_progress.v1", "phase": "started", "completed": 0})
    for name, bank in banks.items():
        row = _run(replication, tf, model, learned, base_adapter, bank, kernel, output / name, name); rows[name] = row; _write(output / "progress.json", {"schema": "bayesfilter.neutra.banana_l10_confirmation_progress.v1", "phase": name, "completed": len(rows), "latest_hmc_passed": row["hmc_passed"]})
    manifest = {"schema": "bayesfilter.neutra.banana_l10_confirmation_manifest.v1", "plan": PLAN.as_posix(), "discovery_root": DISCOVERY.as_posix(), "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(), "device": str(logical[0]), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "memory_policy": memory_policy, "dtype": "float64", "jit_compile": True, "tf32_enabled": False, "training_seed": 15, "training_updates": 6000, "transport_state_hash": state_hash, "kernel": kernel, "warmup_results_per_chain": 2000, "retained_results_per_chain": RETAINED, "banks": list(banks), "trust_basis": "owner_designated_managed_session_visible_gpu_trusted", "wall_seconds": time.perf_counter() - started}
    result = {"schema": "bayesfilter.neutra.banana_l10_confirmation_result.v1", "manifest": manifest, "training": training, "proposal_audit": audit, "banks": rows, "decision": {"promotion": False, "status": "l10_confirmation_complete", "nonclaims": ["no universal L10 default", "no superiority", "no SSL-LSTM transfer", "no production readiness"]}, "wall_seconds": time.perf_counter() - started}
    _write(output / "progress.json", {"schema": "bayesfilter.neutra.banana_l10_confirmation_progress.v1", "phase": "complete", "completed": len(rows)}); _write(output / "run_manifest.json", manifest); _write(output / "result.json", result); _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.banana_l10_confirmation_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"], "banks": {k: v["hmc_passed"] for k, v in rows.items()}}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
