#!/usr/bin/env python3
"""Fresh CPU/XLA plain-HMC authority for q=20 seed-B comparison."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-seed-b-plain-hmc-reference-plan-2026-08-07.md"
SCRIPT = Path(__file__).resolve()
SCHEMA = "bayesfilter.ssl_lstm.q20_seed_b_plain_hmc_reference.v1"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
GRID = (3, 5, 9, 13, 18, 25)
WORKERS = 25
ROWS_PER_WORKER = 4
CAMPAIGN_CAP_SECONDS = 20_000.0
ROOT_SEED = (20260807, 88001)
INITIAL_STATES = (
    (0.73311370, 0.17273238, 0.58942510, 0.15892059),
    (0.73311370, 0.17273238, 0.58942510, 0.15892059),
    (0.44667563, -0.24131804, -0.58769660, 0.11989041),
    (0.44667563, -0.24131804, -0.58769660, 0.11989041),
)
TUNING_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-seed-b-plain-hmc-reference-2026-08-07/r1/tuning")
SEQUENTIAL_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-seed-b-plain-hmc-reference-2026-08-07/r1/sequential")


class CampaignError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).rstrip(b"\n")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if hasattr(value, "numpy"):
        return safe(value.numpy().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(type(value))


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise CampaignError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def _check_cpu() -> Any:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise CampaignError("plain reference requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise CampaignError("plain reference can see a GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def _lineage() -> Any:
    from bayesfilter.inference.hmc_fixed_metric_grid_search import FixedMetricSearchLineage
    from bayesfilter.testing.ssl_lstm_q20_plain_hmc_reference_worker import expected_lineage_payload
    return FixedMetricSearchLineage(**expected_lineage_payload())


def _policy() -> Any:
    from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy
    return HMCAcceptancePolicy()


def _worker_environment() -> tuple[tuple[str, str], ...]:
    return (("CUDA_VISIBLE_DEVICES", "-1"), ("TF_CPP_MIN_LOG_LEVEL", "2"), ("TF_NUM_INTRAOP_THREADS", "1"), ("TF_NUM_INTEROP_THREADS", "1"), ("OMP_NUM_THREADS", "1"))


def _source_hashes() -> Mapping[str, str]:
    return {
        "script": sha256(SCRIPT),
        "worker": sha256(ROOT / "bayesfilter/testing/ssl_lstm_q20_plain_hmc_reference_worker.py"),
        "target": sha256(ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"),
        "plan": sha256(PLAN),
    }


def _run_manifest(args: argparse.Namespace, *, mode: str) -> Mapping[str, Any]:
    return {
        "command": " ".join(sys.argv),
        "mode": mode,
        "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": sys.version.split()[0],
        "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
        "jit_compile": True,
        "dtype": "float64",
        "source_sha256": _source_hashes(),
    }


def _run_tuning(args: argparse.Namespace) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_fixed_metric_grid_search import FixedMetricGridExecutionConfig, FixedMetricGridSearchConfig, run_fixed_metric_grid_search
    started = time.perf_counter()
    config = FixedMetricGridSearchConfig(l_grid=GRID, root_seed=ROOT_SEED, initial_step_size=0.05, screen_num_results=64, extension_num_results=128, refinement_rounds=0)
    execution = FixedMetricGridExecutionConfig(mode="process_parallel", max_workers=WORKERS, worker_factory_locator="bayesfilter.testing.ssl_lstm_q20_plain_hmc_reference_worker:q20_plain_hmc_worker_factory", worker_environment=_worker_environment())
    result = run_fixed_metric_grid_search(config=config, lineage=_lineage(), acceptance_policy=_policy(), execution=execution)
    payload = {"schema": SCHEMA, "mode": "tuning", "status": "TUNING_SUCCEEDED" if result.survivors else "NO_SURVIVOR", "target_signature": TARGET_SIGNATURE, "target_type": "plain_q20_posterior_no_transport", "grid_private": result.payload(), "grid_public": result.public_summary(), "round0_complete": len(result.round0_candidates) == len(GRID), "source_sha256": {"script": sha256(SCRIPT), "plan": sha256(PLAN)}, "wall_seconds": time.perf_counter() - started, "run_manifest": {"command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"), "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import", "jit_compile": True, "dtype": "float64", "worker_count": WORKERS, "rows_per_worker": ROWS_PER_WORKER, "cap_seconds": float(args.cap_seconds)}, "nonclaims": ["plain HMC tuning only", "no convergence or posterior claim", "no NeuTra claim"]}
    payload["tuning_hash"] = stable_hash(payload)
    write_json(args.output_root / "tuning-result.json", payload, replace=True)
    return payload


def _load_kernel(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if payload.get("status") != "TUNING_SUCCEEDED" or payload.get("round0_complete") is not True:
        raise CampaignError("plain tuning is not a complete successful grid")
    private = payload.get("grid_private")
    survivors = private.get("survivor_pairs", ()) if isinstance(private, Mapping) else ()
    if not survivors:
        raise CampaignError("plain tuning has no survivor")
    selected = sorted(survivors, key=lambda row: (int(row["num_leapfrog_steps"]), float(row["tuned_step_size"])))[0]
    leapfrog = int(selected["num_leapfrog_steps"])
    if leapfrog < 2:
        raise CampaignError("L=1 is forbidden")
    return {"num_leapfrog_steps": leapfrog, "step_size": float(selected["tuned_step_size"]), "selection_rule": "smallest_L_then_step_tie_break_no_stochastic_ranking", "tuning_hash": payload["tuning_hash"], "lineage": private["lineage"]}


def _sequential(args: argparse.Namespace) -> Mapping[str, Any]:
    tf = _check_cpu()
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, _run_archived_sequential_neutra_hmc
    from bayesfilter.testing.ssl_lstm_q20_plain_hmc_reference_worker import PlainQ20Adapter
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
    kernel = _load_kernel(args.tuning_root / "tuning-result.json")
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    adapter = PlainQ20Adapter(target)
    if adapter.target_signature() != TARGET_SIGNATURE:
        raise CampaignError("plain sequential target signature mismatch")
    initial = tf.constant(INITIAL_STATES, tf.float64)
    config = SequentialNeuTraHMCConfig(seed=(20260807, 89001), step_size=kernel["step_size"], num_leapfrog_steps=kernel["num_leapfrog_steps"], warmup_chunk_size=500, warmup_min_results=2000, warmup_window_results=1000, warmup_max_results=2000, retained_chunk_size=500, retained_min_results=1000, retained_max_results=1000, warmup_rhat_max=1.05, retained_rhat_max=1.01, bulk_ess_min=400.0, tail_ess_min=400.0, acceptance_min=0.35, acceptance_max=0.95, use_xla=True, target_status_required=True)
    started = time.perf_counter()
    result = _run_archived_sequential_neutra_hmc(adapter, initial, config, archive_root=args.output_root / "archive", archive_label="plain-q20")
    payload = {"schema": SCHEMA, "mode": "sequential", "status": "SEQUENTIAL_SCREEN_PASSED" if result.passed else "SEQUENTIAL_SCREEN_FAILED", "target_signature": TARGET_SIGNATURE, "adapter_signature": adapter.adapter_signature(), "kernel": kernel, "initial_states": safe(initial), "result": safe(result), "source_sha256": {"script": sha256(SCRIPT), "plan": sha256(PLAN)}, "wall_seconds": time.perf_counter() - started, "run_manifest": {"command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"), "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import", "jit_compile": True, "dtype": "float64", "cap_seconds": float(args.cap_seconds)}, "nonclaims": ["plain-HMC sequential screen only", "no model adequacy or superiority claim", "native divergence unavailability is not zero divergences"]}
    payload["sequential_hash"] = stable_hash(payload)
    write_json(args.output_root / "sequential-result.json", payload, replace=True)
    return payload


def _preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    tf = _check_cpu()
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
    from bayesfilter.testing.ssl_lstm_q20_plain_hmc_reference_worker import PlainQ20Adapter
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    adapter = PlainQ20Adapter(target)
    points = tf.constant(INITIAL_STATES, tf.float64)
    values, scores = adapter.log_prob_and_grad(points)
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy() and tf.reduce_all(tf.math.is_finite(scores)).numpy()):
        raise CampaignError("plain target preflight nonfinite")
    payload = {"schema": SCHEMA, "mode": "preflight", "status": "PREFLIGHT_PASSED", "target_signature": adapter.target_signature(), "adapter_signature": adapter.adapter_signature(), "initial_states": safe(points), "values": safe(values), "scores": safe(scores), "jit_compile": True, "cpu_only": True, "source_sha256": {"script": sha256(SCRIPT), "plan": sha256(PLAN)}}
    write_json(args.output_root / "preflight.json", payload, replace=True)
    return payload


def _rate(args: argparse.Namespace) -> Mapping[str, Any]:
    tf = _check_cpu()
    from bayesfilter.inference.hmc import FullChainHMCConfig, build_reusable_full_chain_tfp_hmc_runner
    from bayesfilter.inference.hmc_tuning import HMCTuningPolicy
    from bayesfilter.testing.ssl_lstm_q20_plain_hmc_reference_worker import PlainQ20Adapter
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
    adapter = PlainQ20Adapter(batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"))
    config = FullChainHMCConfig(num_results=2, num_burnin_steps=1, step_size=0.05, num_leapfrog_steps=3, seed=(20260807, 89101), use_xla=True, trace_policy="standard", target_status_trace_policy="per_chain_step", tuning_policy=None, target_scope=adapter.target_scope)
    runner = build_reusable_full_chain_tfp_hmc_runner(adapter, tf.constant(INITIAL_STATES, tf.float64), config)
    first = runner.run(seed=config.seed, step_size=0.05)
    warm = runner.run(seed=(20260807, 89102), step_size=0.05)
    payload = {"schema": SCHEMA, "mode": "rate", "status": "RATE_PASSED", "num_results": 2, "num_leapfrog_steps": 3, "first_seconds": float(first.metadata["sample_chain_call_s"]), "warm_seconds": float(warm.metadata["sample_chain_call_s"]), "seconds_per_transition_leapfrog": float(warm.metadata["sample_chain_call_s"]) / 6.0, "target_signature": adapter.target_signature(), "adapter_signature": adapter.adapter_signature(), "jit_compile": True, "cpu_only": True, "source_sha256": _source_hashes(), "run_manifest": _run_manifest(args, mode="rate"), "nonclaims": ["timing canary only", "no tuning or posterior claim"]}
    write_json(args.output_root / "rate.json", payload, replace=True)
    return payload


def _one_chain_rate_worker(seed: tuple[int, int]) -> Mapping[str, Any]:
    tf = _check_cpu()
    from bayesfilter.inference.hmc import FullChainHMCConfig, build_reusable_full_chain_tfp_hmc_runner
    from bayesfilter.testing.ssl_lstm_q20_plain_hmc_reference_worker import PlainQ20Adapter
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
    adapter = PlainQ20Adapter(batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"))
    state = tf.constant([INITIAL_STATES[0]], tf.float64)
    config = FullChainHMCConfig(num_results=2, num_burnin_steps=1, step_size=0.05, num_leapfrog_steps=3, seed=seed, use_xla=True, trace_policy="standard", target_status_trace_policy="per_chain_step", target_scope=adapter.target_scope)
    runner = build_reusable_full_chain_tfp_hmc_runner(adapter, state, config)
    runner.run(seed=seed, step_size=0.05)
    warm = runner.run(seed=(seed[0], seed[1] + 1), step_size=0.05)
    return {"seed": list(seed), "warm_seconds": float(warm.metadata["sample_chain_call_s"]), "seconds_per_transition_leapfrog": float(warm.metadata["sample_chain_call_s"]) / 6.0}


def _parallel_chain_rate(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    previous = {key: os.environ.get(key) for key, _ in _worker_environment()}
    os.environ.update(dict(_worker_environment()))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
            rows = tuple(executor.map(_one_chain_rate_worker, tuple((20260807, 89200 + index) for index in range(4))))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    payload = {"schema": SCHEMA, "mode": "parallel-chain-rate", "status": "RATE_PASSED", "rows": rows, "critical_path_seconds_per_transition_leapfrog": max(float(row["seconds_per_transition_leapfrog"]) for row in rows), "wall_seconds": time.perf_counter() - started, "workers": 4, "cpu_only": True, "jit_compile": True, "source_sha256": _source_hashes(), "run_manifest": _run_manifest(args, mode="parallel-chain-rate"), "nonclaims": ["four-process one-chain timing canary only", "no tuning or posterior claim"]}
    write_json(args.output_root / "parallel-chain-rate.json", payload, replace=True)
    return payload


def _resource_gate(args: argparse.Namespace) -> Mapping[str, Any]:
    rate_path = args.rate_artifact
    parallel_rate_path = args.parallel_rate_artifact
    rate_payload = json.loads(rate_path.read_text(encoding="ascii"))
    if rate_payload.get("status") != "RATE_PASSED":
        raise CampaignError("resource gate requires a passed rate canary")
    rate = float(rate_payload["seconds_per_transition_leapfrog"])
    parallel_payload = json.loads(parallel_rate_path.read_text(encoding="ascii"))
    if parallel_payload.get("status") != "RATE_PASSED":
        raise CampaignError("resource gate requires a passed parallel-chain rate canary")
    parallel_rate = float(parallel_payload["critical_path_seconds_per_transition_leapfrog"])
    # Three screens and three 128-draw extensions are the complete grid policy.
    per_candidate_transitions = 64 + 1 + 3 * (1 + 64) + 3 * (1 + 128)
    # Each grid worker runs four chains in one process. With six candidates and
    # 25 workers, wall time is the slowest candidate, not total work / 25.
    grid_work = max(int(L) for L in GRID) * per_candidate_transitions
    grid_seconds = grid_work * rate
    # Minimum authority: 2,000 warm-up + 1,000 retained at the smallest allowed L.
    sequential_work = 3 * 3_000
    # The implemented sequential controller is one four-chain process. The
    # one-chain parallel canary is explanatory only and is not used here.
    sequential_seconds = sequential_work * rate
    total = grid_seconds + sequential_seconds
    payload = {
        "schema": SCHEMA,
        "mode": "resource-gate",
        "status": "RESOURCE_VETO_BEFORE_TUNING" if total > float(args.cap_seconds) else "RESOURCE_GATE_PASSED",
        "rate_artifact": rate_path.as_posix(),
        "rate_artifact_sha256": sha256(rate_path),
        "seconds_per_transition_leapfrog": rate,
        "parallel_chain_rate_artifact": parallel_rate_path.as_posix(),
        "parallel_chain_rate_artifact_sha256": sha256(parallel_rate_path),
        "parallel_chain_critical_path_seconds_per_transition_leapfrog": parallel_rate,
        "grid": {"l_values": list(GRID), "per_candidate_transition_leapfrog_work": grid_work, "candidate_count": len(GRID), "concurrent_candidate_workers": len(GRID), "projected_seconds_with_concurrent_four_chain_workers": grid_seconds},
        "minimum_sequential_authority": {"warmup_results_per_chain": 2000, "retained_results_per_chain": 1000, "leapfrog_steps": 3, "transition_leapfrog_work": sequential_work, "projected_seconds_with_one_four_chain_process": sequential_seconds},
        "projected_total_seconds": total,
        "cap_seconds": float(args.cap_seconds),
        "margin_over_cap_seconds": total - float(args.cap_seconds),
        "projection_basis": "implemented_four_chain_process_route; one_chain_parallel_canary_explanatory_only",
        "nonclaims": ["resource projection only", "no plain-HMC tuning or posterior claim", "no NeuTra rejection"],
    }
    write_json(args.output_root / "resource-gate.json", payload, replace=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "rate", "parallel-chain-rate", "resource-gate", "tuning", "sequential"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tuning-root", type=Path, default=TUNING_ROOT)
    parser.add_argument("--cap-seconds", type=float, default=CAMPAIGN_CAP_SECONDS)
    parser.add_argument("--rate-artifact", type=Path)
    parser.add_argument("--parallel-rate-artifact", type=Path)
    args = parser.parse_args()
    args.output_root = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    args.tuning_root = (ROOT / args.tuning_root).resolve() if not args.tuning_root.is_absolute() else args.tuning_root.resolve()
    if not args.output_root.is_relative_to(ROOT) or not args.tuning_root.is_relative_to(ROOT):
        parser.error("output and tuning roots must remain inside the repository")
    if not 0.0 < float(args.cap_seconds) <= CAMPAIGN_CAP_SECONDS:
        parser.error("cap-seconds must be in (0,20000]")
    if args.mode == "resource-gate":
        if args.rate_artifact is None:
            parser.error("resource-gate requires --rate-artifact")
        if args.parallel_rate_artifact is None:
            parser.error("resource-gate requires --parallel-rate-artifact")
        args.rate_artifact = (ROOT / args.rate_artifact).resolve() if not args.rate_artifact.is_absolute() else args.rate_artifact.resolve()
        args.parallel_rate_artifact = (ROOT / args.parallel_rate_artifact).resolve() if not args.parallel_rate_artifact.is_absolute() else args.parallel_rate_artifact.resolve()
        if not args.rate_artifact.is_relative_to(ROOT):
            parser.error("rate-artifact must remain inside repository")
        if not args.parallel_rate_artifact.is_relative_to(ROOT):
            parser.error("parallel-rate-artifact must remain inside repository")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        _preflight(args)
    elif args.mode == "rate":
        _rate(args)
    elif args.mode == "parallel-chain-rate":
        _parallel_chain_rate(args)
    elif args.mode == "resource-gate":
        result = _resource_gate(args)
        if result["status"] != "RESOURCE_GATE_PASSED":
            return 3
    elif args.mode == "tuning":
        _run_tuning(args)
    else:
        _sequential(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
