"""Compare unchanged saved-trial assembly placement; no HMC or admission."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def differences(actual, expected, prefix="", limit=12):
    """Locate mismatches without dumping traces or accepting a tolerance."""
    if actual == expected:
        return []
    if isinstance(actual, dict) and isinstance(expected, dict):
        paths = []
        for key in sorted(actual.keys() | expected.keys()):
            paths += differences(actual.get(key), expected.get(key), prefix + "/" + key)
            if len(paths) >= limit:
                break
        return paths[:limit]
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        paths = []
        for i, (left, right) in enumerate(zip(actual, expected)):
            paths += differences(left, right, prefix + "/" + str(i))
            if len(paths) >= limit:
                break
        return paths[:limit]
    return [prefix]


def worker(args):
    sys.path.insert(0, str(args.source))
    from scripts.run_hmc_v7_release_prices import check_source
    signature = check_source(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.inference import hmc_candidate_set_execution as execution
    from bayesfilter.inference import hmc_acceptance_statistics as statistics
    from bayesfilter.inference.hmc_acceptance_trials import _assemble_trials
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem

    spec = json.loads((args.tuning / "execution_spec.json").read_text())["execution"]
    record_bytes = args.record.read_bytes()
    record = json.loads(record_bytes)
    work = HMCWorkItem.from_payload(record["work"])
    expected = record["trials"][work.trial_range[0]:work.trial_range[1]]
    write(args.output / "worker-manifest.json", dict(
        source_manifest_sha256=signature,
        record_sha256=hashlib.sha256(record_bytes).hexdigest(), memory_policy=memory,
        tensorflow_version=tf.__version__, tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        jit_compile=True, original_graph_choices_preserved=True, trials=len(expected),
        scope="diagnostic placement/reference; no HMC, replay, default or release authority"))
    rows = []
    payloads = {}
    for placement, temperature in [("GPU", "cold"), ("CPU", "cold"),
                                   ("CPU", "warm"), ("GPU", "warm")]:
        costs = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        originals = []

        def instrument(owner, name):
            original = getattr(owner, name)
            originals.append((owner, name, original))

            def timed(*positional, **keywords):
                started = time.perf_counter()
                try:
                    return original(*positional, **keywords)
                finally:
                    costs[name]["calls"] += 1
                    costs[name]["seconds"] += time.perf_counter() - started
            setattr(owner, name, timed)

        with tf.device("/" + placement + ":0"):
            runtime = SimpleNamespace(
                config=execution.HMCCandidateExecutionConfig.from_payload(spec["config"]),
                scope=SimpleNamespace(payload=lambda: spec["scope"]),
                initial_active_state=execution._tensor_from_payload(spec["initial_active_state"]))
            runtime.health_failures = MethodType(execution.HMCCandidateExecutionBinding.health_failures, runtime)
            runtime.analyze_trial = MethodType(execution.HMCCandidateExecutionBinding.analyze_trial, runtime)
            marker_device = tf.constant(1).device
            instrument(runtime, "analyze_trial")
            instrument(statistics, "complete_trial_scores")
            for name in ("_tensor_from_payload", "_tensor_payload"):
                instrument(execution, name)
            started = time.perf_counter()
            try:
                actual = _assemble_trials(runtime, work, record["chunks"])
            finally:
                elapsed = time.perf_counter() - started
                for owner, name, original in reversed(originals):
                    setattr(owner, name, original)
        actual = json.loads(json.dumps(actual, allow_nan=False))
        paths = differences(actual, expected)
        rows.append(dict(placement=placement, temperature=temperature, wall_seconds=elapsed,
                         exact_saved_payloads=not paths, differing_paths=paths,
                         marker_device=marker_device, component_seconds=dict(costs)))
        payloads[placement] = actual
        write(args.output / "progress.json", dict(rows=rows, release_ready=False))
    parity = payloads["CPU"] == payloads["GPU"] and all(r["exact_saved_payloads"] for r in rows)
    write(args.output / "result.json", dict(
        status="exact_placement_parity" if parity else "placement_parity_failed",
        rows=rows, cross_device_equal=payloads["CPU"] == payloads["GPU"],
        cross_device_differing_paths=differences(payloads["CPU"], payloads["GPU"]),
        allocator=tf.config.experimental.get_memory_info("GPU:0"),
        native_sampling=False, release_ready=False,
        interpretation="Component times overlap; timings are descriptive attribution only."))
    return 0 if parity else 1


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "tuning", "record", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    for name in ("source", "tuning", "record", "output"):
        setattr(args, name, getattr(args, name).resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TF_FORCE_GPU_ALLOW_GROWTH="true",
                      TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    os.sched_setaffinity(0, {24, 25, 26, 27})
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    script = args.output / "runner.py"
    script.write_bytes(Path(__file__).read_bytes())
    command = [sys.executable, str(script), *sys.argv[1:], "--worker"]
    started = time.monotonic()
    manifest = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(),
                    environment=sys.executable, gpu=args.gpu, resource="gpu", cpu_affinity=[24,25,26,27],
                    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md",
                    result=str(args.output / "result.json"), native_sampling=False,
                    runner_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    write(args.output / "manifest.json", manifest)
    with (args.output / "run.log").open("x") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=110).returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = dict(manifest, exit_code=code, wall_seconds=time.monotonic() - started)
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("exit_code", "wall_seconds", "result")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
