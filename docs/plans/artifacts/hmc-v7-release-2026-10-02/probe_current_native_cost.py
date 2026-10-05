"""Diagnostic replay of preselected original native batches; no tuning authority."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", type=Path, action="append", required=True)
p.add_argument("--gpu", required=True)
args = p.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
sys.path.insert(0, str(args.source.resolve()))
from scripts.run_hmc_v7_release_prices import check_source
signature = check_source(args.source.resolve())
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.inference.hmc import FullChainHMCConfig
from bayesfilter.inference.hmc_replicated_batch import ReplicatedTrialBatchRunner
from bayesfilter.inference.hmc_candidate_set_execution import (
    HMCCandidateExecutionBinding, HMCCandidateExecutionConfig,
    _rebuild_geometry, _tensor_from_payload, _tensor_payload, _trace_payload)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem, _json_native_sha256
from bayesfilter.inference.hmc_acceptance_trials import _assemble_trials


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def inventory():
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        devices=subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).splitlines(),
        processes=subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits"], text=True).splitlines(),
        cpu_pressure=Path("/proc/pressure/cpu").read_text())


observations = [inventory()]
write("worker-manifest.json", dict(source_manifest_sha256=signature, gpu_uuid=args.gpu,
    memory_policy=memory, pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
    tensorflow_version=tf.__version__, dtype="float64", jit_compile=True,
    tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
    plan="docs/plans/artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-04.md"))
rows = []
for model in args.model:
    spec = read(model/"tuning/execution_spec.json")["execution"]
    config = read(model/"configuration.json")
    saved = read(model/"tuning/tuning_checkpoint.json")["result"]
    candidates = {c["candidate_id"]:c for c in saved["candidates"]}
    work = min((w for w in saved["work_items"] if w["stage"] == "measurement"
        and w["predecessor_work_id"] is None
        and candidates[w["candidate_id"]]["leapfrog_steps"] == 25), key=lambda w:w["ordinal"])
    observation = next(o["observation"] for o in saved["observations"] if o["work_item_id"] == work["work_item_id"])
    evidence_path = model/"tuning/numerical_evidence"/(observation["numerical_evidence_hash"]+".json")
    evidence = read(evidence_path)
    assert _json_native_sha256(evidence) == evidence_path.stem
    chunks = evidence["chunks"]
    assert len(chunks) == 32 and all(c["count"] == 68 for c in chunks)
    candidate = candidates[work["candidate_id"]]
    seeds = [c["seed"] for c in chunks]
    assert len({tuple(seed) for seed in seeds}) == 32
    target = ValidationTarget(config["target"], config["parameters"], config["data"])
    adapter, _ = _rebuild_geometry(target, spec["layers"], spec["target_scope"])
    starts = _tensor_from_payload(spec["initial_active_state"])
    runtime = SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(spec["config"]),
        initial_active_state=starts, scope=SimpleNamespace(payload=lambda:spec["scope"]))
    runtime.health_failures=lambda initial,samples,trace:HMCCandidateExecutionBinding.health_failures(runtime,initial,samples,trace)
    runtime.analyze_trial=lambda initial,samples,trace:HMCCandidateExecutionBinding.analyze_trial(runtime,initial,samples,trace)
    native = ReplicatedTrialBatchRunner(adapter, starts, FullChainHMCConfig(
        num_results=68, num_burnin_steps=0, step_size=candidate["epsilon"],
        num_leapfrog_steps=25, seed=tuple(seeds[0]), use_xla=True,
        target_scope=spec["target_scope"], target_status_trace_policy=runtime.config.target_status_trace_policy,
        capture_candidate_health=True), batch_size=32)
    states = tf.broadcast_to(starts, [32, *starts.shape])
    recorded_native = sum(c["runtime"]["sample_chain_call_s"] for c in chunks)
    for repetition in range(3):
        samples, trace, metadata = native.run(states=states, seeds=seeds,
            step_size=candidate["epsilon"], num_leapfrog_steps=25)
        assert "GPU:0" in samples.device
        decoded = [(samples[:,i], tf.nest.map_structure(lambda t:t[:,i], trace)) for i in range(32)]
        exact = all(_tensor_payload(s) == c["samples"] and _trace_payload(t) == c["trace"]
            for (s,t),c in zip(decoded,chunks))
        before = time.monotonic()
        assembled = _assemble_trials(runtime, HMCWorkItem.from_payload(evidence["work"]),
            chunks, decoded_chunks=decoded)
        assembly = time.monotonic()-before
        exact_records = json.dumps(assembled,sort_keys=True,allow_nan=False) == json.dumps(evidence["trials"],sort_keys=True,allow_nan=False)
        row = dict(model=config["target"], case=config["case_id"], repetition=repetition,
            work_id=work["work_item_id"], evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            seeds=seeds, sample_device=samples.device, native_seconds=metadata["sample_chain_call_s"],
            recorded_native_seconds=recorded_native, assembly_seconds=assembly,
            exact_raw_tensors=exact, exact_full_records=exact_records,
            trace_count=native._runner.experimental_get_tracing_count())
        rows.append(row);observations.append(inventory())
        write("calls.json",rows);write("resource-observations.json",observations)
        if not exact or not exact_records:
            raise ValueError("diagnostic replay differs from original saved batch")
check_source(args.source.resolve())
write("result.json",dict(status="exact_original_native_batches_replayed", rows=rows,
    source_manifest_sha256=signature, memory_policy=memory, gpu_uuid=args.gpu,
    numerical_sampling=True, tuning_authority=False, release_ready=False,
    interpretation="Preselected original L25 measurement batches replayed on unchanged source and affinity. Cold/warm and prior/current costs are descriptive, with observed resources; no new price, speed ranking or independent delivery evidence."))
write("resources.json",observations)
(args.output/"runner.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps({"status":"exact_original_native_batches_replayed", "calls":[{k:v for k,v in r.items() if k not in ('seeds','work_id')} for r in rows]}))
