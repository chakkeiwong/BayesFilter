"""Saved-work CPU reference profile, with no sampling or tuning authority."""
import argparse
import cProfile
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--tuning", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
sys.path.insert(0, str(args.source.resolve()))
from scripts.run_hmc_v7_release_prices import check_source
signature = check_source(args.source.resolve())
from bayesfilter.inference.hmc_candidate_set_execution import (
    HMCCandidateExecutionBinding, HMCCandidateExecutionConfig,
    _tensor_from_payload, _trace_from_payload)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem, _json_native_sha256
from bayesfilter.inference.hmc_acceptance_trials import _assemble_trials

spec = json.loads((args.tuning/"execution_spec.json").read_text())["execution"]
checkpoint = json.loads((args.tuning/"tuning_checkpoint.json").read_text())
first = min((w for w in checkpoint["result"]["work_items"] if w["stage"] == "measurement"),
            key=lambda w: w["ordinal"])
for path in (args.tuning/"numerical_evidence").glob("*.json"):
    evidence = json.loads(path.read_text())
    if evidence["work"]["work_item_id"] == first["work_item_id"]:
        break
else:
    raise ValueError("missing first measurement evidence")
assert _json_native_sha256(evidence) == path.stem
work = HMCWorkItem.from_payload(evidence["work"])
assert work.predecessor_work_id is None
chunks = evidence["chunks"]
runtime = SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(spec["config"]),
    initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
    scope=SimpleNamespace(payload=lambda: spec["scope"]))
runtime.health_failures = lambda initial,samples,trace: HMCCandidateExecutionBinding.health_failures(runtime,initial,samples,trace)
runtime.analyze_trial = lambda initial,samples,trace: HMCCandidateExecutionBinding.analyze_trial(runtime,initial,samples,trace)
decoded = [(_tensor_from_payload(c["samples"]), _trace_from_payload(c["trace"])) for c in chunks]
timings = []
before = time.monotonic()
baseline = _assemble_trials(runtime, work, chunks, decoded_chunks=decoded)
timings.append(dict(kind="cold_plain", seconds=time.monotonic()-before))
before = time.monotonic()
baseline = _assemble_trials(runtime, work, chunks, decoded_chunks=decoded)
timings.append(dict(kind="warm_plain", seconds=time.monotonic()-before))
profile = cProfile.Profile()
before = time.monotonic()
profile.enable()
observed = _assemble_trials(runtime, work, chunks, decoded_chunks=decoded)
profile.disable()
timings.append(dict(kind="warm_instrumented", seconds=time.monotonic()-before))
assert json.dumps(baseline, sort_keys=True, allow_nan=False) == json.dumps(observed, sort_keys=True, allow_nan=False)
profile.dump_stats(str(args.output/"profile.pstats"))
stats = pstats.Stats(profile)
functions = []
for (file,line,name),(primitive,total,self_time,cumulative,callers) in stats.stats.items():
    if "bayesfilter" in file:
        functions.append(dict(file=file,line=line,name=name,primitive_calls=primitive,
            total_calls=total,self_seconds=self_time,cumulative_seconds=cumulative))
functions.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
from bayesfilter.inference import hmc_candidate_set_execution as execution
from bayesfilter.inference import hmc_verification as verification
from bayesfilter.inference import hmc_acceptance_statistics as statistics
timed = {}
originals = []


def wrap(owner, name, label):
    original = getattr(owner, name)
    originals.append((owner, name, original))
    timed[label] = dict(calls=0, inclusive_seconds=0.)

    @wraps(original)
    def measured(*values, **keywords):
        start = time.monotonic()
        try:
            return original(*values, **keywords)
        finally:
            timed[label]["calls"] += 1
            timed[label]["inclusive_seconds"] += time.monotonic()-start
    setattr(owner, name, measured)


try:
    for owner, names in ((runtime, ("analyze_trial", "health_failures")),
            (verification, ("evaluate_hmc_trial_health", "_validate_signed_proxy_summary",
                "_validate_valid_acceptance_summary", "_validate_v5_roles", "_all_close", "_all_finite")),
            (statistics, ("complete_trial_scores",)),
            (execution, ("_tensor_payload", "_trace_payload"))):
        for name in names:
            wrap(owner, name, name)
    before = time.monotonic()
    measured = _assemble_trials(runtime, work, chunks, decoded_chunks=decoded)
    timings.append(dict(kind="warm_explicit_timers", seconds=time.monotonic()-before))
finally:
    for owner, name, original in reversed(originals):
        setattr(owner, name, original)
assert json.dumps(baseline, sort_keys=True, allow_nan=False) == json.dumps(measured, sort_keys=True, allow_nan=False)
assert timed["analyze_trial"]["calls"] == timed["complete_trial_scores"]["calls"] == len(baseline)
result = dict(status="exact_plain_instrumented_serial_records", trials=len(baseline),
    source_manifest_sha256=signature, input_path=str(path.resolve()),
    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), work_id=work.work_item_id,
    timings=timings, functions=functions, explicit_timers=timed, gpu_intentionally_hidden=True,
    numerical_sampling=False, release_ready=False,
    interpretation="CPU reference localization of saved serial trial analysis. This environment's cProfile misses some warmed Python frames; use explicit timer counts for named-function localization. Inclusive times overlap; no GPU price, speed ranking, concurrency adoption or new numerical authority.",
    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md")
check_source(args.source.resolve())
with (args.output/"result.json").open("x") as stream:
    stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
with (args.output/"runner.py").open("xb") as stream:
    stream.write(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in result.items() if k != "functions"}))
print(json.dumps(timed,indent=2))
