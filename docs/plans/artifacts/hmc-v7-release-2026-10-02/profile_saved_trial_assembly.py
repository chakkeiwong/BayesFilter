"""Profile a saved trial assembler, without target calls or admission authority."""
import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

p = argparse.ArgumentParser()
for arg in ('source', 'tuning', 'record', 'output'):
    p.add_argument('--'+arg, type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--worker', action='store_true')
args = p.parse_args()
for name in ('source', 'tuning', 'record', 'output'):
    setattr(args, name, getattr(args, name).resolve())
os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TF_FORCE_GPU_ALLOW_GROWTH='true',
    TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2', BAYESFILTER_PRELOAD_CUSTOM_OP='0')
os.sched_setaffinity(0, {24,25,26,27})

def write(name, value):
    (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

if not args.worker:
    args.output.mkdir(parents=True, exist_ok=False)
    script = args.output/'runner.py'
    script.write_bytes(Path(__file__).read_bytes())
    command = [sys.executable, str(script), *sys.argv[1:], '--worker']
    started = time.monotonic()
    manifest = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(),
        environment=sys.executable, gpu=args.gpu, resource='gpu',
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md',
        runner_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), native_sampling=False)
    write('manifest.json', manifest)
    with (args.output/'run.log').open('x') as log:
        try:
            code=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=55).returncode
        except subprocess.TimeoutExpired:
            code=124
    write('receipt.json', {**manifest, 'exit_code':code, 'wall_seconds':time.monotonic()-started})
    print(json.dumps({'exit_code':code,'wall_seconds':time.monotonic()-started}))
    raise SystemExit(code)

sys.path.insert(0, str(args.source))
from scripts.run_hmc_v7_release_prices import check_source
signature = check_source(args.source)
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
from bayesfilter.inference.hmc_candidate_set_execution import (
    HMCCandidateExecutionBinding, HMCCandidateExecutionConfig, _tensor_from_payload)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem
from bayesfilter.inference.hmc_acceptance_trials import _assemble_trials

spec = json.loads((args.tuning/'execution_spec.json').read_text())['execution']
raw_bytes=args.record.read_bytes()
raw=json.loads(raw_bytes)
with tf.device('/GPU:0'):
    runtime=SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(spec['config']),
        scope=SimpleNamespace(payload=lambda: spec['scope']),
        initial_active_state=_tensor_from_payload(spec['initial_active_state']))
    runtime.health_failures=MethodType(HMCCandidateExecutionBinding.health_failures, runtime)
    runtime.analyze_trial=MethodType(HMCCandidateExecutionBinding.analyze_trial, runtime)
    work=HMCWorkItem.from_payload(raw['work'])
    write('worker-manifest.json', dict(source_manifest_sha256=signature,
        record_sha256=hashlib.sha256(raw_bytes).hexdigest(), memory_policy=memory,
        device=runtime.initial_active_state.device, tensorflow_version=tf.__version__,
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(), jit_compile=True,
        scope='raw trial reconstruction diagnostic; no binding, replay or release authority'))
    profile=cProfile.Profile()
    start=time.monotonic()
    profile.enable()
    trials=_assemble_trials(runtime, work, raw['chunks'])
    profile.disable()
    elapsed=time.monotonic()-start
    profile.dump_stats(str(args.output/'profile.pstats'))
    report=io.StringIO()
    pstats.Stats(profile,stream=report).sort_stats('cumulative').print_stats(35)
    (args.output/'profile.txt').write_text(report.getvalue())
    expected=raw['trials'][work.trial_range[0]:work.trial_range[1]]
    assert json.loads(json.dumps(trials)) == expected
    write('result.json', dict(status='exact_saved_trial_payloads', trials=len(trials),
        wall_seconds=elapsed, native_sampling=False, release_ready=False))
