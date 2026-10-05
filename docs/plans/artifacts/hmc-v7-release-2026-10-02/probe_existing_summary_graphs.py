"""Exact saved-data parity for composing existing summary graphs; no sampling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
for field in ('source', 'record', 'output'):
    p.add_argument('--'+field, type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--worker', action='store_true')
a=p.parse_args()
for field in ('source', 'record', 'output'):
    setattr(a, field, getattr(a, field).resolve())
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, TF_FORCE_GPU_ALLOW_GROWTH='true',
    TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2', BAYESFILTER_PRELOAD_CUSTOM_OP='0')
os.sched_setaffinity(0, {24,25,26,27})

def write(name, data):
    (a.output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

if not a.worker:
    a.output.mkdir(parents=True,exist_ok=False)
    runner=a.output/'runner.py';runner.write_bytes(Path(__file__).read_bytes())
    command=[sys.executable,str(runner),*sys.argv[1:],'--worker']
    started=time.monotonic()
    manifest=dict(command=command,environment=sys.executable,resource='gpu',
        started_utc=datetime.now(timezone.utc).isoformat(),gpu_uuid=a.gpu,
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md',native_sampling=False)
    write('manifest.json',manifest)
    with (a.output/'run.log').open('x') as log:
        try: code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=115).returncode
        except subprocess.TimeoutExpired: code=124
    receipt={**manifest,'exit_code':code,'wall_seconds':time.monotonic()-started}
    write('receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('exit_code','wall_seconds')}))
    raise SystemExit(code)

sys.path.insert(0,str(a.source))
from scripts.run_hmc_v7_release_prices import check_source
source_hash=check_source(a.source)
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
from bayesfilter.inference.hmc_verification import _acceptance_summary_function, HMCAcceptancePolicy
from bayesfilter.inference.hmc_acceptance_statistics import _trial_summary_program
from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload, _trace_from_payload

with tf.device('/GPU:0'):
    raw=a.record.read_bytes(); record=json.loads(raw)
    health=_acceptance_summary_function(); scores=_trial_summary_program(True)
    policy=HMCAcceptancePolicy(min_normalized_return_displacement=0.)
    args=(tf.constant(policy.block_count,tf.int32),
          tf.constant(policy.min_decisions_per_chain,tf.int32),
          tf.constant(policy.chain_count,tf.int32),
          tf.constant(policy.max_abs_log_accept_energy_proxy,tf.float64))
    data=[]
    for trial in record['trials'][:64]:
        trace=_trace_from_payload(trial['trace'])
        data.append((_tensor_from_payload(trial['samples'])[3:],
            trace['log_accept_ratio'][3:],trace['is_accepted'][3:]))
    def scalar(values):
        samples,log_accept,accepted=values
        return health(samples,log_accept,accepted,*args),scores(log_accept)
    expected=[scalar(row) for row in data]
    batch_data=tf.nest.map_structure(lambda *rows:tf.stack(rows),*data)
    signature=tf.nest.map_structure(lambda v:tf.TensorSpec(v.shape,v.dtype),expected[0])

    @tf.function(input_signature=[tf.TensorSpec([None,65,4,2],tf.float64),
        tf.TensorSpec([None,65,4],tf.float64),tf.TensorSpec([None,65,4],tf.bool)],
        autograph=False,jit_compile=False)
    def grouped(samples,log_accept,accepted):
        return tf.map_fn(scalar,(samples,log_accept,accepted),
                        fn_output_signature=signature,parallel_iterations=1)

    write('worker-manifest.json',dict(source_manifest_sha256=source_hash,
        record_sha256=hashlib.sha256(raw).hexdigest(),memory_policy=memory,
        device=batch_data[0].device,tensorflow_version=tf.__version__,
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        outer_jit_compile=False,score_jit_compile=True,health_summary_jit_compile=False,
        scope='diagnostic graph orchestration; exact outputs only, no admission or sampling'))
    expected_batch=tf.nest.map_structure(lambda *rows:tf.stack(rows),*expected)
    rows=[]
    for size in (1,8,32):
        for repeat in range(3):
            for mode in (('scalar','grouped') if repeat%2==0 else ('grouped','scalar')):
                start=time.monotonic()
                if mode=='scalar':
                    results=[scalar(row) for row in data]
                    actual=tf.nest.map_structure(lambda *values:tf.stack(values),*results)
                else:
                    results=[grouped(*(v[i:i+size] for v in batch_data)) for i in range(0,64,size)]
                    actual=tf.nest.map_structure(lambda *values:tf.concat(values,0),*results)
                # Materialize before ending the measured call; validate afterward.
                materialized=tf.nest.map_structure(lambda v:v.numpy().tolist(),actual)
                elapsed=time.monotonic()-start
                for x,y in zip(tf.nest.flatten(actual),tf.nest.flatten(expected_batch)):
                    tf.debugging.assert_equal(x,y)
                rows.append(dict(batch_size=size,repeat=repeat,mode=mode,
                    wall_seconds=elapsed,exact_parity=True))
                write('calls.json',rows)
    write('result.json',dict(status='exact_existing_graph_parity_passed',trials=64,
        graph_traces=grouped.experimental_get_tracing_count(),rows=rows,
        allocator=tf.config.experimental.get_memory_info('GPU:0'),release_ready=False,
        timing_scope='descriptive; no full-procedure cost or runtime superiority claim'))
