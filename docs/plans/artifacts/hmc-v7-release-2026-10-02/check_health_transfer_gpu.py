"""Trusted GPU parity and bounded timing for the health-transfer repair."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def worker(args):
    sys.path.insert(0,str(args.source));os.chdir(args.source)
    from scripts.run_hmc_v7_release_prices import check_source
    signature=check_source(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    from tests.test_hmc_health_transfer import reference_namespace
    from bayesfilter.inference import hmc_verification as verification
    from bayesfilter.inference.hmc_candidate_set_execution import (
        HMCCandidateExecutionBinding,_tensor_from_payload,_trace_from_payload)
    from types import SimpleNamespace
    import pytest
    write(args.output/'worker-manifest.json',dict(source_manifest_sha256=signature,memory_policy=memory,
        gpu_uuid=args.gpu,tensorflow_version=tf.__version__,jit_compile=True,
        test_device_scope=os.environ['BAYESFILTER_TEST_DEVICE_SCOPE'],
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        data='first 32 lexically sorted saved QR chunk paths',random_seeds='original chunk seeds; no new HMC sampling',
        reference='frozen source-08 health functions; raw trace comparison only'))
    with tf.device('/GPU:0'):
        code=pytest.main(['-q','tests/test_hmc_health_transfer.py',
            '--junitxml='+str(args.output/'tests.xml')])
    if code:return code
    reference=reference_namespace()
    files=sorted(args.chunks.glob('*.json'))[:32]
    assert len(files)==32
    # Every member uses these existing explicit release-health thresholds.
    policy=verification.HMCAcceptancePolicy(min_normalized_return_displacement=0.)
    rows=[]
    for index,path in enumerate(files):
        raw=path.read_bytes();chunk=json.loads(raw)
        with tf.device('/GPU:0'):
            initial=tf.identity(_tensor_from_payload(chunk['initial_state']))
            samples=tf.identity(_tensor_from_payload(chunk['samples']))
            trace=tf.nest.map_structure(tf.identity,_trace_from_payload(chunk['trace']))
            binding=SimpleNamespace(initial_active_state=initial,
                config=SimpleNamespace(use_xla=True,target_status_trace_policy='per_chain_step'))
            kwargs=dict(samples=samples[3:],log_accept_ratio=trace['log_accept_ratio'][3:],
                is_accepted=trace['is_accepted'][3:],target_log_prob=trace['target_log_prob'][3:],policy=policy)
            outcomes={};timing={}
            for method in (('old','new') if index%2==0 else ('new','old')):
                before=time.monotonic()
                if method=='old':
                    reasons=reference['health_failures'](binding,initial,samples,trace)
                    health=reference['_evaluate_hmc_health_context'](**kwargs)
                    health=health.evidence if isinstance(health,verification._AcceptanceHealthContext) else health
                else:
                    reasons=HMCCandidateExecutionBinding.health_failures(binding,initial,samples,trace)
                    health=verification.evaluate_hmc_trial_health(**kwargs)
                outcomes[method]=(reasons,health.payload())
                timing[method]=time.monotonic()-before
            assert outcomes['old']==outcomes['new'],path.name
            rows.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                seed=chunk['seed'],sample_device=samples.device,timing_seconds=timing,exact_parity=True))
            write(args.output/'trace-checks.json',rows)
    write(args.output/'result.json',dict(status='exact_health_parity_passed',checked_traces=len(rows),
        total_reference_seconds=sum(r['timing_seconds']['old'] for r in rows),
        total_repaired_seconds=sum(r['timing_seconds']['new'] for r in rows),
        allocator=tf.config.experimental.get_memory_info('GPU:0'),
        timing_scope='descriptive paired diagnostic; not a full replay price or runtime guarantee',
        native_sampling=False,release_ready=False))
    return 0


def main():
    p=argparse.ArgumentParser()
    for name in ('source','output','chunks'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    for name in ('source','output','chunks'):setattr(args,name,getattr(args,name).resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        BAYESFILTER_TEST_DEVICE_SCOPE='visible',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0,{24,25,26,27})
    if args.worker:return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    runner=args.output/'runner.py';runner.write_bytes(Path(__file__).read_bytes())
    command=[sys.executable,str(runner),*sys.argv[1:],'--worker'];started=time.monotonic()
    record=dict(command=command,environment=sys.executable,resource='gpu',
        started_utc=datetime.now(timezone.utc).isoformat(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        git_commit=json.loads((args.source.parent/'assembly.json').read_text())['git_commit'],
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md')
    write(args.output/'manifest.json',record)
    with (args.output/'run.log').open('x') as log:
        try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=115).returncode
        except subprocess.TimeoutExpired:code=124
    record.update(exit_code=code,wall_seconds=time.monotonic()-started)
    write(args.output/'receipt.json',record);print(json.dumps(record));return code


if __name__=='__main__':raise SystemExit(main())
