"""Instrument one real public-tuner work item; no full-price/release claim."""
import argparse
from collections import defaultdict
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


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def worker(args):
    sys.path.insert(0,str(args.source))
    os.chdir(args.source)
    from scripts.run_hmc_v7_release_prices import check_source
    signature=check_source(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    from bayesfilter.testing.acceptance_decision_models import run_model
    import bayesfilter.inference as api
    from bayesfilter.inference import hmc_acceptance_trials as trials
    from bayesfilter.inference import hmc_candidate_set_execution as execution
    from bayesfilter.inference import hmc_candidate_set_checkpoint as checkpoint
    from bayesfilter.inference import hmc_candidate_set_tuning as controller

    configuration=full_search_configuration('nonlinear',seed=(20261002,2502),
        wall_seconds=1800,replicated_trial_batch_size=32)
    manifest=dict(source_manifest_sha256=signature,configuration=configuration,
        gpu_uuid=args.gpu,memory_policy=memory,tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        jit_compile=True,dtype='float64',tensorflow_version=tf.__version__,
        CPU_affinity=sorted(os.sched_getaffinity(0)),original_base_seed=[20261002,2502],
        max_work_items=1,mode=args.mode,scope='intentional partial-search diagnostic',
        started_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'worker-manifest.json',manifest)
    public=api.tune_hmc_kernel
    def bounded(**kwargs):
        return public(**kwargs,max_work_items=1)
    api.tune_hmc_kernel=bounded
    spans=defaultdict(lambda:dict(calls=0,seconds=0.0))
    originals=[]
    def instrument(owner,name):
        original=getattr(owner,name)
        originals.append((owner,name,original))
        def timed(*pos,**kw):
            before=time.monotonic()
            try:return original(*pos,**kw)
            finally:
                spans[name]['calls']+=1
                spans[name]['seconds']+=time.monotonic()-before
        setattr(owner,name,timed)
    if args.mode=='instrumented':
        for owner,names in (
            (trials,['observe','validate_chunks','_assemble_trials','_analyze_trials','_prior']),
            (execution,['_report_rhat','_json_copy','_tensor_payload','_trace_payload']),
            (execution.HMCCandidateExecutionBinding,['_run_replicated_batch','analyze_trial','health_failures']),
            (checkpoint,['write_numerical_tuning_checkpoint']),
            (controller.HMCTuningCandidateSetController,['_apply_observation'])):
            for name in names:instrument(owner,name)
    profile=cProfile.Profile()
    before=time.monotonic()
    try:
        if args.mode=='instrumented':profile.enable()
        result=run_model(configuration,args.output/'model',manifest=manifest)
    finally:
        profile.disable()
        for owner,name,original in reversed(originals):setattr(owner,name,original)
        api.tune_hmc_kernel=public
        write(args.output/'components.json',dict(components=dict(spans),
            interpretation='Inclusive nested spans overlap; do not sum them.'))
        if args.mode=='instrumented':
            profile.dump_stats(str(args.output/'profile.pstats'))
            report=io.StringIO()
            pstats.Stats(profile,stream=report).sort_stats('cumulative').print_stats(40)
            (args.output/'profile.txt').write_text(report.getvalue())
    assert result['completion_status']=='partial_budget'
    assert result['checkpoint_recomputed'] is True
    assert result['evidence_accounting']['unique_complete_trials']==32
    assert result['evidence_accounting']['attempted_work_outside_complete_trials']==0
    devices=set()
    for f in (args.output/'model/tuning/numerical_chunks').glob('*.json'):
        chunk=json.loads(f.read_text())
        devices.add(chunk['samples_device'])
        assert chunk['runtime']['jit_compile'] and chunk['runtime']['use_xla']
    assert devices and all(d.endswith('/device:GPU:0') for d in devices)
    write(args.output/'result.json',dict(status='checked_partial_work',
        mode=args.mode,wall_seconds=time.monotonic()-before,sample_devices=sorted(devices),
        release_ready=False,complete_search_price=False))


def compared_result(root):
    def outputs(mode):
        tuning=root/mode/'model/tuning'
        saved=json.loads((tuning/'tuning_checkpoint.json').read_text())
        hashes=saved['numerical_evidence_hashes']
        assert len(hashes)==1
        raw=json.loads((tuning/'numerical_evidence'/(hashes[0]+'.json')).read_text())
        trials=[{k:t[k] for k in ('ordinal','seed','samples','trace','health','scores')}
                for t in raw['trials']]
        return dict(trials=trials,work=raw['work'],candidate=raw['candidate'],
            analysis=raw['analysis'],rhat=raw['rhat_reporting_only'],
            states=saved['result']['candidate_states'])
    assert outputs('plain')==outputs('instrumented'), 'instrumentation changed numerical output'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--mode',choices=['plain','instrumented'])
    args=p.parse_args()
    args.source,args.output=args.source.resolve(),args.output.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0,{8,9,10,11})
    if args.mode:
        worker(args)
        return 0
    args.output.mkdir(parents=True,exist_ok=False)
    runner=args.output/'runner.py'
    runner.write_bytes(Path(__file__).read_bytes())
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),environment=sys.executable,
        git_commit=json.loads((args.source.parent/'assembly.json').read_text())['git_commit'],
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md',
        gpu_uuid=args.gpu,resource='gpu',budget_seconds=400,attempts=[],
        source_manifest_sha256=hashlib.sha256((args.source.parent/'source-manifest.json').read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),release_ready=False)
    write(args.output/'manifest.json',record)
    started=time.monotonic()
    for mode in ('plain','instrumented'):
        slot=args.output/mode;slot.mkdir()
        command=[sys.executable,str(runner),'--source',str(args.source),'--output',str(slot),
                 '--gpu',args.gpu,'--mode',mode]
        before=time.monotonic()
        with (slot/'run.log').open('x') as log:
            try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:code=124
        record['attempts'].append(dict(mode=mode,command=command,exit_code=code,
                                      wall_seconds=time.monotonic()-before))
        write(args.output/'progress.json',record)
        if code:break
    status='incomplete_diagnostic'
    if len(record['attempts'])==2 and all(a['exit_code']==0 for a in record['attempts']):
        try:compared_result(args.output)
        except (AssertionError,KeyError,ValueError,OSError) as exc:
            status='exact_parity_failed';record['comparison_error']=str(exc)
        else:status='exact_public_work_parity'
    record.update(status=status,wall_seconds=time.monotonic()-started)
    write(args.output/'result.json',record)
    print(json.dumps({k:record[k] for k in ('status','wall_seconds')}))
    return 0 if status=='exact_public_work_parity' else 2


if __name__=='__main__':raise SystemExit(main())
