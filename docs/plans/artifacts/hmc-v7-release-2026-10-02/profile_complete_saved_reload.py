"""Profile an actual saved member group on its bound GPU; no HMC sampling."""
import argparse
import cProfile
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import signal
import subprocess
import sys
import time


def write(path, payload):
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def worker(args):
    sys.path.insert(0,str(args.source));os.chdir(args.source)
    from scripts.run_hmc_v7_release_prices import check_source
    signature=check_source(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.inference import load_hmc_candidate_retained_runners
    config=json.loads((args.model/'configuration.json').read_text())
    target=ValidationTarget(config['target'],config['parameters'],config['data'])
    members=sorted(args.model.glob('*-member.json'))
    expected=json.loads((args.model/'tuning/tuning_checkpoint.json').read_text())['result']['verified_candidate_ids']
    write(args.output/'worker-manifest.json',dict(source_manifest_sha256=signature,
        configuration=config, memory_policy=memory,gpu_uuid=args.gpu,
        members=[str(p) for p in members],expected_ids=expected,
        tensorflow_version=tf.__version__,jit_compile=True,
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(),native_sampling=False))
    from bayesfilter.inference import hmc_candidate_set_retained as retained
    from bayesfilter.inference import hmc_acceptance_trials as trials
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionBinding
    components=defaultdict(lambda: dict(calls=0,seconds=0.0))
    originals=[]
    def instrument(owner,name):
        original=getattr(owner,name)
        originals.append((owner,name,original))
        def timed(*args,**kwargs):
            start=time.perf_counter()
            try:
                return original(*args,**kwargs)
            finally:
                components[name]['calls']+=1
                components[name]['seconds']+=time.perf_counter()-start
        setattr(owner,name,timed)
    for owner,names in [(retained,['_checked_payload','_validate_member_set','_checked_member']),
                        (trials,['_assemble_trials','validate_chunks','initialize_seed_registry','_prior','_analyze_trials']),
                        (HMCCandidateExecutionBinding,['analyze_trial','evidence_analysis'])]:
        for name in names:instrument(owner,name)
    def expired(*_):
        raise TimeoutError('bounded grouped replay profile')
    signal.signal(signal.SIGALRM,expired)
    profile=cProfile.Profile();started=time.monotonic();status='incomplete'
    try:
        signal.alarm(840);profile.enable()
        runners=load_hmc_candidate_retained_runners(members,adapter=target)
        profile.disable()
        assert set(runners)==set(expected)
        status='grouped_reload_passed'
    except TimeoutError:
        status='profile_deadline_without_complete_reload'
    finally:
        profile.disable();signal.alarm(0)
        for owner,name,original in reversed(originals):setattr(owner,name,original)
        write(args.output/'components.json',dict(components=components,scope='Inclusive times overlap; do not sum. Diagnostic instrumentation changes timing.'))
        profile.dump_stats(str(args.output/'profile.pstats'))
        stream=io.StringIO()
        pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(60)
        (args.output/'profile.txt').write_text(stream.getvalue())
        write(args.output/'result.json',dict(status=status,wall_seconds=time.monotonic()-started,
            native_sampling=False,release_ready=False,allocator=tf.config.experimental.get_memory_info('GPU:0')))
    return 0 if status=='grouped_reload_passed' else 2


def main():
    p=argparse.ArgumentParser()
    for name in ('source','model','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    for name in ('source','model','output'):setattr(args,name,getattr(args,name).resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0,{24,25,26,27})
    if args.worker:return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    runner=args.output/'runner.py';runner.write_bytes(Path(__file__).read_bytes())
    command=[sys.executable,str(runner),*sys.argv[1:],'--worker']
    started=time.monotonic()
    record=dict(command=command,started_utc=datetime.now(timezone.utc).isoformat(),
        environment=sys.executable,resource='gpu',runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        git_commit=json.loads((args.source.parent/'assembly.json').read_text())['git_commit'],
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md',
        scope='saved group profile; no native sampling or release/default/posterior claim')
    write(args.output/'manifest.json',record)
    with (args.output/'run.log').open('x') as log:
        try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=870).returncode
        except subprocess.TimeoutExpired:code=124
    record.update(exit_code=code,wall_seconds=time.monotonic()-started)
    write(args.output/'receipt.json',record);print(json.dumps({key:record[key] for key in ('exit_code','wall_seconds')}));return code


if __name__=='__main__':raise SystemExit(main())
