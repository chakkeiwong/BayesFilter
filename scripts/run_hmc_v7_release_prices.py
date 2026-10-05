"""Price frozen full-search v7 cases on one explicitly selected shared GPU.

This is development pricing only. It neither launches confirmation nor marks a
release. Every slot, failure and enclosing cost is retained in a fresh root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEVELOPMENT_SEEDS = {
    'lgssm_qr': (20261002, 2501),
    'nonlinear': (20261002, 2502),
    'funnel_residual': (20261002, 2503),
}


def development_seeds(cases):
    """Preserve original development streams when pricing a subset or order."""
    if not cases or len(cases) != len(set(cases)):
        raise ValueError('development cases must be nonempty and unique')
    if any(case not in DEVELOPMENT_SEEDS for case in cases):
        raise ValueError('unknown development case')
    return {case: DEVELOPMENT_SEEDS[case] for case in cases}


def validate_price_budget(cases, wall_seconds, closeout_seconds, budget_seconds):
    if not 1 <= wall_seconds <= 3600 or not 1 <= closeout_seconds <= 3600:
        raise ValueError('search and closeout caps must be in [1, 3600] seconds')
    if budget_seconds < len(cases)*(wall_seconds+closeout_seconds):
        raise ValueError('fund the full development slot inventory including closeout')


def write(path, payload):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def check_source(source):
    manifest = source.parent/'source-manifest.json'
    records = json.loads(manifest.read_text())
    for name, expected in records.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen release source changed: '+name)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def full_search_delivered(result):
    """A checked member from an interrupted search does not price full work."""
    exported = result.get('verified_candidate_ids', [])
    verified = {cid for cid, state in result.get('candidate_states', {}).items()
                if state == 'verified'}
    return (result.get('completion_status') == 'complete'
            and result.get('checkpoint_recomputed') is True
            and result.get('expectation_met') is True
            and bool(exported) and len(exported) == len(set(exported))
            and set(exported) == verified)


def worker(args):
    root,source = args.output.resolve(),args.source.resolve()
    started = time.monotonic()
    signature = check_source(source)
    sys.path.insert(0,str(source));os.chdir(source)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu
    assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') == 'true'
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'],text=True)
    rows = [row for row in inventory.splitlines() if row.split(',')[0].strip() == args.gpu]
    if len(rows) != 1 or int(rows[0].split(',')[1]) < 4096:
        write(root/'result.json',{'status':'resource_deferred','wall_seconds':time.monotonic()-started,
                                 'inventory':inventory.splitlines(),'gpu_initialized':False})
        return 3
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    from bayesfilter.testing.acceptance_decision_models import run_model
    config = json.loads(args.config.read_text())
    manifest = {'command':sys.argv,'environment':sys.executable,'tensorflow_version':tf.__version__,
        'git_commit':json.loads((source.parent/'assembly.json').read_text())['git_commit'],
        'source':str(source),'source_manifest_sha256':signature,
        'gpu_uuid':args.gpu,'gpu_inventory':inventory.splitlines(),'memory_policy':memory,
        'tf32':tf.config.experimental.tensor_float_32_execution_enabled(),
        'jit_compile':True,'dtype':'float64','seed':config['seed'],'data':config['data'],
        'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'plan':'docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md',
        'configuration_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'development full-search price, no posterior/default/release claim',
        'started_utc':datetime.now(timezone.utc).isoformat()}
    write(root/'manifest.json',manifest)
    result = run_model(config,root/'model',manifest=manifest)
    devices = set()
    # Check persisted attempted native chunks, even when no member was exported.
    for path in (root/'model/tuning/numerical_chunks').glob('*.json'):
        chunk = json.loads(path.read_text())
        devices.add(chunk['samples_device'])
        assert chunk['runtime']['jit_compile'] and chunk['runtime']['use_xla']
    assert devices and all('GPU:0' in device for device in devices), devices
    assert result['checkpoint_recomputed']
    complete = full_search_delivered(result)
    write(root/'result.json',{'status':('complete' if complete else
        'partial_delivery' if result['verified_candidate_ids'] else 'not_delivered'),
        'full_search_delivered':complete,
        'positive_member_delivery':bool(result['verified_candidate_ids']),
        'expectation_met':result['expectation_met'],'completion_status':result['completion_status'],
        'verified_members':len(result['verified_candidate_ids']),'candidates':len(result['candidates']),
        'repairs':len(result['repair_actions']),'sample_devices':sorted(devices),
        'allocator':tf.config.experimental.get_memory_info('GPU:0'),
        'wall_seconds':time.monotonic()-started,'evidence_accounting':result['evidence_accounting'],
        'default_promotion':False,'release_ready':False})
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--wall-seconds',type=int,default=1800)
    parser.add_argument('--closeout-seconds',type=int,default=180)
    parser.add_argument('--budget-seconds',type=int,default=6000)
    parser.add_argument('--trial-batch-size',type=int,default=1)
    parser.add_argument('--cases',nargs='+',default=['lgssm_qr','nonlinear','funnel_residual'])
    parser.add_argument('--config',type=Path)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0,{8,9,10,11})
    if args.config:
        return worker(args)
    seeds = development_seeds(args.cases)
    validate_price_budget(args.cases,args.wall_seconds,args.closeout_seconds,args.budget_seconds)
    root,source = args.output.resolve(),args.source.resolve()
    root.mkdir(parents=True,exist_ok=False)
    frozen_runner = root/'runner.py'
    frozen_runner.write_bytes(Path(__file__).read_bytes())
    signature = check_source(source)
    sys.path.insert(0,str(source))
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    configs = {}
    for case in args.cases:
        path = root/(case+'-config.json')
        write(path,full_search_configuration(case,seed=seeds[case],wall_seconds=args.wall_seconds,
                                            replicated_trial_batch_size=args.trial_batch_size))
        configs[case] = path
    record = {'schema':'bayesfilter.hmc_v7_full_search_prices.v1','cases':args.cases,
        'source_manifest_sha256':signature,'gpu_uuid':args.gpu,'budget_seconds':args.budget_seconds,
        'search_cap_seconds':args.wall_seconds,'closeout_cap_seconds':args.closeout_seconds,
        'attempts':[],'status':'running','release_ready':False,'confirmation_authorized_by_price':False}
    write(root/'progress.json',record)
    started = time.monotonic()
    for case in args.cases:
        remaining = args.budget_seconds-(time.monotonic()-started)
        if remaining < args.wall_seconds+args.closeout_seconds:
            record['attempts'].append({'case':case,'status':'budget_deferred','wall_seconds':0.})
            continue
        slot = root/case
        slot.mkdir(exist_ok=False)
        command = [sys.executable,str(frozen_runner),'--source',str(source),
            '--output',str(slot),'--gpu',args.gpu,'--config',str(configs[case])]
        before = time.monotonic()
        with (slot/'run.log').open('x') as log:
            try:
                code = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=args.wall_seconds+args.closeout_seconds).returncode
            except subprocess.TimeoutExpired:
                code = 124
        attempt = {'case':case,'command':command,'exit_code':code,'wall_seconds':time.monotonic()-before}
        result_path = slot/'result.json'
        if result_path.exists():
            attempt['result'] = json.loads(result_path.read_text())
        record['attempts'].append(attempt)
        write(root/'progress.json',record)
        # Unknown code/import failures invalidate the harness. Keep remaining
        # slots explicitly unstarted; do not repeat an invalid experiment.
        if code not in (0,2,3,124):
            record['status'] = 'harness_failure'
            break
    record['wall_seconds'] = time.monotonic()-started
    record['unstarted_cases'] = [case for case in args.cases if case not in {a['case'] for a in record['attempts']}]
    if record['status'] == 'running':
        record['status'] = 'complete' if all(a.get('exit_code') == 0 for a in record['attempts']) else 'incomplete_prices'
    record['point_forecast_32_per_family_seconds'] = (
        32*sum(a['wall_seconds'] for a in record['attempts']) if record['status']=='complete' else None)
    record['forecast_scope'] = 'descriptive single-price extrapolation; not a runtime guarantee or confirmation design'
    write(root/'result.json',record);write(root/'progress.json',record)
    print(json.dumps({'status':record['status'],'wall_seconds':record['wall_seconds']}))
    return 0 if record['status']=='complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
