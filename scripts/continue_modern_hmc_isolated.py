"""Fresh, source-isolated MacroFinance diagnostic within an existing allocation.

Preflight and prices are engineering diagnostics. A bounded qualification
tranche preserves the full candidate/statistical design without promising that
it finishes. Historical source-invalid checkpoints are never imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts.modern_hmc_source_snapshot import capture, validate_snapshot, audit_imports

PLAN = 'docs/plans/bayesfilter-modern-hmc-isolated-continuation-2026-10-26.md'
INPUT = 'docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/residual-qualification-20261024-01/macrofinance-inputs.json'
PARENT = ROOT/'docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/diagnostic-extension-20261025-01'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def stage_budget(remaining, requested, reserve=0.):
    if (not all(math.isfinite(v) for v in (remaining,requested,reserve))
            or reserve < 0 or remaining <= reserve or requested <= 0):
        raise ValueError('no funded work remains after reserves')
    return min(requested, remaining-reserve)


def prepare(output, *, plan=PLAN, qualification_seed=(20261026,6101),
            pricing_seed=(20261026,5100), reload_seed=(20261026,2001)):
    plan_path = (ROOT/plan).resolve()
    if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
        raise ValueError('plan must be an existing repository file')
    for seed in (qualification_seed, pricing_seed, reload_seed):
        if len(seed) != 2 or any(type(v) is not int or not 0 <= v < 2**31-100 for v in seed):
            raise ValueError('seed namespaces require two bounded nonnegative int32 components')
    if len({tuple(seed) for seed in (qualification_seed, pricing_seed, reload_seed)}) != 3:
        raise ValueError('qualification, pricing and reload seed namespaces must differ')
    output.mkdir(parents=True, exist_ok=False)
    manifest = capture(ROOT, '/home/ubuntu/python/MacroFinance', output/'source',
        extra_paths=[INPUT, str(plan_path.relative_to(ROOT)), 'AGENTS.md'])
    write(output/'isolated-source.json', manifest)
    cohort = read(PARENT/'frozen-cohort.json')
    cohort.update(qualification_seed=list(qualification_seed), reload_seed=list(reload_seed),
        pricing_seed=list(pricing_seed),
        prior_run=str(PARENT/'qualification-03'),
        source_semantics='fresh current rejection checks; old evidence not requalified',
        prior_veto_rule='Historical vetoes remain recorded; no posterior or default promotion')
    write(output/'frozen-cohort.json', cohort)
    write(output/'proposal-design.json', read(PARENT/'proposal-design.json'))
    write(output/'preparation.json', dict(plan=str(plan_path.relative_to(ROOT)), source=manifest['source_root'],
        created_utc=datetime.now(timezone.utc).isoformat(),
        historical_evidence_reused=False, completion_promised=False))
    return manifest


def price(campaign):
    from scripts import run_modern_hmc_macrofinance_frozen_diagnostic as frozen
    from scripts.diagnose_modern_hmc_macrofinance_public import build
    from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus, selected_nvidia_gpu
    if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true' or not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):
        raise ValueError('trusted selected GPU and memory growth required before import')
    out = campaign/'price-01/evidence'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = read(campaign/'isolated-source.json')
    paths = validate_snapshot(source, expected_root=ROOT)
    cohort = read(campaign/'frozen-cohort.json')
    plan = read(campaign/'preparation.json')['plan'] if (campaign/'preparation.json').exists() else PLAN
    pricing_seed = cohort.get('pricing_seed', [20261026,5100])
    inputs = ROOT/INPUT
    adapter, memory = build(read(inputs), device='GPU', gpu_evidence=campaign/'preflight/evidence')
    import tensorflow as tf
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateRecord
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload, _trace_payload
    bound = frozen.frozen_binding(adapter, read(inputs), paths, read(campaign/'proposal-design.json'), cohort)
    manifest = dict(plan=plan, source_sha256=source['source_sha256'], git_commit=source['git_commit'],
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        physical_gpu=selected_nvidia_gpu(query_nvidia_smi_gpus()), memory_policy=memory,
        dtype='float64', use_xla=True, tf32_enabled=False, data_version='unchanged 48-period synthetic fixture',
        loaded_local_modules=audit_imports(source), qualification_evidence=False)
    write(out/'manifest.json', manifest)
    rows = []
    for ordinal, (length, epsilon) in enumerate(((l,e) for l,eps in frozen.pairs(cohort) for e in eps), 1):
        candidate = HMCTuningCandidateRecord.create(bound.scope, leapfrog_steps=length,
            epsilon=epsilon, creation_ordinal=ordinal)
        # One cold call, then two complete warm prices at each declared pair.
        for repeat in range(3 if ordinal == 1 else 2):
            seed = (pricing_seed[0], pricing_seed[1]+ordinal*4+repeat)
            write(out/'progress.json', dict(L=length,epsilon=epsilon,repeat=repeat,completed_calls=len(rows)))
            start = time.monotonic()
            result = bound._run(candidate,bound.initial_active_state,64,seed)
            failures = list(bound.health_failures(bound.initial_active_state,result.samples,result.trace))
            if 'device:GPU:' not in result.samples.device or not result.metadata['use_xla']:
                raise ValueError('full-trial price did not execute GPU/XLA')
            row = dict(L=length,epsilon=epsilon,repeat=repeat,seed=seed,num_results=64,chain_count=4,
                discarded_prefix=0,transition_profile='hmc_uniform_duration',cold_call=ordinal==1 and repeat==0,
                elapsed_seconds=time.monotonic()-start,health_failures=failures,
                native_seconds=result.metadata['sample_chain_call_s'],qualification_evidence=False)
            write(out/f'trial-{len(rows):02}.json',dict(**row,samples=_tensor_payload(result.samples),
                trace=_trace_payload(result.trace)))
            rows.append(row)
            write(out/'progress.json',dict(rows=rows))
    validate_snapshot(source, expected_root=ROOT)
    write(out/'result.json',dict(status='completed',source_unchanged=True,
        execution=dict(rows=[]),elapsed_seconds=time.monotonic()-started,
        loaded_local_modules=audit_imports(source),allocator_bytes=tf.config.experimental.get_memory_info('GPU:0')))
    # Compatibility with the existing all-pair price validator: put all fresh
    # prices in its nominee input. It ignores the explicitly cold call.
    write(campaign/'nominee-price-01/result.json',dict(rows=rows,status='completed',qualification_evidence=False))
    return 0


def run(campaign, seconds):
    source = read(campaign/'isolated-source.json')
    plan = read(campaign/'preparation.json')['plan'] if (campaign/'preparation.json').exists() else PLAN
    validate_snapshot(source, expected_root=ROOT)
    from scripts.run_modern_hmc_filter_health_gpu import Reassessment, stop_worker
    from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus
    env = dict(os.environ, PYTHONPATH=str(ROOT),
        BAYESFILTER_DIAGNOSTIC_PLAN=plan,
        BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT=source['consumer_root'],
        BAYESFILTER_DIAGNOSTIC_GPU_EVIDENCE=str(campaign/'preflight/evidence'),
        TF_FORCE_GPU_ALLOW_GROWTH='true', PYTHONDONTWRITEBYTECODE='1', TF_CPP_MIN_LOG_LEVEL='2',
        TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', OMP_NUM_THREADS='2')
    started = time.monotonic()
    manifest = dict(status='running',active_run=True,plan=plan,command=[sys.executable,*sys.argv],
        git_commit=source['git_commit'],max_wall_seconds=seconds,stages=[],
        started_utc=datetime.now(timezone.utc).isoformat(),source_manifest=str(campaign/'isolated-source.json'))
    write(campaign/'manifest.json',manifest)

    def execute(name, command, cap, extra_env=None):
        began = time.monotonic()
        receipt = dict(stage=name, command=command, cap_seconds=cap, status='running')
        manifest['stages'].append(receipt)
        child = None
        try:
            with (campaign/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command,cwd=ROOT,env=env | (extra_env or {}),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                receipt['pid']=child.pid
                while child.poll() is None:
                    elapsed=time.monotonic()-began
                    manifest.update(heartbeat_utc=datetime.now(timezone.utc).isoformat(),
                                    elapsed_seconds=time.monotonic()-started)
                    write(campaign/'manifest.json',manifest)
                    if elapsed >= cap:
                        stop_worker(child)
                        raise TimeoutError(name+' inclusive cap exhausted')
                    time.sleep(2)
                receipt.update(exit_code=child.returncode,status='completed' if child.returncode==0 else 'stopped')
                return child.returncode
        finally:
            stop_worker(child)
            receipt['elapsed_seconds']=time.monotonic()-began
            write(campaign/'manifest.json',manifest)

    try:
        # Waiting for shared capacity is charged to the total allocation; only
        # the numerical preflight has the short debugging ceiling.
        cap=stage_budget(seconds,seconds,1020.)
        command=[sys.executable,str(ROOT/'scripts/run_modern_hmc_filter_health_gpu.py'),
            '--inputs',str(ROOT/INPUT),'--output',str(campaign/'preflight'),
            '--max-wall-seconds',str(cap-10),'--worker-wall-seconds','600',
            '--task','filter-health']
        if execute('preflight',command,cap):
            raise RuntimeError('fresh GPU preflight failed; inspect before qualification')
        remaining=seconds-(time.monotonic()-started)
        cap=stage_budget(remaining,1800.,1020.)
        monitor=Reassessment(); wait_start=time.monotonic(); selected=None
        while selected is None:
            now=time.monotonic()
            if now-wait_start>=cap:
                raise TimeoutError('pricing capacity wait exhausted')
            if monitor.due(now,waiting=True):
                observation=monitor.observe(now,query_nvidia_smi_gpus())
                write(campaign/'pricing-device.json',observation)
                selected=observation['selected_for_next_worker']
            if selected is None:time.sleep(1)
        cap-=time.monotonic()-wait_start
        if execute('pricing',[sys.executable,str(Path(__file__).resolve()),'--price','--campaign',str(campaign)],
                   cap,dict(CUDA_VISIBLE_DEVICES=selected['uuid'])):
            raise RuntimeError('fresh all-pair pricing failed; inspect before qualification')
        remaining=seconds-(time.monotonic()-started)
        if remaining<=1020:
            raise TimeoutError('allocation exhausted before tuning')
        command=[sys.executable,str(ROOT/'scripts/run_modern_hmc_macrofinance_frozen_diagnostic.py'),
            '--campaign',str(campaign),'--output',str(campaign/'qualification-01'),
            '--wall-seconds',str(remaining-10),'--allow-partial-budget']
        execute('qualification',command,remaining)
        outcome=read(campaign/'qualification-01/manifest.json')
        manifest['status']=outcome['status']
        manifest['qualification_result']=str(campaign/'qualification-01/evidence/result.json')
        validate_snapshot(source,expected_root=ROOT)
    except Exception as exc:
        manifest.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    finally:
        manifest.update(active_run=False,elapsed_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write(campaign/'manifest.json',manifest)
    return 0 if manifest['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--price',action='store_true')
    parser.add_argument('--wall-seconds',type=float)
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--qualification-seed', type=int, nargs=2, default=(20261026,6101))
    parser.add_argument('--pricing-seed', type=int, nargs=2, default=(20261026,5100))
    parser.add_argument('--reload-seed', type=int, nargs=2, default=(20261026,2001))
    args=parser.parse_args()
    args.campaign=args.campaign.resolve()
    if args.prepare:
        source=prepare(args.campaign, plan=args.plan, qualification_seed=args.qualification_seed,
                       pricing_seed=args.pricing_seed, reload_seed=args.reload_seed)
        print(json.dumps(dict(campaign=str(args.campaign),source=source['source_root'],
                              source_file_count=len(source['source_sha256']))))
        return 0
    if args.price:
        sys.path.insert(0,os.environ['BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT'])
        return price(args.campaign)
    if args.wall_seconds is None or not math.isfinite(args.wall_seconds) or args.wall_seconds<=1020:
        parser.error('funded total allowance must exceed closeout reserves')
    return run(args.campaign,args.wall_seconds)


if __name__=='__main__':
    raise SystemExit(main())
