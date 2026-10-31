"""Bounded frozen-grid controlled-SMC qualification supervisor.

Source workers are independent reference computations. The existing TF
GPU/XLA consumer performs all candidate numerics. Inference is deferred
until all frozen pairs are complete.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import run_lorenz_downstream_qualification as common

ROOT, BASE, PYTHON = common.ROOT, common.BASE, common.PYTHON
SELF = Path(__file__).resolve()
PLAN, RESULT = common.PLAN, common.RESULT
INFERENCE = 'docs/benchmarks/csmc_qualification_inference_tf.py'
FILES = common.FILES + ('docs/benchmarks/run_lorenz_qualification.py', INFERENCE,
                         'scripts/run_controlled_smc_kdm.py')
CAPS = dict(cpu=32000.,gpu=100000.)
STOP = threading.Event()
LOCK = threading.RLock()

class CandidateRejected(Exception):
    pass

def source_spec(template, hashes, cases):
    return dict(template, phase='UNTOUCHED_QUALIFICATION', candidate_sha256=hashes,
                preserve_random_tapes=True, stop_on_additional_failure=True, cases=cases)

def freeze(output, current):
    calibration = BASE/'csmc-qualification-20261009-grid-185030'
    template=json.loads((calibration/'cell00/spec.json').read_text())
    grid=json.loads((calibration/'spec.json').read_text())['grid']
    hashes={p:common.sha(ROOT/p) for p in FILES}
    previous_seeds=set()
    previous_seeds.update(r['seed'] for r in json.loads((BASE/'csmc-qualification-20261009-165753/spec.json').read_text())['cases'])
    for path in calibration.glob('cell*/spec.json'):
        previous_seeds.update(r['seed'] for r in json.loads(path.read_text())['cases'])
    cells=[]
    all_seeds=set()
    for i,forcing in enumerate(grid):
        template['source_grid_index']=i if i<40 else None
        cases=[dict(id=i*320+j,forcing=forcing,seed=271100001+i*10000+j,control=j%32==0)
               for j in range(320)]
        seeds={r['seed'] for r in cases}
        if len(seeds)!=320 or seeds & (previous_seeds|all_seeds):
            raise ValueError('Non-independent or reused qualification seeds')
        all_seeds.update(seeds)
        cell=output/f'cell{i:02d}'
        cell.mkdir()
        spec=source_spec(template,hashes,cases)
        common.write(cell/'spec.json',spec)
        for chunk in range(10):
            child=cell/f'chunk{chunk:02d}'
            child.mkdir()
            common.write(child/'spec.json',source_spec(template,hashes,cases[chunk*32:(chunk+1)*32]))
        cells.append(cell)
    for rel in FILES:
        saved=output/'sources'/rel
        saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,saved)
    spec=dict(schema='csmc_full_grid_qualification_v1',grid=grid,pairs_per_cell=320,
        total_pairs=13120,seed_formula='271100001+10000*cell+repetition',
        independent_of_calibration=True,candidate_sha256=hashes,
        source_sha256=template['source_sha256'],octave_version=template['octave_version'],
        data_sha256={p:h for p,h in template['source_sha256'].items() if p.endswith('.mat')},
        cells={c.name:common.sha(c/'spec.json') for c in cells},plan=PLAN,
        plan_sha256=common.sha(ROOT/PLAN),qualification_margins=dict(mean_likelihood_factor=1.01,variance_ratio=1.05,additional_failures=0),
        inference=dict(bootstrap_resamples=20000,seed=[261109,'cell'],
            intervals='envelope_paired_percentile_and_influence_normal_90percent',
            variance_limit='upper_one_sided_95percent',once_at_final_n=True,
            minimum_pairs=300,maximum_likelihood_share=.05,minimum_residual_effective_count=30,
            maximum_influence_variance_share=.10),
        budgets=CAPS,maximum_cpu_workers=4,source_worker_cap_seconds=150,
        tf_cell_cap_seconds=2700,minimum_disk_bytes_at_launch=280_000_000_000,
        disk_stop_bytes=50_000_000_000,gpu_uuid=common.GPU,
        candidate='LorenzRefinement.run/centered_qr/none/FP64/GPU/XLA',
        nonclaims=['full-paper replication','original MATLAB equivalence','superiority','default-readiness'])
    common.write(output/'qualification-spec.json',spec)
    for cell in cells:
        common.verify(cell)
    return spec,cells

def combine_source(cell):
    rows,arrays=[],{}
    for chunk in sorted(cell.glob('chunk*')):
        spec=common.verify(chunk)
        summary=json.loads((chunk/'source-summary.json').read_text())
        if summary['rows_sha256']!=common.sha(chunk/'source-rows.json'):
            raise ValueError('Source row hash changed')
        chunk_rows=json.loads((chunk/'source-rows.json').read_text())
        if len(chunk_rows)!=32 or summary['pristine_controls']!=1:
            raise ValueError('Missing source control or rows')
        if [r['seed'] for r in chunk_rows]!=[c['seed'] for c in spec['cases']]:
            raise ValueError('Source seed order differs')
        rows.extend(chunk_rows)
        for name,digest in summary['arrays_sha256'].items():
            if common.sha(chunk/name)!=digest:
                raise ValueError('Source tape changed')
            if name in arrays:
                if name!='observations.bin' or arrays[name]!=digest:
                    raise ValueError('Duplicate or different source array')
            else:
                (cell/name).symlink_to(chunk/name)
                arrays[name]=digest
    common.write(cell/'source-rows.json',rows)
    common.write(cell/'source-summary.json',dict(status='PASS',runs=len(rows),
        valid=sum(r['valid'] for r in rows),arrays_sha256=arrays,
        rows_sha256=common.sha(cell/'source-rows.json'),pristine_controls=10,
        cpu_only=True,gpu_intentionally_hidden=True))

def worker(mode, destination, manifest, output, requested_limit):
    resource='gpu' if mode=='tf' else 'cpu'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=common.GPU if resource=='gpu' else '-1',
               TF_FORCE_GPU_ALLOW_GROWTH='true',OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
               BAYESFILTER_PRELOAD_CUSTOM_OP='0',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
    command=[PYTHON,'-B',str(SELF),mode,'--output',str(destination)]
    with LOCK:
        if STOP.is_set():
            raise RuntimeError('Launch cancelled after preceding failure')
        reserved=sum(e['timeout_seconds']+5 for e in manifest['executions']
                     if e['status']=='RUNNING' and e['resource']==resource)
        remaining=CAPS[resource]-manifest['charged_seconds'][resource]-reserved
        limit=min(requested_limit,remaining-5)
        if limit<60 or shutil.disk_usage(BASE).free<50_000_000_000:
            STOP.set()
            raise RuntimeError('Compute or disk continuation veto')
        log=(destination/(mode+'-worker.log')).open('x')
        started=time.monotonic()
        try:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            log.close()
            raise
        record=dict(mode=mode,resource=resource,command=command,pid=child.pid,status='RUNNING',
                    output=str(destination),started_unix_seconds=time.time(),timeout_seconds=limit)
        manifest['executions'].append(record)
        common.write(output/'manifest.json',manifest)
    returncode=None
    reason=None
    try:
        while child.poll() is None:
            if STOP.is_set() or time.monotonic()-started>limit:
                reason='CANCELLED' if STOP.is_set() else 'TIMEOUT'
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                break
            time.sleep(.25)
        returncode=child.wait()
    finally:
        log.close()
        with LOCK:
            elapsed=time.monotonic()-started
            manifest['charged_seconds'][resource]+=elapsed
            record.update(status='COMPLETE' if returncode==0 else reason or 'FAILED',
                          wall_seconds=elapsed,returncode=returncode,ended_unix_seconds=time.time())
            common.write(output/'manifest.json',manifest)
    print(json.dumps({k:record[k] for k in ('mode','output','status','wall_seconds','returncode')}),flush=True)
    if returncode==20 and mode=='tf' and (destination/'candidate-rejection.json').exists():
        STOP.set()
        raise CandidateRejected(str(destination/'candidate-rejection.json'))
    if returncode!=0:
        STOP.set()
        raise RuntimeError('Worker failed: '+str(destination))
    return record

def analyze(output):
    # Explicit CPU reference-statistics exception; CUDA hidden before TF import.
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import csmc_qualification_inference_tf as statistics
    spec=json.loads((output/'qualification-spec.json').read_text())
    for rel,digest in spec['candidate_sha256'].items():
        if common.sha(ROOT/rel)!=digest:
            raise ValueError('Frozen candidate or inference changed')
    strata=[]
    global_pass=True
    for i,cell_name in enumerate(spec['cells']):
        cell=output/cell_name
        common.verify(cell)
        source=json.loads((cell/'source-rows.json').read_text())
        port=json.loads((cell/'tf-rows.json').read_text())
        ss=json.loads((cell/'source-summary.json').read_text())
        ts=json.loads((cell/'tf-summary.json').read_text())
        if ss['rows_sha256']!=common.sha(cell/'source-rows.json') or ts['rows_sha256']!=common.sha(cell/'tf-rows.json'):
            raise ValueError('Final rows changed')
        if len(source)!=320 or len(port)!=320:
            raise ValueError('Incomplete fixed sample size')
        if [(r['id'],r['seed'],r['forcing']) for r in source] != [(r['id'],r['seed'],r['forcing']) for r in port]:
            raise ValueError('Unpaired outcomes')
        extra=[s['id'] for s,p in zip(source,port) if s['valid'] and not p['valid']]
        valid=[(s,p) for s,p in zip(source,port) if s['valid'] and p['valid']]
        inference=statistics.infer([s['terminal_logZ'] for s,p in valid],
                                   [p['terminal_logZ'] for s,p in valid],i)
        ns=sum(s['valid'] for s in source)
        pass_cell=inference.get('likelihood_pass',False) and inference.get('variance_pass',False) and not extra
        global_pass=global_pass and pass_cell
        strata.append(dict(cell=cell_name,forcing=source[0]['forcing'],attempted=320,source_valid=ns,
            port_valid=sum(p['valid'] for p in port),valid_pairs=len(valid),
            conditional_valid_pair_estimand=ns<320,
            source_failure_ids=[s['id'] for s in source if not s['valid']],
            matched_failure_ids=[s['id'] for s,p in zip(source,port) if not s['valid'] and not p['valid']],
            additional_port_failure_ids=extra,pointwise_criteria_met=pass_cell,inference=inference,
            extra_failure_upper95=1-.05**(1/ns) if not extra and ns else None,
            stopping_counts_source={str(k):sum(s['valid'] and s['stage_count']==k for s in source) for k in range(2,6)},
            stopping_counts_port={str(k):sum(p['valid'] and p['stage_count']==k for p in port) for k in range(2,6)}))
    summary=dict(status='PASS' if global_pass else 'NOT_QUALIFIED',replication='INCOMPLETE',
        scope='FROZEN_ORIGINAL_LORENZ_GRID_PLUS_GENERATING_PARAMETER',
        scientific_verdict='GLOBAL_CONDITIONAL_VALID_CASE_CRITERIA_MET' if global_pass else 'GLOBAL_CRITERIA_NOT_ESTABLISHED',
        global_criteria_met=global_pass,unconditional_likelihood_claim=all(s['source_valid']==320 for s in strata),
        no_superiority_claim=True,qualification_spec_sha256=common.sha(output/'qualification-spec.json'),
        strata=strata,cpu_only_statistics=True,gpu_intentionally_hidden_for_statistics=True)
    common.write(output/'summary.json',summary)

def supervise(output):
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_controlled_smc_kdm as master
    current=master.status()
    if current['running']:
        raise RuntimeError('Existing campaign run')
    previous=[r for r in current['csmc_qualification'] if r['stage']=='CSMC_UNTOUCHED_QUALIFICATION']
    for resource in CAPS:
        CAPS[resource]-=sum(r['used_seconds'][resource] for r in previous)
        if CAPS[resource]<=0 or current['tranche_remaining_seconds'][resource]<CAPS[resource]:
            raise RuntimeError('Insufficient cumulative qualification budget')
    if shutil.disk_usage(BASE).free<280_000_000_000:
        raise RuntimeError('Insufficient storage for preserved tapes')
    output.mkdir(exist_ok=False)
    spec,cells=freeze(output,current)
    manifest=dict(status='RUNNING',stage='CSMC_UNTOUCHED_QUALIFICATION',resource='mixed',
        started_unix_seconds=time.time(),supervisor_pid=os.getpid(),command=[PYTHON,'-B',str(SELF),'run','--output',str(output)],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment=PYTHON,plan=PLAN,result_file=RESULT,qualification_spec_sha256=common.sha(output/'qualification-spec.json'),
        output_root=str(output),executions=[],charged_seconds=dict(cpu=0.,gpu=0.),phase_budget_seconds=CAPS,
        completed_cells=0,planned_cells=41,planned_pairs=13120,maximum_cpu_workers=4)
    common.write(output/'manifest.json',manifest)
    master.refresh_checkpoint()
    try:
        for cell in cells:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(worker,'source',child,manifest,output,150) for child in sorted(cell.glob('chunk*'))]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                except BaseException:
                    STOP.set()
                    raise
            combine_source(cell)
            worker('tf',cell,manifest,output,2700)
            manifest['completed_cells']+=1
            common.write(output/'manifest.json',manifest)
            master.refresh_checkpoint()
        worker('analyze',output,manifest,output,600)
        manifest['status']='COMPLETE'
    except CandidateRejected as exc:
        manifest.update(status='CANDIDATE_REJECTED',rejection=str(exc))
        common.write(output/'summary.json',dict(status='CANDIDATE_REJECTED',replication='INCOMPLETE',
            scope='PARTIAL_QUALIFICATION_NO_CONFIDENCE_INTERVALS',reason='ADDITIONAL_TF_FAILURE',rejection=str(exc),
            next_action='Q3 localized repair; used seeds are now development'))
    except BaseException as exc:
        manifest.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
        common.write(output/'summary.json',dict(status='FAILED',replication='INCOMPLETE',
            scope='PARTIAL_QUALIFICATION_NO_CONFIDENCE_INTERVALS',error=str(exc),next_action='Inspect preserved logs and repair within remaining caps'))
    finally:
        STOP.set()
        manifest.update(ended_unix_seconds=time.time(),wall_seconds=time.time()-manifest['started_unix_seconds'])
        common.write(output/'manifest.json',manifest)
        ledger_path=BASE/'reference-diagnostics.json'
        ledger=json.loads(ledger_path.read_text())
        for resource in ('cpu','gpu'):
            ledger.append(dict(id=output.name+'_'+resource,resource=resource,charged_seconds=manifest['charged_seconds'][resource],
                summary=str((output/'summary.json').relative_to(ROOT)),summary_sha256=common.sha(output/'summary.json'),
                plan_file=PLAN,result_file=RESULT,stage=manifest['stage']+'_'+manifest['status'],
                next_action='Inspect full frozen inference if complete; otherwise follow the recorded repair trigger.',
                after_attempt=current['attempts'],executions=[dict(path=str(output.relative_to(ROOT)),resource=resource,
                    wall_seconds=manifest['charged_seconds'][resource])]))
        common.write(ledger_path,ledger)
        master.refresh_checkpoint()
    print(json.dumps(dict(status=manifest['status'],charged_seconds=manifest['charged_seconds'],output=str(output))),flush=True)
    return 0 if manifest['status']=='COMPLETE' else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','source','tf','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(BASE) or (args.mode=='run' and output.parent!=BASE):
        parser.error('Output outside campaign root')
    if args.mode=='run':
        raise SystemExit(supervise(output))
    {'source':common.source,'tf':common.port,'analyze':analyze}[args.mode](output)

if __name__=='__main__':
    main()
