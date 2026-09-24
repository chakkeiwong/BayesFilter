#!/usr/bin/env python3
"""A11 diagnostic campaign: protected-chart pair fitting and exact physical IS.

TensorFlow/GPU/XLA numerical kernels; explicit host setup, CPU TT-SVD and
reference/reporting exceptions. No NumPy runtime or full-gradient claim.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from docs.benchmarks import run_observation_tt_robust_guide as a10
from docs.benchmarks import run_observation_tt_independent_filtering as prior
from docs.benchmarks import run_observation_tt_warm_improvement as warm

PLAN=ROOT/'docs/plans/observation-aware-tt-master-amendment-11-protected-fitting-20260917.md'
write,sha=prior.write,prior.sha
REFERENCE=dict(particle_levels=(65536,131072,262144,524288),replicates=8,
               mean_mcse_limit=.01,logz_mcse_limit=.05,reference_t=2.364624251,
               replicate_seed_stride=10000)


def sequence_seed(stage,dimension,sequence,attempt=1):
    if stage not in ('smoke','calibration','confirmation') or dimension not in (1,4) or attempt not in (1,2):
        raise ValueError('Unreviewed A11 seed scope')
    size={'smoke':1,'calibration':6,'confirmation':12}[stage]
    if not 0<=sequence<size:raise ValueError('Sequence outside reviewed panel')
    start={'smoke':80+2*(attempt-1),'calibration':12*(attempt-1),'confirmation':24+24*(attempt-1)}[stage]
    return -1800000000+10000000*(start+(0 if dimension==1 else size)+sequence)


def baseline_config(dimension):
    return dict(warm.config(4,4096 if dimension==1 else 1024,.001),fraction=1.)


def candidates(dimension):
    baseline=baseline_config(dimension)
    profiles=[('half',.5,baseline['rows'],4,3,4),('quarter',.25,baseline['rows'],4,3,4),
              ('rows',.25,4096,4,3,4),('sweeps',.25,4096,4,3,8),
              ('rank',.25,4096,4,5,8),('degree',.25,4096,5,3,8),
              ('combined',.25,4096,5,5,8),('broad-capacity',1.,4096,4,5,8)]
    result={};seen=set()
    for name,fraction,rows,degree,rank,sweeps in profiles:
        for l1 in (1e-5,.001):
            cfg=dict(warm.config(degree,rows,l1),rank=rank,sweeps=sweeps,fraction=fraction)
            identity=dict(cfg,rank=1 if dimension==1 else rank)
            key=json.dumps(identity,sort_keys=True)
            if key not in seen:result[f'{name}-l{l1:g}']=cfg;seen.add(key)
    return result


def valid_metric(entry,name):
    metric=entry['metrics'].get(name)
    return bool(entry['reference_pass'] and metric and name not in entry['failures']
                and math.isfinite(metric['regimes']['all']['mse'])
                and metric['log_evidence_screen']['passed']
                and metric['cdf_bracket_failures']==0 and metric['consumer_invalid_steps']==0)


def nominate(entries,dimension):
    panel=[e for e in entries if e['dimension']==dimension]
    if len(panel)!=6 or not all(e['reference_pass'] for e in panel):
        raise ValueError('Calibration requires six valid references per dimension')
    rows=[]
    for name,cfg in candidates(dimension).items():
        valid=all(valid_metric(e,name) for e in panel)
        mse=sum(e['metrics'][name]['regimes']['all']['mse'] for e in panel)/len(panel) if valid else None
        rows.append(dict(name=name,config=cfg,valid=valid,mse=mse))
    viable=[r for r in rows if r['valid']]
    if not viable:raise ValueError('No valid A11 calibration fit')
    choice=min(viable,key=lambda r:(r['mse'],r['config']['degree'],r['config']['rank'],
                                   r['config']['sweeps'],r['config']['rows'],r['name']))
    return dict(**choice,fit_selection=rows)


def select_epsilon(entries,dimension,nominee):
    panel=[e for e in entries if e['dimension']==dimension]
    name=nominee['name'];curves=[];selected=.05
    baseline_mse=sum(e['metrics'][name]['regimes']['all']['mse'] for e in panel)/len(panel)
    baseline_low=sum(e['metrics'][name]['low_ess_fraction'] for e in panel)/len(panel)
    for epsilon in (.05,.1,.2):
        arm=name if epsilon==.05 else f'{name}-e{epsilon:g}'
        valid=all(valid_metric(e,arm) for e in panel)
        mse=sum(e['metrics'][arm]['regimes']['all']['mse'] for e in panel)/len(panel) if valid else None
        low=sum(e['metrics'][arm]['low_ess_fraction'] for e in panel)/len(panel) if valid else None
        passed=valid and mse<=1.1*baseline_mse and low<=baseline_low
        if passed:selected=epsilon
        curves.append(dict(epsilon=epsilon,valid=valid,mse=mse,low_ess_fraction=low,non_harm=passed))
    return dict(**nominee,epsilon=selected,epsilon_curve=curves)


def chart_diagnostics(model,guide,charts,fraction):
    reference=robust.stable_charts(model,guide);records=[]
    for t,(chart,scale) in enumerate(zip(charts,reference)):
        normalized=tf.linalg.triangular_solve(scale.factor,chart.factor)
        eigenvalues=tf.linalg.eigvalsh(normalized@tf.transpose(normalized))
        minimum=float(tf.reduce_min(eigenvalues))
        if minimum < fraction*(1.-1e-10):raise ValueError('Protected chart covariance bound failed')
        records.append(dict(time=t,fraction=fraction,relative_eigenvalues=eigenvalues,
                            relative_floor_margin=minimum-fraction))
    return records


def fit_path(context,name,cfg,budget):
    dest=context['dest']/name;dest.mkdir(exist_ok=True)
    model,guide=context['model'],context['guide']
    charts=robust.blended_charts(model,guide,cfg['fraction'])
    diagnostics=chart_diagnostics(model,guide,charts,cfg['fraction'])
    first=joint.make_sgqf_joint_step(model,guide[0][1],None,0)
    path=[first];retained=first.retained_proposal
    for t,y in enumerate(tf.unstack(context['observations'])[1:],1):
        budget()
        step,record=warm.fit_step(model,y,guide,retained,t,context['seed']+2000000,
                                  cfg,budget,charts=charts,audit=False)
        path.append(step);retained=step.retained_proposal
        write(dest/f'fit-{t:02d}.json',record)
    write(dest/'path.json',dict(config=cfg,steps=len(path),charts=diagnostics,
                               initial_proposal='analytic_sgqf_joint'))
    return path


def evaluate(context,name,path,epsilon,budget,build_seconds=0.):
    defended=[robust.PhysicalDefenseStep(step,context['model'],epsilon) for step in path]
    a10.evaluate(context,name,context['guide'],defended,'tt_physical_defense',budget,build_seconds,4)
    dest=context['dest']/name
    runs=[json.loads((dest/f'particles-r{r}.json').read_text()) for r in range(4)]
    steps=[s for run in runs for s in run['steps']]
    metric=context['entry']['metrics'][name]
    metric['low_ess_fraction']=sum(float(s['ess'])<.05*512 for s in steps)/len(steps)
    metric['physical_epsilon']=epsilon
    context['entry']['times'][name]['build_reused']=build_seconds==0.
    write(context['dest']/'summary.json',context['entry'])


def run_fit(context,name,cfg,budget):
    try:
        started=time.monotonic();path=fit_path(context,name,cfg,budget)
        build=time.monotonic()-started
        evaluate(context,name,path,.05,budget,build)
        return path
    except (ValueError,tf.errors.OpError) as exc:
        context['entry']['failures'][name]=repr(exc)
        write(context['dest']/'summary.json',context['entry'])
        return None


def evaluate_defense(context,name,path,epsilon,budget):
    try:
        if path is None:raise ValueError('Nominee fit failed')
        evaluate(context,name,path,epsilon,budget)
    except (ValueError,tf.errors.OpError) as exc:
        context['entry']['failures'][name]=repr(exc)
        write(context['dest']/'summary.json',context['entry'])


def infer(entries):
    report={}
    for d in (1,4):
        panel=[e for e in entries if e['dimension']==d]
        output={'reference_valid':sum(e['reference_pass'] for e in panel),'sequences':len(panel),
                'statistical_ranking_supported':False,'contrasts':{},'heuristic_losses':[]}
        for name in ('nominee','stronger-defense'):
            if not any(name in e['metrics'] for e in panel):continue
            valid=len(panel)==12 and all(valid_metric(e,name) and valid_metric(e,'baseline') for e in panel)
            if valid:
                x=tf.constant([e['metrics'][name]['regimes']['all']['mse'] for e in panel],D)
                b=tf.constant([e['metrics']['baseline']['regimes']['all']['mse'] for e in panel],D)
                indices=tf.random.stateless_uniform([9999,len(panel)],[1172026,d],maxval=len(panel),dtype=tf.int32)
                xx=tf.reduce_mean(tf.gather(x,indices),axis=1);bb=tf.reduce_mean(tf.gather(b,indices),axis=1)
                ordered=tf.sort((xx-bb)/bb);tail=.05/(2*4)
                ci=[float(ordered[int(tail*9999)]),float(ordered[int((1.-tail)*9999)])]
                change=float((tf.reduce_mean(x)-tf.reduce_mean(b))/tf.reduce_mean(b))
                output['contrasts'][name]=dict(relative_mse_change=change,simultaneous_interval=ci,
                     statistically_lower_mse=ci[1]<0.,non_harm=ci[1]<=.10)
                output['statistical_ranking_supported'] |= ci[1]<0. or ci[0]>0.
            else:output['contrasts'][name]=dict(status='invalid_or_incomplete_panel_no_ranking')
        for name in ('baseline','nominee','stronger-defense'):
            for regime in ('near_zero','ordinary','large'):
                valid=[e for e in panel if e['reference_pass'] and name in e['metrics']
                       and e['metrics'][name]['regimes'][regime]['count']>0
                       and all(h in e['metrics'] for h in warm.HEURISTICS)]
                if not valid:continue
                def conditional(arm):
                    return sum(e['metrics'][arm]['regimes'][regime]['mse']*
                               e['metrics'][arm]['regimes'][regime]['count'] for e in valid)/sum(
                               e['metrics'][arm]['regimes'][regime]['count'] for e in valid)
                value=conditional(name)
                for heuristic in warm.HEURISTICS:
                    other=conditional(heuristic)
                    if value>other:output['heuristic_losses'].append(dict(candidate=name,regime=regime,
                         heuristic=heuristic,candidate_mse=value,heuristic_mse=other,evidence='descriptive_veto'))
        output['promotion_veto']=(bool(output['heuristic_losses']) or output['reference_valid']!=12
            or any(not all(valid_metric(e,name) for e in panel) for name in output['contrasts']))
        report[str(d)]=output
    return report


def initialize():
    global tf,D,robust,joint,lib,base
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    devices=tf.config.list_physical_devices('GPU')
    if not devices:raise RuntimeError('Trusted GPU required')
    growth=[]
    for device in devices:
        tf.config.experimental.set_memory_growth(device,True)
        if not tf.config.experimental.get_memory_growth(device):raise RuntimeError('Memory growth required')
        growth.append(dict(name=device.name,growth=True,details=tf.config.experimental.get_device_details(device)))
    from bayesfilter.highdim import observation_guided_tt_tf as lib
    from bayesfilter.highdim import observation_robust_guide_tf as robust
    from bayesfilter.highdim import pair_block_tt_tf as pair
    from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal as logrho
    from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as projection
    from docs.benchmarks import run_observation_aware_tt_complete as base
    D=tf.float64
    for module in (base,prior,warm,a10):module.tf=tf;module.D=D
    base.lib=lib;prior.base=base
    for name,value in dict(lib=lib,pair=pair,joint=joint,projection=projection,base=base,logrho=logrho).items():setattr(warm,name,value)
    for name,value in dict(lib=lib,robust=robust,joint=joint,base=base).items():setattr(a10,name,value)
    dependencies=[Path(__file__),PLAN,Path(a10.__file__),Path(warm.__file__),Path(prior.__file__),Path(base.__file__),
                  Path(lib.__file__),Path(robust.__file__),Path(pair.__file__),Path(joint.__file__),Path(projection.__file__),prior.FIXTURE]
    return growth,dependencies


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','calibration','confirmation'),required=True)
    parser.add_argument('--attempt',type=int,choices=(1,2),default=1)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--calibration-root')
    parser.add_argument('--wall-budget-seconds',type=float,required=True)
    args=parser.parse_args();started=time.monotonic()
    limits={'smoke':900.,'calibration':10800.,'confirmation':5400.}
    if not 0<args.wall_budget_seconds<=limits[args.stage]:raise ValueError('Unreviewed attempt budget')
    out=Path(args.output_root).resolve();out.mkdir(parents=True,exist_ok=False)
    log=open(out/'command.log','a',buffering=1);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    growth,dependencies=initialize()
    frozen=None
    if args.stage=='confirmation':
        if not args.calibration_root:raise ValueError('Confirmation needs frozen calibration')
        path=Path(args.calibration_root)/'selected-controls.json'
        source=json.loads((path.parent/'run_manifest.json').read_text())
        if source['status']!='COMPLETE' or source['selected_controls_sha256']!=sha(path):raise ValueError('Invalid frozen controls')
        for dependency in dependencies:
            if source['source_hashes'].get(str(dependency.resolve()))!=sha(dependency):
                raise ValueError(f'Calibration source drift: {dependency}')
        frozen=json.loads(path.read_text());dependencies.append(path);write(out/'selected-controls.json',frozen)
    snapshot=out/'source-snapshot';snapshot.mkdir()
    for dependency in dependencies:
        relative=dependency.resolve().relative_to(ROOT) if dependency.resolve().is_relative_to(ROOT) else Path(dependency.name)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dependency,target)
    manifest=dict(status='RUNNING',stage=args.stage,attempt=args.attempt,started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=sys.argv,
        environment=sys.prefix,python=sys.version,tensorflow=tf.__version__,cpu_only=False,gpu_intentionally_hidden=False,
        jit_compile=True,dtype='float64',tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_memory_policy=dict(schema='a11_growth_v1',mode='memory_growth',devices=growth),
        trust_basis='escalated_gpu_access',source_classification='extension_or_invention',plan=str(PLAN),
        result=str(out/'result.json'),wall_budget_seconds=args.wall_budget_seconds,reference_policy=REFERENCE,
        source_hashes={str(p.resolve()):sha(p) for p in dependencies},data_version='A11 fresh stateless SV sequences; A10 fixed parameters',
        seeds=dict(namespace='-1800000000+10000000*block',attempt=args.attempt,stage=args.stage,
                   fit='+2000000; validation+100000',particle='+3000000+1000*r',reference='+4000000+1000000*level+10000*r',bootstrap=1172026),
        setup_exceptions='Host guide/row/feature setup, checked charts, CPU TT-SVD, reference/statistical reporting; no end-to-end XLA claim')
    result=dict(status='RUNNING',stage=args.stage,sequences=[],default_ready=False)
    write(out/'run_manifest.json',manifest)
    def budget():
        if time.monotonic()-started>args.wall_budget_seconds:raise TimeoutError('A11 attempt wall budget exhausted')
    contexts=[]
    try:
        fixture=json.loads(prior.FIXTURE.read_text());size={'smoke':1,'calibration':6,'confirmation':12}[args.stage]
        for d in (1,4):
            for sequence in range(size):
                budget();dest=out/f'd{d}-s{sequence:02d}';dest.mkdir()
                seed=sequence_seed(args.stage,d,sequence,args.attempt);data=fixture['dimensions'][str(d)]
                model=lib.SVModel(tf.constant(data['A'],D),tf.constant(data['P0'],D),fixture['beta'],fixture['sigma'])
                states,observations=prior.data_generator(model,4 if args.stage=='smoke' else 20)(tf.constant(seed,tf.int32))
                write(dest/'data.json',dict(seed=seed,states=states,observations=observations,A=model.transition,P0=model.covariance0,beta=model.beta,sigma=model.sigma))
                entry=dict(dimension=d,sequence=sequence,data_seed=seed,metrics={},failures={},times={});result['sequences'].append(entry)
                tic=time.monotonic();guide,records=robust.build_guide_path(model,observations)
                write(dest/'guide.json',records);entry['guide_seconds']=time.monotonic()-tic
                entry['guide_methods']={k:sum(r['method']==k for r in records) for k in sorted({r['method'] for r in records})}
                tic=time.monotonic();refmeans,refz,refpass,refinfo=prior.references(model,observations,dest,seed+4000000,budget,**REFERENCE)
                entry.update(reference_pass=refpass,reference=refinfo,reference_seconds=time.monotonic()-tic)
                context=dict(model=model,observations=observations,guide=guide,seed=seed,dest=dest,entry=entry,refmeans=refmeans,refz=refz,paths={})
                contexts.append(context)
                run_fit(context,'baseline',baseline_config(d),budget)
                profiles=candidates(d) if args.stage=='calibration' else (
                    {'maximum-capacity':dict(warm.config(5,4096,.001),rank=5,sweeps=8,fraction=.25)}
                    if args.stage=='smoke' else {'nominee':frozen[str(d)]['config']})
                for name,cfg in profiles.items():
                    context['paths'][name]=run_fit(context,name,cfg,budget)
                    write(out/'result.json',result)
                    print(json.dumps(dict(sequence=dest.name,arm=name,failed=name in entry['failures'],elapsed=time.monotonic()-started)),flush=True)
                if args.stage=='confirmation' and frozen[str(d)]['epsilon']!=.05:
                    path=context['paths']['nominee']
                    evaluate_defense(context,'stronger-defense',path,frozen[str(d)]['epsilon'],budget)
                if args.stage!='calibration':
                    for name in warm.HEURISTICS:a10.run_arm(context,name,guide,budget)
                if args.stage=='smoke' and (not refpass or entry['failures'] or
                        not all(valid_metric(entry,n) for n in ('baseline','maximum-capacity'))):
                    raise ValueError('A11 smoke validity/reference screen failed')
                write(dest/'summary.json',entry);write(out/'result.json',result)
                print(json.dumps(dict(completed=dest.name,reference_pass=refpass,elapsed=time.monotonic()-started)),flush=True)
        if args.stage=='calibration':
            nominations={str(d):nominate(result['sequences'],d) for d in (1,4)}
            write(out/'fit-nominations.json',nominations)
            for context in contexts:
                d=context['entry']['dimension'];name=nominations[str(d)]['name'];path=context['paths'][name]
                for epsilon in (.1,.2):evaluate_defense(context,f'{name}-e{epsilon:g}',path,epsilon,budget)
                write(out/'result.json',result)
            frozen={str(d):select_epsilon(result['sequences'],d,nominations[str(d)]) for d in (1,4)}
            write(out/'selected-controls.json',frozen);manifest['selected_controls_sha256']=sha(out/'selected-controls.json')
        elif args.stage=='confirmation':result['inference']=infer(result['sequences'])
        drift=[p for p,h in manifest['source_hashes'].items() if sha(Path(p))!=h]
        if drift:raise RuntimeError(f'Source drift: {drift}')
        result['status']='COMPLETE';manifest['status']='COMPLETE'
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc));manifest.update(status='FAILED',error=repr(exc))
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
                        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'))
        write(out/'result.json',result);write(out/'run_manifest.json',manifest)


if __name__=='__main__':main()
