#!/usr/bin/env python3
"""A10 bounded d<=4 SV repair experiment; optional extension, no HMC claim.

Numerical kernels use TensorFlow/GPU/XLA. Host guide/row/feature setup, dense
Gaussian projection and CPU TT-SVD are explicit setup/reference exceptions.
Statistics are post-run TensorFlow diagnostics, not a NumPy runtime.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from docs.benchmarks import run_observation_tt_warm_improvement as warm
from docs.benchmarks import run_observation_tt_independent_filtering as prior

PLAN=ROOT/'docs/plans/observation-aware-tt-master-amendment-10-robust-guide-20260916.md'
write,plain,sha=prior.write,prior.plain,prior.sha
HEURISTICS=warm.HEURISTICS


def sequence_seed(stage,dimension,sequence,confirmation_block_start=24):
    if confirmation_block_start not in (24,48):
        raise ValueError('Unreviewed confirmation seed block')
    block=(0 if dimension==1 else 3)+sequence if stage=='calibration' else confirmation_block_start+(0 if dimension==1 else 12)+sequence
    return 1000000000+10000000*block


def fit_path(context,name,guide,cfg,stable,budget):
    dest=context['dest']/name;dest.mkdir()
    charts=robust.stable_charts(context['model'],guide) if stable else None
    first=joint.make_sgqf_joint_step(context['model'],guide[0][1],None,0)
    path=[first];retained=first.retained_proposal
    records=[]
    for t,y in enumerate(tf.unstack(context['observations'])[1:],1):
        step,record=warm.fit_step(context['model'],y,guide,retained,t,context['seed']+2000000,
                                  cfg,budget,charts=charts,audit=False)
        path.append(step);retained=step.retained_proposal;records.append(record)
        write(dest/f'fit-{t:02d}.json',record)
    write(dest/'path.json',dict(config=cfg,stable_charts=stable,steps=len(path),
                              chart_covariance_minimum=[tf.reduce_min(tf.linalg.eigvalsh(s.current_chart.factor@tf.transpose(s.current_chart.factor))) for s in path]))
    return path


def evaluate(context,name,guide,path,method,budget,build_seconds=0.,replicates=4):
    dest=context['dest']/name;dest.mkdir(exist_ok=True)
    started=time.monotonic();runs=[]
    for r in range(replicates):
        budget()
        value,_=base.particle_filter(context['model'],context['observations'],guide,path,method,512,
                                    context['seed']+3000000+1000*r,True)
        runs.append(value);write(dest/f'particles-r{r}.json',value)
    e=context['entry']
    metrics=prior.sequence_metrics(context['model'],context['observations'],{name:runs},context['refmeans'],context['refz'],512)[name]
    refse=float(e['reference'].get('log_evidence_mcse',0.))
    allowance=.15+3.182446*(metrics['log_evidence_mcse']**2+refse**2)**.5
    metrics['log_evidence_screen']=dict(allowance=allowance,passed=abs(metrics['log_evidence_bias'])<=allowance)
    metrics['cdf_bracket_failures']=sum(int(not bool(s['cdf_bracket_valid'])) for r in runs for s in r['steps'] if 'cdf_bracket_valid' in s)
    metrics['consumer_invalid_steps']=sum(int(not bool(s['finite'])) for r in runs for s in r['steps'] if 'finite' in s)
    metrics['maximum_cdf_residual']=max([float(s['cdf_residual']) for r in runs for s in r['steps'] if 'cdf_residual' in s] or [0.])
    e['metrics'][name]=metrics
    e['times'][name]=dict(build_seconds=build_seconds,particle_seconds=time.monotonic()-started,particle_replicates=replicates,
                           build_reused=method=='tt_physical_defense')
    write(context['dest']/'summary.json',e)


def run_arm(context,name,guide,budget,*,cfg=None,stable=False,path=None,epsilon=None,replicates=4):
    try:
        if guide is None and name not in ('transition','stationary_prior'):
            raise ValueError('Original SGQF guide failed; baseline unavailable')
        started=time.monotonic();method=name
        if cfg is not None:
            path=fit_path(context,name,guide,cfg,stable,budget);method='tt_sgqf_initialized'
        elif epsilon is not None:
            if path is None:raise ValueError('Protected candidate fit failed')
            path=[robust.PhysicalDefenseStep(s,context['model'],epsilon) for s in path];method='tt_physical_defense'
        elif name=='sgqf_joint':
            path=[joint.make_sgqf_joint_step(context['model'],g[1],None if t==0 else guide[t-1][1],t) for t,g in enumerate(guide)]
        build=time.monotonic()-started
        evaluate(context,name,guide,path,method,budget,build,replicates)
        return path
    except (ValueError,tf.errors.OpError) as exc:
        context['entry']['failures'][name]=repr(exc)
        dest=context['dest']/name;dest.mkdir(exist_ok=True)
        write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),fallback=False))
        write(context['dest']/'summary.json',context['entry'])
        return None


def eligible_metric(entry,name):
    m=entry['metrics'].get(name)
    return bool(entry['reference_pass'] and m and m['log_evidence_screen']['passed']
                and not m['cdf_bracket_failures'] and not m['consumer_invalid_steps'])


def freeze(entries):
    controls={};ledger=[]
    for d in (1,4):
        selected={};rows=[e for e in entries if e['dimension']==d]
        for family in ('guide','stable'):
            table=[]
            for l1 in (1e-5,.001):
                name=f'{family}-l{l1:g}'
                valid=[e for e in rows if eligible_metric(e,name)]
                mse=sum(e['metrics'][name]['regimes']['all']['mse'] for e in valid)/len(valid) if valid else None
                table.append(dict(name=name,l1=l1,eligible=len(valid)==len(rows)==3,mse=mse,valid_sequences=len(valid)))
            candidates=[x for x in table if x['eligible']]
            # Preserve a representative even if no candidate is admissible;
            # fresh confirmation remains diagnostic and cannot repair calibration.
            chosen=min(candidates,key=lambda x:x['mse']) if candidates else next(x for x in table if x['l1']==.001)
            selected[family]=dict(l1=chosen['l1'],calibration_admissible=chosen['eligible'])
            ledger.append(dict(dimension=d,family=family,candidates=table,selected=chosen))
        l1=selected['stable']['l1'];curve=[]
        stable_name=f'stable-l{l1:g}'
        for eps in (.05,.2,.5):
            name=f'full-l{l1:g}-e{eps:g}'
            valid=[e for e in rows if eligible_metric(e,name) and eligible_metric(e,stable_name)]
            delta=sum(e['metrics'][name]['regimes']['all']['mse']-e['metrics'][stable_name]['regimes']['all']['mse'] for e in valid)/len(valid) if valid else None
            baseline=sum(e['metrics'][stable_name]['regimes']['all']['mse'] for e in valid)/len(valid) if valid else None
            okay=len(valid)==len(rows)==3 and delta<=.1*baseline
            curve.append(dict(epsilon=eps,valid_sequences=len(valid),mean_delta=delta,reference_mse=baseline,
                              calibration_nonharm_pass=okay,second_moment_multiplier=1/eps))
        passing=[r for r in curve if r['calibration_nonharm_pass']]
        epsilon=passing[0]['epsilon'] if passing else .5
        selected['epsilon']=epsilon;selected['defense_calibration_admissible']=bool(passing)
        controls[str(d)]=selected
        ledger.append(dict(dimension=d,family='epsilon',curve=curve,selected=epsilon,
                           selection='smallest_passing' if passing else 'failed_calibration_diagnostic_representative'))
    return dict(status='FROZEN',controls=controls,selection_ledger=ledger,
                nonclaim='Frozen diagnostics do not establish promotion; failed calibration remains a veto.')


def infer(entries):
    contrasts=[];heuristics=[]
    for d in (1,4):
        rows=[e for e in entries if e['dimension']==d]
        for name in ('guide','stable','full'):
            matched=[e for e in rows if eligible_metric(e,name) and eligible_metric(e,'baseline')]
            delta=[e['metrics'][name]['regimes']['all']['mse']-e['metrics']['baseline']['regimes']['all']['mse'] for e in matched]
            # The actual non-harm statistic includes the 10% baseline margin.
            excess=[e['metrics'][name]['regimes']['all']['mse']-1.1*e['metrics']['baseline']['regimes']['all']['mse'] for e in matched]
            full=len(matched)==len(rows)==12
            ci=nonharm=None
            if full:
                index=tf.random.stateless_uniform([9999,12],[109277777,d],minval=0,maxval=12,dtype=tf.int32)
                def interval(values):
                    draws=tf.sort(tf.reduce_mean(tf.gather(tf.constant(values,D),index),axis=1))
                    # Bonferroni across 6 predeclared dimension/arm contrasts.
                    return [float(draws[41]),float(draws[9957])]
                ci=interval(delta);nonharm=interval(excess)
            contrasts.append(dict(dimension=d,candidate=name,matched_sequences=len(matched),full_coverage=full,
                mean_delta=sum(delta)/len(delta) if delta else None,paired_interval=ci,
                nonharm_excess_interval=nonharm,nonharm_supported=nonharm is not None and nonharm[1]<=0,
                improvement_supported=ci is not None and ci[1]<0,inference='exploratory_sequence_bootstrap'))
            for heuristic,regime in itertools.product(HEURISTICS,('all','near_zero','ordinary','large')):
                valid=[e for e in rows if eligible_metric(e,name) and eligible_metric(e,heuristic)
                       and e['metrics'][name]['regimes'][regime]['count']]
                differences=[e['metrics'][name]['regimes'][regime]['mse']-e['metrics'][heuristic]['regimes'][regime]['mse'] for e in valid]
                value=sum(differences)/len(differences) if differences else None
                heuristics.append(dict(dimension=d,candidate=name,heuristic=heuristic,regime=regime,sequences=len(valid),
                    mean_delta=value,observed_loss=value is not None and value>0,role='descriptive_promotion_veto_not_ranking'))
    return dict(contrasts=contrasts,heuristic_contrasts=heuristics,
                heuristic_dominance_verdict='PROMOTION_VETO' if any(h['observed_loss'] for h in heuristics) else 'NO_OBSERVED_LOSS',
                default_ready=False)


def main():
    global tf,D,lib,robust,joint,base
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','calibration','confirmation'),required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--calibration-root')
    parser.add_argument('--confirmation-block-start',type=int,choices=(24,48),default=24)
    parser.add_argument('--wall-budget-seconds',type=float,default=5400)
    args=parser.parse_args();started=time.monotonic()
    out=Path(args.output_root).resolve();out.mkdir(parents=True,exist_ok=False)
    log=open(out/'command.log','a',buffering=1);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
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
    for module in (base,prior,warm):module.tf=tf;module.D=D
    base.lib=lib;prior.base=base
    for name,value in dict(lib=lib,pair=pair,joint=joint,projection=projection,base=base,logrho=logrho).items():setattr(warm,name,value)
    dependencies=[Path(__file__),PLAN,Path(warm.__file__),Path(prior.__file__),Path(base.__file__),Path(lib.__file__),
                  Path(robust.__file__),Path(pair.__file__),Path(joint.__file__),Path(projection.__file__),prior.FIXTURE]
    frozen=None
    if args.stage=='confirmation':
        if not args.calibration_root:raise ValueError('Confirmation needs completed calibration')
        path=Path(args.calibration_root)/'selected-controls.json'
        source=json.loads((path.parent/'run_manifest.json').read_text())
        if source['status']!='COMPLETE' or source['selected_controls_sha256']!=sha(path):raise ValueError('Invalid frozen controls')
        frozen=json.loads(path.read_text());dependencies.append(path);write(out/'selected-controls.json',frozen)
    manifest=dict(status='RUNNING',stage=args.stage,started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=sys.argv,
        environment=sys.prefix,python=sys.version,tensorflow=tf.__version__,cpu_only=False,gpu_intentionally_hidden=False,
        jit_compile=True,dtype='float64',tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_memory_policy=dict(schema='a10_growth_v1',mode='memory_growth',devices=growth),
        trust_basis='escalated_gpu_access',classification='bounded_diagnostic_extension',source_classification='extension_or_invention',
        plan=str(PLAN),result=str(out/'result.json'),wall_budget_seconds=args.wall_budget_seconds,
        source_hashes={str(p.resolve()):sha(p) for p in dependencies},data_version='A10 fresh stateless SV sequences; fixed model parameters only',
        seeds=dict(block=f'1e9+1e7*k; cal k0..5, confirm k{args.confirmation_block_start}..{args.confirmation_block_start+23}',fit='block+2e6; validation+1e5,audit+2e5',
                   particle='block+3e6+1000*r+t',reference='block+4e6+1e6*level+1000*r+t',bootstrap=109277777),
        setup_exceptions='Host guide/row/feature setup, positive-rule CPU eigensolve, CPU TT-SVD and reference/statistical reporting; no end-to-end XLA claim')
    write(out/'run_manifest.json',manifest)
    result=dict(status='RUNNING',stage=args.stage,sequences=[],default_ready=False)
    def budget():
        if time.monotonic()-started>args.wall_budget_seconds:raise TimeoutError('A10 attempt wall budget exhausted')
    try:
        fixture=json.loads(prior.FIXTURE.read_text())
        jobs=[(d,s,None) for d in (1,4) for s in range(1 if args.stage=='smoke' else (3 if args.stage=='calibration' else 12))]
        if args.stage=='smoke':jobs.extend([(4,4,'exposed'),(4,10,'exposed'),(4,5,'a10-scale')])
        for d,sequence,exposed in jobs:
            budget();dest=out/(f'{exposed}-d4-s{sequence:02d}' if exposed else f'd{d}-s{sequence:02d}');dest.mkdir()
            data=fixture['dimensions'][str(d)]
            model=lib.SVModel(tf.constant(data['A'],D),tf.constant(data['P0'],D),fixture['beta'],fixture['sigma'])
            if exposed:
                source=ROOT/(f'docs/benchmarks/artifacts/observation_tt_robust_guide_20260916/attempt-confirmation-01/d4-s{sequence:02d}/data.json' if exposed=='a10-scale' else f'docs/benchmarks/artifacts/observation_tt_warm_improvement_20260916/attempt-confirmation-02/d4-s{sequence:02d}/data.json')
                saved=json.loads(source.read_text());observations=tf.constant(saved['observations'],D)
                states=tf.constant(saved['states'],D);seed=saved['seed'] if exposed=='a10-scale' else 1100000000+sequence*10000
                manifest.setdefault('diagnostic_sources',{})[str(source)]=sha(source)
            else:
                seed=(1080000000+d*10000) if args.stage=='smoke' else sequence_seed(args.stage,d,sequence,args.confirmation_block_start)
                states,observations=prior.data_generator(model,3 if args.stage=='smoke' else 20)(tf.constant(seed,tf.int32))
            write(dest/'data.json',dict(seed=seed,states=states,observations=observations,A=model.transition,P0=model.covariance0,beta=model.beta,sigma=model.sigma,exposed=bool(exposed)))
            entry=dict(dimension=d,sequence=sequence,data_seed=seed,exposed=bool(exposed),metrics={},failures={},times={})
            result['sequences'].append(entry)
            original=None;tic=time.monotonic()
            try:
                original,records=lib.build_guide_path(model,observations);write(dest/'original-guide.json',records)
            except (ValueError,tf.errors.OpError) as exc:
                entry['failures']['original_guide']=repr(exc);write(dest/'original-guide-failure.json',dict(error=repr(exc)))
            entry['original_guide_seconds']=time.monotonic()-tic;tic=time.monotonic()
            guide,records=robust.build_guide_path(model,observations)
            write(dest/'robust-guide.json',records)
            entry['robust_guide_seconds']=time.monotonic()-tic
            entry['guide_methods']={k:sum(r['method']==k for r in records) for k in sorted({r['method'] for r in records})}
            entry['rejected_rules']=sum(r['status']=='invalid' for step in records for r in step['rules'])
            if args.stage=='smoke':
                # Engineering-only filter moments are not a scientific reference.
                refmeans,refz=tf.stack([g[1].mean for g in guide]),tf.constant(0.,D)
                refpass=False;refinfo=dict(kind='smoke_no_accuracy_claim',passed=False)
            else:
                refmeans,refz,refpass,refinfo=prior.references(model,observations,dest,seed+4000000,budget,replicate_seed_stride=1000)
            entry.update(reference_pass=refpass,reference=refinfo)
            context=dict(model=model,observations=observations,guide=guide,seed=seed,dest=dest,entry=entry,refmeans=refmeans,refz=refz)
            cfg=warm.config(4,4096 if d==1 else 1024,.001)
            # The shared MCSE estimator requires independent repetitions even
            # for a smoke; one repetition produces an undefined variance.
            reps=4
            run_arm(context,'baseline',original,budget,cfg=cfg,replicates=reps)
            if args.stage=='calibration':
                for l1 in (1e-5,.001):
                    trial=dict(cfg,l1=l1)
                    run_arm(context,f'guide-l{l1:g}',guide,budget,cfg=trial,replicates=reps)
                    path=run_arm(context,f'stable-l{l1:g}',guide,budget,cfg=trial,stable=True,replicates=reps)
                    for eps in (.05,.2,.5):
                        run_arm(context,f'full-l{l1:g}-e{eps:g}',guide,budget,path=path,epsilon=eps,replicates=reps)
            else:
                controls=frozen['controls'][str(d)] if frozen else dict(guide=dict(l1=.001),stable=dict(l1=.001),epsilon=.2)
                run_arm(context,'guide',guide,budget,cfg=dict(cfg,l1=controls['guide']['l1']),replicates=reps)
                path=run_arm(context,'stable',guide,budget,cfg=dict(cfg,l1=controls['stable']['l1']),stable=True,replicates=reps)
                run_arm(context,'full',guide,budget,path=path,epsilon=controls['epsilon'],replicates=reps)
            for name in HEURISTICS:run_arm(context,name,guide,budget,replicates=reps)
            write(dest/'summary.json',entry);write(out/'result.json',result)
            print(json.dumps(dict(completed=str(dest.name),guide_methods=entry['guide_methods'],failures=list(entry['failures']),elapsed=time.monotonic()-started)),flush=True)
        if args.stage=='calibration':
            frozen=freeze(result['sequences']);write(out/'selected-controls.json',frozen)
            manifest['selected_controls_sha256']=sha(out/'selected-controls.json')
        elif args.stage=='confirmation':result['inference']=infer(result['sequences'])
        drift=[p for p,h in manifest['source_hashes'].items() if sha(Path(p))!=h]
        if drift:raise RuntimeError(f'Source drift: {drift}')
        result['status']='COMPLETE';manifest['status']='COMPLETE'
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc));manifest.update(status='FAILED',error=repr(exc))
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
                        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'))
        write(out/'result.json',result);write(out/'run_manifest.json',manifest)


if __name__=='__main__':main()
