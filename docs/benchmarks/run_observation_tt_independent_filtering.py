#!/usr/bin/env python3
"""A06 independent filtering diagnostic; frozen algorithm, no default claims."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PLAN=ROOT/'docs/plans/observation-aware-tt-master-amendment-06-independent-filtering-20260915.md'
FIXTURE=ROOT/'docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/diagnostic-02/downstream_fixture.json'
METHODS=('transition','stationary_prior','sgqf_gaussian','sgqf_joint','tt_predictive','tt_guided','tt_pair_block','tt_sgqf_safeguard')
HEURISTICS=METHODS[:4]


def plain(value):
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    if hasattr(value,'numpy'): value=value.numpy()
    if hasattr(value,'tolist'): value=value.tolist()
    return value


def write(path,value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def data_generator(model,T):
    d=model.dimension
    @tf.function(input_signature=[tf.TensorSpec([],tf.int32)],jit_compile=True,autograph=False)
    def generate(seed):
        innovations=tf.random.stateless_normal([T,d],tf.stack([seed,0]),dtype=D)
        obsnoise=tf.random.stateless_normal([T,d],tf.stack([seed,1]),dtype=D)
        states=tf.TensorArray(D,size=T)
        x=tf.linalg.matvec(tf.linalg.cholesky(model.covariance0),innovations[0])
        states=states.write(0,x)
        def body(t,x,states):
            x=tf.linalg.matvec(model.transition,x)+model.sigma*innovations[t]
            return t+1,x,states.write(t,x)
        _,_,states=tf.while_loop(lambda t,*_:t<T,body,(tf.constant(1),x,states))
        x=states.stack()
        return x,model.beta*tf.exp(.5*x)*obsnoise
    return generate


def enhanced_path(model,observations,guide,seed,dest,budget):
    """Fixed per-step training/selection rule; audit generated after selection."""
    d=model.dimension
    first=joint.make_sgqf_joint_step(model,guide[0][1],None,0)
    selected_steps=[first];retained=first.retained_proposal
    write(dest/'selection-t0.json',dict(selected='exact_sgqf_joint',reason='frozen initial rule'))
    for t,y in enumerate(tf.unstack(observations)[1:],1):
        budget();tic=time.monotonic()
        current,condition=guide[t][1],guide[t-1][1]
        sg=joint.make_sgqf_joint_step(model,current,condition,t)
        m,C=projection.paired_gaussian(model,current,condition)
        gaussian=lib.Chart.from_moments(m,C)
        coefficients,_=projection.coefficient_kernel(2*d)(m,C)
        with tf.device('/CPU:0'):
            warm,compression,_=projection.pair_svd(coefficients)
        def make_panel(n,offset,scale=None):
            coords,logw,rowinfo=lib.joint_sgqf_row_sampler(model,current,condition,n,seed+t+offset)
            rows=tf.stack([coords[:,:d],coords[:,d:]],axis=-1)
            x,z=current.forward(rows[:,:,0]),condition.forward(rows[:,:,1])
            logtarget=model.observation_log_prob(x,y)+model.transition_log_prob(x,z)+retained.physical_log_density(z)+current.logdet+condition.logdet-logrho(coords)
            if scale is None: scale=tf.reduce_logsumexp(logtarget+logw)-tf.math.log(tf.cast(n,D))
            target=tf.exp(.5*(logtarget-scale));weights=tf.exp(logw)
            lib.finite(target,'enhanced target')
            target_weight=weights*target**2;target_weight/=tf.reduce_sum(target_weight)
            rowinfo.update(target_weight_ess=1/tf.reduce_sum(target_weight**2),maximum_target_weight=tf.reduce_max(target_weight))
            return dict(rows=rows,target=target,weights=weights,scale=scale,info=rowinfo)
        def h2(amplitude,panel):
            w,b=panel['weights'],panel['target']
            value=1-tf.reduce_sum(w*amplitude*b)/tf.sqrt(tf.reduce_sum(w*amplitude**2)*tf.reduce_sum(w*b**2))
            lib.finite(value,'selection discrepancy')
            if float(value)<-1e-12: raise ValueError('negative empirical Hellinger')
            return tf.maximum(0.,value)
        def score(cores,panel):
            h=pair.evaluate_pair_cores(cores,panel['rows']);Z=pair.pair_total_mass(cores)
            tf.debugging.assert_positive(Z)
            return h2(tf.sqrt((1-1e-5)*h*h/Z+1e-5),panel)
        def gauss_score(panel):
            coords=tf.reshape(panel['rows'],[-1,2*d])
            return h2(tf.exp(.5*(gaussian.log_prob(coords)-logrho(coords))),panel)
        train=make_panel(1024,0);validation=make_panel(4096,100000,train['scale'])
        feature=pair._pair_features(train['rows'],3)
        fits={};options={'exact_sgqf_joint':sg};scores={'exact_sgqf_joint':gauss_score(validation)}
        for family,initial in [('generic',pair.initial_pair_cores(d)),('sgqf',warm)]:
            values=pair.evaluate_pair_cores(initial,train['rows'])
            scalar=tf.reduce_sum(train['weights']*values*train['target'])/tf.reduce_sum(train['weights']*values**2)
            tf.debugging.assert_positive(scalar,'nonpositive initialization scale')
            initial=(initial[0]*scalar,*initial[1:]);grid=[]
            for l1 in (0.,1e-5,1e-3):
                budget()
                cores,diag=pair.fit_pair_features(feature,train['target'],train['weights'],degree=3,rank=3,sweeps=4,proximal_steps=128,penalty=l1,initial=initial,jit_compile=True)
                residual=pair.evaluate_pair_cores(cores,validation['rows'])-validation['target']
                rms=tf.sqrt(tf.reduce_sum(validation['weights']*residual**2)/tf.reduce_sum(validation['weights']*validation['target']**2))
                lib.finite(rms,'fit validation')
                grid.append(dict(l1=l1,cores=cores,diagnostics=diag,validation_rms=rms,validation_h2=score(cores,validation)))
            chosen=min(grid,key=lambda g:float(g['validation_rms']))
            cores=chosen['cores'];Z=pair.pair_total_mass(cores);tau=Z*1e-5/(1-1e-5)
            ret=lib.PairRetainedProposal(cores,current,Z,tau,t)
            options[family]=lib.PairTTStep(cores,current,condition,tau,chosen['diagnostics'],t,ret)
            scores[family]=chosen['validation_h2'];fits[family]=dict(selected_l1=chosen['l1'],grid=grid)
        name=min(scores,key=lambda k:float(scores[k]))
        selection=dict(time=t,selected=name,validation_h2=scores,fit_seed=seed+t,audit_used=False)
        write(dest/f'selection-t{t}.json',selection)
        audit=make_panel(8192,200000,train['scale'])
        audit_scores={'exact_sgqf_joint':gauss_score(audit)}
        for family in ('generic','sgqf'): audit_scores[family]=score(options[family].cores,audit)
        selected=options[name]
        write(dest/f'fit-t{t}.json',dict(selection=selection,fits=fits,audit_h2=audit_scores,
            audit_loss_vs_sgqf=bool(audit_scores[name]>audit_scores['exact_sgqf_joint']),
            row_diagnostics=dict(train=train['info'],validation=validation['info'],audit=audit['info']),
            target_scale=train['scale'],conversion_compression= compression,
            current_mean=current.mean,current_factor=current.factor,condition_mean=condition.mean,
            condition_factor=condition.factor,wall_seconds=time.monotonic()-tic))
        selected_steps.append(selected);retained=selected.retained_proposal
    return selected_steps


def mean_se(values):
    x=tf.stack(values);mean=tf.reduce_mean(x,axis=0);n=int(x.shape[0])
    return mean,tf.sqrt(tf.reduce_sum((x-mean)**2,axis=0)/(n*(n-1)))


def references(model,observations,dest,seed,budget):
    d=model.dimension
    if d==1:
        coarse=base.scalar_grid_reference(model,observations,801)
        fine=base.scalar_grid_reference(model,observations,1201)
        mean_gap=tf.reduce_max(tf.abs(tf.stack([s['mean'] for s in coarse['steps']])-tf.stack([s['mean'] for s in fine['steps']])))/tf.sqrt(model.covariance0[0,0])
        log_gap=tf.abs(coarse['log_evidence']-fine['log_evidence'])
        passed=bool(mean_gap<=1e-6 and log_gap<=1e-6)
        output=dict(kind='scalar_grid',passed=passed,mean_gap=mean_gap,log_gap=log_gap,coarse=coarse,fine=fine)
        write(dest/'reference.json',output)
        return tf.stack([tf.reshape(s['mean'],[1]) for s in fine['steps']]),fine['log_evidence'],passed,output
    levels=[];previous=None
    for level,N in enumerate((32768,65536,131072)):
        budget();runs=[]
        for r in range(4):
            budget();value,_=base.particle_filter(model,observations,None,None,'transition',N,seed+1000000*level+10*r,True)
            runs.append(value)
        means,se=mean_se([tf.stack([s['mean'] for s in run['steps']]) for run in runs])
        logz,logse=mean_se([run['log_evidence'] for run in runs])
        scale=tf.sqrt(tf.linalg.diag_part(model.covariance0))
        current=dict(N=N,mean=means,se=se,logz=logz,logse=logse,runs=runs)
        if previous is not None:
            gap=tf.abs(means-previous['mean'])/scale
            allowance=3.182446*tf.sqrt(se**2+previous['se']**2)/scale+.01
            passed=bool(tf.reduce_max(se/scale)<=.02 and logse<=.10 and tf.reduce_all(gap<=allowance) and tf.abs(logz-previous['logz'])<=3.182446*tf.sqrt(logse**2+previous['logse']**2)+.10)
            current.update(passed=passed,max_mean_mcse=tf.reduce_max(se/scale),max_mean_gap=tf.reduce_max(gap))
        else: passed=False
        levels.append(current);write(dest/f'reference-{N}.json',current)
        if previous is not None and (passed or N==131072):
            output=dict(kind='bootstrap_multiscale',passed=passed,selected_N=N,
                mean_mcse=tf.reduce_max(se/scale),log_evidence_mcse=logse,levels=[k['N'] for k in levels])
            write(dest/'reference.json',output)
            return means,logz,passed,output
        previous=current


def sequence_metrics(model,observations,runs,refmeans,refz):
    scale=tf.linalg.diag_part(model.covariance0)
    magnitude=tf.abs(observations)/model.beta
    masks={'all':tf.ones_like(magnitude,dtype=tf.bool),'near_zero':magnitude<=.5,
           'ordinary':(magnitude>.5)&(magnitude<2),'large':magnitude>=2}
    table={}
    for name,replicates in runs.items():
        errors=tf.stack([(tf.stack([s['mean'] for s in r['steps']])-refmeans)**2/scale for r in replicates])
        avg=tf.reduce_mean(errors,axis=0)
        values={}
        for label,mask in masks.items():
            count=int(tf.reduce_sum(tf.cast(mask,tf.int32)))
            values[label]=dict(count=count,mse=float(tf.reduce_mean(tf.boolean_mask(avg,mask))) if count else None)
        z,zse=mean_se([r['log_evidence'] for r in replicates])
        table[name]=dict(regimes=values,log_evidence_bias=z-refz,log_evidence_mcse=zse)
    return plain(table)


def infer(records):
    """Paired sequence bootstrap with simultaneous standardized maximum."""
    labels=[];deltas=[];counts=[];screens=[]
    # Use a joint resampling index to retain cross-method/regime covariance.
    for d in (1,4):
        entries=[r for r in records if r['dimension']==d]
        for h in HEURISTICS:
            for regime in ('all','near_zero','ordinary','large'):
                values=[];total=0;n=0;valid=True
                for entry in entries:
                    table=entry['metrics']
                    if not entry['reference_pass'] or 'tt_sgqf_safeguard' not in table or h not in table:
                        valid=False;values.append(None);continue
                    c,b=table['tt_sgqf_safeguard']['regimes'][regime],table[h]['regimes'][regime]
                    if c['count']==0: values.append(None);continue
                    values.append(c['mse']-b['mse']);total+=c['count'];n+=1
                while len(values)<12: values.append(None);valid=False
                labels.append(dict(dimension=d,heuristic=h,regime=regime))
                deltas.append([0. if x is None else x for x in values]);counts.append([x is not None for x in values])
                screens.append(dict(complete=valid,sequence_count=n,coordinate_times=total,coverage_pass=(n>=8 and total>=24)))
    X=tf.constant(deltas,D);mask=tf.cast(tf.constant(counts),D)
    n=tf.reduce_sum(mask,axis=1)
    mean=tf.math.divide_no_nan(tf.reduce_sum(X,axis=1),n)
    se=tf.sqrt(tf.math.divide_no_nan(tf.reduce_sum(mask*(X-mean[:,None])**2,axis=1),n*(n-1)))
    index=tf.random.stateless_uniform([9999,12],[920000,0],minval=0,maxval=12,dtype=tf.int32)
    bx=tf.gather(X,index,axis=1);bm=tf.gather(mask,index,axis=1)
    bn=tf.reduce_sum(bm,axis=-1)
    boot=tf.math.divide_no_nan(tf.reduce_sum(bx*bm,axis=-1),bn)
    standardized=tf.math.divide_no_nan(tf.abs(boot-mean[:,None]),se[:,None])
    # Ineligible regimes have no interval and cannot contaminate the maximum.
    eligible=[s['complete'] and s['coverage_pass'] for s in screens]
    if any(eligible):
        valid_columns=tf.reduce_all(tf.boolean_mask(bn>0,eligible),axis=0)
        maxima=tf.boolean_mask(tf.reduce_max(tf.boolean_mask(standardized,eligible),axis=0),valid_columns)
        sorted_max=tf.sort(maxima)
        if int(sorted_max.shape[0])<9900: raise ValueError('Too many empty eligible bootstrap regime draws')
        critical=tf.gather(sorted_max,int(.95*(int(sorted_max.shape[0])-1)))
        valid_resamples=int(sorted_max.shape[0])
    else:
        critical=tf.constant(0.,D);valid_resamples=0
    contrasts=[]
    for i,label in enumerate(labels):
        width=float(critical*se[i]) if eligible[i] else None;center=float(mean[i])
        contrasts.append(dict(**label,**screens[i],mean_delta=center,se=float(se[i]),
            lower=center-width if eligible[i] else None,upper=center+width if eligible[i] else None,
            half_width=width,zero_se=bool(se[i]==0),interval_eligible=eligible[i],
            precision_pass=eligible[i] and width<=.01,empirical_heuristic_loss=center>0))
    passes=all(c['interval_eligible'] and c['precision_pass'] and c['upper']<=0 for c in contrasts)
    return dict(simultaneous_critical=critical,contrasts=contrasts,candidate_advances=passes,
        heuristic_dominance_verdict='PROMOTION_VETO' if any(c['empirical_heuristic_loss'] for c in contrasts if c['complete']) else 'NO_OBSERVED_LOSS_OR_INCOMPLETE',
        default_ready=False,bootstrap_valid_resamples=valid_resamples)


def main():
    global tf,D,lib,pair,joint,projection,base,logrho
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',required=True)
    parser.add_argument('--wall-budget-seconds',type=float,default=2400)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();start=time.monotonic()
    out=Path(args.output_root).resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    devices=tf.config.list_physical_devices('GPU')
    if not devices: raise RuntimeError('Trusted GPU required')
    growth=[]
    for device in devices:
        tf.config.experimental.set_memory_growth(device,True)
        if not tf.config.experimental.get_memory_growth(device): raise RuntimeError('Growth required')
        growth.append(dict(name=device.name,growth=True,details=tf.config.experimental.get_device_details(device)))
    from bayesfilter.highdim import observation_guided_tt_tf as lib
    from bayesfilter.highdim import pair_block_tt_tf as pair
    from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal as logrho
    from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as projection
    from docs.benchmarks import run_observation_aware_tt_complete as base
    D=tf.float64;base.tf=tf;base.lib=lib;base.D=D
    dependencies=[Path(__file__).resolve(),PLAN,Path(lib.__file__),Path(pair.__file__),Path(joint.__file__),Path(projection.__file__),Path(base.__file__),ROOT/'bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py']
    manifest=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        plan=str(PLAN),result=str(out/'result.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=ROOT,text=True),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies},model_source_sha256=sha(FIXTURE),
        environment=sys.prefix,python=sys.version,tensorflow=tf.__version__,cpu_only=False,gpu_intentionally_hidden=False,
        jit_compile=True,numerical_dtype='float64',tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_memory_policy=dict(schema='a06_growth_v1',mode='memory_growth',devices=growth),
        trust_basis='escalated_gpu_access',classification='mechanics_smoke' if args.smoke else 'independent_sequence_diagnostic',
        reference_exceptions='Gaussian setup/projection SVD, scalar grid and post-run statistics are TF/stdlib reference exceptions.',
        data_version='A06 stateless SV sequences; parameters only from old fixture, old observations unused',seeds=dict(data='916000+100*d+i',fit='917000+10000*d+100*i',particle='918000+10000*d+100*i+10*r',reference='919000+10000*d+100*i+10*r'),
        attempts_budget=dict(numerical_total_seconds=2700,per_attempt_seconds=args.wall_budget_seconds))
    controls=dict(methods=METHODS,dimensions=[1,4],sequences=12,horizon=20,particles=512,replicates=4,
        degree=3,rank=3,sweeps=4,proximal_steps=128,defensive_mass=1e-5,l1_grid=[0,1e-5,1e-3],
        rows=[1024,4096,8192],selector='minimum_validation_H2_tie_SGQF',frozen_before_observations=True)
    write(out/'frozen-controls.json',controls);write(out/'run_manifest.json',manifest)
    result=dict(status='RUNNING',sequences=[],default_ready=False)
    def budget():
        if time.monotonic()-start>args.wall_budget_seconds: raise TimeoutError('A06 attempt budget exhausted')
    try:
        fixture=json.loads(FIXTURE.read_text())
        for d in ([1] if args.smoke else [1,4]):
            data=fixture['dimensions'][str(d)]
            model=lib.SVModel(tf.constant(data['A'],D),tf.constant(data['P0'],D),fixture['beta'],fixture['sigma'])
            T=3 if args.smoke else 20
            generate=data_generator(model,T)
            for seq in range(1 if args.smoke else 12):
                budget();tic=time.monotonic();dest=out/f'd{d}-s{seq:02d}';dest.mkdir()
                data_seed=931001 if args.smoke else 916000+100*d+seq
                states,observations=generate(tf.constant(data_seed,tf.int32))
                write(dest/'data.json',dict(seed=data_seed,states=states,observations=observations,A=model.transition,P0=model.covariance0,beta=model.beta,sigma=model.sigma))
                stage=time.monotonic();guide=None;failures={}
                try:
                    guide,records=lib.build_guide_path(model,observations)
                    write(dest/'guide.json',records)
                except (ValueError,tf.errors.OpError) as exc: failures['guide']=repr(exc)
                times={'guide':time.monotonic()-stage}
                print(f'd{d} s{seq} reference',flush=True);stage=time.monotonic()
                refmeans,refz,refpass,refinfo=references(model,observations,dest,919000+10000*d+100*seq,budget)
                times['reference']=time.monotonic()-stage
                runs={}
                for method in METHODS:
                    budget();stage=time.monotonic();methoddir=dest/method;methoddir.mkdir()
                    if guide is None and method not in ('transition','stationary_prior'):
                        failures[method]='shared guide unavailable';continue
                    path=None;fitseed=917000+10000*d+100*seq
                    print(f'd{d} s{seq} {method}',flush=True)
                    try:
                        if method=='sgqf_joint':
                            path=[joint.make_sgqf_joint_step(model,g[1],None if t==0 else guide[t-1][1],t) for t,g in enumerate(guide)]
                        elif method in ('tt_predictive','tt_guided'):
                            path=lib.build_tt_path(model,observations,guide,guided=method=='tt_guided',seed=fitseed+(1000000 if method=='tt_predictive' else 2000000))
                        elif method=='tt_pair_block': path=lib.build_pair_tt_path(model,observations,guide,seed=fitseed+3000000)
                        elif method=='tt_sgqf_safeguard': path=enhanced_path(model,observations,guide,fitseed,methoddir,budget)
                        fitseconds=time.monotonic()-stage
                        if path is not None and method not in ('sgqf_joint','tt_sgqf_safeguard'):
                            write(methoddir/'proposals.json',[dict(time=s.time_index,cores=s.cores,tau=s.tau,fit_diagnostics=s.fit_diagnostics,
                                current_mean=s.current_chart.mean,current_factor=s.current_chart.factor,
                                condition_mean=None if s.conditioning_chart is None else s.conditioning_chart.mean,
                                condition_factor=None if s.conditioning_chart is None else s.conditioning_chart.factor) for s in path])
                        repetitions=[]
                        for r in range(4):
                            budget();value,_=base.particle_filter(model,observations,guide,path,method,512,918000+10000*d+100*seq+10*r,True)
                            repetitions.append(value);write(methoddir/f'particles-r{r}.json',value)
                        runs[method]=repetitions;times[method]=dict(fit_seconds=fitseconds,total_seconds=time.monotonic()-stage)
                    except (ValueError,tf.errors.OpError) as exc:
                        failures[method]=repr(exc);times[method]=dict(total_seconds=time.monotonic()-stage)
                        write(methoddir/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),fallback_used=False))
                metrics=sequence_metrics(model,observations,runs,refmeans,refz)
                entry=dict(dimension=d,sequence=seq,reference_pass=refpass,reference=refinfo,metrics=metrics,
                    failures=failures,times=times,wall_seconds=time.monotonic()-tic)
                write(dest/'summary.json',entry);result['sequences'].append(plain(entry));write(out/'result.json',result)
        if not args.smoke: result['inference']=infer(result['sequences'])
        result['status']='COMPLETE';manifest['status']='COMPLETE'
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc));manifest.update(status='FAILED',error=repr(exc));traceback.print_exc();raise
    finally:
        manifest['wall_seconds']=time.monotonic()-start;manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        manifest['gpu_allocator']=tf.config.experimental.get_memory_info('GPU:0')
        manifest['source_unchanged']=all(sha(ROOT/p)==v for p,v in manifest['source_hashes'].items())
        write(out/'result.json',result);write(out/'run_manifest.json',manifest)


if __name__=='__main__': main()
