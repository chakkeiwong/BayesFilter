#!/usr/bin/env python3
"""A05 safety calibration and exact Gaussian consumer diagnostic, not promotion."""
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
PLAN=ROOT/'docs/plans/observation-aware-tt-master-amendment-05-defense-consumer-20260915.md'
OLD=ROOT/'docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914'
A04=ROOT/'docs/benchmarks/artifacts/observation_tt_sgqf_initialization_20260915/attempt-01'


def plain(value):
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    if hasattr(value,'numpy'): value=value.numpy()
    if hasattr(value,'tolist'): value=value.tolist()
    return value


def write(path,value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--wall-budget-seconds',type=float,default=300.)
    args=parser.parse_args()
    start=time.monotonic()
    out=Path(args.output_root).resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    physical=tf.config.list_physical_devices('GPU')
    if not physical: raise RuntimeError('Trusted GPU access required')
    growth=[]
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu,True)
        verified=tf.config.experimental.get_memory_growth(gpu)
        if not verified: raise RuntimeError('Unverified growth')
        growth.append(dict(device=gpu.name,growth=verified,details=tf.config.experimental.get_device_details(gpu)))
    from bayesfilter.highdim import observation_guided_tt_tf as lib
    from bayesfilter.highdim import pair_block_tt_tf as pair
    from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal
    from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as proj
    D=tf.float64
    dependencies=[Path(__file__).resolve(),PLAN,ROOT/'bayesfilter/highdim/sgqf_joint_consumer_tf.py',
        ROOT/'bayesfilter/highdim/pair_block_tt_tf.py',ROOT/'bayesfilter/highdim/observation_guided_tt_tf.py',
        ROOT/'bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py',ROOT/'docs/benchmarks/run_observation_aware_tt_complete.py',
        ROOT/'docs/benchmarks/observation_tt_sgqf_projection_diagnostic.py']
    manifest=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv],plan=str(PLAN),result=str(out/'result.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=ROOT,text=True),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies},inputs={},
        environment=sys.prefix,python=sys.version,tensorflow=tf.__version__,numerical_dtype='float64',
        cpu_only=False,gpu_intentionally_hidden=False,jit_compile=True,
        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),trust_basis='escalated_gpu_access',
        gpu_memory_policy=dict(schema='a05_growth_v1',mode='memory_growth',devices=growth),
        reference_exceptions='One-time Gaussian setup and post-run row diagnostics use TensorFlow outside XLA; repeated sampler kernels are XLA.',
        data_version='exposed A04 cases with frozen coefficients; new numerical rows only',
        seeds=dict(stress=915000,rows='915100+1000*d+10*t+r',replicates=[0,1,2]),row_count=8192)
    result=dict(status='RUNNING',calibration=[],stress=[],cases=[],filter_promoted=False)
    write(out/'run_manifest.json',manifest);write(out/'result.json',result)
    def load(path):
        manifest['inputs'][str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    def budget():
        if time.monotonic()-start>args.wall_budget_seconds: raise TimeoutError('A05 attempt budget')
    def h2(a,b,w):
        value=1.-tf.reduce_sum(w*a*b)/tf.sqrt(tf.reduce_sum(w*a*a)*tf.reduce_sum(w*b*b))
        tf.debugging.assert_greater_equal(value,tf.constant(-1e-12,D))
        return tf.maximum(0.,value)
    def quantiles(x):
        values=tf.sort(tf.reshape(x,[-1]));n=int(values.shape[0])
        return {str(q):tf.gather(values,int(q*(n-1))) for q in (0.,.01,.5,.99,1.)}
    try:
        eps=2.220446049250313e-16;stress=eps**.5;N=512
        grid=[0.,1e-8,1e-6,1e-5,1e-4,1e-3,.005,.01,.05]
        lower=(N-1)*stress/(1+(N-1)*stress)
        upper=.25/(N-1+.25)
        result['derived_interval']=[lower,upper]
        for a in grid:
            healthy=a/((1-a)*.25+a)
            rescue=a/((1-a)*stress+a)
            result['calibration'].append(dict(a=a,healthy_expected_changes=N*healthy,
                stress_expected_unprotected=N*(1-rescue),bounds_pass=(a>0 and lower<=a<=upper)))
        # Both dimensions are tested against the same analytical conditions.
        for d in (1,4):
            for nonconstant in (False,True):
                values=[0.,eps,stress,.25,1.,4.]
                v=tf.concat([tf.repeat(tf.sqrt(tf.constant(values,D)),N)[:,None],tf.zeros([N*6,d-1],D)],axis=1)
                n=int(v.shape[0])
                noise=tf.random.stateless_normal([n,d],[915000+d,0],dtype=D)
                uniform=tf.random.stateless_uniform([n,d],[915000+d,1],minval=1e-12,maxval=1-1e-12,dtype=D)
                mixture=tf.random.stateless_uniform([n],[915000+d,2],dtype=D)
                hu=tf.constant([1.,.2 if nonconstant else 0.],D)
                hu/=tf.linalg.norm(hu)
                first=tf.reshape(hu[:,None]*tf.constant([[0.,1.]],D),[1,2,2,1])
                unit=tf.reshape(tf.constant([1.,0.,0.,0.],D),[1,2,2,1])
                base=(first,)+(unit,)*(d-1)
                sampler=pair.compiled_pair_sampler([c.shape for c in base],True)
                for a in grid:
                    budget()
                    if a==0.: continue  # Undefined at zero mass, excluded analytically.
                    reference=None
                    for scale in (1.,1e-4,1e4):
                        cores=(first*scale,*base[1:])
                        Z=pair.pair_total_mass(cores);tau=Z*a/(1-a)
                        x,logq,diag=sampler(cores,v,tau,mixture,uniform,noise)
                        if not bool(diag['finite']) or float(diag['cdf_residual'])>1e-8:
                            raise ValueError('Stress sampler validity veto')
                        c=pair.conditional_normalizer_batched(pair.pair_conditional_cores(cores,v))
                        rows=tf.stack([x,v],axis=-1);h=pair.evaluate_pair_cores(cores,rows)
                        exact=tf.math.log(h*h+tau)+_log_standard_normal(x)-tf.math.log(c+tau)
                        tf.debugging.assert_near(logq,exact,atol=1e-10,rtol=1e-10)
                        tf.debugging.assert_equal(x[:N],noise[:N])
                        if not nonconstant:
                            tf.debugging.assert_near(logq,_log_standard_normal(x),atol=1e-10,rtol=1e-10)
                        if reference is None: reference=(x,logq)
                        else:
                            tf.debugging.assert_near(x,reference[0],atol=1e-9,rtol=1e-9)
                            tf.debugging.assert_near(logq,reference[1],atol=1e-9,rtol=1e-9)
                        result['stress'].append(dict(d=d,nonconstant=nonconstant,a=a,scale=scale,
                            max_density_error=tf.reduce_max(tf.abs(logq-exact)),**diag))
                    # Common random numbers: same polynomial branch must return same x.
                    if nonconstant:
                        x0,_,_=sampler(base,v[N:],tf.constant(0.,D),mixture[N:],uniform[N:],noise[N:])
                        c0=pair.conditional_normalizer_batched(pair.pair_conditional_cores(base,v[N:]))
                        same=mixture[N:]<c0/(c0+a/(1-a))
                        tf.debugging.assert_near(tf.boolean_mask(reference[0][N:],same),tf.boolean_mask(x0,same),atol=1e-10,rtol=1e-10)
        passing=[row['a'] for row in result['calibration'] if row['bounds_pass']]
        if not passing: raise ValueError('No defense candidate passes both bounds')
        selected=min(passing)
        result['selected_defensive_mass']=selected
        result['safety_candidate_status']='OPTIONAL_BOUNDED_SAMPLER_SAFETY_ONLY'
        write(out/'frozen-safety-selection.json',dict(selected=selected,criterion='derived rescue/non-harm bounds plus sampler validity',interval=[lower,upper],selected_before_target_diagnostics=True))
        fixture=load(OLD/'diagnostic-02/downstream_fixture.json')
        oldmanifest=load(OLD/'campaign-02/run_manifest.json')
        for name in ('bayesfilter/highdim/observation_guided_tt_tf.py','bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py'):
            if sha(ROOT/name)!=oldmanifest['source_hashes'][name]: raise RuntimeError('Target source drift')
        for d in (1,4):
            data=fixture['dimensions'][str(d)]
            model=lib.SVModel(tf.constant(data['A'],D),tf.constant(data['P0'],D),fixture['beta'],fixture['sigma'])
            saved=load(OLD/f'campaign-02/d{d}/tt_pair_block_proposals.json')
            for t,incoming in ((1,'gaussian'),(1,'recursive_tt'),(17,'recursive_tt'),(18,'recursive_tt')):
                record,prev=saved[t],saved[t-1]
                current=lib.Chart(tf.constant(record['mean'],D),tf.constant(record['factor'],D))
                condition=lib.Chart(tf.constant(record['condition_mean'],D),tf.constant(record['condition_factor'],D))
                previouscores=tuple(tf.constant(c,D) for c in prev['cores'])
                retained=lib.PairRetainedProposal(previouscores,condition,pair.pair_total_mass(previouscores),tf.constant(prev['tau'],D),t-1)
                sgstep=joint.make_sgqf_joint_step(model,current,condition,t)
                mean,covariance=proj.paired_gaussian(model,current,condition)
                sggauss=lib.Chart.from_moments(mean,covariance)
                for r in range(3):
                    budget();identifier=f'd{d}-t{t}-{incoming}-r{r}'
                    case=load(A04/(identifier+'.json'))
                    coordinates,logw,rowinfo=lib.joint_sgqf_row_sampler(model,current,condition,8192,915100+1000*d+10*t+r)
                    rows=tf.stack([coordinates[:,:d],coordinates[:,d:]],axis=-1)
                    x,z=current.forward(rows[:,:,0]),condition.forward(rows[:,:,1])
                    priorlog=condition.log_prob(z) if incoming=='gaussian' else retained.physical_log_density(z)
                    logs=model.observation_log_prob(x,tf.constant(case['observation'],D))+model.transition_log_prob(x,z)+priorlog+current.logdet+condition.logdet-_log_standard_normal(coordinates)
                    scale=tf.reduce_logsumexp(logs+logw)-tf.math.log(tf.constant(8192.,D))
                    target=tf.exp(.5*(logs-scale));w=tf.exp(logw)
                    normalized_target_rows=w*target**2/tf.reduce_sum(w*target**2)
                    gaussamp=tf.exp(.5*(sggauss.log_prob(tf.reshape(rows,[-1,2*d]))-_log_standard_normal(coordinates)))
                    output=dict(id=identifier,dimension=d,time=t,incoming=incoming,row_diagnostics=rowinfo,
                        target_row_ess=1/tf.reduce_sum(normalized_target_rows**2),
                        maximum_target_row_share=tf.reduce_max(normalized_target_rows),
                        sgqf_h2=h2(gaussamp,target,w),product_h2=h2(tf.ones_like(target),target,w),families={})
                    draw,logq,_=joint.sample_joint_step(sgstep,z[:512],915900+r)
                    tf.debugging.assert_near(logq,sgstep.conditional_log_density(draw,z[:512]),atol=1e-10,rtol=1e-10)
                    for family in ('generic','sgqf'):
                        fit=case['fits'][family]
                        frozen=next(g for g in fit['grid'] if g['l1']==fit['selected_l1'])
                        cores=tuple(tf.constant(c,D) for c in frozen['cores'])
                        h=pair.evaluate_pair_cores(cores,rows);Z=pair.pair_total_mass(cores)
                        c=pair.conditional_normalizer_batched(pair.pair_conditional_cores(cores,rows[:,:,1]));ratio=c/Z
                        scores=[]
                        for a in (0.,selected,.05):
                            amplitude=tf.sqrt((1-a)*h*h/Z+a)
                            e=a/((1-a)*ratio+a) if a else tf.zeros_like(ratio)
                            hval=h2(amplitude,target,w)
                            scores.append(dict(a=a,h2=hval,conditional_gaussian_fraction=quantiles(e),
                                worst_reference_over_q_bound=tf.reduce_max(tf.math.divide_no_nan(tf.ones_like(e),e)) if a else None,
                                healthy_rows=int(tf.reduce_sum(tf.cast(ratio>=.25,tf.int32))),
                                stress_rows=int(tf.reduce_sum(tf.cast(ratio<=stress,tf.int32))),
                                observed_loss_to_sgqf=bool(hval>output['sgqf_h2'])))
                        output['families'][family]=dict(c_over_Z=quantiles(ratio),scores=scores)
                    result['cases'].append(output)
                    write(out/(identifier+'.json'),output);write(out/'result.json',result)
                    print(identifier,flush=True)
        result.update(status='COMPLETE',consumer_mechanics='PASS',
            heuristic_dominance_verdict='PROMOTION_VETO' if any(s['observed_loss_to_sgqf'] for c in result['cases'] for f in c['families'].values() for s in f['scores'] if s['a']==selected) else 'DESCRIPTIVE_SCREEN_PASS',
            statistically_supported_ranking=False)
        manifest['status']='COMPLETE'
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc));manifest.update(status='FAILED',error=repr(exc))
        traceback.print_exc();raise
    finally:
        manifest['wall_seconds']=time.monotonic()-start
        manifest['gpu_allocator']=tf.config.experimental.get_memory_info('GPU:0')
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        write(out/'result.json',result);write(out/'run_manifest.json',manifest)


if __name__=='__main__': main()
