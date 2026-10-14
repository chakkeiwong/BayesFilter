"""Diagnostic comparison of normalized guides and original filtering marginals.

Reference Python arithmetic below is used only for independent tiny controls.
All experiment kernels and uncertainty calculations use TensorFlow/GPU/XLA.
"""
import argparse
import bisect
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
ROOT = Path('/home/chakwong/BayesFilter')
sys.path.insert(0, str(ROOT))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
D = tf.float64
CAMPAIGN = ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
PREVIOUS = CAMPAIGN/'attempt15_c3_figure4'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load('controlled_snapshot', 'controlled_smc_tf.py')
g = load('guided_snapshot', 'gaussian_guided_pf_tf.py')
r = load('grid_reference_snapshot', 'controlled_smc_filter_reference_tf.py')


def plain_reference(method, observations, noise, uniforms, q):
    """Independent scalar Python diagnostic, with explicit normalized densities."""
    def normal(x, m, v):
        return -.5*(math.log(2*math.pi*v)+(x-m)**2/v)
    previous, total, rows = [0.]*len(noise[0]), 0., []
    for t, y in enumerate(observations):
        states, weights = [], []
        for j, old in enumerate(previous):
            m, v = (.99*old, q) if t else (0., 1.)
            if method == 'ekf':
                p = 1/(1+math.exp(-m)); ey = 50*p; slope = 50*p*(1-p)
                sy = slope*slope*v+slope; cov = v*slope
                cv = v*slope/sy
            else:
                points = [m-math.sqrt(3*v),m,m+math.sqrt(3*v)]
                w = [1/6,2/3,1/6]; p = [1/(1+math.exp(-x)) for x in points]
                ey = sum(wi*50*pi for wi,pi in zip(w,p))
                sy = sum(wi*((50*pi-ey)**2+50*pi*(1-pi)) for wi,pi in zip(w,p))
                cov = sum(wi*(xi-m)*(50*pi-ey) for wi,xi,pi in zip(w,points,p))
                cv = v-cov**2/sy
            cm = m+cov/sy*(y-ey)
            x = cm+math.sqrt(cv)*noise[t][j]
            p = 1/(1+math.exp(-x))
            likelihood = math.lgamma(51)-math.lgamma(y+1)-math.lgamma(51-y)+y*math.log(p)+(50-y)*math.log1p(-p)
            weights.append(likelihood+normal(x,m,v)-normal(x,cm,cv)); states.append(x)
        maximum = max(weights); norm = maximum+math.log(sum(math.exp(w-maximum) for w in weights))
        total += norm-math.log(len(weights)); weights = [math.exp(w-norm) for w in weights]
        mean = sum(x*w for x,w in zip(states,weights))
        var = sum(w*(x-mean)**2 for x,w in zip(states,weights))
        cdf=[]; running=0.
        for w in weights:
            running+=w; cdf.append(running)
        cdf[-1]=1.
        indices=[bisect.bisect_left(cdf,(j+uniforms[t])/len(weights)) for j in range(len(weights))]
        previous=[states[j] for j in indices]
        rows.append(dict(mean=mean,variance=var,logZ=total,x=states,ancestors=indices))
    return rows


def as_list(x):
    return x.numpy().tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); out=Path(args.output)
    (out/'estimates').mkdir(); (out/'references').mkdir()
    started=time.monotonic(); rows=[]; checks=[]; grid_checks=[]; traces=[]; costs=[]
    context={}; peak_replay=0.
    observations=json.loads((PREVIOUS/'observations.json').read_text())
    y=tf.constant(observations,D); ybatch=tf.tile(y[:,None],[1,10])
    old_rows=[json.loads(line) for line in (PREVIOUS/'rows.jsonl').read_text().splitlines()]
    old={(row['method'],round(row['process_variance']*100)-1,row['repetition']):row for row in old_rows}
    logg=c.binomial_logistic_observation(50); moments=g.binomial_moments(50)
    regime_names=['all','extreme','central','other']
    extreme=(y<=5)|(y>=45); central=(y>=15)&(y<=35)
    masks=tf.cast(tf.stack([tf.ones_like(extreme),extreme,central,~(extreme|central)]),D)
    counts=as_list(tf.reduce_sum(masks,axis=1)); present=[n>0 for n in counts]
    features=['logZ']+['mean_squared_error_'+name for name in regime_names]+['variance_absolute_error_'+name for name in regime_names]+['prefix_logZ_rmse']
    def reduce_estimates(mean,variance,logz,ref):
        denominator=tf.maximum(tf.reduce_sum(masks,axis=1),1.)[:,None]
        mse=tf.linalg.matmul(masks,(mean-ref[:,0,None])**2)/denominator
        variance_error=tf.linalg.matmul(masks,tf.abs(variance-ref[:,1,None]))/denominator
        prefix=tf.sqrt(tf.reduce_mean((logz-ref[:,2,None])**2,axis=0))
        return tf.concat([logz[-1:, :],mse,variance_error,prefix[None,:]],axis=0)
    reducers={b:tf.function(reduce_estimates,input_signature=[
        tf.TensorSpec([3000,b],D)]*3+[tf.TensorSpec([3000,3],D)],jit_compile=True,autograph=False) for b in (1,10)}
    def guide(h,n,b,method):
        return g.ScalarGaussianGuidedPF(observations=h,particles=n,replicas=b,alpha=.99,
            initial_mean=0.,initial_variance=1.,observation_log_density=logg,
            observation_mean=moments[0],observation_variance=moments[1],
            observation_derivative=moments[2],guide=method)
    def controlled(n,q,b):
        return c.ScalarGaussianControlledSMC(observations=3000,particles=n,replicas=b,alpha=.99,
            process_variance=q,initial_mean=0.,initial_variance=1.,observation_log_density=logg,
            precision_policy='source_scalar_projection')
    def require(name,actual,expected,tolerance):
        error=float(tf.reduce_max(tf.abs(tf.convert_to_tensor(actual,D)-tf.convert_to_tensor(expected,D))).numpy())
        passed=math.isfinite(error) and error<=tolerance
        checks.append(dict(name=name,error=error,tolerance=tolerance,passed=passed))
        if not passed: raise RuntimeError(name+' failed '+str(error))
    def progress():
        (out/'progress.json').write_text(json.dumps(dict(status='RUNNING',context=context,completed_rows=len(rows),wall_seconds=time.monotonic()-started))+'\n')
    def record(method,grid_index,repstart,batch,estimate,reference,seconds,stream):
        nonlocal peak_replay
        if not bool(estimate['valid'].numpy()): raise RuntimeError('Invalid '+method)
        values=reducers[batch](estimate['mean'],estimate['variance'],estimate['logZ'],reference)
        data=as_list(tf.transpose(values))
        for j,metrics in enumerate(data):
            rep=repstart+j
            if method in ('csmc','bpf128','bpf5529'):
                previous=old[('bpf' if method=='bpf5529' else 'csmc',grid_index,rep)]
                expected=previous['stages'][0 if method=='bpf128' else -1]['logZ']
                error=abs(metrics[0]-expected);peak_replay=max(peak_replay,error)
                if error>1e-8+1e-12*abs(expected): raise RuntimeError('Preserved stream/policy replay mismatch')
            else: error=None
            row=dict(method=method,process_variance=(grid_index+1)/100,repetition=rep,
                seed=20260929,stream=stream,batch=batch,batch_replica=j,
                metrics=dict(zip(features,metrics)),replay_absolute_error=error,
                minimum_ess=float(tf.reduce_min(estimate['ess'][:,j]).numpy()),
                allocated_batch_seconds=seconds/batch,valid=True)
            rows.append(row)
            with (out/'rows.jsonl').open('a') as handle: handle.write(json.dumps(row,allow_nan=False)+'\n')
        tensor=tf.stack([estimate['mean'],estimate['variance'],estimate['logZ'],estimate['ess']],axis=-1)
        tf.io.write_file(str(out/'estimates'/f'{method}_g{grid_index:02d}_r{repstart:03d}.tensor'),tf.io.serialize_tensor(tensor))
        progress()
    try:
        # Independent source-length-free preflight precedes every collection.
        tiny_y=[0.,25.,50.,4.,33.,12.]
        tiny_noise=as_list(tf.random.stateless_normal([6,1,7],[20260929,4300000],dtype=D))
        tiny_uniform=as_list(tf.random.stateless_uniform([6,1],[20260929,4300001],dtype=D))
        for method in ('ekf','quadrature3'):
            expected=plain_reference(method,tiny_y,[x[0] for x in tiny_noise],[x[0] for x in tiny_uniform],.12)
            actual=guide(6,7,1,method).forward(tf.constant(tiny_y,D)[:,None],tf.constant(tiny_noise,D),tf.constant(tiny_uniform,D),tf.constant(.12,D))
            for key in ('logZ','mean','variance','x','ancestors'):
                require(method+'_independent_sequential_'+key,tf.cast(actual[key][:,0],D),[row[key] for row in expected],0. if key=='ancestors' else 1e-9)
        pilots=json.loads((CAMPAIGN/'attempt20_c3_guided_validation/summary.json').read_text())['costs']
        guide_estimate=sum(max(row['seconds']/row['replicas'] for row in pilots if row['method']==method and not row['includes_compile'])*2000 for method in ('ekf','quadrature3'))
        replay_estimate=sum(sum(s['forward_seconds'] for s in row['stages'] if s['iteration'] in (0,3))/row['batch'] for row in old_rows)
        reference_engines=[r.GridFilterReference(observations=3000,points=p,bound=bound,alpha=.99,initial_mean=0.,initial_variance=1.,observation_log_density=logg) for p,bound in ((1025,32.),(2049,32.),(2561,40.))]
        reference_seconds=0.
        references=[]
        for index in range(20):
            context=dict(stage='reference',grid_index=index);progress();start=time.monotonic()
            results=[engine.forward(y,tf.constant((index+1)/100,D)) for engine in reference_engines]
            if not all(bool(x['valid'].numpy()) for x in results):raise RuntimeError('Invalid grid oracle')
            errors={key:max(float(tf.reduce_max(tf.abs(result[key]-results[-1][key])).numpy()) for result in results[:-1]) for key in ('mean','variance','logZ')}
            mass_loss=max(float(tf.reduce_max(x['predictive_mass_loss']).numpy()) for x in results)
            ok=errors['logZ']<=1e-6 and errors['mean']<=1e-7 and errors['variance']<=1e-7 and mass_loss<=1e-10
            grid_checks.append(dict(process_variance=(index+1)/100,errors=errors,predictive_mass_loss=mass_loss,passed=ok))
            (out/'grid-convergence.json').write_text(json.dumps(grid_checks,indent=2)+'\n')
            if not ok:raise RuntimeError('Grid convergence failed; do not interpret Monte Carlo errors')
            reference=tf.stack([results[-1][key] for key in ('mean','variance','logZ')],axis=-1)
            references.append(reference)
            tf.io.write_file(str(out/'references'/f'g{index:02d}.tensor'),tf.io.serialize_tensor(reference))
            elapsed=time.monotonic()-start;reference_seconds+=elapsed
            if index==0:
                estimate=1.5*(guide_estimate+replay_estimate+20*elapsed)+120
                (out/'cost-check.json').write_text(json.dumps(dict(estimated_total_seconds=estimate,cap=7200,guide_estimate=guide_estimate,replay_estimate=replay_estimate,first_reference_seconds=elapsed),indent=2)+'\n')
                if estimate>7200:raise RuntimeError('Conservative projected cost exceeds phase cap')
        guides={method:guide(3000,128,10,method) for method in ('ekf','quadrature3')}
        for grid_index,reference in enumerate(references):
            q=(grid_index+1)/100
            for method,engine in guides.items():
                for repstart in range(0,100,10):
                    context=dict(stage='guided',method=method,grid_index=grid_index,repetition=repstart)
                    stream=(5000000 if method=='ekf' else 6000000)+grid_index*10000+(repstart//10)*100
                    start=time.monotonic()
                    result=engine.forward(ybatch,tf.random.stateless_normal([3000,10,128],[20260929,stream],dtype=D),tf.random.stateless_uniform([3000,10],[20260929,stream+1],dtype=D),tf.constant(q,D))
                    result['valid'].numpy();seconds=time.monotonic()-start
                    record(method,grid_index,repstart,10,result,reference,seconds,stream)
            for n,batch in ((128,10),(5529,1)):
                engine=controlled(n,q,batch if batch>1 else None)
                for method,iteration in ([('bpf128',0),('csmc',3)] if n==128 else [('bpf5529',0)]):
                    for repstart in range(0,100,batch):
                        context=dict(stage='replay',method=method,grid_index=grid_index,repetition=repstart)
                        shape=[3000,batch] if batch>1 else [3000]
                        stream=((1000000+grid_index*10000+(repstart//10)*100) if n==128 else (3000000+grid_index*10000+repstart*2))+iteration*2
                        if method=='csmc':
                            raw=tf.io.read_file(str(PREVIOUS/'fit-diagnostics'/f'g{grid_index:02d}_r{repstart:03d}_i2_policy.tensor'))
                            policy=tf.ensure_shape(tf.io.parse_tensor(raw,D),[3000,10,3])
                        else:policy=tf.zeros(shape+[3],D)
                        start=time.monotonic()
                        result=engine.forward(policy,ybatch if batch>1 else y,tf.random.stateless_normal(shape+[n],[20260929,stream],dtype=D),tf.random.stateless_uniform(shape,[20260929,stream+1],dtype=D))
                        original=engine.original_filtering(policy,ybatch if batch>1 else y,result['x'])
                        original['valid'].numpy();seconds=time.monotonic()-start
                        if batch==1: original={key:value if key=='valid' else value[:,None] for key,value in original.items()}
                        record(method,grid_index,repstart,batch,original,reference,seconds,stream)
                traces.append(dict(grid_index=grid_index,particles=n,forward=engine.forward.experimental_get_tracing_count(),original_filtering=engine.original_filtering.experimental_get_tracing_count()))
            (out/'trace-counts.json').write_text(json.dumps(traces,indent=2)+'\n')
        for method,engine in guides.items():
            traces.append(dict(method=method,forward=engine.forward.experimental_get_tracing_count()))
        for i,engine in enumerate(reference_engines):traces.append(dict(reference_grid=i,forward=engine.forward.experimental_get_tracing_count()))
        if any(value!=1 for row in traces for key,value in row.items() if key in ('forward','original_filtering')):raise RuntimeError('Unexpected retracing')
        if len(rows)!=10000:raise RuntimeError('Incomplete comparison matrix')
        # Statistics are fixed before collection; bootstrap entire independent
        # repetitions. BPF128 and cSMC share streams, so their contrast is paired.
        def statistic(values):
            means=tf.reduce_mean(values,axis=-2)
            logz=values[...,0]
            variance=tf.math.reduce_variance(logz,axis=-1)*100/99
            relative=tf.math.log(variance/tf.reduce_mean(logz,axis=-1)**2)/tf.math.log(tf.constant(10.,D))
            return tf.concat([means,relative[...,None]],axis=-1)
        point=tf.function(statistic,input_signature=[tf.TensorSpec([100,10],D)],jit_compile=True,autograph=False)
        bootstrap=tf.function(lambda v,idx:statistic(tf.gather(v,idx)),input_signature=[tf.TensorSpec([100,10],D),tf.TensorSpec([3999,100],tf.int32)],jit_compile=True,autograph=False)
        def ci(values):
            ordered=tf.sort(values,axis=0)
            return as_list(ordered[99]),as_list(ordered[3899])
        feature_names=features+['log10_relative_variance_logZ']
        summaries=[];comparisons=[];observed_losses=[];supported_losses=[]
        for grid_index in range(20):
            estimates={};samples={}
            common=tf.random.stateless_uniform([3999,100],[20260929,7000000+grid_index],minval=0,maxval=100,dtype=tf.int32)
            for method_index,method in enumerate(('csmc','bpf128','bpf5529','ekf','quadrature3')):
                selected=sorted([row for row in rows if row['method']==method and round(row['process_variance']*100)-1==grid_index],key=lambda row:row['repetition'])
                if [row['repetition'] for row in selected]!=list(range(100)):raise RuntimeError('Missing/duplicate repetition')
                values=tf.constant([[row['metrics'][key] for key in features] for row in selected],D)
                indices=common if method in ('csmc','bpf128') else tf.random.stateless_uniform([3999,100],[20260929,7100000+grid_index*10+method_index],minval=0,maxval=100,dtype=tf.int32)
                estimates[method]=point(values);samples[method]=bootstrap(values,indices)
                lower,upper=ci(samples[method])
                summaries.append(dict(method=method,process_variance=(grid_index+1)/100,point=as_list(estimates[method]),ci_lower=lower,ci_upper=upper,
                    reference_logZ=float(references[grid_index][-1,2].numpy()),collection_seconds=sum(row['allocated_batch_seconds'] for row in selected)))
            for method in ('bpf128','bpf5529','ekf','quadrature3'):
                difference=as_list(estimates['csmc']-estimates[method]);lower,upper=ci(samples['csmc']-samples[method])
                row=dict(comparator=method,process_variance=(grid_index+1)/100,paired=method=='bpf128',difference=difference,ci_lower=lower,ci_upper=upper)
                comparisons.append(row)
                for regime in range(1,4):
                    index=1+regime
                    if present[regime] and difference[index]>0:
                        loss=dict(comparator=method,process_variance=(grid_index+1)/100,regime=regime_names[regime],difference=difference[index],ci=[lower[index],upper[index]])
                        observed_losses.append(loss)
                        if lower[index]>0:supported_losses.append(loss)
        (out/'uncertainty.json').write_text(json.dumps(dict(features=feature_names,regimes=dict(zip(regime_names,counts)),pointwise_only=True,bootstrap_replicates=3999,summaries=summaries,contrasts=comparisons),indent=2)+'\n')
        summary=dict(status='PASS',scope='neuroscience_normalized_guides_and_original_filtering_comparisons',
            completed_rows=len(rows),independent_checks=checks,reference_convergence=grid_checks,
            trace_counts=traces,maximum_replay_absolute_error=peak_replay,
            heuristic_dominance=dict(verdict='PROMOTION_VETO_OBSERVED_UNDERPERFORMANCE' if observed_losses else 'NO_OBSERVED_UNDERPERFORMANCE',observed_losses=observed_losses,pointwise_supported_losses=supported_losses,simultaneous_ranking=False),
            gpu_memory_policy=memory,tensorflow=tf.__version__,jit_compile=True,tf32=False,
            device=references[0].device,allocator=tf.config.experimental.get_memory_info('GPU:0'),
            host_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            wall_seconds=time.monotonic()-started,reference_seconds=reference_seconds,
            replication='NEUROSCIENCE_COMPARISON_SCOPE_ONLY',all_replications_complete=False,
            nonclaims=['online causal latency','equal GPU cost','global simultaneous ranking','Lorenz/RBF replication','K empirical learning replication'])
        (out/'progress.json').write_text(json.dumps(dict(status='COMPLETE',completed_rows=len(rows),wall_seconds=time.monotonic()-started))+'\n')
    except BaseException as exc:
        summary=dict(status='FAIL',error=repr(exc),context=context,completed_rows=len(rows),checks=checks,reference_convergence=grid_checks,
            gpu_memory_policy=memory,tensorflow=tf.__version__,jit_compile=True,wall_seconds=time.monotonic()-started)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        raise
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],completed_rows=len(rows),observed_losses=len(observed_losses),pointwise_supported_losses=len(supported_losses))))


if __name__=='__main__':main()
