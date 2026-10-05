#!/usr/bin/env python3
"""Independent diagnostic completion checks; no runtime or admission role."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/plans/artifacts'


def source(name):
    for suffix in ('02','01'):
        folder=BASE/('surrogate-hmc-marginal-weight-20261005-'+suffix)/name
        if (folder/'result.json').exists():return folder
    raise FileNotFoundError(name)


def prior_trajectory_reference(tf,m,spec,observed,n):
    """Naive full-prior trajectory importance sampling, deliberately no resampling."""
    dtype=tf.float64;d,p,t=(spec[k] for k in ('d','p','t'));model=spec['model']
    @tf.function(input_signature=[tf.TensorSpec([p],dtype),tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
    def evaluate(theta,seed):
        if spec['kind']=='m13':mean,covariance=spec['fixture'].m0,spec['fixture'].P0
        else:mean,covariance=model.initial_distribution_fn(theta)
        x=mean+tf.random.stateless_normal([n,d],seed,dtype=dtype)@tf.transpose(tf.linalg.cholesky(covariance))
        rows=tf.broadcast_to(theta,[n,p])
        q=model.process_covariance if model.process_covariance_fn is None else model.process_covariance_fn(theta)
        chol=tf.linalg.cholesky(q)
        def body(i,x,logw):
            x=model.transition_mean_fn(rows,x)+tf.random.stateless_normal([n,d],seed+[0,i+1],dtype=dtype)@tf.transpose(chol)
            if model.observation_log_density_fn is not None:
                logg=model.observation_log_density_fn(rows,x,observed[i])
            else:
                residual=observed[i]-model.observation_fn(x)
                c=tf.linalg.cholesky(model.observation_covariance)
                z=tf.transpose(tf.linalg.triangular_solve(c,tf.transpose(residual)))
                logg=-.5*tf.reduce_sum(z*z,axis=1)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(c)))-.5*spec['m']*tf.constant(math.log(2*math.pi),dtype)
            return i+1,x,logw+logg
        _,_,logw=tf.while_loop(lambda i,*_:i<t,body,(tf.constant(0),x,tf.zeros([n],dtype)))
        return tf.reduce_logsumexp(logw)-tf.math.log(tf.cast(n,dtype))
    return evaluate


def gaussian_reference(tf,m,spec,observed):
    """Kalman / moment-matched KSC / EKF comparator; AD is diagnostic only."""
    dtype=tf.float64;d,p=(spec[k] for k in ('d','p'));kind=spec['kind']
    if kind.startswith('lg'):return m.kalman_reference(spec,observed)
    mixture=m.ksc_1998_log_chi_square_mixture()
    weights,means,variances=(tf.cast(getattr(mixture,k),dtype) for k in ('weights','means','variances'))
    mixture_mean=tf.reduce_sum(weights*means)
    mixture_variance=tf.reduce_sum(weights*(variances+means*means))-mixture_mean**2
    @tf.function(input_signature=[tf.TensorSpec([p],dtype)],jit_compile=True,autograph=False)
    def evaluate(theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            if kind=='ksc':
                mean,covariance=spec['model'].initial_distribution_fn(theta)
                f=tf.reshape(theta[0],[1,1]);q=tf.ones([1,1],dtype)
                r=tf.reshape(mixture_variance,[1,1])
            else:
                fixture=spec['fixture'];mean,covariance=fixture.m0,fixture.P0
                f=fixture.A;q=fixture.Q*tf.exp(2*theta[0]);r=fixture.R*tf.exp(2*theta[1])
            total=tf.zeros([],dtype)
            for i in range(spec['t']):
                mean=tf.linalg.matvec(f,mean);covariance=f@covariance@tf.transpose(f)+q
                if kind=='ksc':
                    predicted=mean+2*tf.math.log(theta[1])+mixture_mean
                    h=tf.ones([1,1],dtype);residual=observed[i]-predicted
                else:
                    x,y=mean[0],mean[1];radius2=x*x+y*y+1e-12;radius=tf.sqrt(radius2)
                    predicted=tf.stack([radius,tf.atan2(y,x)])
                    z=tf.zeros([],dtype)
                    h=tf.stack([tf.stack([x/radius,y/radius,z,z]),tf.stack([-y/radius2,x/radius2,z,z])])
                    delta=observed[i]-predicted;pi=tf.constant(math.pi,dtype)
                    residual=tf.stack([delta[0],tf.math.floormod(delta[1]+pi,2*pi)-pi])
                s=h@covariance@tf.transpose(h)+r;c=tf.linalg.cholesky(s)
                solved=tf.linalg.cholesky_solve(c,residual[:,None])[:,0]
                total+=-.5*(spec['m']*tf.constant(math.log(2*math.pi),dtype)+2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(c)))+tf.reduce_sum(residual*solved))
                gain=tf.transpose(tf.linalg.cholesky_solve(c,h@covariance))
                mean+=tf.linalg.matvec(gain,residual)
                stable=tf.eye(d,dtype=dtype)-gain@h
                covariance=stable@covariance@tf.transpose(stable)+gain@r@tf.transpose(gain)
        return dict(value=total,score=tape.gradient(total,theta))
    return evaluate


def run(args,tf,m):
    if args.phase=='tile':
        from bayesfilter.highdim.ledh_marginal_weights_tf import gaussian_mixture_log_density_tangent
        from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
        n=4096;dtype=tf.float64
        x=tf.linspace(tf.constant(-2.,dtype),tf.constant(2.,dtype),n)[:,None];dx=.1*tf.sin(x)
        mu=tf.constant([-.6,.8],dtype);dmu=tf.constant([.12,-.2],dtype)
        v=tf.constant([.8,1.2],dtype);dv=tf.constant([.1,.2],dtype);dlw=tf.constant([.3,-.3],dtype)
        inputs=(x,dx,tf.repeat(mu,n//2)[:,None],tf.repeat(dmu,n//2)[:,None],
            tf.repeat(v,n//2)[:,None,None],tf.repeat(dv,n//2)[:,None,None],
            tf.fill([n],-tf.math.log(tf.cast(n,dtype))),tf.repeat(dlw,n//2))
        function=tf.function(gaussian_mixture_log_density_tangent,
            input_signature=[tf.TensorSpec(z.shape,dtype) for z in inputs],jit_compile=True,autograph=False)
        values,tangent,valid,force_valid=function(*inputs)
        def reduced(h):
            variance=v+h*dv;residual=x+h*dx-(mu+h*dmu)[None]
            terms=tf.nn.log_softmax(h*dlw)[None]-.5*(tf.constant(math.log(2*math.pi),dtype)+tf.math.log(variance)[None]+residual**2/variance[None])
            return tf.reduce_logsumexp(terms,axis=1)
        error=float(tf.reduce_max(tf.abs(values-reduced(0.))))
        derivatives=[dict(step=h,maximum_error=float(tf.reduce_max(tf.abs(tangent-(reduced(h)-reduced(-h))/(2*h))))) for h in (1e-4,3e-5)]
        return dict(passed=bool(valid & force_valid) and error<1e-11 and all(r['maximum_error']<1e-7 for r in derivatives),
            n=n,tile_size=select_transport_chunks(n).row_chunk_size,value_error=error,derivative_checks=derivatives)
    rows=[]
    for name in m.PHASES:
        folder=source(name);result=json.loads((folder/'result.json').read_text())
        if not result.get('scope'):
            rows.append(dict(name=name,status='candidate rejected before claim'));continue
        data=json.loads((folder/'data.json').read_text())
        spec=m.specification(name,tf.float64);observed=tf.constant(data['observations']['claim'],tf.float64)
        n=data['n']
        if args.phase=='precision':
            point=1 if spec['kind']=='ksc' else len(spec['points'])-1
            theta=tf.cast(tf.constant(spec['points'][point],tf.float32),tf.float64)
            observed=tf.cast(tf.cast(observed,tf.float32),tf.float64)
            design=m.designs(spec,n,'claim',0,tf.float64)
            for policy in ('ancestor','marginal_mixture'):
                function=m.compile_filter(spec,n,result['scope']['controls'],policy,tf.float64)
                precise=m.serial(function(theta,*design,observed));single=result['claim_rows'][point][0][policy]
                current_spec=m.specification(name,tf.float32)
                current_function=m.compile_filter(current_spec,n,result['scope']['controls'],policy,tf.float32)
                current=m.serial(current_function(tf.cast(theta,tf.float32),*[tf.cast(v,tf.float32) for v in design],tf.cast(observed,tf.float32)))
                finite=precise['valid'] and single['valid']
                parity=(current['valid']==single['valid'] and current['value']==single['value'] and current['score']==single['score'])
                rows.append(dict(name=name,point=point,policy=policy,fp64=precise,fp32=single,
                    repaired_fp32=current,bitwise_saved_run_parity=parity,
                    absolute_value_difference=abs(precise['value']-single['value']) if finite else None,
                    scaled_score_difference=math.sqrt(sum((.1*(a-b))**2 for a,b in zip(precise['score'],single['score']))/spec['p']) if finite else None))
        else:
            naive=prior_trajectory_reference(tf,m,spec,observed,n)
            gaussian=gaussian_reference(tf,m,spec,observed)
            for i,point in enumerate(spec['points']):
                theta=tf.constant(point,tf.float64);reference=result['references'][i]
                g=m.serial(gaussian(theta))
                values=[float(naive(theta,tf.constant([20261010,rep+i*1000],tf.int32))) for rep in range(32)]
                mean=lambda x:sum(x)/len(x)
                candidate=mean([abs(row['marginal_mixture']['value']-reference['value']) for row in result['claim_rows'][i]])
                competitors=dict(prior_trajectory_IS=mean([abs(v-reference['value']) for v in values]),
                    bootstrap_PF=mean([abs(row['bootstrap_value']-reference['value']) for row in result['claim_rows'][i]]),
                    gaussian_filter=abs(g['value']-reference['value']))
                rows.append(dict(name=name,point=i,n=n,candidate_mean_absolute_value_error=candidate,
                    competitor_mean_absolute_value_errors=competitors,prior_trajectory_values=values,gaussian=g,
                    oracle_equivalent_comparator='gaussian_filter' if spec['kind'].startswith('lg') else None,
                    descriptive_underperformance=[key for key,value in competitors.items() if candidate>value],
                    statistically_supported_ranking=False,
                    interpretation='diagnostic falsification; numerical oracle distance remains primary, no new selection or tuning'))
        m.save(args.output/(args.phase+'-progress.json'),rows)
        print(args.phase,name,'complete',flush=True)
    return dict(rows=rows,scope='independent diagnostics; conditional on saved datasets; no admission authority')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('tile','precision','heuristics'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='GPU-68251639-fe82-8f81-3ccc-2953c32e805b')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['TF_NUM_INTRAOP_THREADS']='6';os.environ['TF_NUM_INTEROP_THREADS']='2'
    sys.path.insert(0,str(ROOT));import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import marginal_weight_campaign as m
    started=time.monotonic()
    result=run(args,tf,m)
    result['manifest']=dict(command=sys.argv,wall_seconds=time.monotonic()-started,gpu=args.gpu,
        gpu_memory_policy=memory,tensorflow=tf.__version__,tf32=True,jit_compile=True,dtype='float64 diagnostic reference',
        plan='docs/plans/surrogate-hmc-marginal-weight-follow-through-2026-10-05.md',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        random_seeds='prior_IS [20261010, replicate+1000*point], 32 replicas; precision replays claim replica 0; tile deterministic',
        allocator=tf.config.experimental.get_memory_info('GPU:0'))
    m.save(args.output/(args.phase+'.json'),result)
    print(args.phase,'COMPLETE',round(result['manifest']['wall_seconds'],2),flush=True)
    if args.phase=='tile' and not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
