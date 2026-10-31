"""Optional GPU diagnostic of an algebraically equivalent Lorenz Bellman fit.

It uses repository Gaussian/QR/projection kernels. It is not a runtime default:
this experiment separates recursion arithmetic from the subsequent safety issue.
"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,'/home/chakwong/BayesFilter')
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
from bayesfilter.replication.lorenz_refinement_tf import LorenzRefinement
from bayesfilter.replication.multivariate_controlled_smc_tf import covariance_from_precision,constrain_precision_increment,quadratic
D=tf.float64


def absolute_integral(mean,precision,logdet,a,b,c):
    covariance,ld,margin,valid=covariance_from_precision(precision+2*a)
    shift=tf.einsum('bni,ij->bnj',mean,precision)-b[:,None,:]
    factor=tf.linalg.cholesky(precision+2*a)
    new_mean=tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(factor,tf.linalg.matrix_transpose(shift)))
    difference=new_mean-mean
    integral=.5*(ld-logdet)[:,None]-quadratic(new_mean,a,b,c)
    integral-=.5*tf.einsum('bni,ij,bnj->bn',difference,precision,difference)
    return integral,tf.reduce_all(valid)&tf.reduce_all(tf.math.is_finite(integral))


def make_fit(engine,jit):
    h,b,d,n=engine.h,engine.b,engine.d,engine.n
    specs=list(engine.fit.input_signature)+[tf.TensorSpec([h,b,6],D)]
    def body(old_a,old_b,old_c,x,weights,y):
        arrays=[tf.TensorArray(D,size=h,element_shape=shape) for shape in
            ([b,d,d],[b,d],[b],[b,n],[b],[b],[b,d,d])]
        obs_a=tf.broadcast_to(tf.linalg.diag(tf.constant([5000.]*6+[0.]*2,D)),[b,d,d])
        def step(t,na,nb,nc,valid,arrays):
            future,good=tf.cond(t<h-1,
                lambda:absolute_integral(engine.f(x[t]),engine.p,engine.ldq,na,nb,nc),
                lambda:(tf.zeros([b,n],D),tf.constant(True)))
            fit=engine._regress(x[t],-future)
            # Add the exact observation quadratic; remove the old polynomial to
            # give the unchanged projection routine its required increment.
            obs_b=-tf.pad(y[t],[[0,0],[0,2]])/1e-4
            obs_c=5000*tf.reduce_sum(y[t]**2,-1)
            da=obs_a-old_a[t];db=obs_b-old_b[t];dc=obs_c-old_c[t]
            mu,s=fit['center'],fit['scale']
            fit['A']+=da;fit['b']+=db;fit['c']+=dc
            fit['centered_A']+=da*s[:,:,None]*s[:,None,:]
            fit['centered_b']+=s*(db+2*tf.einsum('bi,bij->bj',mu,da))
            fit['centered_c']+=tf.einsum('bi,bij,bj->b',mu,da,mu)+tf.reduce_sum(db*mu,-1)+dc
            fit,safety=constrain_precision_increment(tf.where(t==0,engine.p0,engine.p),old_a[t],fit,
                engine.precision_policy,engine.precision_floor,engine.precision_eigen_tolerance,engine.precision_eigen_refinement,
                engine.precision_activation_margin)
            na,nb,nc=old_a[t]+fit['A'],old_b[t]+fit['b'],old_c[t]+fit['c']
            _,_,margin,pd=covariance_from_precision(tf.where(t==0,engine.p0,engine.p)+2*na)
            valid &= good&fit['valid']&tf.reduce_all(pd)
            values=(na,nb,nc,future,margin,safety['protection_applied'],fit['projection_added_A'])
            return t-1,na,nb,nc,valid,[arr.write(t,v) for arr,v in zip(arrays,values)]
        _,_,_,_,valid,arrays=tf.while_loop(lambda t,*_:t>=0,step,
            (tf.constant(h-1),tf.zeros_like(old_a[0]),tf.zeros_like(old_b[0]),tf.zeros_like(old_c[0]),tf.constant(True),arrays),parallel_iterations=1)
        result=dict(zip(('A','b','c','future_integral','precision_margin','protection_applied','projection_added_A'),[a.stack() for a in arrays]))
        result['valid']=valid;return result
    return tf.function(body,input_signature=specs,jit_compile=jit,autograph=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic();campaign=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    summary=dict(status='FAIL',scope='EQUIVALENT_ABSOLUTE_BELLMAN_ARITHMETIC_DIAGNOSTIC',
        device='GPU/XLA plus CPU graph diagnostic',gpu_memory_policy=memory,dtype='float64',tf32=False,jit_compile=True,
        comparisons={},replication='INCOMPLETE',candidate_admitted=False)
    def load(folder,k):return tf.io.parse_tensor(tf.io.read_file(str(folder/(k+'.tensor'))),D)
    def compare(a,b):
        if a.shape!=b.shape:raise ValueError('Comparison shape mismatch')
        diff=tf.abs(a-b);scaled=diff/(1e-7+1e-9*tf.abs(b))
        return dict(passed=bool(tf.reduce_all(tf.math.is_finite(scaled)&(scaled<=1)).numpy()),
            max_absolute=float(tf.reduce_max(diff).numpy()),max_scaled=float(tf.reduce_max(scaled).numpy()))
    try:
        candidate=campaign/'attempt108_c2_lorenz_bounded_integral_safety/covariance_floor_0.001'
        values=[load(candidate,k)[3] for k in ('A','b','c','x','log_incremental_weight')]
        y=load(campaign/'attempt76_c2_lorenz_batch_validation/rejected_batch','y')
        engine=LorenzRefinement(forcing=2.5,replicas=4,precision_policy='nearest_covariance',precision_floor=.001,
            precision_eigen_tolerance=8*2.220446049250313e-16,precision_eigen_refinement=True,precision_stable_integral=True).engine
        gpu_fit,cpu_fit=make_fit(engine,True),make_fit(engine,False)
        with tf.device('/GPU:0'):gpu=gpu_fit(*values,y)
        with tf.device('/CPU:0'):cpu=cpu_fit(*values,y)
        for label,result in (('gpu',gpu),('cpu_graph',cpu)):
            folder=args.output/label;folder.mkdir()
            for k,v in result.items():tf.io.write_file(str(folder/(k+'.tensor')),tf.io.serialize_tensor(v))
        summary['valid']={label:bool(result['valid'].numpy()) for label,result in (('gpu',gpu),('cpu_graph',cpu))}
        summary['comparisons']={k:compare(gpu[k],cpu[k]) for k in ('A','b','c','future_integral','projection_added_A')}
        summary['traces']=[gpu_fit.experimental_get_tracing_count(),cpu_fit.experimental_get_tracing_count()]
        summary['status']='PASS' if all(summary['valid'].values()) and summary['traces']==[1,1] else 'FAIL'
        summary['cross_backend_arithmetic_agreement']=all(v['passed'] for v in summary['comparisons'].values())
    except Exception as exc:
        summary['error']=repr(exc);import traceback;traceback.print_exc()
    summary['wall_seconds']=time.monotonic()-start
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
