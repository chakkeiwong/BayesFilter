"""GPU/XLA exact-grid mechanics, not approximation or posterior evidence."""
import json,os,time
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.ledh_exact_block_transport_tf import exact_block_transport_with_tangent
from bayesfilter.highdim.ledh_unified_reset_tf import batched_sinkhorn_contract_e_reset_triple_with_tangent
out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])
records=[]
for dtype in (tf.float64,tf.float32):
    n=4096
    base=tf.cast(tf.random.stateless_normal([1,n,1],[818,917],dtype=tf.float64),dtype)
    delta=tf.cast(tf.random.stateless_normal([1,n,1],[828,917],dtype=tf.float64),dtype)*.1
    logw=tf.cast(tf.random.stateless_normal([1,n],[838,917],dtype=tf.float64),dtype)*.1
    dlogw=tf.cast(tf.random.stateless_normal([1,n],[848,917],dtype=tf.float64),dtype)*.1
    design=base-tf.reduce_mean(base,1,keepdims=True)
    design/=tf.sqrt(tf.reduce_mean(tf.square(design),1,keepdims=True))
    def evaluate(theta):
        x=base+theta*delta
        w=tf.nn.softmax(logw+theta*dlogw,axis=-1)
        dw=w*(dlogw-tf.reduce_sum(w*dlogw,-1,keepdims=True))
        cov=(1+.1*tf.square(x))[:,:,:,None]
        dcov=(.2*x*delta)[:,:,:,None]
        reset=batched_sinkhorn_contract_e_reset_triple_with_tangent(x,delta[None],cov,dcov[None],w,dw[None],design,
            epsilon=2.,sinkhorn_steps=4,balance_steps=4,ridge=1e-5)
        # The trace-sized transport is discarded after inspecting row invariants.
        row_error=tf.reduce_max(tf.abs(tf.reduce_sum(reset[4],-1)-1))
        drow_error=tf.reduce_max(tf.abs(tf.reduce_sum(reset[5],-1)))
        return reset[0],reset[1][0],reset[2],reset[3][0],reset[6],row_error,drow_error
    fn=tf.function(evaluate,input_signature=[tf.TensorSpec([],dtype)],jit_compile=True)
    started=time.monotonic()
    theta=tf.constant(.4,dtype)
    v,dv,c,dc,valid,row,drow=fn(theta)
    steps=[]
    epsilon=2.**(-52 if dtype==tf.float64 else -23)
    for multiplier in (.125,.25,.5,1.,2.,4.,8.):
        h=tf.constant(epsilon**(1/3)*multiplier,dtype)
        plus=fn(theta+h);minus=fn(theta-h)
        value_error=float(tf.reduce_max(tf.abs((plus[0]-minus[0])/(2*h)-dv)/tf.maximum(1.,tf.abs(dv))))
        covariance_error=float(tf.reduce_max(tf.abs((plus[2]-minus[2])/(2*h)-dc)/tf.maximum(1.,tf.abs(dc))))
        steps.append(dict(h=float(h),value_error=value_error,covariance_error=covariance_error,
            valid=bool(tf.reduce_all(plus[4]) & tf.reduce_all(minus[4]))))
    records.append(dict(dtype=dtype.name,n=n,k=2048,block_grid=[2,2],steps=steps,
        row_sum_error=float(row),row_tangent_error=float(drow),
        seconds=time.monotonic()-started,valid=bool(tf.reduce_all(valid))))
    (out/'block-validation.json').write_text(json.dumps(dict(classification='engineering_only',memory_policy=memory,
        records=records,allocator=tf.config.experimental.get_memory_info('GPU:0')),indent=2)+'\n')
    # Master-program derivative budgets, assessed over the rounding/truncation
    # ladder; the previous single float32 step is retained as failed evidence.
    limit=1e-6 if dtype==tf.float64 else 1e-2
    assert bool(tf.reduce_all(valid)) and all(r['valid'] for r in steps)
    assert min(max(r['value_error'],r['covariance_error']) for r in steps)<limit
    assert float(row)<(1e-11 if dtype==tf.float64 else 1e-4)
    assert float(drow)<(1e-11 if dtype==tf.float64 else 1e-4)
    print(records[-1],flush=True)
(out/'block-validation.json').write_text(json.dumps(dict(classification='engineering_only',memory_policy=memory,
    records=records,allocator=tf.config.experimental.get_memory_info('GPU:0')),indent=2)+'\n')
