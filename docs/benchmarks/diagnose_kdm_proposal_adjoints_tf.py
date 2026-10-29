"""GPU/XLA diagnostic local VJPs at identical pinned-author inputs/adjoints.

NumPy loads independent reference arrays only; all computations use TensorFlow.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')
ROOT=Path('/home/chakwong/BayesFilter');sys.path.insert(0,str(ROOT))

def main(output,reference):
    import numpy as np
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    source=json.loads((reference/'summary.json').read_text())
    graphs={};rows=[]
    def function(name,inputs,body):
        key=(name,tuple((tuple(x.shape),x.dtype.name) for x in inputs))
        if key not in graphs:
            graphs[key]=tf.function(body,input_signature=[tf.TensorSpec(x.shape,x.dtype) for x in inputs],
                                    autograph=False,jit_compile=True)
        return graphs[key](*inputs)
    def dense(x,w,b,g):
        with tf.GradientTape() as tape:
            tape.watch((x,w,b));y=tf.matmul(x,w,transpose_b=True)+b
        return (y,*tape.gradient(y,(x,w,b),output_gradients=g))
    def sigmoid(z,scale,g):
        with tf.GradientTape() as tape:
            tape.watch(z);y=(tf.sigmoid(z)*2-1)*scale
        return tape.gradient(y,z,output_gradients=g)
    def normal(q,g):
        with tf.GradientTape() as tape:
            tape.watch(q)
            mag=tf.sqrt(tf.square(q[...,2])+tf.square(q[...,3]))
            y=tf.concat([tf.clip_by_value(q[...,:2],-10000.,10000.),q[...,2:]/mag[...,None]],-1)
        return y,tape.gradient(y,q,output_gradients=g)
    def check(label,actual,expected,record):
        expected=tf.convert_to_tensor(expected,actual.dtype)
        err=tf.abs(actual-expected)
        scale=err/(tf.constant(128*2**-23,actual.dtype)*(1+tf.abs(expected)))
        finite=bool(tf.reduce_all(tf.math.is_finite(actual)).numpy())
        rows.append(dict(**record,operation=label,passed=finite and bool(tf.reduce_all(scale<=1).numpy()),
            max_absolute_error=float(tf.reduce_max(err).numpy()) if finite else None,
            max_scaled_error=float(tf.reduce_max(scale).numpy()) if finite else None))
    with tf.device('/GPU:0'):
        for record in source['records']:
            arrays=np.load(record['path'])
            for dtype in (tf.float32,tf.float64):
                c=lambda x:tf.convert_to_tensor(x,dtype)
                for t in range(16):
                    info=dict(variant=record['variant'],update=record['update'],time=t,dtype=dtype.name)
                    for layer in (0,2,4):
                        tag=f't{t}_l{layer}_'
                        inputs=[c(arrays[tag+'x']),c(arrays[f'l{layer}_w']),c(arrays[f'l{layer}_b']),c(arrays[tag+'g'])]
                        outputs=function('dense',inputs,dense)
                        for name,value in zip(('y','gx','gw','gb'),outputs):
                            check('dense'+str(layer)+'/'+name,value,arrays[tag+name],info)
                    shape=arrays[f't{t}_raw'].shape
                    logits=c(arrays[f't{t}_l4_y'].reshape(shape))
                    grad=function('sigmoid',[logits,c(arrays['residual_scale']),c(arrays[f't{t}_raw_g'])],sigmoid)
                    check('sigmoid_residual/vjp',grad,arrays[f't{t}_l4_g'].reshape(shape),info)
                    norm,grad=function('normal',[c(arrays[f't{t}_raw']),c(arrays[f't{t}_norm_g'])],normal)
                    check('normalize/forward',norm,arrays[f't{t}_norm'],info)
                    check('normalize/vjp',grad,arrays[f't{t}_raw_g'],info)
            arrays.close()
            print(json.dumps(dict(variant=record['variant'],update=record['update'],checks=len(rows),
                failed=sum(not r['passed'] for r in rows))),flush=True)
    assert all(g.experimental_get_tracing_count()==1 for g in graphs.values())
    result=dict(schema='kdm_common_adjoint_diagnostic_v1',records=rows,checks=len(rows),
        failures=[r for r in rows if not r['passed']],memory_policy=memory,tensorflow=tf.__version__,
        jit_compile=True,tf32=False,all_graphs_trace_once=True,source_replay_exact=True,
        scope='LOCAL_DERIVATIVES_AT_COMMON_INPUTS_NOT_COMPLETE_TRAINING',
        float64_role='REFERENCE_WITH_SAME_ROUNDED_FLOAT32_INPUTS_AND_ADJOINTS',
        tolerance_role='EXPLANATORY_FP32_SCREEN_FOR_BOTH_ARMS_NOT_REPLICATION_ACCEPTANCE')
    output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();main(args.output,args.reference)
