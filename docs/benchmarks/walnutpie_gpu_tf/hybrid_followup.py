"""Diagnostic-only transfer/dispatch and CPU-thread sensitivity measurements."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['cpu8','overhead'],required=True)
    p.add_argument('--gpu',default='2');a=p.parse_args()
    cpu=a.mode=='cpu8'
    os.environ.update(CUDA_VISIBLE_DEVICES='-1' if cpu else a.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',BAYESFILTER_PRELOAD_CUSTOM_OP='0',
        TF_NUM_INTRAOP_THREADS='8' if cpu else '4',TF_NUM_INTEROP_THREADS='2')
    import tensorflow as tf
    tf.config.set_soft_device_placement(False)
    tf.config.threading.set_intra_op_parallelism_threads(8 if cpu else 4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    growth=[]
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
        value=tf.config.experimental.get_memory_growth(gpu)
        if value is not True:raise RuntimeError('growth not verified')
        growth.append(dict(name=gpu.name,growth=value))
    if not cpu and not growth:raise RuntimeError('GPU required for overhead measurement')
    from .hybrid_benchmark import data_fixture,make_full,timed,write
    from .core import Config
    source=json.loads((a.root/'result.json').read_text())
    if source['status']!='completed':raise ValueError('main experiment must complete first')
    out=a.root/a.mode;out.mkdir(exist_ok=False)
    source_file=Path(__file__);(out/source_file.name).write_bytes(source_file.read_bytes())
    result=dict(mode=a.mode,command=sys.argv,started_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,
        tensorflow=tf.__version__,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),plan=source['plan'],main_result=str(a.root/'result.json'),
        gpu_intentionally_hidden=cpu,growth=growth,intra_threads=8 if cpu else 4,tf32=False,jit_compile=True,
        role='explanatory diagnostic, not an independent sampler-efficiency ranking')
    begin=time.monotonic()
    if cpu:
        cfg=Config(**source['config']);data=data_fixture(tf,source['selected_model']['observations'],128)
        hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in zip(('X','W','a','y'),data)}
        if hashes!=source['selected_model']['data_sha256']:raise ValueError('different data')
        init,sample,value=make_full(tf,cfg,[tf.TensorSpec(v.shape,v.dtype) for v in data])
        q=tf.fill([4,128],tf.constant(.03,tf.float64));seed=tf.constant([[20261028+i,0] for i in range(4)],tf.int32)
        state=init(q,seed,tf.constant(.2,tf.float64),*data)
        _,first=timed(tf,lambda:sample(state,*data));result['first_call_seconds']=first
        records=[]
        for block in range(7):
            key=20261029+1000*block
            q=.03*tf.random.stateless_normal([4,128],[key,99],dtype=tf.float64,alg='philox')
            seed=tf.constant([[key+i,0] for i in range(4)],tf.int32)
            state=init(q,seed,tf.constant(.2,tf.float64),*data)
            r,t=timed(tf,lambda:sample(state,*data))
            tf.debugging.assert_equal(r.state.valid,tf.ones([4],tf.bool))
            work={name:int(tf.reduce_sum(getattr(r,name)).numpy()) for name in ('evaluations','macros')}
            ref=next(x for x in source['records'] if x['seed']==key)['work']['cpu']
            if any(work[k]!=ref[k] for k in work):raise ValueError('CPU8 changed workload')
            records.append(dict(seed=key,seconds=t,work=work))
        result.update(records=records,median_seconds=statistics.median(r['seconds'] for r in records),
            trace_count=sample.experimental_get_tracing_count(),peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    else:
        # Real transport sizes for this candidate; work-free arithmetic cannot
        # be used to estimate the cost of the actual model/integrator.
        from .hybrid import make_hybrid
        from .hybrid_benchmark import neural_model
        cfg=Config(**source['config'])
        h=make_hybrid(tf,lambda q:neural_model(tf,q),cfg)
        fs=h.signature['send_float'];ins=h.signature['send_int'];fr=h.signature['return_float'];ir=h.signature['return_int']
        with tf.device('/CPU:0'):
            f=tf.ones(fs.shape,fs.dtype);i=tf.ones(ins.shape,ins.dtype)
        @tf.function(input_signature=[fs,ins],jit_compile=True,autograph=False)
        def echo(f,i):return f[:,:fr.shape[-1]]+tf.cast(i[:,0,None],tf.float64)*1e-30,i
        def exchange(compile_echo):
            with tf.device('/GPU:0'):
                gf,gi=tf.identity(f),tf.identity(i)
                if compile_echo:gf,gi=echo(gf,gi)
            with tf.device('/CPU:0'):cf,ci=tf.identity(gf),tf.identity(gi)
            cf.numpy();ci.numpy()
        exchange(False);exchange(True)
        rows=[]
        for block in range(7):
            times={}
            for mode in (('raw','compiled_echo') if block%2==0 else ('compiled_echo','raw')):
                t=time.perf_counter()
                for _ in range(30):exchange(mode=='compiled_echo')
                times[mode]=time.perf_counter()-t
            rows.append(dict(block=block,thirty_round_trips_seconds=times))
        result.update(records=rows,median_thirty_round_trips_seconds={k:statistics.median(r['thirty_round_trips_seconds'][k] for r in rows) for k in ('raw','compiled_echo')},
            true_leaf_bytes=h.bytes_per_round_trip,raw_round_trip_bytes=2*(fs.shape.num_elements()*8+ins.shape.num_elements()*4),
            note='Raw copy returns the request-sized buffer, conservatively larger than the real response. Echo omits model/tree work. Neither is a causal decomposition.',
            trace_count=echo.experimental_get_tracing_count(),allocator=tf.config.experimental.get_memory_info('GPU:0'))
    result.update(status='completed',wall_seconds=time.monotonic()-begin)
    write(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
