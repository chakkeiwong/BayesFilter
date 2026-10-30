"""Diagnostic TF solver operations with oracle branches, not an autonomous filter.

All numerical work uses TensorFlow. Python dispatches independent saved systems
and writes diagnostic artifacts; no NumPy or callback enters a numerical graph.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','reference'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cpu-reference',action='store_true')
    args=parser.parse_args();start=time.monotonic()
    if args.cpu_reference:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    sys.path.insert(0,str(args.root))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    details=[];memory={'mode':'CPU-only reference; GPU intentionally hidden'};selected_name=None
    if not args.cpu_reference:
        physical=tf.config.list_physical_devices('GPU')
        details=[dict(name=d.name,**tf.config.experimental.get_device_details(d)) for d in physical]
        selected=[d for d,info in zip(physical,details) if '4080 SUPER' in info.get('device_name','')]
        assert len(selected)==1,details
        tf.config.set_visible_devices(selected,'GPU')
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        selected_name=selected[0].name
        assert tf.config.list_logical_devices('GPU')
        rel='bayesfilter/runtime/gpu_memory_policy.py'
        assert sha(args.root/rel)==sha(args.output/'sources'/rel)
    else:assert not tf.config.list_physical_devices('GPU')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    dtype=tf.float64;jit=not args.cpu_reference;device='/CPU:0' if args.cpu_reference else '/GPU:0'
    inputs={};fixture_path=args.reference/'fixture.json'
    reference_manifest=json.loads((args.reference/'manifest.json').read_text())
    assert sha(fixture_path)==reference_manifest['output_sha256']['fixture.json']
    index=json.loads(fixture_path.read_text());assert index['source_coefficients_exact']==229
    inputs[str(fixture_path)]=sha(fixture_path)
    arrays={}
    for name,item in index['arrays'].items():
        path=args.reference/item['file'];assert sha(path)==item['sha256'];inputs[str(path)]=sha(path)
        with tf.device('/CPU:0'):
            arrays[name]=tf.reshape(tf.io.decode_raw(path.read_bytes(),dtype,little_endian=True),item['shape'])
    if jit:
        from tensorflow.compiler.tf2xla.ops import gen_xla_ops
        from tensorflow.compiler.xla import xla_data_pb2
        precision=xla_data_pb2.PrecisionConfig(operand_precision=[xla_data_pb2.PrecisionConfig.HIGHEST]*2).SerializeToString()
    signature=[tf.TensorSpec([45,45],dtype),tf.TensorSpec([45],dtype)]

    @tf.function(input_signature=signature,jit_compile=jit,autograph=False)
    def chol(g,h):
        return tf.linalg.cholesky_solve(tf.linalg.cholesky(g),h[:,None])[:,0]

    @tf.function(input_signature=signature,jit_compile=jit,autograph=False)
    def lu(g,h):
        return tf.linalg.solve(g,h[:,None])[:,0]

    @tf.function(input_signature=signature,jit_compile=jit,autograph=False)
    def svd(g,h):
        if jit:
            s,u,v=gen_xla_ops.xla_svd(g,max_iter=100,epsilon=2.**-52,precision_config=precision)
        else:
            s,u,v=tf.linalg.svd(g,full_matrices=False,compute_uv=True)
        retained=s>tf.cast(2.**-53,dtype)*s[0]
        projected=tf.linalg.matvec(u,h,transpose_a=True)
        scaled=tf.where(retained,tf.math.divide_no_nan(projected,s),tf.zeros_like(s))
        return tf.linalg.matvec(v,scaled),s,tf.reduce_sum(tf.cast(retained,tf.int32))

    @tf.function(input_signature=signature,jit_compile=jit,autograph=False)
    def default_svd(g,h):
        s,u,v=tf.linalg.svd(g,full_matrices=False,compute_uv=True)
        return tf.linalg.matvec(v,tf.math.divide_no_nan(tf.linalg.matvec(u,h,transpose_a=True),s))

    @tf.function(input_signature=[tf.TensorSpec([512,45],dtype),tf.TensorSpec([512],dtype),
        tf.TensorSpec([45],dtype),tf.TensorSpec([45],dtype)],jit_compile=jit,autograph=False)
    def metrics(design,target,actual,reference):
        pred=tf.linalg.matvec(design,actual);ref=tf.linalg.matvec(design,reference)
        centered=design-tf.reduce_mean(design,axis=0,keepdims=True)
        cp=tf.linalg.matvec(centered,actual);cr=tf.linalg.matvec(centered,reference)
        error=tf.abs(pred-ref);ce=tf.abs(cp-cr)
        tolerance=tf.constant(1e-7,dtype)+tf.constant(1e-9,dtype)*tf.abs(ref)
        ct=tf.constant(1e-7,dtype)+tf.constant(1e-9,dtype)*tf.abs(cr)
        return (tf.reduce_max(error),tf.reduce_max(ce),tf.reduce_all(error<=tolerance),
            tf.reduce_all(ce<=ct),tf.sqrt(tf.reduce_mean((pred-target)**2)),
            tf.sqrt(tf.reduce_mean((ref-target)**2)),tf.reduce_max(tf.abs(actual-reference)),
            tf.reduce_all(tf.math.is_finite(actual)))

    def number(x):
        value=float(x.numpy());return value if math.isfinite(value) else None

    def measure(d,y,b,ref):
        m=metrics(d,y,b,ref)
        return dict(prediction_max_error=number(m[0]),centered_prediction_max_error=number(m[1]),
            prediction_screen=bool(m[2].numpy()),centered_prediction_screen=bool(m[3].numpy()),
            residual_rms=number(m[4]),reference_residual_rms=number(m[5]),
            coefficient_max_error=number(m[6]),finite=bool(m[7].numpy()))

    rows=[];controls=[];betas=[];spectra=[];default_control=None
    with tf.device(device):
        # Independently known diagonal and dense SPD solutions expose orientation,
        # rank and reconstruction mistakes before ill-conditioned real systems.
        known=tf.cast(tf.range(1,46),dtype)/45.
        diag=tf.cast(tf.range(1,46),dtype)
        v=tf.math.sin(tf.cast(tf.range(45),dtype))
        dense=tf.linalg.diag(diag)+v[:,None]*v[None,:]
        if jit:
            h=tf.linalg.matvec(dense,known)
            default_error=number(tf.reduce_max(tf.abs(default_svd(dense,h)-known)))
            default_control=dict(max_error=default_error,passed=default_error is not None and default_error<1e-11,role='rejected baseline control; not the tested compatibility candidate')
            (args.output/'default-svd-control-hlo.txt').write_text(default_svd.experimental_get_compiler_ir(dense,h)(stage='hlo'))
        for name,g in (('diagonal_spd',tf.linalg.diag(diag)),('dense_spd',dense)):
            h=tf.linalg.matvec(g,known)
            for solver,call in (('chol',chol),('lu',lu),('svd',lambda a,b:svd(a,b)[0])):
                error=number(tf.reduce_max(tf.abs(call(g,h)-known)))
                assert error is not None and error<1e-11,(name,solver,error)
                controls.append(dict(name=name,solver=solver,max_error=error,passed=True))
        singular=tf.linalg.diag(tf.concat([tf.zeros([1],dtype),diag[1:]],0))
        b,s,rank=svd(singular,tf.linalg.matvec(singular,known))
        expected=tf.concat([tf.zeros([1],dtype),known[1:]],0)
        error=number(tf.reduce_max(tf.abs(b-expected)))
        assert int(rank.numpy())==44 and error is not None and error<1e-11
        controls.append(dict(name='singular_diagonal',solver='svd',rank=44,max_error=error,passed=True))
        (args.output/'controls.json').write_text(json.dumps(dict(controls=controls,default_svd_control=default_control),indent=2)+'\n')
        for i,source in enumerate(index['rows']):
            g=tf.identity(arrays['gram'][i]);h=tf.identity(arrays['rhs'][i]);rank=None;spectrum=None
            if source['branch']=='Positive Definite':b=chol(g,h)
            elif source['branch']=='Full':b=lu(g,h)
            else:
                assert source['branch']=='Singular'
                b,spectrum,retained=svd(g,h);rank=int(retained.numpy())
            result=measure(arrays['design'][i],arrays['target'][i],b,arrays['beta'][i])
            direct=measure(arrays['design'][i],arrays['target'][i],lu(g,h),arrays['beta'][i])
            rank_matches=source['rank'] is None or source['rank']==rank
            passed=result['finite'] and result['prediction_screen'] and result['centered_prediction_screen'] and rank_matches
            row=dict(index=i,label=source['label'],source_branch=source['branch'],source_rank=source['rank'],
                rank=rank,rank_matches=rank_matches,source_chol_rcond=source['chol_rcond'],
                compatibility_passed=passed,conditional=result,unconditional_lu=direct,
                minimum_singular_ratio=number(spectrum[-1]/spectrum[0]) if spectrum is not None else None)
            with (args.output/'rows.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
            rows.append(row);betas.append(b);spectra.append(spectrum if spectrum is not None else tf.zeros([45],dtype))
            if (i+1)%32==0 or i+1==229:
                print(json.dumps(dict(progress='systems_completed',count=i+1,compatible=sum(r['compatibility_passed'] for r in rows))),flush=True)
    with tf.device('/CPU:0'):
        for name,value in (('conditional_beta',tf.stack(betas)),('svd_spectra',tf.stack(spectra))):
            tf.io.write_file(str(args.output/(name+'.tensor')),tf.io.serialize_tensor(value))
    groups={}
    for label,predicate in (('all',lambda r:True),('crossover',lambda r:r['index']<128),
                            ('offset',lambda r:r['index']>=128),
                            ('positive_definite',lambda r:r['source_branch']=='Positive Definite'),
                            ('singular',lambda r:r['source_branch']=='Singular')):
        group=[r for r in rows if predicate(r)]
        groups[label]=dict(count=len(group),compatible=sum(r['compatibility_passed'] for r in group),
            rank_matches=sum(r['rank_matches'] for r in group),finite=sum(r['conditional']['finite'] for r in group),
            prediction_max_error=max((r['conditional']['prediction_max_error'] or 0) for r in group),
            centered_prediction_max_error=max((r['conditional']['centered_prediction_max_error'] or 0) for r in group))
    passed=all(r['compatibility_passed'] for r in rows)
    summary=dict(status='PASS',scope='FORCED_SOURCE_BRANCH_TF_SOLVER_DIAGNOSTIC',replication='INCOMPLETE',
        scientific_verdict='FORCED_BRANCH_COMPATIBLE' if passed else 'FORCED_BRANCH_ARITHMETIC_MISMATCH',
        default_svd_control=default_control,svd_convergence_epsilon=2.**-52 if jit else None,svd_max_sweeps=100 if jit else None,
        compatibility_passed=passed,diagnostic_only=True,branch_oracle=True,controls=controls,rows=rows,groups=groups,
        source_coefficients_exact=229,source_svd_relative_cutoff=2.**-53,
        input_sha256=inputs,tensorflow=tf.__version__,device=device,jit_compile=jit,tf32=False,dtype='float64',
        cpu_only=args.cpu_reference,gpu_intentionally_hidden=args.cpu_reference,
        gpu_memory_policy=memory,gpu_devices=details,selected_physical_device=selected_name,
        trace_counts={n:f.experimental_get_tracing_count() for n,f in (('chol',chol),('lu',lu),('svd',svd),('default_svd',default_svd),('metrics',metrics))},
        allocator=tf.config.experimental.get_memory_info('GPU:0') if not args.cpu_reference else None,
        algorithm_seconds=time.monotonic()-start,statistical_ranking='not tested',default_readiness='not established')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','scientific_verdict','groups','trace_counts','algorithm_seconds')}),flush=True)


if __name__=='__main__':main()
