"""Master-program matrix families; engineering validity, not posterior evidence."""
import json,os,sys
from strict_json import strict_numbers
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
gpu=sys.argv[1]=='gpu'
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True) if gpu else {'mode':'cpu_only_gpu_intentionally_hidden'}
if gpu:tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky,cholesky_differential
out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])
records=[]
for dtype in (tf.float32,tf.float64):
    epsilon=2.**(-23 if dtype==tf.float32 else -52)
    fixtures=[]
    for scale in (1e-20,1e-10,1.,1e10,1e20):fixtures.append((f'scale-{scale}',tf.eye(2,dtype=dtype)*scale,True))
    rotation=tf.constant([[.8,-.6],[.6,.8]],dtype)
    for power in (0,4,8,12,16):
        matrix=tf.linalg.diag(tf.constant([1.,10.**-power],dtype))
        fixtures.extend([(f'condition-{power}',matrix,None),
                         (f'rotated-{power}',tf.matmul(tf.matmul(rotation,matrix),rotation,transpose_b=True),None)])
    for delta in (.01,32*epsilon,4*epsilon,0.):
        fixtures.append((f'cancellation-{delta}',tf.constant([[1.,1-delta],[1-delta,1.]],dtype),None))
    for name,matrix in [('zero',[[0.,0.],[0.,0.]]),('negative',[[-1.,0.],[0.,1.]]),
                        ('indefinite',[[1.,2.],[2.,1.]]),('asymmetric',[[1.,.1],[0.,1.]])]:
        fixtures.append((name,tf.constant(matrix,dtype),False))
    for number in (float('nan'),float('inf')):
        for index in ((0,1),(1,0)):
            fixtures.append((f'nonfinite-{number}-{index}',tf.tensor_scatter_nd_update(tf.eye(2,dtype=dtype),[index],[tf.cast(number,dtype)]),False))
    matrices=tf.stack([r[1] for r in fixtures])
    tangents=.25*matrices
    def check(a,da):
        valid,chol=safe_cholesky(a)
        tangent=cholesky_differential(chol,da,valid)
        return valid,chol,tangent
    signature=[tf.TensorSpec(matrices.shape,dtype),tf.TensorSpec(matrices.shape,dtype)]
    for mode in ('eager','graph','xla'):
        fn=check if mode=='eager' else tf.function(check,input_signature=signature,jit_compile=mode=='xla')
        valid,chol,dchol=fn(matrices,tangents)
        for i,(name,matrix,expected) in enumerate(fixtures):
            ok=bool(valid[i]);factor=chol[i];derivative=dchol[i]
            assert bool(tf.reduce_all(tf.math.is_finite(factor)))
            assert bool(tf.reduce_all(tf.math.is_finite(derivative)))
            if expected is not None:assert ok==expected,(dtype.name,mode,name,ok)
            represented=tf.cast(matrix,tf.float64)
            finite=bool(tf.reduce_all(tf.math.is_finite(represented)))
            scale=tf.reduce_max(tf.abs(represented)) if finite else tf.constant(1.,tf.float64)
            normalized=represented/tf.where(scale>0,scale,1.)
            residual=None;symmetry=None;eigenvalues=None
            if finite:
                symmetry=float(tf.reduce_max(tf.abs(normalized-tf.transpose(normalized))))
                eigenvalues=tf.linalg.eigvalsh((represented+tf.transpose(represented))/2).numpy().tolist()
                if ok:
                    reconstructed=tf.matmul(tf.cast(factor,tf.float64),tf.cast(factor,tf.float64),transpose_b=True)
                    residual=float(tf.norm((reconstructed-represented)/scale)/tf.maximum(tf.norm(normalized),tf.constant(1e-300,tf.float64)))
                    assert residual<64*epsilon,(dtype.name,mode,name,residual)
            if not ok:tf.debugging.assert_equal(factor,tf.eye(2,dtype=dtype))
            records.append(dict(dtype=dtype.name,mode=mode,family=name,matrix=matrix.numpy().tolist(),
                expected=expected,valid=ok,eigenvalues_of_represented_symmetrization=eigenvalues,
                normalized_symmetry_error=symmetry,relative_factor_residual=residual,
                finite_tangent=bool(tf.reduce_all(tf.math.is_finite(derivative)))))
    healthy=tf.eye(2,dtype=dtype)
    for leading in ((),(1,),(2,1)):
        matrix=tf.broadcast_to(healthy,[*leading,2,2])
        fn=tf.function(check,input_signature=[tf.TensorSpec(matrix.shape,dtype)]*2,jit_compile=True)
        mask,factor,derivative=fn(matrix,tf.zeros_like(matrix))
        assert tuple(mask.shape)==leading
        tf.debugging.assert_equal(derivative,tf.zeros_like(derivative))
        records.append(dict(dtype=dtype.name,mode='xla',shape=list(matrix.shape),valid_shape=list(mask.shape),zero_tangent=True))
    good,bad=tf.eye(2,dtype=dtype),tf.fill([2,2],tf.constant(float('nan'),dtype))
    vv,cc,dd=check(tf.stack([good,good]),tf.stack([good,bad]))
    tf.debugging.assert_equal(vv,[True,True])
    assert bool(tf.reduce_all(tf.math.is_finite(dd[0]))) and not bool(tf.reduce_all(tf.math.is_finite(dd[1])))
    records.append(dict(dtype=dtype.name,independently_invalid_tangent_detected=True,healthy_neighbor_finite=True))
    (out/'matrix-families.json').write_text(json.dumps(strict_numbers(dict(classification='engineering_only',memory_policy=memory,
        records=records,underflow_rule='normalize matrices in float64 before taking reference norms')),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(records=len(records),passed=True)),flush=True)
