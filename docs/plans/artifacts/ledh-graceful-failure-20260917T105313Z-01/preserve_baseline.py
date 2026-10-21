"""CPU-only diagnostic: freeze the current canonical engineering fixture."""
import json
from strict_json import strict_numbers
from pathlib import Path
import runpy
import sys
import traceback
import tensorflow as tf

repo=Path.cwd()
out=Path(sys.argv[1])
fixture=runpy.run_path(str(repo/'tests/highdim/test_ledh_canonical_batch_fused.py'))
inputs=fixture['_fixture'](211,n=16,horizon=2)
initial,covs,noises,observations=inputs
design=fixture['_reset_design'](16)
theta=tf.constant([.6],tf.float64)
controls=fixture['_FULL_UNION']
def host(value):
    if tf.is_tensor(value): return value.numpy().tolist()
    if isinstance(value,(tuple,list)): return [host(x) for x in value]
    if isinstance(value,dict): return {k:host(v) for k,v in value.items()}
    return value

record={'classification':'current full-feature engineering reference; no scientific admission',
        'gpu_intentionally_hidden':True, 'dtype':'float64','seed':211,
        'tensorflow':tf.__version__, 'controls':controls,'substeps':10,
        'inputs':dict(zip(('initial_states','initial_covariances','noises','observations'),host(inputs))),
        'theta':host(theta),'reset_design':host(design), 'checks':{}}
def check(name,fn):
    try: record['checks'][name]={'result':host(fn())}
    except Exception as exc: record['checks'][name]={'exception':type(exc).__name__,'message':str(exc)[:3000]}
    out.write_text(json.dumps(strict_numbers(record),indent=2,allow_nan=False)+'\n')
    print(name,record['checks'][name] if name!='healthy_trace' else 'saved',flush=True)

check('healthy',lambda:fixture['canonical_value_and_analytical_score'](
    fixture['_single_model'](),theta,*inputs,flow_substeps=10,with_score=True,
    reset_design=design,**controls))
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_canonical_score_stages_tf import _sigma_points_with_tangent
check('upper_nan',lambda:safe_cholesky(tf.constant([[1.,float('nan')],[0.,1.]],tf.float64)))
check('infinite_input',lambda:safe_cholesky(tf.constant([[float('inf'),0.],[0.,1.]],tf.float64)))
check('shape_2_1',lambda:safe_cholesky(tf.eye(2,batch_shape=[2,1],dtype=tf.float64))[0].shape.as_list())
check('invalid_sigma',lambda:_sigma_points_with_tangent(
    tf.zeros([1,2],tf.float64),-tf.eye(2,batch_shape=[1],dtype=tf.float64),
    tf.zeros([1,2],tf.float64),tf.eye(2,batch_shape=[1],dtype=tf.float64),2.,1e-12))
check('invalid_actual_batch',lambda:fixture['canonical_batch_fused_value_score'](
    fixture['_fused_model'](),theta[None,:],tf.ones([1,1,1],tf.float64),
    initial,-covs,noises,observations,substeps=10,reset_design=design,**controls))
print('Saved',out,flush=True)
