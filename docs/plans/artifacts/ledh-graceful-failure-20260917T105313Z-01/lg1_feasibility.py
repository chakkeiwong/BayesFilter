"""Mechanics-only LG1 scope/cost smoke, with intentionally synthetic observations."""
import json,os,time
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.ledh_linear_parameter_adapter_tf import linear_gaussian_parameter_model
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score
dtype=tf.float32;n=256;t=10
model=linear_gaussian_parameter_model(tf.ones([1,1],dtype),tf.ones([1,1],dtype),
    tf.ones([1,1],dtype),.5*tf.ones([1,1],dtype),infer_process_scale=False)
states=tf.random.stateless_normal([n,1],[20260917,801],dtype=dtype)
covs=tf.ones([n,1,1],dtype)
noise=tf.random.stateless_normal([t,n,1],[20260917,802],dtype=dtype)
obs=tf.random.stateless_normal([t,1],[20260917,803],dtype=dtype)
design=tf.tile(tf.constant([[1.],[-1.]],dtype),[n//2,1])
controls=dict(substeps=10,reset_policy='contract_e',reset_design=design,
    reset_epsilon=2.,reset_sinkhorn_steps=8,reset_balance_steps=8,reset_ridge=1e-5,
    correction_steps=2,correction_strength=.2,correction_lm_damping=1e-3,
    correction_lm_scale_floor=1e-6,correction_trust_radius=.1,
    pairwise_steps=2,pairwise_strength=.02,pairwise_rms_cap=2.,coordinate_cap=.98,
    coordinate_cap_power=8)
@tf.function(input_signature=[tf.TensorSpec([1,1],dtype)],jit_compile=True)
def fn(theta):
    return canonical_batch_fused_value_score(model,theta,tf.ones([1,1,1],dtype),
        states,covs,noise,obs,**controls)
started=time.monotonic();value,score,status=fn(tf.constant([[.8]],dtype))
baseline=[value.numpy().tolist(),score.numpy().tolist()]
compile_time=time.monotonic()-started
report=dict(classification='mechanics_smoke_only',n=n,horizon=t,memory_policy=memory,
    tf32_enabled=True,jit_compile=True,compile_first_seconds=compile_time,
    value_score=baseline,status={k:v.numpy().tolist() for k,v in status.items()},timings=[])
for _ in range(3):
    started=time.monotonic();v,s,st=fn(tf.constant([[.8]],dtype))
    current=[v.numpy().tolist(),s.numpy().tolist()]
    report['timings'].append(time.monotonic()-started)
    assert current==baseline,'nondeterministic fixed target'
report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
report['passed']=bool(tf.reduce_all(status['value_valid'] & status['force_valid']))
out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])/'feasibility.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
assert report['passed'], 'LG1 full-feature preflight status failure'
