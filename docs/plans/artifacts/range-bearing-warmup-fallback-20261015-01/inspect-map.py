"""CPU-only failure localization; no runtime or tuning policy is changed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
os.environ['TF_NUM_INTRAOP_THREADS']='4'
os.environ['TF_NUM_INTEROP_THREADS']='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root))
signal.alarm(600)
from dataclasses import replace
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from scripts.run_surrogate_hmc_five_model_master import restore_selected_scope
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_score_tf import Models,build_maps
from bayesfilter.highdim.analytical_jet_tf import Jet

start=time.monotonic()
selection=root/'docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-other-four-advance-01/range_bearing/filter-tuning/range_bearing/filter-selection.json'
case=restore_selected_scope(runtime,runtime.configure_prior(runtime.load_case('range_bearing'),'literature-model-priors-v1'),selection)
archive=json.loads(Path(__file__).with_name('archive-summary.json').read_text())
q=tf.constant([archive['events'][0]['before'],archive['events'][0]['after']],tf.float64)
theta,_=case.chart(tf.cast(q,tf.float32))
program=make_batched_score_filter(case.spec,runtime.N,13,case.controls,tf.float32,jit_compile=False,batch_size=2)
print('PREFIX_START',flush=True)
out=program(theta,case.observations[:13],case.fixed[0],case.fixed[1][:13],case.fixed[2][:13],case.beta)
assert bool(tf.reduce_all(out['valid']))
rows=[]
for dtype in (tf.float32,tf.float64):
    spec=case.spec if dtype==tf.float32 else runtime.configure_precision(case,'fp64_recursion_fp32_pairs',case.tuning_seed_offset).spec
    th=tf.cast(theta[1],dtype)
    x=Jet(tf.cast(out['particles'][1],dtype),tf.cast(out['particle_tangent'][1],dtype))
    # particle_tangent is [P,N,D] for this selected batch row.
    models=Models(spec,th,tf.eye(2,dtype=dtype))
    m=models.models[0].transition_mean_fn(th,x.v)
    h=models.models[0].observation_jacobian_fn(m)
    Q=tf.cast(models.models[0].process_covariance,dtype)
    R=tf.cast(models.models[0].observation_covariance,dtype)
    L=tf.linalg.cholesky(Q)
    Rbatch=tf.broadcast_to(R,[runtime.N,2,2])
    S=tf.linalg.matmul(tf.linalg.matmul(L,tf.linalg.matmul(h,tf.linalg.solve(Rbatch,h),transpose_a=True),transpose_a=True),L)
    mu=tf.linalg.eigvalsh(S)
    for steps in (8,16,32):
        maps=build_maps(models,x,tf.cast(case.observations[13],dtype),replace(case.controls,flow_steps=steps))
        eig=tf.linalg.eigvalsh(.5*(maps[1].v+tf.linalg.matrix_transpose(maps[1].v)))
        margins=eig[...,0]/tf.reduce_max(tf.abs(eig),-1)
        flat=int(tf.argmin(tf.reshape(margins,[-1])))
        component,particle=divmod(flat,runtime.N)
        singular=tf.linalg.svd(maps[2].v[component,particle],compute_uv=False)
        rows.append(dict(dtype=dtype.name,flow_steps=steps,valid=bool(maps[6]),minimum_margin=float(maps[8]),
            minimum_by_component=tf.reduce_min(margins,-1),worst_component=component,worst_particle=particle,
            covariance_eigenvalues=eig[component,particle],map_singular_values=singular,
            local_information_eigenvalues=mu[particle],first_euler_factors=1-mu[particle]/(2*steps),
            information_max=tf.reduce_max(mu),parameter_theta=th,ancestor=x.v[particle],predicted_mean=m[particle]))
result=dict(role='CPU graph prefix and eager map diagnostic; not tuning or GPU evidence',gpu_intentionally_hidden=True,
    jit_compile=False,command=[sys.executable,*sys.argv],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_scope=case.scope(),prefix_valid=out['valid'],rows=rows,wall_seconds=time.monotonic()-start)
dest=Path(__file__).with_name('map-diagnostic.json')
runtime.save(dest,result)
print(json.dumps(runtime.json_ready(dict(rows=rows,wall_seconds=result['wall_seconds'])),indent=2),flush=True)
print('ARTIFACT',dest,flush=True)
