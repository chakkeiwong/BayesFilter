"""Bounded CPU graph debugging replay; not tuning or GPU parity evidence."""
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
from dataclasses import replace
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root))
signal.alarm(600)
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from scripts.run_surrogate_hmc_five_model_master import restore_selected_scope
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS,VALIDITY_FIELDS

start=time.monotonic()
selection=root/'docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-other-four-advance-01/range_bearing/filter-tuning/range_bearing/filter-selection.json'
case=runtime.configure_prior(runtime.load_case('range_bearing'),'literature-model-priors-v1')
case=restore_selected_scope(runtime,case,selection)
mode=sys.argv[1] if len(sys.argv)>1 else 'uniform'
if mode != 'uniform':
    case=runtime.configure_precision(case,mode,case.tuning_seed_offset)
    # This is a diagnostic arithmetic comparison, not a retuned scope.
    case.fixed=tuple(tf.cast(x,case.force_dtype) for x in case.fixed)
label=mode
if len(sys.argv)>2:
    steps=int(sys.argv[2])
    case=replace(case,controls=replace(case.controls,flow_steps=steps))
    label=mode+'-steps'+str(steps)
archive=json.loads(Path(__file__).with_name('archive-summary.json').read_text())
q=tf.constant([archive['events'][0]['before'],archive['events'][0]['after']],tf.float64)
theta,jac=case.chart(tf.cast(q,case.force_dtype))
program=make_batched_score_filter(case.spec,runtime.N,case.horizon,case.controls,case.force_dtype,
    jit_compile=False,batch_size=2,precision_policy=case.precision_policy)
print('REPLAY_START',label,flush=True)
out=program(theta,case.observations,*case.fixed,case.beta)
rows=[]
for i in range(2):
    completed=int(out['steps_completed'][i]); failed=[]
    for t in range(completed):
        bad=[name for j,name in enumerate(VALIDITY_FIELDS) if not bool(out['validity_trace'][i,t,j])]
        if bad:
            failed.append(dict(observation=t+1,failed=bad,
                trace={name:float(out['trace'][i,t,j]) for j,name in enumerate(TRACE_FIELDS)}))
    pv,ps=case.log_prior(q[i])
    rows.append(dict(q=q[i],theta=theta[i],valid=out['valid'][i],steps_completed=completed,
        value=out['value'][i],score=out['score'][i],prior_value=pv,prior_score=ps,failures=failed,
        max_covariance_error=tf.reduce_max(out['trace'][i,:completed,6])))
result=dict(role='CPU non-XLA diagnostic replay; not GPU parity or promotion',gpu_intentionally_hidden=True,
    jit_compile=False,mode=mode,flow_steps=case.controls.flow_steps,command=[sys.executable,*sys.argv],
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    selection=str(selection.relative_to(root)),case_scope=case.scope(),
    covariance_tolerance=out['covariance_tolerance'],rows=rows,wall_seconds=time.monotonic()-start)
dest=Path(__file__).with_name('replay-'+label+'.json')
runtime.save(dest,result)
print(json.dumps(runtime.json_ready(dict(rows=rows,wall_seconds=result['wall_seconds'])),indent=2),flush=True)
print('ARTIFACT',dest,flush=True)
