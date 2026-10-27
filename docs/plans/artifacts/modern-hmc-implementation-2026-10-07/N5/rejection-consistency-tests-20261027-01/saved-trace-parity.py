import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import json, importlib.util
import tensorflow as tf
from bayesfilter.inference.hmc_candidate_set_execution import _trace_from_payload
from bayesfilter.inference.hmc_modern_kernels import ReusableModernHMCRunner
from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
from tests.test_hmc_candidate_set_execution import GaussianTarget
root=Path('docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/source-isolated-20261026-01')
spec=importlib.util.spec_from_file_location('bayesfilter.inference._baseline_modern_kernel',root/'source/BayesFilter/bayesfilter/inference/hmc_modern_kernels.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
q=tf.constant([[-1.,-.4],[-.2,.1],[.4,-.2],[1.,.5]],tf.float64);rows=[]
for family in ('hmc','mams'):
 for trajectory in ('fixed','uniform_duration'):
  for jit in (False,True):
   kw=dict(profile=HMCTransitionProfile(family,trajectory),num_results=8,use_xla=jit,target_status_trace_policy='per_chain_step')
   before=old.ReusableModernHMCRunner(GaussianTarget(),q,**kw);after=ReusableModernHMCRunner(GaussianTarget(),q,**kw)
   args=dict(current_state=q,seed=(20261027,88),step_size=.15,num_leapfrog_steps=7)
   a=before.run(**args);b=after.run(**args)
   assert bool(tf.reduce_all(a.samples==b.samples))
   for key in a.trace:
    for x,y in zip(tf.nest.flatten(a.trace[key]),tf.nest.flatten(b.trace[key])):assert bool(tf.reduce_all(x==y)),key
   rows.append(dict(family=family,trajectory=trajectory,jit=jit,positions_equal=True,original_trace_equal=True))
checkpoint=json.loads((root/'qualification-01/evidence/tuning/tuning_checkpoint.json').read_text())['result'];failures=[]
for row in checkpoint['observations']:
 o=row['observation']
 if not o.get('hard_vetoes'):continue
 path=root/'qualification-01/evidence/tuning/numerical_evidence'/(o['numerical_evidence_hash']+'.json')
 d=json.loads(path.read_text());trial=d['trials'][-1];trace=_trace_from_payload(trial['trace']);hit=trace['first_failure_stage']!=0
 indices=tf.where(hit).numpy().tolist();details=[]
 for i,j in indices:
  details.append({k:bool(trace[k][i,j]) if trace[k].dtype==tf.bool else int(trace[k][i,j]) for k in ('first_failure_stage','first_failure_substep','first_failure_value_finite','first_failure_score_finite','first_failure_momentum_finite','is_accepted')})
 failures.append(dict(L=d['candidate']['leapfrog_steps'],epsilon=d['candidate']['epsilon'],trial_ordinal=trial['ordinal'],failed_transitions=indices,details=details,old_rejection_preserved=True))
out=dict(healthy_path_parity=rows,saved_first_failures=failures,old_artifacts_modified=False,claim='CPU diagnostic parity and saved-trace localization only')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
