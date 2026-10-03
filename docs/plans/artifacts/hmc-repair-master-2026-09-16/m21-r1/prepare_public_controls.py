"""Build, but do not execute, the M21 public HMC pilot inventory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source-r1'))
from bayesfilter.testing.inference_validation.designs import ScenarioSpec,ValidationDesign
from bayesfilter.testing.inference_validation.execution import plan_suite
from bayesfilter.testing.inference_validation.storage import write_json
PLAN='docs/plans/bayesfilter-hmc-repair-m21-public-controls-2026-09-22.md'
counts={'warmup_min_results':2000,'warmup_check_window_results':1000,'warmup_chunk_results':500,'warmup_max_results':10000,'retained_min_results':1000,'retained_chunk_results':500,'retained_max_results':10000}
models=[('gaussian',{'scale':1.},None,.05),('rotated_gaussian',{'condition':100.,'angle':.6},None,.05),
 ('beta_binomial',{'alpha':2.,'beta':3.,'n':12},[5,12],.005),
 ('lgssm_location',{'tau':2.,'sigma':.5,'state_variance':1.,'rho':.6,'n':6},[1.,-.3,.4,.5,1.2,-.2],.025)]
rows=[]
for name,params,data,tol in models:
 options={'plan_file':PLAN,'native_search':True,'preparation_preset':'standard','metric_evidence_policy':'finite_window',
 'bootstrap_initialization_rounds':20,'metric_probe_num_results':16,'preparation_max_restarts':3,'preparation_bound_expansion_steps':1,
 'member_rule':'declared_l_first','posterior_members':'selected','posterior_settings':counts,
 'fixed_comparator':{'warmup_results':2000,'retained_results':10000}}
 if data is not None:options.update(data=data,data_provenance='fixed synthetic observations declared before execution')
 rows.append(ValidationDesign(design_id='m21-public-pilot-'+name,engine='stopping',scenario=ScenarioSpec(name,'ordinary',parameters=params),
  replications=1,draws=500,seed=2026092214,budget_seconds=600,purpose='complete ordinary public pipeline and exact-reference stopping pilot',
  numerical_provenance=PLAN,device='cpu_reference',posterior_cap=10000,mcse_tolerance=tol,options=options))
suite={'schema':'bayesfilter.inference_validation_suite.v1','suite_id':'m21-public-controls-pilot','profile':'master',
 'profiles':{'master':['stopping']},'designs':[d.payload() for d in rows]}
write_json(ROOT/'public-controls-pilot.json',suite)
write_json(ROOT/'public-controls-pilot-plan.json',plan_suite(suite))
print(len(rows),'public HMC pilot fits; budget',plan_suite(suite)['budget_by_device'])
