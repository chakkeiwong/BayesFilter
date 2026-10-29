"""Independent CPU-only diagnostics; does not modify runtime implementations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['BAYESFILTER_TEST_DEVICE_SCOPE']='cpu'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys, importlib.util, json, math, itertools, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,'/home/chakwong/python/src')
OUT=Path(__file__).resolve().parent

def load(name,rel):
 s=importlib.util.spec_from_file_location(name,ROOT/rel)
 m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m

g=load('audit_generic','docs/benchmarks/run_sqmc_generic_lgssm.py')
h=load('audit_horizon','docs/benchmarks/run_sqmc_horizon_transfer.py')
c=load('audit_ten','docs/benchmarks/run_sqmc_10d_t120_tuned.py')
import bayesfilter.highdim.ledh_canonical_score_tf as core
from bayesfilter.highdim.sqmc_tf import hilbert_integer_keys, randomized_halton_joint, inverse_cdf_ancestor_indices
DT=tf.float64
theta=g._p44_theta(3)
m,set_direction=g._p44_nonlinear_model(theta,3)
parts=g.P44_LGSSM._physical_parts(theta,3)
x=tf.constant([[1.,2.,3.]],DT); dx=tf.zeros_like(x)
set_direction(tf.one_hot(0,4,dtype=DT))
eps=tf.constant(1e-5,DT); v=tf.one_hot(0,4,dtype=DT)
mp,_=g._p44_nonlinear_model(theta+eps*v,3); mm,_=g._p44_nonlinear_model(theta-eps*v,3)
fd=(mp.transition_mean_fn(theta+eps*v,x)-mm.transition_mean_fn(theta-eps*v,x))/(2*eps)
report={'classification':'CPU-only deterministic correctness probes; GPU intentionally hidden','tensorflow_version':tf.__version__,'generic_covariance':{'candidate_Q':tf.linalg.diag_part(m.process_covariance).numpy().tolist(),'reference_Q':tf.linalg.diag_part(parts['transition_covariance']).numpy().tolist(),'candidate_R':tf.linalg.diag_part(m.observation_covariance).numpy().tolist(),'reference_R':tf.linalg.diag_part(parts['observation_covariance']).numpy().tolist(),'reference_raw_initial_mean':parts['raw_initial_mean'].numpy().tolist(),'reference_raw_initial_covariance':tf.linalg.diag_part(parts['raw_initial_covariance']).numpy().tolist(),'candidate_initial_mean':[0.,0.,0.],'candidate_initial_covariance':[1.,1.,1.]},'generic_transition_tangent':{'callback':m.transition_mean_tangent_fn(theta,x,dx).numpy().tolist(),'finite_difference_rebuilt_model':fd.numpy().tolist()}}

# Intercept only the executor to inspect exact consumer-to-core arguments.
original=core.canonical_value_and_analytical_score
captures=[]
def captured(model,theta,initial,covs,noises,observations,**kw):
 captures.append((model,theta,initial,covs,noises,observations,kw))
 return tf.constant(float('-inf'),DT),tf.zeros([1],DT)
core.canonical_value_and_analytical_score=captured
controls={'reset_epsilon':16.,'reset_sinkhorn_steps':8,'reset_balance_steps':8,'correction_steps':4,'correction_strength':.2,'pairwise_steps':4,'pairwise_strength':.03}
try:
 tt=tf.constant([.95,.90,.6,.8],DT); obs=tf.constant([[.2,-.1]],DT)
 a=h._evaluate_sqmc('repaired_permutation',controls,obs,tt,717,2,8,1)
 b=h._evaluate_sqmc('repaired_permutation_ablation',controls,obs,tt,717,2,8,1)
 ca,cb=captures[-2:]
 equal_tensors=all(bool(tf.reduce_all(tf.equal(ca[i],cb[i]))) for i in range(1,6))
 equal_kwargs=all(bool(tf.reduce_all(tf.equal(ca[6][k],cb[6][k]))) if tf.is_tensor(ca[6][k]) else ca[6][k]==cb[6][k] for k in ca[6])
 report['ablation_wiring']={'same_tensor_inputs':equal_tensors,'same_executor_options':equal_kwargs,'base_cap':ca[6]['coordinate_cap'],'ablation_cap':cb[6]['coordinate_cap'],'base_policy':ca[6]['ancestry_policy'],'ablation_policy':cb[6]['ancestry_policy']}
 report['invalid_value_classification']={'horizon_valid':a['valid'],'horizon_value':str(a['value'])}
 c.STATE_DIM=2; c.PARTICLE_COUNT=8; c.HORIZON=1
 z=c._evaluate_route_on_seed('iid_dual_cap',controls,obs,tt,717)
 cm=captures[-1][0]
 report['invalid_value_classification']['ten_dim_adapter_valid']=z['valid']
 report['transfer_transition_tangent']={'callback':cm.transition_mean_tangent_fn(tt,tf.ones([1,2],DT),tf.zeros([1,2],DT)).numpy().tolist(),'correct_phi0_direction':[[1.,0.]],'Q_direction_callback':cm.process_covariance_tangent_fn(tt).numpy().tolist(),'correct_q_scale_direction':[[1.2,0.],[0.,1.2]]}
finally: core.canonical_value_and_analytical_score=original

# Actual same-finite-program directional derivative, fresh model per theta.
initial=tf.random.stateless_normal([12,3],[31,2],dtype=DT)
covs=tf.eye(3,batch_shape=[12],dtype=DT)
noises=tf.random.stateless_normal([1,12,3],[31,3],dtype=DT)
obs=tf.constant([[.2,-.1,.3]],DT)
kwargs=dict(flow_substeps=2,reset_policy='contract_e',reset_design=g._reset_design(12,3),reset_epsilon=32.,reset_sinkhorn_steps=16,reset_balance_steps=16,correction_steps=0,pairwise_steps=0)
def evaluate(th,j=0):
 model,direction=g._p44_nonlinear_model(th,3); direction(tf.one_hot(j,4,dtype=DT))
 val,sc=original(model,th,initial,covs,noises,obs,with_score=True,**kwargs)
 return float(val),float(sc[0])
value,score=evaluate(theta)
rows=[]
for j in range(4):
 direction=tf.one_hot(j,4,dtype=DT)
 plus,_=evaluate(theta+eps*direction,j); minus,_=evaluate(theta-eps*direction,j)
 _,analytical=evaluate(theta,j)
 rows.append({'parameter':j,'analytical':analytical,'central_difference':(plus-minus)/(2*float(eps))})
report['generic_actual_finite_program_derivatives']={'value':value,'directions':rows,'finite':math.isfinite(value),'epsilon':float(eps)}

# Stronger Hilbert property than parity between two wrappers of one implementation.
hilbert=[]
for d,bits in [(2,2),(2,3),(3,2)]:
 coords=list(itertools.product(range(2**bits),repeat=d))
 points=tf.constant([[(x+.5)/(2**bits) for x in row] for row in coords],DT)
 keys=hilbert_integer_keys(points,bits=bits).numpy().tolist()
 ordered=[coords[i] for i in sorted(range(len(coords)),key=lambda i:keys[i])]
 gaps=[sum(abs(a-b) for a,b in zip(left,right)) for left,right in zip(ordered,ordered[1:])]
 hilbert.append({'dimension':d,'bits':bits,'unique_keys':len(set(keys)),'points':len(coords),'max_adjacent_manhattan_distance':max(gaps)})
report['hilbert_grid_checks']=hilbert
raw,u,v=randomized_halton_joint(num_particles=12,state_dimension=3,seed=17,salt=3001,dtype=DT)
indices=inverse_cdf_ancestor_indices(u,tf.fill([12],tf.constant(1/12,DT)))
report['uniform_weights_do_not_imply_permutation']={'indices':indices.numpy().tolist(),'unique_count':len(set(indices.numpy().tolist())),'N':12}
(OUT/'deterministic-probes.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
