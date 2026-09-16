"""CPU/XLA diagnostic: current native loop versus frozen substep implementation."""
import importlib.util
import json
import sys
import time
from pathlib import Path
repo=Path(sys.argv[1]);sys.path.insert(0,str(repo))
from bayesfilter.score_study.runtime import configure_runtime
runtime=configure_runtime(device="CPU",tf32=False,jit_compile=True)
import tensorflow as tf
from bayesfilter.highdim import ledh_canonical_score_tf as current
from bayesfilter.score_study.canonical_adapter_tf import gaussian_direction_inputs
from bayesfilter.score_study.nonlinear_tf import direction_inputs
path=repo.parent/"younis-score-ratio-20260915/bayesfilter/highdim/ledh_canonical_score_tf.py"
spec=importlib.util.spec_from_file_location("score_loop_reference",path)
reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
records=[];start=time.monotonic()
for kind,d,o in (("affine",2,1),("nonlinear",1,1)):
 for steps in (1,2,4):
  kernels=[]
  for module in (reference,current):
   def make_kernel(module):
    @tf.function(input_signature=[tf.TensorSpec([6],tf.float64),tf.TensorSpec([6],tf.float64)],jit_compile=True)
    def kernel(theta,direction):
     z=tf.random.stateless_normal([8,d],[271,19],dtype=tf.float64)
     args=gaussian_direction_inputs(theta,direction,z,d,o) if kind=="affine" else direction_inputs(theta,direction,z,.2,.12)
     model,x,dx,P,dP=args[:5]
     cov=tf.broadcast_to(P,[8,d,d]);dcov=tf.broadcast_to(dP,[8,d,d])
     r_inv=tf.linalg.inv(model.observation_covariance)
     dR=model.observation_covariance_tangent_fn(theta)
     return module._flow_substeps_with_tangent(model,x,dx,x,dx,cov,dcov,tf.ones([o],tf.float64)*.3,model.observation_covariance,dR,r_inv,-r_inv@dR@r_inv,substeps=steps,eye=tf.eye(d,dtype=tf.float64))
    return kernel
   kernels.append(make_kernel(module))
  theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],tf.float64);direction=tf.constant([.1,.2,-.3,.4,.5,-.2],tf.float64)
  old,new=(k(theta,direction) for k in kernels)
  errors=[]
  for a,b in zip(old,new):
   tf.debugging.assert_near(a,b,atol=5e-10,rtol=5e-10)
   errors.append(float(tf.reduce_max(tf.abs(a-b)).numpy()))
  graph=kernels[1].get_concrete_function().graph.as_graph_def()
  nodes=list(graph.node)+[node for f in graph.library.function for node in f.node_def]
  loops=sum(n.op in ("While","StatelessWhile") for n in nodes)
  if loops<1:raise AssertionError("native flow loop absent")
  records.append(dict(model=kind,substeps=steps,max_absolute_errors=errors,native_loops=loops,traces=kernels[1].experimental_get_tracing_count()))
result=dict(status="pass",records=records,runtime=runtime,wall_seconds=time.monotonic()-start,reference=str(path),scope="substep parity only; finite-program derivative independently tested in consumer tests")
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":"pass","cases":len(records),"seconds":result["wall_seconds"]}))
