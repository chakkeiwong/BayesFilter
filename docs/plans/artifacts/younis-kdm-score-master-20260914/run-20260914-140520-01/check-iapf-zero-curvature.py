"""CPU-only differential reference run against the preceding frozen checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path.cwd()))
from bayesfilter.score_study.runtime import configure_runtime
configure_runtime(device="CPU",tf32=False,jit_compile=True)
import tensorflow as tf
from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel,make_recursive_fit_kernel
from bayesfilter.score_study.iapf_fit_tf import make_density_recursive_fit_kernel
dtype=tf.float64;N=16;T=2
theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype);obs=tf.constant([[.5],[-.3]],dtype)
clouds=tf.random.stateless_normal([T,N,1],[91,1],dtype=dtype)
out={}
for name,fitter in (("quadratic",make_recursive_fit_kernel(1,1,N,T,.01)),
    ("density",make_density_recursive_fit_kernel(1,1,N,T,4.,.2,4.,2000,30,1e-7,.01))):
    fit=fitter(theta,obs,clouds)
    kernel=make_fitted_twist_kernel(1,1,N,T)
    result=kernel(theta,obs,tf.random.stateless_normal([N,1],[91,2],dtype=dtype),
        tf.random.stateless_normal([T,N,1],[91,3],dtype=dtype),
        tf.random.stateless_uniform([T+1,N],[91,4],dtype=dtype),
        tf.random.stateless_uniform([T,N],[91,5],dtype=dtype),*fit[:3])
    out[name]={"fit":[v.numpy().tolist() for v in fit],"result":[v.numpy().tolist() for v in result]}
record={"source":str(Path.cwd()),"revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "command":sys.argv,"cpu_only_gpu_hidden":True,"dtype":"float64","jit_compile":True,
    "sources":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("bayesfilter/score_study").glob("*.py")},"outputs":out}
Path(sys.argv[1]).write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
print("Saved independent zero-curvature reference:",sys.argv[1])
