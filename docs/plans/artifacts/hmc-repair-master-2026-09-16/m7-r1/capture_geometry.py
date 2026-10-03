"""Diagnostic pre/post extraction comparison; geometry only, GPUs hidden by launcher."""
import json, pickle, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO))
from bayesfilter.inference.hmc_kernel_tuning import HMCGeometryInitializationConfig, initialize_hmc_kernel_geometry
class Adapter:
    parameter_dim=2
    def parameter_names(self): return ("a","b")
cases={
    "identity": ({},{}),
    "dense_hessian": ({}, {"negative_hessian":[[4.,1.],[1.,3.]]}),
    "dense_covariance": ({}, {"initial_covariance":[[2.,.3],[.3,.8]]}),
    "scales": ({}, {"parameter_scales":[2.,3.]}),
    "precedence": ({}, {"negative_hessian":[[4.,0.],[0.,9.]],"initial_covariance":[[9.,0.],[0.,9.]],"parameter_scales":[5.,6.]}),
    "regularization": ({"eigenvalue_floor":.25}, {"negative_hessian":[[2.,0.],[0.,-.5]]}),
    "fallback": ({"allow_geometry_fallback":True}, {"negative_hessian":[[1.,float("nan")],[0.,1.]],"parameter_scales":[2.,3.]}),
    "fixed_identity": ({"mass_policy":"fixed_identity"}, {"initial_covariance":[[2.,0.],[0.,3.]]}),
    "bad_shape": ({}, {"initial_covariance":[[1.]]}),
    "bad_scales": ({}, {"parameter_scales":[1.,0.]}),
    "nonfinite_hessian": ({}, {"negative_hessian":[[1.,float("nan")],[0.,1.]]}),
}
result={}
for name,(options,hints) in cases.items():
    cfg=HMCGeometryInitializationConfig(covariance_jitter=0.,seed=(11,22),**options)
    try:
        value=initialize_hmc_kernel_geometry(adapter=Adapter(),initial_position=[.2,-.3],config=cfg,**hints)
        result[name]={"payload":value.payload(include_mass_arrays=True),"artifact_hash":value.artifact_hash}
    except Exception as error:
        result[name]={"exception":type(error).__name__,"message":str(error)}
path=Path(sys.argv[1]); path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
if len(sys.argv)>2:
    assert json.loads(path.read_text())==json.loads(Path(sys.argv[2]).read_text()), "geometry values, metadata, hashes or errors changed"
print("geometry cases:",len(result),"equal_to_baseline:",len(sys.argv)>2)
