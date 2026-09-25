"""CPU-only localization; preserve the failed fit's inputs and diagnostics."""
import json
from pathlib import Path
import sys

artifact_root=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.cwd()))
from bayesfilter.score_study.runtime import configure_runtime
configure_runtime(device="CPU",tf32=False,jit_compile=True)
from bayesfilter.score_study import iapf_fit_tf
from bayesfilter.score_study.adapters import evaluate_gaussian
from bayesfilter.score_study.registry import default_registry

original=iapf_fit_tf.make_density_recursive_fit_kernel
saved={}
def diagnostic_factory(*args,**kwargs):
    kernel=original(*args,**kwargs)
    def diagnostic(theta,observations,clouds):
        result=kernel(theta,observations,clouds)
        saved.update(theta=theta.numpy().tolist(),observations=observations.numpy().tolist(),
                     clouds=clouds.numpy().tolist(),centers=result[0].numpy().tolist(),
                     covariances=result[1].numpy().tolist(),log_floors=result[2].numpy().tolist(),
                     valid=bool(result[3].numpy()),converged=bool(result[4].numpy()),
                     diagnostics=result[5].numpy().tolist(),fitter_arguments=args)
        return result
    return diagnostic
iapf_fit_tf.make_density_recursive_fit_kernel=diagnostic_factory
study=json.loads((artifact_root/'iapf-mechanics-study.json').read_text())
try:
    evaluate_gaussian(study['rows'][0],{'study':study,'registry':default_registry()})
except Exception as exc:
    saved['failure']=str(exc)
with (artifact_root/'iapf-fit-failure-diagnostic.json').open('x') as stream:
    json.dump(saved,stream,indent=2,allow_nan=False)
print(json.dumps({key:saved[key] for key in ('valid','converged','diagnostics','failure')}))
