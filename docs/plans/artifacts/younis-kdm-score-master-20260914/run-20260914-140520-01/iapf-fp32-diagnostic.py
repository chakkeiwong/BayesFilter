"""CPU-only FP32 underflow localization on the saved failed cloud."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path.cwd()))
from bayesfilter.score_study.runtime import configure_runtime
configure_runtime(device="CPU",tf32=False,jit_compile=True)
import tensorflow as tf
from bayesfilter.score_study.iapf_fit_tf import make_density_recursive_fit_kernel
from bayesfilter.score_study.contracts import DiagnosticFailure
root=Path(__file__).resolve().parent
saved=json.loads((root/'iapf-fit-failure-diagnostic.json').read_text())
args=saved['fitter_arguments'];args[-2]='float32'
kernel=make_density_recursive_fit_kernel(*args)
start=time.monotonic()
out=kernel(*(tf.constant(saved[key],tf.float32) for key in ('theta','observations','clouds')))
result={'classification':'CPU-only FP32 debugging; GPU intentionally hidden',
        'source_root':str(Path.cwd()),'valid':bool(out[3]),'converged':bool(out[4]),
        'diagnostics':out[5].numpy().tolist(),'wall_seconds':time.monotonic()-start}
encoded=DiagnosticFailure('diagnostic',result).diagnostics
with Path(sys.argv[1]).open('x') as f: json.dump(encoded,f,indent=2,allow_nan=False)
print(json.dumps(encoded,indent=2))
