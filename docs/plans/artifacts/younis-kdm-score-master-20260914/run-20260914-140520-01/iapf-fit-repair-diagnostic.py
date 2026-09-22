"""CPU-only fixed-input optimizer localization, never proposal-quality evidence."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path.cwd()))
from bayesfilter.score_study.runtime import configure_runtime
runtime=configure_runtime(device="CPU",tf32=False,jit_compile=True)
import tensorflow as tf
from bayesfilter.score_study.iapf_fit_tf import make_density_recursive_fit_kernel
root=Path(__file__).resolve().parent
saved=json.loads((root/'iapf-fit-failure-diagnostic.json').read_text())
kernel=make_density_recursive_fit_kernel(*saved['fitter_arguments'])
start=time.monotonic()
out=kernel(*(tf.constant(saved[key],tf.float64) for key in ('theta','observations','clouds')))
result={'classification':'CPU-only fixed-input debugging; GPU intentionally hidden',
        'source_root':str(Path.cwd()),'valid':bool(out[3]),'converged':bool(out[4]),
        'centers':out[0].numpy().tolist(),'covariances':out[1].numpy().tolist(),
        'diagnostics':out[5].numpy().tolist(),'wall_seconds':time.monotonic()-start}
with (root/'iapf-fit-repair-diagnostic.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
