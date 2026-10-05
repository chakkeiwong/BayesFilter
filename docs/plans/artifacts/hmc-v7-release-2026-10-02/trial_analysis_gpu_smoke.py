"""Four saved trials through the exact concurrent analysis path; no sampling."""
import hashlib, json, os, sys, time
from contextlib import nullcontext
from pathlib import Path

source = Path(sys.argv[1]).resolve(); output = Path(sys.argv[2]).resolve()
output.mkdir(parents=False, exist_ok=False)
sys.path.insert(0, str(source))
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
from bayesfilter.inference import hmc_acceptance_trials as trials
from tests.test_hmc_trial_analysis_workers import assembly_fixture
from dataclasses import replace
runtime, work, chunks = assembly_fixture('healthy', 68)
runtime.config = replace(runtime.config, use_xla=True, trial_analysis_workers=1)
with tf.device('/GPU:0'):
    started=time.monotonic(); serial=trials._assemble_trials(runtime, work, chunks)
    serial_seconds=time.monotonic()-started
    runtime.config=replace(runtime.config, trial_analysis_workers=4)
    started=time.monotonic(); concurrent=trials._assemble_trials(runtime, work, chunks)
    concurrent_seconds=time.monotonic()-started

def normalize(v): return json.loads(json.dumps(v, allow_nan=False))
assert normalize(serial)==normalize(concurrent)
manifest={'schema':'bayesfilter.hmc_trial_analysis_gpu_smoke.v1','source':str(source),
 'source_manifest_sha256':hashlib.sha256((source.parent/'source-manifest.json').read_bytes()).hexdigest(),
 'memory_policy':memory,'tensorflow_version':tf.__version__,'device_scope':'/GPU:0',
 'trial_count':4,'sampling':False,'jit_compile':True}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result={**manifest,'status':'exact_serial_concurrent_gpu_analysis','serial_seconds':serial_seconds,
 'concurrent_seconds':concurrent_seconds,'records_equal':True,'release_ready':False,
 'interpretation':'Four saved diagnostic trials only; no sampling, delivery, tuning or release claim.'}
(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
