"""CPU-only exact failure localization; unchanged SGQF update, no repair/tuning."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
from pathlib import Path
import hashlib
import json
import time
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import tensorflow as tf
from bayesfilter.highdim import observation_guided_tt_tf as lib

def plain(x):
    if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [plain(v) for v in x]
    if tf.is_tensor(x): return x.numpy().tolist()
    return x

start=time.monotonic()
data_path=ROOT/'docs/benchmarks/artifacts/observation_tt_independent_filtering_20260915/attempt-01/d4-s08/data.json'
data=json.loads(data_path.read_text())
model=lib.SVModel(tf.constant(data['A'],tf.float64),tf.constant(data['P0'],tf.float64),data['beta'],data['sigma'])
original=lib.sgqf_update
records=[]

def observed_update(model,predictive,observation,clouds):
    record={'time':len(records),'observation':observation,'predictive_mean':predictive.mean,
            'predictive_covariance':predictive.factor @ tf.transpose(predictive.factor)}
    try:
        posterior,info=original(model,predictive,observation,clouds)
        record.update(status='COMPLETE',info=info)
        return posterior,info
    except ValueError as error:
        record.update(status='FAILED',error=str(error))
        raise
    finally:
        records.append(plain(record))

lib.sgqf_update=observed_update
try:
    lib.build_guide_path(model,tf.constant(data['observations'],tf.float64))
    status='NO_FAILURE_CPU'
except ValueError:
    status='FAILURE_REPRODUCED_CPU'
finally:
    lib.sgqf_update=original

result={'status':status,'cpu_only':True,'gpu_intentionally_hidden':True,
        'classification':'unchanged_cpu_debug_replay_no_candidate_repair',
        'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest(),
        'records':records,'wall_seconds':time.monotonic()-start}
Path(__file__).with_name('guide-failure-replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':status,'failing_time':records[-1]['time'],'observation':records[-1]['observation'],'wall_seconds':result['wall_seconds']}))
