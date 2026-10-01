"""Tiny GPU/XLA replay mechanics smoke; no learned-map quality claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH','').lower()!='true':
    raise RuntimeError('memory growth must precede TensorFlow import')
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
args.output.mkdir(parents=True,exist_ok=False)
start=time.monotonic()
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
from bayesfilter.inference.neutra_transport import NeuTraTransport,NeuTraTransportConfig
from bayesfilter.inference.neutra_fab import FABConfig,FABTrainer
flow=NeuTraTransport(NeuTraTransportConfig.hoffman_author_iaf(2,conditional_scale_cap=2.,seed=(12,31),dtype='float32'))
config=FABConfig(16,3,3,1,.2,.001,.9,.999,1e-8,128,64,2,10.,None,False,.65,1.02,
    transition_operator='metropolis')
def target(x):
    return -.5*tf.reduce_sum(x*x,-1)-math.log(2*math.pi),-x,tf.reduce_all(tf.math.is_finite(x),-1)
trainer=FABTrainer(flow,target,config,target_signature='a'*64,seed=(8,9))
rows=[]
for i in range(6):
    r=trainer.step()
    assert bool(r['valid'])
    rows.append({'pass':trainer.pass_index,'updates':len(r['updates'])})
assert int(trainer.optimizer.iterations)==2
indices=trainer.replay_sample(trainer.replay,tf.constant([4,8]))
assert int(tf.size(tf.unique(indices).y))==32
payload=json.loads(json.dumps(trainer.checkpoint(),allow_nan=False))
second=FABTrainer(NeuTraTransport(flow.config),target,config,target_signature='a'*64,seed=(8,9))
second.restore(payload)
assert second.checkpoint()['checkpoint_hash']==payload['checkpoint_hash']
result={'status':'passed','purpose':'GPU_XLA_replay_mechanics_only','command':sys.argv,
    'environment':sys.executable,'memory_policy':memory,'device':flow.trainable_variables[0].device,
    'config':asdict(config),'seed':[8,9],'target':'2d_normal_reference','rows':rows,
    'optimizer_updates':int(trainer.optimizer.iterations),'distinct_indices':32,
    'checkpoint_roundtrip':True,'wall_seconds':time.monotonic()-start,
    'source_sha256':hashlib.sha256((ROOT/'bayesfilter/inference/neutra_fab.py').read_bytes()).hexdigest(),
    'plan':'docs/plans/bayesfilter-fab-iaf-plan-2026-09-25.md'}
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
