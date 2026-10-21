"""CPU-only diagnostic of the saved range-bearing warmup; no inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import tensorflow as tf

start = time.monotonic()
root = Path(__file__).resolve().parents[4]
run = root / 'docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-hmc-01/range_bearing/inference/public-tuning'
manifest = json.loads((run / 'manifest.json').read_text())
nom = json.loads((run / 'nomination/manifest.json').read_text())
checksums = {}

def read_tensor(base, record, fallback_dtype='float64'):
    p = base / record['path']
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != record['sha256']:
        raise ValueError(f'Checksum mismatch: {p}')
    checksums[str(p.relative_to(root))] = digest
    return tf.io.parse_tensor(tf.io.read_file(str(p)), tf.as_dtype(record.get('dtype', fallback_dtype)))

def plain(x):
    return x.numpy().tolist()

records = manifest['warmup']
tensors = {k: read_tensor(run, records[k]) for k in
           ('states', 'force_fallback', 'is_accepted', 'delta_h', 'divergence', 'log_accept_ratio')}
center = read_tensor(run / 'nomination', nom['tensors']['center'])
factor = read_tensor(run / 'nomination', nom['tensors']['factor'])
q = center + tf.linalg.matmul(tensors['states'], factor, transpose_b=True)
initial_q = tf.constant(manifest['config']['initial_chain_positions'], tf.float64)
indices = plain(tf.where(tensors['force_fallback']))
events = []
for t, c in indices:
    events.append(dict(transition=t+1, chain=c+1, accepted=bool(tensors['is_accepted'][t,c]),
        before=plain(initial_q[c] if t == 0 else q[t-1,c]), after=plain(q[t,c]),
        delta_h=float(tensors['delta_h'][t,c]),
        log_accept_ratio=float(tensors['log_accept_ratio'][t,c])))
out = dict(schema='range_bearing_warmup_archive_diagnostic_v1',
    role='CPU-only saved-run diagnosis; no inference or tuning',
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    command=[sys.executable,*sys.argv], gpu_intentionally_hidden=True,
    source=str(run.relative_to(root)), status=manifest['sequence']['status'],
    warmup_shape=plain(tf.shape(q)),
    fallback_count=len(indices), fallback_count_by_chain=plain(tf.reduce_sum(tf.cast(tensors['force_fallback'],tf.int32),0)),
    fallback_accepted_count=sum(e['accepted'] for e in events),
    acceptance_by_chain=plain(tf.reduce_mean(tf.cast(tensors['is_accepted'],tf.float64),0)),
    divergences=int(tf.reduce_sum(tf.cast(tensors['divergence'],tf.int32))),
    max_abs_delta_h=float(tf.reduce_max(tf.abs(tensors['delta_h']))),
    q_min=plain(tf.reduce_min(q,[0,1])), q_max=plain(tf.reduce_max(q,[0,1])),
    center=plain(center), factor=plain(factor),
    step_size=plain(read_tensor(run/'nomination',nom['tensors']['step_size'])),
    num_leapfrog_steps=plain(read_tensor(run/'nomination',nom['tensors']['num_leapfrog_steps'],'int32')),
    events=events, checksums=checksums, wall_seconds=time.monotonic()-start)
path = Path(__file__).with_name('archive-summary.json')
path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('checksums','events')},indent=2))
print('FIRST_FALLBACKS', json.dumps(events[:6]))
print('ARTIFACT',path)
