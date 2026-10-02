"""Fresh deterministic diagnostic of optional LEDH moment safety (not tuning)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--cpu-reference', action='store_true')
args = parser.parse_args()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
if args.cpu_reference:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
import tensorflow as tf

output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
growth = [{'name': g.name, 'growth': tf.config.experimental.get_memory_growth(g)} for g in gpus]
if not args.cpu_reference:
    assert growth and all(item['growth'] for item in growth)
tf.config.experimental.enable_tensor_float_32_execution(not args.cpu_reference)
device = '/CPU:0' if args.cpu_reference else '/GPU:0'
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.highdim.sqmc_campaign_tf import reset_design
from bayesfilter.highdim.moment_safety_tf import SAFETY_KEYS
manifest = dict(command=sys.argv, git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root,text=True).strip(),
    environment=sys.executable, tensorflow=tf.__version__, device=device, gpu_intentionally_hidden=args.cpu_reference,
    memory_policy='memory_growth' if gpus else 'CPU reference; GPU intentionally hidden', devices=growth,
    jit_compile=True, tf32=not args.cpu_reference, seeds=[197,811],
    plan='docs/plans/ledh-moment-safety-repair-20261001.md',
    result='docs/benchmarks/ledh-moment-safety-results-20261001.md',
    data_version='fresh deterministic synthetic fixtures v1', source_sha256={})
files=['bayesfilter/highdim/higher_moment_contract_e.py','bayesfilter/highdim/moment_safety_tf.py',
       'bayesfilter/highdim/ledh_unified_correction_tf.py','bayesfilter/highdim/ledh_canonical_score_tf.py',
       'bayesfilter/highdim/sqmc_campaign_tf.py','docs/benchmarks/run_ledh_moment_safety_20261001.py']
for name in files:
    raw=(root/name).read_bytes()
    manifest['source_sha256'][name]=hashlib.sha256(raw).hexdigest()
    dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
(output/'working-tree.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=root))

def save_manifest(status):
    manifest.update(status=status, wall_seconds=time.monotonic()-started, output=str(output))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

save_manifest('running')
base=dict(correction_steps=3,strength=.12,diagonal_lm_damping=.01,diagonal_lm_scale_floor=1e-4,
    diagonal_trust_radius=.5,pairwise_correction_steps=2,pairwise_strength=.03,pairwise_particle_rms_cap=2.,
    coordinatewise_standardized_cap=.98,coordinatewise_standardized_identity_radius=8.)
arms={
    'protected_no_fit':dict(moment_safety=True,correction_steps=0,pairwise_correction_steps=0),
    'original_cap':dict(coordinatewise_standardized_identity_radius=0.),
    'identity_core':{},
    'small_strength':dict(strength=.03,pairwise_strength=.0075),
    'guarded':dict(moment_safety=True)}
regimes=['healthy','skew','mixture','outlier','concentrated','near_collinear','binary_design']

def fixture(n,d,dtype,regime,seed):
    q=reset_design(n,d,dtype,'normal_quantiles')
    perturb=tf.random.stateless_normal([n,d],[seed,d],dtype=dtype)*tf.cast(.04,dtype)
    source=q+perturb
    if regime=='skew': source=q+.3*q*q+perturb
    if regime=='mixture': source=q+tf.cast(q[:,0:1]>0,dtype)*2.+perturb
    if regime=='outlier': source=tf.tensor_scatter_nd_add(source,[[0]],[tf.ones([d],dtype)*40.])
    if regime=='near_collinear' and d>1:
        source=tf.concat([source[:,:1],source[:,:1]+source[:,1:]*.02],axis=1)
    w=tf.nn.softmax(tf.linspace(tf.cast(-.3,dtype),tf.cast(.3,dtype),n))
    if regime=='concentrated':
        w=tf.nn.softmax(tf.concat([tf.constant([12.],dtype),tf.zeros([n-1],dtype)],0))
    points=reset_design(n,d,dtype,'repeated_axes') if regime=='binary_design' else q
    dx=tf.random.stateless_normal([n,d,1],[seed,3],dtype=dtype)*.01
    dp=tf.random.stateless_normal([n,d,1],[seed,5],dtype=dtype)*.01
    dw=tf.linspace(tf.cast(-1e-5,dtype),tf.cast(1e-5,dtype),n)[:,None]
    return source,w,dx,dw,points,dp

def build(n,d,dtype,overrides):
    controls={**base,**overrides}
    signatures=[tf.TensorSpec(shape,dtype) for shape in ([n,d],[n],[n,d,1],[n,1],[n,d],[n,d,1])]
    @tf.function(input_signature=signatures,jit_compile=True,autograph=False)
    def kernel(x,w,dx,dw,p,dp):
        result=higher_moment_shape_jvp(x,w,dx,dw,p,dp,**controls)
        terms=[];metrics={}
        for family,target in [('skew','skew'),('kurtosis','kurtosis'),
               ('pairwise_co_skew','pairwise_co_skew'),('pairwise_co_kurtosis','pairwise_co_kurtosis')]:
            residual=result[family+'_residual']
            metrics[family+'_error']=tf.linalg.norm(residual)
            terms.append(tf.reduce_sum(tf.square(residual/tf.maximum(tf.cast(1.,dtype),tf.abs(result['target_'+target])))))
        particles=result['particles'];mean=tf.reduce_sum(w[:,None]*x,axis=0)
        centered=x-mean;cov=tf.einsum('n,ni,nj->ij',w,centered,centered)
        om=tf.reduce_mean(particles,axis=0);oc=particles-om
        metrics.update(loss=tf.add_n(terms),valid=result['valid'],
            finite=tf.reduce_all(tf.math.is_finite(particles)) & tf.reduce_all(tf.math.is_finite(result['particles_tangent'])),
            mean_error=tf.reduce_max(tf.abs(om-mean)),covariance_error=tf.reduce_max(tf.abs(tf.transpose(oc)@oc/tf.cast(n,dtype)-cov))/tf.maximum(tf.linalg.norm(cov),tf.cast(1e-20,dtype)),
            maximum_particle=tf.reduce_max(tf.abs(particles)),maximum_tangent=tf.reduce_max(tf.abs(result['particles_tangent'])),
            cap_fraction=result['fraction_coordinatewise_cap_active'])
        metrics.update({key:result[key] for key in SAFETY_KEYS})
        return metrics
    return kernel

rows=[]
try:
    dimensions=[2] if args.cpu_reference else [1,2,3,10]
    dtypes=[tf.float64] if args.cpu_reference else [tf.float64,tf.float32]
    for dtype in dtypes:
        for d in dimensions:
            n=48*d
            kernels={name:build(n,d,dtype,overrides) for name,overrides in arms.items()}
            for regime in regimes:
                for seed in manifest['seeds']:
                    tensors=fixture(n,d,dtype,regime,seed)
                    case=f'{dtype.name}-d{d}-{regime}-s{seed}'
                    inputs_dir=output/'inputs'/case;inputs_dir.mkdir(parents=True)
                    for index,tensor in enumerate(tensors):
                        tf.io.write_file(str(inputs_dir/f'{index}.tensor'),tf.io.serialize_tensor(tensor))
                    for name,kernel in kernels.items():
                        if time.monotonic()-started>1500:
                            raise RuntimeError('bounded campaign wall time exceeded')
                        before=time.monotonic()
                        with tf.device(device): result=kernel(*tensors)
                        record={key:value.numpy().item() for key,value in result.items()}
                        record.update(case=case,dimension=d,dtype=dtype.name,regime=regime,seed=seed,arm=name,
                                      wall_seconds=time.monotonic()-before,device=result['loss'].device)
                        rows.append(record)
                    with (output/'rows.jsonl').open('a') as stream:
                        stream.write(''.join(json.dumps(row)+'\n' for row in rows[-len(arms):]))
                    print(case, 'guarded loss=',rows[-1]['loss'],'baseline=',rows[-5]['loss'],
                          'trials=',rows[-1]['moment_safety_trials'],flush=True)
    comparisons=[]
    for index in range(0,len(rows),len(arms)):
        group={row['arm']:row for row in rows[index:index+len(arms)]}
        guarded=group['guarded'];epsilon=2.**(-23 if guarded['dtype']=='float32' else -52)
        baseline=guarded['moment_safety_baseline_loss'];tol=32*epsilon*(1+abs(baseline))
        losses={arm:row['loss'] for arm,row in group.items()}
        comparisons.append(dict(case=guarded['case'],safety_pass=guarded['valid'] and guarded['finite'] and guarded['moment_safety_final_loss']<=baseline+tol,
            losses=losses,heuristic_dominance=all(guarded['loss']<=group[arm]['loss']+tol for arm in ('protected_no_fit','original_cap','small_strength'))))
    decision=dict(safety_pass=all(row['safety_pass'] for row in comparisons),
        heuristic_dominance=all(row['heuristic_dominance'] for row in comparisons),
        comparisons=comparisons,default_promotion=False,statistically_supported_ranking=False)
    (output/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    save_manifest('complete')
except BaseException as error:
    manifest['error']=repr(error);save_manifest('failed');raise
