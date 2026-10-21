"""Frozen-input revision comparison; diagnostic only, never admission or tuning."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('prepare', 'core', 'wrapper'), required=True)
parser.add_argument('--repo', required=True)
parser.add_argument('--campaign', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--label', required=True)
args = parser.parse_args()
sys.path.insert(0, args.repo)
started = time.monotonic()
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
gpu_run = args.mode != 'prepare'
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=gpu_run)
tf.config.experimental.enable_tensor_float_32_execution(True)
dtype = tf.float32
root = Path(args.out)
campaign = Path(args.campaign)
root.mkdir(parents=True, exist_ok=True)
input_file = root / 'inputs.json'
def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
def host(value):
    return value.numpy().tolist()
if args.mode == 'prepare':
    from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target
    curves = json.loads((campaign / 'lg1-cap-repair-curves-01/curves.json').read_text())
    name = 'cap16-halton-N1024-flow3'
    selected = next(c for c in curves['candidate_list'] if c['name'] == name)
    dataset = json.loads((campaign / 'lg1-data-01/data/LG1-calibration-02.json').read_text())
    target = make_linear_fixture_target(dataset, selected['n'], selected['controls'],
        dtype=dtype, draw_design='scrambled_halton_gaussian')
    original = next(c for c in curves['curves'] if c['name'] == name and c['replicate'] == 2)
    record = next(r for r in original['records'] if r['phi'] == -.9)
    frozen = dict(controls=selected['controls'], phi=-.9, n=selected['n'],
        base=host(target.initial_base), noises=host(target.noises),
        observations=host(target.observations), design=host(target.controls['reset_design']),
        dataset=dataset, manifest=dict(target.manifest), original_record=record)
    write(input_file, frozen)
    write(root / 'prepare.json', dict(cpu_only_gpu_intentionally_hidden=True,
        seconds=time.monotonic()-started, input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest()))
    print('Prepared frozen inputs:', input_file, flush=True)
    sys.exit(0)

frozen = json.loads(input_file.read_text())
n = frozen['n']
base = tf.constant(frozen['base'], dtype)
noises = tf.constant(frozen['noises'], dtype)
observations = tf.constant(frozen['observations'], dtype)
design = tf.constant(frozen['design'], dtype)
settings = dict(frozen['controls'], reset_design=design)
settings['flow_substeps'] = settings.pop('substeps')
if args.mode == 'core':
    from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel, canonical_value_and_analytical_score
    model = NonlinearScoreModel(
        transition_mean_fn=lambda theta, x: theta[0]*x,
        transition_mean_tangent_fn=lambda theta, x, dx: x+theta[0]*dx,
        observation_fn=lambda x: x,
        observation_jacobian_fn=lambda x: tf.ones([tf.shape(x)[0],1,1], x.dtype),
        observation_tangent_fn=lambda x, dx: dx,
        process_covariance=tf.ones([1,1],dtype),
        observation_covariance=tf.constant([[.5]],dtype))
    @tf.function(input_signature=[tf.TensorSpec([1],dtype)], jit_compile=True)
    def evaluate(theta):
        p = 1/(1-tf.square(theta[0]))
        dp = 2*theta[0]*p*p
        states = base*tf.sqrt(p)
        dstates = base*(dp/(2*tf.sqrt(p)))
        covs = tf.ones([n,1,1],dtype)*p
        dcovs = tf.ones_like(covs)*dp
        value, score, trace = canonical_value_and_analytical_score(model,theta,
            states,covs,noises,observations,initial_state_tangent=dstates,
            initial_covariance_tangent=dcovs,with_score=True,return_trace=True,**settings)
        means=[]; covariances=[]; ess=[]; reset_means=[]; reset_covariances=[]
        for step in trace:
            weights=step['posterior_weights']
            x=step['children'][:,0]
            mean=tf.reduce_sum(weights*x)
            means.append(mean)
            covariances.append(tf.reduce_sum(weights*tf.square(x-mean)))
            ess.append(1/tf.reduce_sum(tf.square(weights)))
            reset=step['states_after_reset'][:,0]
            reset_mean=tf.reduce_mean(reset)
            reset_means.append(reset_mean)
            reset_covariances.append(tf.reduce_mean(tf.square(reset-reset_mean)))
        summaries=dict(filtering_mean=tf.stack(means),filtering_covariance=tf.stack(covariances),
            filtering_ess=tf.stack(ess),post_reset_mean=tf.stack(reset_means),
            post_reset_covariance=tf.stack(reset_covariances))
        snapshots={k:tf.stack([step[k] for step in trace]) for k in
            ('children','posterior_weights','states_after_reset','predicted_covariances')}
        return value,score,summaries,snapshots
    value, score, summary, snapshots = evaluate(tf.constant([frozen['phi']],dtype))
    result=dict(log_likelihood=float(value),score=host(score),**{k:host(v) for k,v in summary.items()})
    for key, tensor in snapshots.items():
        (root/f'{args.label}-{key}.tensor').write_bytes(tf.io.serialize_tensor(tensor).numpy())
else:
    from dataclasses import replace
    from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target
    target=make_linear_fixture_target(frozen['dataset'],n,frozen['controls'],
        dtype=dtype,draw_design='scrambled_halton_gaussian')
    # Preparation used CPU whitening, while this process also has a GPU.
    # Record device-reduction differences, then use the frozen inputs exactly.
    regenerated_input_difference={key:float(tf.reduce_max(tf.abs(a-b))) for key,a,b in
        (('base',target.initial_base,base),('noises',target.noises,noises),
         ('observations',target.observations,observations),('design',target.controls['reset_design'],design))}
    target=replace(target,initial_base=base,noises=noises,observations=observations,
        controls=dict(frozen['controls'],reset_design=design))
    value,score,status=target.compile(1,filter_moments=True)(tf.constant([[frozen['phi']]],dtype))
    result=dict(log_likelihood=float(value[0])+math.log(1.9),score=float(score[0,0]),
        value_valid=bool(status['value_valid'][0]),force_valid=bool(status['force_valid'][0]))
    result['regenerated_input_difference']=regenerated_input_difference
    for key in ('filtering_mean','filtering_covariance','filtering_ess','post_reset_mean','post_reset_covariance'):
        result[key]=host(tf.reshape(status[key],[10]))
result.update(mode=args.mode,label=args.label,repo=args.repo,command=sys.argv,
    input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),memory_policy=memory,
    seconds=time.monotonic()-started,tf_version=tf.__version__,dtype='float32',
    tf32=True,jit_compile=True,visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
    allocator=tf.config.experimental.get_memory_info('GPU:0'))
write(root/f'{args.label}.json',result)
print(json.dumps({k:result[k] for k in ('label','log_likelihood','score','seconds','allocator')}),flush=True)
