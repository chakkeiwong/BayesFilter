"""CPU FP64/XLA diagnostic-only pure proposal controls; no admission."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import run_observation_parent_score_diagnostic as base
tf, DT = base.tf, base.DT


def has_row_budget(elapsed, budget, completed_durations):
    """Reserve time for the next complete paired row and final serialization."""
    reserve = 1.25*max(completed_durations)+5 if completed_durations else 5
    return elapsed+reserve < budget


def checkpoint_results(output, rows, reference_score, manifest, start, status):
    """Keep a terminally readable summary after every completed paired row."""
    summaries, paired = base.summary_rows(rows, reference_score)
    for summary, label in zip(summaries, ['identity_only', 'conditional_uniform_only', 'conditional_adapted_only']):
        summary['method'] = label
    elapsed = time.monotonic()-start
    for name, payload in [
        ('rows.json', rows),
        ('manifest.json', dict(manifest, wall_seconds=elapsed, completed_seeds=[r['seed'] for r in rows])),
        ('result.json', dict(summary=summaries, paired=paired, wall_seconds=elapsed,
            status=status, completed_streams=len(rows), requested_streams=len(manifest['seeds']),
            reference_oracle_status='not_certified', default_readiness=False))]:
        temporary = output/(name+'.tmp')
        base.save(temporary, payload)
        temporary.replace(output/name)
    return summaries, elapsed


def conditional_only_hook():
    mixed = base.make_hook([1, 2], base.CONTROLS)
    # Draw from the conditional branch and override only the proposal density.
    # Density slot zero retains the transition-mixture target numerator.
    def analytical_hook(models, maps, beta, uniforms, innovations, observation):
        child, densities, valid, selected = mixed(models, maps, tf.constant([0., 1., 0.], DT),
                                                  uniforms, innovations, observation)
        return child, densities, valid, selected, densities[:, :, 1]
    return analytical_hook


def conditional_filter(n, horizon):
    return base.make_batched_score_filter(base.SPEC,n,horizon,base.CONTROLS,DT,True,2,
                                         diagnostic_proposal_hook=conditional_only_hook())


def control_smoke(observations):
    fn = conditional_filter(32, 2)
    inputs = list(base.inputs(32,2,281001,observations)); inputs[-1]=tf.constant([1.,0.,0.],DT)
    center = fn(tf.zeros([2,3],DT),*inputs)
    # Fully adapted proposal has constant weights for each fixed parent cloud.
    # This catches accidentally changing the numerator along with q.
    tf.debugging.assert_near(center['trace'][1,:,2],tf.constant([32.,32.],DT),atol=1e-8,rtol=1e-10)
    checks=[]
    for k in range(3):
        delta=tf.broadcast_to(tf.one_hot(k,3,dtype=DT)[None]*1e-5,[2,3])
        lo,hi=fn(-delta,*inputs),fn(delta,*inputs)
        fd=(hi['value']-lo['value'])/2e-5
        error=tf.abs(fd-center['score'][:,k])/(1+tf.abs(center['score'][:,k]))
        same=tf.reduce_all(lo['diagnostic_parent_indices']==hi['diagnostic_parent_indices'],[1,2])
        valid=center['valid'] & lo['valid'] & hi['valid']
        checks.append(dict(parameter=k,scaled_error=base.plain(error),same_parents=base.plain(same),valid=base.plain(valid)))
        tf.debugging.assert_equal(same & valid,[True,True])
        tf.debugging.assert_less(error,tf.constant([2e-3,2e-3],DT))
    return dict(passed=True,checks=checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--budget-seconds', type=float, default=400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    obs = json.loads((base.ZDIR/'sir_austria-input-dataset.json').read_text())['observations']
    reference = json.loads((base.ZROOT/'038-quadratic-sir_d18-rank20-fit17-T10/result.json').read_text())
    prefix_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(tf.constant(obs[:10],DT)).numpy())).hexdigest()
    assert prefix_hash == reference['proposals'][0]['observation_sha256']
    sources = [Path(__file__), Path(base.__file__), base.ROOT/'bayesfilter/testing/observation_parent_score_diagnostic_tf.py',
               base.ROOT/'bayesfilter/highdim/covariance_proposal_score_batch_tf.py']
    manifest = dict(command=sys.argv, git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=sys.executable, tensorflow=tf.__version__, cpu_only=True, gpu_status='intentionally hidden before import',
        dtype='float64', jit_compile=True, threads=4, n=1008, horizon=10, theta=[0.,0.,0.],
        observation_prefix_sha256=prefix_hash, seeds=list(range(281101,281101+args.seeds)),
        plan='docs/plans/ledh-parent-selection-score-comparison-20261028.md',
        source_sha256={str(p.relative_to(base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        controls=vars(base.CONTROLS), budget_seconds=args.budget_seconds)
    base.save(args.output/'manifest.json',manifest)
    base.save(args.output/'smoke.json',control_smoke(obs))
    theta, stress_obs, initial, noise, uniforms = base.read_saved_stress()
    stress_inputs=[stress_obs[:2],initial,noise[:2],uniforms[:2],tf.constant([1.,0.,0.],DT)]
    stress_identity=base.compact(base.build(1008,2)(theta[None],*stress_inputs))
    stress_conditional=base.compact(conditional_filter(1008,2)(tf.broadcast_to(theta[None],[2,3]),*stress_inputs))
    base.save(args.output/'stress.json',dict(identity=stress_identity,conditional=stress_conditional))
    identity = base.build(1008,10)
    conditional = conditional_filter(1008,10)
    rows, completed_durations = [], []
    for seed in manifest['seeds']:
        if not has_row_budget(time.monotonic()-start, args.budget_seconds, completed_durations):
            break
        row_start = time.monotonic()
        inputs = list(base.inputs(1008,10,seed,obs)); inputs[-1] = tf.constant([1.,0.,0.],DT)
        first = base.compact(identity(tf.zeros([1,3],DT),*inputs))
        rest = base.compact(conditional(tf.zeros([2,3],DT),*inputs))
        row = dict(seed=seed)
        for key in ('valid','steps','value','score','score_trace','ess','covariance_error','failed_checks','nonfinite_score'):
            row[key] = first[key]+rest[key]
        rows.append(row)
        completed_durations.append(time.monotonic()-row_start)
        checkpoint_results(args.output, rows, reference['estimates'][-1]['score'],
                           manifest, start, 'in_progress')
        print(json.dumps(dict(seed=seed,valid=row['valid'],score=row['score'],elapsed=time.monotonic()-start)),flush=True)
    summaries, elapsed = checkpoint_results(args.output, rows, reference['estimates'][-1]['score'],
        manifest, start, 'diagnostic_complete' if len(rows)==args.seeds else 'budget_stopped')
    print(json.dumps(dict(stage='complete',summary=summaries,wall_seconds=elapsed)),flush=True)


if __name__ == '__main__':
    main()
