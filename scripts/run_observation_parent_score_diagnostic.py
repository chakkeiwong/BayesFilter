"""Bounded, CPU FP64/XLA score diagnostic; never a production admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
START = time.monotonic()
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.testing.observation_parent_score_diagnostic_tf import make_hook

SPEC = NonlinearSQMCSpec('sir_d18')
CONTROLS = ProposalControls(proposal_policy="original_three_branch", flow_steps=16, reset_steps=40, reset_epsilon=1., correction_steps=0)
DT = tf.float64
ZROOT = Path('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-zhao-horizons-20261006-01')
ZDIR = ZROOT/'037-zhao-sir_d18-rank20-fit17-runtime-repair'


def plain(x):
    return x.numpy().tolist()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def finite_plain(x):
    return plain(tf.where(tf.math.is_finite(x), x, tf.zeros_like(x)))


def inputs(n, horizon, seed, observations=None):
    initial = tf.random.stateless_normal([n, 18], [seed, 11], dtype=DT)
    noise = tf.random.stateless_normal([horizon, n, 18], [seed, 12], dtype=DT)
    uniforms = tf.random.stateless_uniform([horizon, n, 2], [seed, 13], dtype=DT)
    obs = tf.constant(observations[:horizon], DT)
    return obs, initial, noise, uniforms, tf.constant([1/3]*3, DT)


def build(n, horizon, modes=None):
    return make_batched_score_filter(SPEC, n, horizon, CONTROLS, dtype=DT,
        jit_compile=True, batch_size=1 if modes is None else len(modes),
        diagnostic_proposal_hook=None if modes is None else make_hook(modes, CONTROLS))


def compact(out):
    valid = plain(out['valid'])
    return dict(valid=valid, steps=plain(out['steps_completed']),
        value=finite_plain(out['value']), score=finite_plain(out['score']),
        score_trace=finite_plain(out['score_trace']),
        ess=finite_plain(out['trace'][:, :, 2]),
        covariance_error=finite_plain(out['trace'][:, :, 6]),
        failed_checks=plain(tf.reduce_all(out['validity_trace'], axis=1)),
        nonfinite_score=plain(tf.reduce_any(~tf.math.is_finite(out['score']), axis=1)),
        invalid_numeric_serialization='nonfinite numeric fields replaced by zero ONLY in diagnostic JSON; valid flag remains authoritative')


def smoke(outdir, observations):
    n, horizon, h = 32, 2, 1e-5
    args = inputs(n, horizon, 281001, observations)
    original = build(n, horizon)(tf.zeros([1, 3], DT), *args)
    augmented = build(n, horizon, [0])(tf.zeros([1, 3], DT), *args)
    parity = {key: float(tf.reduce_max(tf.abs(original[key]-augmented[key])))
              for key in ('value', 'score', 'particles', 'particle_tangent')}
    modes, theta = [], []
    for mode in range(3):
        for offset in [None]+[(k, sign) for k in range(3) for sign in (-1, 1)]:
            q = [0., 0., 0.]
            if offset is not None:
                q[offset[0]] = offset[1]*h
            modes.append(mode); theta.append(q)
    result = build(n, horizon, modes)(tf.constant(theta, DT), *args)
    checks = []
    for mode in range(3):
        base = mode*7
        for k in range(3):
            lo, hi = base+1+2*k, base+2+2*k
            fd = float((result['value'][hi]-result['value'][lo])/(2*h))
            analytical = float(result['score'][base, k])
            same = bool(tf.reduce_all(result['diagnostic_parent_indices'][lo] == result['diagnostic_parent_indices'][hi]))
            error = abs(fd-analytical)/(1+abs(analytical))
            checks.append(dict(mode=mode, parameter=k, step=h, finite_difference=fd, analytical=analytical,
                same_parents=same, scaled_error=error, passed=same and error<2e-3))
    passed = all(bool(x) for x in plain(result['valid'])) and max(parity.values())<1e-8 and all(x['passed'] for x in checks)
    note = dict(passed=passed, parity=parity, finite_difference=checks, rows=compact(result), wall_seconds=time.monotonic()-START)
    save(outdir/'smoke.json', note)
    print(json.dumps(dict(stage='smoke', passed=passed, parity=parity, checks=checks, wall_seconds=note['wall_seconds'])), flush=True)
    return passed


def read_saved_stress():
    cfg = json.loads((ROOT/'docs/plans/artifacts/ledh-failed-case-mechanisms-20261027/interventions-20261027T095847Z/config.json').read_text())
    def load(name):
        info = cfg['inputs'][name]
        raw = (Path(cfg['data'])/info['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != info['sha256']:
            raise ValueError('saved input hash mismatch')
        return tf.cast(tf.io.parse_tensor(raw, tf.as_dtype(info['dtype']))[0], DT)
    return load('theta'), load('observations'), load('initial')-SPEC.initial_mean(DT), load('normal'), tf.stack([load('channel_u'), load('component_u')], -1)


def summary_rows(rows, reference):
    summaries = []
    for mode, name in enumerate(['original', 'conditional_uniform', 'conditional_adapted']):
        valid = [r for r in rows if r['valid'][mode] and not r['nonfinite_score'][mode]]
        values = [[r['score'][mode][k] for r in valid] for k in range(3)]
        means = [statistics.mean(v) for v in values] if valid else None
        se = [statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None for v in values] if valid else None
        rmse = [math.sqrt(statistics.mean((v-reference[k])**2 for v in values[k])) for k in range(3)] if valid else None
        summaries.append(dict(method=name, valid_streams=len(valid), total_streams=len(rows), mean_score=means, mcse=se, rmse_to_approximate_zhao_cui=rmse))
    paired = []
    for mode in (1, 2):
        for k in range(3):
            deltas = [(r['score'][mode][k]-reference[k])**2-(r['score'][0][k]-reference[k])**2
                      for r in rows if r['valid'][mode] and r['valid'][0]]
            mean = statistics.mean(deltas) if deltas else None
            se = statistics.stdev(deltas)/math.sqrt(len(deltas)) if len(deltas)>1 else None
            # Small-n t interval is descriptive; heavy tails and reference bias remain.
            critical = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365}.get(len(deltas))
            interval = [mean-critical*se, mean+critical*se] if critical and se is not None else None
            paired.append(dict(mode=mode, parameter=k, n=len(deltas), mean_squared_distance_change=mean,
                approximate_paired_t95=interval, interpretation='distance to saved approximation, not proof of accuracy ranking'))
    return summaries, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'comparison'], required=True)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--budget-seconds', type=float, default=650)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads((ZDIR/'sir_austria-input-dataset.json').read_text())
    source = json.loads((ZROOT/'038-quadratic-sir_d18-rank20-fit17-T10/result.json').read_text())
    proposal = source['proposals'][0]
    if data['observation_sha256'] != proposal['source_observation_sha256']:
        raise ValueError('Zhao-Cui source observation identity mismatch')
    prepared = json.loads((ZDIR/'target-preparation.json').read_text())
    prepared = prepared[0] if isinstance(prepared, list) else prepared
    assert prepared['theta'] == [0., 0., 0.]
    obs = data['observations']
    actual_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(tf.constant(obs,DT)).numpy())).hexdigest()
    prefix_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(tf.constant(obs[:10],DT)).numpy())).hexdigest()
    if actual_hash != proposal['source_observation_sha256'] or prefix_hash != proposal['observation_sha256']:
        raise ValueError('computed observation tensor hash mismatch')
    files = [Path(__file__), ROOT/'bayesfilter/testing/observation_parent_score_diagnostic_tf.py',
             ROOT/'bayesfilter/highdim/covariance_proposal_score_batch_tf.py']
    manifest = dict(command=sys.argv, git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        executable=sys.executable, tensorflow=tf.__version__, dtype='float64', jit_compile=True,
        cpu_only=True, gpu_status='intentionally hidden before TensorFlow import', threads=4,
        plan='docs/plans/ledh-parent-selection-score-comparison-20261028.md',
        input_file=str(ZDIR/'sir_austria-input-dataset.json'), source_observation_sha256=data['observation_sha256'],
        zhao_cui_oracle_status=source['oracle_status'], theta=prepared['theta'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        controls=vars(CONTROLS), seeds=list(range(281101,281101+args.seeds)), budget_seconds=args.budget_seconds)
    save(args.output/'manifest.json', manifest)
    if args.phase == 'smoke':
        passed = smoke(args.output, obs)
        if not passed:
            raise SystemExit(2)
    else:
        theta, stress_obs, initial, noise, uniforms = read_saved_stress()
        fstress = build(1008, 2, [0, 1, 2])
        st = fstress(tf.broadcast_to(theta[None],[3,3]), stress_obs[:2], initial, noise[:2], uniforms[:2], tf.constant([1/3]*3,DT))
        stress = compact(st)
        stress['theta'] = plain(theta)
        save(args.output/'stress.json', stress)
        print(json.dumps(dict(stage='stress', valid=stress['valid'], score=stress['score'], ess=stress['ess'], wall_seconds=time.monotonic()-START)), flush=True)
        fn = build(1008, 10, [0, 1, 2])
        rows = []
        for seed in manifest['seeds']:
            if time.monotonic()-START > args.budget_seconds:
                break
            t0 = time.monotonic()
            out = fn(tf.zeros([3,3],DT), *inputs(1008,10,seed,obs))
            row = dict(seed=seed, **compact(out), seconds=time.monotonic()-t0)
            rows.append(row)
            save(args.output/'rows.json',rows)
            print(json.dumps(dict(stage='paired', seed=seed, valid=row['valid'], score=row['score'], seconds=row['seconds'], elapsed=time.monotonic()-START)),flush=True)
        estimate = source['estimates'][-1]
        summaries,paired = summary_rows(rows,estimate['score'])
        result = dict(status='diagnostic_complete', reference=estimate['score'],
            reference_conditional_mcse=estimate['conditional_path_jackknife']['standard_error'],
            reference_oracle_status=source['oracle_status'], reference_limit='one fit, conditional path error excludes support/rank/numerical bias',
            summary=summaries,paired=paired,stress=stress, wall_seconds=time.monotonic()-START,
            heuristic_dominance='incomplete: separate pure identity and fully adapted controls required before promotion',
            statistically_supported_accuracy_ranking=False, default_readiness=False)
        save(args.output/'result.json', result)
        print(json.dumps(dict(stage='complete',summary=summaries,wall_seconds=result['wall_seconds'])),flush=True)
    manifest['wall_seconds'] = time.monotonic()-START
    save(args.output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
