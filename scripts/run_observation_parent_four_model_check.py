"""Diagnostic only: one saved configuration per model, no tuning or promotion."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / 'docs/plans/observation-parent-four-model-check-20261028.md'
OLD = ROOT / 'docs/plans/artifacts/ledh-independent-validation-20261008-01'
RB = ROOT / 'docs/plans/artifacts/ledh-range-bearing-covariance-20261009-07'
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

def load(path):
    return json.loads(Path(path).read_text())

def clean(x):
    if hasattr(x, 'numpy'):
        return clean(x.numpy().tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x

def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def preparation(model, tf):
    source_files = []
    if model == 'range_bearing':
        from run_range_bearing_covariance_comparison import specs, checked_reference
        fixture, spec = specs(tf, tf.float64, 20)
        path = RB / 'data/1704-t20.json'
        data = load(path)
        tune = load(RB / 'tuning.json')['selected']['new']
        source_files += [path, RB / 'tuning.json']
        reference_files = [RB / f'references/1704-0-{n}.json' for n in (16384, 65536)]
        refs = [checked_reference(p) for p in reference_files]
        for r in refs:
            assert r['valid'] and r['data_seed'] == 1704 and r['point'] == 0 and r['replicates'] == 16
            assert all(math.isfinite(v) for v in [r['value'], *r['score'], *r['mcse']])
        reference = dict(kind='finite_particle_Fisher_reference_not_exact', value=refs[-1]['value'],
                         score=refs[-1]['score'], mcse=refs[-1]['mcse'], refinements=refs)
        source_files += reference_files
        theta, names = [0., 0.], ['log_process_scale', 'log_observation_scale']
        target = 'M13_wrapped_range_bearing'
        controls, beta = tune['controls'], tune['beta']
    else:
        entries = {'lgssm': '001-filter-lgssm-d26100831', 'ksc': '004-filter-ksc-d26100831',
                   'predator_prey': '005-filter-predator_prey-d26100831'}
        path = OLD / entries[model] / 'manifest.json'
        saved = load(path)
        data, theta, target = saved['data'], saved['theta'], saved['target_id']
        controls, beta = saved['tuning']['scope']['controls'], saved['tuning']['beta']
        source_files.append(path)
        fixture = None
        if model == 'lgssm':
            from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
            spec = LGSSMSpec('p44', 3)
            names = ['transition_parameter', 'log_process_variance', 'log_observation_variance', 'initial_mean_scale']
            with tf.device('/CPU:0'):
                value, score = spec.reference_value_and_score(tf.constant(theta, tf.float64),
                                                              tf.constant(data['observations'], tf.float64))
            reference = dict(kind='exact_Kalman', value=float(value), score=score.numpy().tolist())
        elif model == 'ksc':
            from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec, checked_reference
            spec = KSCSpec()
            names = ['persistence_parameter', 'log_observation_scale']
            reference = dict(checked_reference(tuple(theta), tuple(r[0] for r in data['observations'])))
            reference['kind'] = 'independently_refined_grid'
        else:
            from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
            spec = NonlinearSQMCSpec('predator_prey')
            names = ['r', 'carrying_capacity', 'half_saturation', 's', 'u', 'v']
            fits = []
            for fit, folder in [(41, '007-score-predator_prey-d26100831-fit41'),
                                (53, '013-score-predator_prey-d26100831-fit53')]:
                rp = OLD / folder / 'result.json'
                rr = load(rp)
                rm = rp.with_name('manifest.json')
                reference_manifest = load(rm)
                assert reference_manifest['theta0'] == theta
                assert reference_manifest['parameter_names'] == names
                parent_folder = ('002-zhao-predator_prey-d26100831-fit41' if fit == 41
                                 else '009-zhao-predator_prey-d26100831-fit53')
                parent_data_path = OLD / parent_folder / 'pp-input-dataset.json'
                parent_data = load(parent_data_path)
                assert parent_data['observations'] == data['observations']
                assert parent_data['target_id'] == target
                source_files += [rm, parent_data_path]
                proposal = rr['proposals'][0]
                assert proposal['observation_sha256'] == data['observation_sha256']
                assert proposal['source_observation_sha256'] == data['observation_sha256']
                assert proposal['horizon'] == len(data['observations'])
                assert proposal['joint_density_parity_max'] < 1e-8
                estimate = min(rr['estimates'], key=lambda x: x['radius'])
                fits.append(dict(fit_seed=fit, value=proposal['log_likelihood'], score=estimate['score'],
                                 value_conditional_se=proposal['conditional_log_likelihood_jackknife_se'],
                                 score_conditional_se=estimate['conditional_path_jackknife']['standard_error'],
                                 radii=[dict(radius=e['radius'], score=e['score']) for e in rr['estimates']],
                                 source=rp.relative_to(ROOT).as_posix()))
                source_files.append(rp)
            reference = dict(kind='adapted_Zhao_Cui_not_certified', fits=fits,
                             value=statistics.mean(f['value'] for f in fits),
                             score=[statistics.mean(f['score'][j] for f in fits) for j in range(6)],
                             uncertainty='conditional SE excludes fit variation, support error, and radius bias')
        assert spec.target_id == target
    assert len(reference['score']) == len(theta)
    assert all(math.isfinite(v) for v in [reference['value'], *reference['score']])
    return spec, fixture, data, theta, names, target, controls, beta, reference, source_files

def designs(tf, spec, fixture, n, t, seed):
    if fixture is not None:
        from run_range_bearing_covariance_comparison import inputs
        base, noise, uniform, _ = inputs(tf, fixture, n, seed, tf.float64)
        return base, noise[:t], uniform[:t]
    with tf.device('/CPU:0'):
        return (tf.random.stateless_normal([n, spec.dimension], [seed, 11], dtype=tf.float64),
                tf.random.stateless_normal([t, n, spec.dimension], [seed, 12], dtype=tf.float64),
                tf.random.stateless_uniform([t, n, 2], [seed, 13], dtype=tf.float64))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lgssm', 'ksc', 'predator_prey', 'range_bearing'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(args.output / 'status.json', dict(status='preparing', command=[sys.executable, *sys.argv]))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, TRACE_FIELDS, VALIDITY_FIELDS
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
    spec, fixture, data, theta, names, target, settings, beta, reference, sources = preparation(args.model, tf)
    n, t, seeds = 1008, len(data['observations']), list(range(281301, 281309))
    controls = ProposalControls(**settings, proposal_policy='original_three_branch')
    supported = args.model in ('lgssm', 'predator_prey')
    policies = ['original_three_branch'] + (['conditional_uniform', 'conditional_observation_weighted'] if supported else [])
    manifest = dict(model=args.model, target_id=target, theta=theta, parameter_names=names, particles=n, horizon=t,
                    seeds=seeds, controls=dataclasses.asdict(controls), beta=beta, policies=policies,
                    conditional_supported=supported, unsupported_reason=None if supported else 'conditional implementation requires affine Gaussian observations',
                    data=data, reference=reference, source_files={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    command=[sys.executable, *sys.argv], environment=sys.executable, tensorflow=tf.__version__,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    cpu_only=False, device='GPU', dtype='float64', jit_compile=True, tf32=False,
                    precision_exception='diagnostic FP64 reference comparison; not default-readiness evidence',
                    memory_policy=memory, plan=str(PLAN.relative_to(ROOT)), result_file=str(args.output / 'result.json'),
                    inference_status='descriptive_only_no_ranking_or_promotion', input_design='paired stateless draws; RB inputs FP32-rounded per saved harness',
                    smoke='N=16,T=2; original route; no research claim')
    save(args.output / 'manifest.json', manifest)
    q, obs = tf.constant([theta], tf.float64), tf.constant(data['observations'], tf.float64)
    b = tf.constant(beta, tf.float64)
    with tf.device('/GPU:0'):
        smoke = make_batched_score_filter(spec, 16, 2, dataclasses.replace(controls, flow_steps=2, reset_steps=2), batch_size=1)
        initial, noise, uniform = designs(tf, spec, fixture, 16, 2, seeds[0])
        out = smoke(q, obs[:2], initial, noise, uniform, b)
        save(args.output / 'smoke.json', {k: out[k] for k in ['value', 'score', 'valid', 'steps_completed']})
        assert bool(out['valid'][0]) and int(out['steps_completed'][0]) == 2, 'smoke invalid; inspect artifact'
    rows = []
    for policy in policies:
        with tf.device('/GPU:0'):
            program = make_batched_score_filter(spec, n, t, dataclasses.replace(controls, proposal_policy=policy), batch_size=1)
        arms = [(policy, b)]
        if policy == 'original_three_branch':
            arms.append(('transition_only', tf.constant([1., 0., 0.], tf.float64)))
        for arm, weights in arms:
            for seed in seeds:
                initial, noise, uniform = designs(tf, spec, fixture, n, t, seed)
                tick = time.monotonic()
                with tf.device('/GPU:0'):
                    out = program(q, obs, initial, noise, uniform, weights)
                value, score = float(out['value'][0]), out['score'][0].numpy().tolist()
                finite = all(math.isfinite(v) for v in [value, *score])
                valid = bool(out['valid'][0]) and int(out['steps_completed'][0]) == t and finite
                trace = out['trace'][0].numpy().tolist()
                row = dict(arm=arm, seed=seed, value=value, score=score, finite=finite, valid=valid,
                           steps_completed=int(out['steps_completed'][0]), wall_seconds=time.monotonic()-tick,
                           value_error=value-reference['value'], score_error=[v-r for v,r in zip(score,reference['score'])],
                           ess_mean=statistics.mean(r[2] for r in trace), ess_min=min(r[2] for r in trace),
                           trace=trace, trace_fields=TRACE_FIELDS, validity_trace=out['validity_trace'][0], validity_fields=VALIDITY_FIELDS)
                rows.append(clean(row))
                save(args.output / 'rows.json', rows)
                save(args.output / 'status.json', dict(status='running', completed=len(rows), latest_arm=arm, latest_seed=seed, wall_seconds=time.monotonic()-started))
                print(json.dumps({k: clean(row[k]) for k in ['arm','seed','value','score','valid','ess_mean','wall_seconds']}), flush=True)
    summary = {}
    for arm in dict.fromkeys(r['arm'] for r in rows):
        group = [r for r in rows if r['arm'] == arm]
        good = [r for r in group if r['valid']]
        entry = dict(attempted=len(group), valid=len(good), hard_veto='PASS' if len(good)==len(group) else 'INVALID_ROWS',
                     ranking='not statistically established')
        if good:
            entry.update(value_mean=statistics.mean(r['value'] for r in good),
                         value_bias=statistics.mean(r['value_error'] for r in good),
                         value_rmse=math.sqrt(statistics.mean(r['value_error']**2 for r in good)),
                         score_mean=[statistics.mean(r['score'][j] for r in good) for j in range(len(theta))],
                         score_bias=[statistics.mean(r['score_error'][j] for r in good) for j in range(len(theta))],
                         score_rmse=[math.sqrt(statistics.mean(r['score_error'][j]**2 for r in good)) for j in range(len(theta))],
                         ess_mean=statistics.mean(r['ess_mean'] for r in good),
                         ess_min=min(r['ess_min'] for r in good),
                         value_sd=statistics.stdev(r['value'] for r in good) if len(good)>1 else None,
                         score_sd=[statistics.stdev(r['score'][j] for r in good) for j in range(len(theta))] if len(good)>1 else None)
        summary[arm] = entry
    hashes = {}
    for module in list(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).is_file() and Path(p).resolve().is_relative_to(ROOT):
            hashes[str(Path(p).resolve().relative_to(ROOT))] = sha(p)
    manifest.update(loaded_source_sha256=hashes, wall_seconds=time.monotonic()-started,
                    allocator=tf.config.experimental.get_memory_info('GPU:0'))
    save(args.output / 'manifest.json', manifest)
    save(args.output / 'result.json', dict(model=args.model, reference=reference, summary=summary,
         interpretation='descriptive only; errors for valid full-horizon runs; no canonical admission or default promotion',
         wall_seconds=time.monotonic()-started))
    save(args.output / 'status.json', dict(status='complete', completed=len(rows), wall_seconds=time.monotonic()-started))

if __name__ == '__main__':
    main()
