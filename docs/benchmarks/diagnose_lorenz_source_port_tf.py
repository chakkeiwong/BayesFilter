"""GPU/XLA localization on the author's captured inputs; no replication claim.

Shared fitting/weight endpoints are compiled. Host loops enumerate independent
diagnostic cases; reporting reductions are a diagnostic eager exception.
No NumPy numerical path is imported.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    source = Path(__file__).with_name('multivariate_controlled_smc_tf.py')
    spec = importlib.util.spec_from_file_location('lorenz_core_snapshot', source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    dtype = tf.float64
    fixture = json.loads((args.reference/'fixture.json').read_text())
    assert fixture['schema'] == 'lorenz_author_capture_v1'
    checks, comparisons, fits, traces = [], [], [], {}
    summary = dict(status='FAIL', scope='AUTHOR_SOURCE_PORT_LOCALIZATION',
        replication='INCOMPLETE', scientific_verdict='NOT_CHECKED',
        tensorflow=tf.__version__, jit_compile=True, tf32=False, dtype='float64',
        gpu_memory_policy=memory, checks=checks, comparisons=comparisons, fits=fits,
        trace_counts=traces, input_sha256=hashlib.sha256((args.reference/'fixture.json').read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        endpoints=['MultivariateGaussianControlledSMC.regress -> _regress',
                   'MultivariateGaussianControlledSMC.cloud_weights -> _cloud_weights',
                   'MultivariateGaussianControlledSMC.fit -> _fit'],
        nonclaims=['complete replication', 'cause of attempt76 failure', 'solver superiority',
                   'common-random-draw forward parity', 'precision repair', 'default readiness'])

    def plain(value):
        if isinstance(value, dict):
            return {k: plain(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def flag(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise RuntimeError(name)

    def scalar(value):
        return float(value.numpy())

    def load(name):
        item = fixture['arrays'][name]
        payload = (args.reference/item['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise RuntimeError('Input hash mismatch: '+name)
        with tf.device('/CPU:0'):
            decoded = tf.reshape(tf.io.decode_raw(payload, dtype, little_endian=True), item['shape'])
        return tf.identity(decoded)

    def compare(name, actual, expected, role='repair_trigger', atol=1e-7, rtol=1e-9):
        flag('shape_'+name, actual.shape == expected.shape)
        finite = bool(tf.reduce_all(tf.math.is_finite(actual)).numpy())
        delta = tf.abs(actual-expected)
        scaled = delta/(atol+rtol*tf.abs(expected)) if atol+rtol else delta
        axes = list(range(1, len(actual.shape)))
        by_time = tf.reduce_max(scaled, axis=axes) if axes else scaled
        times = by_time.numpy().tolist()
        failed = [t for t,v in enumerate(times) if not math.isfinite(v) or v > 1.]
        if atol+rtol == 0:
            failed = [t for t,v in enumerate(times) if not math.isfinite(v) or v != 0.]
        record = dict(name=name, role=role, passed=finite and not failed, finite=finite,
            max_absolute_error=scalar(tf.reduce_max(delta)),
            max_scaled_error=scalar(tf.reduce_max(scaled)),
            first_time_in_backward_order=max(failed) if failed else None,
            max_scaled_error_by_time=times, atol=atol, rtol=rtol)
        comparisons.append(record)
        if role == 'continuation_veto':
            flag(name, record['passed'])
        return record

    def engine(forcing, solver, replicas=1):
        covariance = lambda value: [[value if i == j else 0. for j in range(8)] for i in range(8)]
        return core.MultivariateGaussianControlledSMC(observations=101, particles=512,
            replicas=replicas, dimension=8, observation_dimension=6, initial_mean=[0.]*8,
            initial_covariance=covariance(.01), transition_covariance=covariance(.001),
            transition_mean=lambda x: core.lorenz_rk4(x, tf.constant(forcing, dtype)),
            observation_log_density=lambda y,x: -.5/1e-4*tf.reduce_sum((y-x[...,:6])**2,-1),
            solver=solver, jit_compile=True, precision_policy='none')

    def save(prefix, result):
        for key,value in result.items():
            with tf.device('/CPU:0'):
                tf.io.write_file(str(args.output/(prefix+'_'+key+'.tensor')), tf.io.serialize_tensor(value))

    try:
        flag('six_cases', len(fixture['cases']) == 6)
        flag('sixteen_refinements', sum(c['refinements'] for c in fixture['cases']) == 16)
        with tf.device('/GPU:0'):
            local_engines = {s: engine(4.8801, s, replicas=101)
                             for s in ('source_normal_equations', 'centered_qr')}
            engines = {(f,s): engine(f,s) for f in (2.5,4.8801,8.5)
                       for s in local_engines}
            for case in fixture['cases']:
                case_id, forcing = case['case'], case['forcing']
                flag('source_replay_case%d' % case_id, case['exact_source_replay'])
                y = load('case%d_y' % case_id)[:,None,:]
                for stage in range(case['refinements']+1):
                    prefix = 'case%d_stage%d' % (case_id, stage)
                    values = {k: load(prefix+'_'+k) for k in ('A','b','c','x','lw','logZ','ess')}
                    a,b,c,x,lw = [values[k][:,None] for k in ('A','b','c','x','lw')]
                    weights = engines[forcing,'centered_qr'].cloud_weights(a,b,c,y,x)
                    save(prefix+'_weights', weights)
                    for key, refkey in (('log_incremental_weight','lw'), ('logZ','logZ'), ('ess','ess')):
                        compare(prefix+'_'+key, weights[key][:,0], values[refkey])
                    summary['device'] = weights['logZ'].device
                    flag('GPU_'+prefix, 'GPU:0' in summary['device'])
                    fits.append(dict(case=case_id, stage=stage, forcing=forcing, seed=case['seed'],
                        operation='source_policy_weights', valid=bool(weights['valid'].numpy())))
                    if stage == case['refinements']:
                        continue
                    target, beta = load(prefix+'_Value'), load(prefix+'_BetaHat')
                    reference = dict(A=tf.einsum('hk,kij->hij', beta, local_engines['centered_qr'].basis),
                                     b=beta[:,1:9], c=beta[:,0])
                    edges = local_engines['centered_qr']._design(tf.gather(values['x'], [0,511], axis=1))
                    compare(prefix+'_design_order', edges, load(prefix+'_design_edges'),
                            role='continuation_veto', atol=0., rtol=0.)
                    predicted = core.quadratic(values['x'], reference['A'], reference['b'], reference['c'])
                    for solver, local_engine in local_engines.items():
                        local = local_engine.regress(values['x'], target)
                        save(prefix+'_'+solver+'_local', local)
                        for key in ('A','b','c'):
                            compare(prefix+'_'+solver+'_local_'+key, local[key], reference[key])
                        compare(prefix+'_'+solver+'_local_prediction',
                            core.quadratic(values['x'], local['A'], local['b'], local['c']), predicted)
                        full = engines[forcing,solver].fit(a,b,c,x,lw)
                        save(prefix+'_'+solver+'_full', full)
                        for key in ('A','b','c'):
                            compare(prefix+'_'+solver+'_full_'+key, full[key][:,0],
                                    load('case%d_stage%d_%s' % (case_id,stage+1,key)))
                        fits.append(dict(case=case_id, stage=stage+1, forcing=forcing, seed=case['seed'],
                            operation=solver, local_valid=bool(local['valid'].numpy()),
                            full_valid=bool(full['valid'].numpy()),
                            source_target_rss=scalar(tf.reduce_sum((predicted-target)**2)),
                            local_target_rss=scalar(tf.reduce_sum(local['residual_sum_squares'])),
                            minimum_raw_or_standardized_gram_ratio=scalar(tf.reduce_min(local['gram_eigenvalue_ratio'])),
                            minimum_full_precision_margin=scalar(tf.reduce_min(full['precision_margin']))))
                    (args.output/'progress.json').write_text(json.dumps(plain(dict(
                        last_completed_case=case_id, last_completed_refinement=stage+1,
                        fits=fits, comparisons=len(comparisons))), allow_nan=False)+'\n')
            for solver,eng in local_engines.items():
                traces[solver+'_local'] = eng.regress.experimental_get_tracing_count()
            for (forcing,solver),eng in engines.items():
                traces[str(forcing)+'_'+solver+'_fit'] = eng.fit.experimental_get_tracing_count()
                if solver == 'centered_qr':
                    traces[str(forcing)+'_weights'] = eng.cloud_weights.experimental_get_tracing_count()
        flag('bounded_single_trace_per_used_endpoint', all(n == 1 for n in traces.values()))
        flag('all_fits_compared', len([f for f in fits if f['operation'] != 'source_policy_weights']) == 32)
        mismatches = [v['name'] for v in comparisons if not v['passed']]
        invalid = [v for v in fits if not all(v.get(k,True) for k in ('valid','local_valid','full_valid'))]
        summary.update(status='PASS', scientific_verdict='PORT_MISMATCH_REQUIRES_LOCALIZATION_REVIEW'
            if mismatches or invalid else 'MATCHED_CAPTURE_SCOPE_PASS',
            mismatching_comparisons=mismatches, invalid_routes=invalid,
            gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'))
    except Exception as exc:
        summary['exception'] = type(exc).__name__+': '+str(exc)
    summary['algorithm_seconds'] = time.monotonic()-started
    (args.output/'summary.json').write_text(json.dumps(plain(summary), indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:summary.get(k) for k in ('status','scientific_verdict','exception')}))
    if summary['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
