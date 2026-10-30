"""Independent-reference diagnostic for native repetition batching, not a claim run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
ROOT = Path('/home/chakwong/BayesFilter')
sys.path.insert(0, str(ROOT))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--observations', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    candidate = load('candidate_core', Path(__file__).with_name('controlled_smc_tf.py'))
    baseline = load('preserved_scalar_core', args.baseline)
    data = json.loads(Path(args.observations).read_text())
    checks, timings = [], []

    def compare(name, got, want, exact=False):
        if exact:
            passed = bool(tf.reduce_all(tf.equal(got, want)).numpy())
            error, units = 0. if passed else 1., 0. if passed else float('inf')
        else:
            error_tensor = tf.abs(got-want)
            scale = 1e-7+1e-7*tf.abs(want)
            error = float(tf.reduce_max(error_tensor).numpy())
            units = float(tf.reduce_max(error_tensor/scale).numpy())
            passed = bool(tf.reduce_all(tf.logical_and(tf.math.is_finite(got), error_tensor <= scale)).numpy())
        checks.append(dict(name=name, passed=passed, max_abs=error, max_tolerance_units=units))
        (output/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
        if not passed:
            raise RuntimeError('Batch parity failed: '+name)

    def factory(mod, h, n, replicas=None):
        kwargs = dict(observations=h, particles=n, alpha=.99, process_variance=.1,
                      initial_mean=0, initial_variance=1,
                      observation_log_density=mod.binomial_logistic_observation(50),
                      precision_policy='source_scalar_projection')
        if mod is candidate and replicas is not None:
            kwargs['replicas'] = replicas
        return mod.ScalarGaussianControlledSMC(**kwargs)

    def run_case(label, h, n, b, refinements, repetitions, checked_replicas, case_index):
        core = factory(candidate, h, n, b)
        reference = factory(baseline, h, n)
        scalar = factory(candidate, h, n) if label == 'short' else None
        y_scalar = tf.constant(data[:h], tf.float64)
        y = tf.tile(y_scalar[:, None], [1, b])
        for repetition in range(repetitions):
            policy = tf.zeros([h, b, 3], tf.float64)
            reference_policies = [tf.zeros([h, 3], tf.float64) for _ in range(checked_replicas)]
            scalar_policy = tf.zeros([h, 3], tf.float64)
            seconds = 0.
            for iteration in range(refinements+1):
                started = time.monotonic()
                stream = 200000+case_index*1000+repetition*100+iteration*2
                noise = tf.random.stateless_normal([h, b, n], [20260929, stream], dtype=tf.float64)
                uniform = tf.random.stateless_uniform([h, b], [20260929, stream+1], dtype=tf.float64)
                result = core.forward(policy, y, noise, uniform)
                if not bool(result['valid'].numpy()):
                    raise RuntimeError('Invalid batched forward '+label)
                fit = None
                if iteration < refinements:
                    fit = core.fit(policy, result['x'], result['log_incremental_weight'])
                    if not bool(fit['valid'].numpy()):
                        raise RuntimeError('Invalid batched fit '+label)
                seconds += time.monotonic()-started
                # Reference loops are diagnostic-only; candidate has no row loop.
                if repetition == 0:
                    for replica in range(checked_replicas):
                        ref = reference.forward(reference_policies[replica], y_scalar,
                                                noise[:, replica, :], uniform[:, replica])
                        prefix = f'{label}.i{iteration}.r{replica}'
                        compare(prefix+'.valid', result['valid'], ref['valid'], True)
                        for key in ('x', 'log_incremental_weight', 'logZ', 'ess', 'ancestors'):
                            compare(prefix+'.'+key, result[key][:, replica], ref[key], key == 'ancestors')
                        if fit is not None:
                            ref_fit = reference.fit(reference_policies[replica], ref['x'], ref['log_incremental_weight'])
                            compare(prefix+'.policy', fit['policy'][:, replica], ref_fit['policy'])
                            compare(prefix+'.projected', fit['projected'][:, replica], ref_fit['projected'], True)
                            reference_policies[replica] = ref_fit['policy']
                        if scalar is not None and replica == 0:
                            single = scalar.forward(scalar_policy, y_scalar, noise[:, 0], uniform[:, 0])
                            for key in ('x', 'log_incremental_weight', 'logZ', 'ess', 'ancestors'):
                                compare(prefix+'.scalar_wrapper.'+key, single[key], ref[key], key == 'ancestors')
                            if fit is not None:
                                single_fit = scalar.fit(scalar_policy, single['x'], single['log_incremental_weight'])
                                compare(prefix+'.scalar_wrapper.policy', single_fit['policy'], reference_policies[0])
                                scalar_policy = single_fit['policy']
                if fit is not None:
                    policy = fit['policy']
            traces = {'forward': core.forward.experimental_get_tracing_count()}
            if refinements:
                traces['fit'] = core.fit.experimental_get_tracing_count()
            if any(count != 1 for count in traces.values()):
                raise RuntimeError('Unexpected retracing '+label)
            row = dict(case=label, h=h, particles=n, batch=b, refinements=refinements,
                       repetition=repetition, includes_compilation=repetition == 0,
                       candidate_seconds=seconds, seconds_per_replica=seconds/b,
                       comparison_time_included=False, traces=traces,
                       final_logZ=result['logZ'][-1].numpy().tolist())
            timings.append(row)
            (output/'timings.json').write_text(json.dumps(timings, indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('case','repetition','candidate_seconds','seconds_per_replica')}), flush=True)
        return result['logZ'].device

    try:
        run_case('short', 64, 32, 3, 2, 1, 3, 0)
        run_case('full_csmc', 3000, 128, 10, 3, 2, 1, 1)
        device = run_case('full_bpf', 3000, 5529, 4, 0, 2, 4, 2)
        status, error = 'PASS', None
    except Exception as exc:
        status, error, device = 'FAIL', str(exc), None
    warm = {row['case']:row['seconds_per_replica'] for row in timings if row['repetition'] == 1}
    summary = dict(status=status, scope='NATIVE_REPETITION_BATCH_PARITY_AND_COST_ONLY',
                   replication='INCOMPLETE', error=error, checks=len(checks),
                   passed_checks=sum(c['passed'] for c in checks), timings=timings,
                   warm_seconds_per_replica=warm, gpu_memory_policy=memory,
                   gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
                   host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   tensorflow=tf.__version__, dtype='float64', tf32=False,
                   jit_compile=True, device=device, pfor=False, numpy_runtime=False,
                   per_replica_candidate_loop=False, reference_loop='diagnostic_only')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
