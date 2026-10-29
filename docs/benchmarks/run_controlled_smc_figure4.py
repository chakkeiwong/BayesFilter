"""Execute and report the frozen Figure 4 right replication matrix.

Algorithms use the validated TensorFlow snapshot. This diagnostic benchmark
driver orchestrates independent calls and reports source statistics. Its
execution PASS is not scientific promotion or completion of the master.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
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
spec = importlib.util.spec_from_file_location('controlled_smc_snapshot', Path(__file__).with_name('controlled_smc_tf.py'))
core_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_module)
D = tf.float64
REPETITIONS = 100
BOOTSTRAPS = 3999


def metrics(values, axis):
    mean = tf.reduce_mean(values, axis=axis)
    variance = tf.math.reduce_variance(values, axis=axis)*REPETITIONS/(REPETITIONS-1)
    log_rvar = tf.math.log(variance/(mean*mean))/tf.math.log(tf.constant(10., D))
    log_mean_z = tf.reduce_logsumexp(values, axis=axis)-tf.math.log(tf.constant(REPETITIONS, D))
    return dict(mean_logZ=mean, variance_logZ=variance,
                log10_relative_variance_logZ=log_rvar, log_mean_Z=log_mean_z)


point_metrics = tf.function(lambda values: metrics(values, 0),
                            input_signature=[tf.TensorSpec([100, 20], D)],
                            jit_compile=True, autograph=False)
bootstrap_metrics = tf.function(
    lambda values, indices: metrics(tf.gather(values, indices), 1),
    input_signature=[tf.TensorSpec([100, 20], D), tf.TensorSpec([BOOTSTRAPS, 100], tf.int32)],
    jit_compile=True, autograph=False)


def as_list(value):
    return tf.reshape(value, [-1]).numpy().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    (output/'fit-diagnostics').mkdir()
    started = time.monotonic()
    rows, trace_rows, device = [], [], None
    context = {}
    reference_directory = ROOT/'.localresources/code/heng-controlled-smc/StateSpaceModels/NeuroScience'
    # Independent CPU diagnostic reader for the MAT container only. It does not
    # supply any numerical operation to the TensorFlow filtering consumer.
    code = ('from scipy.io import loadmat; import json,sys; from pathlib import Path; '
            'p=Path(sys.argv[1]); c=loadmat(p/"NormConst_cSMC.mat"); b=loadmat(p/"NormConst_BPF.mat"); '
            'print(json.dumps(dict(y=loadmat(p/"Data_NeuroScience.mat")["y"].ravel().tolist(), '
            'csmc=c["cSMC_LogNormConst"].tolist(), bpf=b["BPF_LogNormConst"].tolist(), '
            'csmc_n=int(c["cSMC_N"].item()), csmc_i=int(c["cSMC_I"].item()), bpf_n=int(b["BPF_N"].item()))))')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    source = json.loads(subprocess.check_output([sys.executable, '-B', '-c', code, str(reference_directory)], env=env, text=True))
    y_values = source.pop('y')
    (output/'observations.json').write_text(json.dumps(y_values)+'\n')
    (output/'author-reference.json').write_text(json.dumps(source)+'\n')
    if (len(y_values) != 3000 or (source['csmc_n'], source['csmc_i'], source['bpf_n']) != (128, 3, 5529)
            or any(y < 0 or y > 50 or int(y) != y for y in y_values)):
        raise ValueError('Source settings/data mismatch')
    for method in ('csmc', 'bpf'):
        if len(source[method]) != 100 or any(len(row) != 20 for row in source[method]):
            raise ValueError('Source reference matrix shape mismatch')
    y_scalar = tf.constant(y_values, D)
    y_batch = tf.tile(y_scalar[:, None], [1, 10])
    fixture = tf.tile(tf.range(100, dtype=D)[:, None], [1, 20])
    check = point_metrics(fixture)
    if (float(tf.reduce_max(tf.abs(check['mean_logZ']-49.5)).numpy()) > 1e-12 or
            float(tf.reduce_max(tf.abs(check['variance_logZ']-tf.constant(100*101/12, D))).numpy()) > 1e-9):
        raise RuntimeError('Sample-variance reporting identity failed')

    def core(n, q, batch=None):
        return core_module.ScalarGaussianControlledSMC(
            observations=3000, particles=n, alpha=.99, process_variance=q,
            initial_mean=0, initial_variance=1,
            observation_log_density=core_module.binomial_logistic_observation(50),
            precision_policy='source_scalar_projection', replicas=batch)

    def run_batch(engine, method, grid_index, repetition_start, batch, refinements):
        nonlocal device, context
        batch_started = time.monotonic()
        n = 128 if method == 'csmc' else 5529
        prefix = [3000, batch] if batch > 1 else [3000]
        policy = tf.zeros(prefix+[3], D)
        stage_rows = []
        for iteration in range(refinements+1):
            context = dict(method=method, grid_index=grid_index, repetition_start=repetition_start, iteration=iteration)
            stream = ((1000000+grid_index*10000+(repetition_start//10)*100)
                      if method == 'csmc' else (3000000+grid_index*10000+repetition_start*2)) + iteration*2
            noise = tf.random.stateless_normal(prefix+[n], [20260929, stream], dtype=D)
            uniform = tf.random.stateless_uniform(prefix, [20260929, stream+1], dtype=D)
            step_start = time.monotonic()
            result = engine.forward(policy, y_batch if batch > 1 else y_scalar, noise, uniform)
            valid = bool(result['valid'].numpy())
            device = result['logZ'].device
            stage = dict(iteration=iteration, stream=stream, logZ=as_list(result['logZ'][-1]),
                         minimum_ess=as_list(tf.reduce_min(result['ess'], axis=0)),
                         forward_seconds=time.monotonic()-step_start)
            if not valid:
                (output/'failure-diagnostics.json').write_text(json.dumps(dict(context=context, forward=stage), indent=2)+'\n')
                raise RuntimeError('Invalid forward call; no seed is discarded')
            if iteration < refinements:
                fit_start = time.monotonic()
                fit = engine.fit(policy, result['x'], result['log_incremental_weight'])
                policy = fit['policy']
                valid = bool(fit['valid'].numpy())
                stage.update(fit_seconds=time.monotonic()-fit_start,
                             minimum_gram_eigenvalue_ratio=as_list(tf.reduce_min(fit['gram_eigenvalue_ratio'], axis=0)),
                             maximum_fit_residual=as_list(tf.reduce_max(fit['max_residual'], axis=0)),
                             projection_count=as_list(tf.reduce_sum(tf.cast(fit['projected'], tf.int32), axis=0)))
                # Keep computed diagnostics and fitted policies, not just minima.
                for key, value in fit.items():
                    name = f'g{grid_index:02d}_r{repetition_start:03d}_i{iteration}_{key}.tensor'
                    tf.io.write_file(str(output/'fit-diagnostics'/name), tf.io.serialize_tensor(value))
                if not valid:
                    (output/'failure-diagnostics.json').write_text(json.dumps(dict(context=context, fit=stage), indent=2)+'\n')
                    raise RuntimeError('Invalid fit; preserve seed and diagnose before retry')
            stage_rows.append(stage)
        wall_seconds = time.monotonic()-batch_started
        produced = []
        for replica in range(batch):
            per_stage = []
            for stage in stage_rows:
                per_stage.append({key:(value[replica] if isinstance(value, list) else value)
                                  for key, value in stage.items()})
            row = dict(method=method, process_variance=(grid_index+1)/100,
                       repetition=repetition_start+replica, batch=batch, batch_replica=replica,
                       particles=n, refinements=refinements, seed=20260929,
                       logZ=per_stage[-1]['logZ'], stages=per_stage,
                       allocated_batch_seconds=wall_seconds/batch,
                       includes_first_compilation=repetition_start == 0, valid=True)
            produced.append(row)
        with (output/'rows.jsonl').open('a') as handle:
            for row in produced:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
        rows.extend(produced)

    try:
        for grid_index in range(20):
            variance = (grid_index+1)/100
            cell_started = time.monotonic()
            engines = {'csmc':core(128, variance, 10), 'bpf':core(5529, variance)}
            for method, batch, refinements in (('csmc', 10, 3), ('bpf', 1, 0)):
                for repetition_start in range(0, 100, batch):
                    run_batch(engines[method], method, grid_index, repetition_start, batch, refinements)
                    progress = dict(status='RUNNING', cell=grid_index+1, total_cells=20,
                                    method=method, completed_method_repetitions=repetition_start+batch,
                                    completed_rows=len(rows), wall_seconds=time.monotonic()-started,
                                    scientific_verdict='PENDING_SOURCE_REVIEW_AND_HEURISTIC_COMPLETION')
                    (output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                traces = {'forward':engines[method].forward.experimental_get_tracing_count()}
                if refinements:
                    traces['fit'] = engines[method].fit.experimental_get_tracing_count()
                if any(count != 1 for count in traces.values()):
                    raise RuntimeError('Unexpected retracing in '+method)
                trace_rows.append(dict(method=method, variance=variance, traces=traces))
            print(json.dumps(dict(cell=grid_index+1, variance=variance,
                                  completed_rows=len(rows), cell_seconds=time.monotonic()-cell_started)), flush=True)
        if len(rows) != 4000:
            raise RuntimeError('Incomplete experiment matrix')
        arrays = {}
        for method in ('csmc', 'bpf'):
            by_key = {(round(r['process_variance']*100)-1, r['repetition']):r for r in rows if r['method'] == method}
            if len(by_key) != 2000:
                raise RuntimeError('Missing or duplicate experiment row')
            arrays['local_'+method] = tf.constant([[by_key[g,r]['logZ'] for g in range(20)] for r in range(100)], D)
            arrays['author_'+method] = tf.constant(source[method], D)
            if method == 'csmc':
                arrays['local_bpf128'] = tf.constant([[by_key[g,r]['stages'][0]['logZ'] for g in range(20)] for r in range(100)], D)
        point, bootstrap, interval, curves = {}, {}, {}, {}
        def ci(value):
            sorted_values = tf.sort(value, axis=0)
            return tf.gather(sorted_values, [math.floor(.025*(BOOTSTRAPS-1)), math.ceil(.975*(BOOTSTRAPS-1))])
        for index, (name, values) in enumerate(arrays.items()):
            point[name] = point_metrics(values)
            indices = tf.random.stateless_uniform([BOOTSTRAPS, 100], [20260929, 3900000+index],
                                                  maxval=100, dtype=tf.int32)
            bootstrap[name] = bootstrap_metrics(values, indices)
            interval[name] = {key:ci(value) for key, value in bootstrap[name].items()}
            if any(not bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in point[name].values()):
                raise RuntimeError('Nonfinite reporting statistic')
            curves[name] = {key:dict(estimate=value.numpy().tolist(),
                                    pointwise_95_interval=interval[name][key].numpy().tolist())
                            for key, value in point[name].items()}
        contrasts = {}
        for left, right in (('local_csmc','author_csmc'), ('local_bpf','author_bpf'), ('local_csmc','local_bpf')):
            key = 'log10_relative_variance_logZ'
            contrasts[left+'_minus_'+right] = dict(
                estimate=(point[left][key]-point[right][key]).numpy().tolist(),
                pointwise_95_interval=ci(bootstrap[left][key]-bootstrap[right][key]).numpy().tolist())
        (output/'curves.json').write_text(json.dumps(dict(
            variances=[(i+1)/100 for i in range(20)], statistic='log10(sample_var(logZ)/mean(logZ)^2)',
            degrees_of_freedom=99, bootstrap_resamples=BOOTSTRAPS, simultaneous_intervals=False,
            curves=curves, contrasts=contrasts), indent=2, allow_nan=False)+'\n')
        status, error = 'PASS', None
    except Exception as exc:
        status, error = 'FAIL', str(exc)
    summary = dict(status=status, scope='FIGURE4_RIGHT_SOURCE_MATRIX_EXECUTION_AND_UNCERTAINTY_ONLY',
                   replication='INCOMPLETE_MASTER', error=error, last_context=context,
                   scientific_verdict='PENDING_SOURCE_CURVE_REVIEW_AND_HEURISTIC_COMPLETION',
                   heuristic_dominance='INCOMPLETE_GUIDED_AND_TEMPORAL_COMPARATORS',
                   completed_rows=len(rows), required_rows=4000, trace_rows=trace_rows,
                   statistics_traces=dict(point=point_metrics.experimental_get_tracing_count(),
                                          bootstrap=bootstrap_metrics.experimental_get_tracing_count()),
                   gpu_memory_policy=memory, gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
                   host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   tensorflow=tf.__version__, dtype='float64', tf32=False, jit_compile=True,
                   device=device, wall_seconds=time.monotonic()-started, seed=20260929,
                   pfor=False, numpy_runtime=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'progress.json').write_text(json.dumps(dict(status=status, completed_rows=len(rows), error=error), indent=2)+'\n')
    (output/'result.md').write_text(
        '# Figure 4 right execution\n\n'+
        f'Execution status: {status}. Completed {len(rows)} of 4000 required rows.\n\n'+
        'The source-curve uncertainty tables are in curves.json when complete. '+
        'Scientific source-curve review and the guided/temporal heuristic comparisons remain pending. '+
        'No whole-master completion or broad superiority claim follows from this execution.\n'+
        (f'\nFailure: {error}\n' if error else ''))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
