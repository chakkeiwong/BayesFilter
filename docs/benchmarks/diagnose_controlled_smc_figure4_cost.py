"""Diagnostic-only cost probe of the source Figure 4 right configurations.

This is not a variance replication or statistical ranking. The algorithm is
the separately validated TensorFlow source snapshot; Python orchestrates
independent runs and records diagnostics outside its numerical graphs.
"""
import argparse
import importlib.util
import json
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
spec = importlib.util.spec_from_file_location(
    'controlled_smc_snapshot', Path(__file__).with_name('controlled_smc_tf.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    data = ROOT / '.localresources/code/heng-controlled-smc/StateSpaceModels/NeuroScience/Data_NeuroScience.mat'
    # Isolated diagnostic MAT reader; never imported by the algorithmic path.
    command = [sys.executable, '-B', '-c',
               'from scipy.io import loadmat; import json,sys; print(json.dumps(loadmat(sys.argv[1])["y"].ravel().tolist()))',
               str(data)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    y_values = json.loads(subprocess.check_output(command, text=True, env=env))
    (output / 'observations.json').write_text(json.dumps(y_values) + '\n')
    h = len(y_values)
    if h != 3000:
        raise ValueError('Unexpected source observation horizon')
    d = tf.float64
    y = tf.constant(y_values, d)
    rows = []
    cores = []
    healthy = True
    device = None
    for grid_index, variance in enumerate((.01, .10, .20)):
        for method, particles, refinements in (('csmc', 128, 3), ('bpf', 5529, 0)):
            core = module.ScalarGaussianControlledSMC(
                observations=h, particles=particles, alpha=.99,
                process_variance=variance, initial_mean=0, initial_variance=1,
                observation_log_density=module.binomial_logistic_observation(50),
                precision_policy='source_scalar_projection')
            cores.append(core)
            for repetition in range(2):
                started = time.monotonic()
                policy = tf.zeros([h, 3], d)
                stages = []
                run_healthy = True
                for iteration in range(refinements + 1):
                    stream = 100000 + grid_index * 1000 + (0 if method == 'csmc' else 500) + repetition * 20 + iteration * 2
                    normal = tf.random.stateless_normal([h, particles], seed=[20260929, stream], dtype=d)
                    uniform = tf.random.stateless_uniform([h], seed=[20260929, stream + 1], dtype=d)
                    forward_start = time.monotonic()
                    result = core.forward(policy, y, normal, uniform)
                    logz = float(result['logZ'][-1].numpy())
                    forward_seconds = time.monotonic() - forward_start
                    run_healthy = run_healthy and bool(result['valid'].numpy())
                    device = result['logZ'].device
                    stage = dict(iteration=iteration, stream=stream, logZ=logz,
                                 forward_seconds=forward_seconds,
                                 minimum_ess=float(tf.reduce_min(result['ess']).numpy()))
                    if iteration < refinements and run_healthy:
                        fit_start = time.monotonic()
                        fit = core.fit(policy, result['x'], result['log_incremental_weight'])
                        policy = fit['policy']
                        fit_valid = bool(fit['valid'].numpy())
                        stage.update(fit_seconds=time.monotonic() - fit_start,
                                     minimum_gram_eigenvalue_ratio=float(tf.reduce_min(fit['gram_eigenvalue_ratio']).numpy()),
                                     projection_count=int(tf.reduce_sum(tf.cast(fit['projected'], tf.int32)).numpy()))
                        run_healthy = run_healthy and fit_valid
                    stages.append(stage)
                    if not run_healthy:
                        break
                traces = {'forward': core.forward.experimental_get_tracing_count()}
                if refinements:
                    traces['fit'] = core.fit.experimental_get_tracing_count()
                run_healthy = run_healthy and all(v == 1 for v in traces.values())
                row = dict(method=method, particles=particles, refinements=refinements,
                           process_variance=variance, repetition=repetition,
                           includes_first_compilation=repetition == 0,
                           wall_seconds=time.monotonic() - started,
                           stages=stages, traces=traces, valid=run_healthy)
                rows.append(row)
                with (output / 'rows.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=True) + '\n')
                print(json.dumps({k: row[k] for k in ('method', 'process_variance', 'repetition', 'wall_seconds', 'valid')}), flush=True)
                healthy = healthy and run_healthy
                if not healthy:
                    break
            if not healthy:
                break
        if not healthy:
            break
    complete = healthy and len(rows) == 12
    estimate = None
    if complete:
        combined = {}
        for row in rows:
            key = (row['process_variance'], row['repetition'])
            combined[key] = combined.get(key, 0.) + row['wall_seconds']
        cold = max(value for (_, repetition), value in combined.items() if repetition == 0)
        warm = max(value for (_, repetition), value in combined.items() if repetition == 1)
        estimate = dict(max_combined_cold_seconds=cold, max_combined_warm_seconds=warm,
                        planning_seconds=1.5 * (20 * cold + 20 * 99 * warm) + 60,
                        formula='1.5 * (20 * max_cold + 20 * 99 * max_warm) + 60',
                        guarantee=False)
    summary = dict(status='PASS' if complete else 'FAIL',
                   scope='FIGURE4_SOURCE_CONFIGURATION_COST_AND_VALIDITY_PILOT_ONLY',
                   replication='INCOMPLETE', ranking='NOT_TESTED', rows=rows,
                   heuristic_dominance='NOT_EVALUATED_COST_PROBE_ONLY',
                   full_source_experiment_cost_estimate=estimate,
                   observations=h, seed=20260929, gpu_memory_policy=memory,
                   gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
                   host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   tensorflow=tf.__version__, dtype='float64', tf32=False,
                   jit_compile=True, device=device, pfor=False, numpy_runtime=False)
    # NaN diagnostics are retained on failure; a failed artifact cannot pass.
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=not complete) + '\n')
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
