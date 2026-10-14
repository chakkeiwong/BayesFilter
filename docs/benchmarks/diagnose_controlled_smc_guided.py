"""Diagnostic/reference controls and cost pilot for the guided-filter repair."""
import argparse
import importlib.util
import json
import math
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
D = tf.float64


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load('controlled_snapshot', 'controlled_smc_tf.py')
g = load('guided_snapshot', 'gaussian_guided_pf_tf.py')
r = load('grid_reference_snapshot', 'controlled_smc_filter_reference_tf.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    checks, costs = [], []
    started = time.monotonic()
    def check(name, actual, expected, tolerance=1e-10):
        actual, expected = tf.convert_to_tensor(actual, D), tf.convert_to_tensor(expected, D)
        error = float(tf.reduce_max(tf.abs(actual-expected)).numpy())
        passed = math.isfinite(error) and error <= tolerance
        checks.append(dict(name=name, max_absolute_error=error, tolerance=tolerance, passed=passed))
        if not passed:
            raise RuntimeError(name+' failed: '+str(error))
    def valid(name, result):
        check(name, float(bool(result['valid'].numpy())), 1., 0.)
    def controlled(h, n, b, q, logg):
        return c.ScalarGaussianControlledSMC(observations=h, particles=n, replicas=b,
            alpha=.99, process_variance=q, initial_mean=.2, initial_variance=1.3,
            observation_log_density=logg, precision_policy='source_scalar_projection')
    def guided(h, n, b, method, logg, moments, jit=True):
        return g.ScalarGaussianGuidedPF(observations=h, particles=n, replicas=b,
            alpha=.99, initial_mean=.2, initial_variance=1.3, observation_log_density=logg,
            observation_mean=moments[0], observation_variance=moments[1],
            observation_derivative=moments[2], guide=method, jit_compile=jit)
    try:
        linear = (lambda x: 1.2*x+.1, lambda x: tf.ones_like(x)*.4,
                  lambda x: tf.ones_like(x)*1.2)
        logg = lambda y, x: g.normal_log_density(y, 1.2*x+.1, tf.constant(.4, D))
        means = tf.constant([[-3., -.1, .3, 2.], [-1., 0., 1., 4.]], D)
        variance, y = tf.constant(.7, D), tf.constant([.4, -.8], D)
        exact_v = .7*.4/(1.2**2*.7+.4)
        exact_m = means+.7*1.2/(1.2**2*.7+.4)*(y[:, None]-1.2*means-.1)
        for method in ('ekf', 'quadrature3'):
            engine = guided(8, 4, 2, method, logg, linear)
            proposal = engine.proposal(means, variance, y)
            valid(method+'_linear_proposal_valid', proposal)
            check(method+'_exact_conditional_mean', proposal['mean'], exact_m)
            check(method+'_exact_conditional_variance', proposal['variance'], exact_v)
            testx = proposal['mean']+tf.sqrt(proposal['variance'])*means
            weights = (logg(y[:, None], testx)+g.normal_log_density(testx, means, variance)
                       -g.normal_log_density(testx, proposal['mean'], proposal['variance']))
            check(method+'_normalized_importance_identity', weights,
                  g.normal_log_density(y[:, None], 1.2*means+.1, tf.constant(1.2**2*.7+.4, D)))
            data = tf.reshape(tf.linspace(tf.constant(-1., D), tf.constant(1., D), 16), [8, 2])
            noise = tf.random.stateless_normal([8, 2, 4], [20260929, 4100000], dtype=D)
            uniform = tf.random.stateless_uniform([8, 2], [20260929, 4100001], dtype=D)
            result = engine.forward(data, noise, uniform, tf.constant(.1, D))
            valid(method+'_sequential_valid', result)
            # Deliberate graph/no-XLA reference exception for the same call chain.
            reference = guided(8, 4, 2, method, logg, linear, False).forward(data, noise, uniform, tf.constant(.1, D))
            for key in ('mean', 'variance', 'logZ', 'x', 'log_incremental_weight', 'ancestors'):
                check(method+'_graph_parity_'+key, tf.cast(result[key], D), tf.cast(reference[key], D), 1e-9)
            check(method+'_one_trace', engine.forward.experimental_get_tracing_count(), 1, 0)
        # Grid integral compared with a separate ordinary Python Kalman recursion.
        data = [.2, -.4, .8, 1.1, -.5, .3, .7, -.2]
        oracle = r.GridFilterReference(observations=8, points=513, bound=16., alpha=.99,
            initial_mean=.2, initial_variance=1.3, observation_log_density=logg)
        grid = oracle.forward(tf.constant(data, D), tf.constant(.1, D))
        km, kv, kz, expected = .2, 1.3, 0., {key:[] for key in ('mean', 'variance', 'logZ')}
        for t, obs in enumerate(data):
            if t:
                km, kv = .99*km, .99**2*kv+.1
            residual, innovation = obs-(1.2*km+.1), 1.2**2*kv+.4
            kz += -.5*(math.log(2*math.pi*innovation)+residual**2/innovation)
            km, kv = km+kv*1.2/innovation*residual, kv*.4/innovation
            for key, value in zip(('mean','variance','logZ'), (km,kv,kz)):
                expected[key].append(value)
        for key in expected:
            check('grid_kalman_'+key, grid[key], expected[key], 1e-9)
        # Corrected original-target weights must equal direct p/q weights at t=0.
        core = controlled(8, 4, 2, .1, logg)
        policy = tf.broadcast_to(tf.constant([.2, -.15, .07], D), [8,2,3])
        result = core.forward(policy, tf.constant(data, D)[:, None]*tf.ones([1,2], D), noise, uniform)
        original = core.original_filtering(policy, tf.constant(data, D)[:, None]*tf.ones([1,2], D), result['x'])
        cm, cv, _, _ = c.gaussian_twist(tf.constant(.2, D), 1.3, policy[0, :, None, :])
        x = result['x'][0]
        logw = logg(tf.constant(data[0], D), x)+g.normal_log_density(x, tf.constant(.2, D), tf.constant(1.3, D))-g.normal_log_density(x, cm, cv)
        check('untwist_direct_initial_normalizer', original['logZ'][0], tf.reduce_logsumexp(logw, -1)-math.log(4))
        check('untwist_direct_initial_mean', original['mean'][0], tf.reduce_sum(tf.nn.softmax(logw,-1)*x,-1))
        check('untwist_final_normalizer', original['logZ'][-1], result['logZ'][-1])
        zero = tf.zeros_like(policy)
        result0 = core.forward(zero, tf.constant(data, D)[:, None]*tf.ones([1,2], D), noise, uniform)
        original0 = core.original_filtering(zero, tf.constant(data, D)[:, None]*tf.ones([1,2], D), result0['x'])
        check('untwist_zero_policy_prefixes', original0['logZ'], result0['logZ'])
        # Full-length validity/cost pilot at three noise regimes, never used for ranking.
        path = ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928/attempt15_c3_figure4/observations.json'
        observations = tf.constant(json.loads(path.read_text()), D)
        counts = tf.tile(observations[:, None], [1,10])
        moments = g.binomial_moments(50)
        for method in ('ekf', 'quadrature3'):
            engine = g.ScalarGaussianGuidedPF(observations=3000, particles=128, replicas=10,
                alpha=.99, initial_mean=0., initial_variance=1., observation_log_density=c.binomial_logistic_observation(50),
                observation_mean=moments[0], observation_variance=moments[1], observation_derivative=moments[2], guide=method)
            for index, q in enumerate((.01,.1,.2)):
                noise = tf.random.stateless_normal([3000,10,128], [20260929,4200000+index], dtype=D)
                uniform = tf.random.stateless_uniform([3000,10], [20260929,4200010+index], dtype=D)
                start = time.monotonic()
                result = engine.forward(counts, noise, uniform, tf.constant(q,D))
                valid(method+'_full_q'+str(q), result)
                costs.append(dict(method=method, q=q, seconds=time.monotonic()-start,
                    replicas=10, includes_compile=index==0,
                    minimum_ess=float(tf.reduce_min(result['ess']).numpy()),
                    minimum_relative_variance=float(tf.reduce_min(result['relative_margin_min']).numpy())))
            check(method+'_full_one_trace',engine.forward.experimental_get_tracing_count(),1,0)
        # Extremes chosen in advance; no covariance clipping or floor.
        for method in ('ekf','quadrature3'):
            engine = guided(2,4,2,method,c.binomial_logistic_observation(50),moments)
            for q in (.01, 1.):
                proposal=engine.proposal(tf.constant([[-20.,-8.,8.,20.]]*2,D),tf.constant(q,D),tf.constant([0.,50.],D))
                valid(method+'_extreme_'+str(q),proposal)
        summary=dict(status='PASS',scope='normalized_guides_original_target_and_cost_validation',
            checks=checks,costs=costs,algorithm_seconds=time.monotonic()-started,
            gpu_memory_policy=memory,tensorflow=tf.__version__,jit_compile=True,tf32=False,
            device=result['logZ'].device,allocator=tf.config.experimental.get_memory_info('GPU:0'),
            host_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            replication='NOT_COMPLETE',nonjit_exception='same-input graph parity only')
    except BaseException as exc:
        summary=dict(status='FAIL',error=repr(exc),checks=checks,costs=costs,
                     gpu_memory_policy=memory,tensorflow=tf.__version__,jit_compile=True)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        raise
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],checks=len(checks),costs=costs)))


if __name__=='__main__':
    main()
