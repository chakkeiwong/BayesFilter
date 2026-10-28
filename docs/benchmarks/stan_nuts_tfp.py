"""Native TFP NUTS with Stan-style warm-up, standalone research API.

The transition is stock TFP, not the custom Stan tree or native Stan RNG.
An enclosing stable-signature XLA function must drive repeated one_step or
sample_chain calls. make_functions supplies compiled warm-up and sampling.
See nuts-stan-tfp.md for changed seed/result semantics and scope.
"""
from typing import NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

if __package__:
    from . import stan_nuts_warmup as adaptation
else:
    import stan_nuts_warmup as adaptation


class NativeTFPResults(NamedTuple):
    inner_results: object
    seed: object  # Next adapter stream seed; not TFP's last-used seed.
    seed_initialized: object
    valid: object


class StanNoUTurnSampler(tfp.mcmc.TransitionKernel):
    """Stock TFP NUTS in physical coordinates with a frozen inverse metric.

    The name is retained for callers; only warm-up is Stan-style. State is
    FP64 [chains, dimension]. The target must return independent FP64 rows.
    Diagonal inverse mass uses elementwise scale, dense inverse mass a lower
    Cholesky bijector, and unit mass has no coordinate wrapper.

    seed_state is now a single stateless int32[2] pair. With no explicit
    one_step seed, the adapter splits/advances this pair. Explicit seeds
    override the stream for that step and produce a new continuation seed.
    TFP maintains independent row draws internally, not per-chain counters.
    Direct-port [chains,2] random states are deliberately rejected.

    Keep previous_kernel_results together with their corresponding position.
    Native transformed kernels reuse the cached transformed position and
    gradients. Changing a target/metric requires a fresh bootstrap. Arbitrary
    external adaptation wrappers, nested states and sharding are not covered.
    """
    def __init__(self, target_log_prob_fn, dimension, chains=1, step_size=1.,
                 inverse_metric=None, metric='diag', max_tree_depth=10,
                 step_size_jitter=0., max_energy_diff=1000., seed_state=None,
                 name='stan_style_native_tfp_nuts'):
        if type(dimension) is not int or dimension < 1:
            raise ValueError('dimension must be a positive static integer')
        if type(chains) is not int or chains < 1:
            raise ValueError('chains must be a positive static integer')
        if type(max_tree_depth) is not int or not 1 <= max_tree_depth <= 20:
            raise ValueError('max_tree_depth must be in [1,20]; TFP instruction storage grows with depth')
        if metric not in ('unit', 'diag', 'dense'):
            raise ValueError('metric must be unit, diag or dense')
        if step_size_jitter != 0.:
            raise ValueError('Native TFP adapter supports zero jitter only')
        if not 0. < max_energy_diff < float('inf'):
            raise ValueError('max_energy_diff must be finite and positive')
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn, dimension=dimension,
            chains=chains, step_size=step_size, inverse_metric=inverse_metric,
            metric=metric, max_tree_depth=max_tree_depth,
            step_size_jitter=step_size_jitter, max_energy_diff=max_energy_diff,
            seed_state=seed_state, name=name)
        self._c, self._d = chains, dimension
        self._epsilon = tf.broadcast_to(tf.convert_to_tensor(step_size, tf.float64), [chains])
        shape = [chains, dimension, dimension] if metric == 'dense' else [chains, dimension]
        if inverse_metric is None:
            inverse_metric = (tf.eye(dimension, batch_shape=[chains], dtype=tf.float64)
                              if metric == 'dense' else tf.ones(shape, tf.float64))
        m = tf.broadcast_to(tf.convert_to_tensor(inverse_metric, tf.float64), shape)
        axes = [-2, -1] if metric == 'dense' else [-1]
        valid = (tf.math.is_finite(self._epsilon) & (self._epsilon > 0.)
                 & tf.reduce_all(tf.math.is_finite(m), axis=axes))

        def checked_target(q):
            q = tf.ensure_shape(q, [chains, dimension])
            value = target_log_prob_fn(q)
            if not tf.is_tensor(value) or value.dtype != tf.float64:
                raise TypeError('target must return FP64 Tensor [chains]')
            return tf.ensure_shape(value, [chains])

        native = tfp.mcmc.NoUTurnSampler(
            checked_target, self._epsilon[:, None], max_tree_depth=max_tree_depth,
            max_energy_diff=max_energy_diff, unrolled_leapfrog_steps=1,
            parallel_iterations=1)
        self._transformed = metric != 'unit'
        if metric == 'dense':
            factor = tf.linalg.cholesky(m)
            valid &= tf.reduce_all(tf.abs(m-tf.linalg.matrix_transpose(m)) <= 1e-8, [-2, -1])
            valid &= tf.reduce_all(tf.math.is_finite(factor), [-2, -1])
            valid &= tf.reduce_all(tf.linalg.diag_part(factor) > 0., -1)
            bijector = tfp.bijectors.ScaleMatvecTriL(factor)
        elif metric == 'diag':
            valid &= tf.reduce_all(m > 0., -1)
            bijector = tfp.bijectors.Scale(tf.sqrt(m))
        else:
            valid &= tf.reduce_all(m == 1., -1)
        self._native = (tfp.mcmc.TransformedTransitionKernel(native, bijector)
                        if self._transformed else native)
        self._parameters_valid = valid
        self._seed = (tf.zeros([2], tf.int32) if seed_state is None else self._pair(seed_state))
        self._seed_initialized = tf.constant(seed_state is not None)

    @staticmethod
    def _pair(seed):
        pair = tf.convert_to_tensor(seed, tf.int32)
        if pair.shape != tf.TensorShape([2]):
            raise ValueError('seed must be one stateless int32[2] pair, not per-chain counters')
        return pair

    @property
    def parameters(self):
        return dict(self._parameters)

    @property
    def is_calibrated(self):
        return self._native.is_calibrated

    @property
    def target_log_prob_fn(self):
        return self._parameters['target_log_prob_fn']

    @property
    def step_size(self):
        return self._epsilon

    @property
    def name(self):
        return self._parameters['name']

    def native_results(self, results):
        """Native NUTS results; target/gradient are in transformed coordinates."""
        inner = results.inner_results
        return inner.inner_results if self._transformed else inner

    def _health(self, q, inner):
        kr = inner.inner_results if self._transformed else inner
        ok = tf.reduce_all(tf.math.is_finite(q), -1) & tf.math.is_finite(kr.target_log_prob)
        ok &= tf.reduce_all(tf.math.is_finite(kr.grads_target_log_prob[0]), -1)
        ok &= ~tf.math.is_nan(kr.log_accept_ratio) & (kr.log_accept_ratio < float('inf'))
        return ok

    def bootstrap_results(self, init_state):
        q = tf.ensure_shape(tf.convert_to_tensor(init_state, tf.float64), [self._c, self._d])
        inner = self._native.bootstrap_results(q)
        return NativeTFPResults(inner, self._seed, self._seed_initialized,
                                self._parameters_valid & self._health(q, inner))

    def one_step(self, current_state, previous_kernel_results, seed=None):
        previous = previous_kernel_results
        rng = previous.seed if seed is None else self._pair(seed)
        initialized = previous.seed_initialized if seed is None else tf.constant(True)
        keys = tf.random.experimental.stateless_split(rng, 2)
        q, inner = self._native.one_step(current_state, previous.inner_results, seed=keys[1])
        valid = previous.valid & initialized & self._health(q, inner)
        return q, NativeTFPResults(inner, keys[0], initialized, valid)

    @classmethod
    def from_warmup(cls, target_log_prob_fn, warmed, config, *, name='stan_style_native_tfp_nuts'):
        """Host bridge from stan_nuts_warmup's native-TFP adaptation only."""
        if not hasattr(warmed, 'metric_factor'):
            raise TypeError('Use stan_nuts_warmup.make_functions; direct-port RNG/results are incompatible')
        frozen = adaptation.freeze(warmed)
        a = frozen.metric_factor
        metric = (tf.matmul(a, a, transpose_b=True) if config.metric == 'dense'
                  else tf.square(tf.linalg.diag_part(a)))
        return cls(target_log_prob_fn, config.dimension, chains=config.chains,
                   step_size=frozen.step_size, inverse_metric=metric, metric=config.metric,
                   max_tree_depth=config.depth, seed_state=frozen.seed, name=name)


require_valid = adaptation.require_valid


def make_functions(tf, tfp, log_prob, config):
    """Return existing Stan-style native-TFP warmup and native TFP sample_chain.

    Same signature/results as stan_nuts_warmup.make_functions; no retracing on
    tuned values. Dense factors and diagonal scales are frozen input tensors.
    Config comes from stan_nuts_warmup.Config. Sampling preserves every draw.
    """
    warmup, _ = adaptation.make_functions(tf, tfp, log_prob, config)
    c, d = config.chains, config.dimension

    @tf.function(input_signature=[tf.TensorSpec([c, d], tf.float64),
        tf.TensorSpec([c], tf.float64), tf.TensorSpec([c, d, d], tf.float64),
        tf.TensorSpec([2], tf.int32)], autograph=False, jit_compile=True)
    def sample(q, e, a, rng):
        inverse = (tf.matmul(a, a, transpose_b=True) if config.metric == 'dense'
                   else tf.square(tf.linalg.diag_part(a)))
        kernel = StanNoUTurnSampler(log_prob, d, chains=c, step_size=e,
            inverse_metric=inverse, metric=config.metric, max_tree_depth=config.depth,
            seed_state=rng)
        initial = kernel.bootstrap_results(q)
        good_factor = tf.reduce_all(tf.math.is_finite(a), [-2, -1])
        good_factor &= tf.reduce_all(tf.linalg.diag_part(a) > 0., -1)
        good_factor &= tf.reduce_all(a == tf.linalg.band_part(a, -1, 0), [-2, -1])
        if config.metric == 'diag':
            good_factor &= tf.reduce_all(a == tf.linalg.diag(tf.linalg.diag_part(a)), [-2, -1])
        initial = initial._replace(valid=initial.valid & good_factor)

        def trace(state, results):
            kr = kernel.native_results(results)
            return tf.stack((tf.exp(tf.minimum(kr.log_accept_ratio, 0.)),
                tf.cast(kr.is_accepted, tf.float64), tf.cast(kr.leapfrogs_taken, tf.float64),
                tf.cast(kr.reach_max_depth, tf.float64), tf.cast(kr.has_divergence, tf.float64),
                tf.cast(results.valid, tf.float64)), -1)

        out = tfp.mcmc.sample_chain(config.draws, q, kernel=kernel,
            previous_kernel_results=initial, trace_fn=trace,
            return_final_kernel_results=True, parallel_iterations=1)
        final = out.final_kernel_results
        token = tf.reduce_sum(out.all_states)+tf.reduce_sum(out.trace)
        return adaptation.SampleResult(out.all_states[-1], final.seed, out.all_states,
                                       out.trace, final.valid, token)

    return warmup, sample
