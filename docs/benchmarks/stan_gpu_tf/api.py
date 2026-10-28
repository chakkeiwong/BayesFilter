"""Stan Euclidean NUTS for differentiable, row-independent TensorFlow models.

The shared direct port owns tree construction, leapfrog and adaptation. This
module adds explicit runtime model tensors, an enclosing full-fit computation,
and static per-chain Philox batching. Python loops here expand a bounded graph
at trace time; there is no Python callback or host sampling at execution time.

FP64 unconstrained positions are [chains, dimension]; model(q, *data) returns
FP64 [chains]. The caller includes parameter-transform log Jacobians and keeps
rows independent. Arbitrary TensorFlow ops are supported only if their forward
and position-gradient operations compile for the selected XLA backend.
"""
from typing import NamedTuple
from .. import stan_nuts as core

Config = core.Config


class StaticChainKernel(core.Kernel):
    """Same TF Philox calls/streams, with no runtime loop over chains."""
    def random(self, seed, mask, *, normal=False, size=()):
        if self.tape is not None or self.c == 1:
            return super().random(seed, mask, normal=normal, size=size)
        tf = self.tf
        fn = tf.random.stateless_normal if normal else tf.random.stateless_uniform
        value = tf.stack([fn(size, seed[i], dtype=self.dtype, alg='philox')
                          for i in range(self.c)], axis=0)
        return value, seed + tf.stack((tf.zeros([self.c], tf.int32),
                                      tf.cast(mask, tf.int32)), axis=-1)


class FitResult(NamedTuple):
    initialization_valid: object
    initialization_attempts: object
    warmup: object
    sampling: object
    valid: object
    completion_token: object


class Functions(NamedTuple):
    fit: object
    initialize: object
    warmup: object
    sample: object
    value_gradient: object


def make_functions(tf, model, config, model_signature=(), *, rng_layout='static'):
    """Build stable-signature XLA functions; all data inputs remain runtime tensors.

    fit(q, present, epsilon, inverse_metric, seeds, *data) performs the complete
    lifecycle inside XLA and preserves warmup separately. Seeds are int32 [C,2]
    with distinct stream keys and nonnegative counters, as in the direct port.
    For chunk continuation use sample(q, epsilon, inverse_metric, seeds, *data).
    Changing data changes the target: restart adaptation before posterior use.

    Static layout expands C RNG calls at trace time (linear graph growth). The
    'map' baseline retains the original runtime loop for controlled comparisons.
    Neither layout reproduces Stan's MIXMAX random stream. Raw compiled methods
    return validity flags; callers must check them at a host reporting boundary.
    No differentiation through the accept/reject sampler is promised.
    """
    if rng_layout not in ('static', 'map'):
        raise ValueError('rng_layout must be static or map')
    kind = StaticChainKernel if rng_layout == 'static' else core.Kernel
    signature = tuple(model_signature)
    def named_model(q, *data):
        with tf.name_scope('tf_model'):
            return model(q, *data)
    initialize, warmup, sample = core.make_functions(
        tf, named_model, config, model_signature=signature, kernel_type=kind)
    c, d = config.chains, config.dimension
    qspec = tf.TensorSpec([c, d], tf.float64)
    metric_shape = [c, d, d] if config.metric == 'dense' else [c, d]

    @tf.function(input_signature=[qspec, tf.TensorSpec([c, d], tf.bool),
        tf.TensorSpec([c], tf.float64), tf.TensorSpec(metric_shape, tf.float64),
        tf.TensorSpec([c, 2], tf.int32), *signature], autograph=False, jit_compile=True)
    def fit(q, present, epsilon, inverse_metric, seeds, *data):
        q, seeds, initial_valid, attempts = initialize(q, present, seeds, *data)
        w = warmup(q, epsilon, inverse_metric, seeds, *data)
        s = sample(w.position, w.step_size, w.inverse_metric, w.seed, *data)
        valid = initial_valid & w.valid & s.valid
        return FitResult(initial_valid, attempts, w, s, valid,
                         w.completion_token + s.completion_token)

    @tf.function(input_signature=[qspec, *signature], autograph=False, jit_compile=True)
    def value_gradient(q, *data):
        with tf.name_scope('tf_model_value_and_gradient'):
            kernel = kind(tf, lambda x: model(x, *data), config)
            potential, score = kernel.value_score(q)
            return -potential, -score

    return Functions(fit, initialize, warmup, sample, value_gradient)
