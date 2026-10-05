"""Compiled independent-coordinate composition of the scalar retained route.

This preserves the existing fixed-design extension, not coupled multivariate
TT or source-faithful TT-cross. It is a filtering evaluator, not a batch-native
NeuTra training target. Model callbacks reuse the public scalar value and
analytical parameter-score formulas with a tensor-valued fixed sigma.
"""

from collections import OrderedDict

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.models import StochasticVolatilitySSM
from bayesfilter.highdim.scalar_retained_native_tf import ScalarRetainedProgram

D = tf.float64
_CACHE = OrderedDict()


class _CoordinateModel(StochasticVolatilitySSM):
    """Tensor callback view; its physical inputs are validated by the panel API."""

    def __init__(self, sigma, mixture):
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "mixture", mixture)

    def observation_log_density(self, theta, x_t, y_t, t):
        from bayesfilter.highdim.sv_mixture_cut4 import (
            ExactTransformedSVSSM,
            KSCMixtureTransformedSVSSM,
        )
        kind = ExactTransformedSVSSM if self.mixture is None else KSCMixtureTransformedSVSSM
        return kind.observation_log_density(self, theta, x_t, y_t, t)

    def observation_log_density_parameter_score(self, theta, x_t, y_t, t):
        from bayesfilter.highdim.sv_mixture_cut4 import (
            ExactTransformedSVSSM,
            KSCMixtureTransformedSVSSM,
        )
        kind = ExactTransformedSVSSM if self.mixture is None else KSCMixtureTransformedSVSSM
        return kind.observation_log_density_parameter_score(self, theta, x_t, y_t, t)


def make_sv_panel_retained_program(config, observation_shape, *, mixture=None,
                                   derivative_config=None, jit_compile=True):
    """One stable signature across parameter/data changes at a fixed panel shape."""
    key = (id(config), tuple(observation_shape), id(mixture), derivative_config, bool(jit_compile))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key][2:]
    horizon, dimension = observation_shape
    if horizon < 1 or dimension < 1:
        raise ValueError("observations require nonempty [time, coordinate] shape")
    model = _CoordinateModel(tf.constant(1.0, D), mixture)
    scalar = ScalarRetainedProgram(model, config, (horizon, 1), 257, 321,
                                   derivative_config, jit_compile)
    output_spec = tf.nest.map_structure(
        lambda value: tf.TensorSpec(value.shape, value.dtype),
        scalar.call.get_concrete_function().structured_outputs)
    normal = tfp.distributions.Normal(tf.constant(0.0, D), tf.constant(1.0, D))

    @tf.function(input_signature=[tf.TensorSpec(observation_shape, D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension], D)], jit_compile=jit_compile, autograph=False)
    def evaluate(observations, gamma, beta, sigma):
        theta = tf.stack((normal.quantile(gamma), tf.math.log(beta)), axis=1)

        def coordinate(inputs):
            local_theta, data, local_sigma = inputs
            return scalar.numerical(local_theta, data[:, None],
                                    _CoordinateModel(local_sigma, mixture))

        coordinates = tf.map_fn(coordinate, (theta, tf.transpose(observations), sigma),
                                fn_output_signature=output_spec, parallel_iterations=1)
        histories = coordinates["history"]
        return {
            "coordinates": coordinates, "theta": theta,
            "log_likelihood": tf.reduce_sum(coordinates["log_likelihood"]),
            "log_normalizers": tf.reduce_sum(histories["increment"], axis=0),
            "score": tf.reshape(coordinates["score"], [-1]),
            "mean_path": tf.transpose(histories["mean"]),
            "covariance_path": tf.linalg.diag(tf.transpose(histories["variance"])),
        }

    _CACHE[key] = (config, mixture, evaluate, scalar)
    if len(_CACHE) > 8:
        _CACHE.popitem(last=False)
    return evaluate, scalar


def validate_panel_history(scalar, histories):
    """Validate completed coordinate histories without rerunning any filter."""
    flattened = tf.nest.map_structure(
        lambda value: tf.reshape(value, [-1, *value.shape[2:]]), histories)
    scalar.validate(flattened)
