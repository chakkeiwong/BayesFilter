"""Independent FP64 actual-SV quadrature reference; no DCTR or HMC admission.

The returned score differentiates the finite quadrature, including stationary
initialization. Domain/refinement diagnostics are empirical, not error bounds.
"""
from dataclasses import dataclass
import math

import tensorflow as tf

DTYPE = tf.float64
LOG_TWO_PI = math.log(2*math.pi)
PARTITION_SEEDS = {"calibration": 11100, "validation": 11200, "claim": 11300}
PROBES = tuple((a, b) for a in (.1, .5, .9) for b in (.1, .5, .9)) + (
    (.6, .4), (.2, .8), (.8, .2))
GRID_LADDER = ((128, 12.), (256, 12.), (512, 12.), (512, 16.), (1024, 16.), (1280, 20.))


@tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
def generate_dataset(seed):
    initial = tf.random.stateless_normal([], seed, dtype=DTYPE)/tf.sqrt(tf.constant(1-.6**2, DTYPE))
    process = tf.random.stateless_normal([20], seed+[0, 1], dtype=DTYPE)
    observation = tf.random.stateless_normal([20], seed+[0, 2], dtype=DTYPE)
    states = tf.scan(lambda previous, noise: .6*previous+noise, process, initializer=initial)
    raw = tf.cast(.4*tf.exp(.5*states)*observation, tf.float32)
    return initial, states, raw


def raw_observation_log_density(x, beta, observation):
    residual_square = tf.square(observation/beta)*tf.exp(-x)
    return -.5*(LOG_TWO_PI + 2*tf.math.log(beta) + x + residual_square)


def transformed_observation_log_density(x, beta, transformed_observation):
    residual = transformed_observation - 2*tf.math.log(beta) - x
    return .5*(residual-tf.exp(residual)-LOG_TWO_PI)


@tf.function(input_signature=[tf.TensorSpec([None, 2], DTYPE)], jit_compile=True, autograph=False)
def chart_and_log_prior(theta):
    physical = .1+.8*(.5*tf.math.erfc(-theta/tf.sqrt(tf.constant(2., DTYPE))))
    derivative = .8*tf.exp(-.5*theta*theta-.5*LOG_TWO_PI)
    return physical, derivative, tf.reduce_sum(-.5*theta*theta-.5*LOG_TWO_PI, axis=-1)


@dataclass(frozen=True)
class SVReference:
    physical_value_score: object
    unconstrained_value_score: object
    grid: object
    weights: object
    specification: dict


def make_sv_reference(*, intervals=1024, half_width=16., horizon=20, batch_size=4):
    if type(intervals) is not int or intervals < 2 or not math.isfinite(half_width) or half_width <= 0:
        raise ValueError("positive domain and at least two grid intervals required")
    if type(horizon) is not int or horizon < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive fixed horizon/batch required")
    grid = tf.linspace(tf.constant(-half_width, DTYPE), tf.constant(half_width, DTYPE), intervals+1)
    step = 2*half_width/intervals
    weights = tf.concat([tf.constant([.5*step], DTYPE), tf.fill([intervals-1], tf.constant(step, DTYPE)),
                         tf.constant([.5*step], DTYPE)], 0)
    signature = [tf.TensorSpec([batch_size, 2], DTYPE), tf.TensorSpec([horizon], DTYPE)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def physical_value_score(theta, observations):
        gamma, beta = theta[:, 0], theta[:, 1]
        one_minus_square = 1-gamma*gamma
        initial = tf.exp(.5*tf.math.log(one_minus_square[:, None])
                         -.5*one_minus_square[:, None]*grid[None, :]**2-.5*LOG_TWO_PI)*weights[None, :]
        initial_log_derivative = gamma[:, None]*grid[None, :]**2-gamma[:, None]/one_minus_square[:, None]
        derivative = tf.stack([initial*initial_log_derivative, tf.zeros_like(initial)], -1)
        residual = grid[None, None, :]-gamma[:, None, None]*grid[None, :, None]
        transition = tf.exp(-.5*(residual*residual+LOG_TWO_PI))*weights[None, None, :]
        transition_derivative = transition*residual*grid[None, :, None]
        total = tf.zeros([batch_size], DTYPE)
        score = tf.zeros([batch_size, 2], DTYPE)
        lost = 1-tf.reduce_sum(initial, axis=1)

        def update(index, mass, derivative, total, score, lost):
            def predict():
                predicted = tf.linalg.matvec(transition, mass, transpose_a=True)
                differentiated = tf.einsum("bij,bip->bjp", transition, derivative)
                direct = tf.linalg.matvec(transition_derivative, mass, transpose_a=True)
                return predicted, differentiated+tf.stack([direct, tf.zeros_like(direct)], -1)
            predicted, d_predicted = tf.cond(index == 0, lambda: (mass, derivative), predict)
            lost = tf.maximum(lost, 1-tf.reduce_sum(predicted, axis=1))
            obs = raw_observation_log_density(grid[None, :], beta[:, None], observations[index])
            shift = tf.reduce_max(obs, axis=1)
            scaled = tf.exp(obs-shift[:, None])
            unnormalized = predicted*scaled
            obs_beta = (tf.square(observations[index]/beta[:, None])*tf.exp(-grid[None, :])-1)/beta[:, None]
            d_unnormalized = d_predicted*scaled[:, :, None] + unnormalized[:, :, None]*tf.stack(
                [tf.zeros_like(obs_beta), obs_beta], -1)
            normalization = tf.reduce_sum(unnormalized, axis=1)
            d_normalization = tf.reduce_sum(d_unnormalized, axis=1)
            mass = unnormalized/normalization[:, None]
            derivative = (d_unnormalized-mass[:, :, None]*d_normalization[:, None, :])/normalization[:, None, None]
            total += shift+tf.math.log(normalization)
            score += d_normalization/normalization[:, None]
            return index+1, mass, derivative, total, score, lost

        _, mass, derivative, total, score, lost = tf.while_loop(
            lambda index, *_: index < horizon, update,
            (tf.constant(0), initial, derivative, total, score, lost),
            parallel_iterations=1, maximum_iterations=horizon)
        return dict(log_likelihood=total, score=score, maximum_predicted_domain_loss=lost,
                    final_mass=mass, final_mass_derivative=derivative)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def unconstrained_value_score(theta, observations):
        physical, derivative, prior = chart_and_log_prior(theta)
        result = physical_value_score(physical, observations)
        return result["log_likelihood"]+prior, result["score"]*derivative-theta

    return SVReference(physical_value_score, unconstrained_value_score, grid, weights,
        dict(model="actual_sv", scope="SV20-attempt1", sigma=1., horizon=horizon, batch_size=batch_size,
             intervals=intervals, half_width=half_width, dtype="float64", jit_compile=True,
             observation_law="Normal(0,beta^2*exp(x))", stationary_initialization=True,
             transition_before_observation=True, renormalize_truncated_transition=False,
             physical_prior="independent Uniform(.1,.9)", chart=".1+.8*Phi(z)",
             posterior_reference_qualified=False, role="independent quadrature reference"))


def normal_quadrature(order):
    """Independent standard-normal Gaussian quadrature by Jacobi eigensystem."""
    off = tf.sqrt(tf.cast(tf.range(1, order), DTYPE))
    jacobi = tf.linalg.diag(off, k=1)+tf.linalg.diag(off, k=-1)
    nodes, vectors = tf.linalg.eigh(jacobi)
    return nodes, vectors[0]**2


def direct_short_likelihood(theta, observations, order=128):
    """Independent one/two-observation reference; no filtering recursion."""
    if observations.shape[0] not in (1, 2):
        raise ValueError("short Gaussian reference requires one or two observations")
    nodes, weights = normal_quadrature(order)
    gamma, beta = theta[:, 0], theta[:, 1]
    first = nodes[None, :]/tf.sqrt(1-gamma[:, None]**2)
    first_log = raw_observation_log_density(first, beta[:, None], observations[0])
    if observations.shape[0] == 1:
        return tf.reduce_logsumexp(first_log+tf.math.log(weights)[None, :], axis=1)
    second = gamma[:, None, None]*first[:, :, None]+nodes[None, None, :]
    second_log = raw_observation_log_density(second, beta[:, None, None], observations[1])
    total = first_log[:, :, None]+second_log
    return tf.reduce_logsumexp(total+tf.math.log(weights)[None, :, None]
                              +tf.math.log(weights)[None, None, :], axis=[1, 2])
