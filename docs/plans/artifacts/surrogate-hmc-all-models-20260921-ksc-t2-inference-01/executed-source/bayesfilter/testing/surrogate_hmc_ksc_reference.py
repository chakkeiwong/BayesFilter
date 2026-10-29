"""Exact T2 repository KSC model reference for surrogate-HMC comparisons.

The 49 Gaussian components integrate the latent states exactly. This diagnostic
reference is distinct from a moment-collapsed UKF and from actual SV. Parameter
posterior integration still needs its own numerical qualification.
"""
from dataclasses import dataclass
import math

import tensorflow as tf

from bayesfilter.highdim.sv_mixture_cut4 import ksc_1998_log_chi_square_mixture


MODEL_ID = "M03-KSC-T2"
LAW = "stationary Gaussian AR1; seven-component KSC transformed observation"
PRIOR = "independent Uniform(.1,.9)"
CHART = "theta=.1+.8*Phi(q); physical prior plus Jacobian once"
FUNCTION_NAMES = (
    "standardized_gamma", "standardized_beta", "centered_gamma_second_moment",
    "centered_beta_second_moment", "standardized_cross_moment",
    "gamma_cdf_q10", "gamma_cdf_q50", "gamma_cdf_q90",
    "beta_cdf_q10", "beta_cdf_q50", "beta_cdf_q90",
    "predictive_z3_le_minus2", "predictive_z3_le_zero",
)
MARGINS = (.1,) * 5 + (.02,) * 6 + (.03,) * 2
# Validate the repository constants once, before graph tracing. Its constructor
# performs eager configuration checks and is not a numerical kernel.
_MIXTURE = ksc_1998_log_chi_square_mixture()


def make_ksc_dataset(*, partition, situation="nominal"):
    """Tiny CPU fixed-data generation exception, without learned training."""
    roles = {"calibration": 1, "validation": 2, "claim": 3, "diagnostic": 4}
    situations = {"nominal": 0, "stress": 1}
    if partition not in roles or situation not in situations:
        raise ValueError("unknown KSC data role or situation")
    seeds = {name: [20260921, 23000 + 100 * roles[partition]
                   + 10 * situations[situation] + i]
             for i, name in enumerate(("initial", "process", "component", "observation"))}
    truth = [.5, .5] if situation == "nominal" else [.88, .12]
    with tf.device("/CPU:0"):
        mixture = ksc_1998_log_chi_square_mixture()
        gamma, beta = (tf.constant(v, tf.float64) for v in truth)
        x0 = tf.random.stateless_normal([], seeds["initial"], dtype=tf.float64) / tf.sqrt(1 - gamma**2)
        process = tf.random.stateless_normal([2], seeds["process"], dtype=tf.float64)
        noise = tf.random.stateless_normal([2], seeds["observation"], dtype=tf.float64)
        components = tf.random.stateless_categorical(tf.math.log(mixture.weights)[None], 2,
                                                     seeds["component"])[0]
        x1 = gamma * x0 + process[0]
        x2 = gamma * x1 + process[1]
        states = tf.stack([x1, x2])
        observations = (states + 2 * tf.math.log(beta) + tf.gather(mixture.means, components)
                        + tf.sqrt(tf.gather(mixture.variances, components)) * noise)
        return dict(model_id=MODEL_ID, horizon=2, attempt=1, replicate=0,
                    partition=partition, situation=situation, law=LAW, prior=PRIOR, chart=CHART,
                    observations=tf.cast(observations, tf.float32).numpy().tolist(),
                    observation_dtype="float32", observation_coordinates="transformed z, generated directly",
                    raw_transform_applied=False, truth=truth, states=states.numpy().tolist(),
                    initial_state=float(x0), component_labels=components.numpy().tolist(), seeds=seeds,
                    transition_before_observation=True,
                    generation="CPU TensorFlow tiny fixed-input diagnostic exception; no training")


def check_dataset(dataset):
    if (dataset.get("model_id"), dataset.get("horizon"), dataset.get("attempt"),
        dataset.get("replicate"), dataset.get("law"), dataset.get("prior"),
        dataset.get("chart"), dataset.get("observation_dtype"), dataset.get("raw_transform_applied")) != (
            MODEL_ID, 2, 1, 0, LAW, PRIOR, CHART, "float32", False):
        raise ValueError("wrong KSC scope, law, prior, chart or data coordinates")
    if dataset.get("partition") not in ("calibration", "validation", "claim", "diagnostic"):
        raise ValueError("unknown data partition")
    if dataset.get("situation") not in ("nominal", "stress"):
        raise ValueError("unknown data situation")
    observations = tf.constant(dataset["observations"], tf.float32)
    if (observations.shape != [2] or not bool(tf.reduce_all(tf.math.is_finite(observations)))
            or observations.numpy().tolist() != dataset["observations"]):
        raise ValueError("two finite once-rounded FP32 observations required")
    return tf.cast(observations, tf.float64)


def physical_parameters(q):
    q = tf.convert_to_tensor(q, tf.float64)
    return tf.constant(.1, tf.float64) + tf.constant(.8, tf.float64) * .5 * tf.math.erfc(
        -q / tf.sqrt(tf.constant(2., tf.float64)))


def component_statistics(theta, observations):
    """Batched exact likelihood, mixture weights and conditional x2 moments."""
    theta = tf.convert_to_tensor(theta, tf.float64)
    mixture = _MIXTURE
    first = tf.repeat(tf.range(7), 7)
    second = tf.tile(tf.range(7), [7])
    gamma, beta = theta[:, :1], theta[:, 1:2]
    stationary = 1 / (1 - gamma**2)
    cross = gamma * stationary
    a = stationary + tf.gather(mixture.variances, first)[None]
    b = stationary + tf.gather(mixture.variances, second)[None]
    determinant = a * b - cross**2
    r1 = observations[0] - 2 * tf.math.log(beta) - tf.gather(mixture.means, first)[None]
    r2 = observations[1] - 2 * tf.math.log(beta) - tf.gather(mixture.means, second)[None]
    inv_r1 = (b * r1 - cross * r2) / determinant
    inv_r2 = (a * r2 - cross * r1) / determinant
    log_weights = tf.math.log(tf.gather(mixture.weights, first) * tf.gather(mixture.weights, second))
    terms = log_weights[None] - .5 * (
        tf.math.log(determinant) + tf.constant(2 * math.log(2 * math.pi), tf.float64)
        + r1 * inv_r1 + r2 * inv_r2)
    likelihood = tf.reduce_logsumexp(terms, axis=1)
    weights = tf.nn.softmax(terms, axis=1)
    conditional_mean = cross * inv_r1 + stationary * inv_r2
    # Schur complement; do not clip a negative variance into apparent validity.
    conditional_variance = stationary - (
        b * cross**2 - 2 * cross**2 * stationary + a * stationary**2) / determinant
    return likelihood, weights, conditional_mean, conditional_variance


def predictive_probabilities(theta, observations):
    likelihood, weights, mean, variance = component_statistics(theta, observations)
    mixture = _MIXTURE
    gamma, beta = theta[:, :1], theta[:, 1:2]
    next_mean = (gamma * mean + 2 * tf.math.log(beta))[:, :, None] + mixture.means[None, None]
    next_variance = (gamma**2 * variance + 1)[:, :, None] + mixture.variances[None, None]
    cuts = tf.constant([-2., 0.], tf.float64)
    probabilities = .5 * tf.math.erfc(
        -(cuts[None, None, None] - next_mean[:, :, :, None])
        / tf.sqrt(2 * next_variance[:, :, :, None]))
    predictive = tf.reduce_sum(probabilities * weights[:, :, None, None]
                               * mixture.weights[None, None, :, None], axis=[1, 2])
    return likelihood, predictive


@dataclass(frozen=True)
class KSCReference:
    physical_value_score: object
    unconstrained_value_score: object
    statistics: object
    specification: dict


def make_ksc_reference(dataset, *, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive fixed reference batch size required")
    observations = check_dataset(dataset)
    signature = [tf.TensorSpec([batch_size, 2], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def physical_value_score(theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            likelihood = component_statistics(theta, observations)[0]
        return likelihood, tape.gradient(likelihood, theta)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def unconstrained_value_score(q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            likelihood = component_statistics(physical_parameters(q), observations)[0]
            value = likelihood - .5 * tf.reduce_sum(q**2, axis=1) - tf.constant(math.log(2 * math.pi), tf.float64)
        return value, tape.gradient(value, q)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def statistics(theta):
        return predictive_probabilities(theta, observations)

    return KSCReference(physical_value_score, unconstrained_value_score, statistics,
                        dict(model=MODEL_ID, method="exact 49-component Gaussian latent marginalization",
                             dtype="float64", jit_compile=True, batch_size=batch_size,
                             observation_law=LAW, prior=PRIOR, chart=CHART,
                             latent_grid_truncation=False, posterior_integration_qualified=False))
