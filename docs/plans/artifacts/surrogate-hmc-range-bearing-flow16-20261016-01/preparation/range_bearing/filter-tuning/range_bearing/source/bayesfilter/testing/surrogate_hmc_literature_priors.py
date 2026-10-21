"""Proper priors for the five-model surrogate-HMC campaign, with analytic scores.

Source settings, adaptations and derivations are documented in
docs/papers/surrogate-hmc-priors-2026-10-15/README.md. These are priors for the
existing benchmark likelihoods, not replications of all five source models.
All log densities below are with respect to the sampled q coordinates.
"""
from __future__ import annotations

import math
import tensorflow as tf
import tensorflow_probability as tfp


PRIOR_ID = "literature-model-priors-v1"
BENCHMARK_PRIOR_ID = "benchmark-normal-v1"
MODELS = ("lgssm", "ksc", "predator_prey", "sir_d18", "range_bearing")
# Coordinate scales are inherited preconditioning hypotheses, not prior widths.
CHART_SCALES = {"lgssm": (.4, .5, .5, .2), "ksc": (.5, .5),
                "predator_prey": (1.,) * 6, "sir_d18": (.35,) * 3,
                "range_bearing": (math.log(2.),) * 2}
PP_LOWER = (.1, 110., 20., .1, 0., 0.)
PP_UPPER = (1.1, 130., 30., 1.1, 1., 1.)
SIR_DAYS_PER_BENCHMARK_UNIT = 45.
SIR_POPULATION_PER_REGION = 500.


def specification(name):
    """Serializable settings issued by the same module as the density callable."""
    if name not in MODELS:
        raise ValueError("unknown literature-prior model: " + name)
    common = dict(id=PRIOR_ID, coordinates="q", normalized=True,
        chart="bounded_logistic" if name == "predator_prey" else "affine",
        chart_scales=list(CHART_SCALES[name]),
        jacobian="physical-to-theta and theta-to-q Jacobians included",
        evidence="docs/papers/surrogate-hmc-priors-2026-10-15/README.md")
    settings = {
        "predator_prey": dict(source="Zhao-Cui 2024 section 6.4",
            parameter_order=["r", "K", "a", "s", "u", "v"],
            independent_uniform_lower=list(PP_LOWER), independent_uniform_upper=list(PP_UPPER),
            adaptation="published physical prior unchanged; logistic sampling chart"),
        "ksc": dict(source="Zhao-Cui 2024 section 6.2 empirical SV",
            gamma="Phi(theta[0])", beta="exp(theta[1])", innovation_variance=1.,
            beta_prior_on_gamma_plus_one_over_two=[20., 1.5],
            gamma_conditioning_interval=[0., 1.], log_beta_normal_mean=0.,
            log_beta_normal_variance=1.25,
            adaptation="condition on sigma^2=1 and gamma>0; preserve current likelihood/initial law"),
        "lgssm": dict(source="Dahlin-Schon 2019 sections 4,5; Sarkka-Nummenmaa 2009 section III",
            phi="0.55*tanh(theta[0])", phi_normal_mean=0., phi_normal_variance=.5,
            phi_conditioning_interval=[-.55, .55],
            process_variance_multiplier="exp(theta[1])",
            observation_variance_multiplier="exp(theta[2])",
            inverse_gamma_shape=1., inverse_gamma_scale=1.,
            initial_mean_amplitude="theta[3]", mean_normal_mean=0., mean_normal_variance=1.,
            adaptation="restricted AR prior; IG family also applied to process scale; location prior transferred"),
        "sir_d18": dict(source="Grinsztajn et al. 2021 section 4.3; Sarkka-Nummenmaa 2009 section III",
            days_per_benchmark_time_unit=SIR_DAYS_PER_BENCHMARK_UNIT,
            population_per_region=SIR_POPULATION_PER_REGION,
            infection_per_day="500*0.1*exp(theta[0])/45",
            recovery_per_day="18*exp(theta[1])/45",
            infection_positive_normal_mean_sd=[2., 1.], recovery_positive_normal_mean_sd=[.4, .5],
            observation_variance_multiplier="exp(2*theta[2])",
            inverse_gamma_shape=1., inverse_gamma_scale=1.,
            adaptation="declared synthetic time calibration; Gaussian variance prior replaces inapplicable NB dispersion prior"),
        "range_bearing": dict(source="Sarkka-Nummenmaa 2009 section III; Sarkka-Hartikainen 2013 sections 2,3",
            process_variance_multiplier="exp(2*theta[0])",
            observation_variance_multiplier="exp(2*theta[1])",
            inverse_gamma_shape=1., inverse_gamma_scale=1.,
            adaptation="independent static IG priors on common Q/R multipliers, preserving fixed covariance shapes"),
    }
    return dict(common, **settings[name])


def chart(name, q, theta0):
    """Map q to model theta and return the diagonal derivative dtheta/dq."""
    center = tf.cast(theta0, q.dtype)
    if name == "predator_prey":
        lower, upper = tf.constant(PP_LOWER, q.dtype), tf.constant(PP_UPPER, q.dtype)
        z0 = tf.math.log(center - lower) - tf.math.log(upper - center)
        z = z0 + q
        fraction = tf.math.sigmoid(z)
        return lower + (upper - lower) * fraction, (upper - lower) * fraction * tf.math.sigmoid(-z)
    scales = tf.constant(CHART_SCALES[name], q.dtype)
    return center + scales * q, scales


def _normal_log_prob(x, mean, sd):
    return -.5 * tf.square((x - mean) / sd) - math.log(sd) - .5 * math.log(2. * math.pi)


def _log_variance_ig11(theta, exponent):
    """IG(shape=1,scale=1) on v=exp(exponent*theta), including dv/dtheta."""
    inverse_variance = tf.exp(-exponent * theta)
    return math.log(exponent) - exponent * theta - inverse_variance, exponent * (inverse_variance - 1.)


def log_prob_and_score(name, q, theta0):
    """Return normalized log p(q) and its analytic gradient; any leading batch."""
    theta, _ = chart(name, q, theta0)
    if name == "predator_prey":
        lower, upper = tf.constant(PP_LOWER, q.dtype), tf.constant(PP_UPPER, q.dtype)
        center = tf.cast(theta0, q.dtype)
        z = tf.math.log(center - lower) - tf.math.log(upper - center) + q
        # Uniform box widths cancel the logistic chart's widths exactly.
        terms = -tf.nn.softplus(-z) - tf.nn.softplus(z)
        return tf.reduce_sum(terms, axis=-1), 1. - 2. * tf.math.sigmoid(z)
    scales = tf.constant(CHART_SCALES[name], q.dtype)
    if name == "lgssm":
        t = tf.math.tanh(theta[..., 0])
        phi = .55 * t
        log_dphi = math.log(.55) + 2. * (math.log(2.) + theta[..., 0] - tf.nn.softplus(2. * theta[..., 0]))
        dphi = tf.exp(log_dphi)
        phi_term = _normal_log_prob(phi, 0., math.sqrt(.5)) - math.log(math.erf(.55)) + log_dphi
        q_term, q_score = _log_variance_ig11(theta[..., 1], 1.)
        r_term, r_score = _log_variance_ig11(theta[..., 2], 1.)
        terms = tf.stack([phi_term, q_term, r_term, _normal_log_prob(theta[..., 3], 0., 1.)], axis=-1)
        scores = tf.stack([-phi * dphi / .5 - 2. * t, q_score, r_score, -theta[..., 3]], axis=-1)
    elif name == "ksc":
        z, log_beta = theta[..., 0], theta[..., 1]
        normal = tfp.distributions.Normal(tf.zeros([], q.dtype), tf.ones([], q.dtype))
        log_pdf = normal.log_prob(z)
        gamma = normal.cdf(z)
        log_sf = normal.log_survival_function(z)
        log_left_mass = tf.math.log1p(-tf.math.betainc(
            tf.constant(20., q.dtype), tf.constant(1.5, q.dtype), tf.constant(.5, q.dtype)))
        log_beta_normalizer = math.lgamma(20.) + math.lgamma(1.5) - math.lgamma(21.5)
        gamma_term = (19. * (tf.math.log1p(gamma) - math.log(2.))
            + .5 * (log_sf - math.log(2.)) - log_beta_normalizer
            - math.log(2.) - log_left_mass + log_pdf)
        gamma_score = 19. * tf.exp(log_pdf) / (1. + gamma) - .5 * tf.exp(log_pdf - log_sf) - z
        terms = tf.stack([gamma_term, _normal_log_prob(log_beta, 0., math.sqrt(1.25))], axis=-1)
        scores = tf.stack([gamma_score, -log_beta / 1.25], axis=-1)
    elif name == "sir_d18":
        # Source beta*S*I/N equals benchmark kappa*S*I after changing time units.
        beta = (SIR_POPULATION_PER_REGION * .1 / SIR_DAYS_PER_BENCHMARK_UNIT) * tf.exp(theta[..., 0])
        gamma = (18. / SIR_DAYS_PER_BENCHMARK_UNIT) * tf.exp(theta[..., 1])
        terms, scores = [], []
        for x, mu, sd in ((beta, 2., 1.), (gamma, .4, .5)):
            log_mass = math.log(.5 * (1. + math.erf(mu / (sd * math.sqrt(2.)))))
            terms.append(_normal_log_prob(x, mu, sd) - log_mass + tf.math.log(x))
            scores.append(1. - x * (x - mu) / (sd * sd))
        noise_term, noise_score = _log_variance_ig11(theta[..., 2], 2.)
        terms = tf.stack([*terms, noise_term], axis=-1)
        scores = tf.stack([*scores, noise_score], axis=-1)
    elif name == "range_bearing":
        terms, scores = _log_variance_ig11(theta, 2.)
    else:
        raise ValueError("unknown literature-prior model: " + name)
    return tf.reduce_sum(terms + tf.math.log(scales), axis=-1), scores * scales
