"""Framework-free, frozen model definitions for the K0--K7 SSM campaign.

These are synthetic diagnostic hypotheses, not inference defaults. Priors are
defined on raw coordinates. Data and algorithm seeds are separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CampaignProfile:
    case: str
    target: str
    family: str
    horizon: int
    names: tuple[str, ...]
    raw_names: tuple[str, ...]
    prior_mean: tuple[float, ...]
    prior_scale: tuple[float, ...]
    rho: float = 0.6
    process_sd: float = 0.2
    observation_sd: float = 0.2
    persistence_cap: float = 0.999
    reference: str = "grid"

    def payload(self):
        return asdict(self)


_PERSISTENCE_MEAN = math.atanh(0.6 / 0.999)
_LOG_SD = math.log(0.2)
_MULTI_RAW = ("a11_raw", "a22_raw", "a33_raw", "a44_raw", "a21_raw",
              "a31_raw", "a32_raw", "a41_raw", "a42_raw", "a43_raw",
              "log_q1", "log_q2", "log_q3", "log_q4",
              "log_r1", "log_r2", "log_r3", "log_r4")
PROFILES = {
    p.target: p for p in (
        CampaignProfile("K0", "ssm_campaign_location", "location", 32,
            ("location",), ("location",), (0.,), (2.,), reference="analytic"),
        CampaignProfile("K1", "ssm_campaign_interior", "persistence_noise", 32,
            ("rho", "observation_sd"), ("rho_raw", "log_observation_sd"),
            (_PERSISTENCE_MEAN, _LOG_SD), (1., 1.)),
        CampaignProfile("K2", "ssm_campaign_near_unit", "persistence_noise", 32,
            ("rho", "observation_sd"), ("rho_raw", "log_observation_sd"),
            (_PERSISTENCE_MEAN, _LOG_SD), (1., 1.), rho=0.97),
        CampaignProfile("K3", "ssm_campaign_small_noise", "persistence_noise", 32,
            ("rho", "observation_sd"), ("rho_raw", "log_observation_sd"),
            (_PERSISTENCE_MEAN, _LOG_SD), (1., 1.), observation_sd=0.02),
        CampaignProfile("K4", "ssm_campaign_two_noises", "two_noises", 16,
            ("process_sd", "observation_sd"), ("log_process_sd", "log_observation_sd"),
            (_LOG_SD, _LOG_SD), (1., 1.)),
        CampaignProfile("K5", "ssm_campaign_two_noises_long", "two_noises", 128,
            ("process_sd", "observation_sd"), ("log_process_sd", "log_observation_sd"),
            (_LOG_SD, _LOG_SD), (1., 1.)),
        CampaignProfile("K6", "ssm_campaign_multivariate", "multivariate", 120,
            _MULTI_RAW, _MULTI_RAW, (), (), reference="unavailable"),
        CampaignProfile("K7", "ssm_campaign_nonlinear", "nonlinear", 16,
            ("rho", "beta"), ("rho", "beta"), (0.70, 0.80), (0.20, 0.20),
            rho=0.70, process_sd=0.25, observation_sd=0.30),
    )
}


def get_profile(target):
    try:
        return PROFILES[target]
    except KeyError as exc:
        raise ValueError(f"unknown campaign target: {target}") from exc


def validate_parameters(target, parameters):
    """Model edits require a new named profile rather than ignored overrides."""
    if parameters:
        raise ValueError(f"{target} is a frozen campaign profile; parameters must be empty")
    return get_profile(target)


def generate_data(target, seed):
    """TF synthetic observations; callers must establish CPU/device policy first."""
    import tensorflow as tf
    p = get_profile(target)
    if p.family == "multivariate":
        raise ValueError("K6 consumes its existing checked fixture, not generated data")
    noise = tf.random.stateless_normal([p.horizon + 1, 3], seed=seed, dtype=tf.float64)
    if p.family == "nonlinear":
        state = tf.stack((0.5 * noise[0, 0], tf.sqrt(tf.constant(0.2, tf.float64)) * noise[0, 1]))
        def step(previous, innovation):
            m = p.rho * previous[0] + p.process_sd * innovation[0]
            k = 0.55 * previous[1] + 0.8 * tf.math.tanh(m)
            return tf.stack((m, k))
        states = tf.scan(step, noise[1:], initializer=state)
        observations = tf.reduce_sum(states, axis=1) + p.observation_sd * noise[1:, 2]
    else:
        initial = p.process_sd / math.sqrt(1. - p.rho**2) * noise[0, 0]
        states = tf.scan(lambda x, z: p.rho*x + p.process_sd*z, noise[1:, 0], initializer=initial)
        observations = states + p.observation_sd * noise[1:, 1]
        if p.family == "location":
            observations += tf.constant(0.4, tf.float64)
    return observations.numpy().tolist()
