"""Independent conditional-innovation value and score authority for Austria T1.

This is an extension/invention diagnostic.  It evaluates the exact finite
program for ``p_theta(z0) f_theta(z1|z0) g_theta(y1|z1)`` using fixed standard
normal innovations.  It is not a Zhao-Cui transport or a trained child.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import (
    LatentPreclipSIRSSM,
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)


DTYPE = tf.float64
PARAMETER_DIM = 3
STATE_DIM = 18
REFERENCE_ID = "zhao_cui_austria_sir_t1_conditional_innovation_reference_v1"
CLASSIFICATION = "extension_or_invention"


def _tensor_hash(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    return hashlib.sha256(bytes(serialized.numpy())).hexdigest()


def _rows(value: tf.Tensor, width: int, name: str) -> tf.Tensor:
    result = tf.convert_to_tensor(value, DTYPE)
    if result.shape.rank != 2 or result.shape[1] != width:
        raise ValueError(f"{name} must have shape [sample,{width}]")
    tf.debugging.assert_all_finite(result, f"{name} must be finite")
    return result


def _theta(value: tf.Tensor) -> tf.Tensor:
    result = tf.reshape(tf.convert_to_tensor(value, DTYPE), [PARAMETER_DIM])
    tf.debugging.assert_all_finite(result, "theta must be finite")
    return result


@dataclass(frozen=True)
class ConditionalInnovationCloud:
    """Fixed base innovations and the corresponding latent T1 draw."""

    initial_noise: tf.Tensor
    transition_noise: tf.Tensor
    z0: tf.Tensor
    z1: tf.Tensor
    log_observation: tf.Tensor
    seed: int
    role: str

    def __post_init__(self) -> None:
        initial = _rows(self.initial_noise, STATE_DIM, "initial_noise")
        transition = _rows(self.transition_noise, STATE_DIM, "transition_noise")
        z0 = _rows(self.z0, STATE_DIM, "z0")
        z1 = _rows(self.z1, STATE_DIM, "z1")
        log_observation = tf.reshape(tf.convert_to_tensor(self.log_observation, DTYPE), [-1])
        if initial.shape != transition.shape or z0.shape != z1.shape:
            raise ValueError("innovation and latent clouds must have matching shapes")
        if initial.shape != z0.shape or log_observation.shape != (initial.shape[0],):
            raise ValueError("cloud fields must share a static sample count")
        if int(initial.shape[0]) < 2:
            raise ValueError("conditional innovation cloud requires at least two rows")
        tf.debugging.assert_all_finite(log_observation, "log_observation must be finite")
        if not str(self.role):
            raise ValueError("cloud role must be nonempty")
        object.__setattr__(self, "initial_noise", initial)
        object.__setattr__(self, "transition_noise", transition)
        object.__setattr__(self, "z0", z0)
        object.__setattr__(self, "z1", z1)
        object.__setattr__(self, "log_observation", log_observation)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "role", str(self.role))

    @property
    def sample_count(self) -> int:
        return int(self.initial_noise.shape[0])

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "reference_id": REFERENCE_ID,
            "classification": CLASSIFICATION,
            "role": self.role,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "initial_noise_sha256": _tensor_hash(self.initial_noise),
            "transition_noise_sha256": _tensor_hash(self.transition_noise),
            "z0_sha256": _tensor_hash(self.z0),
            "z1_sha256": _tensor_hash(self.z1),
            "log_observation_sha256": _tensor_hash(self.log_observation),
            "sampling_law": "p_0(z0) f_0(z1|z0) via fixed standard normals",
            "event_order": "z0_then_transition_to_z1_then_observe_y1",
        }


def _validate_noise(initial_noise: tf.Tensor, transition_noise: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    initial = _rows(initial_noise, STATE_DIM, "initial_noise")
    transition = _rows(transition_noise, STATE_DIM, "transition_noise")
    if initial.shape != transition.shape:
        raise ValueError("initial_noise and transition_noise must have the same shape")
    if int(initial.shape[0]) < 2:
        raise ValueError("at least two innovation rows are required")
    return initial, transition


def make_conditional_innovation_cloud(
    *,
    theta: tf.Tensor,
    sample_count: int,
    seed: int,
    role: str,
    model: LatentPreclipSIRSSM | None = None,
) -> ConditionalInnovationCloud:
    """Generate one stateless cloud and evaluate its observation likelihood."""

    n = int(sample_count)
    if n < 2:
        raise ValueError("sample_count must be at least two")
    parameters = _theta(theta)
    if not bool(tf.reduce_all(parameters == 0.0).numpy()):
        raise ValueError("conditional reference clouds must be generated at theta_ref=0")
    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    if not isinstance(active_model, LatentPreclipSIRSSM):
        raise TypeError("model must be LatentPreclipSIRSSM")
    roots = tf.random.experimental.stateless_split(tf.constant([int(seed), 1701], tf.int32), 2)
    initial_noise = tf.random.stateless_normal([n, STATE_DIM], roots[0], dtype=DTYPE)
    transition_noise = tf.random.stateless_normal([n, STATE_DIM], roots[1], dtype=DTYPE)
    scaled = active_model.physical_model.scaled_model(parameters)
    initial_chol = tf.linalg.cholesky(scaled.initial_covariance)
    z0 = scaled.initial_mean[tf.newaxis, :] + tf.linalg.matmul(
        initial_noise, initial_chol, transpose_b=True
    )
    z1 = active_model.transition_push_from_standard_normal(
        parameters, z0, transition_noise, 1
    )
    _states, observations, _all = generate_sealed_lane_b_dataset()
    log_observation = active_model.observation_log_density(
        parameters, z1, observations[0], 1
    )
    return ConditionalInnovationCloud(
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        z0=z0,
        z1=z1,
        log_observation=log_observation,
        seed=int(seed),
        role=role,
    )


def _draw_latents(
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    model: LatentPreclipSIRSSM,
) -> tuple[tf.Tensor, tf.Tensor]:
    parameters = _theta(theta)
    initial, transition = _validate_noise(initial_noise, transition_noise)
    scaled = model.physical_model.scaled_model(parameters)
    initial_chol = tf.linalg.cholesky(scaled.initial_covariance)
    z0 = scaled.initial_mean[tf.newaxis, :] + tf.linalg.matmul(
        initial, initial_chol, transpose_b=True
    )
    z1 = model.transition_push_from_standard_normal(parameters, z0, transition, 1)
    return z0, z1


def _log_observation(
    theta: tf.Tensor, z1: tf.Tensor, model: LatentPreclipSIRSSM
) -> tf.Tensor:
    _states, observations, _all = generate_sealed_lane_b_dataset()
    return model.observation_log_density(theta, z1, observations[0], 1)


def _proposal_origin_log_density(
    z0: tf.Tensor, z1: tf.Tensor, model: LatentPreclipSIRSSM
) -> tf.Tensor:
    origin = tf.zeros([PARAMETER_DIM], DTYPE)
    return model.initial_log_density(origin, z0) + model.transition_log_density(
        origin, z0, z1, 1
    )


def _target_log_density(
    theta: tf.Tensor, z0: tf.Tensor, z1: tf.Tensor, model: LatentPreclipSIRSSM
) -> tf.Tensor:
    parameters = _theta(theta)
    return (
        model.initial_log_density(parameters, z0)
        + model.transition_log_density(parameters, z0, z1, 1)
        + _log_observation(parameters, z1, model)
    )


def finite_log_value(
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    *,
    model: LatentPreclipSIRSSM | None = None,
) -> tf.Tensor:
    """Return the log of the finite innovation Monte Carlo value program."""

    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    origin = tf.zeros([PARAMETER_DIM], DTYPE)
    z0, z1 = _draw_latents(origin, initial_noise, transition_noise, active_model)
    log_ratio = _target_log_density(theta, z0, z1, active_model) - _proposal_origin_log_density(
        z0, z1, active_model
    )
    value = tf.reduce_logsumexp(log_ratio) - tf.math.log(
        tf.cast(tf.shape(log_ratio)[0], DTYPE)
    )
    tf.debugging.assert_all_finite(value, "finite innovation log value")
    return value


def finite_value_and_analytical_score(
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    *,
    model: LatentPreclipSIRSSM | None = None,
) -> Mapping[str, tf.Tensor]:
    """Evaluate value, complete-data ratio score, ESS, and score MCSE."""

    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    parameters = _theta(theta)
    origin = tf.zeros([PARAMETER_DIM], DTYPE)
    z0, z1 = _draw_latents(origin, initial_noise, transition_noise, active_model)
    proposal_log = _proposal_origin_log_density(z0, z1, active_model)
    target_log = _target_log_density(parameters, z0, z1, active_model)
    log_observation = _log_observation(parameters, z1, active_model)
    log_ratio = target_log - proposal_log
    log_value = tf.reduce_logsumexp(log_ratio) - tf.math.log(
        tf.cast(tf.shape(log_ratio)[0], DTYPE)
    )
    weights = tf.nn.softmax(log_ratio)
    ratio = tf.exp(log_ratio - tf.reduce_max(log_ratio))
    ratio_mean = tf.reduce_mean(ratio)
    ratio_standard_error = tf.math.reduce_std(ratio) / tf.sqrt(
        tf.cast(tf.shape(ratio)[0], DTYPE)
    )
    value_standard_error = ratio_standard_error / ratio_mean
    _states, observations, _all = generate_sealed_lane_b_dataset()
    complete_score = (
        active_model.initial_log_density_parameter_score(parameters, z0)
        + active_model.transition_log_density_parameter_score(parameters, z0, z1, 1)
        + active_model.observation_log_density_parameter_score(
            parameters, z1, observations[0], 1
        )
    )
    score = tf.reduce_sum(weights[:, tf.newaxis] * complete_score, axis=0)
    centered = complete_score - score[tf.newaxis, :]
    scaled_weights = tf.exp(log_ratio - tf.reduce_max(log_ratio))
    mean_scaled_weight = tf.reduce_mean(scaled_weights)
    influence = scaled_weights[:, tf.newaxis] * centered / mean_scaled_weight
    variance = tf.reduce_sum(tf.square(influence), axis=0) / tf.cast(
        tf.shape(log_observation)[0] - 1, DTYPE
    )
    standard_error = tf.sqrt(variance / tf.cast(tf.shape(log_observation)[0], DTYPE))
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))
    for name, value in (("log_value", log_value), ("score", score), ("standard_error", standard_error), ("ess", ess), ("value_standard_error", value_standard_error)):
        tf.debugging.assert_all_finite(value, name)
    return {
        "log_value": log_value,
        "score": score,
        "standard_error": standard_error,
        "value_standard_error": value_standard_error,
        "effective_sample_size": ess,
        "log_observation": log_observation,
        "complete_score": complete_score,
        "z0": z0,
        "z1": z1,
    }


def finite_value_and_autodiff_score(
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    *,
    model: LatentPreclipSIRSSM | None = None,
) -> Mapping[str, tf.Tensor]:
    """Differentiate the same finite innovation value, with fixed innovations."""

    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    parameters = tf.convert_to_tensor(theta, DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(parameters)
        value = finite_log_value(
            parameters, initial_noise, transition_noise, model=active_model
        )
    score = tape.gradient(value, parameters)
    if score is None:
        raise ValueError("innovation finite value is disconnected from theta")
    score = tf.reshape(score, [PARAMETER_DIM])
    tf.debugging.assert_all_finite(score, "autodiff innovation score")
    return {"log_value": value, "score": score}


def generate_authority_pair(
    *, sample_count: int = 8192, seed_a: int = 92001, seed_b: int = 92002
) -> Mapping[str, object]:
    """Return two independent origin estimates for the mechanics authority."""

    theta = tf.zeros([PARAMETER_DIM], DTYPE)
    rows = []
    for seed in (int(seed_a), int(seed_b)):
        cloud = make_conditional_innovation_cloud(
            theta=theta,
            sample_count=int(sample_count),
            seed=seed,
            role=f"authority_seed_{seed}",
        )
        estimate = finite_value_and_analytical_score(
            theta, cloud.initial_noise, cloud.transition_noise
        )
        autodiff = finite_value_and_autodiff_score(
            theta, cloud.initial_noise, cloud.transition_noise
        )
        rows.append(
            {
                "seed": seed,
                "cloud": cloud.manifest_payload(),
                "log_value": estimate["log_value"],
                "score": estimate["score"],
                "score_standard_error": estimate["standard_error"],
                "value_standard_error": estimate["value_standard_error"],
                "effective_sample_size": estimate["effective_sample_size"],
                "autodiff_log_value": autodiff["log_value"],
                "autodiff_score": autodiff["score"],
            }
        )
    return {
        "reference_id": REFERENCE_ID,
        "classification": CLASSIFICATION,
        "theta": theta,
        "sample_count": int(sample_count),
        "rows": tuple(rows),
        "source_observation_sha256": _tensor_hash(generate_sealed_lane_b_dataset()[1][0]),
        "nonclaims": (
            "not a Zhao-Cui source-faithful route",
            "not a trained child",
            "not full-horizon value or score",
            "not HMC evidence",
        ),
    }


__all__ = [
    "CLASSIFICATION",
    "ConditionalInnovationCloud",
    "REFERENCE_ID",
    "finite_log_value",
    "finite_value_and_analytical_score",
    "finite_value_and_autodiff_score",
    "generate_authority_pair",
    "make_conditional_innovation_cloud",
]
