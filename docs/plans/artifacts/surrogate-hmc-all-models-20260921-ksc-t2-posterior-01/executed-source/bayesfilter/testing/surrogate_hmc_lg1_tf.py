"""LG1 surrogate-HMC qualification consumer and independent TF Kalman reference.

Diagnostic until scope-specific DCTR selection and public HMC admission exist.
The DCTR force always calls the full shared canonical implementation. The
reference uses scalar Kalman recursions, with the stationary covariance and
change of variables differentiated explicitly; no reduced LEDH is implemented.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.ledh_program_fixtures_tf import fixture_seed, make_linear_fixture_target
from bayesfilter.inference.neural_force_hmc import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation,
)


def warm_start_controls() -> dict:
    """Post-invalidation full-route hypothesis; never a tuning artifact."""
    return dict(substeps=10, reset_policy="contract_e", reset_epsilon=2.,
                reset_sinkhorn_steps=8, reset_balance_steps=8, reset_ridge=1e-5,
                correction_steps=2, correction_strength=.2, correction_lm_damping=1e-3,
                correction_lm_scale_floor=1e-6, correction_trust_radius=.1,
                pairwise_steps=2, pairwise_strength=.02, pairwise_rms_cap=2.,
                coordinate_cap=16., coordinate_cap_power=8)


def make_diagnostic_dataset(*, attempt=4, partition="diagnostic", replicate=0) -> dict:
    """Small CPU diagnostic generation exception, not a training-data lane."""
    with tf.device("/CPU:0"):
        seeds = {str(p): fixture_seed("LG1", 10, partition, attempt, replicate, p) for p in (1, 2, 3)}
        x = tf.random.stateless_normal([], seeds["1"], dtype=tf.float64) / tf.sqrt(tf.constant(1-.8**2, tf.float64))
        initial = float(x)
        process = tf.random.stateless_normal([10], seeds["2"], dtype=tf.float64)
        noise = tf.random.stateless_normal([10], seeds["3"], dtype=tf.float64)
        states, observations = [], []
        for t in range(10):
            x = .8*x + process[t]
            states.append([float(x)])
            observations.append([float(x + tf.sqrt(tf.constant(.5, tf.float64))*noise[t])])
        return dict(model_id="LG1", attempt=attempt, partition=partition, replicate=replicate,
                    horizon=10, observations=observations, states=states, initial_state=[initial],
                    seeds={key: value.numpy().tolist() for key, value in seeds.items()},
                    truth=[.8], transition_before_observation=True,
                    generation="CPU TensorFlow tiny diagnostic generation; not training",
                    classification="synthetic_fixture")


def endpoint_status(value: tf.Tensor) -> dict:
    finite = tf.math.is_finite(value)
    zeros = tf.zeros(tf.shape(value), tf.int32)
    absent = -tf.ones(tf.shape(value), tf.int32)
    return dict(value_valid=finite, force_valid=finite,
                outside_declared_support=tf.zeros_like(finite),
                value_failure_code=tf.where(finite, zeros, tf.ones_like(zeros)),
                force_failure_code=tf.where(finite, zeros, tf.ones_like(zeros)),
                first_failure_time=absent, first_failure_stage=absent,
                first_failure_particle=absent, numerical_failure=~finite)


def kalman_physical(phi: tf.Tensor, observations: tf.Tensor):
    """Same-law log likelihood, total d/dphi and model predictive probability.

    P0=1/(1-phi^2); dP0=2phi/(1-phi^2)^2. Each prediction is
    a=phi*m, V=phi^2*P+1, S=V+.5, e=y-a, K=V/S.
    dK=.5*dV/S^2, dm=da+dK*e+K*de and dP=.5*dK.
    Differentiate -.5*(log S+e^2/S) before the measurement update.
    """
    phi = tf.convert_to_tensor(phi, tf.float64)
    y = tf.convert_to_tensor(observations, tf.float64)
    mean = tf.zeros_like(phi)
    cov = tf.math.reciprocal(1-phi*phi)
    dmean = tf.zeros_like(phi)
    dcov = 2*phi*cov*cov
    value, score = tf.zeros_like(phi), tf.zeros_like(phi)
    def body(i, mean, cov, dmean, dcov, value, score):
        pred = phi*mean
        dpred = mean + phi*dmean
        variance = phi*phi*cov + 1
        dvariance = 2*phi*cov + phi*phi*dcov
        innovation = y[i, 0] - pred
        dinnovation = -dpred
        total = variance + .5
        gain = variance / total
        dgain = .5*dvariance/(total*total)
        value -= .5*(tf.constant(math.log(2*math.pi), tf.float64) + tf.math.log(total) + innovation*innovation/total)
        score -= .5*(dvariance/total + 2*innovation*dinnovation/total - innovation*innovation*dvariance/(total*total))
        mean = pred + gain*innovation
        dmean = dpred + dgain*innovation + gain*dinnovation
        cov = .5*gain
        dcov = .5*dgain
        return i+1, mean, cov, dmean, dcov, value, score
    _, mean, cov, _, _, value, score = tf.while_loop(
        lambda i, *_: i < tf.shape(y)[0], body,
        (0, mean, cov, dmean, dcov, value, score), parallel_iterations=1)
    predicted_mean = phi*mean
    predicted_variance = phi*phi*cov + 1.5
    probability = .5*tf.math.erfc(-predicted_mean/tf.sqrt(2*predicted_variance))
    return value, score, probability


def kalman_unconstrained(z: tf.Tensor, observations: tf.Tensor):
    z = tf.convert_to_tensor(z, tf.float64)
    sig = tf.math.sigmoid(z[:, 0])
    phi = -.95 + 1.9*sig
    value, score, predictive = kalman_physical(phi, observations)
    # Uniform density 1/1.9 cancels the 1.9 in the Jacobian exactly.
    logp = value - tf.nn.softplus(-z[:, 0]) - tf.nn.softplus(z[:, 0])
    gradient = score*1.9*sig*(1-sig) + 1-2*sig
    return logp, gradient[:, None], predictive


@dataclass(frozen=True)
class LG1Consumer:
    fixture: object
    force: FrozenPositionOnlyForce
    endpoint: FrozenTargetPotential
    exact_force: FrozenPositionOnlyForce
    evaluate_force: object
    evaluate_reference: object
    manifest: dict


def make_lg1_consumer(dataset: dict, *, batch_size: int, particle_count=1024,
                      controls=None, jit_compile=True) -> LG1Consumer:
    if dataset.get("model_id") != "LG1":
        raise ValueError("LG1 consumer requires the LG1 statistical law")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    fixture = make_linear_fixture_target(dataset, particle_count,
                                        warm_start_controls() if controls is None else controls,
                                        dtype=tf.float32)
    # The FP64 endpoint consumes exactly the observations seen by DCTR, promoted
    # from its FP32 data representation, to avoid a different observed dataset.
    observations = tf.cast(fixture.observations, tf.float64)
    signature = [tf.TensorSpec([batch_size, 1], tf.float64)]
    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def evaluate_force(z):
        return fixture.unconstrained(tf.cast(z, tf.float32))
    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def evaluate_reference(z):
        return kalman_unconstrained(z, observations)
    def field(z):
        _, gradient, status = evaluate_force(z)
        return PositionOnlyForceEvaluation(-tf.cast(gradient, tf.float64), status["force_valid"])
    def potential(z):
        logp, _, _ = evaluate_reference(z)
        return TargetPotentialEvaluation(-logp, endpoint_status(logp))
    def exact_field(z):
        logp, gradient, _ = evaluate_reference(z)
        return PositionOnlyForceEvaluation(-gradient, tf.math.is_finite(logp) & tf.reduce_all(tf.math.is_finite(gradient), axis=-1))
    identity_payload = dict(law="LG1-stationary-Q1-R.5-transition-before-observation",
                            prior="Uniform(-.95,.95)", chart="bounded_logit_Jacobian_once",
                            observations_sha256=hashlib.sha256(tf.io.serialize_tensor(observations).numpy()).hexdigest(),
                            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    endpoint_identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    force = FrozenPositionOnlyForce(field, identity=fixture.identity + ":fp64-in-fp32-force",
                                    semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    endpoint = FrozenTargetPotential(potential, identity=endpoint_identity)
    exact = FrozenPositionOnlyForce(exact_field, identity=endpoint_identity + ":analytical-kalman-gradient")
    return LG1Consumer(fixture, force, endpoint, exact, evaluate_force, evaluate_reference,
                       dict(endpoint=identity_payload, endpoint_identity=endpoint_identity,
                            force_identity=force.identity, fixture=dict(fixture.manifest),
                            force_semantics=force.semantics, batch_size=batch_size,
                            precision_classification="FP64_INTEGRATOR_FP32_DCTR",
                            jit_compile=jit_compile,
                            classification="full_DCTR_consumer_diagnostic_not_tuning_admission"))
