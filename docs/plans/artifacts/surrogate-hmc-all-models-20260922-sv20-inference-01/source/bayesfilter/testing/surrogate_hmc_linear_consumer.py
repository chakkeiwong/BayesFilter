"""Full canonical DCTR proposal fields with independent linear-model endpoints.

This consumer qualifies wiring and mechanics. Scope-specific control selection
and public HMC tuning are separate prerequisites for retained inference.
Controls are mandatory: a setting from another model is never a default here.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target
from bayesfilter.inference.neural_force_hmc import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation,
)
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status
from bayesfilter.testing.surrogate_hmc_linear_reference import make_linear_reference


@dataclass(frozen=True)
class LinearConsumer:
    fixture: object
    reference: object
    force: FrozenPositionOnlyForce
    endpoint: FrozenTargetPotential
    exact_force: FrozenPositionOnlyForce
    evaluate_force: object
    evaluate_reference: object
    manifest: dict


def make_linear_consumer(dataset, *, controls, batch_size, particle_count=1024):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    reference = make_linear_reference(dataset["model_id"])
    if dataset.get("horizon") != reference.specification["t"]:
        raise ValueError("dataset horizon differs from the declared model scope")
    if dataset.get("transition_before_observation") is not True:
        raise ValueError("transition-before-observation data required")
    fixture = make_linear_fixture_target(dataset, particle_count, controls, dtype=tf.float32)
    # Preserve exactly the observations used by the full DCTR consumer.
    observations = tf.cast(fixture.observations, tf.float64)
    signature = [tf.TensorSpec([batch_size, fixture.parameter_count], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_force(z):
        return fixture.unconstrained(tf.cast(z, tf.float32))

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_reference(z):
        return reference.unconstrained_value_score(z, observations)

    def field(z):
        _, score, status = evaluate_force(z)
        return PositionOnlyForceEvaluation(-tf.cast(score, tf.float64), status["force_valid"])

    def potential(z):
        logp, _ = evaluate_reference(z)
        return TargetPotentialEvaluation(-logp, endpoint_status(logp))

    def exact_field(z):
        logp, score = evaluate_reference(z)
        valid = tf.math.is_finite(logp) & tf.reduce_all(tf.math.is_finite(score), axis=-1)
        return PositionOnlyForceEvaluation(-score, valid)

    root = Path(__file__).resolve().parents[2]
    sources = dict(fixture.manifest["source_sha256"])
    for name in ("bayesfilter/testing/surrogate_hmc_linear_consumer.py",
                 "bayesfilter/testing/surrogate_hmc_linear_reference.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_tf.py"):
        sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    identity = dict(model_id=dataset["model_id"], horizon=reference.specification["t"],
        initial_law="parameter-dependent stationary Gaussian", transition_before_observation=True,
        parameterization=fixture.manifest["parameterization"], prior=fixture.manifest["priors"],
        chart="bounded_logit_uniform_width_cancelled_Jacobian_once",
        observations_sha256=hashlib.sha256(tf.io.serialize_tensor(observations).numpy()).hexdigest(),
        sources=sources, precision="FP64 matrix Kalman reference")
    endpoint_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    force = FrozenPositionOnlyForce(field, identity=fixture.identity + ":fp64-in-fp32-force",
        semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    endpoint = FrozenTargetPotential(potential, identity=endpoint_identity)
    exact = FrozenPositionOnlyForce(exact_field, identity=endpoint_identity + ":total-Kalman-gradient")
    manifest = dict(classification="full_DCTR_consumer_qualification_not_tuning_admission",
        endpoint=identity, endpoint_identity=endpoint_identity, force_identity=force.identity,
        force_semantics=force.semantics, fixture=dict(fixture.manifest), batch_size=batch_size,
        precision="FP64_integrator_FP32_DCTR_FP64_Kalman_endpoint", jit_compile=True)
    return LinearConsumer(fixture, reference, force, endpoint, exact, evaluate_force,
                          evaluate_reference, manifest)
