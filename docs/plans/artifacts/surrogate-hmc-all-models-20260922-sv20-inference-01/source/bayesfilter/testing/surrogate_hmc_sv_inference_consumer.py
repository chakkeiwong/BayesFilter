"""Claim-data actual-SV endpoint with explicit quadrature validity.

The proposal field remains the shared canonical analytical DCTR consumer.
The endpoint retains, and can fail on, the reference's computed domain loss.
This qualification candidate does not issue HMC tuning or posterior admission.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation)
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status
from bayesfilter.testing.surrogate_hmc_sv_consumer import SVConsumer, make_sv_consumer
from bayesfilter.testing.surrogate_hmc_sv_reference import chart_and_log_prior

DOMAIN_LOSS_LIMIT = 1e-9


def reference_validity(log_density, score, domain_loss):
    """Numerical validity, never mathematical prior-support exclusion."""
    return (tf.math.is_finite(log_density)
            & tf.reduce_all(tf.math.is_finite(score), axis=-1)
            & tf.math.is_finite(domain_loss)
            & (domain_loss <= tf.constant(DOMAIN_LOSS_LIMIT, tf.float64)))


@dataclass(frozen=True)
class SVInferenceConsumer(SVConsumer):
    evaluate_reference_diagnostics: object


def make_sv_inference_consumer(dataset, *, controls, batch_size=4):
    if (dataset.get("partition"), dataset.get("attempt"), dataset.get("replicate")) != ("claim", 1, 0):
        raise ValueError("original SV20 attempt1 claim data required")
    base = make_sv_consumer(dataset, controls=controls, batch_size=batch_size, particle_count=1024)
    observations = tf.cast(tf.constant(dataset["raw_observations"], tf.float32), tf.float64)
    signature = [tf.TensorSpec([batch_size, 2], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate(z):
        physical, derivative, prior = chart_and_log_prior(z)
        result = base.reference.physical_value_score(physical, observations)
        logp = result["log_likelihood"] + prior
        score = result["score"] * derivative - z
        loss = result["maximum_predicted_domain_loss"]
        return logp, score, loss, reference_validity(logp, score, loss)

    identity_payload = dict(base_endpoint=base.endpoint.identity,
        domain_loss_limit=DOMAIN_LOSS_LIMIT,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        classification="actual_SV20_claim_reference_endpoint_with_domain_guard")
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()

    def potential(z):
        logp, _, _, valid = evaluate(z)
        checked = tf.where(valid, logp, tf.constant(float("nan"), tf.float64))
        return TargetPotentialEvaluation(-checked, endpoint_status(checked))

    def reference_field(z):
        _, score, _, valid = evaluate(z)
        return PositionOnlyForceEvaluation(-score, valid)

    manifest = dict(base.manifest, endpoint_identity=identity,
        endpoint_validity=identity_payload, data_partition="claim",
        classification="actual_SV20_claim_consumer_candidate_not_tuning_admission")
    return SVInferenceConsumer(**dict(base.__dict__,
        endpoint=FrozenTargetPotential(potential, identity=identity),
        exact_force=FrozenPositionOnlyForce(reference_field, identity=identity+":reference-total-gradient"),
        manifest=manifest), evaluate_reference_diagnostics=evaluate)
