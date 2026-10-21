"""Surrogate-HMC AR2 consumer with independent Kalman reference endpoints."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.ledh_structural_ar2_target_tf import make_ar2_target
from bayesfilter.inference.neural_force_hmc import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation, TargetPotentialEvaluation,
)
from bayesfilter.testing.surrogate_hmc_ar2_reference import make_reference
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status


@dataclass(frozen=True)
class AR2Consumer:
    fixture: object
    force: object
    endpoint: object
    finite_endpoint: object
    exact_force: object
    evaluate_force: object
    evaluate_reference: object
    manifest: dict


def make_ar2_consumer(dataset, *, controls, particle_count, batch_size):
    fixture = make_ar2_target(dataset, particle_count=particle_count, controls=controls)
    _, reference, _ = make_reference(dataset["horizon"])
    observations = tf.cast(fixture.inputs["observations"], tf.float64)
    signature = [tf.TensorSpec([batch_size, 2], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_force(theta):
        return fixture.evaluate(tf.cast(theta, tf.float32))

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_reference(theta):
        return reference(theta, observations)

    def field(theta):
        _, score, status = evaluate_force(theta)
        return PositionOnlyForceEvaluation(-tf.cast(score, tf.float64), status["force_valid"])

    def potential(theta):
        logp, _ = evaluate_reference(theta)
        return TargetPotentialEvaluation(-logp, endpoint_status(logp))

    def finite_potential(theta):
        logp, _, status = evaluate_force(theta)
        return TargetPotentialEvaluation(-tf.cast(logp, tf.float64), status)

    def exact_field(theta):
        logp, score = evaluate_reference(theta)
        return PositionOnlyForceEvaluation(-score, tf.math.is_finite(logp) & tf.reduce_all(tf.math.is_finite(score), -1))

    root = Path(__file__).resolve().parents[2]
    sources = dict(fixture.manifest["source_sha256"])
    for name in ("bayesfilter/testing/surrogate_hmc_ar2_consumer.py", "bayesfilter/testing/surrogate_hmc_ar2_reference.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_tf.py"):
        sources[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest = dict(fixture.manifest, fixture_identity=fixture.identity, source_sha256=sources,
                    endpoint="independent FP64 Kalman likelihood plus normal log-scale prior",
                    force="canonical FP32 analytical finite DCTR score", batch_size=batch_size,
                    force_semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return AR2Consumer(fixture,
        FrozenPositionOnlyForce(field, identity+":DCTR", DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS),
        FrozenTargetPotential(potential, identity+":Kalman"), FrozenTargetPotential(finite_potential, identity+":finite"),
        FrozenPositionOnlyForce(exact_field, identity+":exact", DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS),
        evaluate_force, evaluate_reference, manifest)
