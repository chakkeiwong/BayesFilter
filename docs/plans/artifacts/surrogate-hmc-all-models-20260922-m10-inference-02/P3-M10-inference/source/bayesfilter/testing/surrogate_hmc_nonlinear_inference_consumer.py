"""M10 reference-endpoint consumer candidate; no inference admission by itself.

The shared NonlinearFixture.unconstrained method remains the sole DCTR value
and analytical-force implementation. Claim preparation changes observations
and their repository-issued identity, never the algorithm or original status.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation, DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status
from bayesfilter.testing.surrogate_hmc_nonlinear_consumer import (
    NonlinearConsumer, make_nonlinear_fixture, _hash_tensor)
from bayesfilter.testing.surrogate_hmc_nonlinear_log_reference import make_log_reference
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import PRIOR_SD


def make_inference_consumer(claim, calibration, *, controls, batch_size):
    if (claim.get("role"), claim.get("regime"), claim.get("horizon"), claim.get("dtype")) != ("claim", "nominal", 2, "float32"):
        raise ValueError("original nominal M10 claim dataset required")
    if (calibration.get("role"), calibration.get("regime")) != ("calibration", "nominal"):
        raise ValueError("original nominal M10 calibration dataset required for common fixed banks")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive fixed batch size required")
    # Fixed banks and numerical controls do not depend on calibration observations.
    # Bind the actual claim observations explicitly; do not relabel claim data.
    prepared = make_nonlinear_fixture(calibration, controls=controls)
    observations = tf.constant(claim["raw_observations"], tf.float32)
    tf.debugging.assert_equal(tf.shape(observations), [2, 1])
    tf.debugging.assert_all_finite(observations, "M10 claim observations")
    fixture_manifest = dict(prepared.manifest)
    fixture_manifest.update(data_role="claim", data_regime="nominal",
        control_status="caller supplies controls; launch requires repository-verified scoped selection",
        input_sha256={**fixture_manifest["input_sha256"], "observations": _hash_tensor(observations)})
    sources = dict(fixture_manifest["source_sha256"])
    root = Path(__file__).resolve().parents[2]
    for name in ("surrogate_hmc_nonlinear_inference_consumer.py", "surrogate_hmc_nonlinear_log_reference.py",
                 "surrogate_hmc_nonlinear_quadrature.py"):
        relative = "bayesfilter/testing/"+name
        sources[relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
    fixture_manifest["source_sha256"] = sources
    identity = hashlib.sha256(json.dumps(fixture_manifest, sort_keys=True).encode()).hexdigest()
    fixture = replace(prepared, observations=observations, identity=identity, manifest=MappingProxyType(fixture_manifest))
    reference = make_log_reference(with_score=False)
    signature = [tf.TensorSpec([batch_size, 2], tf.float64)]
    variance = tf.constant(PRIOR_SD**2, tf.float64)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_force(z):
        return fixture.unconstrained(tf.cast(z, tf.float32))

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_reference(z):
        def row(point):
            result = reference(point, tf.cast(observations, tf.float64))
            prior = -tf.reduce_sum(point**2)/(2*variance)-tf.math.log(2*tf.constant(3.141592653589793, tf.float64)*variance)
            return result["log_likelihood"]+prior, result["predictive"], result["valid"]
        # Independent reference evaluation, not a NeuTra training route.
        return tf.map_fn(row, z, fn_output_signature=(tf.TensorSpec([], tf.float64),
            tf.TensorSpec([2], tf.float64), tf.TensorSpec([], tf.bool)), parallel_iterations=1)

    def field(z):
        _, score, status = evaluate_force(z)
        return PositionOnlyForceEvaluation(-tf.cast(score, tf.float64), status["force_valid"])

    def potential(z):
        value, _, valid = evaluate_reference(z)
        value = tf.where(valid, value, tf.constant(float("nan"), tf.float64))
        return TargetPotentialEvaluation(-value, endpoint_status(value))

    endpoint_id = identity+":model-reference-log-quadrature-32-128-v1"
    force = FrozenPositionOnlyForce(field, identity=identity+":FP32-DCTR",
        semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    endpoint = FrozenTargetPotential(potential, identity=endpoint_id)
    manifest = dict(classification="M10_inference_consumer_candidate", fixture=fixture_manifest,
        endpoint_identity=endpoint_id, force_identity=force.identity, force_semantics=force.semantics,
        endpoint="FP64 log quadrature 32/128, full normal log-noise prior, identity coordinates",
        batch_size=batch_size, jit_compile=True, exact_model_likelihood=False,
        precision="FP64_integrator_FP32_DCTR_FP64_reference", inference_admitted=False)
    return NonlinearConsumer(fixture, force, endpoint, None, evaluate_force, evaluate_reference, manifest)
