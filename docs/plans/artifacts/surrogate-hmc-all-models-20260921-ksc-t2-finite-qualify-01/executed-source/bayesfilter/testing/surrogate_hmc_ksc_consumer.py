"""Full canonical DCTR proposal consumer for the repository's T2 KSC model.

Only model callbacks, fixed inputs and typed endpoint/force bindings live here.
Filtering and analytical recursion execute the shared canonical implementation.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import tensorflow as tf

from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
from bayesfilter.highdim.sv_mixture_cut4 import ksc_1998_log_chi_square_mixture
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.inference.neural_force_hmc import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS, FrozenPositionOnlyForce,
    FrozenTargetPotential, PositionOnlyForceEvaluation, TargetPotentialEvaluation,
)
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status
from bayesfilter.testing.surrogate_hmc_sv_consumer import SVConsumer, SVFixtureTarget, actual_sv_model
from bayesfilter.testing.surrogate_hmc_ksc_reference import (
    MODEL_ID, LAW, PRIOR, CHART, check_dataset, make_ksc_reference,
)


def ksc_model(dtype=tf.float32):
    """Physical gamma/beta callbacks; mixture density and its total tangent."""
    mixture = ksc_1998_log_chi_square_mixture()
    weights, means, variances = (tf.cast(v, dtype) for v in
                                (mixture.weights, mixture.means, mixture.variances))
    mean = tf.reduce_sum(weights * means)
    variance = tf.reduce_sum(weights * (variances + (means - mean)**2))
    log_constant = tf.math.log(weights) - .5 * tf.math.log(tf.constant(2 * math.pi, dtype) * variances)

    def components(theta, x, observation):
        residual = observation[0] - x[:, :1] - 2 * tf.math.log(theta[:, 1:2]) - means[None]
        return log_constant[None] - .5 * residual**2 / variances[None], residual

    def density(theta, x, observation):
        return tf.reduce_logsumexp(components(theta, x, observation)[0], axis=1)

    def tangent(theta, x, observation, dx, direction):
        terms, residual = components(theta, x, observation)
        location_score = tf.reduce_sum(tf.nn.softmax(terms, axis=1) * residual / variances[None], axis=1)
        return location_score * (dx[:, 0] + 2 * direction[:, 1] / theta[:, 1])

    # The state law, stationary initial dependence and observation Jacobians are
    # exactly the existing SV callbacks. Replace the observation law/geometry.
    return replace(actual_sv_model(dtype),
        observation_fn=lambda x: x + mean,
        parameter_observation_fn=lambda theta, x: x + 2 * tf.math.log(theta[:, 1:2]) + mean,
        observation_covariance=tf.reshape(variance, [1, 1]),
        observation_log_density_fn=density, observation_log_density_tangent_fn=tangent)


def tensor_sha(tensor):
    return hashlib.sha256(tf.io.serialize_tensor(tensor).numpy()).hexdigest()


def make_ksc_fixture(dataset, *, controls, particle_count=1024):
    observations64 = check_dataset(dataset)
    chunks = select_transport_chunks(particle_count)
    settings = dict(controls)
    if (settings.get("reset_policy") != "contract_e" or settings.get("correction_steps", 0) < 1
            or settings.get("pairwise_steps", 0) < 1 or settings.get("coordinate_cap", 0) <= 0):
        raise ValueError("complete canonical reset, GenUT and both caps required")
    role = {"calibration": 1, "validation": 2, "claim": 3, "diagnostic": 4}[dataset["partition"]]
    situation = {"nominal": 0, "stress": 1}[dataset["situation"]]
    seeds = {name: [20260921, 24000 + 100 * role + 10 * situation + i]
             for i, name in enumerate(("base", "noise", "design"))}
    n = particle_count
    with tf.device("/CPU:0"):
        base = tf.cast(tf.random.stateless_normal([n, 1], seeds["base"], dtype=tf.float64), tf.float32)
        noise = tf.cast(tf.random.stateless_normal([2, n, 1], seeds["noise"], dtype=tf.float64), tf.float32)
        design = tf.cast(tf.random.stateless_normal([n, 1], seeds["design"], dtype=tf.float64), tf.float32)
        design -= tf.reduce_mean(design, axis=0)
        valid, chol = safe_cholesky(tf.matmul(design, design, transpose_a=True) / n, "fixed_residual_design")
        if not bool(valid):
            raise ValueError("invalid KSC residual design")
        design = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(design)))
        observations = tf.cast(observations64[:, None], tf.float32)
    settings["reset_design"] = design
    root = Path(__file__).resolve().parents[2]
    sources = source_closure()
    for name in ("bayesfilter/testing/surrogate_hmc_ksc_consumer.py",
                 "bayesfilter/testing/surrogate_hmc_ksc_reference.py",
                 "bayesfilter/testing/surrogate_hmc_sv_consumer.py",
                 "bayesfilter/testing/surrogate_hmc_sv_reference.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_tf.py",
                 "bayesfilter/highdim/sv_mixture_cut4.py",
                 "bayesfilter/inference/neural_force_hmc.py"):
        sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest = dict(model_id=MODEL_ID, n=n, t=2, d=1, p=2, law=LAW, prior=PRIOR, chart=CHART,
        route="canonical_batch_fused_value_score", reset_contract_id="contract_e_chol_v1",
        derivative="analytical_recursive_all_parameter_dependencies",
        initial_law="stationary parameter-dependent initial state; transition before observation",
        control_status="hypothesis until independent scope selection", chunk_policy=chunks.__dict__,
        seeds=seeds, dtype="float32", tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        jit_compile=True, source_sha256=sources,
        settings={k: v for k, v in settings.items() if k != "reset_design"},
        dataset_sha256=hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest(),
        input_sha256={name: tensor_sha(t) for name, t in
                      (("base", base), ("noise", noise), ("design", design), ("observations", observations))},
        input_generation="tiny CPU fixed-input diagnostic exception; no training")
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return SVFixtureTarget(MODEL_ID, ksc_model(), base, tf.ones([n, 1, 1], tf.float32),
        noise, observations, tf.constant([.1, .1]), tf.constant([.9, .9]),
        MappingProxyType(settings), identity, MappingProxyType(manifest))


def make_ksc_consumer(dataset, *, controls, batch_size, particle_count=1024):
    fixture = make_ksc_fixture(dataset, controls=controls, particle_count=particle_count)
    reference = make_ksc_reference(dataset, batch_size=batch_size)

    @tf.function(input_signature=[tf.TensorSpec([batch_size, 2], tf.float64)], jit_compile=True, autograph=False)
    def evaluate_force(q):
        return fixture.unconstrained(tf.cast(q, tf.float32))

    def field(q):
        _, score, status = evaluate_force(q)
        return PositionOnlyForceEvaluation(-tf.cast(score, tf.float64), status["force_valid"])

    def potential(q):
        value, _ = reference.unconstrained_value_score(q)
        return TargetPotentialEvaluation(-value, endpoint_status(value))

    def exact_field(q):
        value, score = reference.unconstrained_value_score(q)
        valid = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score), axis=-1)
        return PositionOnlyForceEvaluation(-score, valid)

    identity = dict(reference=reference.specification, sources=dict(fixture.manifest["source_sha256"]),
                    data_sha256=fixture.manifest["dataset_sha256"], prior=PRIOR, chart=CHART)
    endpoint_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    force = FrozenPositionOnlyForce(field, identity=fixture.identity + ":fp64-integrator-fp32-field",
                                    semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    endpoint = FrozenTargetPotential(potential, identity=endpoint_id)
    exact = FrozenPositionOnlyForce(exact_field, identity=endpoint_id + ":exact-reference-score")
    manifest = dict(classification="KSC_T2_full_DCTR_consumer_not_tuning_admission",
                    endpoint=identity, endpoint_identity=endpoint_id, force_identity=force.identity,
                    force_semantics=force.semantics, fixture=dict(fixture.manifest), batch_size=batch_size,
                    precision="FP64_integrator_FP32_DCTR_FP64_exact_mixture_endpoint", jit_compile=True)
    return SVConsumer(fixture, reference, force, endpoint, exact, evaluate_force,
                       reference.unconstrained_value_score, manifest)
