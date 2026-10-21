"""M10 full canonical DCTR with a locally checked finite-grid HMC endpoint.

This diagnostic consumer neither truncates the normal prior nor admits an
endpoint for retained inference. Only model callbacks and prior wiring are
implemented here; every DCTR stage calls the shared canonical engine.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel, canonical_batch_fused_value_score)
from bayesfilter.highdim.ledh_failure_status_tf import FailureCode, FailureStage
from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.inference.neural_force_hmc import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation)
from bayesfilter.score_study.gaussian_tf import parameterized_model
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import (
    BASE_THETA, TRANSITION_CURVE, OBSERVATION_CURVE, PRIOR_SD, grid_reference, log_prior)


def nonlinear_noise_model(dtype=tf.float32):
    a, _, h, _, m, _, p, _, q, _, r, _ = parameterized_model(tf.constant(BASE_THETA, dtype), 1, 1)
    c, b = tf.cast(TRANSITION_CURVE, dtype), tf.cast(OBSERVATION_CURVE, dtype)
    observation = lambda x: h[0, 0]*x+b*x*x
    tangent = lambda x, dx: (h[0, 0]+2*b*x)*dx
    return PerPointScoreModel(
        transition_mean_fn=lambda theta, x: a[0, 0]*x+c*tf.sin(x),
        transition_mean_tangent_fn=lambda theta, x, dx, direction: (a[0, 0]+c*tf.cos(x))*dx,
        observation_fn=observation,
        observation_jacobian_fn=lambda x: (h[0, 0]+2*b*x)[:, :, None],
        observation_tangent_fn=tangent,
        observation_jacobian_tangent_fn=lambda x, dx: (2*b*dx)[:, :, None],
        process_covariance=q, observation_covariance=r,
        process_covariance_fn=lambda theta: q*tf.exp(2*theta[0]),
        process_covariance_tangent_fn=lambda theta, direction: 2*q*tf.exp(2*theta[0])*direction[0],
        observation_covariance_fn=lambda theta: r*tf.exp(2*theta[1]),
        observation_covariance_tangent_fn=lambda theta, direction: 2*r*tf.exp(2*theta[1])*direction[1],
        initial_distribution_fn=lambda theta: (m, p),
        initial_distribution_tangent_fn=lambda theta, direction: (tf.zeros_like(m), tf.zeros_like(p)))


@dataclass(frozen=True)
class NonlinearFixture:
    model: object
    initial_base: object
    initial_covariances: object
    noises: object
    observations: object
    controls: object
    identity: str
    manifest: object

    def unconstrained(self, z, *, filter_moments=False):
        z = tf.convert_to_tensor(z, tf.float32)
        finite = tf.reduce_all(tf.math.is_finite(z), -1)
        safe = tf.where(finite[:, None], z, tf.zeros_like(z))
        directions = tf.broadcast_to(tf.eye(2, dtype=z.dtype)[None], [tf.shape(z)[0], 2, 2])
        value, score, status = canonical_batch_fused_value_score(
            self.model, safe, directions, self.initial_base, self.initial_covariances,
            self.noises, self.observations, return_filter_moments=filter_moments,
            **dict(self.controls))
        variance = tf.constant(PRIOR_SD**2, z.dtype)
        prior = -tf.reduce_sum(safe**2, -1)/(2*variance)-tf.math.log(2*math.pi*variance)
        score = score-safe/variance
        status = dict(status)
        status['outside_declared_support'] = tf.zeros_like(finite)
        value_valid = finite & tf.math.is_finite(value+prior)
        force_valid = value_valid & tf.reduce_all(tf.math.is_finite(score), -1)
        # Preserve canonical failures, and describe failures introduced by the
        # prior/coordinate boundary even when the supplied coordinates are finite.
        status['value_failure_code'] = tf.where(status['value_valid'] & ~value_valid,
            int(FailureCode.NONFINITE_INPUT), status['value_failure_code'])
        status['force_failure_code'] = tf.where(status['force_valid'] & ~force_valid,
            tf.where(value_valid, int(FailureCode.INVALID_TANGENT), int(FailureCode.NONFINITE_INPUT)),
            status['force_failure_code'])
        first = (status['first_failure_stage'] < 0) & ~force_valid
        status['first_failure_stage'] = tf.where(first, int(FailureStage.PRIOR), status['first_failure_stage'])
        status['value_valid'] &= value_valid
        status['force_valid'] &= force_valid & status['value_valid']
        status['program_valid'] = status['value_valid'] & status['force_valid']
        status['direction_value_valid'] &= status['value_valid'][:, None]
        status['direction_force_valid'] &= status['force_valid'][:, None]
        return (tf.where(status['value_valid'], value+prior, tf.constant(-float('inf'), z.dtype)),
                tf.where(status['force_valid'][:, None], score, tf.zeros_like(score)), status)

    def compile(self, batch_size, *, filter_moments=False):
        return tf.function(lambda z: self.unconstrained(z, filter_moments=filter_moments),
                           input_signature=[tf.TensorSpec([batch_size, 2], tf.float32)],
                           jit_compile=True, autograph=False)


def _hash_tensor(tensor):
    return hashlib.sha256(tf.io.serialize_tensor(tensor).numpy()).hexdigest()


def make_nonlinear_fixture(dataset, *, controls, particle_count=1024):
    if ((dataset.get('horizon'), dataset.get('dtype')) != (2, 'float32')
            or dataset.get('role') not in ('calibration', 'validation')
            or dataset.get('regime') not in ('nominal', 'stress')):
        raise ValueError('M10 fixture requires frozen calibration or validation data')
    chunks = select_transport_chunks(particle_count)
    settings = dict(controls)
    if (settings.get('reset_policy') != 'contract_e' or settings.get('correction_steps', 0) < 1
            or settings.get('pairwise_steps', 0) < 1 or settings.get('coordinate_cap', 0) <= 0):
        raise ValueError('complete canonical reset and correction required')
    seeds = {str(i): [20260920, 16200+i] for i in (20, 21, 22)}
    with tf.device('/CPU:0'):
        base = tf.cast(tf.random.stateless_normal([particle_count, 1], seeds['20'], dtype=tf.float64), tf.float32)
        noises = tf.cast(tf.random.stateless_normal([2, particle_count, 1], seeds['21'], dtype=tf.float64), tf.float32)
        design = tf.cast(tf.random.stateless_normal([particle_count, 1], seeds['22'], dtype=tf.float64), tf.float32)
    observations = tf.constant(dataset['raw_observations'], tf.float32)
    tf.debugging.assert_equal(tf.shape(observations), [2, 1])
    tf.debugging.assert_all_finite(observations, 'M10 observations')
    settings['reset_design'] = design
    sources = source_closure()
    root = Path(__file__).resolve().parents[2]
    for name in ('bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py',
                 'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
                 'bayesfilter/score_study/gaussian_tf.py', 'bayesfilter/score_study/nonlinear_tf.py'):
        sources[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest = dict(model='M10-T2', particle_count=particle_count, chunk_policy=chunks.__dict__,
        data_role=dataset['role'], data_regime=dataset['regime'],
        prior='independent Normal(0,log(2)^2)', chart='identity; no Jacobian',
        settings={k: v for k, v in settings.items() if k != 'reset_design'}, seeds=seeds,
        dtype='float32', tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()), jit_compile=True,
        control_status='unselected M10 hypothesis; no tuning admission',
        source_sha256=sources, input_generation='tiny CPU fixed-input diagnostic exception',
        input_sha256=dict(base=_hash_tensor(base), noises=_hash_tensor(noises),
                          observations=_hash_tensor(observations), design=_hash_tensor(design)))
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    model = nonlinear_noise_model()
    _, initial_covariance = model.initial_distribution_fn(tf.zeros([2], tf.float32))
    return NonlinearFixture(model, base, tf.broadcast_to(initial_covariance, [particle_count, 1, 1]),
        noises, observations, MappingProxyType(settings), identity, MappingProxyType(manifest))


@dataclass(frozen=True)
class NonlinearConsumer:
    fixture: object
    force: object
    endpoint: object
    reference_force: object
    evaluate_force: object
    evaluate_reference: object
    manifest: dict


def make_nonlinear_consumer(dataset, *, controls, batch_size, particle_count=1024):
    if (dataset.get('role'), dataset.get('regime')) != ('calibration', 'nominal'):
        raise ValueError('M10 local mechanics endpoint requires nominal calibration data')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive integer batch size required')
    fixture = make_nonlinear_fixture(dataset, controls=controls, particle_count=particle_count)
    observations = tf.cast(fixture.observations, tf.float64)
    grid = grid_reference(513, 9.)
    signature = [tf.TensorSpec([batch_size, 2], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_force(z):
        return fixture.unconstrained(tf.cast(z, tf.float32))

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_reference(z):
        def row(point):
            result = grid(point, observations)
            prior, score = log_prior(point)
            return result['log_likelihood']+prior, result['score']+score
        # Independent reference lane only; this is not NeuTra training.
        return tf.map_fn(row, z, fn_output_signature=(tf.TensorSpec([], tf.float64),
            tf.TensorSpec([2], tf.float64)), parallel_iterations=1)

    def field(z):
        _, score, status = evaluate_force(z)
        return PositionOnlyForceEvaluation(-tf.cast(score, tf.float64), status['force_valid'])

    def locally_qualified(z):
        return tf.reduce_all(tf.math.is_finite(z) & (tf.abs(z) <= tf.constant(PRIOR_SD, tf.float64)), -1)

    def potential(z):
        value, _ = evaluate_reference(z)
        qualified_value = tf.where(locally_qualified(z), value, tf.constant(float('nan'), tf.float64))
        return TargetPotentialEvaluation(-qualified_value, endpoint_status(qualified_value))

    def reference_field(z):
        value, score = evaluate_reference(z)
        return PositionOnlyForceEvaluation(-score,
            locally_qualified(z) & tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score), -1))

    identity = dict(model='M10-T2', grid_points=513, grid_radius=9.,
        endpoint_role='finite-grid mechanics endpoint; not globally qualified model likelihood',
        full_prior_posterior_reference=False, prior='independent Normal(0,log(2)^2)', chart='identity',
        observations_sha256=_hash_tensor(observations), sources=dict(fixture.manifest['source_sha256']))
    endpoint_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    force = FrozenPositionOnlyForce(field, identity=fixture.identity+':FP32-DCTR',
        semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    endpoint = FrozenTargetPotential(potential, identity=endpoint_id)
    reference_force = FrozenPositionOnlyForce(reference_field, identity=endpoint_id+':finite-grid-gradient')
    manifest = dict(classification='M10_consumer_mechanics_only', endpoint=identity,
        endpoint_identity=endpoint_id, force_identity=force.identity, force_semantics=force.semantics,
        fixture=dict(fixture.manifest), batch_size=batch_size, retained_inference=False,
        precision='FP64_integrator_FP32_DCTR_FP64_finite_grid', jit_compile=True)
    return NonlinearConsumer(fixture, force, endpoint, reference_force, evaluate_force, evaluate_reference, manifest)
