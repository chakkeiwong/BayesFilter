"""Actual-SV full-DCTR consumer qualification; no control/tuning admission.

Only model callbacks and the prior chart are added here. Filtering, analytical
recursion, Contract E reset and all correction stages use the shared engine.
The independent endpoint is a numerically qualified FP64 quadrature reference.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import PerPointScoreModel
from bayesfilter.highdim.ledh_failure_status_tf import FailureCode
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_program_fixtures_tf import (
    BoundedLEDHFixtureTarget, source_closure,
)
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.inference.neural_force_hmc import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation,
)
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status
from bayesfilter.testing.surrogate_hmc_sv_reference import make_sv_reference


def actual_sv_model(dtype=tf.float32):
    """Physical (gamma,beta) callbacks, with exact raw-return weight density."""
    half = tf.constant(.5, dtype)
    log_two_pi = tf.constant(math.log(2 * math.pi), dtype)
    mu = tf.math.digamma(half) + tf.math.log(tf.constant(2., dtype))
    variance = tf.constant(math.pi**2 / 2, dtype)

    def initial(theta):
        return tf.zeros([1], dtype), tf.reshape(1 / (1 - theta[0]**2), [1, 1])

    def dinitial(theta, direction):
        return tf.zeros([1], dtype), tf.reshape(
            2 * theta[0] * direction[0] / (1 - theta[0]**2)**2, [1, 1])

    def obs(theta, x):
        return x + 2 * tf.math.log(theta[:, 1:2]) + mu

    def dobs(theta, x, dx, direction):
        return dx + 2 * direction[:, 1:2] / theta[:, 1:2]

    def density(theta, x, observation):
        log_beta = tf.math.log(theta[:, 1])
        v = observation[0] - 2 * log_beta - x[:, 0]
        # observation is log(y^2), but weights are densities with respect to dy.
        return -log_beta - half * x[:, 0] - half * tf.exp(v) - half * log_two_pi

    def ddensity(theta, x, observation, dx, direction):
        v = observation[0] - 2 * tf.math.log(theta[:, 1]) - x[:, 0]
        return tf.math.expm1(v) * (half * dx[:, 0] + direction[:, 1] / theta[:, 1])

    def jacobian(x):
        return tf.ones([tf.shape(x)[0], 1, 1], dtype)

    return PerPointScoreModel(
        transition_mean_fn=lambda theta, x: theta[:, :1] * x,
        transition_mean_tangent_fn=lambda theta, x, dx, direction:
            theta[:, :1] * dx + direction[:, :1] * x,
        observation_fn=lambda x: x + mu,
        observation_jacobian_fn=jacobian,
        observation_tangent_fn=lambda x, dx: dx,
        process_covariance=tf.ones([1, 1], dtype),
        observation_covariance=tf.reshape(variance, [1, 1]),
        observation_log_density_fn=density,
        observation_log_density_tangent_fn=ddensity,
        parameter_observation_fn=obs,
        parameter_observation_tangent_fn=dobs,
        parameter_observation_jacobian_fn=lambda theta, x: jacobian(x),
        parameter_observation_jacobian_tangent_fn=lambda theta, x, dx, direction:
            tf.zeros([tf.shape(x)[0], 1, 1], dtype),
        initial_distribution_fn=initial,
        initial_distribution_tangent_fn=dinitial,
    )


class SVFixtureTarget(BoundedLEDHFixtureTarget):
    """Reuse the full physical target with the SV probit chart, never logistic."""

    def unconstrained(self, z):
        z = tf.convert_to_tensor(z, self.initial_base.dtype)
        width = self.upper - self.lower
        phi = tf.exp(-.5 * z**2) / tf.sqrt(tf.constant(2 * math.pi, z.dtype))
        physical = self.lower + width * .5 * tf.math.erfc(-z / tf.sqrt(tf.constant(2., z.dtype)))
        value, score, status = self.physical(physical)
        finite = tf.reduce_all(tf.math.is_finite(z), axis=-1)
        chart_failure = status['outside_declared_support'] | ~finite
        # Finite probit coordinates never leave mathematical prior support.
        # FP32 saturation is a numerical failure, not a support exclusion.
        status['outside_declared_support'] = tf.zeros_like(finite)
        for key in ('value_failure_code', 'force_failure_code'):
            status[key] = tf.where(chart_failure, int(FailureCode.NONFINITE_INPUT), status[key])
        log_jac = tf.reduce_sum(tf.math.log(width) - .5 * z**2
            - tf.constant(.5 * math.log(2 * math.pi), z.dtype), axis=-1)
        transformed = score * width * phi - z
        valid = status['value_valid'] & finite & tf.math.is_finite(value + log_jac)
        force_valid = valid & status['force_valid'] & tf.reduce_all(tf.math.is_finite(transformed), axis=-1)
        status['value_valid'] = valid
        status['force_valid'] = force_valid
        status['program_valid'] = valid & force_valid
        status['direction_value_valid'] &= valid[:, None]
        status['direction_force_valid'] &= force_valid[:, None]
        return (tf.where(valid, value + log_jac, tf.constant(-float('inf'), z.dtype)),
                tf.where(force_valid[:, None], transformed, tf.zeros_like(transformed)), status)


def _hash_tensor(tensor):
    return hashlib.sha256(tf.io.serialize_tensor(tensor).numpy()).hexdigest()


def make_sv_fixture_target(dataset, *, controls, particle_count=1024):
    if (dataset.get('model_id'), dataset.get('horizon'), dataset.get('sigma')) != ('SV20', 20, 1.):
        raise ValueError('actual SV20 unit process variance dataset required')
    if dataset.get('prior') != 'independent Uniform(.1,.9)' or dataset.get('chart') != 'gamma,beta=.1+.8*Phi(z)':
        raise ValueError('wrong SV prior/chart')
    if dataset.get('law') != 'stationary x0; transition then Normal(0,beta^2*exp(x)) observation':
        raise ValueError('wrong SV observation/initial law')
    if dataset.get('observation_dtype') != 'float32':
        raise ValueError('once-rounded FP32 data required')
    n = particle_count
    chunks = select_transport_chunks(n)
    settings = dict(controls)
    if (settings.get('reset_policy') != 'contract_e' or settings.get('correction_steps', 0) < 1
            or settings.get('pairwise_steps', 0) < 1 or settings.get('coordinate_cap', 0) <= 0):
        raise ValueError('complete canonical reset and correction required')
    partition_code = {'calibration': 1, 'validation': 2, 'claim': 3, 'diagnostic': 4}[dataset['partition']]
    seeds = {str(p): [20260920, 12000 + partition_code * 100 + p] for p in (20, 21, 22)}
    if dataset.get('attempt') != 1 or dataset.get('replicate') != 0:
        raise ValueError('new attempt/replicate requires a separately declared fixed design')
    with tf.device('/CPU:0'):
        # Tiny fixed-input generation exception; no learned transport training.
        base = tf.cast(tf.random.stateless_normal([n, 1], seeds['20'], dtype=tf.float64), tf.float32)
        noises = tf.cast(tf.random.stateless_normal([20, n, 1], seeds['21'], dtype=tf.float64), tf.float32)
        design = tf.cast(tf.random.stateless_normal([n, 1], seeds['22'], dtype=tf.float64), tf.float32)
        raw = tf.constant(dataset['raw_observations'], tf.float32)
        if raw.shape != [20] or not bool(tf.reduce_all(tf.math.is_finite(raw) & (raw != 0))):
            raise ValueError('log-return flow requires finite nonzero observations; no floor is permitted')
        if [float(x) for x in raw.numpy().tolist()] != dataset['raw_observations']:
            raise ValueError('input observations were not once-rounded FP32 values')
        observations = 2 * tf.math.log(tf.abs(raw))[:, None]
        design -= tf.reduce_mean(design, axis=0)
        valid, chol = safe_cholesky(tf.matmul(design, design, transpose_a=True) / n, 'fixed_residual_design')
        if not bool(valid):
            raise ValueError('invalid residual design')
        design = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(design)))
    settings['reset_design'] = design
    sources = source_closure()
    root = Path(__file__).resolve().parents[2]
    for name in ('bayesfilter/testing/surrogate_hmc_sv_consumer.py',
                 'bayesfilter/testing/surrogate_hmc_sv_reference.py',
                 'bayesfilter/testing/surrogate_hmc_lg1_tf.py',
                 'bayesfilter/inference/neural_force_hmc.py'):
        sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest = dict(model_id='SV20', n=n, t=20, d=1, p=2,
        route='canonical_batch_fused_value_score', reset_contract_id='contract_e_chol_v1',
        derivative='analytical_recursive_all_parameter_dependencies',
        initial_law='stationary parameter-dependent x0, standard-normal base transformed by shared engine',
        observation_law='raw Normal(0,beta^2*exp(x)); log(y^2) moment geometry only',
        priors='independent Uniform(.1,.9)', coordinate_map='theta=.1+.8*Phi(z); Jacobian once',
        settings={k: v for k, v in settings.items() if k != 'reset_design'},
        seeds=seeds, chunk_policy=chunks.__dict__, dtype='float32',
        tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()), jit_compile=True,
        control_status='warm-start hypothesis, not selected for SV',
        input_generation='CPU fixed-input diagnostic exception; no training',
        input_sha256=dict(base=_hash_tensor(base), noise=_hash_tensor(noises),
            raw_observations=_hash_tensor(raw), flow_observations=_hash_tensor(observations),
            residual_design=_hash_tensor(design)), source_sha256=sources)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return SVFixtureTarget('SV20', actual_sv_model(), base, tf.ones([n, 1, 1], tf.float32),
        noises, observations, tf.constant([.1, .1]), tf.constant([.9, .9]),
        MappingProxyType(settings), identity, MappingProxyType(manifest))


@dataclass(frozen=True)
class SVConsumer:
    fixture: object
    reference: object
    force: FrozenPositionOnlyForce
    endpoint: FrozenTargetPotential
    exact_force: FrozenPositionOnlyForce
    evaluate_force: object
    evaluate_reference: object
    manifest: dict


def make_sv_consumer(dataset, *, controls, batch_size, particle_count=1024):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive integer batch size required')
    fixture = make_sv_fixture_target(dataset, controls=controls, particle_count=particle_count)
    reference = make_sv_reference(batch_size=batch_size)
    observations = tf.cast(tf.constant(dataset['raw_observations'], tf.float32), tf.float64)
    signature = [tf.TensorSpec([batch_size, 2], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_force(z):
        return fixture.unconstrained(tf.cast(z, tf.float32))

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate_reference(z):
        result = reference.unconstrained_value_score(z, observations)
        return result[0], result[1]

    def field(z):
        _, score, status = evaluate_force(z)
        return PositionOnlyForceEvaluation(-tf.cast(score, tf.float64), status['force_valid'])

    def potential(z):
        logp, _ = evaluate_reference(z)
        return TargetPotentialEvaluation(-logp, endpoint_status(logp))

    def exact_field(z):
        logp, score = evaluate_reference(z)
        valid = tf.math.is_finite(logp) & tf.reduce_all(tf.math.is_finite(score), axis=-1)
        return PositionOnlyForceEvaluation(-score, valid)

    identity = dict(model='SV20', reference=reference.specification,
        raw_observations_sha256=_hash_tensor(observations),
        sources=dict(fixture.manifest['source_sha256']),
        precision='FP64 numerical model-reference endpoint',
        initial_law=dataset['law'], prior=dataset['prior'], chart=dataset['chart'])
    endpoint_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    force = FrozenPositionOnlyForce(field, identity=fixture.identity + ':fp64-in-fp32-force',
        semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    endpoint = FrozenTargetPotential(potential, identity=endpoint_id)
    exact = FrozenPositionOnlyForce(exact_field, identity=endpoint_id + ':reference-total-gradient')
    manifest = dict(classification='actual_SV_full_DCTR_qualification_not_tuning_admission',
        endpoint=identity, endpoint_identity=endpoint_id, force_identity=force.identity,
        force_semantics=force.semantics, fixture=dict(fixture.manifest), batch_size=batch_size,
        precision='FP64_integrator_FP32_DCTR_FP64_quadrature_endpoint', jit_compile=True)
    return SVConsumer(fixture, reference, force, endpoint, exact, evaluate_force, evaluate_reference, manifest)
