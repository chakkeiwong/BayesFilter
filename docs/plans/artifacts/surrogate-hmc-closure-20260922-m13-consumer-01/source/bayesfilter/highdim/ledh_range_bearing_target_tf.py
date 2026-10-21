"""Canonical analytical DCTR likelihood for the M13 range/bearing fixture.

Parameters are unbounded log multipliers of the process and observation noise
standard deviations. This adapter adds no parameter prior and confers no tuning
or posterior admission. Numerical failures retain their typed status.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import tensorflow as tf

from bayesfilter.highdim.gaussian_cloud_designs_tf import standard_normal_cloud
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel, canonical_batch_fused_value_score)
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.wrapped_gaussian_tf import alias_policy, wrapped_gaussian_batch_kernel


def _polar_geometry(x):
    p = x[:, :2]
    square = tf.reduce_sum(p*p, axis=-1)
    radius = tf.sqrt(square+tf.constant(1e-12, x.dtype))
    valid = (square > 0.) & tf.reduce_all(tf.math.is_finite(x), -1)
    return p, square, radius, valid


def _checked(value, valid):
    shape = [-1]+[1]*(value.shape.rank-1)
    return tf.where(tf.reshape(valid, shape), value, tf.constant(float("nan"), value.dtype))


def polar_observation(x):
    p, _, radius, valid = _polar_geometry(x)
    return _checked(tf.stack([radius, tf.atan2(p[:, 1], p[:, 0])], -1), valid)


def polar_observation_tangent(x, dx):
    p, square, radius, valid = _polar_geometry(x)
    dp = dx[:, :2]
    radial = tf.reduce_sum(p*dp, -1)/radius
    angular = (p[:, 0]*dp[:, 1]-p[:, 1]*dp[:, 0])/square
    return _checked(tf.stack([radial, angular], -1), valid)


def polar_jacobian(x):
    p, square, radius, valid = _polar_geometry(x)
    angular = tf.stack([-p[:, 1], p[:, 0]], -1)/square[:, None]
    position = tf.stack([p/radius[:, None], angular], 1)
    return _checked(tf.concat([position, tf.zeros_like(position)], -1), valid)


def polar_jacobian_tangent(x, dx):
    p, square, radius, valid = _polar_geometry(x)
    dp = dx[:, :2]
    dot = tf.reduce_sum(p*dp, -1)
    radial = dp/radius[:, None]-p*(dot/radius**3)[:, None]
    angular = (tf.stack([-dp[:, 1], dp[:, 0]], -1)/square[:, None]
               -tf.stack([-p[:, 1], p[:, 0]], -1)*(2*dot/square**2)[:, None])
    position = tf.stack([radial, angular], 1)
    return _checked(tf.concat([position, tf.zeros_like(position)], -1), valid)


def make_range_bearing_model(a, q, r, *, dtype=tf.float32):
    """Both likelihood callbacks use the shared wrapped-density implementation."""
    a, q, r = [tf.convert_to_tensor(v, dtype) for v in (a, q, r)]
    if a.shape != [4, 4] or q.shape != [4, 4] or r.shape != [2, 2]:
        raise ValueError("M13 requires state4 and observation2 matrices")
    wrapped = wrapped_gaussian_batch_kernel(dtype=dtype)

    def density(theta, x, observed):
        covariance = tf.exp(2*theta[:, 1])[:, None, None]*r[None]
        residual = observed-polar_observation(x)
        result = wrapped(residual, covariance, tf.zeros_like(residual), tf.zeros_like(covariance))
        # -inf from a numerical kernel veto is not model zero density.
        return tf.where(result["value_valid"], result["value"], tf.constant(float("nan"), dtype))

    def density_tangent(theta, x, observed, dx, direction):
        covariance = tf.exp(2*theta[:, 1])[:, None, None]*r[None]
        dcovariance = 2*direction[:, 1, None, None]*covariance
        result = wrapped(observed-polar_observation(x), covariance,
                         -polar_observation_tangent(x, dx), dcovariance)
        return result["tangent"]

    return PerPointScoreModel(
        transition_mean_fn=lambda theta, x: x @ tf.transpose(a),
        transition_mean_tangent_fn=lambda theta, x, dx, direction: dx @ tf.transpose(a),
        observation_fn=polar_observation, observation_tangent_fn=polar_observation_tangent,
        observation_jacobian_fn=polar_jacobian,
        observation_jacobian_tangent_fn=polar_jacobian_tangent,
        process_covariance=q, process_covariance_fn=lambda theta: q*tf.exp(2*theta[0]),
        process_covariance_tangent_fn=lambda theta, direction: 2*q*tf.exp(2*theta[0])*direction[0],
        observation_covariance=r, observation_covariance_fn=lambda theta: r*tf.exp(2*theta[1]),
        observation_covariance_tangent_fn=lambda theta, direction: 2*r*tf.exp(2*theta[1])*direction[1],
        observation_log_density_fn=density, observation_log_density_tangent_fn=density_tangent)


@dataclass(frozen=True)
class M13LikelihoodTarget:
    model: object
    inputs: object
    controls: object
    manifest: object
    identity: str

    def evaluate(self, theta, *, filter_moments=False):
        theta = tf.convert_to_tensor(theta, self.inputs["initial_states"].dtype)
        directions = tf.broadcast_to(tf.eye(2, dtype=theta.dtype)[None], [tf.shape(theta)[0], 2, 2])
        value, score, status = canonical_batch_fused_value_score(
            self.model, theta, directions, **dict(self.inputs), **dict(self.controls),
            return_filter_moments=filter_moments)
        status = dict(status)
        status["outside_declared_support"] = tf.zeros_like(status["value_valid"])
        status["value_valid"] &= tf.math.is_finite(value)
        status["force_valid"] &= status["value_valid"] & tf.reduce_all(tf.math.is_finite(score), -1)
        status["program_valid"] = status["value_valid"] & status["force_valid"]
        return (tf.where(status["value_valid"], value, tf.constant(-float("inf"), theta.dtype)),
                tf.where(status["force_valid"][:, None], score, tf.zeros_like(score)), status)

    def compile(self, batch_size, *, filter_moments=False):
        return tf.function(lambda theta: self.evaluate(theta, filter_moments=filter_moments),
            input_signature=[tf.TensorSpec([batch_size, 2], self.inputs["initial_states"].dtype)],
            jit_compile=True, autograph=False)


def make_range_bearing_target(fixture, *, particle_count, controls, dtype=tf.float32,
                             design_seed=20260923, design_partition=0):
    """Freeze designs for a likelihood mechanics check; controls remain untuned."""
    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError("FP32 or FP64 required")
    if fixture.name != "range_bearing_gaussian_tf_tfp_moderate":
        raise ValueError("the declared M13 fixture is required")
    n, horizon = int(particle_count), int(fixture.horizon)
    chunks = select_transport_chunks(n)
    settings = dict(controls)
    if (settings.get("reset_policy") != "contract_e" or settings.get("correction_steps", 0) < 1
            or settings.get("pairwise_steps", 0) < 1 or settings.get("coordinate_cap", 0) <= 0):
        raise ValueError("complete canonical reset and dual-cap corrections required")
    cast = lambda v: tf.cast(v, dtype)
    model = make_range_bearing_model(cast(fixture.A), cast(fixture.Q), cast(fixture.R), dtype=dtype)
    with tf.device("/CPU:0"):
        seeds = {}
        for purpose in (20, 22):
            seed = tf.constant([int(design_seed), 1300], tf.int32)
            for code in (horizon, int(design_partition), purpose):
                seed = tf.random.experimental.stateless_fold_in(seed, code)
            seeds[purpose] = seed.numpy().tolist()
        path = standard_normal_cloud("scrambled_halton_gaussian", num_particles=n,
            dimension=4+4*horizon, seed=seeds[20][0], salt=seeds[20][1], dtype=tf.float64)
        design = standard_normal_cloud("scrambled_halton_gaussian", num_particles=n,
            dimension=4, seed=seeds[22][0], salt=seeds[22][1], dtype=tf.float64)
        design -= tf.reduce_mean(design, 0)
        valid, chol = safe_cholesky(tf.transpose(design) @ design/tf.cast(n, tf.float64), "fixed_M13_design")
        if not bool(valid):
            raise ValueError("singular fixed M13 reset design")
        design = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(design)))
        p0 = tf.cast(fixture.P0, tf.float64)
        m0 = tf.cast(fixture.m0, tf.float64)
        initial = m0+path[:, :4] @ tf.transpose(tf.linalg.cholesky(p0))
        # Same realized rounded particle/noise/reset design in both arithmetic arms.
        rounded = lambda value: tf.cast(tf.cast(value, tf.float32), dtype)
        inputs = dict(initial_states=rounded(initial),
            initial_covariances=tf.broadcast_to(cast(p0), [n, 4, 4]),
            noises=rounded(tf.transpose(tf.reshape(path[:, 4:], [n, horizon, 4]), [1, 0, 2])),
            observations=cast(fixture.observations))
        settings["reset_design"] = rounded(design)
    if inputs["observations"].shape != [horizon, 2]:
        raise ValueError("M13 observation shape differs from the declared horizon")
    root = Path(__file__).resolve().parents[2]
    sources = source_closure()
    for name in ("bayesfilter/highdim/ledh_range_bearing_target_tf.py",
                 "bayesfilter/highdim/gaussian_cloud_designs_tf.py",
                 "bayesfilter/wrapped_gaussian_tf.py"):
        sources[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest = dict(model_id="M13", fixture=fixture.model_definition(), horizon=horizon,
        particle_count=n, state_dimension=4, innovation_dimension=4, observation_dimension=2,
        parameter_dimension=2, parameterization="Q=Q0*exp(2*lambda_Q); R=R0*exp(2*lambda_R)",
        prior=None, quantity="finite_canonical_log_likelihood_only", transition_before_observation=True,
        route="canonical_batch_fused_value_score", score_engine="analytical_recursive_all_parameter_dependencies",
        reset_contract_id="contract_e_chol_v1", control_status="diagnostic_warm_start_not_scope_tuned",
        dtype=dtype.name, tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()) if dtype == tf.float32 else False,
        jit_compile=True, chunk_policy=chunks.__dict__, observation_law="wrapped_correlated_gaussian",
        alias_policy=alias_policy(), alias_limit_role="numerical_validity_not_parameter_support",
        draw_design="scrambled_halton_gaussian_FP32_rounded_once", seeds=seeds,
        design_partition=design_partition, settings={k: v for k, v in settings.items() if k != "reset_design"},
        input_sha256={k: hashlib.sha256(tf.io.serialize_tensor(v).numpy()).hexdigest()
                      for k, v in (inputs | {"reset_design": settings["reset_design"]}).items()},
        source_sha256=sources)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return M13LikelihoodTarget(model, MappingProxyType(inputs), MappingProxyType(settings),
                               MappingProxyType(manifest), identity)
