"""Canonical analytical DCTR for the declared two-noise-scale AR(2) scope.

The finite likelihood is a fixed numerical approximation. Its analytical score
is eligible only as a position-only proposal field until downstream evidence
qualifies a specific inference use. No full-state transition density exists.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import tensorflow as tf

from bayesfilter.highdim.gaussian_cloud_designs_tf import standard_normal_cloud
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import PerPointScoreModel, canonical_batch_fused_value_score
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

PRIOR_SD = math.log(2.)


@dataclass(frozen=True)
class AR2LEDHTarget:
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
        prior = -tf.reduce_sum(.5*(theta/PRIOR_SD)**2+math.log(PRIOR_SD)+.5*math.log(2*math.pi), -1)
        score -= theta/PRIOR_SD**2
        value += prior
        # The Gaussian prior has all of R^2 as support. Numerical failures must
        # retain their typed status; they must never become a support boundary.
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


def make_ar2_target(dataset, *, particle_count, controls, dtype=tf.float32):
    if dataset.get("model_id") != "AR2" or dataset.get("initial_law") != "N(0,I2)" or dataset.get("transition_before_observation") is not True:
        raise ValueError("declared AR2 initial law and timing required")
    horizon = dataset["horizon"]
    n = int(particle_count)
    chunks = select_transport_chunks(n)
    controls = dict(controls)
    if (controls.get("reset_policy") != "contract_e" or controls.get("correction_steps", 0) < 1
            or controls.get("pairwise_steps", 0) < 1 or controls.get("coordinate_cap", 0) <= 0):
        raise ValueError("complete canonical reset and dual-cap corrections required")
    f = tf.constant([[.45, -.15], [1., 0.]], dtype)
    model = PerPointScoreModel(
        transition_mean_fn=lambda theta, x: x @ tf.transpose(f),
        transition_mean_tangent_fn=lambda theta, x, dx, direction: dx @ tf.transpose(f),
        observation_fn=lambda x: x[:, :1], observation_tangent_fn=lambda x, dx: dx[:, :1],
        observation_jacobian_fn=lambda x: tf.broadcast_to(tf.constant([[[1., 0.]]], dtype), [tf.shape(x)[0], 1, 2]),
        observation_jacobian_tangent_fn=lambda x, dx: tf.zeros([tf.shape(x)[0], 1, 2], dtype),
        process_covariance=tf.constant([[.09]], dtype),
        process_covariance_fn=lambda theta: tf.reshape(.09*tf.exp(2*theta[0]), [1, 1]),
        process_covariance_tangent_fn=lambda theta, direction: tf.reshape(.18*tf.exp(2*theta[0])*direction[0], [1, 1]),
        observation_covariance=tf.constant([[.04]], dtype),
        observation_covariance_fn=lambda theta: tf.reshape(.04*tf.exp(2*theta[1]), [1, 1]),
        observation_covariance_tangent_fn=lambda theta, direction: tf.reshape(.08*tf.exp(2*theta[1])*direction[1], [1, 1]),
        innovation_matrix=tf.constant([[1.], [0.]], dtype))
    with tf.device("/CPU:0"):
        partition = {"calibration": 1, "validation": 2, "claim": 3, "diagnostic": 4}[dataset["partition"]]
        seeds = {}
        for purpose in (20, 22):
            seed = tf.constant([20260923, 800], tf.int32)
            for code in (horizon, partition, dataset.get("attempt", 1), dataset.get("replicate", 0), purpose):
                seed = tf.random.experimental.stateless_fold_in(seed, int(code))
            seeds[purpose] = seed.numpy().tolist()
        path = standard_normal_cloud("scrambled_halton_gaussian", num_particles=n, dimension=2+horizon,
                    seed=seeds[20][0], salt=seeds[20][1], dtype=tf.float64)
        design = standard_normal_cloud("scrambled_halton_gaussian", num_particles=n, dimension=2,
                    seed=seeds[22][0], salt=seeds[22][1], dtype=tf.float64)
        design -= tf.reduce_mean(design, 0)
        valid, chol = safe_cholesky(tf.transpose(design) @ design/tf.cast(n, tf.float64), "fixed_AR2_design")
        if not bool(valid):
            raise ValueError("singular fixed AR2 reset design")
        design = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(design)))
        # Freeze the same rounded design in both precision arms. The FP64 arm
        # tests arithmetic transfer, not a different realized particle cloud.
        path = tf.cast(tf.cast(path, tf.float32), dtype)
        design = tf.cast(tf.cast(design, tf.float32), dtype)
        observations = tf.cast(tf.convert_to_tensor(dataset["observations"], tf.float32), dtype)
    if observations.shape != [horizon, 1]:
        raise ValueError("AR2 observation shape differs from declared horizon")
    inputs = dict(initial_states=path[:, :2], initial_covariances=tf.eye(2, batch_shape=[n], dtype=dtype),
                  noises=tf.transpose(path[:, 2:])[:, :, None], observations=observations)
    controls["reset_design"] = design
    root = Path(__file__).resolve().parents[2]
    sources = source_closure()
    for name in ("bayesfilter/highdim/ledh_structural_ar2_target_tf.py", "bayesfilter/highdim/gaussian_cloud_designs_tf.py"):
        sources[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest = dict(model_id="AR2", horizon=horizon, particle_count=n, state_dimension=2,
        innovation_dimension=1, parameter_dimension=2, observation_dimension=1,
        initial_law="N(0,I2)", parameterization="Q=.09*exp(2*lambda_Q)*BB'; R=.04*exp(2*lambda_R)",
        prior=dict(family="independent_normal", mean=[0., 0.], sd=PRIOR_SD), coordinates="unbounded_noise_log_scales",
        transition_before_observation=True, route="canonical_batch_fused_value_score",
        score_engine="analytical_recursive_all_parameter_dependencies", reset_contract_id="contract_e_chol_v1",
        dtype=dtype.name, tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()) if dtype == tf.float32 else False,
        jit_compile=True, chunk_policy=chunks.__dict__, draw_design="scrambled_halton_gaussian",
        seeds=seeds, settings={k: v for k, v in controls.items() if k != "reset_design"},
        input_sha256={k: hashlib.sha256(tf.io.serialize_tensor(v).numpy()).hexdigest()
                      for k, v in (inputs | {"reset_design": design}).items()}, source_sha256=sources)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return AR2LEDHTarget(model, MappingProxyType(inputs), MappingProxyType(controls), MappingProxyType(manifest), identity)
