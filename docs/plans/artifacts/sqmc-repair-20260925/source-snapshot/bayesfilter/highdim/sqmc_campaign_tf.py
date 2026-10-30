"""Common SQMC campaign computation; host diagnostics cannot confer admission.

The default kernel is TensorFlow/XLA. Explicit non-JIT calls are reference
diagnostics. All parameter directions use the shared analytical executor.
"""
from __future__ import annotations

import functools
import math
import time

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.sqmc_tf import randomized_halton_gaussian, randomized_halton_joint
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size


ROUTES = {
    'iid_dual_cap': ('existing_one_to_one', .98),
    'previous_inverse_cdf': ('hilbert_inverse_cdf', .98),
    'repaired_permutation': ('hilbert_permutation_one_to_one', .98),
    'repaired_permutation_ablation': ('hilbert_permutation_one_to_one', .97),
}


def route_settings(route):
    if route not in ROUTES:
        raise ValueError(f'unknown SQMC route: {route}')
    ancestry, cap = ROUTES[route]
    return dict(ancestry_policy=ancestry, coordinate_cap=cap)


def reset_design(n, d, dtype):
    if n < 2 * d or n % (2 * d):
        raise ValueError('Contract-E residual design requires N divisible by 2D')
    return tf.tile(tf.concat([tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)], axis=0), [n // (2 * d), 1])


def random_inputs(route, seed, n, d, horizon, dtype=tf.float64):
    route_settings(route)
    if route == 'iid_dual_cap':
        initial = tf.random.stateless_normal([n, d], [seed, 101], dtype=dtype)
        noise = tf.stack([tf.random.stateless_normal([n, d], [seed, 1001 + t], dtype=dtype) for t in range(horizon)])
        return initial, noise, tf.zeros([horizon, n], dtype)
    initial = randomized_halton_gaussian(num_particles=n, dimension=d, seed=seed, salt=301, dtype=dtype)
    rows = [randomized_halton_joint(num_particles=n, state_dimension=d, seed=seed, salt=3001 + t, dtype=dtype)
            for t in range(horizon)]
    return initial, tf.stack([tf.math.ndtri(row[2]) for row in rows]), tf.stack([row[1] for row in rows])


def numerical_settings(controls):
    required = ('reset_epsilon', 'reset_sinkhorn_steps', 'reset_balance_steps',
                'correction_steps', 'correction_strength', 'pairwise_steps', 'pairwise_strength')
    missing = set(required) - set(controls)
    if missing:
        raise ValueError(f'missing SQMC controls: {sorted(missing)}')
    result = dict(flow_substeps=8, reset_policy='contract_e', reset_ridge=1e-5,
                  correction_lm_damping=.01, correction_lm_scale_floor=.0001,
                  correction_trust_radius=.5, pairwise_rms_cap=2., coordinate_cap_power=8,
                  state_map_policy='adaptive_empirical', hilbert_bits=12)
    allowed = set(result) | set(required)
    if set(controls) - allowed:
        raise ValueError(f'unknown SQMC controls: {sorted(set(controls) - allowed)}')
    result.update(controls)
    if result['reset_policy'] != 'contract_e':
        raise ValueError('SQMC campaign requires Contract E')
    for name, value in result.items():
        if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
            raise ValueError(f'invalid numerical control: {name}')
    for name in ('flow_substeps', 'reset_sinkhorn_steps', 'reset_balance_steps', 'correction_steps',
                 'pairwise_steps', 'coordinate_cap_power', 'hilbert_bits'):
        if type(result[name]) is not int:
            raise ValueError(f'integer numerical control required: {name}')
    if result['reset_epsilon'] <= 0 or result['flow_substeps'] < 1 or result['reset_sinkhorn_steps'] < 1:
        raise ValueError('positive epsilon, flow and Sinkhorn steps required')
    return result


@functools.lru_cache(maxsize=32)
def _kernel(spec, route, n, horizon, dtype_name, settings_tuple, jit_compile):
    dtype = tf.as_dtype(dtype_name)
    settings = dict(settings_tuple)
    settings.update(route_settings(route))
    # Validate repository chunk policy before tracing; core owns its selection.
    select_transport_chunk_size(n)
    design = reset_design(n, spec.dimension, dtype)
    signature = [tf.TensorSpec([spec.parameter_count], dtype), tf.TensorSpec([n, spec.dimension], dtype),
                 tf.TensorSpec([horizon, n, spec.dimension], dtype), tf.TensorSpec([horizon, n], dtype),
                 tf.TensorSpec([horizon, spec.dimension], dtype)]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def compute(theta, initial, noise, uniforms, observations):
        values = tf.TensorArray(dtype, size=spec.parameter_count)
        scores = tf.TensorArray(dtype, size=spec.parameter_count)
        def body(i, values, scores):
            direction = tf.one_hot(i, spec.parameter_count, dtype=dtype)
            model, _ = spec.model(theta, direction)
            states, covs, ds, dc = spec.initial_cloud(theta, initial, direction)
            value, score = canonical_value_and_analytical_score(
                model, theta, states, covs, noise, observations, with_score=True,
                initial_state_tangent=ds, initial_covariance_tangent=dc,
                reset_design=design, process_ancestor_uniforms=uniforms, **settings)
            return i + 1, values.write(i, value), scores.write(i, score[0])
        _, values, scores = tf.while_loop(lambda i, *_: i < spec.parameter_count, body, (0, values, scores))
        values, scores = values.stack(), scores.stack()
        valid = tf.reduce_all(tf.math.is_finite(values)) & tf.reduce_all(tf.math.is_finite(scores))
        # Every directional call must differentiate the same scalar computation.
        tolerance = tf.cast(1e-10 if dtype == tf.float64 else 1e-5, dtype) * (1. + tf.abs(values[0]))
        valid &= tf.reduce_max(tf.abs(values - values[0])) <= tolerance
        return values[0], scores, valid
    return compute


def value_and_score(spec, route, controls, theta, observations, seed, particle_count, *, jit_compile=True, inputs=None):
    theta = tf.convert_to_tensor(theta)
    observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
    horizon = int(observations.shape[0])
    settings = numerical_settings(controls)
    kernel = _kernel(spec, route, particle_count, horizon, theta.dtype.name, tuple(sorted(settings.items())), bool(jit_compile))
    if inputs is None:
        inputs = random_inputs(route, seed, particle_count, spec.dimension, horizon, theta.dtype)
    return kernel(theta, *inputs, observations)


def score_metrics(score, reference):
    """Descriptive errors. Undefined relative errors remain unavailable."""
    score, reference = tf.convert_to_tensor(score), tf.convert_to_tensor(reference)
    error = float(tf.linalg.norm(score - reference).numpy())
    a, b = float(tf.linalg.norm(score).numpy()), float(tf.linalg.norm(reference).numpy())
    cosine = float(tf.reduce_sum(score * reference).numpy()) / (a * b) if a > 0 and b > 0 else None
    return dict(score_l2_error=error, relative_score_error=error / b if b > 0 else None,
                cosine_similarity=cosine, relative_norm_error=abs(a - b) / b if b > 0 else None,
                component_absolute_errors=tf.abs(score - reference).numpy().tolist())


def evaluate_diagnostic(spec, route, controls, observations, theta, seed, particle_count, *, jit_compile=True):
    """Run a complete value/score/oracle diagnostic, never a tuned claim."""
    started = time.perf_counter()
    metadata = dict(target_id=spec.target_id, route=route, seed=seed, jit_compile=jit_compile,
                    evidence_role='diagnostic_only', tuning_status='unvalidated_controls',
                    claim_eligible=False, score_method='analytical_recursive', **route_settings(route))
    try:
        value, score, valid = value_and_score(spec, route, controls, theta, observations, seed, particle_count,
                                             jit_compile=jit_compile)
        oracle_value, oracle_score = spec.reference_value_and_score(theta, observations)
        valid = (bool(valid.numpy()) and math.isfinite(float(value.numpy()))
                 and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
                 and bool(tf.reduce_all(tf.math.is_finite(oracle_score)).numpy())
                 and math.isfinite(float(oracle_value.numpy())))
        if not valid:
            raise ValueError('nonfinite value/score/oracle or inconsistent directional values')
        result = dict(valid=True, value=float(value.numpy()), score=score.numpy().tolist(),
                      oracle_value=float(oracle_value.numpy()), oracle_score=oracle_score.numpy().tolist(),
                      value_error=float((value - oracle_value).numpy()), **score_metrics(score, oracle_score))
        if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
            raise ValueError('nonfinite computed diagnostic metric')
        result['tuning_score'] = result['score_l2_error']
        result['compute_device'] = value.device
        result['tf32'] = bool(tf.config.experimental.tensor_float_32_execution_enabled())
        result['controls'] = numerical_settings(controls)
        return dict(metadata, **result, wall_seconds=time.perf_counter() - started)
    except (ValueError, tf.errors.OpError) as exc:
        return dict(metadata, valid=False, value=None, score=None, error=str(exc),
                    wall_seconds=time.perf_counter() - started)
