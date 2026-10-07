"""Independent diagnostics for the SIR tuning trace and reset derivative."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import math
import numpy as np
import pytest
import tensorflow as tf

from docs.benchmarks import run_ledh_sir_no_oracle_tuning as tuning
from bayesfilter.highdim import sqmc_campaign_tf as common
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec, trace_kernel

D = tf.float64


def _cpu_tf():
    # The test is a small CPU/reference diagnostic; serious campaign evidence
    # is collected by the GPU/XLA runner and records its own device manifest.
    return tuning.prepare_tf("cpu")[0]


def _inputs(tf_module, spec, horizon=3, particles=72, seed=261006701):
    theta = spec.default_theta(tf_module.float64)
    obs = spec.observations(horizon, 261006702, dtype=tf_module.float64,
                            jit_compile=True)
    args = common.random_inputs(tuning.ROUTE, seed, particles, spec.dimension,
                                horizon, tf_module.float64, jit_compile=True)
    return theta, obs, args


def test_dynamic_trace_preserves_legacy_direction_zero_endpoint():
    tf_module = _cpu_tf()
    spec = NonlinearSQMCSpec("sir_d18")
    theta, obs, args = _inputs(tf_module, spec)
    old = trace_kernel(spec, tuning.ROUTE, tuning.BASE, 72, 3, D,
                       jit_compile=True, reset_design_kind=tuning.DESIGN,
                       include_clouds=True, dynamic_direction=False)
    new = trace_kernel(spec, tuning.ROUTE, tuning.BASE, 72, 3, D,
                       jit_compile=True, reset_design_kind=tuning.DESIGN,
                       include_clouds=True, dynamic_direction=True)
    a = old(theta, *args, obs)
    b = new(theta, *args, obs, tf_module.one_hot(0, 3, dtype=D))
    np.testing.assert_allclose(a[0], b[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(a[1], b[1], atol=1e-10, rtol=1e-10)
    assert len(a[2]) == len(b[2]) == 3
    for left, right in zip(a[2], b[2]):
        for key in ("children", "d_children", "states_after_reset",
                    "d_states_after_reset", "posterior_logits",
                    "d_posterior_logits"):
            np.testing.assert_allclose(left[key], right[key], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("coordinate", [0, 1, 2])
def test_sir_reset_score_matches_independent_cloud_finite_difference(coordinate):
    tf_module = _cpu_tf()
    spec = NonlinearSQMCSpec("sir_d18")
    n, horizon = 72, 3
    theta, obs, args = _inputs(tf_module, spec, horizon, n, 261006703)
    direction = tf_module.one_hot(coordinate, 3, dtype=D)
    kernel = trace_kernel(spec, tuning.ROUTE, tuning.BASE, n, horizon, D,
                          jit_compile=True, reset_design_kind=tuning.DESIGN,
                          include_clouds=True, dynamic_direction=True)
    _, _, trace = kernel(theta, *args, obs, direction)
    fields = [tf_module.stack([row[key] for row in trace]) for key in
              ("children", "d_children", "states_after_reset",
               "d_states_after_reset", "posterior_logits",
               "d_posterior_logits")]
    reset = tuning.reset_kernel(spec, n, horizon, tf_module)
    result = reset(theta, direction, *fields, obs)

    def local_log_error(t, step):
        model, _ = spec.model(t, direction)
        v = 1. + 100. * tf_module.exp(2. * t[2])
        before, after = [], []
        x, dx, z, dz, logits, dlogits = fields
        for k in range(horizon - 1):
            def ell(points):
                means = model.transition_mean_fn(t, points)
                predicted = model.observation_fn(means)
                residual = obs[k + 1][None, :] - predicted
                r2 = tf_module.reduce_sum(residual * residual, axis=-1)
                return -.5 * (9. * tf_module.math.log(
                    tf_module.constant(2. * math.pi, D) * v) + r2 / v)
            before.append(tf_module.reduce_logsumexp(
                tf_module.nn.log_softmax(logits[k] + step * dlogits[k]) +
                ell(x[k] + step * dx[k])))
            after.append(tf_module.reduce_logsumexp(ell(z[k] + step * dz[k])) -
                         tf_module.math.log(tf_module.cast(n, D)))
        return tf_module.stack(after) - tf_module.stack(before)

    value = local_log_error(theta, tf_module.constant(0., D))
    eps = tf_module.constant(1e-6, D)
    fd = (local_log_error(theta + eps * direction, eps) -
          local_log_error(theta - eps * direction, -eps)) / (2. * eps)
    np.testing.assert_allclose(result["log_error"], value, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(result["score_error"], fd, atol=2e-6, rtol=2e-6)
    assert bool(tf_module.reduce_all(tf_module.math.is_finite(result["score_error"])))
