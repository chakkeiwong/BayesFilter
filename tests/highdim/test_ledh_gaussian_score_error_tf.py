"""CPU-only diagnostic checks of summaries from the full shared LEDH trace."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.score_study.backward_guide_pilot_tf import (
    REFERENCE_CONTROLS, make_guided_reference_kernel, reference_theta,
)


def test_trace_increments_and_reset_moment_derivatives():
    controls = {**REFERENCE_CONTROLS, "flow_substeps": 4}
    kernel = make_guided_reference_kernel(8, 2, controls, jit_compile=False,
                                         all_directions=False, error_diagnostics=True)
    baseline = make_guided_reference_kernel(8, 2, controls, jit_compile=False,
                                           all_directions=False)
    theta = reference_theta(0.8)
    direction = tf.constant([0.2, -0.4, 0.1, 0.3, 0.0, 0.1], tf.float64)
    operands = (
        direction, tf.constant([[0.2], [-0.3]], tf.float64),
        tf.random.stateless_normal([8, 1], [20261016, 710001], dtype=tf.float64),
        tf.random.stateless_normal([2, 8, 1], [20261016, 710002], dtype=tf.float64),
        tf.zeros([2, 1, 1], tf.float64), tf.zeros([2, 1], tf.float64),
    )
    value, score, diagnostics = kernel(theta, *operands)
    np.testing.assert_allclose([value, score], baseline(theta, *operands)[:2], atol=0, rtol=0)
    np.testing.assert_allclose(tf.reduce_sum(diagnostics["value_increment"]), value, atol=1e-12)
    np.testing.assert_allclose(tf.reduce_sum(diagnostics["score_increment"]), score, atol=1e-12)
    for flag in ("program_valid", "reset_transport_valid", "reset_numerical_valid",
                 "ukf_predict_valid", "ukf_update_valid", "guided_proposal_valid"):
        assert bool(tf.reduce_all(diagnostics[flag]))
    assert bool(tf.reduce_all(diagnostics["transport_column_total_variation_per_time"] <= 1e-4))
    step = 3e-5
    plus = kernel(theta + step * direction, *operands)[2]
    minus = kernel(theta - step * direction, *operands)[2]
    for stage in ("weighted_children", "raw_reset", "pre_cap", "post_cap", "final_reset"):
        for moment in ("mean", "variance"):
            difference = (plus[stage + "_" + moment] - minus[stage + "_" + moment]) / (2 * step)
            np.testing.assert_allclose(diagnostics[stage + "_d_" + moment], difference,
                                       atol=2e-7, rtol=2e-5)
    assert kernel.experimental_get_tracing_count() == 1
