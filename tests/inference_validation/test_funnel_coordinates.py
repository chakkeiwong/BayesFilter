"""Same model law and starts through a distinct ordinary coordinate system."""
import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.testing.inference_validation.procedures import initial_starts
from bayesfilter.testing.inference_validation.references import analytic


def test_funnel_change_of_variables_includes_full_jacobian():
    centered = ValidationTarget("funnel", jit_compile=False)
    noncentered = ValidationTarget("funnel_noncentered", jit_compile=False)
    q = tf.constant([[-9., .3, -2.], [-1., 2., .1], [0., -1., 1.], [8., .2, 3.]], tf.float64)
    model = noncentered.to_model(q)
    with tf.GradientTape() as tape:
        tape.watch(q)
        transformed = centered.log_density(noncentered.to_model(q)) + q[:, 0]
    value, score = noncentered.log_prob_and_grad(q)
    np.testing.assert_allclose(value, transformed, atol=1e-12)
    np.testing.assert_allclose(score, tape.gradient(transformed, q), atol=1e-12)
    np.testing.assert_allclose(value, analytic.log_density("funnel_noncentered", q), atol=1e-12)
    np.testing.assert_allclose(analytic.active_coordinates("funnel_noncentered", model), q, atol=1e-12)
    assert centered.adapter_signature() != noncentered.adapter_signature()


@pytest.mark.parametrize("regime", ["dispersed", "remote"])
def test_coordinate_comparison_preserves_model_start_bank(regime):
    centered = ValidationTarget("funnel", jit_compile=False)
    noncentered = ValidationTarget("funnel_noncentered", jit_compile=False)
    np.testing.assert_allclose(noncentered.to_model(initial_starts(noncentered, regime)),
                               initial_starts(centered, regime), atol=1e-12)


def test_independent_draws_transform_to_identical_model_law():
    centered = analytic.draw("funnel", 200, [41, 53])
    q = analytic.draw("funnel_noncentered", 200, [41, 53])
    np.testing.assert_allclose(analytic.model_coordinates("funnel_noncentered", q), centered, atol=1e-12)
    assert analytic.exact_functionals("funnel_noncentered") == analytic.exact_functionals("funnel")
