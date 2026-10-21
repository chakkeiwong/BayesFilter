"""Independent analytic checks for the M13 diagnostic comparison law."""
import math

import pytest
import tensorflow as tf

from bayesfilter.testing import surrogate_hmc_m13_posterior_assessment as a
from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import (
    payload_hash, reference_summary,
)


def test_linear_density_moments_and_cdf_match_polynomial_integrals():
    axis = tf.constant([0., .2, .6, 1.], a.F64)
    density = 2*axis
    mean, sd = a.linear_density_moments(axis, density)
    assert float(mean) == pytest.approx(2/3, abs=1e-14)
    assert float(sd)**2 == pytest.approx(1/18, abs=1e-14)
    cuts = tf.constant([-.1, .3, .8, 1.1], a.F64)
    tf.debugging.assert_near(a.piecewise_cdf(axis, density, cuts),
                            tf.constant([0., .09, .64, 1.], a.F64), atol=1e-14)


def test_bilinear_sampler_matches_nonuniform_analytic_density():
    # f(x,y)=4xy on [0,1]^2: independent Beta(2,1) coordinates.
    axis = tf.constant([0., .5, 1.], a.F64)
    density = 4*axis[:, None]*axis[None, :]
    points = a.draw_bilinear(axis, density, tf.constant([100000]),
                              tf.constant([719, 21]))
    mean = tf.reduce_mean(points, 0)
    assert all(abs(float(x)-2/3) < .004 for x in mean)
    probabilities = tf.reduce_mean(tf.cast(points <= .5, a.F64), 0)
    assert all(abs(float(x)-.25) < .005 for x in probabilities)


def test_reference_summary_removes_prior_and_preserves_replication_scale():
    axis = tf.linspace(tf.constant(-1., a.F64), tf.constant(1., a.F64), 9)
    prior_cancel = (axis[:, None]**2+axis[None, :]**2)/(2*math.log(2.)**2)
    # Different replicate normalizers, identical uniform posterior shapes.
    logs = prior_cancel[..., None]+tf.math.log(tf.constant([1., 2., 3.], a.F64))
    result = reference_summary(axis, logs)
    tf.debugging.assert_near(result["mean"], tf.zeros([2], a.F64), atol=1e-13)
    tf.debugging.assert_near(result["sd"],
                             tf.fill([2], tf.constant(1/math.sqrt(3), a.F64)), atol=1e-13)
    tf.debugging.assert_near(result["cuts"],
        tf.constant([[-.5, 0., .5], [-.5, 0., .5]], a.F64), atol=1e-12)
    assert float(tf.reduce_max(result["mcse"])) < 1e-12
    with pytest.raises(tf.errors.InvalidArgumentError):
        reference_summary(tf.constant([-1., -.5, 1.], a.F64), logs[:3, :3])


def test_reference_identity_excludes_only_its_own_digest():
    payload = dict(mean=[0., 1.], reference_sha256="old")
    digest = payload_hash(payload)
    assert payload_hash(payload | dict(reference_sha256="new")) == digest
    assert payload_hash(payload | dict(mean=[0., 2.])) != digest


def test_assessment_rejects_unqualified_or_mismatched_coverage():
    points = tf.zeros([4, 1000, 2], a.F64)
    with pytest.raises(ValueError, match="uncertainty qualification"):
        a.assess(points, dict(reference_sha256="reference"),
                 calibration=dict(method=a.METHOD, qualified=False))


def test_binomial_coverage_bound_matches_exact_all_success_probability():
    from bayesfilter.testing.surrogate_hmc_m13_posterior_coverage import coverage_lower
    assert coverage_lower(256, 256) == pytest.approx(.05**(1/256), abs=1e-13)
    assert coverage_lower(0, 256) == 0.
    assert coverage_lower(250, 256) < coverage_lower(251, 256)
    with pytest.raises(ValueError):
        coverage_lower(257, 256)


def test_compiled_sticky_controls_hold_or_refresh_without_changing_support():
    from bayesfilter.testing.surrogate_hmc_m13_posterior_coverage import make_draw_graph
    axis = tf.constant([0., .5, 1.], a.F64)
    draw = make_draw_graph(axis, tf.ones([3, 3], a.F64), batch=2, count=24)
    held = draw(tf.constant(0., a.F64), tf.constant([993, 1]))
    tf.debugging.assert_equal(held, tf.repeat(held[:, :, :1], 24, axis=2))
    refreshed = draw(tf.constant(1., a.F64), tf.constant([993, 2]))
    assert bool(tf.reduce_all(tf.reduce_any(refreshed[:, :, 1:] != refreshed[:, :, :-1], -1)))
    assert bool(tf.reduce_all((refreshed >= 0.) & (refreshed <= 1.)))
    assert draw.experimental_get_tracing_count() == 1


def test_bounded_inverse_cdf_matches_independent_upper_bound_with_zero_mass_cells():
    cdf = tf.constant([0., .1, .1, .5, .7, 1.], a.F64)
    values = tf.constant([0., .099, .1, .5, .699, .7, .999999], a.F64)
    expected = tf.searchsorted(cdf, values, side="right", out_type=tf.int32)
    @tf.function(input_signature=[tf.TensorSpec([7], a.F64)], jit_compile=True)
    def compiled(x):
        return a.inverse_cdf_indices(cdf, x)
    tf.debugging.assert_equal(compiled(values), expected)
