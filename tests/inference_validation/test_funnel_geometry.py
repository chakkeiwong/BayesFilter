"""Independent density, finite-difference Hessian and tail geometry checks."""
import math

import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.funnel_geometry import (
    model_to_latent, residual_geometry_program,
)


@pytest.mark.parametrize("kind,strength", [("exact", None), ("partial", 1.), ("partial_half", .5)])
def test_residual_law_and_derivatives_against_original_funnel(kind, strength):
    z = tf.constant([[-4., .2, -.4], [0., -1., 1.], [3., .5, .6]], tf.float64)

    def original_log_density(latent):
        v = 3. * latent[:, 0]
        s = v / 2. if strength is None else 6. * tf.tanh(strength * v / 12.)
        x = tf.exp(s[:, None]) * latent[:, 1:]
        # Centered funnel log density plus independently assembled map Jacobian.
        return (-.5 * (v / 3.)**2 - math.log(3.) - v
                -.5 * tf.reduce_sum(tf.square(x) * tf.exp(-v[:, None]), axis=1)
                + math.log(3.) + 2. * s)

    with tf.GradientTape() as tape:
        tape.watch(z)
        logp = original_log_density(z)
    score = tape.gradient(logp, z)
    program = residual_geometry_program(kind, jit_compile=False)
    report = program(z)
    tf.debugging.assert_near(-report["potential"], logp, rtol=1e-11, atol=1e-11)
    tf.debugging.assert_near(report["score"], score, rtol=1e-11, atol=1e-11)
    columns = []
    for index in range(3):
        delta = tf.one_hot(index, 3, dtype=tf.float64) * 1e-5
        gradients = []
        for sign in (-1., 1.):
            probe = z + sign * delta
            with tf.GradientTape() as tape:
                tape.watch(probe)
                density = original_log_density(probe)
            gradients.append(tape.gradient(density, probe))
        columns.append(-(gradients[1] - gradients[0]) / 2e-5)
    tf.debugging.assert_near(report["hessian"], tf.stack(columns, axis=-1), rtol=1e-7, atol=1e-7)
    program(z[:1])
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("kind", ["exact", "partial", "partial_half"])
def test_model_inverse_roundtrip_and_exact_gaussian_control(kind):
    z = tf.constant([[-3., 1., 2.], [0., -1., 3.], [2., .5, -1.]], tf.float64)
    report = residual_geometry_program(kind, jit_compile=False)(z)
    model = tf.concat((report["v"][:, None],
                       tf.exp(report["log_scale"][:, None]) * z[:, 1:]), axis=1)
    tf.debugging.assert_near(model_to_latent(model, kind), z, rtol=1e-12, atol=1e-12)
    if kind == "exact":
        tf.debugging.assert_equal(report["hessian"], tf.eye(3, batch_shape=[3], dtype=tf.float64))
        tf.debugging.assert_equal(report["score"], -z)


@pytest.mark.parametrize("kind", ["partial", "partial_half"])
def test_partial_map_remains_unbounded_in_negative_tail(kind):
    z = tf.constant([[-4., 0., 0.], [-8., 0., 0.], [-16., 0., 0.]], tf.float64)
    report = residual_geometry_program(kind, jit_compile=False)(z)
    assert bool(tf.reduce_all(report["child_curvature"][1:] > report["child_curvature"][:-1]))
    # s is bounded below by -6: log child curvature = 2s-v >= -12-v.
    assert bool(tf.reduce_all(report["log_child_curvature"] >= -12. - report["v"]))
