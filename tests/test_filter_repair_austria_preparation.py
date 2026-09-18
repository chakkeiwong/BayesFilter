"""Pinned recurrence and independent derivative checks for Austria SIR setup."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import (
    zhao_cui_austria_sir_parameter_density_training_tf as candidate,
)
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def test_strict_scalar_score_rounding(request):
    """Localize strict consumer parity without changing its numerical gate."""
    from bayesfilter.highdim.sir_latent_preclip_tf import (
        latent_preclip_zhao_cui_sir_austria_model,
    )
    from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
        generate_sealed_lane_b_dataset,
    )

    theta = tf.constant([[0., 0., 0.], [.03, -.02, .01], [-.03, .02, -.01]], D)
    inputs = (theta, tf.random.stateless_normal([16, 18], [8101, 1], dtype=D),
              tf.random.stateless_normal([16, 18], [8101, 2], dtype=D),
              generate_sealed_lane_b_dataset()[1][0])
    actual = candidate.batch_native_t1_from_common_noise(*inputs)
    eager = candidate.batch_native_t1_from_common_noise.python_function(*inputs)
    model = latent_preclip_zhao_cui_sir_austria_model()
    state = actual["z0"]
    means, tangents = tf.function(candidate._batch_native_transition_mean_and_jacobian,
        input_signature=[tf.TensorSpec(theta.shape, D), tf.TensorSpec(state.shape, D)],
        jit_compile=True, autograph=False)(theta, state)
    expected = []
    details = []
    for index in range(3):
        scalar_mean, scalar_jac = model.physical_model.transition_mean_parameter_jacobian(
            theta[index], state[index])
        scalar_score = (
            model.transition_log_density_parameter_score(theta[index], state[index], actual["z1"][index], 1)
            + model.observation_log_density_parameter_score(theta[index], actual["z1"][index], inputs[3], 1))
        expected.append(scalar_score)
        details.append({"mean_max_error": float(tf.reduce_max(tf.abs(means[index] - scalar_mean))),
            "jacobian_max_error": float(tf.reduce_max(tf.abs(tangents[index] - tf.transpose(scalar_jac, [1, 2, 0])))),
            "score_max_error": tf.reduce_max(tf.abs(actual["complete_data_score"][index] - scalar_score), axis=0).numpy().tolist(),
            "z1_eager_max_error": float(tf.reduce_max(tf.abs(actual["z1"][index] - eager["z1"][index])))})
    junit = request.config.getoption("xmlpath")
    if junit:
        (Path(junit).parent / "austria-rounding.json").write_text(
            json.dumps(details, indent=2, allow_nan=False) + "\n")
    tf.debugging.assert_near(actual["complete_data_score"], tf.stack(expected),
                             atol=6e-13, message=str(details))


@pytest.mark.parametrize("count", [3, 7])
@pytest.mark.parametrize("jit", [False, True])
def test_austria_complete_data_score_and_native_substeps(count, jit):
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    theta = tf.constant([[.02, -.03, .01], [-.01, .04, -.02]], D)
    noise = .05 * tf.reshape(tf.sin(tf.cast(tf.range(count * 18), D)), [count, 18])
    inputs = (theta, noise, .4 * noise, tf.linspace(tf.constant(4., D), 8., 9))
    specs = tuple(tf.TensorSpec(x.shape, x.dtype) for x in inputs)
    reference = tf.function(before.batch_native_t1_from_common_noise,
                            input_signature=specs, jit_compile=jit, autograph=False)
    call = tf.function(candidate.batch_native_t1_from_common_noise.python_function,
                       input_signature=specs, jit_compile=jit, autograph=False)
    actual, expected = call(*inputs), reference(*inputs)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-10, atol=1e-10)
    direct = candidate.batch_native_t1_from_common_noise(*inputs)
    np.testing.assert_allclose(direct["complete_data_score"], expected["complete_data_score"],
                               rtol=1e-10, atol=1e-10)
    state = actual["z0"]
    means, tangents = candidate._batch_native_transition_mean_and_jacobian(theta, state)
    direction = tf.constant([.1, -.05, .03], D)
    delta = 1e-5
    plus = candidate._batch_native_transition_mean_and_jacobian(theta + delta * direction, state)[0]
    minus = candidate._batch_native_transition_mean_and_jacobian(theta - delta * direction, state)[0]
    np.testing.assert_allclose(tf.einsum("bnsp,p->bns", tangents, direction),
                               (plus - minus) / (2 * delta), atol=1e-8, rtol=1e-7)
    # A rounding identity must not sever the ordinary differentiable value path.
    @tf.function(input_signature=[tf.TensorSpec(theta.shape, D), tf.TensorSpec(state.shape, D)],
                 jit_compile=jit, autograph=False)
    def ordinary_gradient(parameters, previous):
        with tf.GradientTape() as tape:
            tape.watch(parameters)
            objective = tf.reduce_sum(candidate._batch_native_transition_mean_and_jacobian(
                parameters, previous)[0])
        return tape.gradient(objective, parameters)

    np.testing.assert_allclose(ordinary_gradient(theta, state),
        tf.reduce_sum(tangents, axis=[1, 2]), atol=1e-10, rtol=1e-10)
    assert bool(tf.reduce_all(tf.math.is_finite(means)))
    _graph(call)
    assert "HloModule" in candidate.batch_native_t1_from_common_noise.experimental_get_compiler_ir(
        *inputs)(stage="hlo")
