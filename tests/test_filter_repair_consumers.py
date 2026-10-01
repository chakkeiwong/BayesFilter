"""Consumer checks for native state recurrences and analytical model callbacks."""

import numpy as np
import tensorflow as tf

from bayesfilter.hardbound.joint_target_tf import (
    GATE,
    gate_joint_log_prob,
    gate_joint_log_prob_batched,
    gate_states_from_raws,
)
from bayesfilter.independent_score.sir_observation_simulator_tf import (
    INITIAL_MEAN,
    _transition_mean,
    simulate_observation_paths_from_noise,
)


def test_hardbound_gate_preserves_recurrence_and_dense_gradients():
    params = tf.constant([[0.05, -1.], [-0.1, -0.5]], tf.float64)
    raws = tf.reshape(tf.linspace(tf.constant(-0.3, tf.float64), 0.5, 2 * (GATE.horizon + 1)), [2, GATE.horizon + 1])
    y = tf.linspace(tf.constant(-0.1, tf.float64), 0.2, GATE.horizon)
    expected = []
    for index in range(2):
        x = params[index, 0] + GATE.p0_sd * raws[index, 0]
        states = []
        for date in range(GATE.horizon):
            x = params[index, 0] + GATE.phi * (x - params[index, 0]) + GATE.q_sd * raws[index, date + 1]
            states.append(x)
        np.testing.assert_allclose(gate_states_from_raws(params[index, 0], raws[index]), states, atol=1e-13)
        expected.append(gate_joint_log_prob(y, params[index], raws[index]))

    def run(parameters, noise):
        with tf.GradientTape() as tape:
            tape.watch((parameters, noise))
            value = gate_joint_log_prob_batched(y, parameters, noise)
            total = tf.reduce_sum(value)
        dp, dn = tape.gradient(total, (parameters, noise))
        return value, tf.convert_to_tensor(dp), tf.convert_to_tensor(dn)
    compiled = tf.function(run, input_signature=[tf.TensorSpec(params.shape, tf.float64), tf.TensorSpec(raws.shape, tf.float64)], jit_compile=True, autograph=False)
    value, dp, dn = compiled(params, raws)
    np.testing.assert_allclose(value, expected, atol=1e-12)
    assert bool(tf.reduce_all(tf.math.is_finite(dp)))
    assert bool(tf.reduce_all(tf.math.is_finite(dn)))
    assert compiled.experimental_get_tracing_count() == 1


def test_sir_batched_native_time_matches_explicit_reference():
    from bayesfilter.independent_score.sir_observation_simulator_tf import (
        BASE_KAPPA,
        BASE_NU,
        BASE_OBSERVATION_SCALE,
        STEP,
        SUBSTEPS,
        _rhs,
    )
    batch, horizon = 2, 3
    initial = tf.reshape(tf.sin(tf.cast(tf.range(batch * 18), tf.float64)), [batch, 18])
    process = tf.reshape(0.01 * tf.cos(tf.cast(tf.range(batch * horizon * 18), tf.float64)), [batch, horizon, 18])
    observation = tf.zeros([batch, horizon, 9], tf.float64)
    state = INITIAL_MEAN[None] + initial
    expected = []
    for date in range(horizon):
        reference = state
        for _ in range(SUBSTEPS):
            k1 = _rhs(reference, BASE_KAPPA, BASE_NU)
            k2 = _rhs(reference + 0.5 * STEP * k1, BASE_KAPPA, BASE_NU)
            k3 = _rhs(reference + 0.5 * STEP * k2, BASE_KAPPA, BASE_NU)
            # Preserve the source simulator's half-step fourth stage. Replacing
            # it with standard RK4 changes the simulated model in this campaign.
            k4 = _rhs(reference + 0.5 * STEP * k3, BASE_KAPPA, BASE_NU)
            reference = reference + STEP / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        np.testing.assert_allclose(_transition_mean(state, BASE_KAPPA, BASE_NU), reference, atol=1e-12)
        latent = reference + process[:, date]
        state = tf.reshape(tf.stack([tf.maximum(latent[:, 0::2], 0.), latent[:, 1::2]], axis=2), tf.shape(latent))
        expected.append(state[:, 1::2] + BASE_OBSERVATION_SCALE * observation[:, date])
    value = simulate_observation_paths_from_noise(tf.zeros([3], tf.float64), initial, process, observation)
    np.testing.assert_allclose(value, tf.stack(expected, axis=1), atol=1e-12)
