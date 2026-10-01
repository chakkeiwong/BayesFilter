"""Original finite-update and calibration parity at the Austria consumer."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import zhao_cui_austria_sir_lane_b_tf as lane
from bayesfilter.highdim.stochastic_density_training import make_adam_optimizer
from tests.highdim.test_p75_stochastic_density_training import _batch
from tests.test_filter_repair_remaining_routes import _original
from tests.test_filter_repair_stochastic_training import _authority

D = tf.float64


def test_existing_lane_b_loss_and_optimizer_have_stable_compilation_boundary():
    actual, expected = _authority()
    before = _original("zhao_cui_austria_sir_lane_b_tf")
    functions = (lane.make_compiled_train_step(actual, make_adam_optimizer(actual.config)),
                 before.make_compiled_train_step(expected, make_adam_optimizer(expected.config)))
    batch = _batch()
    inputs = (batch.points, batch.target_values, batch.weights)
    for _ in range(2):
        output = functions[0](*inputs)
        reference = functions[1].python_function(*inputs)
        for value, authority in zip(output, reference, strict=True):
            np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
        for value, authority in zip(actual.variables, expected.variables, strict=True):
            np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    assert functions[0].experimental_get_tracing_count() == 1
    assert "HloModule" in functions[0].experimental_get_compiler_ir(*inputs)(stage="hlo")


def test_compiled_calibration_preserves_scale_and_parameter_updates():
    actual, expected = _authority()
    before = _original("zhao_cui_austria_sir_lane_b_tf")
    for target in (.4, 1.7):
        log_target = tf.math.log(tf.constant(target, D))
        scale = lane.calibrate_trainer_normalizer(actual, log_target)
        reference = before.calibrate_trainer_normalizer(expected, log_target)
        np.testing.assert_allclose(scale, reference, atol=1e-10, rtol=1e-10)
        for value, authority in zip(actual.variables, expected.variables, strict=True):
            np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    assert actual._calibration_program.experimental_get_tracing_count() == 1
    assert "HloModule" in actual._calibration_program.experimental_get_compiler_ir(log_target)(stage="hlo")
    saved = tuple(tf.identity(core) for core in actual.variables)
    with pytest.raises(tf.errors.InvalidArgumentError, match="invalid normalizer"):
        lane.calibrate_trainer_normalizer(actual, tf.constant(-100., D))
    for value, authority in zip(actual.variables, saved, strict=True):
        np.testing.assert_array_equal(value, authority)


def test_compiled_lane_b_rejects_nonfinite_loss_before_assignments():
    trainer, _expected = _authority()
    optimizer = make_adam_optimizer(trainer.config)
    step = lane.make_compiled_train_step(trainer, optimizer)
    state = (*trainer.variables, *optimizer.variables)
    before = tuple(tf.identity(value) for value in state)
    batch = _batch()
    result = step(batch.points, tf.fill(batch.target_values.shape, tf.constant(float("nan"), D)), batch.weights)
    assert not bool(tf.math.is_finite(result[4]))
    for value, reference in zip(state, before, strict=True):
        np.testing.assert_array_equal(value, reference)
