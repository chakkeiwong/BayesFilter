"""Fresh same-input reference checks for native SSL-LSTM forecast recurrences."""

import subprocess
import sys
import types
from functools import lru_cache

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear import ssl_lstm_complexity_predictive_tf as complexity
from bayesfilter.nonlinear import ssl_lstm_complexity_target_tf as target
from bayesfilter.nonlinear import ssl_lstm_predictive_tf as predictive
from tests.test_filter_repair_remaining_routes import BASELINE, ROOT, _graph

D = tf.float64


@lru_cache(maxsize=3)
def _before(name):
    path = f"bayesfilter/nonlinear/{name}.py"
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True)
    module = types.ModuleType(f"predictive_reference_{name}")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASELINE}:{path}", "exec"), module.__dict__)  # noqa: S102 - pinned local oracle
    return module


def _compare(actual, expected):
    for value, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        if value.dtype == tf.bool or value.dtype.is_integer:
            np.testing.assert_array_equal(value, reference)
        else:
            np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("q", [1, 2, 5])
def test_synthetic_preparation_preserves_original_philox_stream(q):
    before = _before("ssl_lstm_complexity_target_tf")
    config = target.make_complexity_config(q)
    fixture = target.make_full_fixture(config)
    frozen = before.make_full_fixture(config)
    _compare(fixture, frozen)
    _compare(target.make_synthetic_observations(config, frozen), before.make_synthetic_observations(config, frozen))
    call = target._synthetic_program(config)
    assert "HloModule" in call.experimental_get_compiler_ir(frozen)(stage="hlo")
    _graph(call)


def _forecast_inputs(count, replications=2):
    return (tf.tile(tf.constant([[.35, -.08, .65, .05]], D), [count, 1]),
        tf.zeros([count, predictive.STATE_DIM], D),
        tf.eye(predictive.STATE_DIM, batch_shape=[count], dtype=D) * .2,
        tf.reshape(tf.sin(tf.cast(tf.range(count * replications * predictive.STATE_DIM), D)),
                   [count, replications, predictive.STATE_DIM]),
        tf.ones([count, replications, predictive.FORECAST_HORIZON, predictive.LATENT_DIM], D) * .12,
        tf.ones([count, replications, predictive.FORECAST_HORIZON, predictive.OBSERVATION_DIM], D) * -.23)


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("jit", [False, True])
def test_draw_and_time_recurrence_matches_pinned_forecast(count, jit):
    before = _before("ssl_lstm_predictive_tf")
    config = predictive.SSLLSTMForecastConfig()
    inputs = _forecast_inputs(count)
    specs = tuple(tf.TensorSpec(x.shape, x.dtype) for x in inputs)
    reference = tf.function(lambda *args: before._forecast_batch_core(*args, config),
                            input_signature=specs, jit_compile=jit, autograph=False)
    candidate = tf.function(lambda *args: predictive._forecast_batch_core(*args, config),
                            input_signature=specs, jit_compile=jit, autograph=False)
    _compare(candidate(*inputs), reference(*inputs))
    if jit:
        assert "HloModule" in candidate.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(candidate)


@pytest.mark.parametrize("count", [2, 5])
def test_innovation_banks_preserve_seeds_and_original_normal_stream(count):
    before = _before("ssl_lstm_predictive_tf")
    config = predictive.SSLLSTMForecastConfig()
    with tf.device("/CPU:0"):
        old = before.make_ssl_lstm_innovation_bank(before.SSLLSTMForecastConfig(), count,
                                                 [17, 31], "independent_arm", 2)
        new = predictive.make_ssl_lstm_innovation_bank(config, count, [17, 31], "independent_arm", 2)
    _compare((new.terminal_standard_normal, new.process_standard_normal, new.observation_standard_normal,
              new.derived_seeds),
             (old.terminal_standard_normal, old.process_standard_normal, old.observation_standard_normal,
              old.derived_seeds))
    call = predictive._innovation_bank_program(count, config.replication_count)
    assert "HloModule" in call.experimental_get_compiler_ir(
        tf.constant([17, 31]), tf.constant(211), tf.constant(2))(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("q,count,horizon", [(1, 2, 3), (2, 3, 5)])
def test_complexity_innovations_preserve_original_stream(q, count, horizon):
    root = tf.constant([23, 41])
    state_dim, replications = 3 * q, 2
    call = complexity._innovation_program(q, state_dim, count, replications, horizon)
    shapes = ([count, replications, state_dim], [count, replications, horizon, q],
              [count, replications, horizon])
    with tf.device("/CPU:0"):
        expected = tuple(tf.random.stateless_normal(shape, complexity._fold(root, family),
            dtype=D, alg="philox") for shape, family in zip(shapes, (5101, 5102, 5103), strict=True))
        _compare(call(root), expected)
    assert "HloModule" in call.experimental_get_compiler_ir(root)(stage="hlo")


@pytest.mark.parametrize("count", [32, 64])
def test_calibration_native_chunks_match_original_orchestration(count, monkeypatch):
    before = _before("ssl_lstm_complexity_predictive_tf")
    # The per-draw numerical kernel has separate pinned-source parity checks.
    monkeypatch.setattr(before, "forecast_complexity_conditional_moments",
                        complexity.forecast_complexity_conditional_moments)
    expected = before.calibrate_complexity_horizon_scales(q=1, draw_count_per_chain=count)
    actual = complexity.calibrate_complexity_horizon_scales(q=1, draw_count_per_chain=count)
    _compare((actual.center, actual.scale), (expected.center, expected.scale))
    assert actual.calibration_signature == expected.calibration_signature
    model = target.complexity_posterior_target(1)
    call = complexity._calibration_forecast_program(model, count)
    assert call._jit_compile
    _graph(call)
    generator = complexity._calibration_innovations_program(1, 3, count)
    seeds = tf.constant(complexity.calibration_seed_roots(1), tf.int32)
    _, *innovations = generator(seeds)
    assert "HloModule" in generator.experimental_get_compiler_ir(seeds)(stage="hlo")
    assert "HloModule" in call.experimental_get_compiler_ir(
        model.config.prior_center, *innovations)(stage="hlo")


def test_forecast_graph_does_not_grow_with_draw_count():
    config = predictive.SSLLSTMForecastConfig()
    counts = [len(_graph(predictive.ssl_lstm_forecast_compiled_program(config, count))) for count in (2, 4)]
    assert counts[0] == counts[1]


@pytest.mark.parametrize("count,chunk", [(2, 2), (4, 2), (5, 2)])
def test_chunked_enclosing_forecast_and_parameter_embedding(count, chunk):
    config = predictive.SSLLSTMForecastConfig()
    inputs = _forecast_inputs(count)
    call = predictive._chunked_forecast_program(config, count, chunk)
    actual, embedded = call(*inputs)
    expected = predictive.ssl_lstm_forecast_compiled_program(config, count)(*inputs)
    _compare(actual, expected)
    np.testing.assert_array_equal(embedded, tf.stack([config.posterior_config.parameter_mask.embed(row)
        for row in tf.unstack(inputs[0])]))
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("horizon", [2, 4])
def test_complete_complexity_forecast_same_input_parity(horizon):
    before = _before("ssl_lstm_complexity_predictive_tf")
    model = target.complexity_posterior_target(1)
    inputs = (model.config.prior_center, tf.ones([2, 3], D) * .1,
              tf.ones([2, horizon, 1], D) * -.15, tf.ones([2, horizon], D) * .2)
    specs = tuple(tf.TensorSpec(x.shape, x.dtype) for x in inputs)
    reference = tf.function(lambda *args: before._single_forecast_core(*args, model, horizon),
                            input_signature=specs, jit_compile=True, autograph=False)
    candidate = tf.function(lambda *args: complexity._single_forecast_core(*args, model, horizon),
                            input_signature=specs, jit_compile=True, autograph=False)
    _compare(candidate(*inputs), reference(*inputs))
    assert "HloModule" in candidate.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(candidate)


def test_complexity_graph_reference_has_no_nested_xla():
    model = target.complexity_posterior_target(1, jit_compile=False)
    program = complexity.complexity_forecast_compiled_program(
        model, draw_count=2, replication_count=2, horizon=2)
    diagnostic = tf.function(program.python_function, input_signature=program.input_signature,
                             jit_compile=False, autograph=False)
    graph = diagnostic.get_concrete_function().graph.as_graph_def()
    assert not any(fn.attr.get("_XlaMustCompile") and fn.attr["_XlaMustCompile"].b
                   for fn in graph.library.function)
    inputs = (tf.tile(model.config.prior_center[None, :], [2, 1]),
              tf.ones([2, 2, 3], D) * .1, tf.ones([2, 2, 2, 1], D) * -.15,
              tf.ones([2, 2, 2], D) * .2)
    _compare(diagnostic(*inputs), program(*inputs))
    assert bool(tf.reduce_all(diagnostic(*inputs)[-1]))
