"""Independent eager-reference parity for complete mapped TT recurrences."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_adapted_tf import run_value_filter_branch_axis_adapted_reference as run_value_filter_branch_axis_adapted
from bayesfilter.highdim.squared_tt_engine_adapted_xla_tf import (
    make_value_filter_branch_axis_adapted_xla, run_value_filter_branch_axis_adapted_xla,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import run_value_filter_branch_axis_gaussian_reference as run_value_filter_branch_axis_gaussian
from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import _run_value_filter_branch_axis_gaussian_xla

D = tf.float64


class ScalarGaussianAdapter:
    state_dim = 1

    def initial_log_density(self, x):
        return -.5 * tf.reduce_sum(x*x/.1 + tf.math.log(tf.constant(2*np.pi*.1, D)), axis=-1)

    def transition_log_density(self, x, previous):
        return -.5 * tf.reduce_sum((x-.8*previous)**2/.05 + tf.math.log(tf.constant(2*np.pi*.05, D)), axis=-1)

    def observation_log_density(self, x, y):
        return -.5 * tf.reduce_sum((x-y)**2/.2 + tf.math.log(tf.constant(2*np.pi*.2, D)), axis=-1)


def fixture(horizon):
    adapter = ScalarGaussianAdapter()
    observations = tf.zeros([horizon,1], D)
    config = EngineConfig(basis_degree=2, rank=2, row_count=64, sweeps=2,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3., seed=7781, row_design="sobol")
    hint = lambda date,y: (tf.zeros([2], D), tf.eye(2,dtype=D)*.005)
    return adapter, observations, config, hint


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("horizon", [1,4])
def test_adapted_value_preserves_full_eager_recurrence(jit, horizon):
    adapter, observations, config, hint = fixture(horizon)
    expected, ed = run_value_filter_branch_axis_adapted(adapter, observations, config,
        predictive_moment_hint=hint)
    actual, ad = run_value_filter_branch_axis_adapted_xla(adapter, observations, config,
        predictive_moment_hint=hint, jit_compile=jit)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose([d["log_increment"] for d in ad], [d["log_increment"] for d in ed],
        rtol=1e-12, atol=1e-12)
    call = make_value_filter_branch_axis_adapted_xla(adapter, observations.shape, config, jit_compile=jit)
    assert call is make_value_filter_branch_axis_adapted_xla(adapter, observations.shape, config, jit_compile=jit)
    assert call.experimental_get_tracing_count() == 1


def test_adapted_invalid_hints_fail_closed():
    adapter, observations, config, _ = fixture(3)
    hint = lambda date,y: (tf.ones([2],D)*10., tf.eye(2,dtype=D))
    with pytest.raises(ValueError, match="containment|non-finite"):
        run_value_filter_branch_axis_adapted_xla(adapter, observations, config, predictive_moment_hint=hint)


@pytest.mark.parametrize("jit",[False,True])
@pytest.mark.parametrize("horizon",[1,4])
@pytest.mark.parametrize("nu",[None,8.])
def test_gaussian_value_preserves_eager_recurrence(jit,horizon,nu):
    adapter,observations,config,_ = fixture(horizon)
    initial = lambda y: (tf.zeros([1],D),tf.eye(1,dtype=D)*.1)
    hint = lambda t,y: (tf.zeros([2],D),tf.eye(2,dtype=D)*.1)
    expected,ed = run_value_filter_branch_axis_gaussian(adapter,observations,config,
        predictive_moment_hint=hint,initial_moment_hint=initial,defensive_nu=nu)
    actual,ad,_ = _run_value_filter_branch_axis_gaussian_xla(adapter,observations,config,
        predictive_moment_hint=hint,initial_moment_hint=initial,defensive_nu=nu,jit_compile=jit)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose([d["log_increment"] for d in ad],[d["log_increment"] for d in ed],
        rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("family", ["bounded", "adapted", "gaussian"])
def test_public_filter_defaults_reach_complete_compiled_recurrence(family, monkeypatch):
    import bayesfilter.highdim.squared_tt_engine_adapted_tf as adapted
    import bayesfilter.highdim.squared_tt_engine_adapted_xla_tf as adapted_xla
    import bayesfilter.highdim.squared_tt_engine_gaussian_tf as gaussian
    import bayesfilter.highdim.squared_tt_gaussian_native_tf as gaussian_xla
    import bayesfilter.highdim.squared_tt_engine_v0_tf as bounded
    import bayesfilter.highdim.squared_tt_engine_xla_tf as bounded_xla

    adapter, observations, config, hint = fixture(3)
    if family == "bounded":
        module, name = bounded_xla, "make_value_filter_branch_axis_xla"
        public, reference, kwargs = bounded.run_value_filter_branch_axis, bounded.run_value_filter_branch_axis_reference, {}
    elif family == "adapted":
        module, name = adapted_xla, "make_value_filter_branch_axis_adapted_xla"
        public, reference = adapted.run_value_filter_branch_axis_adapted, adapted.run_value_filter_branch_axis_adapted_reference
        kwargs = dict(predictive_moment_hint=hint)
    else:
        module, name = gaussian_xla, "make_gaussian_value_filter"
        public, reference = gaussian.run_value_filter_branch_axis_gaussian, gaussian.run_value_filter_branch_axis_gaussian_reference
        kwargs = dict(predictive_moment_hint=hint,
            initial_moment_hint=lambda _: (tf.zeros([1], D), tf.eye(1, dtype=D)*.1))
    programs = []
    factory = getattr(module, name)

    def record(*args, **kwargs):
        program = factory(*args, **kwargs)
        programs.append(program)
        return program

    monkeypatch.setattr(module, name, record)
    value, _ = public(adapter, observations, config, **kwargs)
    expected, _ = reference(adapter, observations, config, **kwargs)
    np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-10)
    assert programs and programs[0]._jit_compile
    assert programs[0].input_signature is not None
    assert programs[0].experimental_get_tracing_count() == 1
