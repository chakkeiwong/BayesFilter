"""Fixed-design fitting pullbacks against the pre-lazy implementation."""

import importlib.util
import subprocess

import pytest
import tensorflow as tf

from bayesfilter.highdim.fixed_tt_native_fit_tf import NativeFixedTTFit
from bayesfilter.highdim.tt import TTCore
from tests.highdim.test_fixed_branch_fit import _config, _grid2, _product_basis

D = tf.float64


@pytest.fixture(scope="module")
def original():
    path = "bayesfilter/highdim/fixed_tt_native_fit_tf.py"
    source = subprocess.check_output(["git", "show", f"c4351f8c:{path}"], text=True)
    spec = importlib.util.spec_from_loader("fixed_fit_pre_lazy_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "fixed_fit_pre_lazy_reference.py", "exec"), module.__dict__)  # noqa: S102
    return module.NativeFixedTTFit


def inputs():
    basis = _product_basis((1, 2))
    points = _grid2()
    weights = tf.linspace(tf.constant(.7, D), tf.constant(1.3, D), points.shape[0])
    config = _config((1, 2, 1), ridge=1e-5, max_sweeps=1)
    cores = (TTCore(tf.constant([[[1., .3], [.2, .4]]], D)),
             TTCore(tf.constant([[[.9], [.4], [.2]], [[.6], [.1], [.5]]], D)))
    target = 1. + .2 * points[:, 0] + .3 * points[:, 1]**2
    return basis, points, weights, config, cores, target


@pytest.mark.parametrize("jit_compile", [False, True])
def test_value_only_fit_never_builds_accepted_update_pullbacks(monkeypatch, jit_compile):
    basis, points, weights, config, cores, target = inputs()

    def unused_pullback(*args, **kwargs):
        raise AssertionError("value-only fitting traced an unused pullback")

    with monkeypatch.context() as patch:
        patch.setattr(tf.GradientTape, "gradient", unused_pullback)
        native = NativeFixedTTFit(basis, points, weights, config, cores, jit_compile=jit_compile)
        program = tf.function(native, input_signature=[tf.TensorSpec(target.shape, D),
            tf.TensorSpec(native.initial.shape, D)], jit_compile=jit_compile, autograph=False)
        result = program(target, native.initial)
        assert bool(result["valid"])
    @tf.function(input_signature=[tf.TensorSpec(target.shape, D)],
                 jit_compile=jit_compile, autograph=False)
    def score(value):
        with tf.GradientTape() as tape:
            tape.watch(value)
            result = program.python_function(value, native.initial)
            loss = tf.reduce_sum(result["cores"])
        return tape.gradient(loss, value)

    gradient = score(target)
    assert gradient is not None and bool(tf.reduce_all(tf.math.is_finite(gradient)))
    assert bool(tf.reduce_any(gradient != 0.))
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
def test_lazy_fit_preserves_heterogeneous_core_target_and_rejection_derivatives(
        original, jit_compile, rejected):
    basis, points, weights, config, cores, target = inputs()
    if rejected:
        target = tf.tensor_scatter_nd_update(target, [[0]], [tf.constant(float("nan"), D)])
    outcomes = []
    for create in (original, NativeFixedTTFit):
        native = create(basis, points, weights, config, cores, jit_compile=jit_compile)
        initial = native.initial
        def scored_program(fit):
            @tf.function(input_signature=[tf.TensorSpec(target.shape, D),
                tf.TensorSpec(fit.initial.shape, D)], jit_compile=jit_compile, autograph=False)
            def program(value, starting):
                with tf.GradientTape() as tape:
                    tape.watch((starting, value))
                    result = fit(value, starting)
                    loss = tf.reduce_sum(result["cores"] * tf.reshape(
                        tf.linspace(tf.constant(.2, D), tf.constant(.8, D), starting.shape.num_elements()),
                        starting.shape))
                return result, tape.gradient(loss, (starting, value))

            return program

        result, gradient = scored_program(native)(target, initial)
        assert bool(result["valid"]) is not rejected
        outcomes.append((result["cores"], result["codes"], gradient))
    tf.debugging.assert_equal(outcomes[1][1], outcomes[0][1])
    for actual, expected in zip(tf.nest.flatten(outcomes[1]), tf.nest.flatten(outcomes[0]), strict=True):
        assert actual is not None and expected is not None
        if actual.dtype.is_floating:
            tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)
    if rejected:
        tf.debugging.assert_equal(outcomes[1][2][1], tf.zeros_like(target))
    else:
        assert bool(tf.reduce_any(outcomes[1][2][0] != 0.))
        assert bool(tf.reduce_any(outcomes[1][2][1] != 0.))
