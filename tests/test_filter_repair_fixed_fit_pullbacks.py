"""Fixed-design fitting pullbacks against the pre-lazy implementation."""

import importlib.util
import subprocess

import pytest
import tensorflow as tf

from bayesfilter.highdim.fixed_tt_native_fit_tf import NativeFixedTTFit
from bayesfilter.highdim.tt import TTCore
from tests.highdim.test_fixed_branch_fit import _config, _grid2, _product_basis
from tests.test_filter_repair_remaining_routes import _graph

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


@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("case", ["shared", "heterogeneous", "mixed_shared", "resource_rejection"])
def test_shared_update_graph_preserves_order_history_and_full_fixed_design_pullback(
        original, jit_compile, case):
    dimension = 4
    ranks = (1, 1, 1, 1, 1) if case == "shared" else (1, 1, 2, 2, 1)
    degrees = (1, 1, 1, 1) if case == "shared" else (1, 2, 2, 1)
    if case == "mixed_shared":
        dimension, ranks, degrees = 5, (1, 2, 2, 2, 2, 1), (1, 2, 2, 2, 1)
    basis = _product_basis(degrees)
    if case == "mixed_shared":
        points = tf.reshape(.8 * tf.sin(tf.cast(tf.range(12 * dimension), D) * .71), [12, dimension])
    else:
        points = tf.reshape(tf.linspace(tf.constant(-.8, D), tf.constant(.9, D), 48), [12, dimension])
    weights = tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), 12)
    # Axis 0 is accepted before the larger axis 1 trips the static budget.
    schedule = tuple(range(dimension)) + tuple(reversed(range(dimension)))
    config = _config(ranks, sweep_order=schedule, ridge=1e-4, max_sweeps=2,
                     column_budget=3 if case == "resource_rejection" else 1000)
    cores = tuple(TTCore(tf.reshape(tf.linspace(tf.constant(.4, D), tf.constant(.9, D),
        ranks[axis] * (degrees[axis]+1) * ranks[axis+1]),
        [ranks[axis], degrees[axis]+1, ranks[axis+1]])) for axis in range(dimension))
    target = 1. + .2 * points[:, 0] + .1 * points[:, 2]**2
    outcomes = []
    for create in (original, NativeFixedTTFit):
        native = create(basis, points, weights, config, cores, jit_compile=jit_compile)

        def scored(fit):
            @tf.function(input_signature=[tf.TensorSpec(target.shape, D),
                tf.TensorSpec(fit.initial.shape, D)], jit_compile=jit_compile, autograph=False)
            def call(value, starting):
                with tf.GradientTape() as tape:
                    tape.watch((value, starting))
                    history = fit(value, starting)
                    loss = tf.reduce_sum(history["cores"] ** 2)
                return history, tape.gradient(loss, (value, starting))
            return call

        program = scored(native)
        history, gradients = program(target, native.initial)
        assert bool(history["valid"]) is (case != "resource_rejection")
        if case == "resource_rejection":
            assert history["codes"].numpy().tolist() == [0, 6] + [-1] * 14
        assert all(gradient is not None for gradient in gradients)
        outcomes.append((history, gradients))
        _graph(program)
        if jit_compile:
            assert "HloModule" in program.experimental_get_compiler_ir(target, native.initial)(stage="hlo")
    for actual, expected in zip(tf.nest.flatten(outcomes[1]), tf.nest.flatten(outcomes[0]), strict=True):
        if actual.dtype.is_floating:
            tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10,
                summarize=48)
        else:
            tf.debugging.assert_equal(actual, expected)


def test_repeated_core_shape_graph_growth_is_linear(record_property):
    sizes = []
    for dimension in (4, 8):
        basis = _product_basis((1,) * dimension)
        points = tf.reshape(tf.linspace(tf.constant(-.8, D), tf.constant(.8, D),
            8 * dimension), [8, dimension])
        weights = tf.ones([8], D)
        cores = tuple(TTCore(tf.constant([[[1.], [.1]]], D)) for _ in range(dimension))
        config = _config((1,) * (dimension+1), ridge=1e-5)
        native = NativeFixedTTFit(basis, points, weights, config, cores)
        program = tf.function(native, input_signature=[tf.TensorSpec([8], D),
            tf.TensorSpec(native.initial.shape, D)], jit_compile=True, autograph=False)
        graph = program.get_concrete_function().graph.as_graph_def()
        sizes.append(len(graph.node) + sum(len(function.node_def) for function in graph.library.function))
        assert bool(program(tf.ones([8], D), native.initial)["valid"])
    print("shared_fit_graph_nodes", sizes)
    record_property("graph_nodes_4d_8d", str(sizes))
    assert sizes[1] < 2.3 * sizes[0]
