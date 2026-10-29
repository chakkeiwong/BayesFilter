"""Pinned parity for shared owned-Legendre basis graphs and their pullbacks."""

import importlib.util
import subprocess
from functools import partial

import pytest
import tensorflow as tf

from bayesfilter import highdim
from bayesfilter.highdim import tt_native_control_tf as candidate

D = tf.float64


@pytest.fixture(scope="module")
def previous():
    source = subprocess.check_output(["git", "show",
        "1da3170e:bayesfilter/highdim/tt_native_control_tf.py"], text=True)
    spec = importlib.util.spec_from_loader("basis_before_shared_legendre", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "basis_before_shared_legendre.py", "exec"), module.__dict__)  # noqa: S102
    return module


def _inputs(dimension=3, *, lebesgue=False, family="common"):
    class CustomLegendre(highdim.LegendreBasis1D):
        def evaluate(self, points):
            return super().evaluate(points) + tf.constant(.125, D)

        def mass_matrix(self, measure):
            return super().mass_matrix(measure) * tf.constant(1.25, D)

    convention = highdim.MeasureConvention(
        density_measure=(highdim.DensityMeasure.REFERENCE_LEBESGUE if lebesgue
                         else highdim.DensityMeasure.REFERENCE_MEASURE),
        mass_measure=(highdim.MassMeasure.REFERENCE_LEBESGUE if lebesgue
                      else highdim.MassMeasure.REFERENCE_MEASURE), reference_weight_name="omega")
    basis_type = CustomLegendre if family == "custom" else highdim.LegendreBasis1D
    parts = tuple(basis_type(highdim.BoundedInterval(-1. - .2*axis, 1. + .4*axis),
        1 + axis % 2 if family == "heterogeneous" else 2) for axis in range(dimension))
    basis = highdim.ProductBasis(parts, convention)
    ranks = (1, *(2 + axis % 2 for axis in range(dimension-1)), 1)
    cores = tuple(highdim.TTCore(tf.reshape(.2*tf.sin(tf.cast(tf.range(
        ranks[axis]*part.basis_dim*ranks[axis+1]), D)) + .3,
        [ranks[axis], part.basis_dim, ranks[axis+1]])) for axis, part in enumerate(parts))
    points = tf.reshape(tf.linspace(tf.constant(-.7, D), .8, 4*dimension), [4, dimension])
    return basis, cores, points


def _differentiate(module, basis, cores, axes, query):
    bounds = tuple(value for part in basis.bases for value in (part.domain.left, part.domain.right))
    values = tuple(core.values for core in cores)
    sources = (query, *bounds, *values)
    with tf.GradientTape() as tape:
        tape.watch(sources)
        rows = module._basis_rows_with_pullback(basis, axes, query, max(core.basis_dim for core in cores))
        masses = module.basis_masses(basis, cores)
        gram = module.gram_chain(cores, masses)[0]
        objective = tf.reduce_sum(tf.square(rows)) + tf.reduce_sum(gram)
    return (rows, masses, gram), tape.gradient(objective, sources,
        unconnected_gradients=tf.UnconnectedGradients.ZERO)


@pytest.mark.parametrize("axes", [(), (1,), (2, 0), (0, 1, 2)])
@pytest.mark.parametrize("lebesgue", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_shared_basis_preserves_values_and_all_query_domain_core_derivatives(previous, axes, lebesgue, jit):
    basis, cores, points = _inputs(lebesgue=lebesgue)
    query = tf.gather(points, tf.constant(axes, tf.int32), axis=1)
    results = []
    for module in (previous, candidate):
        program = tf.function(partial(_differentiate, module, basis, cores, axes),
            input_signature=[tf.TensorSpec(query.shape, D)], jit_compile=jit, autograph=False)
        results.append(program(query))
        if jit:
            assert "HloModule" in program.experimental_get_compiler_ir(query)(stage="hlo")
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
        assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
        if not jit:
            assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                           for function in graph.library.function)
    for actual, expected in zip(tf.nest.flatten(results[1]), tf.nest.flatten(results[0]), strict=True):
        tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("family", ["common", "heterogeneous", "custom"])
@pytest.mark.parametrize("jit", [False, True])
def test_fixed_design_core_matrices_mass_offset_and_fallback_preserve_previous(previous, family, jit):
    basis, cores, points = _inputs(lebesgue=True, family=family)
    shapes = tuple(tuple(core.values.shape) for core in cores)
    bounds = tuple(value for part in basis.bases for value in (part.domain.left, part.domain.right))

    def evaluate(module, query):
        # Fixed fitting coordinates are prepared outside the score tape. The
        # original raw preparation helper has no XLA pullback across its Case
        # branches; the owned derivative path is core_matrices below.
        fixed_rows = module._fixed_basis_rows(basis, query, shapes)
        with tf.GradientTape() as tape:
            tape.watch((query, bounds))
            outputs = (module.core_matrices(basis, query, cores),
                module.basis_masses(basis, cores[1:], axis_offset=1))
            objective = tf.add_n([tf.reduce_sum(tf.square(value)) for value in tf.nest.flatten(outputs)])
        return (fixed_rows, outputs), tape.gradient(objective, (query, bounds),
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

    results = []
    for module in (previous, candidate):
        program = tf.function(partial(evaluate, module),
            input_signature=[tf.TensorSpec(points.shape, D)], jit_compile=jit, autograph=False)
        results.append(program(points))
    for actual, expected in zip(tf.nest.flatten(results[1]), tf.nest.flatten(results[0]), strict=True):
        tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)


def test_shared_basis_graph_grows_less_than_per_axis_polynomial_graph(previous, record_property):
    counts = {"before": [], "after": []}
    for dimension in (4, 8):
        basis, _cores, points = _inputs(dimension)
        for name, module in (("before", previous), ("after", candidate)):
            program = tf.function(partial(module._basis_rows_with_pullback,
                basis, tuple(range(dimension)), width=3),
                input_signature=[tf.TensorSpec(points.shape, D)], jit_compile=True, autograph=False)
            definition = program.get_concrete_function().graph.as_graph_def()
            counts[name].append(len(definition.node) + sum(len(f.node_def) for f in definition.library.function))
    print("basis graph nodes", counts)
    record_property("basis_graph_nodes", str(counts))
    assert all(after < before for after, before in zip(counts["after"], counts["before"], strict=True))
    assert counts["after"][1] - counts["after"][0] < counts["before"][1] - counts["before"][0]


def test_invalid_mass_measure_is_still_rejected(previous):
    basis, cores, _ = _inputs()
    for module in (previous, candidate):
        with pytest.raises(TypeError, match="measure must be a MassMeasure"):
            module.basis_masses(basis, cores, measure="invalid")


def test_incompatible_core_basis_width_still_fails(previous):
    basis, cores, _ = _inputs()
    invalid = tuple(highdim.TTCore(core.values[:, :2, :]) for core in cores)
    for module in (previous, candidate):
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            module.basis_masses(basis, invalid)


@pytest.mark.parametrize("rank_one", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_basis_sharing_preserves_complete_fit_histories_and_pullbacks(previous, monkeypatch, rank_one, jit):
    from bayesfilter.highdim import fixed_tt_native_fit_tf as fitter
    from tests.highdim.test_fixed_branch_fit import _config

    basis, cores, points = _inputs()
    if rank_one:
        cores = tuple(highdim.TTCore(core.values[:1, :, :1]) for core in cores)
    ranks = (1, *(core.right_rank for core in cores))
    config = _config(ranks, sweep_order=(0, 1, 2, 2, 1, 0), max_sweeps=2, ridge=1e-4)
    weights = tf.constant([.7, 1.1, .9, 1.3], D)
    target = 1. + .2*points[:, 0] + .1*points[:, 2]**2

    def score(fit, values, initial):
        with tf.GradientTape() as tape:
            tape.watch((values, initial))
            result = fit(values, initial)
            objective = tf.reduce_sum(tf.square(result["cores"]))
        return result, tape.gradient(objective, (values, initial))

    results = []
    for basis_rows in (previous.fixed_basis_rows, candidate.fixed_basis_rows):
        with monkeypatch.context() as patch:
            patch.setattr(fitter, "fixed_basis_rows", basis_rows)
            fit = fitter.NativeFixedTTFit(basis, points, weights, config, cores, jit_compile=jit)
            program = tf.function(partial(score, fit), input_signature=[
                tf.TensorSpec(target.shape, D), tf.TensorSpec(fit.initial.shape, D)],
                jit_compile=jit, autograph=False)
            result = program(target, fit.initial)
        assert bool(result[0]["valid"])
        results.append(result)
    for actual, expected in zip(tf.nest.flatten(results[1]), tf.nest.flatten(results[0]), strict=True):
        assert actual is not None and expected is not None
        if actual.dtype.is_floating:
            tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)
        else:
            tf.debugging.assert_equal(actual, expected)
