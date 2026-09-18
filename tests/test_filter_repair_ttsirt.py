"""Independent original-source comparisons for the grid-CDF execution repair."""

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.filtering import AffineCoordinateMap, IdentityCoordinateMap
from bayesfilter.highdim.ttsirt_native_tf import transport_arguments, transport_program
from bayesfilter.highdim.ttsirt_proposal_native_tf import proposal_program
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    AlgebraicCoordinateMap,
    compile_fixed_ttsirt_proposal_branch,
)
from tests.highdim.test_zhao_cui_frozen_ttsirt_apf_compiler import (
    _constant_transport,
    _correlated_transport,
)

DTYPE = tf.float64


@pytest.fixture(scope="module")
def original():
    root = Path(__file__).resolve().parents[1]
    modules = []
    for filename in ("transport", "zhao_cui_frozen_proposal_apf_tf"):
        source = subprocess.check_output([
            "git", "show", f"3582b4ac:bayesfilter/highdim/{filename}.py"], cwd=root, text=True)
        spec = importlib.util.spec_from_loader(f"original_{filename}_reference", loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        # Execute only the pinned repository source as an independent reference.
        exec(compile(source, f"frozen_{filename}_reference.py", "exec"), module.__dict__)  # noqa: S102
        modules.append(module)
    return modules


@pytest.mark.parametrize("suffix", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_public_transport_preserves_original_maps_and_density(original, suffix, jit):
    transport = _correlated_transport()
    baseline = original[0].FixedTTSIRTTransport(transport.density,
        original[0].KRCDFConfig(**transport.cdf_config.__dict__))
    uniforms = tf.constant([[0.18, 0.54, 0.81], [0.25, 0.42, 0.76]], DTYPE)
    points = baseline.inverse_transport(uniforms)
    condition, query = points[1:], uniforms[:1]
    if suffix:
        value = transport.conditional_inverse_transport_suffix(condition, query, jit_compile=jit)
        expected = baseline.conditional_inverse_transport_suffix(condition, query)
        tf.debugging.assert_near(value, expected, atol=2e-12, rtol=2e-12)
        for name in ("conditional_forward_transport_suffix", "conditional_forward_log_jacobian_suffix"):
            tf.debugging.assert_near(getattr(transport, name)(condition, expected, jit_compile=jit),
                getattr(baseline, name)(condition, expected), atol=2e-12, rtol=2e-12)
        tf.debugging.assert_near(transport.conditional_proposal_log_density_suffix(
            conditioning_points=condition, generated_points=expected, jit_compile=jit),
            baseline.conditional_proposal_log_density_suffix(conditioning_points=condition, generated_points=expected),
            atol=2e-12, rtol=2e-12)
    else:
        tf.debugging.assert_near(transport.inverse_transport(uniforms, jit_compile=jit), points,
            atol=2e-12, rtol=2e-12)
        for name in ("forward_transport", "forward_log_jacobian", "eval_pdf"):
            tf.debugging.assert_near(getattr(transport, name)(points, jit_compile=jit),
                getattr(baseline, name)(points), atol=2e-12, rtol=2e-12)


def test_transport_vetoes_and_bisection_graph_are_preserved():
    transport = _correlated_transport()
    values = tf.constant([[0.2, 0.7], [0.3, 0.8]], DTYPE)
    counts = []
    for steps in (12, 24):
        candidate = replace(transport, cdf_config=replace(transport.cdf_config, bisection_steps=steps))
        program = transport_program(candidate, "inverse", 2)
        arguments = (*transport_arguments(candidate), tf.zeros([0, 2], DTYPE), values)
        _, code = program(*arguments)
        assert int(code) == 0
        assert "HloModule" in program.experimental_get_compiler_ir(*arguments)(stage="hlo")
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        assert not any(node.op in ("PyFunc", "EagerPyFunc") for node in nodes)
        counts.append(len(nodes))
    assert counts[0] == counts[1]
    with pytest.raises(ValueError, match="INVERSE_BRACKET_FAILURE"):
        transport.inverse_transport(values+1.0)
    with pytest.raises(ValueError, match="max_batch_working_bytes"):
        replace(transport, cdf_config=replace(transport.cdf_config, max_batch_working_bytes=1)).inverse_transport(values)


@pytest.mark.parametrize("suffix", [False, True])
def test_public_transport_complete_enclosing_graph_and_xla_with_invalid_status(suffix):
    from dataclasses import replace

    transport = _correlated_transport()
    transport = replace(transport, cdf_config=replace(transport.cdf_config, grid_size=9, bisection_steps=8))
    points = tf.constant([[.2, .5, .8], [.3, .7, .4]], DTYPE)

    def evaluate(query):
        if suffix:
            condition = tf.constant([[.1, -.3, .2]], DTYPE)
            local = transport.conditional_inverse_transport_suffix(condition, query[:1])
            return (local, transport.conditional_forward_transport_suffix(condition, local),
                transport.conditional_forward_log_jacobian_suffix(condition, local),
                transport.conditional_proposal_log_density_suffix(conditioning_points=condition, generated_points=local))
        local = transport.inverse_transport(query)
        return (local, transport.forward_transport(local), transport.forward_log_jacobian(local),
                transport.eval_pdf(local), transport.proposal_log_density(local_points=local, reference_points=query))

    expected = evaluate(points)
    signature = [tf.TensorSpec(points.shape, DTYPE)]
    graph = tf.function(evaluate, input_signature=signature, jit_compile=False, autograph=False)
    xla = tf.function(evaluate, input_signature=signature, jit_compile=True, autograph=False)
    for result in (graph(points), xla(points)):
        for actual, reference in zip(result, expected, strict=True):
            tf.debugging.assert_near(actual, reference, atol=2e-12, rtol=2e-12)
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in definition.library.function)
    assert not any(node.op in ("PyFunc", "EagerPyFunc")
                   for node in [*definition.node, *(node for fn in definition.library.function for node in fn.node_def)])
    assert xla.experimental_get_tracing_count() == 1
    assert "HloModule" in xla.experimental_get_compiler_ir(points)(stage="hlo")
    for invalid in (points + 1., tf.fill(points.shape, tf.constant(float("nan"), DTYPE))):
        assert not bool(tf.reduce_all(tf.math.is_finite(xla(invalid)[0])))


def _compiler_inputs(steps, map_kind):
    maps = {
        "identity": lambda: IdentityCoordinateMap(1),
        "algebraic": lambda: AlgebraicCoordinateMap(tf.constant([1.5], DTYPE)),
        "affine": lambda: AffineCoordinateMap(tf.constant([0.3], DTYPE), tf.constant([[1.2]], DTYPE)),
    }
    return {"observations": tf.zeros([steps+1, 1], DTYPE), "initial_transport": _constant_transport(1),
        "transition_transports": tuple(_correlated_transport() for _ in range(steps)),
        "coordinate_map": maps[map_kind](),
        "initial_reference_points": tf.constant([[0.13, 0.46, 0.79]], DTYPE),
        "ancestor_uniforms": tf.tile(tf.constant([[0.02, 0.41, 0.99]], DTYPE), [steps, 1]),
        "auxiliary_log_probabilities": tf.tile(tf.math.log(tf.constant([[0.2, 0.3, 0.5]], DTYPE)), [steps, 1]),
        "transition_reference_points": tf.tile(tf.constant([[[0.24, 0.39, 0.72]]], DTYPE), [steps, 1, 1])}


def test_empty_generated_suffix_and_affine_xla():
    transport = _correlated_transport()
    condition, empty = tf.zeros([2, 3], DTYPE), tf.zeros([0, 3], DTYPE)
    for name in ("conditional_inverse_transport_suffix", "conditional_forward_transport_suffix"):
        assert getattr(transport, name)(condition, empty).shape == (0, 3)
    tf.debugging.assert_equal(transport.conditional_forward_log_jacobian_suffix(condition, empty),
        tf.zeros([3], DTYPE))
    matrix = tf.constant([[0.1, 1.4], [1.8, -0.2]], DTYPE)
    coordinate = AffineCoordinateMap(tf.constant([0.3, -0.7], DTYPE), matrix)
    points = tf.constant([[0.2, 0.6], [-0.4, 0.5]], DTYPE)

    @tf.function(jit_compile=True, autograph=False)
    def roundtrip(points):
        physical, forward = coordinate.forward(points)
        reference, inverse = coordinate.inverse(physical)
        return reference, forward, inverse

    reference, forward, inverse = roundtrip(points)
    tf.debugging.assert_near(reference, points, atol=1e-13, rtol=1e-13)
    expected = tf.fill([2], tf.math.log(tf.abs(tf.linalg.det(matrix))))
    tf.debugging.assert_near(forward, expected, atol=1e-13, rtol=1e-13)
    tf.debugging.assert_near(inverse, -expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("steps,map_kind", [(0, "identity"), (2, "identity"), (2, "algebraic"), (2, "affine")])
def test_complete_compiler_matches_original_time_recurrence(original, steps, map_kind):
    inputs = _compiler_inputs(steps, map_kind)
    baseline = original[1].compile_fixed_ttsirt_proposal_branch(**inputs).branch
    for jit in (False, True):
        result = compile_fixed_ttsirt_proposal_branch(**inputs, jit_compile=jit)
        assert result.manifest["jit_compile"] == jit
        for field in ("states", "initial_log_proposal_density", "transition_log_proposal_density"):
            tf.debugging.assert_near(getattr(result.branch, field), getattr(baseline, field), atol=2e-12, rtol=2e-12)
        tf.debugging.assert_equal(result.branch.ancestors, baseline.ancestors)


def test_complete_compiler_hlo_graph_bound_and_changed_coefficient_inputs():
    counts = []
    for steps in (1, 4):
        inputs = _compiler_inputs(steps, "identity")
        program, packed = proposal_program(inputs["initial_transport"], inputs["transition_transports"],
            inputs["coordinate_map"], 3)
        arguments = (*packed, inputs["initial_reference_points"], inputs["ancestor_uniforms"],
            inputs["auxiliary_log_probabilities"], inputs["transition_reference_points"])
        result = program(*arguments)
        assert int(result[-1]) == 0
        assert "HloModule" in program.experimental_get_compiler_ir(*arguments)(stage="hlo")
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        assert not any(node.op in ("PyFunc", "EagerPyFunc") for node in nodes)
        counts.append(len(nodes))
        changed = list(arguments)
        changed[2] = tf.tensor_scatter_nd_add(changed[2], [[0, 0]], tf.constant([0.1], DTYPE))
        assert float(tf.reduce_max(tf.abs(program(*changed)[0] - result[0]))) > 1e-5
        assert program.experimental_get_tracing_count() == 1
    assert counts[0] == counts[1]


def test_complete_compiler_dispatches_heterogeneous_core_schemas(original):
    inputs = _compiler_inputs(3, "identity")
    inputs["transition_transports"] = (
        _constant_transport(2), _correlated_transport(), _constant_transport(2))
    expected = original[1].compile_fixed_ttsirt_proposal_branch(**inputs).branch
    for jit in (False, True):
        result = compile_fixed_ttsirt_proposal_branch(**inputs, jit_compile=jit).branch
        for field in ("states", "initial_log_proposal_density", "transition_log_proposal_density"):
            tf.debugging.assert_near(getattr(result, field), getattr(expected, field),
                                     atol=2e-12, rtol=2e-12)
        tf.debugging.assert_equal(result.ancestors, expected.ancestors)
