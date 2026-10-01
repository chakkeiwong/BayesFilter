"""Pinned and independent checks of the existing core-affine extension."""

from math import prod

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import core_tangent_native_tf as native
from bayesfilter.highdim import (
    zhao_cui_austria_sir_parameter_density_training_tf as candidate,
)
from tests.test_filter_repair_centered_tt import _fresh_parent
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _cores(dimension):
    ranks = (1, *(axis % 3 + 1 for axis in range(1, dimension)), 1)
    shapes = tuple((ranks[axis], axis % 4 + 2, ranks[axis + 1]) for axis in range(dimension))
    parents = tuple(tf.reshape(.2 * tf.sin(tf.cast(tf.range(prod(shape)), D) + axis), shape)
                    for axis, shape in enumerate(shapes))
    tangents = tuple(tuple(.03 * tf.cos(core + index) for core in parents) for index in range(3))
    return parents, tangents


@pytest.mark.parametrize("dimension", [2, 5])
def test_heterogeneous_product_rule_blocks_preserve_values_masks_and_pullbacks(dimension):
    parents, tangents = _cores(dimension)
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    results, gradients = [], []
    for repaired in (True, False):
        with tf.GradientTape() as tape:
            tape.watch((parents, tangents))
            blocks = (native.encode_components(parents, tangents) if repaired else
                      tuple(before.core_tangent_to_residual_component(parent_cores=parents,
                            tangent_cores=row) for row in tangents))
            objective = sum(tf.reduce_sum(tf.square(value)) for value in tf.nest.flatten(blocks))
        results.append(blocks)
        gradients.append(tape.gradient(objective, (parents, tangents)))
    for actual, expected in zip(tf.nest.flatten(results[0]), tf.nest.flatten(results[1]), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(tf.nest.flatten(gradients[0]), tf.nest.flatten(gradients[1]), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    restored = candidate.core_tangent_banks_from_residual_components(
        parent_cores=parents, residual_components=results[0])
    for actual, expected in zip(tf.nest.flatten(restored), tf.nest.flatten(tuple(zip(*tangents, strict=True))), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(candidate.core_affine_released_coordinate_mask(parents),
                                   before.core_affine_released_coordinate_mask(parents))
    program = native.block_program(tuple(tuple(value.shape) for value in parents), 3)
    flat_parent = tf.concat([tf.reshape(core, [-1]) for core in parents], 0)
    flat_tangent = tf.stack([tf.concat([tf.reshape(core, [-1]) for core in row], 0) for row in tangents])
    encoded = program.encode(flat_parent, flat_tangent)
    for call, values in ((program.encode, (flat_parent, flat_tangent)),
                         (program.decode, (flat_parent, encoded)), (program.mask, ())):
        assert "HloModule" in call.experimental_get_compiler_ir(*values)(stage="hlo")
        assert call.experimental_get_tracing_count() == 1
        _graph(call)


@pytest.mark.parametrize("axis,coordinate", [(0, (0, 0, 0)), (1, (2, 0, 0)), (2, (3, 0, 0))])
def test_inverse_encoding_rejects_corrupted_parent_and_zero_blocks(axis, coordinate):
    parents, tangents = _cores(3)
    blocks = native.encode_components(parents, tangents)
    changed = list(blocks[0])
    changed[axis] = tf.tensor_scatter_nd_add(changed[axis], [coordinate], tf.constant([.1], D))
    corrupted = (tuple(changed), *blocks[1:])
    with pytest.raises(tf.errors.InvalidArgumentError, match="block mismatch"):
        candidate.core_tangent_banks_from_residual_components(
            parent_cores=parents, residual_components=corrupted)


def test_block_graph_size_does_not_unroll_core_axes():
    nodes = []
    for count in (2, 10):
        parents, _tangents = _cores(count)
        program = native.block_program(tuple(tuple(value.shape) for value in parents), 3)
        nodes.append(len(_graph(program.encode)))
    assert abs(nodes[1] - nodes[0]) <= 10


def test_explicit_graph_reference_does_not_hide_nested_xla():
    parents, tangents = _cores(2)
    specifications = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, D), tangents)
    reference = tf.function(lambda values: native.encode_components(parents, values),
        input_signature=[specifications], jit_compile=False, autograph=False)
    reference(tangents)
    graph = reference.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in graph.library.function)


def test_complete_core_affine_loss_and_gradient_preserve_pinned_consumer(monkeypatch):
    parent = _fresh_parent()
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    contractions = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values", "CenteredThetaFeatures"):
        monkeypatch.setattr(before, name, getattr(contractions, name))
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 4 * 36), [4, 36])
    target = tf.reshape(tf.linspace(tf.constant(-.2, D), .3, 12), [4, 3])
    options = {"parent": parent, "point_local_points": points, "point_target_score": target,
        "point_importance_log_weight": tf.constant([.2, -.1, .3, -.2], D),
        "global_target_score": target[0], "global_score_standard_error": tf.ones([3], D),
        "prefix_local_points": points[:, :18], "prefix_target_score": target,
        "prefix_score_standard_error": tf.ones([4, 3], D), "point_weight": 1.,
        "global_weight": .2, "prefix_weight": .1, "l2_weight": .001}
    size = 3 * sum(prod(core.shape) for core in parent.cores)
    position = .01 * tf.sin(tf.cast(tf.range(size), D))
    callback = candidate.make_compiled_core_affine_total_score_value_and_gradient(**options)
    actual, gradient = callback(position)
    with tf.GradientTape() as tape:
        tape.watch(position)
        expected = before.core_affine_origin_total_score_loss_arrays(position=position, **options)[0]
    expected_gradient = tape.gradient(expected, position)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10, rtol=1e-10)
    direction, step = .05 * tf.cos(position), 1e-5
    finite_difference = (callback(position + step * direction)[0] - callback(position - step * direction)[0]) / (2 * step)
    np.testing.assert_allclose(tf.reduce_sum(gradient * direction), finite_difference, atol=1e-8, rtol=1e-7)
    assert callback.experimental_get_tracing_count() == 1
    assert "HloModule" in callback.experimental_get_compiler_ir(position)(stage="hlo")
    graph = callback.get_concrete_function(position).graph.as_graph_def()
    nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
    assert not ({node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"})
