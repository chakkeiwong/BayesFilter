"""Pinned seeded-core and gradient checks for compiled TT initialization."""

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import centered_training_native_tf as native
from bayesfilter.highdim import centered_tt_native_tf as contractions
from bayesfilter.highdim import zhao_cui_austria_sir_lane_b_tf as lane
from bayesfilter.highdim import (
    zhao_cui_austria_sir_parameter_density_training_tf as candidate,
)
from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (
    CenteredThetaFeatures,
)
from tests.test_filter_repair_centered_tt import _fresh_parent
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("rank", [1, 3])
def test_balanced_initial_cores_preserve_mixed_width_schema(rank):
    before = _original("zhao_cui_austria_sir_lane_b_tf")
    settings = replace(_fresh_parent().settings, rank=rank)
    basis = ProductBasis(tuple(LegendreBasis1D(BoundedInterval(-1., 1.), axis % 3 + 1)
                               for axis in range(36)), lane.lane_b_measure_convention())
    expected = before.balanced_initial_cores(settings, basis)
    actual = lane.balanced_initial_cores(settings, basis)
    for result, authority in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, authority)
    program = native.balanced_core_program(settings.ranks(), basis.basis_dim_tuple())
    scale = tf.constant(1e-6 / max(rank - 1, 1), D)
    assert "HloModule" in program.experimental_get_compiler_ir(scale)(stage="hlo")
    _graph(program)
    assert program.experimental_get_tracing_count() == 1


def test_balanced_initialization_rejects_invalid_seeded_basis_index():
    # The previous scatter rejects basis index 1 when the first width is 1.
    # Validate before XLA rather than allowing an out-of-bounds update to vanish.
    with pytest.raises(ValueError, match="first-axis basis"):
        native.balanced_core_program((1, 2, 1), (1, 3))


def test_complete_seeded_initialization_preserves_enclosing_graph_boundary():
    parent, features = _fresh_parent(), CenteredThetaFeatures()

    def evaluate():
        residual = candidate.fixed_rank_initial_residual_components(parent=parent,
            features=features, rank=2, seed=1729)
        return residual, candidate.embed_residual_component_with_connected_channels(residual[0],
            target_rank=4, seed=1729, seeded_channel_epsilon=.03)

    graph = tf.function(evaluate, input_signature=[], jit_compile=False, autograph=False)
    compiled = tf.function(evaluate, input_signature=[], jit_compile=True, autograph=False)
    for actual, reference in zip(tf.nest.flatten(compiled()), tf.nest.flatten(graph()), strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-14, rtol=1e-14)
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in definition.library.function)
    _graph(graph)
    assert "HloModule" in compiled.experimental_get_compiler_ir()(stage="hlo")


@pytest.mark.parametrize("rank,seed", [(1, 1729), (3, -17)])
def test_complete_residual_initializer_preserves_original_seeded_draws(rank, seed, monkeypatch):
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    old_lane = _original("zhao_cui_austria_sir_lane_b_tf")
    monkeypatch.setattr(before, "balanced_initial_cores", old_lane.balanced_initial_cores)
    parent = _fresh_parent()
    features = CenteredThetaFeatures()
    options = {"parent": parent, "features": features, "rank": rank, "seed": seed,
               "amplitude_scale": .002, "perturbation_scale": .003}
    expected = before.fixed_rank_initial_residual_components(**options)
    actual = candidate.fixed_rank_initial_residual_components(**options)
    repeated = candidate.fixed_rank_initial_residual_components(**options)
    for result, authority, repeat in zip(tf.nest.flatten(actual), tf.nest.flatten(expected),
                                         tf.nest.flatten(repeated), strict=True):
        np.testing.assert_allclose(result, authority, atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(result, repeat)
    basis = candidate.centered_lane_b_product_basis(order=parent.settings.basis_order,
                                                   num_elems=parent.settings.basis_num_elems)
    balanced = lane.balanced_initial_cores(replace(parent.settings, rank=rank), basis)
    shapes = tuple(tuple(core.shape) for core in balanced)
    program = native.residual_noise_program(shapes, features.feature_count)
    inputs = (contractions.pack_components((balanced,))[0], tf.constant(seed, tf.int32),
              tf.constant(.002, D), tf.constant(.003, D))
    assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
    assert any(node.op in ("While", "StatelessWhile") for node in _graph(program))
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("target_rank,seed", [(2, 1729), (4, 1729), (4, -17)])
def test_connected_channel_initialization_preserves_mixed_widths_draws_and_gradients(target_rank, seed):
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    widths, ranks = (3, 4, 7, 5), (1, 2, 2, 2, 1)
    cores = tuple(tf.reshape(.1 * tf.sin(tf.cast(tf.range(ranks[axis] * width * ranks[axis+1]), D)),
                             [ranks[axis], width, ranks[axis+1]]) for axis, width in enumerate(widths))
    options = {"target_rank": target_rank, "seed": seed, "seeded_channel_epsilon": .03}
    outputs, gradients = [], []
    for module in (candidate, before):
        with tf.GradientTape() as tape:
            tape.watch(cores)
            output = module.embed_residual_component_with_connected_channels(cores, **options)
            objective = sum(tf.reduce_sum(tf.square(value)) for value in output)
        outputs.append(output)
        gradients.append(tape.gradient(objective, cores))
    for result, authority in zip(tf.nest.flatten(outputs[0]), tf.nest.flatten(outputs[1]), strict=True):
        np.testing.assert_allclose(result, authority, atol=1e-14, rtol=1e-14)
    for result, authority, original in zip(*gradients, cores, strict=True):
        np.testing.assert_array_equal(result, authority)
        np.testing.assert_array_equal(result, 2. * original)
    if target_rank > 2:
        packed = contractions.pack_components((candidate.embed_residual_component_at_rank(
            cores, target_rank=target_rank),))[0]
        program = native.connected_channels_program(widths, target_rank, 2)
        assert "HloModule" in program.experimental_get_compiler_ir(
            packed, tf.constant(seed, tf.int32), tf.constant(.03, D))(stage="hlo")
        assert any(node.op in ("While", "StatelessWhile") for node in _graph(program))
        assert program.experimental_get_tracing_count() == 1
