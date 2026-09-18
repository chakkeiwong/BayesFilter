"""Pinned checks of complete centered initializer feature operators."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import centered_initializer_native_tf as native
from bayesfilter.highdim import (
    zhao_cui_austria_sir_parameter_density_training_tf as candidate,
)
from tests.test_filter_repair_centered_tt import _fresh_parent
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("prefix", [1, 18])
def test_additive_and_pair_feature_operators_preserve_pinned_algebra(prefix, monkeypatch):
    parent = _fresh_parent()
    # Nonconstant parent cores make every forward/backward message observable.
    cores = []
    for axis in range(36):
        left, right = (1 if axis == 0 else 2), (1 if axis == 35 else 2)
        shape = (left, 3, right)
        constant = tf.one_hot(0, left, dtype=D)[:, None, None] * tf.ones(shape, D) * tf.one_hot(0, right, dtype=D)[None, None]
        cores.append(constant + .02 * tf.reshape(tf.sin(tf.cast(tf.range(left * 3 * right), D) + axis), shape))
    cores = tuple(cores)
    # Use a simple immutable view; this is a numerical operator test, not an
    # artifact admission check. The complete child test issues a real identity.
    parent = SimpleNamespace(cores=cores, settings=parent.settings)
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    contractions = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values"):
        monkeypatch.setattr(before, name, getattr(contractions, name))
    basis = candidate.centered_lane_b_product_basis(order=2, num_elems=1)
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 2 * 36), [2, 36])
    actual = (native.feature_values(basis, points), native.feature_integrals(cores, basis),
        native.feature_integrals(cores, basis, points[:, :prefix]),
        native.feature_integrals(cores, basis, include_pairs=True),
        native.feature_integrals(cores, basis, points[:, :prefix], include_pairs=True),
        candidate.additive_prefix_score_operator(parent=parent, local_prefix_points=points[:, :prefix]),
        candidate.within_region_pair_prefix_score_operator(parent=parent, local_prefix_points=points[:, :prefix]))
    expected = (before._within_region_feature_values(basis, points), before._parent_additive_cross_features(parent, basis),
        # The original prefix operator includes normalization; compare the full
        # operator below, and use original feature TTs as an independent integral.
        before._within_region_cross_features(parent=parent, basis=basis, local_prefix_points=points[:, :prefix])[:, :108],
        before._within_region_cross_features(parent=parent, basis=basis, local_prefix_points=None),
        before._within_region_cross_features(parent=parent, basis=basis, local_prefix_points=points[:, :prefix]),
        before.additive_prefix_score_operator(parent=parent, local_prefix_points=points[:, :prefix]),
        before.within_region_pair_prefix_score_operator(parent=parent, local_prefix_points=points[:, :prefix]))
    for value, authority in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    prefix_points = points[:, :prefix]
    with tf.GradientTape() as tape:
        tape.watch((cores, prefix_points))
        features = native.feature_integrals(cores, basis, prefix_points, include_pairs=True)
        loss = tf.reduce_sum(features)
    core_gradient, query_gradient = tape.gradient(loss, (cores, prefix_points))
    assert all(gradient is not None for gradient in core_gradient)
    direction = .1 * tf.cos(prefix_points)
    delta = 1e-5
    plus = tf.reduce_sum(native.feature_integrals(cores, basis, prefix_points + delta * direction, include_pairs=True))
    minus = tf.reduce_sum(native.feature_integrals(cores, basis, prefix_points - delta * direction, include_pairs=True))
    np.testing.assert_allclose(tf.reduce_sum(query_gradient * direction), (plus-minus)/(2*delta), atol=1e-8, rtol=1e-7)
    core_direction = .05 * tf.cos(cores[3])
    plus_cores = (*cores[:3], cores[3] + delta * core_direction, *cores[4:])
    minus_cores = (*cores[:3], cores[3] - delta * core_direction, *cores[4:])
    plus = tf.reduce_sum(native.feature_integrals(plus_cores, basis, prefix_points, include_pairs=True))
    minus = tf.reduce_sum(native.feature_integrals(minus_cores, basis, prefix_points, include_pairs=True))
    np.testing.assert_allclose(tf.reduce_sum(core_gradient[3] * core_direction), (plus-minus)/(2*delta), atol=1e-8, rtol=1e-7)
    program = next(row[1].compiled for key, row in native._PROGRAMS.items()
        if key[0] == id(basis) and key[1] == "pair_cross" and key[2][1].shape[1] == prefix)
    packed = candidate.centered_native.pack_components((cores,))[0]
    assert "HloModule" in program.experimental_get_compiler_ir(packed, prefix_points)(stage="hlo")
    assert any(node.op in ("While", "StatelessWhile") for node in _graph(program))


@pytest.mark.parametrize("pair_features", [False, True])
def test_complete_score_initializer_and_coefficient_pullback(pair_features, monkeypatch):
    from dataclasses import fields

    parent = _fresh_parent()
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    contractions = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values"):
        monkeypatch.setattr(before, name, getattr(contractions, name))
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 8 * 36), [8, 36])
    target = tf.reshape(tf.sin(tf.cast(tf.range(24), D)), [8, 3])
    kwargs = {"parent": parent, "local_points": points, "target_complete_data_score": target,
        "importance_log_weight": tf.linspace(tf.constant(-.2, D), .3, 8),
        "ridge_fraction": .1, "global_score_weight": 1., "prefix_local_points": points[:2, :18],
        "prefix_target_score": target[:2], "prefix_score_standard_error": tf.ones([2, 3], D), "prefix_weight": .2}
    name = "target_informed_within_region_pair_score_initialization" if pair_features else "target_informed_additive_score_initialization"
    result = getattr(candidate, name)(**kwargs)
    expected = getattr(before, name)(**kwargs)
    for field in fields(result):
        for value, authority in zip(tf.nest.flatten(getattr(result, field.name)), tf.nest.flatten(getattr(expected, field.name)), strict=True):
            np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    call = getattr(candidate, name)
    with tf.GradientTape() as tape:
        tape.watch(target)
        result = call(**{**kwargs, "target_complete_data_score": target})
        objective = tf.reduce_sum(result.coefficient_rms)
    gradient = tape.gradient(objective, target)
    direction, delta = .1 * tf.cos(target), 1e-5
    plus = call(**{**kwargs, "target_complete_data_score": target + delta * direction})
    minus = call(**{**kwargs, "target_complete_data_score": target - delta * direction})
    finite_difference = (tf.reduce_sum(plus.coefficient_rms) - tf.reduce_sum(minus.coefficient_rms)) / (2 * delta)
    np.testing.assert_allclose(tf.reduce_sum(gradient * direction), finite_difference, atol=1e-8, rtol=1e-7)
    matching = [(key, row) for key, row in native._INITIALIZERS.items() if key[0].__name__ == name and key[1] == id(parent)]
    assert len(matching) == 1
    _key, (_, evaluate, _) = matching[0]
    program = evaluate.compiled
    assert program.experimental_get_tracing_count() == 1
    assert "HloModule" in program.experimental_get_compiler_ir(
        points, target, kwargs["importance_log_weight"], points[:2, :18], target[:2], tf.ones([2, 3], D))(stage="hlo")
