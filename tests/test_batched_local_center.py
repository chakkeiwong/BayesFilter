"""NumPy/GradientTape verification oracles for the opt-in TF location primitive."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    BATCHED_LOCAL_CENTER_NONCLAIMS,
    BatchedLocalCenterConfig,
    BatchedLocalCenterResult,
    locate_batched_local_center,
)
from bayesfilter.inference import batched_local_center as local_center


def _config(**overrides):
    return BatchedLocalCenterConfig(**{"jit_compile": False, **overrides})


def _quadratic(mode=(0.2, -0.15), precision=((2.0, 0.4), (0.4, 1.3))):
    mode = tf.constant(mode, tf.float64)
    precision = tf.constant(precision, tf.float64)

    def target(positions):
        displacement = positions - mode
        score = -tf.linalg.matmul(displacement, precision, transpose_b=True)
        return (
            0.5 * tf.reduce_sum(displacement * score, axis=1),
            score,
            tf.ones([positions.shape[0]], tf.bool),
        )

    return target


def _flat(positions):
    return (
        tf.zeros([positions.shape[0]], tf.float64),
        tf.zeros_like(positions),
        tf.ones(
            [positions.shape[0]],
            tf.bool,
        ),
    )


def _fake_result(function, endpoint, failed=False):
    value, gradient = function(endpoint)
    return SimpleNamespace(
        position=endpoint,
        objective_value=value,
        objective_gradient=gradient,
        converged=tf.ones([endpoint.shape[0]], tf.bool),
        failed=tf.fill([endpoint.shape[0]], failed),
    )


def _accounting(result, batch_size, cfg):
    batches = int(result.target_callback_batches)
    assert int(result.physical_target_rows) == batch_size * batches
    assert batches == 1 + int(result.optimizer_target_batches) + int(
        result.replay_batches
    )
    assert batches <= cfg.maximum_physical_rows_multiplier
    assert int(result.replay_batches) == int(result.rounds_completed) + int(
        tf.reduce_any(result.valid_rows)
    )
    assert int(result.optimizer_target_batches) <= int(result.rounds_completed) * (
        cfg.max_optimizer_callback_batches_per_round
    )
    assert result.trace_count == 1
    json.dumps(result.payload(), allow_nan=False)


def test_public_exports_and_defaults():
    assert local_center.BatchedLocalCenterResult is BatchedLocalCenterResult
    assert locate_batched_local_center.__module__ == local_center.__name__
    assert BATCHED_LOCAL_CENTER_NONCLAIMS
    assert BatchedLocalCenterConfig().maximum_physical_rows_multiplier == 132
    assert BatchedLocalCenterConfig().jit_compile


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("box_radius", 0.0, ValueError),
        ("gradient_tolerance", float("nan"), ValueError),
        ("replay_atol", -1.0, ValueError),
        ("trust_refinement_rounds", 0, ValueError),
        ("max_iterations", 1.5, TypeError),
        ("max_optimizer_callback_batches_per_round", 0, ValueError),
    ],
)
def test_bad_config(field, value, error):
    with pytest.raises(error):
        _config(**{field: value})


@pytest.mark.parametrize(
    "initial,scale",
    [
        ([[[0.0, 0.0]]], [1.0, 1.0]),
        ([[]], []),
        ([[0.0, 0.0]], [1.0]),
        ([[0.0, 0.0]], [1.0, 0.0]),
        ([[0.0, 0.0]], [1.0, float("nan")]),
        ([[float("inf"), 0.0]], [1.0, 1.0]),
    ],
)
def test_input_validation(initial, scale):
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        locate_batched_local_center(_quadratic(), initial, scale, config=_config())


def test_float32_tensor_is_not_silently_promoted():
    with pytest.raises(ValueError):
        locate_batched_local_center(_quadratic(), tf.zeros([2, 2]), [1.0, 1.0])


@pytest.mark.parametrize("scale", [[1.0, 1.0], [0.01, 13.0]])
def test_actual_bounded_chart_chain_rule(scale):
    scale = tf.constant(scale, tf.float64)
    anchor = tf.constant([[0.3, -0.2], [-1.0, 0.4]], tf.float64)
    coordinate = tf.constant([[0.4, -0.1], [100.0, -100.0]], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(coordinate)
        positions, derivative = local_center._bounded_chart(
            anchor, scale, coordinate, 4.0
        )
        value, score, _ = _quadratic()(positions)
        loss = -tf.reduce_sum(value)
    np.testing.assert_allclose(
        -score * derivative, tape.gradient(loss, coordinate), atol=1e-12
    )
    assert bool(tf.reduce_all(tf.abs((positions - anchor) / scale) <= 4.0 + 1e-12))


@pytest.mark.parametrize("batch_size", [1, 3])
def test_rotated_gaussian_recovery_fixed_batch_and_exact_accounting(batch_size):
    calls = tf.Variable(0, dtype=tf.int32)
    target = _quadratic()

    def counted(positions):
        assert positions.shape == (batch_size, 2)
        calls.assign_add(1)
        return target(positions)

    starts = [[-0.5, 0.8], [0.8, -0.7], [1.0, 1.2]][:batch_size]
    cfg = _config()
    result = locate_batched_local_center(counted, starts, [0.5, 2.0], config=cfg)
    assert bool(result.accepted), result.payload()
    np.testing.assert_allclose(result.center, [0.2, -0.15], atol=1e-7)
    assert int(calls) == int(result.target_callback_batches)
    assert result.status == "stationary"
    _accounting(result, batch_size, cfg)


def test_affine_diagonal_reparameterization_and_row_permutation():
    target = _quadratic()
    starts = tf.constant([[-0.6, 0.5], [0.8, -0.7], [1.0, 1.2]], tf.float64)
    scale = tf.constant([0.5, 2.0], tf.float64)
    shift = tf.constant([100.0, -15.0], tf.float64)
    multiplier = tf.constant([0.03, 7.0], tf.float64)

    def transformed(positions):
        value, score, valid = target((positions - shift) / multiplier)
        return value, score / multiplier, valid

    first = locate_batched_local_center(target, starts, scale, config=_config())
    second = locate_batched_local_center(
        transformed,
        shift + tf.reverse(starts, [0]) * multiplier,
        scale * multiplier,
        config=_config(),
    )
    assert bool(first.accepted) and bool(second.accepted)
    np.testing.assert_allclose(
        (second.center - shift) / multiplier, first.center, atol=1e-7
    )
    np.testing.assert_allclose(
        second.center_score * multiplier, first.center_score, atol=1e-7
    )
    np.testing.assert_allclose(first.center_value, second.center_value, atol=1e-12)


@pytest.mark.parametrize("invalid_kind", ["sentinel", "value_nan", "score_nan"])
def test_invalid_rows_do_not_contaminate_valid_starts(invalid_kind):
    def target(positions):
        value, score, _ = _quadratic()(positions)
        domain = positions[:, 0] > 0.0
        if invalid_kind == "score_nan":
            score = tf.where(
                domain[:, None], score, tf.constant(float("nan"), tf.float64)
            )
        else:
            bad_value = 1e200 if invalid_kind == "sentinel" else float("nan")
            value = tf.where(domain, value, tf.constant(bad_value, tf.float64))
        eligible = domain if invalid_kind == "sentinel" else tf.ones_like(domain)
        return value, score, eligible

    cfg = _config()
    result = locate_batched_local_center(
        target, [[0.8, -0.7], [-0.8, 0.7]], [0.5, 2.0], config=cfg
    )
    assert bool(result.accepted), result.payload()
    assert result.valid_rows.numpy().tolist() == [True, False]
    assert not bool(result.optimizer_converged[1])
    assert int(result.invalid_target_rows) > 0
    np.testing.assert_allclose(result.center, [0.2, -0.15], atol=1e-6)
    _accounting(result, 2, cfg)


def test_flat_ties_stop_after_no_actual_movement():
    starts = [[2.0, 1.0], [-2.0, 1.0], [1.0, -2.0]]
    cfg = _config()
    result = locate_batched_local_center(_flat, starts, [1.0, 1.0], config=cfg)
    assert bool(result.accepted)
    assert int(result.rounds_completed) == 1
    assert int(result.selected_evaluation_index) == 0
    np.testing.assert_array_equal(result.best_evaluation_indices, [0, 1, 2])
    np.testing.assert_array_equal(result.center, starts[0])
    _accounting(result, 3, cfg)


def test_global_tie_uses_evaluation_index_not_row_index(monkeypatch):
    def optimizer(function, initial_position, **kwargs):
        function(initial_position)
        endpoint = tf.constant([[-4.0 * np.arctanh(0.25)], [0.0]], tf.float64)
        return _fake_result(function, endpoint)

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(
        _quadratic([0.0], [[1.0]]),
        [[1.0], [0.0]],
        [1.0],
        config=_config(trust_refinement_rounds=1),
    )
    assert bool(result.accepted)
    assert int(result.selected_evaluation_index) == 1


def test_best_exact_callback_survives_lower_optimizer_endpoint(monkeypatch):
    def optimizer(function, initial_position, **kwargs):
        function(
            tf.fill(
                initial_position.shape, tf.constant(4.0 * np.arctanh(0.2), tf.float64)
            )
        )
        return _fake_result(function, initial_position)

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(
        _quadratic([0.8], [[1.0]]),
        [[0.0], [-1.0]],
        [1.0],
        config=_config(trust_refinement_rounds=1),
    )
    assert bool(result.accepted), result.payload()
    np.testing.assert_allclose(result.center, [0.8], atol=1e-14)
    assert float(result.center_value) > float(tf.reduce_max(result.endpoint_values))


@pytest.mark.parametrize("component", ["value", "gradient"])
def test_endpoint_replay_mismatch_vetoes_even_higher_values(monkeypatch, component):
    def optimizer(function, initial_position, **kwargs):
        result = _fake_result(function, initial_position)
        if component == "value":
            result.objective_value += 1.0
        else:
            result.objective_gradient += 1.0
        return result

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(_flat, [[0.0], [1.0]], [1.0], config=_config())
    assert not bool(result.accepted)
    assert result.status == "replay_mismatch"
    assert not bool(tf.reduce_any(result.endpoint_accepted))


@pytest.mark.parametrize("component", ["value", "score", "eligibility", "row_context"])
def test_selected_replay_checks_every_duplicate_value_score_and_eligibility(component):
    def target(positions):
        value, score, eligible = _flat(positions)
        duplicate = tf.reduce_all(positions == positions[:1])
        if component == "value":
            value += tf.cast(duplicate, tf.float64)
        elif component == "score":
            score += tf.cast(duplicate, tf.float64)
        elif component == "eligibility":
            eligible &= ~duplicate
        else:
            value += tf.cast(duplicate, tf.float64) * tf.constant(
                [0.0, 1.0], tf.float64
            )
        return value, score, eligible

    result = locate_batched_local_center(
        target, [[0.0], [1.0]], [1.0], config=_config()
    )
    assert not bool(result.accepted)
    assert result.status == "replay_mismatch"
    assert float(result.center_value) == 0.0
    np.testing.assert_array_equal(result.center_score, [0.0])


def test_saturated_chart_is_not_reported_stationary(monkeypatch):
    def optimizer(function, initial_position, **kwargs):
        return _fake_result(
            function, tf.fill(initial_position.shape, tf.constant(100.0, tf.float64))
        )

    def linear(positions):
        return positions[:, 0], tf.ones_like(positions), tf.ones([2], tf.bool)

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(
        linear, [[0.0], [1.0]], [1.0], config=_config()
    )
    assert bool(result.accepted)
    assert result.status == "localized"
    assert bool(tf.reduce_all(result.chart_saturated))
    assert not bool(tf.reduce_any(result.optimizer_converged))
    assert int(result.rounds_completed) == 2
    np.testing.assert_array_equal(result.center, [9.0])


def test_failed_selected_optimizer_is_not_promoted(monkeypatch):
    def optimizer(function, initial_position, **kwargs):
        return _fake_result(function, initial_position, failed=True)

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(_flat, [[0.0], [1.0]], [1.0], config=_config())
    assert not bool(result.accepted)
    assert result.status == "selected_optimizer_failed"


def test_nonfinite_positions_cannot_enter_ledger(monkeypatch):
    def optimizer(function, initial_position, **kwargs):
        return _fake_result(
            function, tf.fill(initial_position.shape, tf.constant(100.0, tf.float64))
        )

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(
        _flat, [[0.0], [1.0]], [1e308], config=_config()
    )
    assert not bool(result.accepted)
    assert bool(tf.reduce_all(tf.math.is_finite(result.best_positions)))
    assert not bool(tf.reduce_any(result.endpoint_valid))
    assert int(result.invalid_target_rows) > 0


@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("cap", [1, 2, 8])
def test_cap_is_physical_prompt_and_fails_closed(jit_compile, cap):
    cfg = _config(max_optimizer_callback_batches_per_round=cap, jit_compile=jit_compile)
    result = locate_batched_local_center(
        _quadratic(), [[-0.5, 0.8], [0.8, -0.7]], [1.0, 1.0], config=cfg
    )
    assert not bool(result.accepted)
    assert result.status == "callback_cap_exhausted"
    assert int(result.optimizer_target_batches) == cap
    assert int(result.optimizer_callback_attempts) <= cap + 2
    assert bool(result.replay_consistent)
    assert int(result.invalid_target_rows) == 0
    assert not bool(tf.reduce_any(result.endpoint_accepted))
    assert bool(tf.reduce_all(tf.math.is_finite(result.center)))
    _accounting(result, 2, cfg)


def test_all_invalid_batch_is_json_safe_without_optimizer_calls():
    def target(positions):
        value, score, valid = _flat(positions)
        return value, score, ~valid

    cfg = _config()
    result = locate_batched_local_center(target, [[0.0], [1.0]], [1.0], config=cfg)
    assert not bool(result.accepted)
    assert result.status == "initial_target_invalid"
    assert int(result.target_callback_batches) == 1
    assert int(result.optimizer_callback_attempts) == 0
    assert result.payload()["center_value"] is None
    _accounting(result, 2, cfg)


def test_optimizer_exception_propagates_without_a_success_result(monkeypatch):
    def optimizer(*args, **kwargs):
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        locate_batched_local_center(_flat, [[0.0], [1.0]], [1.0], config=_config())


@pytest.mark.parametrize("bad_output", ["shape", "eligibility_dtype", "exception"])
def test_malformed_target_fails_closed(bad_output):
    def target(positions):
        value, score, eligible = _flat(positions)
        if bad_output == "shape":
            return value[:, None], score, eligible
        if bad_output == "eligibility_dtype":
            return value, score, tf.cast(eligible, tf.int32)
        raise ValueError("synthetic target failure")

    with pytest.raises((ValueError, TypeError)):
        locate_batched_local_center(target, [[0.0], [1.0]], [1.0], config=_config())


@pytest.mark.parametrize("case", ["gaussian", "invalid_row", "cap", "all_invalid"])
def test_entire_host_xla_program_matches_non_xla_values_scores_and_center(case):
    quadratic = _quadratic()

    def target(positions):
        values, scores, eligible = quadratic(positions)
        if case == "invalid_row":
            eligible &= positions[:, 0] > 0.0
        elif case == "all_invalid":
            eligible &= False
        return values, scores, eligible

    starts = [[-0.5, 0.8], [0.8, -0.7]]
    limit = 1 if case == "cap" else 64
    reference_cfg = _config(max_optimizer_callback_batches_per_round=limit)
    compiled_cfg = _config(
        jit_compile=True, max_optimizer_callback_batches_per_round=limit
    )
    reference = locate_batched_local_center(
        target, starts, [0.5, 2.0], config=reference_cfg
    )
    compiled = locate_batched_local_center(
        target, starts, [0.5, 2.0], config=compiled_cfg
    )
    expected = case in {"gaussian", "invalid_row"}
    assert bool(reference.accepted) is expected and bool(compiled.accepted) is expected
    assert reference.status == compiled.status
    for name in ("center", "center_value", "center_score"):
        np.testing.assert_allclose(
            getattr(reference, name), getattr(compiled, name), atol=1e-7
        )
    _accounting(reference, 2, reference_cfg)
    _accounting(compiled, 2, compiled_cfg)


def test_lower_endpoint_cannot_move_anchor(monkeypatch):
    def optimizer(function, initial_position, **kwargs):
        function(initial_position)
        return _fake_result(function, initial_position + 1.0)

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(
        _quadratic([0.0], [[1.0]]),
        [[0.0], [0.1]],
        [1.0],
        config=_config(),
    )
    assert bool(result.accepted)
    assert not bool(tf.reduce_any(result.endpoint_accepted))
    assert int(result.rounds_completed) == 1
    np.testing.assert_array_equal(result.center, [0.0])


@pytest.mark.parametrize("component", ["value", "score"])
def test_replay_tolerance_accepts_roundoff_without_replacing_exact_record(component):
    def target(positions):
        value, score, eligible = _flat(positions)
        duplicate = tf.cast(tf.reduce_all(positions == positions[:1]), tf.float64)
        if component == "value":
            value += duplicate * 5e-11
        else:
            score += duplicate * 5e-11
        return value, score, eligible

    result = locate_batched_local_center(
        target, [[0.0], [1.0]], [1.0], config=_config()
    )
    assert bool(result.accepted)
    assert float(result.center_value) == 0.0
    np.testing.assert_array_equal(result.center_score, [0.0])


def test_runtime_has_no_host_decisions_or_direct_numpy_and_public_import_is_lazy():
    tree = ast.parse(Path(local_center.__file__).read_text())
    runtime = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "locate_batched_local_center"
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr
        in {"numpy", "py_function", "numpy_function", "map_fn", "vectorized_map"}
        for node in ast.walk(runtime)
    )
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any("numpy" in ast.unparse(node) for node in imports)
    code = """
import sys
from bayesfilter.inference import locate_batched_local_center
assert callable(locate_batched_local_center)
assert not any(name.startswith(('bayesfilter.inference.hmc', 'filters.',
    'inference.hmc', 'inference.mass_matrix', 'inference.posterior_adapter')) for name in sys.modules)
print('lazy runtime import passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lazy runtime import passed" in result.stdout


def test_cap_status_preserves_distinct_invalid_support_accounting():
    def target(positions):
        values, scores, _ = _quadratic()(positions)
        valid = positions[:, 0] >= 0.0
        return values, scores, valid

    cfg = _config(max_optimizer_callback_batches_per_round=1)
    result = locate_batched_local_center(
        target, [[-0.5, 0.8], [0.8, -0.7]], [1.0, 1.0], config=cfg
    )
    assert not bool(result.accepted)
    assert result.status == "callback_cap_exhausted"
    assert bool(result.cap_exhausted)
    assert int(result.invalid_target_rows) > 0
    assert int(result.optimizer_callback_attempts) <= 3
    assert bool(result.replay_consistent)


@pytest.mark.parametrize("component", ["value", "score"])
def test_cap_does_not_bypass_endpoint_replay(monkeypatch, component):
    def optimizer(function, initial_position, **kwargs):
        result = _fake_result(function, initial_position)
        function(initial_position)
        if component == "value":
            result.objective_value += 1.0
        else:
            result.objective_gradient += 1.0
        return result

    monkeypatch.setattr(local_center.tfp.optimizer, "lbfgs_minimize", optimizer)
    result = locate_batched_local_center(
        _quadratic(), [[-0.5, 0.8], [0.8, -0.7]], [1.0, 1.0],
        config=_config(max_optimizer_callback_batches_per_round=1),
    )
    assert result.status == "callback_cap_exhausted"
    assert not bool(result.accepted)
    assert not bool(result.replay_consistent)


def test_installed_tfp_infinity_bisects_but_nan_aborts():
    def count_requests(sentinel):
        calls = tf.Variable(0, dtype=tf.int32)

        def objective(position):
            request = calls.assign_add(1)
            return tf.cond(
                request <= 1,
                lambda: (tf.reduce_sum(tf.square(position)), 2.0 * position),
                lambda: (tf.constant(sentinel, tf.float64), tf.zeros_like(position)),
            )

        @tf.function(autograph=False)
        def run():
            return local_center.tfp.optimizer.lbfgs_minimize(
                objective, tf.constant([1.0], tf.float64),
                max_iterations=2, max_line_search_iterations=1,
            )

        result = run()
        assert bool(result.failed)
        return int(calls)

    assert count_requests(float("inf")) > 100
    assert count_requests(float("nan")) <= 3


def test_xla_counters_are_accelerator_placeable_and_accounting_is_unchanged():
    """Int32 resources are host-only; int64 counters preserve whole-program XLA.

    This runs on the process-visible default device (CPU oracle or configured
    GPU). The GPU runner must establish and verify memory growth before tests.
    """
    cfg = _config(jit_compile=True)
    result = locate_batched_local_center(
        _quadratic(), [[-0.5, 0.8], [0.8, -0.7], [1.0, 1.2]],
        [0.5, 2.0], config=cfg,
    )
    assert bool(result.accepted) and bool(result.replay_consistent)
    assert result.target_callback_batches.dtype == tf.int64
    assert result.best_evaluation_indices.dtype == tf.int64
    np.testing.assert_allclose(result.center, [0.2, -0.15], atol=1e-7)
    _accounting(result, 3, cfg)
