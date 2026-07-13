from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (
    A0_IMMUTABLE_AGGREGATE_SHA256,
    A0_SIGNATURE_AGGREGATE_SHA256,
    A0_TARGET_LOCK_FILE_SHA256,
    FREE_INDICES,
    FREE_PARAMETER_NAMES,
    FULL_FIXTURE_RAW_SHA256,
    GOLDEN_SIGNATURES_FILE_SHA256,
    MASKED_POSTERIOR_CONTRACT_SHA256,
    OBSERVATION_RAW_SHA256,
    PARAMETER_MASK_SHA256,
    PRIOR_CENTER_RAW_SHA256,
    SSLLSTMParameterMask,
    SSLLSTMPosteriorConfig,
    SSLLSTMPosteriorTarget,
    TARGET_SEMANTIC_SHA256,
    locked_ssl_lstm_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = (
    ROOT
    / "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json"
)
HISTORICAL_PATH = (
    ROOT / "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py"
)
HISTORICAL_SHA256 = "fea73716e1d972a5336e3bdedb733dfc31c4a0bb61cf40cdf877d577d68cbe28"

TRUTH_FREE = np.array([0.35, -0.08, 0.65, 0.05], dtype=np.float64)
PHASE2S_CENTER = np.array(
    [0.5704394246369003, -0.1242247342531544, 0.6609123192759063, 0.1354211218811133],
    dtype=np.float64,
)
SHELL_STEP = 0.25 * 0.35
FD_STEP = 1.0e-5
FD_RTOL = 5.0e-3
FD_ATOL = 8.0e-4
HISTORICAL_SCALE = 8.0 * (2.0**-52)


def _finite_points() -> list[tuple[str, np.ndarray]]:
    points = [("truth_free", TRUTH_FREE.copy()), ("phase2s_center", PHASE2S_CENTER.copy())]
    for index in range(4):
        minus = PHASE2S_CENTER.copy()
        plus = PHASE2S_CENTER.copy()
        minus[index] -= SHELL_STEP
        plus[index] += SHELL_STEP
        points.extend(
            [
                (f"shell_{index}_minus", minus),
                (f"shell_{index}_plus", plus),
            ]
        )
    return points


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        parse_constant=reject,
        object_pairs_hook=pairs,
    )


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _historical_target() -> Any:
    assert hashlib.sha256(HISTORICAL_PATH.read_bytes()).hexdigest() == HISTORICAL_SHA256
    spec = importlib.util.spec_from_file_location("ssl_lstm_a1_historical", HISTORICAL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_filtering_geometry_target()


@pytest.fixture(scope="module")
def target() -> SSLLSTMPosteriorTarget:
    return locked_ssl_lstm_posterior_target()


@pytest.fixture(scope="module")
def historical_target() -> Any:
    return _historical_target()


def test_golden_payloads_and_independent_digests() -> None:
    golden = _strict_json(GOLDEN_PATH)
    mask = SSLLSTMParameterMask()
    config = SSLLSTMPosteriorConfig(parameter_mask=mask)

    assert hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest() == GOLDEN_SIGNATURES_FILE_SHA256
    assert mask.signature_payload() == golden["parameter_mask"]["payload"]
    assert config.signature_payload() == golden["masked_posterior_contract"]["payload"]
    assert _canonical_digest(golden["parameter_mask"]["payload"]) == PARAMETER_MASK_SHA256
    assert _canonical_digest(golden["masked_posterior_contract"]["payload"]) == MASKED_POSTERIOR_CONTRACT_SHA256
    assert mask.signature() == golden["parameter_mask"]["sha256"]
    assert config.signature() == golden["masked_posterior_contract"]["sha256"]


def _leaf_paths(value: Any, prefix: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaf_paths(value[key], prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, prefix + (index,))
    else:
        yield prefix


def _mutate_leaf(payload: Any, path: tuple[Any, ...]) -> Any:
    mutated = copy.deepcopy(payload)
    parent = mutated
    for key in path[:-1]:
        parent = parent[key]
    key = path[-1]
    value = parent[key]
    if isinstance(value, bool):
        parent[key] = not value
    elif isinstance(value, int):
        parent[key] = value + 1
    elif isinstance(value, str):
        parent[key] = value + "_mutated"
    else:
        raise AssertionError(f"unsupported golden leaf type: {type(value)}")
    return mutated


def test_every_golden_semantic_leaf_affects_its_digest() -> None:
    golden = _strict_json(GOLDEN_PATH)
    for role in ("parameter_mask", "masked_posterior_contract"):
        payload = golden[role]["payload"]
        expected = golden[role]["sha256"]
        assert _canonical_digest(payload) == expected
        for path in _leaf_paths(payload):
            assert _canonical_digest(_mutate_leaf(payload, path)) != expected, (role, path)


def test_mask_embed_extract_and_locked_raw_hashes() -> None:
    mask = SSLLSTMParameterMask()
    free = tf.constant(PHASE2S_CENTER, dtype=tf.float64)
    full = mask.embed(free)

    assert mask.full_dimension == 24
    assert mask.free_dimension == 4
    assert mask.free_indices == FREE_INDICES
    assert mask.free_parameter_names == FREE_PARAMETER_NAMES
    tf.debugging.assert_equal(mask.extract(full), free)
    expected = mask.full_values.numpy().copy()
    expected[list(FREE_INDICES)] = PHASE2S_CENTER
    np.testing.assert_array_equal(full.numpy(), expected)
    assert hashlib.sha256(mask.full_values.numpy().astype("<f8").tobytes()).hexdigest() == FULL_FIXTURE_RAW_SHA256


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"full_parameter_names": ("duplicate",) * 24}, ValueError),
        ({"free_parameter_names": (FREE_PARAMETER_NAMES[0],) * 4}, ValueError),
        ({"free_indices": (12, 12, 14, 15)}, ValueError),
        ({"free_indices": (12, 13, 14, 24)}, ValueError),
        ({"free_indices": (13, 12, 14, 15)}, ValueError),
        ({"full_values": tf.zeros([23], tf.float64)}, ValueError),
        ({"full_values": tf.zeros([24], tf.float32)}, TypeError),
        (
            {"full_values": tf.tensor_scatter_nd_update(tf.constant([0.0] * 24, tf.float64), [[0]], [np.nan])},
            ValueError,
        ),
    ],
)
def test_mask_rejects_invalid_contracts(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        SSLLSTMParameterMask(**kwargs)


def test_config_binds_exact_target_inputs() -> None:
    config = SSLLSTMPosteriorConfig()
    assert config.static_config.parameter_dim == 24
    assert config.observations.shape == (30, 1)
    assert config.prior_center.shape == (4,)
    assert config.prior_standard_deviation == 4.0
    assert config.prior_normalized is False
    assert config.filter_name == "svd_ukf"
    assert config.jit_compile is True
    assert config.execution_role == "default_xla"
    assert hashlib.sha256(config.observations.numpy().astype("<f8").tobytes()).hexdigest() == OBSERVATION_RAW_SHA256
    assert hashlib.sha256(config.prior_center.numpy().astype("<f8").tobytes()).hexdigest() == PRIOR_CENTER_RAW_SHA256


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"static_config": SSLLSTMStaticConfig(29, 1, 1, 1)}, ValueError),
        ({"observations": tf.zeros([30, 1], tf.float32)}, TypeError),
        ({"observations": tf.zeros([29, 1], tf.float64)}, ValueError),
        ({"observations": tf.fill([30, 1], tf.constant(np.nan, tf.float64))}, ValueError),
        ({"prior_center": tf.zeros([4], tf.float32)}, TypeError),
        ({"prior_center": tf.zeros([5], tf.float64)}, ValueError),
        ({"prior_standard_deviation": 5.0}, ValueError),
        ({"prior_normalized": True}, ValueError),
        ({"filter_name": "principal_sqrt_ukf"}, ValueError),
        ({"spectral_gap_tolerance": 1.0e-8}, ValueError),
        ({"allow_fixed_null_support": True}, ValueError),
        ({"return_filtered": True}, ValueError),
        ({"jit_compile": False}, ValueError),
        ({"backend": "jax"}, ValueError),
        ({"dtype": "float32"}, ValueError),
    ],
)
def test_config_rejects_target_drift(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        SSLLSTMPosteriorConfig(**kwargs)


def test_explicit_nonxla_debug_config_cannot_publish_evidence() -> None:
    config = SSLLSTMPosteriorConfig(jit_compile=False, execution_role="eager_debug_reference")
    debug_target = SSLLSTMPosteriorTarget(config)
    with pytest.raises(RuntimeError, match="non-XLA debug"):
        debug_target.target_signature()
    with pytest.raises(RuntimeError, match="non-XLA debug"):
        debug_target.adapter_signature()


def test_target_manifest_and_capability_are_four_dimensional(target: SSLLSTMPosteriorTarget) -> None:
    capability = target.value_score_capability()
    manifest = target.adapter_manifest_payload()
    assert target.parameter_dim == 4
    assert target.parameter_names == FREE_PARAMETER_NAMES
    assert target.target_signature() == TARGET_SEMANTIC_SHA256
    assert target.adapter_signature() == MASKED_POSTERIOR_CONTRACT_SHA256
    assert manifest["parameter_dim"] == 4
    assert tuple(manifest["parameter_names"]) == FREE_PARAMETER_NAMES
    assert capability.value_score_authority == "graph_native"
    assert capability.xla_hmc_ready is False
    assert capability.full_chain_xla_diagnostic_ready is False
    assert capability.target_scope == target.target_scope


@pytest.mark.parametrize(
    "value,error",
    [
        (tf.zeros([4], tf.float32), TypeError),
        (tf.zeros([3], tf.float64), ValueError),
        (tf.zeros([1, 4], tf.float64), ValueError),
    ],
)
def test_scalar_surfaces_reject_wrong_dtype_or_shape(
    target: SSLLSTMPosteriorTarget,
    value: tf.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        target.value_and_score(value)


@pytest.mark.parametrize(
    "value,error",
    [
        (tf.zeros([4], tf.float64), ValueError),
        (tf.zeros([0, 4], tf.float64), ValueError),
        (tf.zeros([2, 3], tf.float64), ValueError),
        (tf.zeros([2, 4], tf.float32), TypeError),
        (tf.zeros([1, 2, 4], tf.float64), ValueError),
    ],
)
def test_batch_surface_rejects_wrong_dtype_or_shape(
    target: SSLLSTMPosteriorTarget,
    value: tf.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        target.batch_value_and_score(value)


def test_batch_surface_rejects_unknown_batch_dimension(
    target: SSLLSTMPosteriorTarget,
) -> None:
    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float64)])
    def dynamic_batch(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return target.batch_value_and_score(values)

    with pytest.raises(ValueError, match="static and positive"):
        dynamic_batch.get_concrete_function()


def test_historical_route_matches_at_all_ten_points(
    target: SSLLSTMPosteriorTarget,
    historical_target: Any,
) -> None:
    for name, point in _finite_points():
        current_value, current_score = target.eager_debug_value_and_score(
            tf.constant(point, tf.float64)
        )
        historical_value, historical_score = historical_target._value_and_score_impl(
            tf.constant(point, tf.float64)
        )
        current_value_float = float(current_value.numpy())
        historical_value_float = float(historical_value.numpy())
        current_score_array = np.asarray(current_score.numpy(), dtype=np.float64)
        historical_score_array = np.asarray(historical_score.numpy(), dtype=np.float64)
        value_tolerance = HISTORICAL_SCALE * max(
            1.0,
            abs(current_value_float),
            abs(historical_value_float),
        )
        score_tolerance = HISTORICAL_SCALE * max(
            1.0,
            float(np.max(np.abs(current_score_array))),
            float(np.max(np.abs(historical_score_array))),
        )
        assert abs(current_value_float - historical_value_float) <= value_tolerance, name
        assert float(np.max(np.abs(current_score_array - historical_score_array))) <= score_tolerance, name


def test_score_matches_centered_finite_difference_at_all_ten_points(
    target: SSLLSTMPosteriorTarget,
) -> None:
    for name, point in _finite_points():
        _value, score = target.eager_debug_value_and_score(tf.constant(point, tf.float64))
        finite_difference = np.empty(4, dtype=np.float64)
        for index in range(4):
            plus = point.copy()
            minus = point.copy()
            plus[index] += FD_STEP
            minus[index] -= FD_STEP
            plus_value, _ = target.eager_debug_value_and_score(tf.constant(plus, tf.float64))
            minus_value, _ = target.eager_debug_value_and_score(tf.constant(minus, tf.float64))
            finite_difference[index] = (
                float(plus_value.numpy()) - float(minus_value.numpy())
            ) / (2.0 * FD_STEP)
        np.testing.assert_allclose(
            score.numpy(),
            finite_difference,
            rtol=FD_RTOL,
            atol=FD_ATOL,
            err_msg=name,
        )


def test_eager_cpu_xla_parity_and_compiled_default_at_all_ten_points(
    target: SSLLSTMPosteriorTarget,
) -> None:
    before = target.compiled_scalar_trace_count()
    for name, point in _finite_points():
        tensor = tf.constant(point, tf.float64)
        eager_value, eager_score = target.eager_debug_value_and_score(tensor)
        compiled_value, compiled_score = target.value_and_score(tensor)
        eager_value_float = float(eager_value.numpy())
        compiled_value_float = float(compiled_value.numpy())
        value_tolerance = 1.0e-10 * max(
            1.0,
            abs(eager_value_float),
            abs(compiled_value_float),
        )
        assert abs(compiled_value_float - eager_value_float) <= value_tolerance, name
        np.testing.assert_allclose(
            compiled_score.numpy(),
            eager_score.numpy(),
            rtol=0.0,
            atol=1.0e-8,
            err_msg=name,
        )
    assert target.compiled_scalar_trace_count() >= max(1, before)
    assert target.compiled_scalar_trace_count() == 1


def test_callable_aliases_and_custom_gradient(target: SSLLSTMPosteriorTarget) -> None:
    point = tf.constant(TRUTH_FREE, tf.float64)
    value, score = target.value_and_score(point)
    alias_value, alias_score = target.log_prob_and_grad(point)
    with tf.GradientTape() as tape:
        tape.watch(point)
        log_prob = target.log_prob(point)
    gradient = tape.gradient(log_prob, point)

    tf.debugging.assert_equal(target.value(point), value)
    tf.debugging.assert_equal(target.score(point), score)
    tf.debugging.assert_equal(alias_value, value)
    tf.debugging.assert_equal(alias_score, score)
    tf.debugging.assert_equal(log_prob, value)
    tf.debugging.assert_equal(gradient, score)


@pytest.mark.parametrize("batch_size", [1, 4, 10])
def test_static_batch_xla_shapes_values_order_and_cache(
    target: SSLLSTMPosteriorTarget,
    batch_size: int,
) -> None:
    rows = [point for _name, point in _finite_points()[:batch_size]]
    batch = tf.constant(np.stack(rows), tf.float64)
    values, scores = target.batch_value_and_score(batch)
    assert values.shape == (batch_size,)
    assert scores.shape == (batch_size, 4)
    expected = [target.value_and_score(tf.constant(row, tf.float64)) for row in rows]
    expected_values = np.array([float(value.numpy()) for value, _score in expected])
    actual_values = np.asarray(values.numpy(), dtype=np.float64)
    value_tolerances = 1.0e-10 * np.maximum(
        1.0,
        np.maximum(np.abs(actual_values), np.abs(expected_values)),
    )
    assert np.all(np.abs(actual_values - expected_values) <= value_tolerances)
    np.testing.assert_allclose(
        scores.numpy(),
        np.stack([score.numpy() for _value, score in expected]),
        rtol=0.0,
        atol=1.0e-8,
    )
    repeated_values, repeated_scores = target.batch_value_and_score(batch)
    tf.debugging.assert_equal(values, repeated_values)
    tf.debugging.assert_equal(scores, repeated_scores)
    assert batch_size in target.compiled_batch_sizes()


def _guarded_testing_branch(free: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    free = tf.debugging.check_numerics(free, "testing finite branch received nonfinite input")
    return -0.5 * tf.reduce_sum(tf.square(free)), -free


def test_nonfinite_reject_is_exact_and_skips_guarded_finite_branch() -> None:
    testing_target = SSLLSTMPosteriorTarget(
        finite_branch_callable=_guarded_testing_branch,
        testing_only=True,
    )
    nan_row = tf.constant([np.nan, -0.08, 0.65, 0.05], tf.float64)
    inf_row = tf.constant([np.inf, -0.08, 0.65, 0.05], tf.float64)
    for row in (nan_row, inf_row):
        value, score, status = testing_target.diagnostic_value_and_score(row)
        assert float(value.numpy()) == -1.0e100
        np.testing.assert_array_equal(score.numpy(), np.zeros(4))
        assert int(status.numpy()) == 1
        with tf.GradientTape() as tape:
            tape.watch(row)
            log_prob = testing_target.log_prob(row)
        np.testing.assert_array_equal(tape.gradient(log_prob, row).numpy(), np.zeros(4))

    batch = tf.stack([tf.constant(TRUTH_FREE, tf.float64), nan_row, inf_row])
    values, scores, statuses = testing_target.diagnostic_value_and_score(batch)
    assert statuses.numpy().tolist() == [0, 1, 1]
    assert values.numpy().tolist()[1:] == [-1.0e100, -1.0e100]
    np.testing.assert_array_equal(scores.numpy()[1:], np.zeros([2, 4]))


def test_valid_branch_is_bitwise_equal_to_direct_call_at_both_anchors() -> None:
    testing_target = SSLLSTMPosteriorTarget(
        finite_branch_callable=_guarded_testing_branch,
        testing_only=True,
    )
    for point in (TRUTH_FREE, PHASE2S_CENTER):
        tensor = tf.constant(point, tf.float64)
        direct_value, direct_score = _guarded_testing_branch(tensor)
        wrapped_value, wrapped_score = testing_target.eager_debug_value_and_score(tensor)
        tf.debugging.assert_equal(wrapped_value, direct_value)
        tf.debugging.assert_equal(wrapped_score, direct_score)
        _compiled_value, _compiled_score, status = testing_target.diagnostic_value_and_score(tensor)
        assert int(status.numpy()) == 0


def test_finite_filter_failure_remains_loud() -> None:
    def failing_branch(free: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        del free
        return tf.constant(np.nan, tf.float64), tf.zeros([4], tf.float64)

    debug_config = SSLLSTMPosteriorConfig(
        jit_compile=False,
        execution_role="eager_debug_reference",
    )
    testing_target = SSLLSTMPosteriorTarget(
        debug_config,
        finite_branch_callable=failing_branch,
        testing_only=True,
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="finite target value"):
        testing_target.eager_debug_value_and_score(tf.constant(TRUTH_FREE, tf.float64))


def test_testing_only_target_cannot_publish_provenance_or_artifacts() -> None:
    testing_target = SSLLSTMPosteriorTarget(
        finite_branch_callable=_guarded_testing_branch,
        testing_only=True,
    )
    capability = testing_target.value_score_capability()
    assert capability.value_score_authority == "debug_only"
    assert capability.xla_hmc_ready is False
    assert capability.full_chain_xla_diagnostic_ready is False
    with pytest.raises(RuntimeError, match="testing-only"):
        testing_target.target_signature()
    with pytest.raises(RuntimeError, match="testing-only"):
        testing_target.adapter_signature()
    with pytest.raises(RuntimeError, match="testing-only"):
        testing_target.adapter_manifest_payload()
    with pytest.raises(RuntimeError, match="testing-only"):
        testing_target.assert_production_evidence_target()


def test_nondefault_finite_branch_requires_testing_authority() -> None:
    with pytest.raises(ValueError, match="testing_only"):
        SSLLSTMPosteriorTarget(finite_branch_callable=_guarded_testing_branch)
    with pytest.raises(ValueError, match="finite_branch_callable"):
        SSLLSTMPosteriorTarget(testing_only=True)


def test_lazy_exports_resolve_to_production_module() -> None:
    from bayesfilter import nonlinear

    assert nonlinear.SSLLSTMParameterMask is SSLLSTMParameterMask
    assert nonlinear.SSLLSTMPosteriorConfig is SSLLSTMPosteriorConfig
    assert nonlinear.SSLLSTMPosteriorTarget is SSLLSTMPosteriorTarget
    assert nonlinear.locked_ssl_lstm_posterior_target is locked_ssl_lstm_posterior_target


def test_a0_bindings_are_explicit_and_unchanged() -> None:
    config = SSLLSTMPosteriorConfig()
    bindings = config.signature_payload()["a0_bindings"]
    assert bindings == {
        "dependency_manifest_file_sha256": "2a1e3dcd89c0c5e24f892d14a29ef44329ef3e50c4af40093428082de6806517",
        "immutable_aggregate_sha256": A0_IMMUTABLE_AGGREGATE_SHA256,
        "signature_aggregate_sha256": A0_SIGNATURE_AGGREGATE_SHA256,
        "target_lock_file_sha256": A0_TARGET_LOCK_FILE_SHA256,
        "target_semantic_sha256": TARGET_SEMANTIC_SHA256,
    }
