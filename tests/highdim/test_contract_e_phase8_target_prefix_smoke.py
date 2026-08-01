from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import tensorflow as tf

from docs.benchmarks import (
    emit_contract_e_canonical_lgssm_phase8_target_prefix_smoke as smoke,
)


def _telemetry_fixture() -> dict[str, tf.Tensor]:
    values = {}
    for name, shape in smoke.TELEMETRY_SHAPES.items():
        if name == "active_reset_history":
            values[name] = tf.ones(shape, tf.bool)
        else:
            values[name] = tf.zeros(shape, tf.float64)
    return values


def test_frozen_target_prefix_identity_and_configuration() -> None:
    observations = smoke._target_observations()
    assert observations.shape == (1, 3)
    assert smoke._tensor_sha256(observations) == smoke.EXPECTED_OBSERVATION_SHA256
    assert smoke.THETA == (0.72, 0.55, 0.35, 0.35, 0.45)
    assert smoke.NUM_PARTICLES == 4
    assert smoke.TIME_STEPS == 1
    assert smoke.RIDGE == 4.0
    assert smoke.SINKHORN_STEPS == 2


def test_hmc_chain_factors_match_declared_coordinates() -> None:
    theta = tf.constant(smoke.THETA, tf.float64)
    tf.debugging.assert_equal(
        smoke._hmc_chain_factors(theta),
        tf.constant(
            [1.0 - 0.72**2, 1.0 - 0.55**2, 1.0 - 0.35**2, 0.35, 0.45],
            tf.float64,
        ),
    )


def test_telemetry_schema_accepts_exact_shapes_and_rejects_drift() -> None:
    values = _telemetry_fixture()
    records, checks = smoke._validate_telemetry(values)
    assert set(records) == set(smoke.TELEMETRY_SHAPES)
    assert all(checks.values())

    missing = dict(values)
    del missing["quotient_mass_history"]
    _, missing_checks = smoke._validate_telemetry(missing)
    assert not missing_checks["required_field_set_complete"]

    wrong_shape = dict(values)
    wrong_shape["target_mean_history"] = tf.zeros([1, 1, 2], tf.float64)
    _, shape_checks = smoke._validate_telemetry(wrong_shape)
    assert not shape_checks["all_static_shapes_match"]

    nonfinite = dict(values)
    nonfinite["realized_ridge_history"] = tf.constant([[float("nan")]], tf.float64)
    _, finite_checks = smoke._validate_telemetry(nonfinite)
    assert not finite_checks["all_required_values_finite"]


def test_serialized_repeatability_is_exact_and_key_bound() -> None:
    first = {
        "value": tf.constant([1.0, 2.0], tf.float64),
        "mask": tf.constant([True, False]),
    }
    second = {
        "value": tf.constant([1.0, 2.0], tf.float64),
        "mask": tf.constant([True, False]),
    }
    assert smoke._all_tensor_outputs_identical(first, second)
    second["value"] = tf.constant([1.0, 2.0 + 1.0e-15], tf.float64)
    assert not smoke._all_tensor_outputs_identical(first, second)
    assert not smoke._all_tensor_outputs_identical(first, {"value": first["value"]})


def test_exclusive_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    smoke._write_json_exclusive(path, {"first": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"first": True}
    with pytest.raises(FileExistsError):
        smoke._write_json_exclusive(path, {"second": True})


def test_smoke_source_has_only_canonical_and_kalman_routes() -> None:
    source = Path(smoke.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "ledh_contract_e_canonical_lgssm_tf" in imported
    assert "ledh_contract_e_lgssm_preparation_tf" in imported
    assert "tf_kalman_log_likelihood" in imported
    assert "historical_raw" not in source
    assert "compact_score" not in source
    assert "raw_barycentric" not in source
