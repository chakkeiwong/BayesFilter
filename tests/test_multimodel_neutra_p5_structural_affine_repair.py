from __future__ import annotations

import json

import pytest
import tensorflow as tf

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_plain_hmc as base
from docs.benchmarks.run_multimodel_neutra_p5_structural_affine_geometry import (
    StructuralAffineTargetAdapter,
    _affine_checks,
)
from docs.benchmarks.run_multimodel_neutra_p5_structural_affine_hmc import (
    _load_geometry,
)


class _QuadraticAdapter:
    def log_prob_and_grad(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        return -0.5 * tf.reduce_sum(values * values, axis=-1), -values

    def target_status_telemetry(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        return {
            "status_code": tf.zeros(tf.shape(values)[:-1], tf.int32),
            "valid_pre_regularized_score": tf.ones(tf.shape(values)[:-1], tf.bool),
        }


def test_structural_affine_adapter_round_trip_and_chain_rule() -> None:
    factor = tf.constant(
        [
            [1.2, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.0, 0.0, 0.0],
            [0.0, -0.2, 1.1, 0.0, 0.0],
            [0.2, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.1, 0.0, -0.1, 0.7],
        ],
        tf.float64,
    )
    adapter = StructuralAffineTargetAdapter(
        base_adapter=_QuadraticAdapter(),
        center=tf.constant([0.1, -0.2, 0.3, -0.4, 0.5], tf.float64),
        factor=factor,
        target_signature="a" * 64,
        geometry_sha256="b" * 64,
    )
    z = tf.constant([[0.0, 0.1, -0.2, 0.3, -0.4], [0.5, -0.4, 0.3, -0.2, 0.1]], tf.float64)
    assert _affine_checks(tf, adapter, z)["passed"] is True
    assert adapter.forward(tf.zeros([3, 2, 5], tf.float64)).shape == (3, 2, 5)


def test_geometry_loader_rejects_failed_geometry(tmp_path) -> None:
    result = {
        "passed": False,
        "raw_precision_spd": False,
        "terminal_hessian_stable": True,
        "score_gate_passed": True,
        "affine_checks": {"passed": True},
        "final_geometry": {},
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    hashes = {"result.json": base._file_sha256(result_path)}
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}), encoding="utf-8"
    )
    with pytest.raises(base.P4PlainHMCError, match="was not admitted"):
        _load_geometry(tmp_path)


def test_geometry_loader_rejects_recursive_hash_drift(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": {"result.json": "0" * 64}}), encoding="utf-8"
    )
    with pytest.raises(base.P4PlainHMCError, match="hash mismatch"):
        _load_geometry(tmp_path)
