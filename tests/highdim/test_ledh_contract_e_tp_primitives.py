from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


ROOT = Path(__file__).resolve().parents[2]
WITNESS_PATH = ROOT / "docs/benchmarks/contract_e_score_aware_teacher_projection_2d_lgssm.py"
DTYPE = tf.float64
ACTIVE_INDICES = tf.constant(
    [108, 221, 2317, 2402, 2474, 3942, 4001], tf.int32
)
ROW_SCALE = tf.constant(
    [
        1.0,
        5.1012693387019965,
        4.522840653349041,
        26.022948865981103,
        21.932472504120415,
        20.456087575586782,
        0.6326344960582179,
    ],
    DTYPE,
)
EXPECTED_WEIGHTS = tf.constant(
    [
        0.0013273171630924646,
        0.00029612211860157757,
        0.03197246241434581,
        0.037785821151157827,
        0.039370360941121535,
        0.054164725039877466,
        0.8350831911718027,
    ],
    DTYPE,
)


def _load_witness():
    specification = importlib.util.spec_from_file_location("contract_e_tp_witness", WITNESS_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _witness_inputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    witness = _load_witness()
    teacher = witness._teacher(witness.THETA0)
    log_weights = tf.math.log(teacher["weights"]) + teacher["current_increment"]
    return teacher["candidates"], log_weights, teacher["features"]


def test_dense_square_reproduces_frozen_6561_to_7_witness() -> None:
    points, log_weights, features = _witness_inputs()
    result = tp._contract_e_tp_dense_square_forward_core(
        points, log_weights, features, ACTIVE_INDICES, ROW_SCALE
    )

    np.testing.assert_allclose(result["student_weights"], EXPECTED_WEIGHTS, rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(result["matched_target"], result["target"], rtol=0.0, atol=1.5e-14)
    np.testing.assert_allclose(tf.reduce_sum(result["student_weights"]), 1.0, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(result["condition_number"], 84.26064554729527, rtol=2e-13, atol=0.0)
    assert float(result["minimum_weight"].numpy()) == pytest.approx(
        0.00029612211860157757, rel=2e-12
    )
    assert float(result["scaled_relative_residual"].numpy()) <= float(
        result["forward_error_bound"].numpy()
    )


def test_dense_kkt_preserves_features_with_positive_overcomplete_chart() -> None:
    points = tf.constant([[-1.0], [0.0], [1.0]], DTYPE)
    probabilities = tf.constant([0.2, 0.5, 0.3], DTYPE)
    features = tf.stack([tf.ones([3], DTYPE), points[:, 0]])
    result = tp._contract_e_tp_dense_kkt_forward_core(
        points,
        tf.math.log(probabilities),
        features,
        tf.constant([0, 1, 2], tf.int32),
        tf.ones([2], DTYPE),
        tf.fill([3], tf.constant(1.0 / 3.0, DTYPE)),
        tf.eye(3, dtype=DTYPE),
    )

    np.testing.assert_allclose(result["matched_target"], result["target"], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(result["student_weights"], [0.2833333333333333, 1.0 / 3.0, 0.3833333333333333], rtol=0.0, atol=2e-15)
    assert float(result["minimum_weight"].numpy()) > 0.0


@pytest.mark.parametrize("failure", ["duplicate", "rank", "negative_square", "negative_kkt"])
def test_invalid_charts_fail_closed_without_clipping(failure: str) -> None:
    points = tf.constant([[-1.0], [0.0], [1.0]], DTYPE)
    features = tf.stack([tf.ones([3], DTYPE), points[:, 0]])
    log_weights = tf.math.log(tf.constant([0.2, 0.5, 0.3], DTYPE))

    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        if failure == "duplicate":
            tp._contract_e_tp_dense_square_forward_core(
                points, log_weights, features, tf.constant([0, 0]), tf.ones([2], DTYPE)
            )
        elif failure == "rank":
            rank_features = tf.stack([tf.ones([3], DTYPE), tf.constant([0.0, 0.0, 1.0], DTYPE)])
            tp._contract_e_tp_dense_square_forward_core(
                points, log_weights, rank_features, tf.constant([0, 1]), tf.ones([2], DTYPE)
            )
        elif failure == "negative_square":
            tp._contract_e_tp_dense_square_forward_core(
                points,
                tf.constant([-20.0, -20.0, 0.0], DTYPE),
                features,
                tf.constant([0, 1]),
                tf.ones([2], DTYPE),
            )
        else:
            tp._contract_e_tp_dense_kkt_forward_core(
                points,
                tf.constant([0.0, -20.0, -20.0], DTYPE),
                features,
                tf.constant([0, 1, 2]),
                tf.ones([2], DTYPE),
                tf.fill([3], tf.constant(1.0 / 3.0, DTYPE)),
                tf.eye(3, dtype=DTYPE),
            )


def test_runtime_module_has_no_active_set_selection_or_clipping() -> None:
    source = Path(tp.__file__).read_text(encoding="utf-8")
    forbidden = ("argmin", "argmax", "top_k", "clip_by_value", "relu(", "maximum(weights")
    for token in forbidden:
        assert token not in source
    assert tp.ALGORITHM_ID == "contract_e_tp_experimental_v1"
