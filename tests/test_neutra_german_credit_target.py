"""Focused tests for the TensorFlow German-credit target and exact score."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_german_credit_target import (
    GermanCreditValueScoreAdapter,
    constrained_from_unconstrained,
    german_credit_log_prob_batch,
    german_credit_log_prob_and_score_batch,
    load_german_credit_target_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "german-credit-source"
)
DATA = SOURCE_ROOT / "german.data-numeric"
REFERENCE = SOURCE_ROOT / "logistic_gamma_gate1_reference.json"


def _write_small_fixture(root: Path) -> tuple[Path, Path]:
    data = root / "german-small.data-numeric"
    data.write_text(
        "1 2 1\n2 4 2\n3 8 1\n4 16 2\n",
        encoding="utf-8",
    )
    reference = root / "reference.json"
    reference.write_text(
        json.dumps({"mean": [0.0] * 7, "square": [1.0] * 7}),
        encoding="utf-8",
    )
    return data, reference


def test_committed_source_copy_loads_exact_target_contract() -> None:
    spec = load_german_credit_target_spec(DATA, REFERENCE)
    assert spec.observation_count == 1000
    assert spec.feature_count == 25
    assert spec.dimension == 51
    assert spec.data_sha256 == "2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8"
    assert spec.reference_sha256 == "605fbca76b076bb23cf865f7210ef8e6da2b29c1c87964d13463126e71faeb09"
    assert len(spec.reference_mean) == len(spec.reference_square) == 51


def test_preprocessing_matches_source_no_minimum_subtraction(tmp_path: Path) -> None:
    data, reference = _write_small_fixture(tmp_path)
    spec = load_german_credit_target_spec(data, reference)
    expected = (
        (-1.0 / 3.0, -5.0 / 7.0, 1.0),
        (1.0 / 3.0, -3.0 / 7.0, 1.0),
        (1.0, 1.0 / 7.0, 1.0),
        (5.0 / 3.0, 9.0 / 7.0, 1.0),
    )
    for actual_row, expected_row in zip(spec.design, expected):
        assert actual_row == pytest.approx(expected_row, rel=0.0, abs=1.0e-15)
    assert spec.response == (0.0, 1.0, 0.0, 1.0)


def test_exact_score_matches_tensorflow_autodiff(tmp_path: Path) -> None:
    data, reference = _write_small_fixture(tmp_path)
    spec = load_german_credit_target_spec(data, reference)
    rows = tf.constant(
        (
            (0.2, -0.3, 0.4, -0.2, 0.1, 0.3, -0.1),
            (-0.4, 0.1, 0.2, 0.2, -0.3, 0.05, 0.25),
        ),
        tf.float64,
    )
    value, score = german_credit_log_prob_and_score_batch(spec, rows)
    with tf.GradientTape() as tape:
        tape.watch(rows)
        autodiff_value = german_credit_log_prob_batch(spec, rows)
        autodiff_total = tf.reduce_sum(autodiff_value)
    autodiff_score = tape.gradient(autodiff_total, rows)
    tf.debugging.assert_near(value, autodiff_value, atol=1.0e-12)
    tf.debugging.assert_near(score, autodiff_score, atol=1.0e-11)


def test_zero_state_value_and_constrained_map_match_closed_form(tmp_path: Path) -> None:
    data, reference = _write_small_fixture(tmp_path)
    spec = load_german_credit_target_spec(data, reference)
    zero = tf.zeros((1, spec.dimension), tf.float64)
    value, score = german_credit_log_prob_and_score_batch(spec, zero)
    expected = -spec.observation_count * math.log(2.0) - 0.5 * (spec.feature_count + 1)
    assert float(value[0].numpy()) == pytest.approx(expected, rel=0.0, abs=1.0e-12)
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    constrained = constrained_from_unconstrained(spec, zero)
    assert constrained.shape == (1, 7)
    assert tuple(constrained[0, 3:].numpy().tolist()) == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_adapter_is_batch_native_and_xla_ready(tmp_path: Path) -> None:
    data, reference = _write_small_fixture(tmp_path)
    spec = load_german_credit_target_spec(data, reference)
    adapter = GermanCreditValueScoreAdapter(spec)

    @tf.function(jit_compile=True)
    def compiled(rows: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return adapter.log_prob_and_grad(rows)

    value, score = compiled(tf.zeros((3, spec.dimension), tf.float64))
    assert value.shape == (3,)
    assert score.shape == (3, spec.dimension)
    capability = adapter.value_score_capability()
    assert capability.is_accepted_full_chain_xla_diagnostic_authority
