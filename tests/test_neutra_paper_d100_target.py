"""Focused tests for exact TensorFlow d100 paper targets and samplers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_paper_d100_target import (
    PAPER_D100_DIMENSION,
    PaperD100TargetError,
    PaperD100TargetSpec,
    PaperD100ValueScoreAdapter,
    load_paper_gaussian_spec,
    make_paper_funnel_spec,
    paper_d100_log_prob_and_score_batch,
    paper_d100_log_prob_batch,
    paper_funnel_standardized_residuals,
    sample_paper_d100_exact,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "paper-d100/source-r1/paper_ill_cond_gaussian_d100_constants.json"
)


@pytest.mark.parametrize("target_name", ("funnel", "gaussian"))
def test_exact_scores_match_tensorflow_autodiff(target_name: str) -> None:
    spec = (
        make_paper_funnel_spec()
        if target_name == "funnel"
        else load_paper_gaussian_spec(SOURCE)
    )
    rows = sample_paper_d100_exact(spec, 5, seed=(20260813, 51001))
    with tf.GradientTape() as tape:
        tape.watch(rows)
        autodiff_value = paper_d100_log_prob_batch(spec, rows)
        total = tf.reduce_sum(autodiff_value)
    autodiff_score = tape.gradient(total, rows)
    value, score = paper_d100_log_prob_and_score_batch(spec, rows)
    tf.debugging.assert_near(value, autodiff_value, atol=1.0e-10)
    tf.debugging.assert_near(score, autodiff_score, atol=1.0e-9)


def test_funnel_closed_form_value_score_and_standardization() -> None:
    spec = make_paper_funnel_spec()
    row = tf.concat(
        (
            tf.constant([[0.2]], tf.float64),
            tf.linspace(tf.constant(-0.3, tf.float64), tf.constant(0.4, tf.float64), 99)[
                tf.newaxis, :
            ],
        ),
        axis=1,
    )
    value, score = paper_d100_log_prob_and_score_batch(spec, row)
    y = row[0, 0]
    x = row[0, 1:]
    inverse_variance = tf.exp(-2.0 * y)
    expected_value = -(
        0.5 * tf.square(y)
        + 0.5 * inverse_variance * tf.reduce_sum(tf.square(x))
        + 99.0 * y
    )
    expected_y_score = -y + inverse_variance * tf.reduce_sum(tf.square(x)) - 99.0
    tf.debugging.assert_near(value[0], expected_value, atol=1.0e-12)
    tf.debugging.assert_near(score[0, 0], expected_y_score, atol=1.0e-12)
    tf.debugging.assert_near(score[0, 1:], -inverse_variance * x, atol=1.0e-12)
    tf.debugging.assert_near(
        paper_funnel_standardized_residuals(spec, row),
        x[tf.newaxis, :] * tf.exp(-y),
        atol=1.0e-12,
    )


def test_funnel_exact_sampler_is_deterministic_and_conditionally_standard_normal() -> None:
    spec = make_paper_funnel_spec()
    first = sample_paper_d100_exact(spec, 32768, seed=(20260813, 51002))
    second = sample_paper_d100_exact(spec, 32768, seed=(20260813, 51002))
    tf.debugging.assert_equal(first, second)
    residual = paper_funnel_standardized_residuals(spec, first)
    y = first[:, 0]
    assert first.shape == (32768, PAPER_D100_DIMENSION)
    assert first.dtype == residual.dtype == tf.float64
    assert abs(float(tf.reduce_mean(y).numpy())) < 0.02
    assert abs(float(tf.reduce_mean(tf.square(y)).numpy()) - 1.0) < 0.03
    assert abs(float(tf.reduce_mean(residual).numpy())) < 0.003
    assert abs(float(tf.reduce_mean(tf.square(residual)).numpy()) - 1.0) < 0.004


def test_committed_gaussian_source_contract_and_exact_whitening() -> None:
    spec = load_paper_gaussian_spec(SOURCE)
    assert spec.name == "paper_ill_cond_gaussian"
    assert spec.dimension == 100
    assert spec.constants_hash == (
        "e7421d0837176190bbd82479fb099adc321ce6ca262efccf6911ec1f59c33875"
    )
    rows = sample_paper_d100_exact(spec, 32768, seed=(20260813, 51003))
    centered = rows - tf.constant(spec.mean, tf.float64)[tf.newaxis, :]
    whitened = tf.transpose(
        tf.linalg.triangular_solve(
            tf.constant(spec.cholesky, tf.float64), tf.transpose(centered), lower=True
        )
    )
    assert abs(float(tf.reduce_mean(whitened).numpy())) < 0.003
    assert abs(float(tf.reduce_mean(tf.square(whitened)).numpy()) - 1.0) < 0.004


@pytest.mark.parametrize("target_name", ("funnel", "gaussian"))
def test_adapter_batch_native_xla_program(target_name: str) -> None:
    spec = (
        make_paper_funnel_spec()
        if target_name == "funnel"
        else load_paper_gaussian_spec(SOURCE)
    )
    adapter = PaperD100ValueScoreAdapter(spec)

    @tf.function(jit_compile=True)
    def compiled(rows: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return adapter.log_prob_and_grad(rows)

    rows = sample_paper_d100_exact(spec, 8, seed=(20260813, 51004))
    value, score = compiled(rows)
    assert value.shape == (8,)
    assert score.shape == (8, 100)
    assert adapter.value_score_capability().is_accepted_full_chain_xla_diagnostic_authority


def test_loader_rejects_semantic_hash_tampering(tmp_path: Path) -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["constants"]["mean"][0] = 0.25
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PaperD100TargetError, match="semantic constants hash"):
        load_paper_gaussian_spec(tampered)


def test_invalid_shapes_seeds_counts_and_target_kind_fail_closed() -> None:
    funnel = make_paper_funnel_spec()
    with pytest.raises(ValueError, match="shape"):
        paper_d100_log_prob_batch(funnel, tf.zeros((2, 99), tf.float64))
    with pytest.raises(ValueError, match="exceed one"):
        sample_paper_d100_exact(funnel, 1, seed=(1, 2))
    with pytest.raises(ValueError, match="two integers"):
        sample_paper_d100_exact(funnel, 2, seed=(1, 2, 3))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only for the funnel"):
        paper_funnel_standardized_residuals(
            load_paper_gaussian_spec(SOURCE), tf.zeros((2, 100), tf.float64)
        )
    with pytest.raises(ValueError, match="frozen at dimension 100"):
        PaperD100TargetSpec(name="paper_funnel", dimension=10)


def test_candidate_module_has_no_numpy_or_source_repository_import() -> None:
    source = (ROOT / "bayesfilter/inference/neutra_paper_d100_target.py").read_text(
        encoding="utf-8"
    )
    assert "import numpy" not in source
    assert "from dsge_hmc" not in source
    assert "import dsge_hmc" not in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.while_loop" not in source
