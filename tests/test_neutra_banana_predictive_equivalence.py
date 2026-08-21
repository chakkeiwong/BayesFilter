from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_neutra_banana_predictive_equivalence_2026_08_16.py"


def _module():
    spec = importlib.util.spec_from_file_location("banana_predictive_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_banana_map_preserves_exact_latent_coordinates() -> None:
    runner = _module()
    latent = tf.reshape(tf.cast(tf.range(2 * 8 * 16), tf.float64), [2, 8, 16]) / 10.0
    observed = runner._banana(tf, latent)
    recovered = tf.concat(
        (
            observed[:, :, :1],
            observed[:, :, 1:2]
            - tf.constant(runner.CURVATURE, tf.float64)
            * (tf.square(observed[:, :, 0:1]) - 1.0),
            observed[:, :, 2:],
        ),
        axis=-1,
    )
    tf.debugging.assert_near(recovered, latent, atol=1e-14)


def test_block_mmd_is_zero_for_identical_arm() -> None:
    runner = _module()
    samples = tf.random.stateless_normal([4, 128, 16], seed=[1, 2], dtype=tf.float64)
    result = runner._bootstrap_mmd(
        tf,
        samples,
        samples,
        block_length=32,
        bootstrap_count=8,
        seed=(3, 4),
    )
    assert float(result["point"]) == pytest.approx(0.0, abs=1e-13)
    assert float(result["upper_99"]) >= 0.0


def test_block_mmd_detects_a_large_location_shift() -> None:
    runner = _module()
    left = tf.random.stateless_normal([4, 128, 16], seed=[5, 6], dtype=tf.float64)
    right = left + tf.constant(3.0, tf.float64)
    result = runner._bootstrap_mmd(
        tf,
        left,
        right,
        block_length=32,
        bootstrap_count=8,
        seed=(7, 8),
    )
    assert float(result["point"]) > 0.1


def test_calibration_envelope_reports_both_empirical_quantiles() -> None:
    runner = _module()
    values = tf.constant([0.1, 0.2, 0.3, 0.4], tf.float64)
    assert float(runner._empirical_quantile(tf, values, 0.95)) == pytest.approx(0.4)
    assert float(runner._empirical_quantile(tf, values, 0.99)) == pytest.approx(0.4)


def test_candidate_loader_rejects_missing_or_stale_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _module()
    monkeypatch.setattr(runner, "CANDIDATE_RESULT", Path("/tmp/does-not-exist-result.json"))
    monkeypatch.setattr(runner, "CANDIDATE_ARCHIVE", Path("/tmp/does-not-exist-archive.tftensor"))
    with pytest.raises(FileNotFoundError):
        runner._load_candidate(tf, draw_count=128, offset=0)
