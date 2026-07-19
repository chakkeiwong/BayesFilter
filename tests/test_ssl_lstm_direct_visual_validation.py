import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


PATH = Path("docs/benchmarks/run_ssl_lstm_direct_visual_validation_2026_07_18.py")
SPEC = importlib.util.spec_from_file_location("ssl_lstm_visual", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_orient_pilot_prefix_preserves_chain_axis() -> None:
    raw = tf.reshape(
        tf.range(256 * 4 * 4, dtype=tf.float64),
        [256, 4, 4],
    )
    result = MODULE.orient_pilot_prefix(raw).numpy()
    assert result.shape == (4, 64, 4)
    np.testing.assert_array_equal(result[2, 7], raw.numpy()[7, 2])


def test_orient_pilot_prefix_rejects_transposed_input() -> None:
    with pytest.raises(MODULE.VisualValidationError, match="shape/orientation"):
        MODULE.orient_pilot_prefix(tf.zeros([4, 256, 4], tf.float64))


def _paths(offset: float) -> np.ndarray:
    base = np.arange(4 * 64 * 2 * 10, dtype=np.float64).reshape(4, 64, 2, 10)
    return 0.001 * base + offset


def test_summarize_paths_and_moment_diagnostic_are_finite() -> None:
    left = _paths(0.1)
    right = _paths(0.0) * 1.01
    summary = MODULE.summarize_paths(left)
    diagnostic = MODULE.moment_difference_diagnostic(left, right)
    assert summary["path_count"] == 512
    assert np.asarray(summary["chain_mean"]).shape == (4, 10)
    assert np.asarray(diagnostic["difference"]).shape == (20,)
    assert diagnostic["block_count_per_arm"] == 16
    assert np.isfinite(np.asarray(diagnostic["lower"])).all()
    assert np.all(np.asarray(diagnostic["upper"]) >= np.asarray(diagnostic["lower"]))


def test_moment_diagnostic_rejects_zero_variance() -> None:
    constant = np.ones([4, 64, 2, 10], dtype=np.float64)
    with pytest.raises(MODULE.VisualValidationError, match="non-positive"):
        MODULE.moment_difference_diagnostic(constant, constant)


def test_visual_seed_domains_are_disjoint() -> None:
    assert len(set(MODULE.SEEDS.values())) == 2
    assert set(MODULE.SEEDS.values()).isdisjoint({(20260718, 4101), (20260718, 4201), (20260718, 4301)})
