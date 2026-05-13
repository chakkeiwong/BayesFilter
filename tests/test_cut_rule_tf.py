from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear.cut_tf import tf_cut4g_sigma_point_rule


ROOT = Path(__file__).resolve().parents[1]


def _expectation(rule, values: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(values * rule.mean_weights)


def test_cut4g_rule_reproduces_documented_moment_identities() -> None:
    rule = tf_cut4g_sigma_point_rule(3)
    z = rule.offsets

    np.testing.assert_allclose(tf.reduce_sum(rule.mean_weights).numpy(), 1.0, atol=1e-14)
    np.testing.assert_allclose(
        tf.linalg.matvec(tf.transpose(z), rule.mean_weights).numpy(),
        np.zeros(3),
        atol=1e-14,
    )
    for i in range(3):
        np.testing.assert_allclose(_expectation(rule, z[:, i] ** 2).numpy(), 1.0, atol=1e-14)
        np.testing.assert_allclose(_expectation(rule, z[:, i] ** 4).numpy(), 3.0, atol=1e-14)
    np.testing.assert_allclose(
        _expectation(rule, z[:, 0] ** 2 * z[:, 1] ** 2).numpy(),
        1.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(_expectation(rule, z[:, 0] ** 5).numpy(), 0.0, atol=1e-14)
    assert rule.point_count == 14
    assert rule.polynomial_degree == 5
    assert np.all(rule.mean_weights.numpy() > 0.0)


def test_cut4g_rejects_too_small_dimension() -> None:
    with pytest.raises(ValueError, match="dim >= 3"):
        tf_cut4g_sigma_point_rule(2)


def test_cut_modules_do_not_import_numpy_or_call_dot_numpy() -> None:
    for relative in (
        ("bayesfilter", "nonlinear", "cut_tf.py"),
        ("bayesfilter", "nonlinear", "svd_cut_tf.py"),
    ):
        text = (ROOT.joinpath(*relative)).read_text(encoding="utf-8")
        assert "import numpy" not in text
        assert "from numpy" not in text
        assert ".numpy(" not in text
