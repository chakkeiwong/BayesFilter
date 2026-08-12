from __future__ import annotations

import importlib.util
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_q20_seed_b_posterior_reference_2026_08_07.py"


def _load():
    spec = importlib.util.spec_from_file_location("q20_seed_b_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gauss_hermite_weights_integrate_standard_normal_moments() -> None:
    module = _load()
    nodes, weights = module._gh_nodes_weights(tf, 9)
    tf.debugging.assert_near(tf.reduce_sum(weights), tf.constant(1.0, tf.float64), atol=1e-14)
    tf.debugging.assert_near(tf.reduce_sum(weights * nodes), tf.constant(0.0, tf.float64), atol=1e-14)
    tf.debugging.assert_near(tf.reduce_sum(weights * tf.square(nodes)), tf.constant(1.0, tf.float64), atol=1e-13)


def test_importance_correction_recovers_off_center_gaussian() -> None:
    module = _load()
    proposal_center = tf.constant([0.0, 0.0, 0.0, 0.0], tf.float64)
    factor = tf.eye(4, dtype=tf.float64)
    points, gh_weights, log_proposal = module._mesh_quadrature(
        tf, proposal_center, factor, 9, 1.5
    )
    target_center = tf.constant([0.25, -0.1, 0.2, 0.05], tf.float64)
    log_target = (
        -0.5 * tf.reduce_sum(tf.square(points - target_center), axis=1)
        - 2.0 * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))
    )
    stats = module._stats(tf, points, tf.math.log(gh_weights) + log_target - log_proposal)
    tf.debugging.assert_near(tf.constant(stats["mean"], tf.float64), target_center, atol=2e-3)
    tf.debugging.assert_near(tf.constant(stats["sd"], tf.float64), tf.ones([4], tf.float64), atol=3e-3)


def test_stability_uses_scale_free_metrics() -> None:
    module = _load()
    base = {
        "mean": [0.0, 0.0],
        "sd": [2.0, 4.0],
        "covariance": [[4.0, 0.0], [0.0, 16.0]],
        "quantiles": [[-2.0, -4.0], [0.0, 0.0], [2.0, 4.0]],
    }
    same = module._stability(tf, base, base)
    assert same == {"mean": 0.0, "sd": 0.0, "covariance": 0.0, "quantile": 0.0}
    assert module._stability_passed(same)


def test_source_keeps_reference_and_retained_values_phase_separated() -> None:
    source = SCRIPT.read_text(encoding="ascii")
    reference_body = source.split("def _reference", 1)[1].split("def _compare", 1)[0]
    assert "ARCHIVE_MANIFEST" not in reference_body
    assert "parse_tensor_values=True" not in reference_body
    compare_body = source.split("def _compare", 1)[1].split("def _unweighted_stats", 1)[0]
    assert "parse_tensor_values=True" in compare_body
    assert "reference_hash" in compare_body

