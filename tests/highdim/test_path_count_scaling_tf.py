from __future__ import annotations

import tensorflow as tf
import pytest

from bayesfilter.independent_score.path_count_scaling_tf import (
    summarize_path_count_scaling,
)


def test_invalid_path_filter_is_pairwise_and_threshold_is_a_flag() -> None:
    import importlib.util
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "v7_path_count_runner",
        root / "docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py",
    )
    assert spec is not None and spec.loader is not None
    os.environ["BAYESFILTER_CPU_ONLY_SMOKE"] = "true"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    minus = tf.zeros([4, 2, 1], tf.float64)
    plus = tf.tensor_scatter_nd_update(minus, [[2, 0, 0]], [float("nan")])
    filtered, info = module._filter_paired_cases(
        [(minus, plus)],
        path_count=4,
        policy="remove_invalid_paths",
        threshold=1.0e-3,
    )
    assert info["invalid_path_count"] == 1
    assert info["threshold_flagged"]
    assert info["removal_applied"]
    assert info["removed_pair_count"] == 1
    assert filtered[0][0].shape[0] == filtered[0][1].shape[0] == 3

    preserved, info = module._filter_paired_cases(
        [(minus, plus)],
        path_count=4,
        policy="preserve",
        threshold=1.0e-3,
    )
    assert not info["removal_applied"]
    assert preserved[0][0].shape[0] == 4


def _ladder(amplitude: tuple[float, ...]) -> tuple[tf.Tensor, tf.Tensor]:
    bundles, paths, cells = 10, 24, 3
    path_effect = tf.random.stateless_normal([paths, cells], [511, 1], dtype=tf.float64)
    bundle_noise = tf.random.stateless_normal(
        [bundles, paths, cells], [511, 2], dtype=tf.float64
    )
    fixed_noise = tf.random.stateless_normal([bundles, cells], [511, 3], dtype=tf.float64)
    outputs = tf.stack(
        [path_effect[None, :, :] + value * bundle_noise for value in amplitude]
    )
    fixed = tf.stack([value * fixed_noise for value in amplitude])
    return outputs, fixed


def test_three_level_exact_one_over_n_scaling() -> None:
    outputs, fixed = _ladder((1.0, 2.0**-0.5, 0.5))
    summary = summarize_path_count_scaling(
        outputs,
        fixed,
        counts=(8192, 16384, 32768),
        bootstrap_replicates=200,
    )
    for endpoint in ("audit_adjacent_scaling", "fixed_adjacent_scaling"):
        for row in summary[endpoint]:
            assert row["variance_ratio"] == pytest.approx(0.5, rel=1.0e-10)
            assert row["normalized_1_over_n_efficiency"] == pytest.approx(1.0)
            assert row["scaling_exponent"] == pytest.approx(1.0)
            assert row["classification"] == "compatible_with_1_over_n"
    assert summary["audit_global_exponent"]["point"] == pytest.approx(1.0)
    assert summary["fixed_global_exponent"]["point"] == pytest.approx(1.0)


def test_plateau_is_not_called_variance_reduction() -> None:
    outputs, fixed = _ladder((1.0, 1.0))
    summary = summarize_path_count_scaling(
        outputs, fixed, counts=(8192, 16384), bootstrap_replicates=200
    )
    assert (
        summary["audit_adjacent_scaling"][0]["classification"]
        == "no_supported_variance_reduction"
    )
    assert summary["audit_adjacent_scaling"][0]["variance_ratio"] == pytest.approx(1.0)


def test_faster_than_one_over_n_is_distinguished() -> None:
    outputs, fixed = _ladder((1.0, 0.5))
    summary = summarize_path_count_scaling(
        outputs, fixed, counts=(8192, 16384), bootstrap_replicates=200
    )
    assert summary["audit_adjacent_scaling"][0]["classification"] == "faster_than_1_over_n"
    assert summary["audit_adjacent_scaling"][0]["variance_ratio"] == pytest.approx(0.25)
    assert summary["audit_adjacent_scaling"][0]["scaling_exponent"] == pytest.approx(2.0)


def test_gaussian_mse_scaling_is_reported() -> None:
    outputs, fixed = _ladder((1.0, 2.0**-0.5))
    exact = tf.zeros([24, 3], tf.float64)
    exact_fixed = tf.zeros([3], tf.float64)
    summary = summarize_path_count_scaling(
        outputs,
        fixed,
        counts=(8192, 16384),
        exact_scores=exact,
        exact_fixed_score=exact_fixed,
        bootstrap_replicates=200,
    )
    assert "exact_mse_adjacent_scaling" in summary
    assert "exact_fixed_mse_adjacent_scaling" in summary


def test_non_doubling_ladder_is_rejected() -> None:
    outputs, fixed = _ladder((1.0, 0.5))
    with pytest.raises(ValueError, match="adjacent doublings"):
        summarize_path_count_scaling(
            outputs, fixed, counts=(8192, 20000), bootstrap_replicates=100
        )
