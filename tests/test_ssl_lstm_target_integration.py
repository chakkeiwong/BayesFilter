from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.ssl_lstm_target_integration import (
    TargetIntegrationError,
    calibrate_horizon_scales,
    compare_path_and_conditional_moments,
    conditional_observation_moments,
)
from bayesfilter.nonlinear.ssl_lstm_predictive_tf import (
    SSLLSTMForecastConfig,
    forecast_ssl_lstm_paths,
    make_ssl_lstm_innovation_bank,
)
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    make_ssl_lstm_svd_ukf_components,
)


POINTS = tf.constant(
    [
        [0.35, -0.08, 0.65, 0.05],
        [0.3, -0.1, 0.6, 0.08],
    ],
    tf.float64,
)
RUNNER = (
    Path(__file__).resolve().parents[1]
    / "docs/benchmarks/run_ssl_lstm_neutra_target_integration_2026_07_18.py"
)


@pytest.fixture(scope="module")
def bundle():
    config = SSLLSTMForecastConfig()
    bank_a = make_ssl_lstm_innovation_bank(
        config, 2, tf.constant([20260718, 4101], tf.int32), "independent_arm", 1
    )
    bank_b = make_ssl_lstm_innovation_bank(
        config, 2, tf.constant([20260718, 4102], tf.int32), "independent_arm", 2
    )
    return config, (
        forecast_ssl_lstm_paths(POINTS, bank_a, config),
        forecast_ssl_lstm_paths(POINTS, bank_b, config),
    )


def test_conditional_variance_is_parameter_chart_observation_variance(bundle) -> None:
    config, paths = bundle
    result = conditional_observation_moments(paths[0], config)
    full = paths[0].terminal.full_parameters
    expected = []
    for row in tf.unstack(full):
        components = make_ssl_lstm_svd_ukf_components(
            row,
            config.posterior_config.static_config,
            evidence_path="docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md",
            std_floor=config.posterior_config.std_floor,
        )
        expected.append(tf.square(components.parameters.observation_std[0]))
    expected = tf.stack(expected)
    tf.debugging.assert_near(result.observation_stds, tf.sqrt(expected))
    tf.debugging.assert_near(
        result.variances,
        tf.broadcast_to(expected[:, tf.newaxis, tf.newaxis], [2, 2, 10]),
    )


def test_calibration_scales_are_finite_positive_and_replayable(bundle) -> None:
    config, paths = bundle
    roots = ((20260718, 4101), (20260718, 4102))
    first = calibrate_horizon_scales(paths, seed_roots=roots)
    second = calibrate_horizon_scales(paths, seed_roots=roots)
    tf.debugging.assert_near(first.center, second.center)
    tf.debugging.assert_near(first.scale, second.scale)
    assert first.calibration_signature == second.calibration_signature
    assert bool(tf.reduce_all(first.scale > 0.0))


def test_calibration_rejects_seed_count_or_nonpositive_scale(bundle) -> None:
    config, paths = bundle
    with pytest.raises(TargetIntegrationError, match="one seed root"):
        calibrate_horizon_scales(paths, seed_roots=((1, 2),))
    with pytest.raises(TargetIntegrationError, match="do not match forecast provenance"):
        calibrate_horizon_scales(paths, seed_roots=((9, 9), (8, 8)))
    constant = tuple(
        dataclasses.replace(
            paths_item,
            observations=tf.zeros_like(paths_item.observations),
            observation_means=tf.zeros_like(paths_item.observation_means),
            observation_innovations=tf.zeros_like(paths_item.observation_innovations),
        )
        for paths_item in paths
    )
    with pytest.raises(TargetIntegrationError, match="strictly positive"):
        calibrate_horizon_scales(
            constant,
            seed_roots=tuple(item.provenance.innovation_root_seed for item in paths),
        )


def test_path_and_conditional_feature_contract_is_finite_on_tiny_fixture(bundle) -> None:
    config, paths = bundle
    calibration = calibrate_horizon_scales(
        paths, seed_roots=((20260718, 4101), (20260718, 4102))
    )
    result = compare_path_and_conditional_moments(
        paths, calibration, config, jit_compile=False
    )
    assert type(result.paired_pass) is bool
    tf.debugging.assert_all_finite(result.paired_feature_difference, "paired difference")
    tf.debugging.assert_all_finite(result.paired_standard_error, "paired standard error")
    assert tuple(result.path.feature_estimate.shape) == (20,)
    assert tuple(result.conditional.feature_estimate.shape) == (20,)
    assert tuple(result.standardized_conditional_variances.shape) == (2, 2, 2, 10)


def test_runner_is_immutable_and_has_no_confirmation_input(tmp_path: Path) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "retained_samples" not in source
    assert "phase-7-retained-admission" not in source
    assert "never performs a G/H" in source
    spec = importlib.util.spec_from_file_location("ssl_lstm_target_integration_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.run(existing)
