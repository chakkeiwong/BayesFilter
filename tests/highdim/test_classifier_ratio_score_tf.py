from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import tensorflow as tf

from bayesfilter.independent_score import classifier_ratio_score_tf as ratio
from bayesfilter.independent_score import gaussian_observation_simulator_tf as gaussian
from bayesfilter.independent_score.sir_observation_simulator_tf import (
    fixed_observed_path,
    simulate_observation_paths_from_noise,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "docs/benchmarks/run_sir_classifier_ratio_score_20260813.py"


def test_balanced_dataset_contract_rejects_imbalance() -> None:
    observations = tf.zeros([4, 2, 1], tf.float32)
    ratio.validate_balanced_observation_dataset(
        observations, tf.constant([0, 0, 1, 1], tf.int32)
    )
    with pytest.raises(ValueError, match="balanced"):
        ratio.validate_balanced_observation_dataset(
            observations, tf.constant([0, 0, 0, 1], tf.int32)
        )


def test_score_is_exactly_calibrated_logit_over_two_epsilon() -> None:
    observed = ratio.central_score_from_calibrated_logit(
        tf.constant([0.4, -0.2], tf.float64), 0.1
    )
    tf.debugging.assert_near(
        observed, tf.constant([2.0, -1.0], tf.float64), atol=5e-8
    )
    source = textwrap.dedent(
        inspect.getsource(ratio.central_score_from_calibrated_logit)
    )
    assert "2.0 * epsilon_value" in source


def test_classifier_module_has_no_inference_or_analytical_score_dependency() -> None:
    source = inspect.getsource(ratio)
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == [
        "__future__",
        "dataclasses",
        "math",
        "collections.abc",
        "tensorflow",
    ]
    forbidden = (
        "fisher_identity_simulation_score",
        "complete_data_score",
        "transition_log_density_parameter_score",
        "observation_log_density_parameter_score",
        "resampling",
        "smoothing",
        "particle",
    )
    lowered = source.lower()
    assert not any(token in lowered for token in forbidden)


def test_lightweight_runtime_import_does_not_load_state_estimation_modules() -> None:
    command = (
        "import sys; "
        "import bayesfilter.independent_score.classifier_ratio_score_tf; "
        "import bayesfilter.independent_score.sir_observation_simulator_tf; "
        "bad=[n for n in sys.modules if n.startswith('bayesfilter.') and "
        "('highdim' in n.split('.') or set(n.split('.')) & "
        "{'filters','filtering','particle','particles','smoothing'})]; "
        "assert not bad, bad"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        env={"CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
    )


def test_runner_source_uses_only_independent_observation_simulators() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    bayesfilter_imports = [name for name in imported if name.startswith("bayesfilter")]
    assert bayesfilter_imports == [
        "bayesfilter.runtime.gpu_memory_policy",
        "bayesfilter.independent_score.classifier_ratio_score_tf",
        "bayesfilter.independent_score",
        "bayesfilter.independent_score",
    ]
    assert "selection_data_domain\": 10" in source
    assert "final_data_domain\": 20" in source
    assert "calibrated_logit_at_fixed_observation/(2*epsilon)" in source


def test_standalone_sir_simulator_matches_source_generative_steps() -> None:
    from bayesfilter.highdim.models import parameterized_zhao_cui_sir_austria_model

    model = parameterized_zhao_cui_sir_austria_model()
    theta = tf.constant([0.03, -0.02, 0.04], tf.float64)
    initial_noise = tf.reshape(
        tf.linspace(tf.constant(-0.2, tf.float64), tf.constant(0.2, tf.float64), 36),
        [2, 18],
    )
    transition_noise = tf.reshape(
        tf.linspace(tf.constant(-0.1, tf.float64), tf.constant(0.1, tf.float64), 72),
        [2, 2, 18],
    )
    observation_noise = tf.reshape(
        tf.linspace(tf.constant(-0.15, tf.float64), tf.constant(0.15, tf.float64), 36),
        [2, 2, 9],
    )
    standalone = simulate_observation_paths_from_noise(
        theta, initial_noise, transition_noise, observation_noise
    )
    scaled = model.scaled_model(theta)
    state = scaled.initial_mean[None, :] + initial_noise
    expected = []
    for time_index in range(2):
        latent = model.transition_mean(theta, state) + transition_noise[:, time_index, :]
        state = tf.reshape(
            tf.stack([tf.maximum(latent[:, 0::2], 0.0), latent[:, 1::2]], axis=2),
            tf.shape(latent),
        )
        expected.append(
            state[:, 1::2]
            + 10.0 * tf.exp(theta[2]) * observation_noise[:, time_index, :]
        )
    tf.debugging.assert_near(standalone, tf.stack(expected, axis=1), atol=2e-12)


def test_fixed_observed_path_matches_source_seed_and_time_slice() -> None:
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    _, source_observations = zhao_cui_sir_austria_model().simulate(
        final_time=3, seed=81120
    )
    tf.debugging.assert_equal(fixed_observed_path(81120, 3), source_observations[1:4])


def test_sir_full_generation_has_exact_paired_prefixes() -> None:
    initial_noise = tf.random.stateless_normal([3, 18], [10, 1], dtype=tf.float64)
    transition_noise = tf.random.stateless_normal([3, 50, 18], [10, 2], dtype=tf.float64)
    observation_noise = tf.random.stateless_normal([3, 50, 9], [10, 3], dtype=tf.float64)
    full = simulate_observation_paths_from_noise(
        tf.zeros([3], tf.float64), initial_noise, transition_noise, observation_noise
    )
    prefix = simulate_observation_paths_from_noise(
        tf.zeros([3], tf.float64),
        initial_noise,
        transition_noise[:, :20, :],
        observation_noise[:, :20, :],
    )
    tf.debugging.assert_equal(full[:, :20, :], prefix)


def test_gaussian_oracle_score_matches_density_central_difference() -> None:
    theta = tf.constant([0.03, -0.02, 0.04], tf.float64)
    observations = gaussian.fixed_observed_path(20)
    exact = gaussian.exact_score(theta, observations)

    def log_density(parameters: tf.Tensor) -> tf.Tensor:
        zeros = tf.zeros([1, 20, 9], tf.float64)
        mean = gaussian.simulate_observation_paths_from_noise(parameters, zeros)[0]
        scale = tf.exp(parameters[2])
        residual = (observations - mean) / scale
        return -tf.cast(tf.size(observations), tf.float64) * tf.math.log(scale) - 0.5 * tf.reduce_sum(tf.square(residual))

    epsilon = tf.constant(1.0e-5, tf.float64)
    finite_differences = []
    for coordinate in range(3):
        direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
        finite_differences.append(
            (log_density(theta + epsilon * direction) - log_density(theta - epsilon * direction))
            / (2.0 * epsilon)
        )
    tf.debugging.assert_near(exact, tf.stack(finite_differences), atol=2.0e-7)


def test_epsilon_squared_extrapolation_and_admission() -> None:
    rows = {
        0.01: [2.0002, 2.0003, 2.0001],
        0.02: [2.0008, 2.0009, 2.0007],
        0.04: [2.0032, 2.0033, 2.0031],
        0.08: [2.0128, 2.0129, 2.0127],
    }
    result = ratio.epsilon_squared_extrapolation(rows)
    assert result["reference_admitted"] is True
    assert math.isclose(float(result["intercept"]), 2.0, abs_tol=2.0e-4)
    incomplete = ratio.epsilon_squared_extrapolation({0.01: rows[0.01], 0.02: rows[0.02]})
    assert incomplete["reference_admitted"] is False
    assert incomplete["status"] == "no_classifier_ratio_reference"


def test_quadratic_features_are_centered_as_frozen_plan_requires() -> None:
    train = tf.constant(
        [[[-1.0]], [[1.0]], [[-1.0]], [[1.0]]], dtype=tf.float32
    )
    center = tf.reduce_mean(train, axis=0)
    scale = tf.math.reduce_std(train, axis=0)
    features = ratio._standardize(
        train, center, scale, "mlp_full_path_quadratic"
    )
    tf.debugging.assert_near(tf.reduce_mean(features[:, 1]), 0.0, atol=1.0e-7)
    tf.debugging.assert_equal(features[:, 1], tf.zeros([4], tf.float32))
    quadratic_features = ratio._standardize(
        train, center, scale, "linear_full_path_quadratic"
    )
    tf.debugging.assert_equal(quadratic_features[:, 1], tf.zeros([4], tf.float32))


@pytest.mark.parametrize(
    "architecture", ["linear_full_path", "linear_full_path_quadratic"]
)
def test_convex_heads_use_zero_initialization(architecture: str) -> None:
    model = ratio._make_model(architecture, 4, 17)
    assert all(bool(tf.reduce_all(weight == 0.0).numpy()) for weight in model.weights)


def test_convex_quadratic_head_is_distinct_from_mlp() -> None:
    assert "linear_full_path_quadratic" in ratio.ARCHITECTURES
    assert ratio._make_model("linear_full_path_quadratic", 4, 17).count_params() == 5


def test_linear_ratio_classifier_recovers_gaussian_location_ratio() -> None:
    def data(count: int, seed: int):
        half = count // 2
        negative = tf.random.stateless_normal([half, 2, 1], [seed, 1]) - 0.2
        positive = tf.random.stateless_normal([half, 2, 1], [seed, 2]) + 0.2
        return (
            tf.concat([negative, positive], axis=0),
            tf.concat([tf.zeros([half]), tf.ones([half])], axis=0),
        )

    train, train_y = data(512, 10)
    validation, validation_y = data(128, 20)
    calibration, calibration_y = data(128, 30)
    test, test_y = data(256, 40)
    fit = ratio.fit_ratio_classifier(
        train,
        train_y,
        validation_observations=validation,
        validation_labels=validation_y,
        calibration_observations=calibration,
        calibration_labels=calibration_y,
        test_observations=test,
        test_labels=test_y,
        architecture="linear_full_path",
        seed=50,
        epochs=30,
        minimum_epochs=8,
        patience=6,
        batch_size=128,
        jit_compile=False,
    )
    assert bool(fit.finite.numpy())
    assert 0.55 < float(fit.test_auc.numpy()) < 0.8
    score = ratio.central_score_from_calibrated_logit(
        fit.calibrated_logit(tf.zeros([1, 2, 1], tf.float32)), 0.2
    )
    assert math.isfinite(float(score.numpy()[0]))
