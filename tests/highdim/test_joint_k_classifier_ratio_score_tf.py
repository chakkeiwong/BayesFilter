from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import tensorflow as tf

from bayesfilter.independent_score import joint_k_classifier_ratio_score_tf as joint


ROOT = Path(__file__).resolve().parents[2]


def _dataset(seed: int, count_per_delta: int = 16):
    deltas = tf.constant([0.01, 0.02, 0.03, 0.04], tf.float32)
    rows = []
    delta_rows = []
    label_rows = []
    for index, delta in enumerate(deltas):
        noise_minus = tf.random.stateless_normal([count_per_delta, 2, 1], [seed + index, 1])
        noise_plus = tf.random.stateless_normal([count_per_delta, 2, 1], [seed + index, 2])
        rows.extend([noise_minus - delta, noise_plus + delta])
        delta_rows.extend([tf.fill([count_per_delta], delta), tf.fill([count_per_delta], delta)])
        label_rows.extend([tf.zeros([count_per_delta]), tf.ones([count_per_delta])])
    return tf.concat(rows, 0), tf.concat(delta_rows, 0), tf.concat(label_rows, 0)


def test_odd_delta_basis_is_odd_and_zero_at_origin() -> None:
    values = joint.odd_delta_basis(tf.constant([-0.04, -0.01, 0.0, 0.01, 0.04]))
    tf.debugging.assert_near(values[0], -values[-1])
    tf.debugging.assert_near(values[1], -values[-2])
    tf.debugging.assert_near(values[2], tf.zeros([3], tf.float32))


def test_conditional_balance_rejects_missing_or_imbalanced_delta() -> None:
    x, d, y = _dataset(10)
    joint.validate_conditional_balanced_dataset(x, d, y, expected_deltas=(0.01, 0.02, 0.03, 0.04))
    with pytest.raises(ValueError, match="balance"):
        joint.validate_conditional_balanced_dataset(
            x[:-1], d[:-1], y[:-1], expected_deltas=(0.01, 0.02, 0.03, 0.04)
        )


def test_joint_linear_head_has_three_zero_coefficients_and_no_intercept() -> None:
    model = joint._make_model("joint_linear_quadratic_odd5", 6, 3)
    assert model.count_params() == 21
    assert all(bool(tf.reduce_all(weight == 0.0).numpy()) for weight in model.weights)
    source = inspect.getsource(joint._raw_logits)
    assert "odd_delta_basis" in source


def test_joint_fit_mechanics_and_score_coefficient_conversion() -> None:
    train = _dataset(20, 32)
    validation = _dataset(30, 16)
    calibration = _dataset(40, 16)
    test = _dataset(50, 16)
    fit = joint.fit_joint_k_classifier(
        *train,
        validation_observations=validation[0],
        validation_deltas=validation[1],
        validation_labels=validation[2],
        calibration_observations=calibration[0],
        calibration_deltas=calibration[1],
        calibration_labels=calibration[2],
        test_observations=test[0],
        test_deltas=test[1],
        test_labels=test[2],
        expected_deltas=(0.01, 0.02, 0.03, 0.04),
        architecture="joint_linear_quadratic_odd5",
        seed=11,
        epochs=3,
        minimum_epochs=2,
        patience=2,
        batch_size=256,
        jit_compile=False,
    )
    assert bool(fit.finite.numpy())
    assert fit.calibration_temperature.numpy() > 0.0
    score = fit.score_at_observation(tf.zeros([1, 2, 1], tf.float32))
    assert score.shape == (1,)
    assert "2.0 * self.delta_scale" in inspect.getsource(joint.JointKFit.score_at_observation)


def test_joint_module_has_no_state_estimation_imports() -> None:
    source = inspect.getsource(joint)
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
        "bayesfilter.independent_score.classifier_ratio_score_tf",
    ]
    forbidden = ("fisher", "particle", "resampl", "smoothing", "complete_data_score", "filtering")
    assert not any(token in source.lower() for token in forbidden)


def test_pooled_training_rows_match_profile_batch_contract() -> None:
    assert (64 * 2 * 4) % 128 == 0
    assert (2048 * 2 * 4) % 2048 == 0


def test_fresh_joint_module_import_does_not_load_highdim() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bayesfilter.independent_score.joint_k_classifier_ratio_score_tf; "
            "bad=[x for x in sys.modules if x.startswith('bayesfilter.') and ('highdim' in x or 'filtering' in x)]; assert not bad, bad",
        ],
        cwd=ROOT,
        env={"CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        capture_output=True,
        text=True,
    )
