from __future__ import annotations

import ast
import inspect
import math
import subprocess
import sys
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.independent_score import anchored_orthogonal_ratio_score_tf as anchored


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "docs/benchmarks/run_sir_anchored_orthogonal_ratio_score_20260814.py"


def _dataset(count: int = 16):
    observations, deltas, labels = [], [], []
    for index, delta in enumerate(anchored.DELTAS):
        minus = tf.random.stateless_normal([count, 2, 1], [100 + index, 1], dtype=tf.float32) - delta
        plus = tf.random.stateless_normal([count, 2, 1], [100 + index, 2], dtype=tf.float32) + delta
        observations.extend((minus, plus))
        deltas.extend((tf.fill([count], delta), tf.fill([count], delta)))
        labels.extend((tf.zeros([count]), tf.ones([count])))
    return tf.concat(observations, 0), tf.concat(deltas, 0), tf.concat(labels, 0)


def test_alpha_and_discrete_orthogonality_are_from_declared_grid() -> None:
    radius = tf.constant([d / anchored.DELTA_SCALE for d in anchored.DELTAS], tf.float64)
    alpha = anchored.basis_alpha()
    phi = radius**3 - alpha * radius**5
    tf.debugging.assert_near(tf.reduce_sum(radius * phi), 0.0, atol=1e-13)
    assert math.isclose(alpha, sum(float(r) ** 4 for r in radius.numpy()) / sum(float(r) ** 6 for r in radius.numpy()))


def test_anchored_basis_derivatives_and_condition_number() -> None:
    h = 1.0e-6
    values = anchored.anchored_basis(tf.constant([-h, h], tf.float64), delta_scale=1.0)
    derivatives = (values[1] - values[0]) / (2.0 * h)
    tf.debugging.assert_near(derivatives, tf.constant([1.0, 0.0], derivatives.dtype), atol=1.0e-6, rtol=0.0)
    diagnostics = anchored.basis_diagnostics()
    assert abs(diagnostics["inner_product"]) < 1e-12
    assert math.isfinite(diagnostics["condition_number"])
    assert diagnostics["condition_number"] < 20.0


def test_score_conversion_uses_only_anchored_linear_coefficient() -> None:
    source = inspect.getsource(anchored.AnchoredFit.score_at_observation)
    assert "c0" in source
    assert "2.0 * self.delta_scale" in source
    assert "c1" not in source


def test_synthetic_anchored_logit_recovery_mechanics() -> None:
    observations, deltas, labels = _dataset(16)
    fit = anchored.fit_anchored_classifier(
        observations, deltas, labels,
        validation_observations=observations, validation_deltas=deltas, validation_labels=labels,
        calibration_observations=observations, calibration_deltas=deltas, calibration_labels=labels,
        test_observations=observations, test_deltas=deltas, test_labels=labels,
        architecture="anchored_linear_quadratic", seed=7, epochs=2, minimum_epochs=1,
        patience=1, batch_size=192, jit_compile=False,
    )
    assert bool(fit.finite.numpy())
    assert float(fit.calibration_temperature.numpy()) > 0.0
    assert fit.score_at_observation(tf.zeros([1, 2, 1])).shape == (1,)


def test_conditional_balance_and_undeclared_delta_are_rejected() -> None:
    observations, deltas, labels = _dataset(4)
    anchored.validate_conditional_dataset(observations, deltas, labels)
    with pytest.raises(ValueError, match="balance"):
        anchored.validate_conditional_dataset(observations[:-1], deltas[:-1], labels[:-1])
    with pytest.raises(ValueError, match="undeclared"):
        anchored.validate_conditional_dataset(observations, tf.tensor_scatter_nd_update(deltas, [[0]], [0.007]), labels)


def test_fresh_process_has_no_forbidden_runtime_modules() -> None:
    command = (
        "import sys; import bayesfilter.independent_score.anchored_orthogonal_ratio_score_tf; "
        "bad=[n for n in sys.modules if n.startswith('bayesfilter.') and any(t in n.lower().split('.') "
        "for t in ('highdim','filtering','filters','particle','particles','smoothing','simulation_score_tf'))]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", command], cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": "-1"}, check=True, capture_output=True, text=True)


def test_selection_and_final_domains_are_separate_and_sir_is_gated() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"selection_domain":50' in source
    assert '"final_domain":60' in source
    assert "SIR requires a PASSED full exact oracle" in source
