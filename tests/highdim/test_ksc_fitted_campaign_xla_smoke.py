"""Intentionally CPU-only XLA compatibility smoke, not research evidence."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import importlib.util
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.score_study.ksc_fitted_guides_diagnostic_tf import (
    make_ksc_kernel_fit, make_ksc_quadratic_fit)


@pytest.mark.parametrize("family", ["quadratic", "kernel"])
def test_xla_fits_and_loss_are_finite(family):
    path = Path(__file__).resolve().parents[2] / "scripts/run_ledh_ksc_fitted_guides.py"
    specification = importlib.util.spec_from_file_location("ksc_campaign_smoke", path)
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    theta = tf.constant([1.5, 0.0], tf.float64)
    observations = tf.constant([0.2, -0.4], tf.float64)
    config = {"count": 64, "passes": 2, "family": family, "transition_replicates": 16}
    builder = (make_ksc_quadratic_fit(theta, 64, 2) if family == "quadratic"
               else make_ksc_kernel_fit(theta, 64, 16, 1.0))
    fitted = runner.tensor_fit(tf, builder, config, observations, 691701)
    assert bool(fitted["density_eligible"])
    nodes = tf.linspace(tf.constant(-4.0, tf.float64), 4.0, 256)
    weights = tf.ones([256], tf.float64) / 256.0
    loss = runner.make_loss_program(tf, family, 64)
    result = loss(nodes, weights, -0.5 * tf.square(nodes), -nodes,
                  *runner.fit_parameters(family, fitted))
    assert all(bool(tf.math.is_finite(value)) for value in result.values())
    assert loss.experimental_get_tracing_count() == 1
