"""Independent reference checks for the training campaign's artifact/IS boundary."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import tensorflow as tf


spec = importlib.util.spec_from_file_location(
    "fab_campaign", Path(__file__).resolve().parents[1] / "docs/benchmarks/run_fab_iaf_training_campaign_2026_09_26.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_artifact_tensor_scalar_and_array_round_trip(tmp_path):
    path = tmp_path / "progress.json"
    campaign.save(path, {"loss": tf.constant(1.25, tf.float32),
                         "weights": tf.constant([0.25, 0.75], tf.float64),
                         "invalid": tf.constant(float("inf"), tf.float32)})
    assert json.loads(path.read_text()) == {"loss": 1.25, "weights": [0.25, 0.75], "invalid": None}


def test_independent_bank_weights_use_sampling_density():
    # Target p=N(0,1), bank g=N(2,1). A learned q=p would give p/q=1,
    # which cannot correct these bank draws. Correct log(p/g)=2-2*x.
    x = tf.constant([[-1.0], [0.0], [2.0], [3.0]], tf.float64)
    log_p = campaign.diagonal_gaussian_log_prob(x, [0.], [1.])
    log_g = campaign.diagonal_gaussian_log_prob(x, [2.], [1.])
    assert (log_p - log_g).numpy().tolist() == pytest.approx([4., 2., -2., -4.])
    # Check scale normalization against a scalar reference, not just ratios.
    actual = campaign.diagonal_gaussian_log_prob(tf.constant([[2., -1.]], tf.float64), [1., 1.], [2., 3.])
    expected = -math.log(2 * math.pi * 6) - .5 * ((1 / 2)**2 + (-2 / 3)**2)
    assert float(actual[0]) == pytest.approx(expected, abs=1e-12)


def test_reference_cache_preserves_diagnostic_and_rejects_different_bank(tmp_path):
    class NormalFlow:
        def forward_and_logdet(self, z):
            return z, tf.zeros(tf.shape(z)[0], tf.float64)

        def log_prob(self, x):
            return campaign.diagonal_gaussian_log_prob(x, [0.] * 4, [1.] * 4)

    class Bridge:
        parameter_dim = 4
        signature = "a" * 64

        def value_score_status(self, x, beta):
            del beta
            return NormalFlow().log_prob(x), -x, {"bridge_valid": tf.ones(tf.shape(x)[0], tf.bool)}

    flow, bridge = NormalFlow(), Bridge()
    kwargs = dict(rows=32, seed=(12, 34), reference_cache=tmp_path / "bank.json")
    first = campaign.coverage_diagnostic(flow, flow, bridge, [0.] * 4, [1.] * 4, None, **kwargs)
    cached = campaign.coverage_diagnostic(flow, flow, bridge, [0.] * 4, [1.] * 4, None, **kwargs)
    assert not first["reference_target_cache_reused"] and cached["reference_target_cache_reused"]
    assert first["map_base"] == cached["map_base"]
    assert first["independent_prior_bank"] == cached["independent_prior_bank"]
    assert first["map_base"]["ess"] == pytest.approx(32.)
    with pytest.raises(ValueError, match="identity"):
        campaign.coverage_diagnostic(flow, flow, bridge, [1.] * 4, [1.] * 4, None, **kwargs)
