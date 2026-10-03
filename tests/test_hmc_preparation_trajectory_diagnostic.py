"""CPU diagnostic checks, with an explicitly injected reporting failure."""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference import hmc_warmup
from scripts.hmc_preparation_trajectory_diagnostic import install
from tests.test_hmc_warmup import _GaussianAdapter, _transform


def test_trajectory_diagnostic_preserves_typed_traces_and_replays_endpoint(tmp_path, monkeypatch):
    originals = (tfp.mcmc.sample_chain, hmc_warmup.reviewed_value_score_target_fn,
                 hmc_warmup._validate_operational_window_trace)
    # Have pytest restore the diagnostic's process-local instrumentation.
    monkeypatch.setattr(tfp.mcmc, "sample_chain", originals[0])
    monkeypatch.setattr(hmc_warmup, "reviewed_value_score_target_fn", originals[1])
    monkeypatch.setattr(hmc_warmup, "_validate_operational_window_trace", originals[2])
    install(tmp_path / "diagnostic")
    adapter = hmc_warmup._AffineWarmupAdapter(base_adapter=_GaussianAdapter(tf.eye(2, dtype=tf.float64)),
        transform=_transform(tf.eye(2, dtype=tf.float64)), target_scope="hmc_warmup_gaussian")
    target = hmc_warmup.reviewed_value_score_target_fn(adapter)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(target, step_size=tf.constant(.1, tf.float64), num_leapfrog_steps=3),
        num_adaptation_steps=0)
    state = tf.constant([.2, -.1], tf.float64)

    def trace_fn(state, results):
        inner = results.inner_results
        return dict(is_accepted=inner.is_accepted, log_accept_ratio=inner.log_accept_ratio,
                    target_log_prob=inner.accepted_results.target_log_prob,
                    step_size=results.new_step_size, proposed_step_size=results.new_step_size,
                    consumed_step_size=inner.accepted_results.step_size)

    result = tfp.mcmc.sample_chain(num_results=4, current_state=state, kernel=kernel,
        previous_kernel_results=kernel.bootstrap_results(state), trace_fn=trace_fn,
        return_final_kernel_results=True, seed=(20260920, 61))
    trace = dict(result.trace)
    trace["log_accept_ratio"] = tf.tensor_scatter_nd_update(trace["log_accept_ratio"], [[1]],
                                                         tf.constant([-float("inf")], tf.float64))
    with pytest.raises(ValueError, match="log_accept_ratio: count=1, first_index=1"):
        hmc_warmup._validate_operational_window_trace(trace=trace, expected_draw_count=4,
            step_size_upper_bound=1., target_status_trace_policy="none")
    root = tmp_path / "diagnostic/window-00"
    assert json.loads((root / "is_accepted.tensor.json").read_text())["dtype"] == "bool"
    assert json.loads((root / "proposal-seed.tensor.json").read_text())["dtype"] == "int32"
    report = json.loads((root / "transition-0001/trajectory.json").read_text())
    assert report["first_nonfinite"] is None
    assert report["affine_layers"][0]["transform"]["factor"] == [[1., 0.], [0., 1.]]
    assert all(item["finite_mask_equal"] and item["max_absolute_difference"] < 1e-12
               for item in report["endpoint_comparison"].values())
    assert report["candidate_authority"] is False
