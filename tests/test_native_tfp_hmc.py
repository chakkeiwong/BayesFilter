from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.native_tfp_hmc import (
    NativeTFPFixedKernelHMCConfig,
    run_native_tfp_fixed_kernel_hmc,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability


SCOPE = "native_tfp_gaussian_fixture"


class ReviewedGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "reviewed_gaussian_adapter_v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_fixture",
            target_scope=SCOPE,
            nonclaims=("test fixture",),
        )

    def log_prob_and_grad(self, theta):
        values = tf.cast(tf.convert_to_tensor(theta), tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values)), -values

    def target_status_telemetry(self, theta):
        values = tf.cast(tf.convert_to_tensor(theta), tf.float64)
        valid = tf.reduce_all(tf.math.is_finite(values))
        return {
            "status_code": tf.where(valid, 0, 1),
            "valid_pre_regularized_score": valid,
            "floor_count_value": tf.constant(0, tf.int32),
            "min_innovation_eigenvalue": tf.constant(0.0, tf.float64),
            "innovation_condition_estimate": tf.constant(0.0, tf.float64),
        }


def _config() -> NativeTFPFixedKernelHMCConfig:
    return NativeTFPFixedKernelHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260725, 8),
        target_scope=SCOPE,
    )


def test_native_runner_import_surface_is_tensorflow_tfp_only() -> None:
    import bayesfilter.inference.native_tfp_hmc as module

    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any(
        name == forbidden or name.startswith(forbidden + ".")
        for name in imports
        for forbidden in ("numpy", "scipy", "bayesfilter.inference.hmc")
    )
    source = path.read_text(encoding="utf-8")
    assert "input_signature=()" in source
    assert "tfm.HamiltonianMonteCarlo" in source
    assert "tfm.sample_chain" in source


def test_native_runner_matches_historical_runner_for_identical_seed() -> None:
    from bayesfilter.inference.hmc import FullChainHMCConfig, run_full_chain_tfp_hmc

    adapter = ReviewedGaussianAdapter()
    initial = tf.constant([0.1, -0.2], tf.float64)
    old = run_full_chain_tfp_hmc(
        adapter,
        initial,
        FullChainHMCConfig(
            num_results=4,
            num_burnin_steps=2,
            step_size=0.05,
            num_leapfrog_steps=2,
            seed=(20260725, 8),
            use_xla=False,
            trace_policy="standard",
            target_status_trace_policy="per_chain_step",
            adaptation_policy="fixed_kernel_no_adaptation",
            target_scope=SCOPE,
            chain_execution_mode="tf_function",
        ),
    )
    new = run_native_tfp_fixed_kernel_hmc(adapter, initial, _config())
    np.testing.assert_array_equal(new.samples.numpy(), old.samples.numpy())
    for name in (
        "is_accepted",
        "log_accept_ratio",
        "target_log_prob",
        "proposed_target_log_prob",
        "log_acceptance_correction",
    ):
        np.testing.assert_array_equal(new.trace[name].numpy(), old.trace[name].numpy())
    for name, value in new.trace["target_status_telemetry"].items():
        np.testing.assert_array_equal(
            value.numpy(), old.trace["target_status_telemetry"][name].numpy()
        )
    assert new.metadata["implementation_backend"] == (
        "tensorflow_tensorflow_probability_only"
    )
    assert new.metadata["adaptation_policy"] == "fixed_kernel_no_adaptation"


def test_native_runner_records_valid_status_and_fixed_trace_count() -> None:
    run = run_native_tfp_fixed_kernel_hmc(
        ReviewedGaussianAdapter(),
        tf.constant([0.1, -0.2], tf.float64),
        _config(),
    )
    status = run.diagnostics["target_status_telemetry"]
    assert tuple(run.samples.shape) == (4, 2)
    assert int(run.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert int(status["trace_entry_count"].numpy()) == 4
    assert bool(status["all_status_valid"].numpy()) is True
    assert run.diagnostics["native_divergence_status"] in {
        "available",
        "not_exposed_by_kernel",
    }


def test_native_runner_fails_closed_on_scope_and_unreviewed_authority() -> None:
    adapter = ReviewedGaussianAdapter()
    bad_scope = NativeTFPFixedKernelHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(1, 2),
        target_scope="wrong",
    )
    with pytest.raises(ValueError, match="target_scope mismatch"):
        run_native_tfp_fixed_kernel_hmc(
            adapter, tf.constant([0.0, 0.0], tf.float64), bad_scope
        )

    class DebugAdapter(ReviewedGaussianAdapter):
        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="debug_only",
                xla_hmc_ready=False,
                target_scope=SCOPE,
            )

    with pytest.raises(ValueError, match="reviewed graph value/score authority"):
        run_native_tfp_fixed_kernel_hmc(
            DebugAdapter(), tf.constant([0.0, 0.0], tf.float64), _config()
        )
