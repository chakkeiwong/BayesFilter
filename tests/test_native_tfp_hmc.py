from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.native_tfp_hmc import (
    NativeTFPIndependentChainHMCConfig,
    NativeTFPFixedKernelHMCConfig,
    load_native_tfp_retained_artifact,
    native_tfp_rank_normalized_diagnostics,
    native_tfp_retained_diagnostics,
    probe_native_tfp_independent_chain_graph,
    reviewed_independent_chain_target_fn,
    run_native_tfp_fixed_kernel_hmc,
    run_native_tfp_independent_chains,
    write_native_tfp_retained_artifact,
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


class RetainedStatusGaussianAdapter(ReviewedGaussianAdapter):
    def target_status_telemetry(self, theta):
        raise AssertionError("retained target status must avoid target re-evaluation")

    def retained_target_status_telemetry(self, target_log_prob):
        value = tf.cast(tf.convert_to_tensor(target_log_prob), tf.float64)
        valid = tf.math.is_finite(value)
        return {
            "status_code": tf.where(valid, 0, 1),
            "valid_pre_regularized_score": valid,
            "floor_count_value": tf.zeros(tf.shape(value), tf.int32),
            "min_innovation_eigenvalue": tf.zeros_like(value),
            "innovation_condition_estimate": tf.zeros_like(value),
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


def _independent_config() -> NativeTFPIndependentChainHMCConfig:
    return NativeTFPIndependentChainHMCConfig(
        num_results=8,
        num_burnin_steps=4,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260725, 9),
        chain_count=4,
        target_scope=SCOPE,
    )


def _independent_initial_state() -> tf.Tensor:
    return tf.constant(
        [[-0.4, 0.2], [-0.1, -0.3], [0.2, 0.4], [0.5, -0.2]],
        tf.float64,
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
        for forbidden in (
            "numpy",
            "scipy",
            "bayesfilter.inference.batched_value_score",
            "bayesfilter.inference.hmc",
        )
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


def test_independent_chain_target_matches_rows_and_has_no_cross_chain_gradient() -> None:
    adapter = ReviewedGaussianAdapter()
    positions = _independent_initial_state()
    target = reviewed_independent_chain_target_fn(
        adapter,
        chain_count=4,
        parameter_dim=2,
    )
    with tf.GradientTape() as tape:
        tape.watch(positions)
        values = target(positions)
        selected = values[2]
    gradient = tape.gradient(selected, positions)
    expected_values = -0.5 * tf.reduce_sum(tf.square(positions), axis=1)
    np.testing.assert_array_equal(values.numpy(), expected_values.numpy())
    np.testing.assert_array_equal(gradient[:2].numpy(), np.zeros((2, 2)))
    np.testing.assert_array_equal(gradient[2].numpy(), -positions[2].numpy())
    np.testing.assert_array_equal(gradient[3].numpy(), np.zeros((2,)))


def test_independent_chain_runner_is_finite_moving_and_status_complete() -> None:
    run = run_native_tfp_independent_chains(
        ReviewedGaussianAdapter(),
        _independent_initial_state(),
        _independent_config(),
    )
    status = run.diagnostics["target_status_telemetry"]
    assert tuple(run.samples.shape) == (8, 4, 2)
    assert int(run.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert bool(run.diagnostics["all_chains_moved"].numpy()) is True
    assert tuple(run.diagnostics["acceptance_rate_by_chain"].shape) == (4,)
    assert int(status["trace_entry_count"].numpy()) == 32
    assert bool(status["all_status_valid"].numpy()) is True
    assert run.metadata["target_batching"] == "scalar_rows_tf_while_loop"
    assert run.metadata["trace_count"] == 1
    assert run.metadata["target_status_trace_source"] == (
        "adapter_state_re_evaluation"
    )


def test_independent_chain_runner_uses_adapter_retained_target_status() -> None:
    run = run_native_tfp_independent_chains(
        RetainedStatusGaussianAdapter(),
        _independent_initial_state(),
        _independent_config(),
    )
    status = run.diagnostics["target_status_telemetry"]
    assert int(status["trace_entry_count"].numpy()) == 32
    assert bool(status["all_status_valid"].numpy()) is True
    assert run.metadata["target_status_trace_source"] == (
        "retained_accepted_target_log_prob"
    )


def test_one_step_status_probe_uses_adapter_retained_target_status() -> None:
    result = probe_native_tfp_independent_chain_graph(
        RetainedStatusGaussianAdapter(),
        _independent_initial_state(),
        _independent_config(),
        stage="one_step_status",
    )
    assert bool(result["all_numeric_outputs_finite"].numpy()) is True


@pytest.mark.parametrize(
    "stage", ("target", "status", "bootstrap", "one_step", "one_step_status")
)
def test_independent_chain_graph_probe_is_bounded_and_finite(stage: str) -> None:
    result = probe_native_tfp_independent_chain_graph(
        ReviewedGaussianAdapter(),
        _independent_initial_state(),
        _independent_config(),
        stage=stage,
    )
    assert result["stage"] == stage
    assert result["trace_seconds"] >= 0.0
    assert result["execute_seconds"] >= 0.0
    assert result["trace_count"] == 1
    assert bool(result["all_numeric_outputs_finite"].numpy()) is True
    assert result["diagnostic_role"] == (
        "graph_attribution_only_not_sampling_tuning_or_convergence"
    )


def test_independent_chain_graph_probe_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="stage must be one of"):
        probe_native_tfp_independent_chain_graph(
            ReviewedGaussianAdapter(),
            _independent_initial_state(),
            _independent_config(),
            stage="unknown",
        )


def test_independent_chain_config_requires_split_diagnostic_draw_contract() -> None:
    for num_results in (2, 5):
        with pytest.raises(ValueError, match="even num_results of at least four"):
            NativeTFPIndependentChainHMCConfig(
                num_results=num_results,
                num_burnin_steps=1,
                step_size=0.05,
                num_leapfrog_steps=1,
                seed=(1, 2),
                chain_count=4,
                target_scope=SCOPE,
            )


def test_retained_tensor_artifact_round_trips_and_rejects_corruption(tmp_path) -> None:
    adapter = ReviewedGaussianAdapter()
    config = _independent_config()
    run = run_native_tfp_independent_chains(
        adapter,
        _independent_initial_state(),
        config,
    )
    root = tmp_path / "native-retained"
    written = write_native_tfp_retained_artifact(
        root,
        run,
        adapter=adapter,
        config=config,
    )
    loaded = load_native_tfp_retained_artifact(
        root,
        expected_adapter_signature=adapter.adapter_signature(),
        expected_program_signature=run.metadata["program_signature"],
    )
    assert loaded.manifest_sha256 == written["manifest_sha256"]
    np.testing.assert_array_equal(
        loaded.initial_state.numpy(), _independent_initial_state().numpy()
    )
    np.testing.assert_array_equal(loaded.samples.numpy(), run.samples.numpy())
    np.testing.assert_array_equal(
        loaded.trace["log_accept_ratio"].numpy(),
        run.trace["log_accept_ratio"].numpy(),
    )
    np.testing.assert_array_equal(
        loaded.trace["target_status_telemetry"]["status_code"].numpy(),
        run.trace["target_status_telemetry"]["status_code"].numpy(),
    )
    retained_diagnostics = native_tfp_retained_diagnostics(loaded)
    assert retained_diagnostics["source_manifest_sha256"] == loaded.manifest_sha256
    assert retained_diagnostics["source_sample_layout"] == "draw_chain_parameter"
    assert retained_diagnostics["diagnostic_sample_layout"] == (
        "chain_draw_parameter"
    )
    with pytest.raises(FileExistsError):
        write_native_tfp_retained_artifact(
            root,
            run,
            adapter=adapter,
            config=config,
        )
    mismatched_root = tmp_path / "mismatched-native-retained"
    mismatched_config = NativeTFPIndependentChainHMCConfig(
        num_results=config.num_results,
        num_burnin_steps=config.num_burnin_steps,
        step_size=config.step_size * 2.0,
        num_leapfrog_steps=config.num_leapfrog_steps,
        seed=config.seed,
        chain_count=config.chain_count,
        target_scope=config.target_scope,
    )
    with pytest.raises(ValueError, match="program signature mismatch"):
        write_native_tfp_retained_artifact(
            mismatched_root,
            run,
            adapter=adapter,
            config=mismatched_config,
        )
    assert not mismatched_root.exists()
    appended_path = root / "unbound.tensor"
    appended_path.write_bytes(b"unbound")
    with pytest.raises(RuntimeError, match="file-set drift"):
        load_native_tfp_retained_artifact(root)
    appended_path.unlink()
    sample_path = root / loaded.manifest["tensors"]["samples"]["path"]
    sample_path.write_bytes(sample_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="tensor drift"):
        load_native_tfp_retained_artifact(root)


def test_retained_tensor_artifact_does_not_publish_partial_write(
    tmp_path, monkeypatch
) -> None:
    adapter = ReviewedGaussianAdapter()
    config = _independent_config()
    run = run_native_tfp_independent_chains(
        adapter,
        _independent_initial_state(),
        config,
    )
    original_write_file = tf.io.write_file
    calls = 0

    def fail_second_write(path, contents):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected artifact write failure")
        return original_write_file(path, contents)

    monkeypatch.setattr(tf.io, "write_file", fail_second_write)
    root = tmp_path / "partial-native-retained"
    with pytest.raises(RuntimeError, match="injected artifact write failure"):
        write_native_tfp_retained_artifact(
            root,
            run,
            adapter=adapter,
            config=config,
        )
    assert not root.exists()
    assert list(tmp_path.iterdir()) == []


def test_rank_normalized_diagnostics_distinguish_well_mixed_and_separated_chains() -> None:
    draws = tf.random.stateless_normal(
        (256, 4, 2),
        seed=(20260725, 10),
        dtype=tf.float64,
    )
    well_mixed = tf.transpose(draws, (1, 0, 2))
    separated = well_mixed + tf.constant(
        [[[-4.0]], [[-1.5]], [[1.5]], [[4.0]]], tf.float64
    )
    good = native_tfp_rank_normalized_diagnostics(well_mixed)
    bad = native_tfp_rank_normalized_diagnostics(separated)
    good_rhat = good["rank_normalized_split_rhat"]["maximum"]
    bad_rhat = bad["rank_normalized_split_rhat"]["maximum"]
    assert bool(tf.reduce_all(tf.math.is_finite(good_rhat)).numpy())
    assert bool(tf.reduce_all(good_rhat < 1.10).numpy())
    assert bool(tf.reduce_any(bad_rhat > 1.10).numpy())
    bulk = good["rank_normalized_bulk_tail_ess"]["bulk"]
    tail = good["rank_normalized_bulk_tail_ess"]["tail"]
    assert bool(tf.reduce_all(tf.math.is_finite(bulk)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(tail)).numpy())
    assert good["diagnostic_role"] == "finite_sample_screen_not_convergence_proof"
