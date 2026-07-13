from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    FullChainHMCConfig,
    HMCTuningPolicy,
    ValueScoreCapability,
    build_reusable_full_chain_tfp_hmc_runner,
    run_gaussian_dual_averaging_diagnostic,
    run_full_chain_tfp_hmc,
)


class ReviewedGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "phase4-reviewed-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow",
            evidence_path="tests/test_nonlinear_ssm_phase4_full_chain_hmc.py",
            target_scope="phase4_reviewed_gaussian",
            nonclaims=("tiny full-chain HMC engineering fixture only",),
            full_chain_xla_diagnostic_ready=True,
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values)), -values


class ReviewedBatchedGaussianAdapter(ReviewedGaussianAdapter):
    def adapter_signature(self) -> str:
        return "phase4-reviewed-batched-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow",
            evidence_path="tests/test_nonlinear_ssm_phase4_full_chain_hmc.py",
            target_scope="phase4_reviewed_batched_gaussian",
            nonclaims=("tiny chain-batched full-chain HMC engineering fixture only",),
            full_chain_xla_diagnostic_ready=True,
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values


class TelemetryBatchedGaussianAdapter(ReviewedBatchedGaussianAdapter):
    def adapter_signature(self) -> str:
        return "phase4-telemetry-batched-gaussian-v1"

    def target_status_telemetry(self, theta: tf.Tensor) -> dict[str, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        status_shape = tf.shape(values)[:-1]
        min_eigen = 0.5 + 0.0 * tf.reduce_sum(values, axis=-1)
        condition = 2.0 + 0.0 * tf.reduce_sum(values, axis=-1)
        return {
            "status_code": tf.zeros(status_shape, dtype=tf.int32),
            "valid_pre_regularized_score": tf.ones(status_shape, dtype=tf.bool),
            "floor_count_value": tf.zeros(status_shape, dtype=tf.int32),
            "min_innovation_eigenvalue": tf.cast(min_eigen, tf.float64),
            "innovation_condition_estimate": tf.cast(condition, tf.float64),
            "regularization_derivative_target": tf.zeros(status_shape, dtype=tf.int32),
        }


class NonvalidTelemetryBatchedGaussianAdapter(TelemetryBatchedGaussianAdapter):
    def target_status_telemetry(self, theta: tf.Tensor) -> dict[str, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        status_shape = tf.shape(values)[:-1]
        return {
            "status_code": tf.ones(status_shape, dtype=tf.int32),
            "valid_pre_regularized_score": tf.zeros(status_shape, dtype=tf.bool),
            "floor_count_value": tf.ones(status_shape, dtype=tf.int32),
            "min_innovation_eigenvalue": tf.zeros(status_shape, dtype=tf.float64),
            "innovation_condition_estimate": tf.fill(status_shape, tf.constant(1.0e12, dtype=tf.float64)),
            "regularization_derivative_target": tf.zeros(status_shape, dtype=tf.int32),
        }


class BadBatchedGaussianAdapter(ReviewedBatchedGaussianAdapter):
    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        value = -0.5 * tf.reduce_sum(tf.square(values), axis=-1)
        score = -values[0]
        return value, score


class UnreviewedGaussianAdapter(ReviewedGaussianAdapter):
    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="gradient_tape_fallback",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            nonclaims=("debug fallback only",),
        )


class TargetOnlyXLAGaussianAdapter(ReviewedGaussianAdapter):
    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow",
            evidence_path="tests/test_nonlinear_ssm_phase4_full_chain_hmc.py",
            target_scope="phase4_target_only_gaussian",
            nonclaims=("target-only XLA parity fixture; no full-chain XLA authority",),
        )


class EagerOnlyGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "phase4-eager-only-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="debug_only",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            evidence_path="tests/test_nonlinear_ssm_phase4_full_chain_hmc.py",
            target_scope="phase4_eager_only_gaussian",
            nonclaims=("tiny eager-only chain interface fixture only",),
        )

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        if not tf.executing_eagerly():
            raise RuntimeError("blocked_non_eager")
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values))


def _config(trace_policy: str = "standard") -> FullChainHMCConfig:
    return FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260608, 4),
        use_xla=True,
        trace_policy=trace_policy,
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope="phase4_reviewed_gaussian",
    )


def _batched_config(trace_policy: str = "standard") -> FullChainHMCConfig:
    return FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260610, 7),
        use_xla=True,
        trace_policy=trace_policy,
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope="phase4_reviewed_batched_gaussian",
    )


def test_phase4_cpu_only_hides_gpu_before_tensorflow_runtime_probe() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert tf.config.list_physical_devices("GPU") == []


def test_phase4_tiny_full_chain_hmc_jit_returns_finite_samples_and_metadata() -> None:
    config = _config()
    assert config.chain_execution_mode == "tf_function"
    assert config.signature_payload()["chain_execution_mode"] == "tf_function"

    result = run_full_chain_tfp_hmc(
        ReviewedGaussianAdapter(),
        tf.constant([0.1, -0.2], dtype=tf.float64),
        config,
    )

    assert result.samples.shape == (4, 2)
    assert int(result.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert int(result.diagnostics["finite_sample_count"].numpy()) == 4
    assert np.isfinite(float(result.diagnostics["acceptance_rate"].numpy()))
    assert result.diagnostics["native_divergence_status"] == "not_exposed_by_kernel"
    assert result.diagnostics["divergence_status"] == "not_exposed_by_kernel"
    assert result.diagnostics["divergence_count"] is None
    assert result.diagnostics["divergence_source"] is None
    assert result.diagnostics["hmc_health_diagnostics"]["diagnostic_role"] == (
        "hmc_health_diagnostics_not_native_divergence"
    )
    assert np.isfinite(
        float(result.diagnostics["hmc_health_diagnostics"]["acceptance_rate"].numpy())
    )
    assert result.diagnostics["hmc_health_diagnostics"]["log_accept_ratio"]["available"] is True
    assert result.diagnostics["hmc_health_diagnostics"]["target_log_prob"]["available"] is True
    assert (
        result.diagnostics["hmc_health_diagnostics"]["proposed_target_log_prob"][
            "available"
        ]
        is True
    )
    assert (
        result.diagnostics["hmc_health_diagnostics"]["log_acceptance_correction"][
            "available"
        ]
        is True
    )
    assert set(result.trace) == {
        "is_accepted",
        "log_acceptance_correction",
        "log_accept_ratio",
        "proposed_target_log_prob",
        "target_log_prob",
    }
    assert result.metadata["runtime"] == "tfp.mcmc.sample_chain"
    assert result.metadata["jit_compile"] is True
    assert result.metadata["chain_execution_mode"] == "tf_function"
    assert result.metadata["trace_policy"] == "standard"
    assert result.metadata["adaptation_policy"] == "fixed_kernel_no_adaptation"
    assert result.metadata["trace_unavailability"] == {
        "divergence": (
            "native boolean divergence field not exposed by "
            "TensorFlow Probability HMC kernel results"
        ),
    }
    assert result.metadata["value_score_authority"] == "graph_native"
    assert result.metadata["target_scope"] == "phase4_reviewed_gaussian"
    assert isinstance(result.metadata["program_signature"], str)
    assert result.metadata["first_call_s"] >= 0.0
    assert result.metadata["sample_chain_call_s"] >= 0.0
    assert result.metadata["sample_chain_invocation_count"] == 1
    assert result.metadata["runner_build_s"] >= 0.0
    assert result.metadata["first_sample_chain_compile_execute_s"] == result.metadata["first_call_s"]
    assert result.metadata["warm_call_s"] is None
    assert result.metadata["warm_sample_chain_execute_s"] is None
    assert result.metadata["trace_capture_s"] >= 0.0
    assert result.metadata["trace_capture_timing_scope"] == (
        "post_sample_chain_public_safe_trace_diagnostics_capture"
    )
    assert result.metadata["timing_buckets"]["runner_build_s"].startswith("explanatory_only")
    assert result.metadata["warm_sample_shape"] is None
    assert result.metadata["warm_trace_keys"] == ()
    assert "no sampler convergence claim" in result.metadata["nonclaims"]


def test_phase4_full_chain_hmc_promotes_float32_initial_state_to_fp64_state() -> None:
    config = FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260617, 31),
        use_xla=False,
        trace_policy="standard",
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope="phase4_reviewed_gaussian",
    )

    result = run_full_chain_tfp_hmc(
        ReviewedGaussianAdapter(),
        tf.constant([0.1, -0.2], dtype=tf.float32),
        config,
    )

    assert result.samples.dtype == tf.float64
    assert result.metadata["initial_state_dtype"] == "float64"
    assert int(result.diagnostics["nonfinite_sample_count"].numpy()) == 0


def test_phase4_chain_batched_full_chain_hmc_broadcasts_upstream_gradients() -> None:
    import bayesfilter.inference.hmc as hmc_module

    initial_state = tf.constant(
        [[0.1, -0.2], [0.2, 0.1], [-0.1, 0.3], [0.05, -0.05]],
        dtype=tf.float64,
    )
    target = hmc_module._make_tfp_target_log_prob_fn(
        ReviewedBatchedGaussianAdapter(),
        dtype=tf.float64,
    )

    with tf.GradientTape() as tape:
        tape.watch(initial_state)
        value = target(initial_state)
    gradient = tape.gradient(value, initial_state)
    assert gradient.shape == initial_state.shape
    np.testing.assert_allclose(gradient.numpy(), -initial_state.numpy())

    result = run_full_chain_tfp_hmc(
        ReviewedBatchedGaussianAdapter(),
        initial_state,
        _batched_config(),
    )

    assert result.samples.shape == (4, 4, 2)
    assert int(result.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert int(result.diagnostics["finite_sample_count"].numpy()) == 16
    assert set(result.trace) == {"is_accepted", "log_accept_ratio", "target_log_prob"}
    assert result.trace["is_accepted"].shape == (4, 4)
    assert result.trace["log_accept_ratio"].shape == (4, 4)
    assert result.trace["target_log_prob"].shape == (4, 4)
    assert result.metadata["target_scope"] == "phase4_reviewed_batched_gaussian"
    assert result.metadata["requested_target_scope"] == "phase4_reviewed_batched_gaussian"
    assert result.metadata["jit_compile"] is True
    assert "no sampler convergence claim" in result.metadata["nonclaims"]


def test_phase4_target_status_trace_policy_preserves_default_trace_when_disabled() -> None:
    config = _batched_config()

    assert config.target_status_trace_policy == "none"
    assert config.signature_payload()["target_status_trace_policy"] == "none"

    result = run_full_chain_tfp_hmc(
        TelemetryBatchedGaussianAdapter(),
        tf.constant([[0.1, -0.2], [0.2, 0.1]], dtype=tf.float64),
        config,
    )

    assert set(result.trace) == {"is_accepted", "log_accept_ratio", "target_log_prob"}
    assert "target_status_telemetry" not in result.trace
    assert result.metadata["target_status_trace_policy"] == "none"


def test_phase4_target_status_trace_policy_fails_closed_without_adapter_method() -> None:
    config = FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260618, 10),
        use_xla=False,
        trace_policy="standard",
        target_status_trace_policy="per_chain_step",
        target_scope="phase4_reviewed_batched_gaussian",
    )

    with pytest.raises(TypeError, match="target_status_telemetry"):
        run_full_chain_tfp_hmc(
            ReviewedBatchedGaussianAdapter(),
            tf.constant([[0.1, -0.2], [0.2, 0.1]], dtype=tf.float64),
            config,
        )


def test_phase4_target_status_trace_policy_records_raw_and_summary_telemetry() -> None:
    initial_state = tf.constant(
        [[0.1, -0.2], [0.2, 0.1]],
        dtype=tf.float64,
    )
    config = FullChainHMCConfig(
        num_results=3,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260618, 11),
        use_xla=False,
        trace_policy="standard",
        target_status_trace_policy="per_chain_step",
        target_scope="phase4_reviewed_batched_gaussian",
    )

    result = run_full_chain_tfp_hmc(
        TelemetryBatchedGaussianAdapter(),
        initial_state,
        config,
    )

    assert result.samples.shape == (3, 2, 2)
    assert set(result.trace) == {
        "is_accepted",
        "log_accept_ratio",
        "target_log_prob",
        "target_status_telemetry",
    }
    telemetry = result.trace["target_status_telemetry"]
    assert telemetry["status_code"].shape == (3, 2)
    assert telemetry["valid_pre_regularized_score"].shape == (3, 2)
    summary = result.diagnostics["target_status_telemetry"]
    assert int(summary["trace_entry_count"].numpy()) == 6
    assert int(summary["status_nonvalid_count"].numpy()) == 0
    assert bool(summary["all_status_valid"].numpy()) is True
    assert int(summary["floor_count_total"].numpy()) == 0
    assert int(summary["max_floor_count_value"].numpy()) == 0
    assert float(summary["min_min_innovation_eigenvalue"].numpy()) == pytest.approx(0.5)
    assert float(summary["max_innovation_condition_estimate"].numpy()) == pytest.approx(2.0)
    assert bool(summary["telemetry_failure_veto"].numpy()) is False
    assert result.metadata["target_status_trace_policy"] == "per_chain_step"


def test_phase4_target_status_diagnostics_records_nonvalid_status_as_veto() -> None:
    import bayesfilter.inference.hmc as hmc_module

    telemetry = NonvalidTelemetryBatchedGaussianAdapter().target_status_telemetry(
        tf.constant([[0.1, -0.2], [0.2, 0.1]], dtype=tf.float64)
    )

    summary = hmc_module._target_status_telemetry_diagnostics(telemetry)

    assert int(summary["trace_entry_count"].numpy()) == 2
    assert int(summary["status_nonvalid_count"].numpy()) == 2
    assert bool(summary["all_status_valid"].numpy()) is False
    assert bool(summary["telemetry_failure_veto"].numpy()) is True


def test_phase4_reusable_full_chain_runner_reuses_compiled_shape_and_seed_argument() -> None:
    initial_state = tf.constant(
        [[0.1, -0.2], [0.2, 0.1], [-0.1, 0.3], [0.05, -0.05]],
        dtype=tf.float64,
    )
    config = FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260617, 1),
        use_xla=False,
        trace_policy="standard",
        target_scope="phase4_reviewed_batched_gaussian",
    )

    runner = build_reusable_full_chain_tfp_hmc_runner(
        ReviewedBatchedGaussianAdapter(),
        initial_state,
        config,
    )
    first = runner.run(seed=(20260617, 2), step_size=0.05)
    second = runner.run(seed=(20260617, 3), step_size=0.04)

    assert first.samples.shape == (2, 4, 2)
    assert second.samples.shape == (2, 4, 2)
    assert int(first.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert int(second.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert set(second.trace) == {"is_accepted", "log_accept_ratio", "target_log_prob"}
    assert first.metadata["reusable_runner"] is True
    assert second.metadata["reusable_runner"] is True
    assert first.metadata["sample_chain_invocation_count"] == 1
    assert second.metadata["sample_chain_invocation_count"] == 2
    assert second.metadata["use_xla"] is False
    assert second.metadata["chain_execution_mode"] == "tf_function"
    assert second.metadata["sample_chain_timing_scope"] == (
        "reusable_tf_function_first_call_trace_compile_plus_execute_then_warm_execute"
    )
    assert second.metadata["runner_build_s"] == first.metadata["runner_build_s"]
    assert second.metadata["runner_build_s"] >= 0.0
    assert first.metadata["first_call_s"] >= 0.0
    assert first.metadata["warm_call_s"] is None
    assert second.metadata["first_call_s"] == first.metadata["first_call_s"]
    assert second.metadata["first_sample_chain_compile_execute_s"] == first.metadata["first_call_s"]
    assert second.metadata["warm_call_s"] >= 0.0
    assert second.metadata["warm_sample_chain_execute_s"] == second.metadata["warm_call_s"]
    assert second.metadata["trace_capture_s"] >= 0.0
    assert second.metadata["timing_buckets"]["warm_call_s"].startswith("explanatory_only")
    assert second.metadata["dynamic_inputs"] == ("current_state", "seed", "step_size")
    assert second.metadata["seed_source"] == "runtime_tensor_argument"
    assert second.metadata["current_state_source"] == "runtime_tensor_argument"
    assert second.metadata["step_size_source"] == "runtime_tensor_argument"
    assert second.metadata["target_scope"] == "phase4_reviewed_batched_gaussian"
    assert second.metadata["requested_target_scope"] == "phase4_reviewed_batched_gaussian"
    assert "no sampler convergence claim" in second.metadata["nonclaims"]


def test_phase4_reusable_full_chain_runner_dynamic_l_reuses_xla_trace() -> None:
    initial_state = tf.constant([0.1, -0.2], dtype=tf.float64)
    config = FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260705, 1),
        use_xla=True,
        trace_policy="standard",
        target_scope="phase4_reviewed_gaussian",
    )

    runner = build_reusable_full_chain_tfp_hmc_runner(
        ReviewedGaussianAdapter(),
        initial_state,
        config,
        dynamic_num_leapfrog_steps=True,
    )
    first = runner.run(seed=(20260705, 2), step_size=0.05, num_leapfrog_steps=1)
    second = runner.run(seed=(20260705, 3), step_size=0.05, num_leapfrog_steps=3)

    assert int(first.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert int(second.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert first.metadata["dynamic_num_leapfrog_steps"] is True
    assert second.metadata["dynamic_num_leapfrog_steps"] is True
    assert second.metadata["dynamic_inputs"] == (
        "current_state",
        "seed",
        "step_size",
        "num_leapfrog_steps",
    )
    assert first.metadata["num_leapfrog_steps"] == 1
    assert second.metadata["num_leapfrog_steps"] == 3
    assert runner._runner.experimental_get_tracing_count() == 1


def test_phase4_reusable_full_chain_runner_validates_static_state_shape() -> None:
    initial_state = tf.constant(
        [[0.1, -0.2], [0.2, 0.1]],
        dtype=tf.float64,
    )
    config = FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260617, 4),
        use_xla=False,
        trace_policy="standard",
        target_scope="phase4_reviewed_batched_gaussian",
    )
    runner = build_reusable_full_chain_tfp_hmc_runner(
        ReviewedBatchedGaussianAdapter(),
        initial_state,
        config,
    )

    with pytest.raises(ValueError, match="current_state shape"):
        runner.run(current_state=tf.zeros((3, 2), dtype=tf.float64))
    with pytest.raises(ValueError, match="seed"):
        runner.run(seed=(1, 2, 3))
    with pytest.raises(ValueError, match="step_size"):
        runner.run(step_size=tf.ones((1,), dtype=tf.float64))


def test_phase4_reusable_full_chain_runner_preserves_xla_authority_gate() -> None:
    config = FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260617, 5),
        use_xla=True,
        trace_policy="standard",
        target_scope="phase4_target_only_gaussian",
    )

    with pytest.raises(ValueError, match="target-only XLA readiness is not sufficient"):
        build_reusable_full_chain_tfp_hmc_runner(
            TargetOnlyXLAGaussianAdapter(),
            tf.constant([0.1, -0.2], dtype=tf.float64),
            config,
        )


def test_phase4_custom_gradient_rejects_incompatible_value_score_shapes() -> None:
    import bayesfilter.inference.hmc as hmc_module

    target = hmc_module._make_tfp_target_log_prob_fn(
        BadBatchedGaussianAdapter(),
        dtype=tf.float64,
    )
    initial_state = tf.constant(
        [[0.1, -0.2], [0.2, 0.1], [-0.1, 0.3], [0.05, -0.05]],
        dtype=tf.float64,
    )

    with pytest.raises(Exception, match="batched target score must have rank 2"):
        with tf.GradientTape() as tape:
            tape.watch(initial_state)
            value = target(initial_state)
        tape.gradient(value, initial_state)


def test_phase4_native_divergence_extractor_accepts_only_boolean_kernel_fields() -> None:
    import bayesfilter.inference.hmc as hmc_module

    native = hmc_module._extract_native_divergence_tensor(
        SimpleNamespace(
            log_accept_ratio=tf.constant([1200.0], dtype=tf.float64),
            proposed_results=SimpleNamespace(
                log_acceptance_correction=tf.constant([1200.0], dtype=tf.float64),
                is_divergent=tf.constant([False, True], dtype=tf.bool),
            ),
        )
    )
    proxy = hmc_module._extract_native_divergence_tensor(
        SimpleNamespace(
            divergence=tf.constant([0.0, 1.0], dtype=tf.float64),
            proposed_results=SimpleNamespace(
                log_acceptance_correction=tf.constant([1200.0], dtype=tf.float64),
            ),
        )
    )

    assert native is not None
    np.testing.assert_array_equal(native.numpy(), np.asarray([False, True]))
    assert proxy is None


def test_phase4_xla_full_chain_hmc_fails_closed_without_reviewed_authority() -> None:
    with pytest.raises(ValueError, match="XLA full-chain HMC requires"):
        run_full_chain_tfp_hmc(
            UnreviewedGaussianAdapter(),
            tf.constant([0.1, -0.2], dtype=tf.float64),
            _config(),
        )


def test_phase4_target_only_xla_readiness_does_not_authorize_full_chain_xla() -> None:
    config = FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260615, 1),
        use_xla=True,
        trace_policy="standard",
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope="phase4_target_only_gaussian",
    )

    with pytest.raises(ValueError, match="target-only XLA readiness is not sufficient"):
        run_full_chain_tfp_hmc(
            TargetOnlyXLAGaussianAdapter(),
            tf.constant([0.1, -0.2], dtype=tf.float64),
            config,
        )


def test_phase4_eager_chain_mode_runs_eager_only_target_and_records_metadata() -> None:
    tf_function_config = FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260616, 1),
        use_xla=False,
        trace_policy="standard",
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope="phase4_eager_only_gaussian",
    )
    with pytest.raises(RuntimeError, match="blocked_non_eager"):
        run_full_chain_tfp_hmc(
            EagerOnlyGaussianAdapter(),
            tf.constant([0.1, -0.2], dtype=tf.float64),
            tf_function_config,
        )

    config = FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260616, 1),
        use_xla=False,
        chain_execution_mode="eager",
        trace_policy="standard",
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope="phase4_eager_only_gaussian",
    )
    result = run_full_chain_tfp_hmc(
        EagerOnlyGaussianAdapter(),
        tf.constant([0.1, -0.2], dtype=tf.float64),
        config,
    )

    assert result.samples.shape == (4, 2)
    assert int(result.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert result.metadata["jit_compile"] is False
    assert result.metadata["chain_execution_mode"] == "eager"
    assert result.metadata["target_scope"] == "phase4_eager_only_gaussian"
    assert config.signature_payload()["chain_execution_mode"] == "eager"


def test_phase4_xla_rejects_eager_chain_execution_mode() -> None:
    with pytest.raises(ValueError, match="XLA full-chain HMC requires"):
        FullChainHMCConfig(
            num_results=4,
            num_burnin_steps=2,
            step_size=0.05,
            num_leapfrog_steps=2,
            seed=(20260616, 2),
            use_xla=True,
            chain_execution_mode="eager",
            trace_policy="standard",
            adaptation_policy="fixed_kernel_no_adaptation",
            target_scope="phase4_reviewed_gaussian",
        )


def test_phase4_rejects_invalid_chain_execution_mode() -> None:
    with pytest.raises(ValueError, match="chain_execution_mode"):
        FullChainHMCConfig(
            num_results=4,
            num_burnin_steps=2,
            step_size=0.05,
            num_leapfrog_steps=2,
            seed=(20260616, 3),
            use_xla=False,
            chain_execution_mode="plain_python",
            trace_policy="standard",
            adaptation_policy="fixed_kernel_no_adaptation",
            target_scope="phase4_reviewed_gaussian",
        )


def test_phase4_xla_full_chain_hmc_rejects_target_scope_mismatch() -> None:
    config = FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260608, 4),
        use_xla=True,
        trace_policy="standard",
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope="wrong_scope",
    )

    with pytest.raises(ValueError, match="target_scope mismatch"):
        run_full_chain_tfp_hmc(
            ReviewedGaussianAdapter(),
            tf.constant([0.1, -0.2], dtype=tf.float64),
            config,
        )


def test_phase4_reduced_trace_reports_unavailable_diagnostics_not_zero() -> None:
    result = run_full_chain_tfp_hmc(
        ReviewedGaussianAdapter(),
        tf.constant([0.1, -0.2], dtype=tf.float64),
        _config(trace_policy="reduced"),
    )

    assert set(result.trace) == {"trace_collected"}
    assert result.diagnostics["acceptance_rate"] is None
    assert result.diagnostics["native_divergence_status"] == "unavailable"
    assert result.diagnostics["divergence_status"] == "unavailable"
    assert result.diagnostics["divergence_count"] is None
    assert result.diagnostics["divergence_source"] is None
    assert result.diagnostics["hmc_health_diagnostics"]["diagnostic_role"] == (
        "hmc_health_diagnostics_not_native_divergence"
    )
    assert result.diagnostics["hmc_health_diagnostics"]["acceptance_rate"] is None
    assert result.diagnostics["hmc_health_diagnostics"]["log_accept_ratio"]["available"] is False
    assert result.diagnostics["hmc_health_diagnostics"]["target_log_prob"]["available"] is False
    assert result.metadata["trace_unavailability"] == {
        "is_accepted": "reduced trace policy",
        "log_accept_ratio": "reduced trace policy",
        "target_log_prob": "reduced trace policy",
        "divergence": "reduced trace policy",
    }


def test_phase4_rejects_unreviewed_adaptation_policy() -> None:
    with pytest.raises(ValueError, match="fail-closed"):
        FullChainHMCConfig(
            num_results=4,
            num_burnin_steps=2,
            step_size=0.05,
            num_leapfrog_steps=2,
            seed=(20260608, 4),
            use_xla=True,
            trace_policy="standard",
            adaptation_policy="dual_averaging",
            target_scope="phase4_reviewed_gaussian",
        )


def test_phase4_reviewed_dual_averaging_policy_records_diagnostic_telemetry() -> None:
    policy = HMCTuningPolicy.fixed_mass_dual_averaging(
        num_adaptation_steps=2,
        target_accept_prob=0.75,
        source="tests/test_nonlinear_ssm_phase4_full_chain_hmc.py",
    )
    config = FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260608, 7),
        use_xla=False,
        trace_policy="standard",
        tuning_policy=policy,
        target_scope="phase4_reviewed_gaussian",
    )
    result = run_full_chain_tfp_hmc(
        ReviewedGaussianAdapter(),
        tf.constant([0.1, -0.2], dtype=tf.float64),
        config,
    )

    assert result.metadata["adaptation_policy"] == "dual_averaging_step_size"
    assert result.metadata["tuning_policy"]["label"] == "fixed_mass_dual_averaging"
    assert result.metadata["tuning_policy"]["num_adaptation_steps"] == 2
    assert result.diagnostics["num_adaptation_steps"].numpy() == 2
    assert result.diagnostics["target_accept_prob"].numpy() == pytest.approx(0.75)
    assert np.all(np.isfinite(result.diagnostics["final_step_size"].numpy()))
    assert bool(result.diagnostics["final_step_size_finite"].numpy()) is True
    assert "step_size" in result.trace
    assert "no sampler convergence claim" in result.metadata["nonclaims"]


def test_phase4_reviewed_dual_averaging_policy_compiles_with_xla() -> None:
    policy = HMCTuningPolicy.fixed_mass_dual_averaging(
        num_adaptation_steps=2,
        target_accept_prob=0.75,
        source="tests/test_nonlinear_ssm_phase4_full_chain_hmc.py",
    )
    config = FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260620, 1),
        use_xla=True,
        trace_policy="standard",
        tuning_policy=policy,
        target_scope="phase4_reviewed_gaussian",
    )
    result = run_full_chain_tfp_hmc(
        ReviewedGaussianAdapter(),
        tf.constant([0.1, -0.2], dtype=tf.float64),
        config,
    )

    assert result.samples.shape == (4, 2)
    assert int(result.diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert result.metadata["jit_compile"] is True
    assert result.metadata["adaptation_policy"] == "dual_averaging_step_size"
    assert result.metadata["tuning_policy"]["label"] == "fixed_mass_dual_averaging"
    assert result.metadata["value_score_authority"] == "graph_native"
    assert result.metadata["target_scope"] == "phase4_reviewed_gaussian"
    assert result.metadata["requested_target_scope"] == "phase4_reviewed_gaussian"
    assert result.diagnostics["num_adaptation_steps"].numpy() == 2
    assert result.diagnostics["target_accept_prob"].numpy() == pytest.approx(0.75)
    assert np.all(np.isfinite(result.diagnostics["final_step_size"].numpy()))
    assert bool(result.diagnostics["final_step_size_finite"].numpy()) is True
    assert {
        "is_accepted",
        "log_accept_ratio",
        "target_log_prob",
        "step_size",
        "target_accept_prob",
        "num_adaptation_steps",
    }.issubset(result.trace)
    assert "no sampler convergence claim" in result.metadata["nonclaims"]
    assert "no posterior validity claim" in result.metadata["nonclaims"]


def test_phase4_gaussian_dual_averaging_diagnostic_records_nonclaims() -> None:
    policy = HMCTuningPolicy.fixed_mass_dual_averaging(
        num_adaptation_steps=2,
        target_accept_prob=0.75,
        source="tests/test_nonlinear_ssm_phase4_full_chain_hmc.py",
    )
    result = run_gaussian_dual_averaging_diagnostic(
        policy,
        initial_state=tf.constant([[0.1, -0.2], [0.2, 0.1]], dtype=tf.float64),
        num_results=4,
        num_burnin_steps=2,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260609, 3),
    )

    assert result.policy.label == "fixed_mass_dual_averaging"
    assert result.diagnostics["num_adaptation_steps"] == 2
    assert result.diagnostics["target_accept_prob"] == pytest.approx(0.75)
    assert np.isfinite(result.diagnostics["final_step_size"])
    assert result.diagnostics["final_step_size_finite"] is True
    assert result.diagnostics["reports_posterior_convergence"] is False
    assert "step_size" in result.trace
    assert "no posterior convergence claim" in result.metadata["nonclaims"]


def test_phase4_compiled_sampling_source_does_not_materialize_numpy_inside_path() -> None:
    import bayesfilter.inference.hmc as hmc_module

    source = inspect.getsource(hmc_module.run_full_chain_tfp_hmc)
    source += inspect.getsource(hmc_module._make_tfp_target_log_prob_fn)
    source += inspect.getsource(hmc_module._broadcast_upstream_gradient_to_score)
    source += inspect.getsource(hmc_module._build_sample_chain_runner)
    source += inspect.getsource(hmc_module._standard_trace_fn)
    source += inspect.getsource(hmc_module._native_divergence_trace)
    source += inspect.getsource(hmc_module._extract_native_divergence_tensor)
    source += inspect.getsource(hmc_module._reduced_trace_fn)
    source += inspect.getsource(ReviewedGaussianAdapter.log_prob_and_grad)
    source += inspect.getsource(ReviewedBatchedGaussianAdapter.log_prob_and_grad)

    assert ".numpy(" not in source


def test_phase4_full_chain_target_uses_reviewed_value_score_helper() -> None:
    import bayesfilter.inference.hmc as hmc_module

    source = inspect.getsource(hmc_module._make_tfp_target_log_prob_fn)

    assert "reviewed_value_score_target_fn" in source
    assert "tf.custom_gradient" not in source
    assert "value_with_reviewed_score" not in source
