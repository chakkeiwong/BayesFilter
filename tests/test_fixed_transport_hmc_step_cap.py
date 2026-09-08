from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference import (
    FixedTransportHMCKernelTuningConfig,
    ValueScoreCapability,
    tune_fixed_transport_hmc_kernel,
)
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
    FixedTransportFullChainConfig,
    FixedTransportHMCPolicy,
    FixedTransportReusableRunnerPool,
    build_fixed_transport_value_score_adapter,
    fixed_transport_capped_step_size_setter,
    run_fixed_transport_full_chain_tfp_hmc,
)
from bayesfilter.inference.hmc import FullChainHMCRunResult


class GaussianAdapter:
    parameter_dim = 2

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="step_cap_fixture",
            target_scope="step_cap_fixture",
            nonclaims=("step-cap mechanics fixture only",),
        )

    def adapter_signature(self) -> str:
        return "step-cap-gaussian-v1"

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values


class IdentityTransport:
    parameter_dim = 2

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema": "step_cap_identity_transport.v1",
            "parameter_dim": self.parameter_dim,
        }

    def forward(self, z: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(z, tf.float64)

    def forward_batch(self, z: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(z, tf.float64)

    def log_abs_det_jacobian(self, z: tf.Tensor) -> tf.Tensor:
        del z
        return tf.constant(0.0, tf.float64)

    def log_abs_det_jacobian_batch(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros(tf.shape(tf.convert_to_tensor(z))[:1], tf.float64)

    def pullback_score(self, z: tf.Tensor, score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.convert_to_tensor(score, tf.float64)

    def pullback_score_batch(self, z: tf.Tensor, score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.convert_to_tensor(score, tf.float64)

    def log_abs_det_jacobian_score(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(z, tf.float64))

    def log_abs_det_jacobian_score_batch(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(z, tf.float64))


def _adaptive_config(*, use_xla: bool) -> FixedTransportFullChainConfig:
    return FixedTransportFullChainConfig(
        num_results=8,
        num_burnin_steps=12,
        step_size=0.03,
        num_leapfrog_steps=3,
        seed=(20260822, 401),
        use_xla=use_xla,
        trace_policy="standard",
        target_status_trace_policy="none",
        tuning_policy=FixedTransportHMCPolicy.dual_averaging(
            steps=12, target=0.7, source="step-cap-test"
        ),
        target_scope="step_cap_fixture:fixed_transport",
        chain_execution_mode="tf_function" if use_xla else "eager",
        maximum_candidate_step_size=0.05,
    )


def _adapter() -> tuple[GaussianAdapter, object]:
    base = GaussianAdapter()
    transport = IdentityTransport()
    return (
        build_fixed_transport_value_score_adapter(
            base_adapter=base,
            fixed_transport=transport,
            target_scope="step_cap_fixture:fixed_transport",
            evidence_path=None,
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
        ),
        transport,
    )


def test_step_cap_config_is_positive_and_lineage_visible() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        maximum_candidate_step_size=0.45,
    )
    assert config.payload()["maximum_candidate_step_size"] == pytest.approx(0.45)
    assert config.payload()["maximum_candidate_step_size_role"] == (
        "bayesfilter_mechanics_hard_upper_bound"
    )
    with pytest.raises(ValueError, match="maximum_candidate_step_size"):
        FixedTransportHMCKernelTuningConfig(
            initial_step_size=0.1, maximum_candidate_step_size=0.0
        )


def test_capped_setter_changes_the_step_used_by_inner_hmc() -> None:
    target = lambda value: -0.5 * tf.reduce_sum(tf.square(value), axis=-1)
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target, step_size=tf.constant(0.9, tf.float64), num_leapfrog_steps=3
    )
    kernel_results = hmc.bootstrap_results(tf.zeros((2, 2), tf.float64))
    capped = fixed_transport_capped_step_size_setter(0.2)(
        kernel_results, tf.constant(0.9, tf.float64)
    )
    from tensorflow_probability.python.mcmc import simple_step_size_adaptation as step_api

    applied = step_api.hmc_like_step_size_getter_fn(capped)
    assert float(applied.numpy()) == pytest.approx(0.2)


@pytest.mark.parametrize("use_xla", (False, True))
def test_real_dual_averaging_reports_applied_step_within_cap(use_xla: bool) -> None:
    if use_xla and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        # CPU XLA is still an exact mechanics check; it does not make a GPU claim.
        pass
    adapter, _transport = _adapter()
    result = run_fixed_transport_full_chain_tfp_hmc(
        adapter,
        tf.constant([[0.4, -0.3], [-0.2, 0.1], [0.6, 0.5], [-0.5, -0.4]], tf.float64),
        _adaptive_config(use_xla=use_xla),
    )
    diagnostics = result.diagnostics
    assert diagnostics["step_size_cap_telemetry_complete"] is True
    assert diagnostics["step_size_cap_within_bound"] is True
    assert diagnostics["maximum_candidate_step_size"] == pytest.approx(0.05)
    assert diagnostics["applied_step_size_max"] <= 0.05
    assert np.all(np.asarray(result.trace["applied_step_size"]) <= 0.05)
    assert np.all(np.asarray(result.trace["step_size"]) <= 0.05)


def test_cap_absence_preserves_unbounded_contract_and_records_nonclaim() -> None:
    adapter, _transport = _adapter()
    config = FixedTransportFullChainConfig(
        **{
            **_adaptive_config(use_xla=False).__dict__,
            "maximum_candidate_step_size": None,
            "step_size": 0.1,
            "chain_execution_mode": "eager",
        }
    )
    result = run_fixed_transport_full_chain_tfp_hmc(
        adapter,
        tf.ones((4, 2), tf.float64),
        config,
    )
    assert result.diagnostics["maximum_candidate_step_size"] is None
    assert result.diagnostics["step_size_cap_within_bound"] is None
    assert result.diagnostics["step_size_cap_applied"] is False


def test_adaptive_initial_step_above_cap_fails_closed() -> None:
    adapter, _transport = _adapter()
    config = FixedTransportFullChainConfig(
        **{
            **_adaptive_config(use_xla=False).__dict__,
            "step_size": 0.2,
            "chain_execution_mode": "eager",
        }
    )
    with pytest.raises(ValueError, match="exceeds maximum_candidate_step_size"):
        run_fixed_transport_full_chain_tfp_hmc(
            adapter,
            tf.ones((4, 2), tf.float64),
            config,
        )


def test_reusable_runner_xla_path_applies_cap_without_symbolic_numpy() -> None:
    adapter, _transport = _adapter()
    config = _adaptive_config(use_xla=True)
    pool = FixedTransportReusableRunnerPool()
    result = pool(adapter, tf.ones((4, 2), tf.float64), config)
    assert result.diagnostics["step_size_cap_telemetry_complete"] is True
    assert result.diagnostics["step_size_cap_within_bound"] is True
    assert result.diagnostics["applied_step_size_max"] <= 0.05
    evidence = pool.evidence()
    assert evidence["all_runners_traced_exactly_once"] is True


class MissingCapTelemetryRunner:
    def __call__(self, adapter, initial_state, config):
        state = tf.convert_to_tensor(initial_state, tf.float64)
        shape = (config.num_results, state.shape[0])
        value, score = adapter.log_prob_and_grad(state)
        samples = tf.broadcast_to(state[tf.newaxis, :, :], (config.num_results,) + tuple(state.shape))
        trace = {
            "is_accepted": tf.ones(shape, tf.bool),
            "log_accept_ratio": tf.zeros(shape, tf.float64),
            "target_log_prob": tf.broadcast_to(value[tf.newaxis, :], shape),
            "proposed_target_log_prob": tf.broadcast_to(value[tf.newaxis, :], shape),
            "target_score": tf.broadcast_to(score[tf.newaxis, :, :], (config.num_results,) + tuple(state.shape)),
        }
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics={
                "acceptance_rate": tf.constant(0.7, tf.float64),
                "samples_all_finite": True,
                "log_accept_ratio_finite": True,
                "target_log_prob_finite": True,
                "proposed_target_log_prob_finite": True,
                "target_score_finite": True,
                "final_step_size": tf.constant(0.1, tf.float64),
                "final_step_size_finite": True,
                "divergence_status": "available",
                "divergence_count": 0,
            },
            metadata={"runtime": "missing-cap-telemetry-fixture"},
        )


class CappedFakeHMC(MissingCapTelemetryRunner):
    def __call__(self, adapter, initial_state, config):
        result = super().__call__(adapter, initial_state, config)
        state = tf.convert_to_tensor(initial_state, tf.float64)
        offsets = tf.cast(
            tf.range(1, config.num_results + 1)[:, tf.newaxis, tf.newaxis],
            tf.float64,
        )
        samples = state[tf.newaxis, :, :] + 0.01 * offsets
        trace = dict(result.trace)
        trace["log_accept_ratio"] = tf.fill(
            tf.shape(trace["log_accept_ratio"]), tf.math.log(tf.constant(0.7, tf.float64))
        )
        diagnostics = dict(result.diagnostics)
        cap = config.maximum_candidate_step_size
        diagnostics.update(
            {
                "maximum_candidate_step_size": cap,
                "step_size_cap_telemetry_complete": True,
                "step_size_cap_within_bound": True,
                "step_size_cap_applied": False,
                "requested_step_size_max": float(config.step_size),
                "applied_step_size_max": float(config.step_size),
                "applied_step_size_min": float(config.step_size),
            }
        )
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics=diagnostics,
            metadata=result.metadata,
        )


def test_tuner_fails_closed_when_cap_telemetry_is_missing() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.03,
        maximum_candidate_step_size=0.05,
        leapfrog_grid=(3,),
        chain_count=4,
        budget_schedule=(2,),
        tune_num_results=2,
        screen_num_results=2,
        screen_num_burnin_steps=1,
        verification_num_results=2,
        verification_num_burnin_steps=1,
        acceptance_band=(0.6, 0.8),
        repair_band=(0.5, 0.9),
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="step_cap_fixture:fixed_transport",
    )
    base = GaussianAdapter()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=IdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=MissingCapTelemetryRunner(),
    )
    assert result.passed is False
    assert any("step_size_cap_telemetry" in veto for veto in result.hard_vetoes)


def test_tuner_stops_before_building_runner_for_repaired_step_above_cap() -> None:
    from dataclasses import replace

    class HighAcceptanceRunner(MissingCapTelemetryRunner):
        def __call__(self, adapter, initial_state, config):
            result = super().__call__(adapter, initial_state, config)
            diagnostics = dict(result.diagnostics)
            diagnostics.update(
                {
                    "maximum_candidate_step_size": 0.15,
                    "step_size_cap_telemetry_complete": True,
                    "step_size_cap_within_bound": True,
                    "applied_step_size_max": 0.1,
                }
            )
            if config.tuning_policy.num_adaptation_steps == 0:
                diagnostics["acceptance_rate"] = tf.constant(0.9, tf.float64)
            return replace(result, diagnostics=diagnostics)

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        maximum_candidate_step_size=0.15,
        leapfrog_grid=(3,),
        chain_count=4,
        budget_schedule=(2, 2),
        tune_num_results=2,
        screen_num_results=2,
        screen_num_burnin_steps=1,
        verification_num_results=2,
        verification_num_burnin_steps=1,
        acceptance_band=(0.6, 0.8),
        repair_band=(0.5, 0.9),
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="step_cap_fixture:fixed_transport",
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=HighAcceptanceRunner(),
    )
    assert result.passed is False
    assert "tune_initial_step_size_exceeds_configured_cap" in result.hard_vetoes


def test_successful_handoff_binds_the_configured_cap() -> None:
    from bayesfilter.inference import build_verified_fixed_transport_hmc_handoff_from_tuning_result

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        maximum_candidate_step_size=0.2,
        leapfrog_grid=(3,),
        chain_count=4,
        budget_schedule=(2,),
        tune_num_results=2,
        screen_num_results=2,
        screen_num_burnin_steps=1,
        verification_num_results=2,
        verification_num_burnin_steps=1,
        acceptance_band=(0.6, 0.8),
        repair_band=(0.5, 0.9),
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="step_cap_fixture:fixed_transport",
    )
    base = GaussianAdapter()
    transport = IdentityTransport()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=CappedFakeHMC(),
    )
    assert result.passed is True
    assert result.tuning_scope_payload["maximum_candidate_step_size"] == pytest.approx(0.2)
    assert result.final_kernel_payload["maximum_candidate_step_size"] == pytest.approx(0.2)
    handoff = build_verified_fixed_transport_hmc_handoff_from_tuning_result(
        tuning_result=result,
        base_adapter=base,
        fixed_transport=transport,
    )
    assert handoff.handoff_payload["maximum_candidate_step_size"] == pytest.approx(0.2)


def test_full_chain_signature_changes_when_cap_changes() -> None:
    first = _adaptive_config(use_xla=False).signature_payload()
    second = FixedTransportFullChainConfig(
        **{
            **_adaptive_config(use_xla=False).__dict__,
            "maximum_candidate_step_size": 0.06,
        }
    ).signature_payload()
    assert first != second
