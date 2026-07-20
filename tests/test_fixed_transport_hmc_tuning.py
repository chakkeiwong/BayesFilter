from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
from bayesfilter.inference import (
    FixedTransportHMCKernelTuningConfig,
    ValueScoreCapability,
    tune_fixed_transport_hmc_kernel,
)
from bayesfilter.inference.hmc import FullChainHMCRunResult


class CountingGaussianAdapter:
    parameter_dim = 2

    def __init__(self, *, authority: str = "graph_native") -> None:
        self.authority = authority
        self.shapes: list[tuple[int, ...]] = []

    def adapter_signature(self) -> str:
        return f"counting-gaussian-{self.authority}-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority=self.authority,
            xla_hmc_ready=self.authority == "graph_native",
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="fixed_transport_tuning_fixture",
            target_scope="gaussian_fixture",
            nonclaims=("tiny fixed-transport tuning fixture only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        self.shapes.append(tuple(values.shape.as_list()))
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values


class CountingIdentityTransport:
    parameter_dim = 2

    def __init__(self) -> None:
        self.batch_calls = 0
        self.scalar_calls = 0

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema": "counting_identity_transport.v1",
            "parameter_dim": self.parameter_dim,
            "kind": "identity",
        }

    def forward(self, z: tf.Tensor) -> tf.Tensor:
        self.scalar_calls += 1
        return tf.convert_to_tensor(z, dtype=tf.float64)

    def forward_batch(self, z_batch: tf.Tensor) -> tf.Tensor:
        self.batch_calls += 1
        return tf.convert_to_tensor(z_batch, dtype=tf.float64)

    def log_abs_det_jacobian(self, z: tf.Tensor) -> tf.Tensor:
        del z
        return tf.constant(0.0, dtype=tf.float64)

    def log_abs_det_jacobian_batch(self, z_batch: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        return tf.zeros(tf.shape(values)[:1], dtype=tf.float64)

    def pullback_score(self, z: tf.Tensor, theta_score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.convert_to_tensor(theta_score, dtype=tf.float64)

    def pullback_score_batch(
        self, z_batch: tf.Tensor, theta_score_batch: tf.Tensor
    ) -> tf.Tensor:
        del z_batch
        return tf.convert_to_tensor(theta_score_batch, dtype=tf.float64)

    def log_abs_det_jacobian_score(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(z, dtype=tf.float64))

    def log_abs_det_jacobian_score_batch(self, z_batch: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(z_batch, dtype=tf.float64))


class FakeHMC:
    def __init__(self, *, acceptance: float = 0.72, final_step: float = 0.17) -> None:
        self.acceptance = float(acceptance)
        self.final_step = float(final_step)
        self.calls: list[dict[str, object]] = []

    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        state = tf.convert_to_tensor(initial_state, dtype=tf.float64)
        value, _score = adapter.log_prob_and_grad(state)
        state_shape = tuple(state.shape.as_list())
        assert len(state_shape) == 2
        self.calls.append(
            {
                "state_shape": state_shape,
                "num_results": config.num_results,
                "num_burnin_steps": config.num_burnin_steps,
                "num_leapfrog_steps": config.num_leapfrog_steps,
                "tuning_policy": config.tuning_policy.label,
                "target_scope": config.target_scope,
            }
        )
        samples = tf.zeros((config.num_results,) + state_shape, dtype=tf.float64)
        trace_shape = (config.num_results, state_shape[0])
        trace = {
            "is_accepted": tf.ones(trace_shape, dtype=tf.bool),
            "log_accept_ratio": tf.fill(
                trace_shape,
                tf.math.log(tf.constant(self.acceptance, tf.float64)),
            ),
            "target_log_prob": tf.broadcast_to(
                tf.reshape(tf.convert_to_tensor(value, dtype=tf.float64), (1, state_shape[0])),
                trace_shape,
            ),
        }
        diagnostics = {
            "acceptance_rate": tf.constant(self.acceptance, dtype=tf.float64),
            "finite_sample_count": tf.size(samples),
            "nonfinite_sample_count": tf.constant(0, dtype=tf.int32),
            "final_step_size": tf.constant(self.final_step, dtype=tf.float64),
            "final_step_size_finite": tf.constant(True),
            "target_accept_prob": None
            if config.tuning_policy.target_accept_prob is None
            else tf.constant(config.tuning_policy.target_accept_prob, dtype=tf.float64),
            "num_adaptation_steps": tf.constant(config.tuning_policy.num_adaptation_steps, dtype=tf.int32),
            "trace_policy": config.trace_policy,
            "divergence_status": "not_available",
            "divergence_count": tf.constant(0, dtype=tf.int32),
        }
        metadata = {
            "runtime": "fake_rank2_hmc_runner",
            "initial_state_shape": state_shape,
            "target_scope": config.target_scope,
            "windowed_mass_adaptation_used": False,
        }
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )


class StepSensitiveFakeHMC(FakeHMC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        self.acceptance = 0.95 if float(config.step_size) < 4.0 else 0.70
        self.final_step = float(config.step_size)
        return super().__call__(adapter, initial_state, config)


class ArchiveFakeHMC(FakeHMC):
    def __init__(self, *, folded_scale_mismatch: bool = False) -> None:
        super().__init__(acceptance=0.70, final_step=0.2)
        self.folded_scale_mismatch = bool(folded_scale_mismatch)

    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        if int(config.num_results) != 1000:
            return result
        draws = tf.random.stateless_normal(
            (1000, 4, 2), seed=(20260713, 91), dtype=tf.float64
        )
        if self.folded_scale_mismatch:
            draws = draws * tf.constant([0.5, 1.0, 2.0, 3.0], tf.float64)[
                tf.newaxis, :, tf.newaxis
            ]
        return FullChainHMCRunResult(
            samples=draws,
            trace=result.trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )


class InvalidTelemetryFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        diagnostics = dict(result.diagnostics)
        diagnostics["target_status_telemetry"] = {
            "telemetry_failure_veto": tf.constant(True),
            "all_status_valid": tf.constant(False),
            "status_nonvalid_count": tf.constant(1, dtype=tf.int32),
        }
        return FullChainHMCRunResult(
            samples=result.samples,
            trace=result.trace,
            diagnostics=diagnostics,
            metadata=result.metadata,
        )


class DivergentFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        diagnostics = dict(result.diagnostics)
        diagnostics["divergence_status"] = "available"
        diagnostics["divergence_count"] = tf.constant(1, dtype=tf.int32)
        return FullChainHMCRunResult(
            samples=result.samples,
            trace=result.trace,
            diagnostics=diagnostics,
            metadata=result.metadata,
        )


class EnergyTailFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        trace = dict(result.trace)
        trace["log_accept_ratio"] = tf.fill(
            tf.shape(trace["log_accept_ratio"]),
            tf.constant(-1001.0, tf.float64),
        )
        return FullChainHMCRunResult(
            samples=result.samples,
            trace=trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )


def _config() -> FixedTransportHMCKernelTuningConfig:
    return FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.11,
        leapfrog_grid=(5, 7),
        chain_count=4,
        budget_schedule=(3,),
        tune_num_results=2,
        screen_num_results=2,
        screen_num_burnin_steps=1,
        verification_num_results=2,
        verification_num_burnin_steps=1,
        acceptance_band=(0.60, 0.85),
        repair_band=(0.50, 0.95),
        chain_execution_mode="eager",
        target_scope="gaussian_fixture_fixed_transport",
    )


def test_fixed_transport_defaults_match_owner_acceptance_policy() -> None:
    config = FixedTransportHMCKernelTuningConfig(initial_step_size=0.1)
    assert config.target_accept_prob == 0.70
    assert config.acceptance_band == (0.65, 0.75)
    assert config.repair_band == (0.55, 0.85)
    assert config.maximum_absolute_energy_error == 1000.0


def _modern_config(**overrides) -> FixedTransportHMCKernelTuningConfig:
    values = {
        "initial_step_size": 0.2,
        "leapfrog_grid": (5,),
        "chain_count": 4,
        "budget_schedule": (3,),
        "tune_num_results": 2,
        "screen_num_results": 2,
        "screen_num_burnin_steps": 1,
        "verification_num_results": 1000,
        "verification_num_burnin_steps": 10,
        "require_modern_rank_normalized_verification": True,
        "verification_min_retained_results_per_chain": 1000,
        "verification_rhat_max": 1.01,
        "acceptance_band": (0.65, 0.75),
        "repair_band": (0.50, 0.95),
        "chain_execution_mode": "eager",
        "target_scope": "gaussian_fixture_fixed_transport",
        "fixed_grid_base_step_size_candidates": (0.2,),
        "fixed_grid_scale_candidates": (1.0,),
        "fixed_grid_num_leapfrog_steps": 5,
        "fixed_grid_max_attempts": 1,
    }
    values.update(overrides)
    return FixedTransportHMCKernelTuningConfig(**values)


def test_fixed_transport_hmc_tuner_selects_frozen_identity_z_kernel(tmp_path: Path) -> None:
    base = CountingGaussianAdapter()
    transport = CountingIdentityTransport()
    fake_hmc = FakeHMC()

    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=np.zeros(2),
        config=_config(),
        output_dir=tmp_path,
        run_full_chain=fake_hmc,
    )

    assert result.passed
    assert result.artifact_path is not None
    assert Path(result.artifact_path).exists()
    artifact_payload = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    assert artifact_payload["artifact_path"] == result.artifact_path
    assert result.selected_candidate is not None
    payload = result.final_kernel_payload
    assert payload is not None
    assert payload["runtime"] == "bayesfilter.inference.tune_fixed_transport_hmc_kernel"
    assert payload["mass_policy"] == "fixed_identity_z"
    assert payload["windowed_mass_adaptation_used"] is False
    assert payload["mass_adaptation_used"] is False
    assert payload["rank2_chain_batched_target_required"] is True
    assert payload["proposal_dynamics_identity"] == "exact_transformed_gradient"
    mass_payload = payload["identity_z_mass_artifact_payload"]
    assert mass_payload["position_role"] == "fixed_neutra_initial_z"
    assert mass_payload["covariance_source"] == "fixed_identity_z"
    assert mass_payload["matrix_used_for_square_root"] == "identity_z"
    assert "windowed_stage_artifact_hash" not in payload
    assert {call["state_shape"] for call in fake_hmc.calls} == {(4, 2)}
    assert transport.batch_calls > 0
    assert transport.scalar_calls == 0
    assert all(shape == (4, 2) for shape in base.shapes)


def test_fixed_transport_hmc_tuner_public_imports() -> None:
    assert bayesfilter.tune_fixed_transport_hmc_kernel is tune_fixed_transport_hmc_kernel
    assert bayesfilter.FixedTransportHMCKernelTuningConfig is FixedTransportHMCKernelTuningConfig
    assert "tune_fixed_transport_hmc_kernel" in bayesfilter.__all__


def test_fixed_transport_hmc_tuner_forbids_gradient_tape_fallback() -> None:
    with pytest.raises(ValueError, match="gradient_tape_fallback"):
        tune_fixed_transport_hmc_kernel(
            base_adapter=CountingGaussianAdapter(authority="gradient_tape_fallback"),
            fixed_transport=CountingIdentityTransport(),
            initial_position=np.zeros(2),
            config=_config(),
            run_full_chain=FakeHMC(),
        )


def test_fixed_transport_hmc_tuner_records_no_viable_candidate() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_config(),
        run_full_chain=FakeHMC(acceptance=0.99),
    )

    assert not result.passed
    assert result.final_status == "no_viable_candidate"
    assert result.final_kernel_payload is None
    assert result.selected_candidate_index is None
    assert any(
        "screen_acceptance_above_repair_band" in candidate.repair_triggers
        or "screen_acceptance_outside_repair_band" in candidate.hard_vetoes
        or "verification_acceptance_outside_repair_band" in candidate.hard_vetoes
        for candidate in result.candidates
    )


def test_fixed_transport_hmc_tuner_runs_bayesfilter_fixed_grid_scale_repair() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.11,
        leapfrog_grid=(5, 7),
        chain_count=4,
        budget_schedule=(3,),
        tune_num_results=2,
        screen_num_results=2,
        screen_num_burnin_steps=1,
        verification_num_results=2,
        verification_num_burnin_steps=1,
        acceptance_band=(0.65, 0.75),
        repair_band=(0.50, 0.95),
        chain_execution_mode="eager",
        target_scope="gaussian_fixture_fixed_transport",
        fixed_grid_base_step_size_candidates=(0.05, 0.5),
        fixed_grid_scale_candidates=(0.1, 0.2, 1.0, 5.0, 9.0),
        fixed_grid_num_leapfrog_steps=5,
        fixed_grid_max_attempts=5,
        fixed_grid_fallback_acceptance_max=0.85,
    )

    fake_hmc = StepSensitiveFakeHMC()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=fake_hmc,
    )

    assert result.passed
    assert result.fixed_grid_scale_selection_payload is not None
    scale_payload = result.fixed_grid_scale_selection_payload
    assert scale_payload["artifact_type"] == (
        "bayesfilter_fixed_transport_hmc_grid_scale_repair"
    )
    assert scale_payload["status"] == "accepted_in_band"
    assert scale_payload["selected_scale"] == 9.0
    assert len(scale_payload["attempts"]) == 5
    assert scale_payload["attempts"][-1]["acceptance_class"] == "in_band"
    assert len(result.candidates) == 1
    candidate_payload = result.candidates[0].payload()
    assert candidate_payload["handoff_source"] == "fixed_grid_scale_probe"
    assert candidate_payload["ladder"] is None
    assert candidate_payload["fixed_kernel_step_size"] == 4.5
    assert candidate_payload["selected_step_size"] == 4.5
    assert [call["tuning_policy"] for call in fake_hmc.calls] == [
        "fixed_kernel_screen",
    ] * 6
    assert [call["num_results"] for call in fake_hmc.calls[:-1]] == [2] * 5
    assert fake_hmc.calls[-1]["num_results"] == config.verification_num_results
    assert fake_hmc.calls[-1]["num_burnin_steps"] == (
        config.verification_num_burnin_steps
    )
    assert all(
        call["num_leapfrog_steps"] == config.fixed_grid_num_leapfrog_steps
        for call in fake_hmc.calls
    )
    assert result.final_kernel_payload is not None
    assert result.final_kernel_payload["step_size"] == 4.5


def test_modern_verification_requires_four_chains_and_1000_draws() -> None:
    with pytest.raises(ValueError, match="exactly four chains"):
        _modern_config(chain_count=3)
    with pytest.raises(ValueError, match="at least the configured retained"):
        _modern_config(verification_num_results=999)


def test_modern_verification_passes_from_real_iid_archive() -> None:
    fake_hmc = ArchiveFakeHMC()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(),
        run_full_chain=fake_hmc,
    )

    assert result.passed
    assert result.final_kernel_payload is not None
    diagnostics = result.final_kernel_payload["verification_diagnostics"]
    modern = diagnostics["modern_rank_normalized_verification"]
    assert diagnostics["sample_shape"] == (1000, 4, 2)
    assert diagnostics["modern_verification_coordinate_system"] == (
        "raw_target_coordinates"
    )
    assert modern["passed"] is True
    assert modern["rhat_definition"] == (
        "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
    )
    assert [call["num_results"] for call in fake_hmc.calls] == [2, 1000]


def test_modern_verification_folded_rhat_vetoes_in_band_acceptance() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(),
        run_full_chain=ArchiveFakeHMC(folded_scale_mismatch=True),
    )

    assert not result.passed
    assert result.final_kernel_payload is None
    assert len(result.candidates) == 1
    assert result.candidates[0].final_status == "hard_veto"
    scale = result.fixed_grid_scale_selection_payload
    assert scale is not None
    attempt = scale["attempts"][0]
    assert attempt["probe_diagnostics"]["modern_rank_normalized_verification"] is None
    assert attempt["pilot_acceptance_rate"] == pytest.approx(0.70)
    modern = result.candidates[0].verification_diagnostics[
        "modern_rank_normalized_verification"
    ]
    assert modern["max_rank_normalized_split_rhat"] < 1.01
    assert modern["max_folded_rank_normalized_split_rhat"] > 1.01
    assert "verification_modern_rank_folded_rhat_failed" in result.candidates[
        0
    ].hard_vetoes


def test_verification_target_status_failure_is_a_hard_veto() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(target_status_trace_policy="per_chain_step"),
        run_full_chain=InvalidTelemetryFakeHMC(),
    )

    assert not result.passed
    assert len(result.candidates) == 0
    scale = result.fixed_grid_scale_selection_payload
    assert scale is not None
    assert "verification_target_status_telemetry_failure" in scale["attempts"][0][
        "probe_hard_vetoes"
    ]


def test_native_divergence_is_a_hard_veto() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(),
        run_full_chain=DivergentFakeHMC(),
    )

    assert not result.passed
    scale = result.fixed_grid_scale_selection_payload
    assert scale is not None
    assert "verification_native_divergence_detected" in scale["attempts"][0][
        "probe_hard_vetoes"
    ]


def test_untruncated_energy_tail_is_a_hard_veto_before_promotion() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(),
        run_full_chain=EnergyTailFakeHMC(acceptance=0.70),
    )
    assert not result.passed
    scale = result.fixed_grid_scale_selection_payload
    assert scale is not None
    assert "verification_energy_error_exceeds_limit" in scale["attempts"][0][
        "probe_hard_vetoes"
    ]
