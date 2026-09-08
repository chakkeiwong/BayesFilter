from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
import bayesfilter.inference.fixed_transport_hmc_tuning_tf as fixed_tuning
from bayesfilter.inference import (
    FixedTransportHMCKernelTuningConfig,
    FixedTransportReusableRunnerPool,
    ValueScoreCapability,
    build_verified_fixed_transport_hmc_handoff_from_tuning_result,
    tune_fixed_transport_hmc_kernel,
)
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
    FixedTransportFullChainConfig,
    FixedTransportHMCPolicy,
    build_fixed_transport_value_score_adapter,
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

    def __init__(self, *, manifest_tag: str = "default") -> None:
        self.batch_calls = 0
        self.scalar_calls = 0
        self.manifest_tag = str(manifest_tag)

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema": "counting_identity_transport.v1",
            "parameter_dim": self.parameter_dim,
            "kind": "identity",
            "manifest_tag": self.manifest_tag,
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
                "initial_state": state.numpy().tolist(),
                "num_results": config.num_results,
                "num_burnin_steps": config.num_burnin_steps,
                "num_leapfrog_steps": config.num_leapfrog_steps,
                "tuning_policy": config.tuning_policy.label,
                "target_scope": config.target_scope,
                "seed": tuple(int(value) for value in config.seed),
            }
        )
        draw_offset = tf.cast(
            tf.range(1, config.num_results + 1)[:, tf.newaxis, tf.newaxis],
            tf.float64,
        )
        chain_scale = tf.cast(
            tf.range(1, state_shape[0] + 1)[tf.newaxis, :, tf.newaxis],
            tf.float64,
        )
        samples = state[tf.newaxis, :, :] + 0.01 * draw_offset * chain_scale
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
            "proposed_target_log_prob": tf.broadcast_to(
                tf.reshape(tf.convert_to_tensor(value, dtype=tf.float64), (1, state_shape[0])),
                trace_shape,
            ),
            "target_score": tf.broadcast_to(
                tf.convert_to_tensor(_score, dtype=tf.float64)[tf.newaxis, :, :],
                (config.num_results, state_shape[0], state_shape[1]),
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
            "divergence_status": "available",
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


class StuckFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        state = tf.convert_to_tensor(initial_state, tf.float64)
        samples = tf.broadcast_to(
            state[tf.newaxis, :, :],
            (config.num_results,) + tuple(state.shape.as_list()),
        )
        return FullChainHMCRunResult(
            samples=samples,
            trace=result.trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )


class MoveThenReturnFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        state = tf.convert_to_tensor(initial_state, tf.float64)
        first = state[tf.newaxis, :, :] + 0.25
        remainder = tf.broadcast_to(
            state[tf.newaxis, :, :],
            (max(config.num_results - 1, 0),) + tuple(state.shape.as_list()),
        )
        return FullChainHMCRunResult(
            samples=tf.concat((first, remainder), axis=0),
            trace=result.trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )


class EfficiencyRankingFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        if int(config.num_results) != 64 or int(config.num_burnin_steps) != 32:
            return result
        state = tf.convert_to_tensor(initial_state, tf.float64)
        noise = tf.random.stateless_normal(
            (64,) + tuple(state.shape.as_list()),
            seed=(20260821, int(config.num_leapfrog_steps)),
            dtype=tf.float64,
        )
        draws = tf.cumsum(noise, axis=0) if config.num_leapfrog_steps == 5 else noise
        return FullChainHMCRunResult(
            samples=state[tf.newaxis, :, :] + draws,
            trace=result.trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )


class LeapfrogAcceptanceFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        self.acceptance = 0.99 if config.num_leapfrog_steps == 5 else 0.72
        return super().__call__(adapter, initial_state, config)


class HeldoutFailureFakeHMC(EfficiencyRankingFakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        self.acceptance = (
            0.99
            if int(config.num_results) == 10
            and int(config.num_burnin_steps) == 5
            else 0.72
        )
        return super().__call__(adapter, initial_state, config)


class AllLeapfrogEfficiencyFakeHMC(FakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        self.acceptance = 0.70
        self.final_step = 0.1 + 0.001 * int(config.num_leapfrog_steps)
        result = super().__call__(adapter, initial_state, config)
        if int(config.num_results) != 16 or int(config.num_burnin_steps) != 8:
            return result
        state = tf.convert_to_tensor(initial_state, tf.float64)
        draws = tf.random.stateless_normal(
            (16,) + tuple(state.shape.as_list()),
            seed=config.seed,
            dtype=tf.float64,
        )
        return FullChainHMCRunResult(
            samples=state[tf.newaxis, :, :] + draws,
            trace=result.trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )


class AcceptanceRepairFakeHMC(FakeHMC):
    def __init__(self, *, direction: str) -> None:
        super().__init__()
        self.direction = str(direction)

    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        step = float(config.step_size)
        self.final_step = step
        if self.direction == "lower":
            self.acceptance = 0.40 if step >= 0.1 else 0.70
        elif self.direction == "higher":
            self.acceptance = 0.90 if step <= 0.1 else 0.70
        else:
            raise AssertionError("invalid repair direction fixture")
        return super().__call__(adapter, initial_state, config)


class MissingSelectionTelemetryFakeHMC(EfficiencyRankingFakeHMC):
    def __init__(self, *, missing_key: str) -> None:
        super().__init__()
        self.missing_key = str(missing_key)

    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        if int(config.num_results) != 64 or int(config.num_burnin_steps) != 32:
            return result
        trace = dict(result.trace)
        trace.pop(self.missing_key)
        return FullChainHMCRunResult(
            samples=result.samples,
            trace=trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
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


class UnavailableDivergenceFakeHMC(ArchiveFakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        diagnostics = dict(result.diagnostics)
        diagnostics["divergence_status"] = "not_exposed_by_kernel"
        diagnostics["divergence_count"] = None
        return FullChainHMCRunResult(
            samples=result.samples,
            trace=result.trace,
            diagnostics=diagnostics,
            metadata=result.metadata,
        )


class EnergyTailFakeHMC(ArchiveFakeHMC):
    def __call__(self, adapter, initial_state, config) -> FullChainHMCRunResult:
        result = super().__call__(adapter, initial_state, config)
        trace = dict(result.trace)
        if int(config.num_results) >= 1000:
            trace["log_accept_ratio"] = tf.tensor_scatter_nd_update(
                trace["log_accept_ratio"],
                indices=((0, 0),),
                updates=(tf.constant(-1001.0, tf.float64),),
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
        use_xla=False,
        target_scope="gaussian_fixture_fixed_transport",
    )


def test_fixed_transport_defaults_match_owner_acceptance_policy() -> None:
    config = FixedTransportHMCKernelTuningConfig(initial_step_size=0.1)
    assert config.target_accept_prob == 0.70
    assert config.acceptance_band == (0.65, 0.75)
    assert config.repair_band == (0.55, 0.85)
    assert config.maximum_absolute_energy_error == 1000.0
    assert config.use_xla is True


@pytest.mark.parametrize(
    "overrides",
    (
        {"leapfrog_grid": (1, 5)},
        {
            "fixed_grid_base_step_size_candidates": (0.1,),
            "fixed_grid_scale_candidates": (1.0,),
            "fixed_grid_num_leapfrog_steps": 1,
        },
    ),
)
def test_fixed_transport_tuning_forbids_one_leapfrog_step(overrides) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        FixedTransportHMCKernelTuningConfig(initial_step_size=0.1, **overrides)


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
        "use_xla": False,
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


def test_fixed_transport_hmc_tuner_passes_through_declared_resource_stop(
    tmp_path: Path,
) -> None:
    class CampaignResourceStop(RuntimeError):
        pass

    def resource_stop(_adapter, _initial_state, _config):
        raise CampaignResourceStop("fixture resource refusal")

    with pytest.raises(CampaignResourceStop, match="fixture resource refusal"):
        tune_fixed_transport_hmc_kernel(
            base_adapter=CountingGaussianAdapter(),
            fixed_transport=CountingIdentityTransport(),
            initial_position=np.zeros(2),
            config=_config(),
            output_dir=tmp_path,
            run_full_chain=resource_stop,
            passthrough_exceptions=(CampaignResourceStop,),
        )

    assert not (tmp_path / _config().output_filename).exists()


def test_reusable_pool_resource_hook_propagates_before_compilation(
    tmp_path: Path,
) -> None:
    class CampaignResourceStop(RuntimeError):
        pass

    def resource_stop(_adapter, _initial_state, _config) -> None:
        raise CampaignResourceStop("pool resource refusal")

    pool = FixedTransportReusableRunnerPool(before_run=resource_stop)
    config = replace(_config(), chain_execution_mode="tf_function")
    with pytest.raises(CampaignResourceStop, match="pool resource refusal"):
        tune_fixed_transport_hmc_kernel(
            base_adapter=CountingGaussianAdapter(),
            fixed_transport=CountingIdentityTransport(),
            initial_position=np.zeros(2),
            config=config,
            output_dir=tmp_path,
            run_full_chain=pool,
            passthrough_exceptions=(CampaignResourceStop,),
        )

    assert pool.evidence()["runner_count"] == 0
    assert not (tmp_path / config.output_filename).exists()


def test_fixed_transport_hmc_tuner_still_converts_undeclared_runtime_error() -> None:
    def runtime_error(_adapter, _initial_state, _config):
        raise RuntimeError("fixture numerical runtime error")

    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_config(),
        run_full_chain=runtime_error,
    )

    assert result.passed is False
    assert "tune_samples_nonfinite_or_missing" in result.hard_vetoes


def test_fixed_transport_hmc_tuner_uses_and_records_explicit_initial_state_bank() -> None:
    bank = ((-1.0, 0.2), (-0.8, -0.1), (0.9, 0.3), (1.1, -0.2))
    fake_hmc = FakeHMC()
    config = FixedTransportHMCKernelTuningConfig(
        **{**_config().__dict__, "initial_state_bank": bank}
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.asarray(bank[0]),
        config=config,
        run_full_chain=fake_hmc,
    )

    assert result.passed
    assert fake_hmc.calls
    assert all(call["initial_state"] == [list(row) for row in bank] for call in fake_hmc.calls)
    diagnostics = result.selected_candidate.verification_diagnostics
    assert diagnostics["initial_state_all_zero"] is False
    assert diagnostics["initial_state_bank"] == [list(row) for row in bank]
    assert result.identity_z_mass_artifact_payload["position"] == list(bank[0])


def test_fixed_transport_hmc_tuner_broadcasts_nonzero_initial_position() -> None:
    fake_hmc = FakeHMC()
    start = np.asarray([0.4, -0.7])
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=start,
        config=_config(),
        run_full_chain=fake_hmc,
    )

    assert result.passed
    expected = [start.tolist()] * 4
    assert all(call["initial_state"] == expected for call in fake_hmc.calls)
    assert result.identity_z_mass_artifact_payload["position"] == start.tolist()


def test_nominal_acceptance_cannot_promote_stuck_chains() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_config(),
        run_full_chain=StuckFakeHMC(),
    )

    assert not result.passed
    assert "verification_chain_without_movement" in result.hard_vetoes


def test_move_then_return_to_initial_state_counts_as_movement() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_config(),
        run_full_chain=MoveThenReturnFakeHMC(),
    )

    assert result.passed
    diagnostics = result.selected_candidate.verification_diagnostics
    assert diagnostics["all_chains_moved"] is True
    assert all(value > 0.0 for value in diagnostics["maximum_displacement_by_chain"])


def test_dual_averaging_does_not_validate_fixed_grid_fallback_against_pass_band() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        acceptance_band=(0.65, 0.90),
        repair_band=(0.35, 0.95),
    )
    assert config.fixed_grid_fallback_acceptance_max == 0.85
    assert config.payload()["fixed_grid_fallback_acceptance_max_role"] == (
        "not_applicable_to_dual_averaging"
    )

    with pytest.raises(ValueError, match="contain the pass-band upper bound"):
        FixedTransportHMCKernelTuningConfig(
            initial_step_size=0.1,
            acceptance_band=(0.65, 0.90),
            repair_band=(0.35, 0.95),
            fixed_grid_base_step_size_candidates=(0.1,),
            fixed_grid_scale_candidates=(1.0,),
        )


def test_replicated_efficiency_policy_screens_every_ladder_nominee_then_holds_out_winner() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        **{
            **_config().__dict__,
            "selection_policy": "replicated_min_bulk_ess_per_gradient",
            "selection_replications": 1,
            "selection_num_results": 64,
            "selection_num_burnin_steps": 32,
            "selection_acceptance_band": (0.35, 0.95),
        }
    )
    fake_hmc = EfficiencyRankingFakeHMC()
    base = CountingGaussianAdapter()
    transport = CountingIdentityTransport()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=fake_hmc,
    )

    assert result.passed
    assert result.selected_candidate_index == 1
    rows = result.candidate_selection_payload["candidate_rows"]
    assert [row["candidate_index"] for row in rows if row["selection_evidence"]] == [
        0,
        1,
    ]
    assert all(row["eligible"] for row in rows)
    selection_calls = [
        call
        for call in fake_hmc.calls
        if call["num_results"] == 64 and call["num_burnin_steps"] == 32
    ]
    assert [call["num_leapfrog_steps"] for call in selection_calls] == [5, 7]
    selection = result.candidate_selection_payload
    assert selection["candidate_verification_serves_as_final"] is False
    assert selection["post_selection_candidate_only_verification_used"] is True
    heldout = selection["heldout_verification"]
    assert heldout["final_status"] == "passed"
    assert heldout["diagnostic_role"] == "post_selection_heldout_verification"
    assert tuple(heldout["config_payload"]["seed"]) not in {
        tuple(replication["config_payload"]["seed"])
        for row in rows
        for replication in row["selection_evidence"]["diagnostics"]["replications"]
    }
    assert result.final_kernel_payload["post_selection_candidate_only_verification_used"] is True
    assert result.final_kernel_payload["verification_diagnostics"] == heldout["diagnostics"]

    handoff = build_verified_fixed_transport_hmc_handoff_from_tuning_result(
        tuning_result=result,
        base_adapter=base,
        fixed_transport=transport,
    )
    assert handoff.num_leapfrog_steps == 7
    assert handoff.transformed_adapter.adapter_signature() == (
        result.transformed_adapter_signature
    )


def test_replicated_efficiency_traverses_all_l_values_and_tunes_epsilon_independently() -> None:
    leapfrogs = (3, 5, 9, 13, 18, 25)
    config = FixedTransportHMCKernelTuningConfig(
        **{
            **_config().__dict__,
            "leapfrog_grid": leapfrogs,
            "selection_policy": "replicated_min_bulk_ess_per_gradient",
            "selection_replications": 3,
            "selection_num_results": 16,
            "selection_num_burnin_steps": 8,
            "selection_acceptance_band": (0.65, 0.75),
            "verification_num_results": 20,
            "verification_num_burnin_steps": 10,
        }
    )
    fake_hmc = AllLeapfrogEfficiencyFakeHMC()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=fake_hmc,
    )

    assert result.passed
    assert tuple(candidate.num_leapfrog_steps for candidate in result.candidates) == leapfrogs
    assert tuple(candidate.selected_step_size for candidate in result.candidates) == tuple(
        0.1 + 0.001 * leapfrog for leapfrog in leapfrogs
    )
    adaptation_calls = [
        call for call in fake_hmc.calls if call["tuning_policy"] == "fixed_mass_dual_averaging"
    ]
    assert [call["num_leapfrog_steps"] for call in adaptation_calls] == list(leapfrogs)
    selection_calls = [
        call
        for call in fake_hmc.calls
        if call["num_results"] == 16 and call["num_burnin_steps"] == 8
    ]
    assert len(selection_calls) == 3 * len(leapfrogs)
    assert {
        leapfrog: sum(
            call["num_leapfrog_steps"] == leapfrog for call in selection_calls
        )
        for leapfrog in leapfrogs
    } == {leapfrog: 3 for leapfrog in leapfrogs}
    heldout_calls = [
        call
        for call in fake_hmc.calls
        if call["num_results"] == 20 and call["num_burnin_steps"] == 10
    ]
    assert len(heldout_calls) == 1
    seed_ledger = result.candidate_selection_payload["seed_ledger"]
    assert seed_ledger["all_seeds_unique"] is True
    assert seed_ledger["seed_count"] == len(fake_hmc.calls)
    assert len({call["seed"] for call in fake_hmc.calls}) == len(fake_hmc.calls)


def test_failed_post_selection_holdout_suppresses_kernel_without_trying_runner_up(
    tmp_path: Path,
) -> None:
    config = FixedTransportHMCKernelTuningConfig(
        **{
            **_config().__dict__,
            "selection_policy": "replicated_min_bulk_ess_per_gradient",
            "selection_replications": 1,
            "selection_num_results": 64,
            "selection_num_burnin_steps": 32,
            "selection_acceptance_band": (0.35, 0.95),
            "verification_num_results": 10,
            "verification_num_burnin_steps": 5,
        }
    )
    fake_hmc = HeldoutFailureFakeHMC()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        output_dir=tmp_path,
        run_full_chain=fake_hmc,
    )

    selection = result.candidate_selection_payload
    assert result.passed is False
    assert result.final_status == "heldout_verification_failed"
    assert selection["nominated_candidate_index"] == 1
    assert selection["selected_candidate_index"] is None
    assert selection["heldout_verification"]["final_status"] == "hard_veto"
    assert result.final_kernel_payload is None
    assert result.final_kernel_hash is None
    heldout_calls = [
        call
        for call in fake_hmc.calls
        if call["num_results"] == 10 and call["num_burnin_steps"] == 5
    ]
    assert len(heldout_calls) == 1
    assert heldout_calls[0]["num_leapfrog_steps"] == 7
    artifact = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    assert artifact["final_kernel_payload"] is None
    assert artifact["final_kernel_hash"] is None
    with pytest.raises(ValueError, match="did not authorize"):
        build_verified_fixed_transport_hmc_handoff_from_tuning_result(
            tuning_result=result,
            base_adapter=CountingGaussianAdapter(),
            fixed_transport=CountingIdentityTransport(),
        )


@pytest.mark.parametrize("missing_key", ("proposed_target_log_prob", "target_score"))
def test_selection_requires_complete_target_value_and_score_telemetry(
    missing_key: str,
) -> None:
    config = FixedTransportHMCKernelTuningConfig(
        **{
            **_config().__dict__,
            "selection_policy": "replicated_min_bulk_ess_per_gradient",
            "selection_replications": 1,
            "selection_num_results": 64,
            "selection_num_burnin_steps": 32,
            "selection_acceptance_band": (0.35, 0.95),
        }
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=MissingSelectionTelemetryFakeHMC(missing_key=missing_key),
    )

    assert result.passed is False
    assert result.candidate_selection_payload["nominated_candidate_index"] is None
    assert any(missing_key in veto for veto in result.hard_vetoes)


@pytest.mark.parametrize(
    ("direction", "expected_second_step", "expected_trigger"),
    (
        ("lower", 0.05, "screen_acceptance_below_repair_band"),
        ("higher", 0.2, "screen_acceptance_above_repair_band"),
    ),
)
def test_dual_averaging_screen_repairs_epsilon_in_the_correct_direction(
    direction: str,
    expected_second_step: float,
    expected_trigger: str,
) -> None:
    config = FixedTransportHMCKernelTuningConfig(
        **{
            **_config().__dict__,
            "initial_step_size": 0.1,
            "leapfrog_grid": (5,),
            "budget_schedule": (3, 6),
            "acceptance_band": (0.65, 0.75),
            "repair_band": (0.55, 0.85),
        }
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=AcceptanceRepairFakeHMC(direction=direction),
    )

    assert result.passed
    rounds = result.candidates[0].ladder_result["rounds"]
    assert rounds[0]["initial_step_size"] == 0.1
    assert rounds[0]["repair_triggers"] == (expected_trigger,)
    assert rounds[1]["initial_step_size"] == expected_second_step


def test_resource_stop_inside_efficiency_selection_is_not_scientific_failure() -> None:
    class CampaignResourceStop(RuntimeError):
        pass

    config = FixedTransportHMCKernelTuningConfig(
        **{
            **_config().__dict__,
            "selection_policy": "replicated_min_bulk_ess_per_gradient",
            "selection_num_results": 64,
            "selection_num_burnin_steps": 32,
            "selection_acceptance_band": (0.35, 0.95),
        }
    )
    fake_hmc = FakeHMC()

    def stop_at_selection(adapter, initial_state, run_config):
        if run_config.num_results == 64 and run_config.num_burnin_steps == 32:
            raise CampaignResourceStop("selection compute ceiling")
        return fake_hmc(adapter, initial_state, run_config)

    with pytest.raises(CampaignResourceStop, match="selection compute ceiling"):
        tune_fixed_transport_hmc_kernel(
            base_adapter=CountingGaussianAdapter(),
            fixed_transport=CountingIdentityTransport(),
            initial_position=np.zeros(2),
            config=config,
            run_full_chain=stop_at_selection,
            passthrough_exceptions=(CampaignResourceStop,),
        )


def test_rejected_candidate_never_runs_efficiency_selection() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        **{
            **_config().__dict__,
            "selection_policy": "replicated_min_bulk_ess_per_gradient",
            "selection_num_results": 64,
            "selection_num_burnin_steps": 32,
            "selection_acceptance_band": (0.35, 0.95),
        }
    )
    fake_hmc = LeapfrogAcceptanceFakeHMC()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=fake_hmc,
    )

    assert result.passed
    rows = result.candidate_selection_payload["candidate_rows"]
    assert rows[0]["ladder_or_compatibility_verification_passed"] is False
    assert rows[0]["independent_verification_passed"] is None
    assert rows[0]["selection_evidence"] is None
    assert rows[1]["selection_evidence"]["final_status"] == "passed"
    selection_calls = [
        call
        for call in fake_hmc.calls
        if call["num_results"] == 64 and call["num_burnin_steps"] == 32
    ]
    assert [call["num_leapfrog_steps"] for call in selection_calls] == [7]


def test_verified_handoff_rejects_target_scope_substitution() -> None:
    base = CountingGaussianAdapter()
    transport = CountingIdentityTransport()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=np.zeros(2),
        config=_config(),
        run_full_chain=FakeHMC(),
    )
    tampered = replace(
        result,
        config=replace(result.config, target_scope="different_target_scope"),
    )

    with pytest.raises(ValueError, match="lineage mismatch"):
        build_verified_fixed_transport_hmc_handoff_from_tuning_result(
            tuning_result=tampered,
            base_adapter=base,
            fixed_transport=transport,
        )


def test_verified_handoff_rejects_transport_substitution() -> None:
    base = CountingGaussianAdapter()
    tuned_transport = CountingIdentityTransport(manifest_tag="tuned")
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=tuned_transport,
        initial_position=np.zeros(2),
        config=_config(),
        run_full_chain=FakeHMC(),
    )

    with pytest.raises(ValueError, match="lineage mismatch"):
        build_verified_fixed_transport_hmc_handoff_from_tuning_result(
            tuning_result=result,
            base_adapter=base,
            fixed_transport=CountingIdentityTransport(manifest_tag="substituted"),
        )


def test_tuning_artifact_binds_source_closure_scope_route_and_seed_domains() -> None:
    fake_hmc = FakeHMC()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_config(),
        run_full_chain=fake_hmc,
    )

    paths = {row["path"] for row in result.source_dependency_closure["files"]}
    assert {
        "bayesfilter/inference/fixed_transport_hmc_tuning_tf.py",
        "bayesfilter/inference/fixed_transport_hmc_tuning.py",
        "bayesfilter/inference/fixed_transport_hmc_mechanics_tf.py",
        "bayesfilter/inference/hmc_convergence.py",
        "bayesfilter/inference/tuning_contract.py",
        "bayesfilter/inference/posterior_adapter.py",
        "bayesfilter/inference/__init__.py",
    } <= paths
    assert result.tuning_scope_payload["target_scope"] == result.config.target_scope
    assert result.route_record_payload["role"] == "active"
    assert result.route_record_payload["artifact_authority"] is True
    seeds = [call["seed"] for call in fake_hmc.calls]
    assert len(seeds) == len(set(seeds))


def test_fixed_transport_hmc_tuner_public_imports() -> None:
    assert (
        bayesfilter.tune_fixed_transport_hmc_kernel is tune_fixed_transport_hmc_kernel
    )
    assert (
        bayesfilter.FixedTransportHMCKernelTuningConfig
        is FixedTransportHMCKernelTuningConfig
    )
    assert (
        bayesfilter.FixedTransportReusableRunnerPool is FixedTransportReusableRunnerPool
    )
    assert "tune_fixed_transport_hmc_kernel" in bayesfilter.__all__


def test_reusable_runner_pool_reuses_dynamic_mechanics_and_rejects_target_crossing() -> (
    None
):
    base = CountingGaussianAdapter()
    transport = CountingIdentityTransport()
    adapter = build_fixed_transport_value_score_adapter(
        base_adapter=base,
        fixed_transport=transport,
        target_scope="gaussian_fixture_fixed_transport",
        evidence_path=None,
        xla_hmc_ready=False,
        full_chain_xla_diagnostic_ready=False,
    )
    state = tf.zeros((4, 2), tf.float64)
    config = FixedTransportFullChainConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260821, 1),
        use_xla=False,
        trace_policy="standard",
        target_status_trace_policy="none",
        tuning_policy=FixedTransportHMCPolicy.fixed(source="test"),
        target_scope=adapter.target_scope,
        chain_execution_mode="tf_function",
    )
    pool = FixedTransportReusableRunnerPool()
    pool(adapter, state, config)
    pool(
        adapter,
        state,
        replace(
            config,
            step_size=0.09,
            num_leapfrog_steps=4,
            seed=(20260821, 2),
        ),
    )
    evidence = pool.evidence()
    assert evidence["runner_count"] == 1
    assert evidence["total_call_count"] == 2
    assert evidence["all_runners_traced_exactly_once"] is True

    equivalent_but_distinct_adapter = build_fixed_transport_value_score_adapter(
        base_adapter=base,
        fixed_transport=transport,
        target_scope="gaussian_fixture_fixed_transport",
        evidence_path=None,
        xla_hmc_ready=False,
        full_chain_xla_diagnostic_ready=False,
    )
    assert (
        equivalent_but_distinct_adapter.adapter_signature()
        == adapter.adapter_signature()
    )
    with pytest.raises(ValueError, match="adapter object boundaries"):
        pool(equivalent_but_distinct_adapter, state, config)
    with pytest.raises(ValueError, match="target_scope mismatch"):
        pool(adapter, state, replace(config, target_scope="different_scope"))


def test_fixed_transport_public_route_avoids_legacy_numpy_backed_modules() -> None:
    command = (
        "import sys; "
        "import bayesfilter.inference.fixed_transport_hmc_tuning; "
        "forbidden=('bayesfilter.inference.hmc',"
        "'bayesfilter.inference.hmc_budget_ladder',"
        "'bayesfilter.inference.generic_hmc_tuning',"
        "'bayesfilter.runtime.runner'); "
        "print(','.join(name for name in forbidden if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert completed.stdout.strip() == ""


def test_real_tfp_xla_route_uses_shared_scalar_step_and_zero_chain_bank() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        leapfrog_grid=(2,),
        chain_count=4,
        budget_schedule=(2,),
        tune_num_results=2,
        screen_num_results=4,
        screen_num_burnin_steps=2,
        verification_num_results=4,
        verification_num_burnin_steps=2,
        acceptance_band=(0.01, 0.999999),
        repair_band=(0.001, 0.9999999),
        fixed_grid_fallback_acceptance_max=0.9999995,
        use_xla=True,
        target_scope="gaussian_fixture_fixed_transport",
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=tf.zeros((2,), tf.float64),
        config=config,
    )

    assert len(result.candidates) == 1
    runner_evidence = result.full_chain_runner_evidence
    assert runner_evidence is not None
    assert runner_evidence["runner_count"] == 2
    assert runner_evidence["total_call_count"] == 3
    assert runner_evidence["all_runners_traced_exactly_once"] is True
    assert result.final_kernel_payload["full_chain_runner_evidence"] == runner_evidence
    ladder = result.candidates[0].ladder_result
    assert ladder is not None
    assert ladder["shared_scalar_step_across_chain_bank"] is True
    assert ladder["runtime_numerical_backend"] == "tensorflow_tfp_only"
    tune_config = ladder["rounds"][0]["tune_config"]
    assert tune_config["tuning_policy"]["label"] == "fixed_mass_dual_averaging"
    assert tune_config["use_xla"] is True
    if result.candidates[0].verification_diagnostics:
        diagnostics = result.candidates[0].verification_diagnostics
        assert diagnostics["initial_state_shape"] == (4, 2)
        assert diagnostics["initial_state_all_zero"] is True


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
        use_xla=False,
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


def test_diagnostic_modern_verification_has_the_same_archive_requirements() -> None:
    with pytest.raises(ValueError, match="exactly four chains"):
        _modern_config(
            chain_count=3,
            require_modern_rank_normalized_verification=False,
            report_modern_rank_normalized_verification=True,
        )
    with pytest.raises(ValueError, match="at least the configured retained"):
        _modern_config(
            verification_num_results=999,
            require_modern_rank_normalized_verification=False,
            report_modern_rank_normalized_verification=True,
        )


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


def test_diagnostic_modern_rhat_is_reported_without_vetoing_healthy_mechanics() -> None:
    base = CountingGaussianAdapter()
    transport = CountingIdentityTransport()
    config = _modern_config(
        require_modern_rank_normalized_verification=False,
        report_modern_rank_normalized_verification=True,
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=ArchiveFakeHMC(folded_scale_mismatch=True),
    )

    assert result.passed
    assert result.final_kernel_payload is not None
    diagnostics = result.final_kernel_payload["verification_diagnostics"]
    modern = diagnostics["modern_rank_normalized_verification"]
    assert modern["passed"] is False
    assert modern["max_folded_rank_normalized_split_rhat"] > 1.01
    assert diagnostics["modern_rank_normalized_verification_role"] == (
        "diagnostic_only_not_handoff_gate"
    )
    assert "verification_modern_rank_folded_rhat_failed" not in result.hard_vetoes
    assert config.payload()["modern_rank_normalized_verification_role"] == (
        "diagnostic_only_not_handoff_gate"
    )
    handoff = build_verified_fixed_transport_hmc_handoff_from_tuning_result(
        tuning_result=result,
        base_adapter=base,
        fixed_transport=transport,
    )
    assert handoff.handoff_hash
    assert handoff.handoff_payload["final_kernel_hash"] == result.final_kernel_hash


def test_diagnostic_modern_rhat_computation_error_cannot_veto_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_diagnostic(*_args, **_kwargs):
        raise RuntimeError("diagnostic negative control")

    monkeypatch.setattr(
        fixed_tuning, "rank_normalized_split_rhat_summary", fail_diagnostic
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(
            require_modern_rank_normalized_verification=False,
            report_modern_rank_normalized_verification=True,
        ),
        run_full_chain=ArchiveFakeHMC(),
    )

    assert result.passed
    diagnostics = result.final_kernel_payload["verification_diagnostics"]
    modern = diagnostics["modern_rank_normalized_verification"]
    assert modern["passed"] is False
    assert modern["diagnostic_error"] == {
        "type": "RuntimeError",
        "message": "diagnostic negative control",
    }
    assert diagnostics["modern_rank_normalized_verification_role"] == (
        "diagnostic_only_not_handoff_gate"
    )
    assert "verification_runtime_error" not in result.hard_vetoes


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


def test_unavailable_native_divergence_is_recorded_but_not_a_veto() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(),
        run_full_chain=UnavailableDivergenceFakeHMC(),
    )

    assert result.passed
    scale = result.fixed_grid_scale_selection_payload
    assert scale is not None
    assert scale["attempts"][0]["probe_hard_vetoes"] == ()
    diagnostics = scale["attempts"][0]["probe_diagnostics"]
    assert diagnostics["divergence_status"] == "not_exposed_by_kernel"
    assert diagnostics["divergence_count"] is None
    assert diagnostics["native_divergence_interpretation"] == (
        "unavailable is not zero divergences"
    )


def test_finite_log_accept_energy_tail_is_explanatory_only() -> None:
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=CountingGaussianAdapter(),
        fixed_transport=CountingIdentityTransport(),
        initial_position=np.zeros(2),
        config=_modern_config(),
        run_full_chain=EnergyTailFakeHMC(),
    )
    assert result.passed
    scale = result.fixed_grid_scale_selection_payload
    assert scale is not None
    assert scale["attempts"][0]["probe_hard_vetoes"] == ()
    diagnostics = result.selected_candidate.verification_diagnostics
    assert diagnostics["max_abs_log_accept_energy_proxy"] == pytest.approx(1001.0)
    assert diagnostics["log_accept_energy_proxy_alert"] is True
    assert diagnostics["log_accept_energy_proxy_role"] == "explanatory_alert_only"
