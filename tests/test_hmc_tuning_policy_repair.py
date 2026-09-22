"""Focused tests for the guide-directed fixed-transport tuning policy.

These are mechanics/policy tests.  They do not establish posterior validity or
HMC performance for a scientific target.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    FixedTransportHMCKernelTuningConfig,
    ValueScoreCapability,
    build_verified_fixed_transport_hmc_handoff_from_tuning_result,
    tune_fixed_transport_hmc_kernel,
)
from bayesfilter.inference.hmc import FullChainHMCRunResult


class _GaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "policy-repair-gaussian-adapter-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="policy_repair_fixture",
            target_scope="policy_repair_fixture",
            nonclaims=("policy fixture only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor):
        theta = tf.convert_to_tensor(theta, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(theta), axis=-1), -theta


class _IdentityTransport:
    parameter_dim = 2

    def manifest_payload(self):
        return {
            "schema": "policy_repair_identity_transport.v1",
            "parameter_dim": 2,
        }

    def forward(self, z):
        return tf.convert_to_tensor(z, tf.float64)

    def forward_batch(self, z):
        return tf.convert_to_tensor(z, tf.float64)

    def log_abs_det_jacobian(self, z):
        del z
        return tf.constant(0.0, tf.float64)

    def log_abs_det_jacobian_batch(self, z):
        return tf.zeros(tf.shape(tf.convert_to_tensor(z))[:1], tf.float64)

    def pullback_score(self, z, score):
        del z
        return tf.convert_to_tensor(score, tf.float64)

    def pullback_score_batch(self, z, score):
        del z
        return tf.convert_to_tensor(score, tf.float64)

    def log_abs_det_jacobian_score(self, z):
        return tf.zeros_like(tf.convert_to_tensor(z, tf.float64))

    def log_abs_det_jacobian_score_batch(self, z):
        return tf.zeros_like(tf.convert_to_tensor(z, tf.float64))


class _IIDFixedKernel:
    """Small deterministic-seed fixture with complete transition telemetry."""

    def __init__(self, *, acceptance: float = 0.70):
        self.acceptance = float(acceptance)

    def __call__(self, adapter, initial_state, config):
        state = tf.convert_to_tensor(initial_state, tf.float64)
        chains = int(state.shape[0])
        draws = tf.random.stateless_normal(
            (int(config.num_results), chains, int(state.shape[1])),
            seed=tuple(int(value) for value in config.seed),
            dtype=tf.float64,
        )
        value, score = adapter.log_prob_and_grad(state)
        trace_shape = (int(config.num_results), chains)
        trace = {
            "is_accepted": tf.ones(trace_shape, tf.bool),
            "log_accept_ratio": tf.fill(
                trace_shape, tf.math.log(tf.constant(self.acceptance, tf.float64))
            ),
            "target_log_prob": tf.broadcast_to(value[tf.newaxis, :], trace_shape),
            "proposed_target_log_prob": tf.broadcast_to(value[tf.newaxis, :], trace_shape),
            "target_score": tf.broadcast_to(
                score[tf.newaxis, :, :],
                (int(config.num_results), chains, int(state.shape[1])),
            ),
        }
        return FullChainHMCRunResult(
            samples=draws,
            trace=trace,
            diagnostics={
                "acceptance_rate": tf.constant(self.acceptance, tf.float64),
                "finite_sample_count": tf.size(draws),
                "nonfinite_sample_count": tf.constant(0, tf.int32),
                "final_step_size": tf.constant(config.step_size, tf.float64),
                "final_step_size_finite": tf.constant(True),
                "target_accept_prob": tf.constant(0.70, tf.float64),
                "num_adaptation_steps": tf.constant(0, tf.int32),
                "trace_policy": config.trace_policy,
                "divergence_status": "available",
                "divergence_count": tf.constant(0, tf.int32),
            },
            metadata={
                "runtime": "policy_repair_iid_fixture",
                "initial_state_shape": tuple(int(v) for v in state.shape),
                "target_scope": config.target_scope,
                "windowed_mass_adaptation_used": False,
            },
        )


def test_harmonic_oscillator_phase_invalidates_directional_inference() -> None:
    def phase(step: float, leapfrog: int) -> float:
        return leapfrog * math.acos(1.0 - step * step / 2.0)

    # These two stable steps lie near different trajectory phases at L=5.
    phase_near_return = phase(1.205189, 5)
    phase_near_reversal = phase(0.628978, 5)
    assert abs(phase_near_return - 2.0 * math.pi) < 0.25
    assert abs(phase_near_reversal - math.pi) < 0.25
    assert phase_near_return > phase_near_reversal


def test_measured_policy_rejects_underspecified_grid() -> None:
    with pytest.raises(ValueError, match="at least two distinct step"):
        FixedTransportHMCKernelTuningConfig(
            initial_step_size=0.1,
            step_size_candidates=(0.1,),
        )
    with pytest.raises(ValueError, match="at least two distinct leapfrog"):
        FixedTransportHMCKernelTuningConfig(
            initial_step_size=0.1,
            step_size_candidates=(0.1, 0.2),
            leapfrog_grid=(5,),
        )
    with pytest.raises(ValueError, match="replicated_min_bulk_ess_per_gradient"):
        FixedTransportHMCKernelTuningConfig(
            initial_step_size=0.1,
            step_size_candidates=(0.1, 0.2),
            leapfrog_grid=(5, 7),
            selection_policy="acceptance_target_distance",
        )


def test_measured_policy_executes_every_joint_pair_and_holds_out() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        step_size_candidates=(0.1, 0.2),
        leapfrog_grid=(3, 5),
        selection_replications=2,
        selection_num_results=16,
        selection_num_burnin_steps=4,
        verification_num_results=8,
        verification_num_burnin_steps=2,
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="policy_repair_fixture",
    )
    runner = _IIDFixedKernel()
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=_GaussianAdapter(),
        fixed_transport=_IdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=runner,
    )
    grid = result.fixed_grid_scale_selection_payload
    assert grid is not None
    assert grid["all_declared_pairs_measured"] is True
    assert grid["directional_inference_used"] is False
    assert grid["candidate_count"] == 4
    assert len(result.candidates) == 4
    selection = result.candidate_selection_payload
    assert selection["all_candidate_pairs_measured"] is True
    assert selection["heldout_verification"]["final_status"] == "passed"
    assert result.final_kernel_payload is not None
    assert result.final_kernel_payload["tuning_policy"] == "measured_joint_grid_v1"
    assert result.final_kernel_payload["verification_diagnostics"][
        "jump_distance_excludes_initial_state"
    ] is True
    assert all(
        candidate.payload()["tuning_candidate_status"] == "eligible_for_selection"
        for candidate in result.candidates
    )


def test_measured_policy_does_not_hard_veto_valid_high_acceptance() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        step_size_candidates=(0.1, 0.2),
        leapfrog_grid=(3, 5),
        selection_replications=2,
        selection_num_results=16,
        selection_num_burnin_steps=4,
        verification_num_results=8,
        verification_num_burnin_steps=2,
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="policy_repair_fixture_high_acceptance",
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=_GaussianAdapter(),
        fixed_transport=_IdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=_IIDFixedKernel(acceptance=0.99),
    )
    assert result.passed is True
    assert result.final_kernel_payload is not None
    assert result.final_kernel_payload["posterior_ready"] is False
    assert result.candidate_selection_payload["heldout_verification"]["final_status"] == (
        "passed"
    )
    assert all(
        "verification_acceptance_outside_pass_band" not in candidate.hard_vetoes
        for candidate in result.candidates
    )
    assert any(
        "verification_acceptance_above_pass_band" in candidate.repair_triggers
        for candidate in result.candidates
    )


def test_measured_policy_rejects_over_cap_grid_before_running() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        FixedTransportHMCKernelTuningConfig(
            initial_step_size=0.1,
            step_size_candidates=(0.1, 0.3),
            leapfrog_grid=(3, 5),
            maximum_candidate_step_size=0.2,
        )


def test_legacy_directional_policy_is_diagnostic_only() -> None:
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        leapfrog_grid=(5, 7),
        tuning_policy="legacy_directional_diagnostic_v1",
        selection_policy="acceptance_target_distance",
        budget_schedule=(1,),
        tune_num_results=2,
        screen_num_results=2,
        screen_num_burnin_steps=1,
        verification_num_results=2,
        verification_num_burnin_steps=1,
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="policy_repair_fixture",
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=_GaussianAdapter(),
        fixed_transport=_IdentityTransport(),
        initial_position=np.zeros(2),
        config=config,
        run_full_chain=_IIDFixedKernel(),
    )
    assert result.final_status == "diagnostic_only_legacy_policy"
    assert result.passed is False
    assert result.route_record_payload["artifact_authority"] is False
    with pytest.raises(ValueError, match="legacy directional results are diagnostic-only"):
        build_verified_fixed_transport_hmc_handoff_from_tuning_result(
            tuning_result=result,
            base_adapter=_GaussianAdapter(),
            fixed_transport=_IdentityTransport(),
        )
