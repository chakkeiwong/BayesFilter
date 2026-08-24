from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import inspect

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
from bayesfilter.inference import (
    FixedSizeHMCChunkConfig,
    SequentialRHatCheckpointWriterConfig,
    SequentialRHatHMCVerificationConfig,
    ValueScoreCapability,
    assert_sequential_rhat_checkpoint_public_reference_safe,
    build_fixed_size_hmc_chunk_runner,
    build_sequential_rhat_hmc_verifier,
    inspect_sequential_rhat_private_checkpoint,
)


class ReviewedBatchedGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "fixed-size-hmc-chunk-reviewed-batched-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_fixed_size_chunk_runner.py",
            target_scope="fixed_size_hmc_chunk_gaussian",
            nonclaims=("tiny fixed-size HMC chunk engineering fixture only",),
            full_chain_xla_diagnostic_ready=True,
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values


class TargetOnlyXLAGaussianAdapter(ReviewedBatchedGaussianAdapter):
    def adapter_signature(self) -> str:
        return "fixed-size-hmc-chunk-target-only-xla-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_fixed_size_chunk_runner.py",
            target_scope="fixed_size_hmc_chunk_target_only",
            nonclaims=("target-only XLA fixture; no full-chain XLA authority",),
        )


def _initial_state() -> tf.Tensor:
    return tf.constant(
        [[0.1, -0.2], [0.2, 0.1], [-0.1, 0.3]],
        dtype=tf.float64,
    )


def _config(*, use_xla: bool = False, max_results: int = 5) -> FixedSizeHMCChunkConfig:
    return FixedSizeHMCChunkConfig(
        max_results=max_results,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260622, 10),
        use_xla=use_xla,
        target_scope="fixed_size_hmc_chunk_gaussian",
        chain_execution_mode="tf_function",
    )


def _standard_verification_trace(
    draws: int,
    *,
    chain_count: int = 3,
    probability: float = 0.70,
) -> Mapping[str, Any]:
    shape = (int(draws), int(chain_count))
    return {
        "is_accepted": tf.ones(shape, dtype=tf.bool),
        "log_accept_ratio": tf.fill(
            shape,
            tf.math.log(tf.constant(float(probability), dtype=tf.float64)),
        ),
        "target_log_prob": -tf.ones(shape, dtype=tf.float64),
    }


def _write_deterministic_sequential_rhat_checkpoints(
    monkeypatch,
    tmp_path,
    *,
    checkpoint_label: str = "phase4",
):
    import bayesfilter.inference.hmc as hmc_module

    class _DivergentChainChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            self.call_count += 1
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            chain_offsets = tf.reshape(
                tf.constant([-1.0, 0.0, 1.0], dtype=tf.float64),
                (1, 3, 1),
            )
            samples = tf.ones((draws, 3, 2), dtype=tf.float64) * chain_offsets
            samples = samples + tf.cast(self.call_count, tf.float64)
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            metadata = {
                "compile_trace_count": 1,
                "first_call_s": 0.01,
                "warm_call_s": None if self.call_count == 1 else 0.001,
                "chunk_call_s": 0.001,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=base,
                trace=_standard_verification_trace(draws),
                diagnostics=diagnostics,
                metadata=metadata,
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _DivergentChainChunkRunner)
    result = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=2,
            max_results=6,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260628, 44),
            chain_count=3,
            rhat_threshold=1.01,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    ).run(
        checkpoint_writer_config=SequentialRHatCheckpointWriterConfig(
            checkpoint_dir=tmp_path,
            checkpoint_label=checkpoint_label,
        )
    )
    manifest_paths = sorted(tmp_path.glob(f"{checkpoint_label}_*_manifest.json"))
    return result, manifest_paths


def test_fixed_size_chunk_runner_changes_active_count_without_retrace() -> None:
    runner = build_fixed_size_hmc_chunk_runner(
        ReviewedBatchedGaussianAdapter(),
        _initial_state(),
        _config(use_xla=False, max_results=5),
    )

    first = runner.run(active_results=2, seed=(20260622, 11))
    second = runner.run(
        active_results=4,
        current_state=first.final_state,
        seed=(20260622, 12),
    )

    assert first.samples.shape == (5, 3, 2)
    assert second.samples.shape == (5, 3, 2)
    assert first.valid_mask.shape == (5,)
    assert second.valid_mask.shape == (5,)
    np.testing.assert_array_equal(
        first.valid_mask.numpy(),
        np.asarray([True, True, False, False, False]),
    )
    np.testing.assert_array_equal(
        second.valid_mask.numpy(),
        np.asarray([True, True, True, True, False]),
    )
    assert first.final_state.shape == (3, 2)
    assert second.final_state.shape == (3, 2)
    assert not np.allclose(first.final_state.numpy(), second.final_state.numpy())
    assert int(first.diagnostics["valid_sample_count"].numpy()) == 2
    assert int(first.diagnostics["invalid_sample_count"].numpy()) == 3
    assert int(second.diagnostics["valid_sample_count"].numpy()) == 4
    assert int(second.diagnostics["invalid_sample_count"].numpy()) == 1
    assert int(first.diagnostics["nonfinite_valid_sample_count"].numpy()) == 0
    assert int(second.diagnostics["nonfinite_valid_sample_count"].numpy()) == 0
    assert first.metadata["runtime"] == "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop"
    assert first.metadata["uses_sample_chain"] is False
    assert second.metadata["fixed_size_chunk_runner"] is True
    assert second.metadata["max_results"] == 5
    assert second.metadata["active_results"] == 4
    assert second.metadata["chunk_invocation_count"] == 2
    assert first.metadata["compile_trace_count"] == 1
    assert second.metadata["compile_trace_count"] == 1
    assert second.metadata["first_call_s"] == first.metadata["first_call_s"]
    assert second.metadata["warm_call_s"] >= 0.0
    assert second.metadata["dynamic_inputs"] == (
        "current_state",
        "seed",
        "step_size",
        "active_results",
    )
    assert np.isfinite(float(second.diagnostics["acceptance_rate"].numpy()))
    assert int(second.diagnostics["acceptance_decision_count"].numpy()) == 12
    assert int(second.diagnostics["log_accept_ratio_nonfinite_count"].numpy()) == 0
    assert int(second.diagnostics["target_log_prob_nonfinite_count"].numpy()) == 0
    assert "no sampler convergence claim" in second.metadata["nonclaims"]


def test_fixed_size_chunk_runner_xla_compiles_tiny_contract() -> None:
    runner = build_fixed_size_hmc_chunk_runner(
        ReviewedBatchedGaussianAdapter(),
        _initial_state(),
        _config(use_xla=True, max_results=2),
    )

    result = runner.run(active_results=1, seed=(20260622, 13))

    assert result.samples.shape == (2, 3, 2)
    assert int(result.diagnostics["valid_sample_count"].numpy()) == 1
    assert int(result.diagnostics["nonfinite_valid_sample_count"].numpy()) == 0
    assert result.metadata["jit_compile"] is True
    assert result.metadata["compile_trace_count"] == 1


@pytest.mark.parametrize(
    ("rhat_admission_policy", "expected_passed"),
    (("rhat_gate", False), ("explanatory_only", True)),
)
def test_sequential_verifier_policy_separates_rhat_telemetry_from_handoff(
    monkeypatch: pytest.MonkeyPatch,
    rhat_admission_policy: str,
    expected_passed: bool,
) -> None:
    """A fake chunk proves the two admission policies without running HMC."""

    import bayesfilter.inference.hmc as hmc_module

    class _HealthyChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self._state = tf.convert_to_tensor(initial_state, dtype=tf.float64)

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            del seed, step_size
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            draw_axis = tf.cast(tf.range(draws), tf.float64)[:, None, None]
            chain_axis = tf.reshape(
                tf.constant([-1.0, 0.0, 1.0, 2.0], dtype=tf.float64),
                (1, 4, 1),
            )
            samples = tf.broadcast_to(
                draw_axis + chain_axis,
                (draws, 4, 2),
            )
            trace = _standard_verification_trace(
                draws,
                chain_count=4,
                probability=0.70,
            )
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(4 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(4 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(4 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=tf.ones((draws,), dtype=tf.bool),
                final_state=base,
                trace=trace,
                diagnostics=diagnostics,
                metadata={
                    "compile_trace_count": 0,
                    "first_call_s": 0.0,
                    "warm_call_s": 0.0,
                    "chunk_call_s": 0.0,
                },
            )

    def _failed_rhat(_samples, *, threshold):
        del threshold
        summary = dict(hmc_module._empty_rhat_summary())
        summary.update(
            {
                "max_rank_normalized_split_rhat": 1.20,
                "max_folded_rank_normalized_split_rhat": 1.15,
                "max_finite_rhat": 1.20,
                "finite_rhat_count": 2,
                "nonfinite_rhat_count": 0,
            }
        )
        return summary

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _HealthyChunkRunner)
    monkeypatch.setattr(hmc_module, "_rhat_summary_from_retained_samples", _failed_rhat)
    config = SequentialRHatHMCVerificationConfig(
        check_interval=64,
        max_results=64,
        num_burnin_steps=0,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260822, 1),
        chain_count=4,
        target_scope="fixed_size_hmc_chunk_gaussian",
        rhat_admission_policy=rhat_admission_policy,
        explicit_start_bank_policy_id="phase7_engineering_probe_bank_v1",
    )
    result = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((4, 2), dtype=tf.float64),
        config,
    ).run()

    assert result.passed is expected_passed
    assert result.rhat_passed is False
    assert result.acceptance_handoff_eligible is True
    assert result.diagnostics["rhat_admission_policy"] == rhat_admission_policy
    assert result.diagnostics["rhat_passed"] is False
    assert result.diagnostics["acceptance_handoff_eligible"] is True
    assert result.diagnostics["promotion_role"] == "non_promoting"
    expected_role = (
        "explanatory_only_not_stopping_or_admission"
        if rhat_admission_policy == "explanatory_only"
        else "fixed_kernel_convergence_gate_not_candidate_ranking"
    )
    assert result.diagnostics["rhat_role"] == expected_role
    assert result.metadata["initial_state_policy"] == (
        "phase7_engineering_probe_bank_v1"
    )
    payload = config.signature_payload()
    assert payload["rhat_admission_policy"] == rhat_admission_policy
    assert payload["explicit_start_bank_policy_id"] == (
        "phase7_engineering_probe_bank_v1"
    )
    gate_config = SequentialRHatHMCVerificationConfig(
        **{
            **config.signature_payload(),
            "acceptance_policy": config.acceptance_policy,
            "rhat_admission_policy": (
                "explanatory_only"
                if rhat_admission_policy == "rhat_gate"
                else "rhat_gate"
            ),
        }
    )
    assert hmc_module.program_signature(gate_config.signature_payload()) != (
        hmc_module.program_signature(config.signature_payload())
    )


def test_sequential_verifier_policy_and_bank_fields_validate() -> None:
    base = dict(
        check_interval=4,
        max_results=4,
        num_burnin_steps=0,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260822, 2),
        chain_count=4,
        target_scope="fixed_size_hmc_chunk_gaussian",
    )
    with pytest.raises(ValueError, match="rhat_admission_policy"):
        SequentialRHatHMCVerificationConfig(
            **base,
            rhat_admission_policy="unknown",
        )
    with pytest.raises(ValueError, match="explicit_start_bank_policy_id"):
        SequentialRHatHMCVerificationConfig(
            **base,
            explicit_start_bank_policy_id=" ",
        )
    default_config = SequentialRHatHMCVerificationConfig(**base)
    assert default_config.explicit_start_bank_policy_id == "explicit_start_bank"
    assert default_config.rhat_admission_policy == "rhat_gate"


def test_sequential_verifier_fails_closed_when_modern_rhat_is_undefined(monkeypatch) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _ScriptedChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self._state = tf.convert_to_tensor(initial_state, dtype=tf.float64)
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            self.call_count += 1
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            samples = tf.zeros((draws, 3, 2), dtype=tf.float64) + tf.reshape(
                tf.linspace(
                    tf.constant(-0.001, dtype=tf.float64),
                    tf.constant(0.001, dtype=tf.float64),
                    draws,
                ),
                (draws, 1, 1),
            )
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            metadata = {
                "compile_trace_count": 1,
                "first_call_s": 0.01,
                "warm_call_s": None if self.call_count == 1 else 0.001,
                "chunk_call_s": 0.001,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=base,
                trace=_standard_verification_trace(draws),
                diagnostics=diagnostics,
                metadata=metadata,
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _ScriptedChunkRunner)
    verifier = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=2,
            max_results=6,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260628, 1),
            chain_count=3,
            rhat_threshold=1.01,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    )

    result = verifier.run()

    assert result.passed is False
    assert result.cap_hit is True
    assert result.retained_sample_count == 6
    assert result.chunk_count == 3
    assert result.max_finite_rhat is None
    assert result.diagnostics["acceptance_evidence"]["decision"] == (
        "inconclusive_evidence"
    )
    assert result.diagnostics["rhat_role"] == (
        "fixed_kernel_convergence_gate_not_candidate_ranking"
    )
    assert result.diagnostics["rhat_definition"] == (
        "max(rank-normalized split R-hat, "
        "folded rank-normalized split R-hat)"
    )
    assert result.diagnostics["max_finite_rhat"] is None
    assert result.diagnostics["privacy_contract"][
        "public_summary_contains_step_size"
    ] is False
    assert result.diagnostics["privacy_contract"][
        "public_summary_contains_leapfrog_count"
    ] is False
    assert result.diagnostics["privacy_contract"][
        "public_summary_contains_mass_matrix"
    ] is False
    assert result.metadata["sample_values_publicized"] is False
    assert result.metadata["chain_handoff_publicized"] is False
    assert result.metadata["checkpointing_enabled"] is False
    assert result.metadata["checkpoint_count"] == 0
    assert result.metadata["checkpoint_references"] == []
    public_text = json.dumps(
        {"diagnostics": result.diagnostics, "metadata": result.metadata},
        sort_keys=True,
    )
    assert "target_log_prob" not in public_text
    assert '"log_accept_ratio":' not in public_text


def test_sequential_verifier_rejects_retained_count_mismatch(monkeypatch) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _ShortMaskChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self._state = tf.convert_to_tensor(initial_state, dtype=tf.float64)

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            del seed, step_size
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            samples = tf.ones((draws, 4, 2), dtype=tf.float64)
            valid = tf.concat(
                (tf.ones((draws - 1,), tf.bool), tf.zeros((1,), tf.bool)),
                axis=0,
            )
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=base,
                trace=_standard_verification_trace(draws, chain_count=4),
                diagnostics={
                    "valid_sample_count": tf.constant(draws, tf.int32),
                    "nonfinite_valid_sample_count": tf.constant(0, tf.int32),
                    "acceptance_rate": tf.constant(0.70, tf.float64),
                    "acceptance_decision_count": tf.constant(4 * draws, tf.int32),
                    "log_accept_ratio_nonfinite_count": tf.constant(0, tf.int32),
                    "target_log_prob_nonfinite_count": tf.constant(0, tf.int32),
                    "divergence_status": "not_exposed_by_kernel",
                    "divergence_count": None,
                },
                metadata={},
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _ShortMaskChunkRunner)
    result = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=64,
            max_results=64,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260712, 73),
            chain_count=4,
            target_scope="fixed_size_hmc_chunk_gaussian",
        ),
    ).run()

    assert result.passed is False
    assert result.cap_hit is False
    assert result.retained_sample_count == 63
    assert result.diagnostics["acceptance_evidence_validity"] == (
        "shared_execution_invalid"
    )
    assert "shared_schema_invalid" in result.diagnostics[
        "engineering_invalidity_reasons"
    ]
    assert "chunk_retained_count_mismatch" in result.diagnostics["hard_vetoes"]
    assert result.diagnostics["chunk_summaries"][0]["expected_active_results"] == 64
    assert result.diagnostics["chunk_summaries"][0]["observed_valid_mask_count"] == 63


def test_sequential_verifier_rejects_mixed_divergence_coverage(monkeypatch) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _MixedDivergenceChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            del seed, step_size
            self.call_count += 1
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            offset = tf.cast((self.call_count - 1) * draws, tf.float64)
            draw = offset + tf.cast(tf.range(draws), tf.float64)
            samples = tf.tile(draw[:, None, None], (1, 4, 2))
            exposed = self.call_count == 1
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=tf.ones((draws,), tf.bool),
                final_state=samples[-1],
                trace=_standard_verification_trace(draws, chain_count=4),
                diagnostics={
                    "valid_sample_count": tf.constant(draws, tf.int32),
                    "nonfinite_valid_sample_count": tf.constant(0, tf.int32),
                    "acceptance_rate": tf.constant(0.70, tf.float64),
                    "acceptance_decision_count": tf.constant(4 * draws, tf.int32),
                    "log_accept_ratio_nonfinite_count": tf.constant(0, tf.int32),
                    "target_log_prob_nonfinite_count": tf.constant(0, tf.int32),
                    "divergence_status": (
                        "available" if exposed else "not_exposed_by_kernel"
                    ),
                    "divergence_count": 0 if exposed else None,
                },
                metadata={},
            )

    monkeypatch.setattr(
        hmc_module,
        "FixedSizeHMCChunkRunner",
        _MixedDivergenceChunkRunner,
    )
    result = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=32,
            max_results=64,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260712, 74),
            chain_count=4,
            target_scope="fixed_size_hmc_chunk_gaussian",
        ),
    ).run()

    assert result.passed is False
    assert result.cap_hit is False
    assert result.retained_sample_count == 64
    assert result.chunk_count == 2
    assert result.diagnostics["acceptance_evidence_validity"] == (
        "shared_execution_invalid"
    )
    assert result.diagnostics["divergence_status"] == "not_collected"
    assert result.diagnostics["divergence_count"] is None
    assert result.diagnostics["divergence_provenance_consistent"] is False
    assert "shared_schema_invalid" in result.diagnostics[
        "engineering_invalidity_reasons"
    ]


def test_sequential_verifier_rejects_period_two_return_path(monkeypatch) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _PeriodTwoChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self._initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float64)

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            draws = int(active_results)
            state_a = tf.convert_to_tensor(current_state, dtype=tf.float64)
            state_b = state_a + tf.constant([1.0, -0.5], dtype=tf.float64)
            samples = tf.stack((state_a, state_b) * (draws // 2), axis=0)
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(4 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(4 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.36, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(4 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=samples[-1],
                trace=_standard_verification_trace(draws, chain_count=4),
                diagnostics=diagnostics,
                metadata={
                    "compile_trace_count": 1,
                    "first_call_s": 0.01,
                    "warm_call_s": None,
                    "chunk_call_s": 0.001,
                },
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _PeriodTwoChunkRunner)
    result = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=64,
            max_results=64,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260712, 72),
            chain_count=4,
            rhat_threshold=1.01,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    ).run()

    evidence = result.diagnostics["acceptance_evidence"]
    assert result.passed is False
    assert result.cap_hit is False
    assert result.retained_sample_count == 64
    assert result.chunk_count == 1
    assert evidence["acceptance_decision"] == "repair_trajectory"
    assert evidence["path_return_fraction_by_chain"] == [1.0] * 4
    assert evidence["candidate_promotion_vetoes"] == [
        "path_return_resonance_detected"
    ]
    assert result.diagnostics["tuning_repair_triggers"] == [
        "trajectory:repair_resonance"
    ]
    assert result.diagnostics["rhat_role"] == (
        "fixed_kernel_convergence_gate_not_candidate_ranking"
    )


@pytest.mark.parametrize(
    ("mode", "expected_validity", "expected_reason", "expected_status_count"),
    (
        ("public_default", "valid", None, None),
        (
            "missing",
            "shared_execution_invalid",
            "required_target_status_telemetry_missing",
            None,
        ),
        ("malformed", "shared_execution_invalid", "shared_schema_invalid", None),
        (
            "nonvalid",
            "candidate_data_invalid",
            "target_status_telemetry_failure",
            64,
        ),
        (
            "callback_exception",
            "shared_execution_invalid",
            "shared_callback_invalid",
            None,
        ),
    ),
)
def test_sequential_verifier_private_retained_health_is_phase7_opt_in(
    monkeypatch,
    mode: str,
    expected_validity: str,
    expected_reason: str | None,
    expected_status_count: int | None,
) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _DeterministicChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self._state = tf.convert_to_tensor(initial_state, dtype=tf.float64)

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            del seed, step_size
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            draw = tf.cast(tf.range(draws), tf.float64)[:, tf.newaxis, tf.newaxis]
            chain = tf.cast(tf.range(4), tf.float64)[tf.newaxis, :, tf.newaxis]
            direction = tf.constant([1.0, -0.5], tf.float64)[tf.newaxis, tf.newaxis, :]
            samples = draw * direction + chain
            probability = tf.fill((draws, 4), tf.constant(0.70, tf.float64))
            diagnostics = {
                "valid_sample_count": tf.constant(draws, tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, tf.int32),
                "acceptance_rate": tf.constant(0.70, tf.float64),
                "acceptance_decision_count": tf.constant(4 * draws, tf.int32),
                "log_accept_ratio_finite_count": tf.constant(4 * draws, tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, tf.int32),
                "log_accept_ratio_max_abs_finite": -tf.math.log(
                    tf.constant(0.70, tf.float64)
                ),
                "target_log_prob_finite_count": tf.constant(4 * draws, tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=tf.ones((draws,), tf.bool),
                final_state=samples[-1] if draws else base,
                trace={
                    "is_accepted": tf.ones((draws, 4), tf.bool),
                    "log_accept_ratio": tf.math.log(probability),
                    "target_log_prob": -tf.reduce_sum(tf.square(samples), axis=-1),
                },
                diagnostics=diagnostics,
                metadata={
                    "compile_trace_count": 1,
                    "first_call_s": 0.01,
                    "warm_call_s": None,
                    "chunk_call_s": 0.001,
                },
            )

    class _StatusAdapter(ReviewedBatchedGaussianAdapter):
        def log_prob_and_grad(self, theta):
            if mode == "callback_exception":
                raise RuntimeError("unexpected target callback failure")
            return super().log_prob_and_grad(theta)

        def target_status_telemetry(self, theta):
            shape = tf.shape(theta)[:-1]
            payload = {
                "status_code": tf.zeros(shape, tf.int32),
                "valid_pre_regularized_score": tf.ones(shape, tf.bool),
                "floor_count_value": tf.zeros(shape, tf.int32),
                "min_innovation_eigenvalue": tf.ones(shape, tf.float64),
                "innovation_condition_estimate": tf.ones(shape, tf.float64),
            }
            if mode == "malformed":
                payload.pop("floor_count_value")
            elif mode == "nonvalid":
                payload["status_code"] = tf.concat(
                    (tf.ones((1,), tf.int32), tf.zeros((3,), tf.int32)),
                    axis=0,
                )
            return payload

    adapter = (
        ReviewedBatchedGaussianAdapter()
        if mode in {"public_default", "missing"}
        else _StatusAdapter()
    )
    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _DeterministicChunkRunner)
    verifier = build_sequential_rhat_hmc_verifier(
        adapter,
        tf.zeros((4, 2), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=64,
            max_results=64,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260712, 1),
            chain_count=4,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    )
    if mode != "public_default":
        verifier._configure_retained_target_health_policy("per_chain_step")

    result = verifier.run()
    diagnostics = result.diagnostics

    assert diagnostics["acceptance_evidence_validity"] == expected_validity
    assert diagnostics["target_status_failure_count"] == expected_status_count
    if mode == "public_default":
        assert result.passed is False
        assert result.cap_hit is True
        assert diagnostics["retained_target_health_policy"] == "disabled"
        assert diagnostics["target_score_health_passed"] is None
        assert diagnostics["retained_target_health_evaluated_draw_count"] == 0
    else:
        assert result.passed is False
        assert diagnostics["retained_target_health_policy"] == "per_chain_step"
        assert expected_reason in diagnostics["engineering_invalidity_reasons"]


def test_sequential_rhat_verifier_respects_minimum_retained_results_for_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesfilter.inference.hmc as hmc_module

    calls: list[int] = []

    class _ScriptedChunkRunner:
        def __init__(self, _adapter: Any, initial_state: Any, _config: Any) -> None:
            self.initial_state = initial_state
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            del seed, step_size
            self.call_count += 1
            draws = int(active_results)
            calls.append(draws)
            base = (
                tf.convert_to_tensor(current_state, dtype=tf.float64)
                if current_state is not None
                else tf.convert_to_tensor(self.initial_state, dtype=tf.float64)
            )
            samples = tf.random.stateless_normal(
                (draws, 4, 2),
                seed=(20260713, self.call_count),
                dtype=tf.float64,
            )
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(4 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(4 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(4 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            metadata = {
                "compile_trace_count": 1,
                "first_call_s": 0.01,
                "warm_call_s": None if self.call_count == 1 else 0.001,
                "chunk_call_s": 0.001,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=samples[-1] if draws else base,
                trace=_standard_verification_trace(draws, chain_count=4),
                diagnostics=diagnostics,
                metadata=metadata,
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _ScriptedChunkRunner)
    verifier = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=200,
            max_results=600,
            min_retained_results_for_pass=600,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260628, 1),
            chain_count=4,
            rhat_threshold=1.01,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    )

    result = verifier.run()

    assert result.passed is True
    assert result.retained_sample_count == 600
    assert result.chunk_count == 3
    assert calls == [200, 200, 200]
    assert result.diagnostics["min_retained_results_for_pass"] == 600
    assert result.diagnostics["minimum_retained_pass_gate_satisfied"] is True
    assert result.metadata["min_retained_results_for_pass"] == 600


def test_sequential_rhat_verifier_stops_at_cap_when_rhat_fails(monkeypatch) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _DivergentChainChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self._state = tf.convert_to_tensor(initial_state, dtype=tf.float64)
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            self.call_count += 1
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            chain_offsets = tf.reshape(
                tf.constant([-1.0, 0.0, 1.0], dtype=tf.float64),
                (1, 3, 1),
            )
            samples = tf.ones((draws, 3, 2), dtype=tf.float64) * chain_offsets
            samples = samples + tf.cast(self.call_count, tf.float64)
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            metadata = {
                "compile_trace_count": 1,
                "first_call_s": 0.01,
                "warm_call_s": None if self.call_count == 1 else 0.001,
                "chunk_call_s": 0.001,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=base,
                trace=_standard_verification_trace(draws),
                diagnostics=diagnostics,
                metadata=metadata,
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _DivergentChainChunkRunner)
    verifier = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=2,
            max_results=6,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260628, 2),
            chain_count=3,
            rhat_threshold=1.01,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    )

    result = verifier.run()

    assert result.passed is False
    assert result.cap_hit is True
    assert result.retained_sample_count == 6
    assert result.chunk_count == 3
    assert result.diagnostics["all_finite_rhat_at_or_below_threshold"] is False


def test_sequential_rhat_verifier_default_does_not_call_checkpoint_writer(
    monkeypatch,
) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _ScriptedChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            self.call_count += 1
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            samples = tf.zeros((draws, 3, 2), dtype=tf.float64)
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            metadata = {
                "compile_trace_count": 1,
                "first_call_s": 0.01,
                "warm_call_s": None if self.call_count == 1 else 0.001,
                "chunk_call_s": 0.001,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=base,
                trace=_standard_verification_trace(draws),
                diagnostics=diagnostics,
                metadata=metadata,
            )

    def _fail_if_called(**_kwargs):
        raise AssertionError("checkpoint writer must be opt-in")

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _ScriptedChunkRunner)
    monkeypatch.setattr(
        hmc_module,
        "_write_sequential_rhat_private_checkpoint",
        _fail_if_called,
    )
    verifier = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=2,
            max_results=6,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260628, 3),
            chain_count=3,
            rhat_threshold=1.01,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    )

    result = verifier.run()

    assert result.metadata["checkpointing_enabled"] is False
    assert result.metadata["checkpoint_count"] == 0
    assert result.metadata["checkpoint_references"] == []


def test_sequential_rhat_verifier_opt_in_writes_private_checkpoints(
    monkeypatch,
    tmp_path,
) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _DivergentChainChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            self.call_count += 1
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            chain_offsets = tf.reshape(
                tf.constant([-1.0, 0.0, 1.0], dtype=tf.float64),
                (1, 3, 1),
            )
            samples = tf.ones((draws, 3, 2), dtype=tf.float64) * chain_offsets
            samples = samples + tf.cast(self.call_count, tf.float64)
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            metadata = {
                "compile_trace_count": 1,
                "first_call_s": 0.01,
                "warm_call_s": None if self.call_count == 1 else 0.001,
                "chunk_call_s": 0.001,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=base,
                trace=_standard_verification_trace(draws),
                diagnostics=diagnostics,
                metadata=metadata,
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _DivergentChainChunkRunner)
    adapter = ReviewedBatchedGaussianAdapter()
    config = SequentialRHatHMCVerificationConfig(
        check_interval=2,
        max_results=6,
        num_burnin_steps=0,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260628, 4),
        chain_count=3,
        rhat_threshold=1.01,
        target_scope="fixed_size_hmc_chunk_gaussian",
        chain_execution_mode="tf_function",
    )
    theta = tf.constant([[0.25, -0.5]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        target_before = tf.reduce_sum(
            hmc_module._make_hmc_target_log_prob_fn(adapter)(theta)
        )
    grad_before = tape.gradient(target_before, theta)

    baseline = build_sequential_rhat_hmc_verifier(
        adapter,
        tf.zeros((2,), dtype=tf.float64),
        config,
    ).run()
    checkpointed = build_sequential_rhat_hmc_verifier(
        adapter,
        tf.zeros((2,), dtype=tf.float64),
        config,
    ).run(
        checkpoint_writer_config=SequentialRHatCheckpointWriterConfig(
            checkpoint_dir=tmp_path,
            checkpoint_label="phase2",
        )
    )
    with tf.GradientTape() as tape:
        tape.watch(theta)
        target_after = tf.reduce_sum(
            hmc_module._make_hmc_target_log_prob_fn(adapter)(theta)
        )
    grad_after = tape.gradient(target_after, theta)

    assert checkpointed.passed == baseline.passed
    assert checkpointed.cap_hit == baseline.cap_hit
    assert checkpointed.retained_sample_count == baseline.retained_sample_count
    assert checkpointed.chunk_count == baseline.chunk_count == 3
    assert checkpointed.max_finite_rhat == baseline.max_finite_rhat
    assert checkpointed.finite_rhat_count == baseline.finite_rhat_count
    assert checkpointed.nonfinite_rhat_count == baseline.nonfinite_rhat_count
    np.testing.assert_allclose(target_after.numpy(), target_before.numpy())
    np.testing.assert_allclose(grad_after.numpy(), grad_before.numpy())

    assert checkpointed.metadata["checkpointing_enabled"] is True
    assert checkpointed.metadata["checkpoint_count"] == checkpointed.chunk_count
    assert checkpointed.diagnostics["checkpoint_count"] == checkpointed.chunk_count
    references = checkpointed.metadata["checkpoint_references"]
    assert len(references) == checkpointed.chunk_count
    assert checkpointed.diagnostics["checkpoint_references"] == references
    public_text = json.dumps(
        {"diagnostics": checkpointed.diagnostics, "metadata": checkpointed.metadata},
        sort_keys=True,
    )
    assert "target_log_prob" not in public_text
    assert '"log_accept_ratio":' not in public_text
    for reference in references:
        assert_sequential_rhat_checkpoint_public_reference_safe(reference)
        public_text = json.dumps(reference, sort_keys=True)
        for forbidden in (
            "/",
            "\\",
            "samples",
            "valid_mask",
            "final_state",
            "trace",
            "target_log_prob",
            "log_accept",
            "step_size",
            "leapfrog",
            "mass",
            "selected_kernel",
            ".tftensor",
        ):
            assert forbidden not in public_text

    manifest_paths = sorted(tmp_path.glob("phase2_*_manifest.json"))
    assert len(manifest_paths) == checkpointed.chunk_count
    required_roles = {
        "samples",
        "valid_mask",
        "final_state",
        "reduced_trace",
        "target_log_prob_summary",
        "log_accept_ratio_summary",
        "rhat_summary",
    }
    manifests = [
        json.loads(manifest_path.read_text(encoding="utf-8"))
        for manifest_path in manifest_paths
    ]
    manifests_by_checkpoint_id = {
        manifest["checkpoint_id"]: manifest for manifest in manifests
    }
    assert set(manifests_by_checkpoint_id) == {
        reference["checkpoint_id"] for reference in references
    }
    for reference in references:
        manifest = manifests_by_checkpoint_id[reference["checkpoint_id"]]
        manifest_core = dict(manifest)
        observed_core_hash = manifest_core.pop("manifest_core_sha256")
        manifest_core_text = json.dumps(
            manifest_core,
            indent=2,
            sort_keys=True,
        ) + "\n"
        assert hashlib.sha256(manifest_core_text.encode("utf-8")).hexdigest() == (
            observed_core_hash
        )
        assert manifest["checkpoint_id"] == reference["checkpoint_id"]
        assert manifest["manifest_core_sha256"] == reference["checkpoint_sha256"]
        assert set(manifest["private_shards"]) == required_roles
        assert manifest["privacy_contract"]["manifest_contains_private_paths"] is True
        assert manifest["privacy_contract"]["manifest_contains_private_raw_tensors"] is False
        assert manifest["privacy_contract"]["public_reference_contains_paths"] is False
        for role, shard in manifest["private_shards"].items():
            assert shard["role"] == role
            assert (tmp_path / os.path.basename(shard["path"])).exists()
            assert shard["sha256"]
            assert shard["bytes"] > 0

    assert not list(tmp_path.glob(".*.tmp"))


def test_sequential_rhat_private_checkpoint_inspection_verifies_hashes_and_stays_public_safe(
    monkeypatch,
    tmp_path,
) -> None:
    result, manifest_paths = _write_deterministic_sequential_rhat_checkpoints(
        monkeypatch,
        tmp_path,
    )
    references = result.metadata["checkpoint_references"]
    latest_reference = references[-1]
    manifests = {
        json.loads(path.read_text(encoding="utf-8"))["checkpoint_id"]: path
        for path in manifest_paths
    }

    summary = inspect_sequential_rhat_private_checkpoint(
        manifest_path=manifests[latest_reference["checkpoint_id"]],
        public_reference=latest_reference,
    )

    assert summary["artifact_type"] == (
        "bayesfilter_sequential_rhat_checkpoint_inspection_summary"
    )
    assert summary["checkpoint_kind"] == "verification_chunk"
    assert summary["checkpoint_id"] == latest_reference["checkpoint_id"]
    assert summary["checkpoint_sha256"] == latest_reference["checkpoint_sha256"]
    assert summary["manifest_core_hash_verified"] is True
    assert summary["private_payload_hashes_verified"] is True
    assert summary["required_private_payloads_present"] is True
    assert summary["private_payload_count"] == 7
    assert summary["latest_chunk_index"] == 2
    assert summary["retained_sample_count"] == 6
    assert summary["valid_sample_count"] == 2
    assert summary["nonfinite_valid_sample_count"] == 0
    assert summary["finite_checks"] == {
        "retained_values_finite": True,
        "private_target_value_health_passed": True,
        "private_acceptance_log_health_passed": True,
        "rhat_all_finite": True,
    }
    assert summary["resume"]["supported"] is False
    assert summary["resume"]["reason_code"] == "incomplete_continuation_contract"
    assert "no resume claim" in summary["nonclaims"]

    public_text = json.dumps(summary, sort_keys=True)
    for forbidden in (
        "/",
        "\\",
        str(tmp_path),
        "samples",
        "valid_mask",
        "final_state",
        "trace",
        "target_log_prob",
        "log_accept",
        "step_size",
        "num_leapfrog_steps",
        "mass",
        "selected_kernel",
        "shape",
        "dtype",
        "serializer",
        ".tftensor",
    ):
        assert forbidden not in public_text


def test_sequential_rhat_private_checkpoint_inspection_rejects_public_reference_mismatch(
    monkeypatch,
    tmp_path,
) -> None:
    result, manifest_paths = _write_deterministic_sequential_rhat_checkpoints(
        monkeypatch,
        tmp_path,
        checkpoint_label="mismatch",
    )
    references = result.metadata["checkpoint_references"]
    first_manifest = {
        json.loads(path.read_text(encoding="utf-8"))["checkpoint_id"]: path
        for path in manifest_paths
    }[references[0]["checkpoint_id"]]

    with pytest.raises(ValueError, match="does not match private manifest"):
        inspect_sequential_rhat_private_checkpoint(
            manifest_path=first_manifest,
            public_reference=references[-1],
        )


def test_sequential_rhat_private_checkpoint_inspection_rejects_manifest_corruption(
    monkeypatch,
    tmp_path,
) -> None:
    result, manifest_paths = _write_deterministic_sequential_rhat_checkpoints(
        monkeypatch,
        tmp_path,
        checkpoint_label="manifestcorrupt",
    )
    references = result.metadata["checkpoint_references"]
    manifests = {
        json.loads(path.read_text(encoding="utf-8"))["checkpoint_id"]: path
        for path in manifest_paths
    }
    manifest_path = manifests[references[-1]["checkpoint_id"]]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["private_diagnostics"]["retained_sample_count"] = 999
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="manifest_core_sha256 mismatch"):
        inspect_sequential_rhat_private_checkpoint(
            manifest_path=manifest_path,
            public_reference=references[-1],
        )


def test_sequential_rhat_private_checkpoint_inspection_rejects_shard_corruption(
    monkeypatch,
    tmp_path,
) -> None:
    result, manifest_paths = _write_deterministic_sequential_rhat_checkpoints(
        monkeypatch,
        tmp_path,
        checkpoint_label="shardcorrupt",
    )
    references = result.metadata["checkpoint_references"]
    manifests = {
        json.loads(path.read_text(encoding="utf-8"))["checkpoint_id"]: path
        for path in manifest_paths
    }
    manifest_path = manifests[references[-1]["checkpoint_id"]]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = tmp_path / os.path.basename(manifest["private_shards"]["rhat_summary"]["path"])
    original = shard_path.read_text(encoding="utf-8")
    replacement_index = next(index for index, char in enumerate(original) if char == "0")
    corrupted = (
        original[:replacement_index]
        + "1"
        + original[replacement_index + 1 :]
    )
    assert len(corrupted.encode("utf-8")) == len(original.encode("utf-8"))
    shard_path.write_text(corrupted, encoding="utf-8")

    with pytest.raises(ValueError, match="shard hash mismatch"):
        inspect_sequential_rhat_private_checkpoint(
            manifest_path=manifest_path,
            public_reference=references[-1],
        )


def test_sequential_rhat_checkpoint_writer_runs_before_hard_veto_stop(
    monkeypatch,
    tmp_path,
) -> None:
    import bayesfilter.inference.hmc as hmc_module

    class _NonfiniteChunkRunner:
        def __init__(self, _adapter, initial_state, _config) -> None:
            self.call_count = 0

        def run(self, *, active_results, current_state=None, seed=None, step_size=None):
            self.call_count += 1
            draws = int(active_results)
            base = tf.convert_to_tensor(current_state, dtype=tf.float64)
            samples = tf.zeros((draws, 3, 2), dtype=tf.float64)
            valid = tf.ones((draws,), dtype=tf.bool)
            diagnostics = {
                "valid_sample_count": tf.constant(draws, dtype=tf.int32),
                "nonfinite_valid_sample_count": tf.constant(1, dtype=tf.int32),
                "acceptance_rate": tf.constant(0.70, dtype=tf.float64),
                "acceptance_decision_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "log_accept_ratio_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "log_accept_ratio_max_abs_finite": tf.constant(0.2, dtype=tf.float64),
                "target_log_prob_finite_count": tf.constant(3 * draws, dtype=tf.int32),
                "target_log_prob_nonfinite_count": tf.constant(0, dtype=tf.int32),
                "target_log_prob_min_finite": tf.constant(-1.0, dtype=tf.float64),
                "target_log_prob_max_finite": tf.constant(-0.1, dtype=tf.float64),
                "divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            metadata = {
                "compile_trace_count": 1,
                "first_call_s": 0.01,
                "warm_call_s": None,
                "chunk_call_s": 0.001,
            }
            return hmc_module.FixedSizeHMCChunkRunResult(
                samples=samples,
                valid_mask=valid,
                final_state=base,
                trace=_standard_verification_trace(draws),
                diagnostics=diagnostics,
                metadata=metadata,
            )

    monkeypatch.setattr(hmc_module, "FixedSizeHMCChunkRunner", _NonfiniteChunkRunner)
    result = build_sequential_rhat_hmc_verifier(
        ReviewedBatchedGaussianAdapter(),
        tf.zeros((2,), dtype=tf.float64),
        SequentialRHatHMCVerificationConfig(
            check_interval=2,
            max_results=6,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(20260628, 5),
            chain_count=3,
            rhat_threshold=1.01,
            target_scope="fixed_size_hmc_chunk_gaussian",
            chain_execution_mode="tf_function",
        ),
    ).run(
        checkpoint_writer_config=SequentialRHatCheckpointWriterConfig(
            checkpoint_dir=tmp_path,
            checkpoint_label="hardveto",
        )
    )

    assert result.passed is False
    assert result.cap_hit is False
    assert result.retained_sample_count == 2
    assert result.chunk_count == 1
    assert result.metadata["checkpoint_count"] == 1
    assert result.diagnostics["checkpoint_count"] == 1
    assert result.diagnostics["hard_vetoes"] == ["nonfinite_retained_samples"]
    references = result.metadata["checkpoint_references"]
    assert len(references) == 1
    assert_sequential_rhat_checkpoint_public_reference_safe(references[0])

    manifest_paths = sorted(tmp_path.glob("hardveto_*_manifest.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["checkpoint_id"] == references[0]["checkpoint_id"]
    assert manifest["private_diagnostics"]["nonfinite_valid_sample_count"] == 1
    assert set(manifest["private_shards"]) == {
        "samples",
        "valid_mask",
        "final_state",
        "reduced_trace",
        "target_log_prob_summary",
        "log_accept_ratio_summary",
        "rhat_summary",
    }
    public_text = json.dumps(
        {"diagnostics": result.diagnostics, "metadata": result.metadata},
        sort_keys=True,
    )
    assert "target_log_prob" not in public_text
    assert '"log_accept_ratio":' not in public_text
    assert str(tmp_path) not in public_text


def test_fixed_size_chunk_runner_rejects_invalid_contracts_and_inputs() -> None:
    with pytest.raises(ValueError, match="max_results"):
        FixedSizeHMCChunkConfig(
            max_results=0,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(1, 2),
        )
    with pytest.raises(ValueError, match="trace_policy"):
        FixedSizeHMCChunkConfig(
            max_results=2,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(1, 2),
            trace_policy="raw",
        )
    with pytest.raises(ValueError, match="chain_execution_mode"):
        FixedSizeHMCChunkConfig(
            max_results=2,
            num_burnin_steps=0,
            step_size=0.05,
            num_leapfrog_steps=1,
            seed=(1, 2),
            use_xla=True,
            chain_execution_mode="eager",
        )

    runner = build_fixed_size_hmc_chunk_runner(
        ReviewedBatchedGaussianAdapter(),
        _initial_state(),
        _config(use_xla=False, max_results=2),
    )
    with pytest.raises(ValueError, match="current_state shape"):
        runner.run(active_results=1, current_state=tf.zeros((2, 2), dtype=tf.float64))
    with pytest.raises(ValueError, match="seed"):
        runner.run(active_results=1, seed=(1, 2, 3))
    with pytest.raises(ValueError, match="step_size"):
        runner.run(active_results=1, step_size=tf.ones((1,), dtype=tf.float64))
    with pytest.raises(ValueError, match="active_results"):
        runner.run(active_results=3)


def test_fixed_size_chunk_runner_preserves_xla_authority_gate() -> None:
    config = FixedSizeHMCChunkConfig(
        max_results=2,
        num_burnin_steps=0,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260622, 14),
        use_xla=True,
        target_scope="fixed_size_hmc_chunk_target_only",
    )

    with pytest.raises(ValueError, match="target-only XLA readiness is not sufficient"):
        build_fixed_size_hmc_chunk_runner(
            TargetOnlyXLAGaussianAdapter(),
            _initial_state(),
            config,
        )


def test_fixed_size_chunk_runner_standard_trace_returns_valid_row_vectors() -> None:
    config = FixedSizeHMCChunkConfig(
        max_results=5,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260622, 15),
        use_xla=False,
        trace_policy="standard",
        target_scope="fixed_size_hmc_chunk_gaussian",
        chain_execution_mode="eager",
    )
    runner = build_fixed_size_hmc_chunk_runner(
        ReviewedBatchedGaussianAdapter(),
        _initial_state(),
        config,
    )

    result = runner.run(active_results=3)
    mask = np.asarray(result.valid_mask.numpy(), dtype=bool)

    assert mask.tolist() == [True, True, True, False, False]
    assert set(("is_accepted", "log_accept_ratio", "target_log_prob")).issubset(
        result.trace
    )
    assert tuple(result.trace["is_accepted"].shape) == (5, 3)
    assert tuple(result.trace["log_accept_ratio"].shape) == (5, 3)
    assert tuple(result.trace["target_log_prob"].shape) == (5, 3)
    assert np.asarray(result.trace["is_accepted"].numpy())[mask].shape == (3, 3)
    assert np.all(np.isfinite(np.asarray(result.trace["log_accept_ratio"].numpy())[mask]))
    assert np.all(np.isfinite(np.asarray(result.trace["target_log_prob"].numpy())[mask]))


def test_fixed_size_chunk_runner_tf_function_accepts_dynamic_acceptance_trace_shape(
    monkeypatch,
) -> None:
    from collections import namedtuple

    import tensorflow_probability as tfp

    _AcceptedResults = namedtuple("_AcceptedResults", ("target_log_prob",))
    _KernelResults = namedtuple(
        "_KernelResults",
        ("is_accepted", "log_accept_ratio", "accepted_results"),
    )

    class _DynamicAcceptanceShapeHMC:
        def __init__(
            self,
            *,
            target_log_prob_fn,
            step_size,
            num_leapfrog_steps,
        ) -> None:
            self.target_log_prob_fn = target_log_prob_fn
            self.step_size = step_size
            self.num_leapfrog_steps = num_leapfrog_steps

        def _results(self, state: tf.Tensor) -> _KernelResults:
            chain_shape = tf.shape(state)[:1]
            dtype = state.dtype
            return _KernelResults(
                is_accepted=tf.ensure_shape(
                    tf.ones(chain_shape, dtype=tf.bool),
                    [None],
                ),
                log_accept_ratio=tf.ensure_shape(
                    tf.zeros(chain_shape, dtype=dtype),
                    [None],
                ),
                accepted_results=_AcceptedResults(
                    target_log_prob=tf.ensure_shape(
                        tf.zeros(chain_shape, dtype=dtype),
                        [None],
                    ),
                ),
            )

        def bootstrap_results(self, current_state: tf.Tensor) -> _KernelResults:
            return self._results(current_state)

        def one_step(
            self,
            state: tf.Tensor,
            previous_kernel_results: _KernelResults,
            *,
            seed=None,
        ) -> tuple[tf.Tensor, _KernelResults]:
            del previous_kernel_results, seed
            next_state = state + tf.cast(0.001, state.dtype)
            return next_state, self._results(next_state)

    monkeypatch.setattr(
        tfp.mcmc,
        "HamiltonianMonteCarlo",
        _DynamicAcceptanceShapeHMC,
    )
    config = FixedSizeHMCChunkConfig(
        max_results=3,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=1,
        seed=(20260705, 8),
        use_xla=False,
        trace_policy="standard",
        target_scope="fixed_size_hmc_chunk_gaussian",
        chain_execution_mode="tf_function",
    )
    runner = build_fixed_size_hmc_chunk_runner(
        ReviewedBatchedGaussianAdapter(),
        _initial_state(),
        config,
    )

    result = runner.run(active_results=2, seed=(20260705, 9))

    assert tuple(result.trace["is_accepted"].shape) == (3, 3)
    assert tuple(result.trace["log_accept_ratio"].shape) == (3, 3)
    assert tuple(result.trace["target_log_prob"].shape) == (3, 3)
    assert int(result.diagnostics["valid_sample_count"].numpy()) == 2
    assert int(result.diagnostics["acceptance_decision_count"].numpy()) == 6
    assert result.metadata["compile_trace_count"] == 1
    assert result.metadata["trace_policy"] == "standard"


def test_fixed_size_chunk_runner_public_exports_and_no_sample_chain_dependency() -> None:
    import bayesfilter.inference.hmc as hmc_module

    assert bayesfilter.FixedSizeHMCChunkConfig is FixedSizeHMCChunkConfig
    assert bayesfilter.build_fixed_size_hmc_chunk_runner is build_fixed_size_hmc_chunk_runner
    assert "FixedSizeHMCChunkConfig" in bayesfilter.__all__
    assert "build_fixed_size_hmc_chunk_runner" in bayesfilter.__all__

    source = inspect.getsource(hmc_module.FixedSizeHMCChunkRunner)
    assert "tf.while_loop" in source
    assert "sample_chain(" not in source
