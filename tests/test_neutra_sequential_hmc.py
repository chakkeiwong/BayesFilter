from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_hmc import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    NeuTraHMCError,
    SequentialNeuTraHMCConfig,
    run_sequential_neutra_hmc,
    sequential_chunk_seed,
)
from bayesfilter.inference import neutra_hmc as neutra_hmc_module


class GaussianStatusAdapter:
    parameter_dim = 2

    def __init__(self, *, status_valid: bool = True) -> None:
        self.status_valid = bool(status_valid)
        self.status_batch_sizes = []

    def log_prob_and_grad(self, z):
        values = tf.convert_to_tensor(z, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

    def latent_to_position(self, z):
        return tf.convert_to_tensor(z, tf.float64)

    def log_prob_and_grad_status(self, z):
        value, score = self.log_prob_and_grad(z)
        count = tf.shape(value)[0]
        self.status_batch_sizes.append(int(z.shape[0]))
        valid = tf.fill((count,), self.status_valid)
        return value, score, {
            "status_code": tf.where(valid, tf.zeros_like(valid, tf.int32), tf.ones_like(valid, tf.int32)),
            "valid_pre_regularized_score": valid,
            "floor_count_value": tf.zeros((count,), tf.int32),
            "min_innovation_eigenvalue": tf.ones((count,), tf.float64),
            "innovation_condition_estimate": tf.ones((count,), tf.float64),
        }


class GaussianStatusWithoutConditionAdapter(GaussianStatusAdapter):
    def log_prob_and_grad_status(self, z):
        value, score, status = super().log_prob_and_grad_status(z)
        status = dict(status)
        status.pop("innovation_condition_estimate")
        return value, score, status


def tiny_config() -> SequentialNeuTraHMCConfig:
    return SequentialNeuTraHMCConfig(
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260722, 1),
        warmup_chunk_size=4,
        warmup_min_results=4,
        warmup_window_results=4,
        warmup_max_results=4,
        retained_chunk_size=4,
        retained_min_results=4,
        retained_max_results=4,
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
        acceptance_max=1.0,
        chain_count=2,
        use_xla=False,
    )


def test_policy_defaults_match_repository_sequential_contract() -> None:
    config = SequentialNeuTraHMCConfig(
        step_size=0.1,
        num_leapfrog_steps=4,
        seed=(1, 2),
    )
    assert config.payload()["policy_id"] == NEUTRA_SEQUENTIAL_HMC_POLICY_ID
    assert config.warmup_min_results == 2000
    assert config.warmup_window_results == 1000
    assert config.warmup_max_results == 10000
    assert config.retained_min_results == 1000
    assert config.retained_max_results == 10000
    assert config.warmup_rhat_max == pytest.approx(1.05)
    assert config.retained_rhat_max == pytest.approx(1.01)
    assert config.delta_h_abs_max == pytest.approx(1000.0)
    assert config.acceptance_min == pytest.approx(0.35)
    assert config.acceptance_max == pytest.approx(0.95)


def test_sequential_hmc_forbids_one_leapfrog_step() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        SequentialNeuTraHMCConfig(
            step_size=0.1,
            num_leapfrog_steps=1,
            seed=(1, 2),
        )


def test_chunk_policy_uses_only_declared_mechanics_vetoes() -> None:
    unavailable = neutra_hmc_module._chunk_policy_vetoes(
        samples_finite=True,
        log_accept_finite=True,
        target_finite=True,
        proposed_finite=True,
        target_score_finite=True,
        delta_h_finite=True,
        target_status_passed=True,
        acceptance_probability_by_chain=(0.35, 0.95),
        acceptance_min=0.35,
        acceptance_max=0.95,
        native_divergence_status="not_exposed_by_kernel",
        native_divergence_count=None,
    )
    assert unavailable == ()

    divergent = neutra_hmc_module._chunk_policy_vetoes(
        samples_finite=True,
        log_accept_finite=True,
        target_finite=True,
        proposed_finite=True,
        target_score_finite=True,
        delta_h_finite=True,
        target_status_passed=True,
        acceptance_probability_by_chain=(0.5, 0.7),
        acceptance_min=0.35,
        acceptance_max=0.95,
        native_divergence_status="available",
        native_divergence_count=1,
    )
    assert divergent == ("positive_native_divergence",)

    acceptance_veto = neutra_hmc_module._chunk_policy_vetoes(
        samples_finite=True,
        log_accept_finite=True,
        target_finite=True,
        proposed_finite=True,
        target_score_finite=True,
        delta_h_finite=True,
        target_status_passed=True,
        acceptance_probability_by_chain=(0.34, 0.96),
        acceptance_min=0.35,
        acceptance_max=0.95,
        native_divergence_status="not_exposed_by_kernel",
        native_divergence_count=None,
    )
    assert acceptance_veto == ("acceptance_probability_outside_declared_bounds",)


def test_sequential_controller_accepts_exact_external_chunk_callback(tmp_path) -> None:
    calls = []

    def run_chunk(state, seed, config):
        state = tf.convert_to_tensor(state, tf.float64)
        calls.append(
            {
                "state_shape": tuple(state.shape),
                "seed": tuple(seed),
                "leapfrog": config.num_leapfrog_steps,
            }
        )
        offsets = tf.reshape(tf.range(1, 5, dtype=tf.float64), (4, 1, 1))
        samples = state[tf.newaxis, :, :] + offsets * 0.01
        shape = (4, 2)
        trace = {
            "is_accepted": tf.ones(shape, tf.bool),
            "log_accept_ratio": tf.fill(
                shape, tf.math.log(tf.constant(0.7, tf.float64))
            ),
            "target_log_prob": tf.zeros(shape, tf.float64),
            "proposed_target_log_prob": tf.zeros(shape, tf.float64),
            "target_score": tf.zeros((4, 2, 2), tf.float64),
            "delta_h": tf.fill(
                shape, -tf.math.log(tf.constant(0.7, tf.float64))
            ),
            "target_status_code": tf.zeros(shape, tf.int32),
            "target_valid_pre_regularized_score": tf.ones(shape, tf.bool),
            "target_floor_count_value": tf.zeros(shape, tf.int32),
            "target_min_innovation_eigenvalue": tf.ones(shape, tf.float64),
        }
        return samples, trace

    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.zeros((2, 2), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "external",
        archive_label="external",
        run_chunk=run_chunk,
    )
    assert calls
    assert {call["state_shape"] for call in calls} == {(2, 2)}
    assert {call["leapfrog"] for call in calls} == {2}
    assert Path(result.archive["manifest_path"]).is_file()
    assert result.diagnostics["hard_vetoes"] == []


def test_target_status_from_trace_vetoes_invalid_transition() -> None:
    diagnostics = neutra_hmc_module._target_status_from_trace(
        {
            "target_status_code": tf.constant(((0, 1),), tf.int32),
            "target_valid_pre_regularized_score": tf.constant(((True, False),)),
            "target_floor_count_value": tf.constant(((0, 1),), tf.int32),
            "target_min_innovation_eigenvalue": tf.constant(((1.0, 0.0),), tf.float64),
        }
    )
    assert diagnostics is not None
    assert diagnostics["passed"] is False
    assert diagnostics["status_nonvalid_count"] == 1


def test_chunk_seeds_are_phase_separated_and_deterministic() -> None:
    first = sequential_chunk_seed((4, 5), phase_index=0, chunk_index=0)
    assert first == sequential_chunk_seed((4, 5), phase_index=0, chunk_index=0)
    assert first != sequential_chunk_seed((4, 5), phase_index=0, chunk_index=1)
    assert first != sequential_chunk_seed((4, 5), phase_index=1, chunk_index=0)


def test_movement_is_measured_from_the_immediate_pre_chunk_state() -> None:
    pre_chunk = tf.constant(((3.0, 4.0), (5.0, 6.0)), tf.float64)
    frozen = tf.repeat(pre_chunk[None, ...], repeats=4, axis=0)
    assert neutra_hmc_module._chain_moved(pre_chunk, frozen).numpy().tolist() == [
        False,
        False,
    ]


def test_tiny_run_archives_warmup_separately_and_excludes_it(tmp_path) -> None:
    adapter = GaussianStatusAdapter()
    result = run_sequential_neutra_hmc(
        adapter,
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="tiny",
    )
    assert result.warmup_results_per_chain == 4
    assert result.retained_results_per_chain in {0, 4}
    assert result.metadata["warmup_excluded_from_posterior"] is True
    assert set(result.diagnostics["warmup"]) >= {
        "hmc_coordinates",
        "model_parameters",
    }
    assert result.archive["warmup_chunk_count"] == 1
    assert list((tmp_path / "run" / "warmup").glob("*samples.tftensor"))
    assert adapter.status_batch_sizes
    assert set(adapter.status_batch_sizes) == {8}
    assert list((tmp_path / "run" / "warmup").glob("*delta_h.tftensor"))
    if result.retained_results_per_chain:
        assert list((tmp_path / "run" / "retained").glob("*samples.tftensor"))


def test_target_status_failure_is_hard_veto(tmp_path) -> None:
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(status_valid=False),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="status-veto",
    )
    assert result.passed is False
    assert result.stop_reason == "hard_veto"
    assert "target_status_veto" in result.diagnostics["hard_vetoes"]


def test_target_status_accepts_q20_schema_without_condition_estimate() -> None:
    diagnostics = neutra_hmc_module._target_status(
        GaussianStatusWithoutConditionAdapter(),
        tf.zeros((4, 2, 2), tf.float64),
    )
    assert diagnostics["passed"] is True
    assert diagnostics["maximum_innovation_condition_estimate"] is None
    assert diagnostics["innovation_condition_estimate_status"] == (
        "not_exposed_by_target"
    )


def test_run_rejects_nonempty_output_root(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "existing").write_text("occupied", encoding="utf-8")
    with pytest.raises(NeuTraHMCError, match="new or empty"):
        run_sequential_neutra_hmc(
            GaussianStatusAdapter(),
            tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
            tiny_config(),
            archive_root=root,
            archive_label="occupied",
        )


def test_budget_refusal_is_archived_as_a_resource_cap(tmp_path) -> None:
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="budget-cap",
        budget_check=lambda _requested_work: False,
    )
    assert result.passed is False
    assert result.stop_reason == "hard_veto"
    assert "campaign_resource_cap" in result.diagnostics["hard_vetoes"]
    assert result.archive["warmup_chunk_count"] == 0


def test_archived_result_payload_uses_defined_schema(tmp_path) -> None:
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="payload-schema",
        budget_check=lambda _requested_work: False,
    )
    payload = result.payload()
    assert payload["schema"] == "bayesfilter.neutra.sequential_hmc_result.v1"
    assert payload["stop_reason"] == "hard_veto"
