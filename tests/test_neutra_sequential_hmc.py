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


def tiny_config() -> SequentialNeuTraHMCConfig:
    return SequentialNeuTraHMCConfig(
        step_size=0.05,
        num_leapfrog_steps=1,
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
