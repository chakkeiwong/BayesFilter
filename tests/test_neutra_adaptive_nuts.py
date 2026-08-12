from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_adaptive_nuts import (
    AdaptiveNeuTraNUTSConfig,
    FrozenNeuTraNUTSChunkRunner,
    FrozenNeuTraNUTSKernel,
    compute_frozen_neutra_nuts_qualification,
    compute_neutra_nuts_adaptation_readiness,
    compute_retained_neutra_nuts_diagnostics,
    read_nuts_adaptation_shard,
    read_nuts_tensor_shard,
    run_windowed_adaptive_neutra_nuts,
    write_nuts_adaptation_shard,
    write_nuts_tensor_shard,
)


class GaussianAdapter:
    parameter_dim = 2

    def log_prob_and_grad(self, value):
        tensor = tf.convert_to_tensor(value, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(tensor), axis=-1), -tensor

    def latent_to_position(self, value):
        return tf.convert_to_tensor(value, tf.float64)

    def log_abs_det_jacobian(self, value):
        tensor = tf.convert_to_tensor(value, tf.float64)
        return tf.zeros(tf.shape(tensor)[:-1], tf.float64)

    def target_status_telemetry(self, value):
        tensor = tf.convert_to_tensor(value, tf.float64)
        shape = tf.shape(tensor)[:-1]
        return {
            "status_code": tf.zeros(shape, tf.int32),
            "valid_pre_regularized_score": tf.ones(shape, tf.bool),
            "invalid_count": tf.zeros(shape, tf.int32),
            "roundoff_repair_count": tf.zeros(shape, tf.int32),
        }


def _initial():
    return tf.constant(((2.0, -2.0), (-2.0, 2.0)), tf.float64)


def test_frozen_kernel_payload_roundtrip() -> None:
    kernel = FrozenNeuTraNUTSKernel(
        dimension=2,
        chain_count=2,
        step_size=0.2,
        position_variance=((1.0, 2.0), (1.5, 0.8)),
        target_accept_prob=0.8,
        max_tree_depth=4,
        max_energy_diff=500.0,
    )
    assert FrozenNeuTraNUTSKernel.from_payload(kernel.payload()) == kernel
    changed = dict(kernel.payload())
    changed["step_size"] = 0.3
    with pytest.raises(RuntimeError, match="hash mismatch"):
        FrozenNeuTraNUTSKernel.from_payload(changed)


def test_gaussian_adaptation_extracts_finite_reconstructible_kernel() -> None:
    result = run_windowed_adaptive_neutra_nuts(
        GaussianAdapter(),
        _initial(),
        AdaptiveNeuTraNUTSConfig(
            dimension=2,
            chain_count=2,
            adaptation_results=100,
            max_tree_depth=4,
            seed=(11, 12),
        ),
    )
    assert result.states.shape == (100, 2, 2)
    assert result.frozen_kernel.step_size > 0.0
    assert tf.reduce_all(tf.constant(result.frozen_kernel.position_variance) > 0.0)
    assert result.trace["has_divergence"].shape == (100, 2)
    assert result.trace["has_divergence"].dtype == tf.bool
    assert result.trace["target_score"].shape == (100, 2, 2)


def test_adaptation_shard_is_excluded_and_not_retained(tmp_path) -> None:
    adapted = run_windowed_adaptive_neutra_nuts(
        GaussianAdapter(),
        _initial(),
        AdaptiveNeuTraNUTSConfig(
            dimension=2,
            chain_count=2,
            adaptation_results=100,
            max_tree_depth=4,
            seed=(13, 14),
        ),
    )
    receipt = write_nuts_adaptation_shard(
        adapted,
        initial_state=_initial(),
        path=tmp_path / "adaptation",
        kernel_payload=adapted.frozen_kernel.payload(),
    )
    assert receipt["excluded_from_posterior"] is True
    shard = read_nuts_adaptation_shard(tmp_path / "adaptation")
    readiness = compute_neutra_nuts_adaptation_readiness(
        GaussianAdapter(),
        shard,
        recent_rhat_draws=100,
        recent_mechanics_draws=20,
        rhat_max_inclusive=2.0,
        max_depth_fraction_exclusive=1.0,
    )
    assert readiness["all_finite"] is True
    with pytest.raises(RuntimeError, match="posterior shards"):
        compute_retained_neutra_nuts_diagnostics(
            GaussianAdapter(),
            [shard],
            rhat_max_exclusive=2.0,
            bulk_ess_min=1.0,
            tail_ess_min=1.0,
            mcse_sd_ratio_max=1.0,
            ebfmi_min_exclusive=0.0,
            max_depth_fraction_exclusive=1.0,
        )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        read_nuts_tensor_shard(tmp_path / "adaptation")


def test_frozen_runner_shard_and_diagnostics(tmp_path) -> None:
    adapted = run_windowed_adaptive_neutra_nuts(
        GaussianAdapter(),
        _initial(),
        AdaptiveNeuTraNUTSConfig(
            dimension=2,
            chain_count=2,
            adaptation_results=100,
            max_tree_depth=4,
            seed=(21, 22),
        ),
    )
    runner = FrozenNeuTraNUTSChunkRunner(
        GaussianAdapter(), adapted.states[-1], adapted.frozen_kernel, chunk_size=100
    )
    first = runner.run(adapted.states[-1], (23, 24))
    qualification_receipt = write_nuts_tensor_shard(
        first,
        path=tmp_path / "qualification",
        role="frozen_qualification",
        block_index=0,
        global_start_index=0,
        kernel_payload=adapted.frozen_kernel.payload(),
    )
    assert qualification_receipt["readback_verified"] is True
    qualification = compute_frozen_neutra_nuts_qualification(
        read_nuts_tensor_shard(tmp_path / "qualification"),
        adapted.frozen_kernel,
        max_depth_fraction_exclusive=1.0,
    )
    assert qualification["all_finite"] is True
    assert qualification["step_size_matches"] is True
    tf.debugging.assert_equal(
        read_nuts_tensor_shard(tmp_path / "qualification")["tensors"]["step_size"],
        tf.constant(adapted.frozen_kernel.step_size, tf.float64),
    )
    receipt = write_nuts_tensor_shard(
        first,
        path=tmp_path / "chunk-000",
        role="posterior",
        block_index=0,
        global_start_index=0,
        kernel_payload=adapted.frozen_kernel.payload(),
    )
    assert receipt["readback_verified"] is True
    shard = read_nuts_tensor_shard(tmp_path / "chunk-000")
    diagnostics = compute_retained_neutra_nuts_diagnostics(
        GaussianAdapter(),
        [shard],
        rhat_max_exclusive=2.0,
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
        mcse_sd_ratio_max=1.0,
        ebfmi_min_exclusive=0.0,
        max_depth_fraction_exclusive=1.0,
    )
    assert diagnostics["all_finite"] is True
    assert diagnostics["divergence_count_by_chain"] == [0, 0]
