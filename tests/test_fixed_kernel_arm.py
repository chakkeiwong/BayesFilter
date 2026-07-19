from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
import bayesfilter.inference.fixed_kernel_arm as arm_module
from bayesfilter.inference import (
    FixedKernelArmConfig,
    FixedKernelArmResult,
    FixedSizeHMCChunkRunResult,
    ValueScoreCapability,
    minimum_latent_ess,
    run_fixed_kernel_arm,
)


class _GaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "fixed-kernel-arm-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow",
            evidence_path="tests/test_fixed_kernel_arm.py",
            target_scope="fixed_kernel_arm_gaussian",
            nonclaims=("tiny fixed-kernel arm fixture only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.cast(tf.convert_to_tensor(theta), tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value


def _config(**overrides) -> FixedKernelArmConfig:
    values = {
        "label": "A",
        "transition_count": 32,
        "step_size": 0.05,
        "num_leapfrog_steps": 1,
        "seed": (20260718, 1),
        "target_scope": "fixed_kernel_arm_gaussian",
        "use_xla": False,
    }
    values.update(overrides)
    return FixedKernelArmConfig(**values)


def _chunk(samples: np.ndarray, *, divergence_available: bool = False):
    draws, chains, _coordinates = samples.shape
    diagnostics = {
        "valid_sample_count": tf.constant(draws, tf.int32),
        "nonfinite_valid_sample_count": tf.constant(0, tf.int32),
        "target_log_prob_nonfinite_count": tf.constant(0, tf.int32),
        "log_accept_ratio_nonfinite_count": tf.constant(0, tf.int32),
        "acceptance_rate": tf.constant(0.7, tf.float64),
        "native_divergence_available": tf.constant(divergence_available),
        "divergence_status": (
            "available" if divergence_available else "not_exposed_by_kernel"
        ),
        "divergence_count": tf.constant(0, tf.int32),
    }
    return FixedSizeHMCChunkRunResult(
        samples=tf.constant(samples, tf.float64),
        valid_mask=tf.ones((draws,), tf.bool),
        final_state=tf.constant(samples[-1], tf.float64),
        trace={},
        diagnostics=diagnostics,
        metadata={
            "runtime": "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop",
            "compile_trace_count": 1,
        },
    )


def test_minimum_latent_ess_matches_direct_tfp_authority() -> None:
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(128, 3, 4))
    from bayesfilter.inference.hmc_warmup import _summed_chain_ess

    expected = _summed_chain_ess(samples)

    report = minimum_latent_ess(samples)

    np.testing.assert_allclose(report["effective_sample_size_by_coordinate"], expected)
    assert report["minimum_effective_sample_size"] == float(np.min(expected))
    assert report["sample_shape"] == (128, 3, 4)
    assert report["posterior_convergence_claim"] is False


def test_minimum_latent_ess_accepts_single_chain_and_fails_closed() -> None:
    report = minimum_latent_ess(np.arange(64.0).reshape(32, 2))
    assert report["chain_count"] == 1
    assert report["single_chain_allowed"] is True

    with pytest.raises(ValueError, match="at least four"):
        minimum_latent_ess(np.zeros((3, 1, 2)))
    bad = np.zeros((8, 1, 2))
    bad[0, 0, 0] = np.nan
    with pytest.raises(tf.errors.InvalidArgumentError, match="finite"):
        minimum_latent_ess(bad)


def test_minimum_latent_ess_checkpoints_match_exact_prefixes_without_vectors() -> None:
    samples = np.random.default_rng(8).normal(size=(16, 1, 3))

    summaries = arm_module.minimum_latent_ess_checkpoints(
        samples, (4, 8, 16), ess_threshold=9.0
    )

    assert tuple(summaries) == ("4", "8", "16")
    for checkpoint in (4, 8, 16):
        direct = minimum_latent_ess(samples[:checkpoint])
        summary = summaries[str(checkpoint)]
        assert summary["state_count"] == checkpoint
        assert summary["minimum_effective_sample_size"] == pytest.approx(
            direct["minimum_effective_sample_size"]
        )
        assert summary["maximum_effective_sample_size"] == pytest.approx(
            direct["maximum_effective_sample_size"]
        )
        assert summary["coordinate_count_below_threshold"] == sum(
            value < 9.0
            for value in direct["effective_sample_size_by_coordinate"]
        )
        assert "effective_sample_size_by_coordinate" not in summary


def test_minimum_latent_ess_checkpoint_validation_fails_closed() -> None:
    samples = np.random.default_rng(12).normal(size=(16, 1, 3))
    for checkpoints in ((), (3,), (8, 8), (12, 8), (17,)):
        with pytest.raises(ValueError, match="checkpoints"):
            arm_module.minimum_latent_ess_checkpoints(samples, checkpoints)
    with pytest.raises(ValueError, match="ess_threshold"):
        arm_module.minimum_latent_ess_checkpoints(
            samples, (8,), ess_threshold=float("nan")
        )


def test_run_fixed_kernel_arm_passes_explicit_state_and_frozen_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    samples = np.random.default_rng(9).normal(size=(32, 1, 2))

    class _Runner:
        def run(self, **kwargs):
            calls.append(kwargs)
            return _chunk(samples)

    def build(_adapter, state, config):
        calls.append({"build_state": state, "build_config": config})
        return _Runner()

    monkeypatch.setattr(arm_module, "build_fixed_size_hmc_chunk_runner", build)
    initial = tf.constant([[0.2, -0.1]], tf.float64)
    config = _config()

    result = run_fixed_kernel_arm(_GaussianAdapter(), initial, config)

    assert isinstance(result, FixedKernelArmResult)
    built = calls[0]
    assert built["build_config"].num_burnin_steps == 0
    assert built["build_config"].num_leapfrog_steps == 1
    assert built["build_config"].trace_policy == "standard"
    assert calls[1]["active_results"] == 32
    assert calls[1]["seed"] == (20260718, 1)
    assert calls[1]["step_size"] == pytest.approx(0.05)
    np.testing.assert_array_equal(calls[1]["current_state"].numpy(), initial.numpy())
    assert result.diagnostics["divergence_status"] == "not_exposed_by_kernel"
    assert result.diagnostics["divergence_count"] is None
    assert result.metadata["adaptation_policy"] == "fixed_kernel_no_adaptation"
    payload = result.payload()
    assert payload["raw_samples_publicized"] is False
    assert payload["final_state_publicized"] is False
    text = json.dumps(payload, sort_keys=True)
    assert '"samples":' not in text
    assert '"final_state":' not in text
    assert payload["metadata"]["privacy_contract"][
        "public_summary_contains_hmc_mechanics"
    ] is True
    assert "ess_checkpoints" not in payload["arm"]
    assert "ess_checkpoint_summaries" not in payload["diagnostics"]


def test_run_fixed_kernel_arm_emits_opt_in_same_chain_checkpoint_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.random.default_rng(13).normal(size=(16, 1, 2))

    class _Runner:
        def run(self, **_kwargs):
            return _chunk(samples)

    monkeypatch.setattr(
        arm_module,
        "build_fixed_size_hmc_chunk_runner",
        lambda *_args, **_kwargs: _Runner(),
    )
    result = run_fixed_kernel_arm(
        _GaussianAdapter(),
        tf.zeros((1, 2), tf.float64),
        _config(transition_count=16, ess_checkpoints=(4, 8, 16)),
    )

    summaries = result.diagnostics["ess_checkpoint_summaries"]
    assert tuple(summaries) == ("4", "8", "16")
    assert result.config.payload()["ess_checkpoints"] == (4, 8, 16)
    assert all(
        "effective_sample_size_by_coordinate" not in summary
        for summary in summaries.values()
    )


def test_run_fixed_kernel_arm_rejects_count_and_nonfinite_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.random.default_rng(10).normal(size=(32, 1, 2))

    class _Runner:
        def __init__(self, result):
            self.result = result

        def run(self, **_kwargs):
            return self.result

    mismatch = _chunk(samples)
    mismatch.diagnostics["valid_sample_count"] = tf.constant(31, tf.int32)
    monkeypatch.setattr(
        arm_module,
        "build_fixed_size_hmc_chunk_runner",
        lambda *_args, **_kwargs: _Runner(mismatch),
    )
    with pytest.raises(ValueError, match="transition budget"):
        run_fixed_kernel_arm(
            _GaussianAdapter(), tf.zeros((1, 2), tf.float64), _config()
        )

    invalid = _chunk(samples)
    invalid.diagnostics["target_log_prob_nonfinite_count"] = tf.constant(1, tf.int32)
    monkeypatch.setattr(
        arm_module,
        "build_fixed_size_hmc_chunk_runner",
        lambda *_args, **_kwargs: _Runner(invalid),
    )
    with pytest.raises(ValueError, match="target evaluation"):
        run_fixed_kernel_arm(
            _GaussianAdapter(), tf.zeros((1, 2), tf.float64), _config()
        )

    invalid_log_accept = _chunk(samples)
    invalid_log_accept.diagnostics["log_accept_ratio_nonfinite_count"] = tf.constant(
        1, tf.int32
    )
    monkeypatch.setattr(
        arm_module,
        "build_fixed_size_hmc_chunk_runner",
        lambda *_args, **_kwargs: _Runner(invalid_log_accept),
    )
    with pytest.raises(ValueError, match="log acceptance ratio"):
        run_fixed_kernel_arm(
            _GaussianAdapter(), tf.zeros((1, 2), tf.float64), _config()
        )


def test_fixed_kernel_arm_config_validation_and_public_exports() -> None:
    with pytest.raises(ValueError, match="transition_count"):
        _config(transition_count=0)
    with pytest.raises(ValueError, match="step_size"):
        _config(step_size=float("nan"))
    with pytest.raises(ValueError, match="num_leapfrog_steps"):
        _config(num_leapfrog_steps=0)
    with pytest.raises(ValueError, match="ess_checkpoints"):
        _config(ess_checkpoints=(8, 8))
    with pytest.raises(ValueError, match="ess_threshold"):
        _config(ess_threshold=0.0)
    assert bayesfilter.FixedKernelArmConfig is FixedKernelArmConfig
    assert bayesfilter.minimum_latent_ess is minimum_latent_ess
    assert bayesfilter.run_fixed_kernel_arm is run_fixed_kernel_arm


def test_real_fixed_kernel_arm_smoke_has_no_adaptation() -> None:
    result = run_fixed_kernel_arm(
        _GaussianAdapter(),
        tf.zeros((1, 2), tf.float64),
        _config(transition_count=16),
    )

    assert result.diagnostics["valid_sample_count"] == 16
    assert result.diagnostics["finite"] is True
    assert result.metadata["num_burnin_steps"] == 0
    assert result.metadata["adaptation_policy"] == "fixed_kernel_no_adaptation"
    assert result.metadata["runtime"] == (
        "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop"
    )


def test_real_fixed_kernel_arm_accepts_single_chain_vector_state() -> None:
    result = run_fixed_kernel_arm(
        _GaussianAdapter(),
        tf.zeros((2,), tf.float64),
        _config(transition_count=16),
    )

    assert result.diagnostics["sample_shape"] == (16, 1, 2)
    assert result.diagnostics["chain_count"] == 1
    assert result.diagnostics["valid_sample_count"] == 16
