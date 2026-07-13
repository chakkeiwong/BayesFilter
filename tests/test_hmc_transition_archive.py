from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import FixedSizeHMCChunkConfig, ValueScoreCapability
from bayesfilter.inference.hmc import build_fixed_size_hmc_chunk_runner
from bayesfilter.inference.hmc_transition_archive import (
    HMCExactMechanicsIdentityPolicy,
    HMCTransitionArchiveConfig,
    TRANSITION_TENSOR_KEYS,
    build_hmc_transition_archive_runner,
    read_hmc_transition_shard,
    summarize_hmc_exact_mechanics_identity,
    write_hmc_transition_shard,
)


class ReviewedGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "hmc-transition-archive-reviewed-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_transition_archive.py",
            target_scope="hmc_transition_archive_gaussian",
            nonclaims=("tiny transition archive fixture only",),
            full_chain_xla_diagnostic_ready=True,
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values


def _initial() -> tf.Tensor:
    return tf.constant([[1.0, 2.0], [-1.0, 0.5], [0.2, -0.7]], tf.float64)


def _config(
    max_results: int,
    *,
    use_xla: bool = False,
    step_size: float = 0.2,
    leapfrog: int = 3,
) -> HMCTransitionArchiveConfig:
    return HMCTransitionArchiveConfig(
        max_results=max_results,
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
        master_seed=(20260712, 29),
        use_xla=use_xla,
        target_scope="hmc_transition_archive_gaussian",
        chain_execution_mode="tf_function" if use_xla else "eager",
    )


def _active(result, key: str) -> np.ndarray:
    mask = np.asarray(result.valid_mask.numpy(), dtype=bool)
    return np.asarray(result.tensors[key].numpy())[mask]


def test_mixed_identity_policy_is_dtype_aware_and_scale_appropriate() -> None:
    policy = HMCExactMechanicsIdentityPolicy()
    assert policy.atol == pytest.approx(2.0e-10)
    assert policy.rtol == pytest.approx(64.0 * np.finfo(np.float64).eps)

    # Exact Phase 29 false positive: the 7.45e-9 residual is binary64
    # cancellation at a 5.69e7 operand scale.
    large_left = np.array([[56893713.15427512]], dtype=np.float64)
    large_right = np.array([[56893713.15427511]], dtype=np.float64)
    large = summarize_hmc_exact_mechanics_identity(
        large_left,
        large_right,
        identity_name="large_scale_roundoff",
        policy=policy,
    )
    assert large["passed"] is True
    assert large["max_absolute_residual"] == pytest.approx(
        7.450580596923828e-09
    )
    assert large["max_scaled_residual"] <= 1.0

    small = summarize_hmc_exact_mechanics_identity(
        np.array([3.0e-10], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        identity_name="small_absolute_failure",
        policy=policy,
    )
    assert small["passed"] is False
    assert small["max_scaled_residual"] > 1.0

    with pytest.raises(ValueError, match="tested float64"):
        summarize_hmc_exact_mechanics_identity(
            np.array([1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            identity_name="untested_dtype",
        )


def test_mixed_identity_rejects_wrong_sign_convention_and_nonfinite_operands() -> None:
    delta_h = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64)
    log_accept = -delta_h
    correct = summarize_hmc_exact_mechanics_identity(
        delta_h,
        -log_accept,
        identity_name="delta_h_equals_negative_log_accept_ratio",
    )
    wrong_sign = summarize_hmc_exact_mechanics_identity(
        delta_h,
        log_accept,
        identity_name="wrong_hamiltonian_sign",
    )
    assert correct["passed"] is True
    assert wrong_sign["passed"] is False
    assert wrong_sign["worst_index"] == (1, 1)
    assert wrong_sign["worst_left_operand"] == pytest.approx(-0.4)
    assert wrong_sign["worst_right_operand"] == pytest.approx(0.4)

    nonfinite = summarize_hmc_exact_mechanics_identity(
        np.array([0.0, np.nan], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        identity_name="nonfinite_operand",
    )
    assert nonfinite["passed"] is False
    assert nonfinite["all_operands_finite"] is False
    assert nonfinite["worst_index"] == (1,)
    assert nonfinite["numeric_summary_available"] is False
    assert nonfinite["max_absolute_residual"] is None
    assert nonfinite["max_scaled_residual"] is None
    assert "NaN" not in json.dumps(nonfinite, allow_nan=False)


def test_mixed_identity_rejects_wrong_kinetic_convention() -> None:
    initial = np.array([[2.0, 4.0]], dtype=np.float64)
    final = np.array([[0.5, 1.0]], dtype=np.float64)
    correction = initial - final
    correct = summarize_hmc_exact_mechanics_identity(
        correction,
        initial - final,
        identity_name="correct_kinetic_correction",
    )
    wrong = summarize_hmc_exact_mechanics_identity(
        correction,
        final - initial,
        identity_name="wrong_kinetic_correction",
    )
    assert correct["passed"] is True
    assert wrong["passed"] is False


def test_archive_shapes_hamiltonian_identities_and_rejection_semantics() -> None:
    runner = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(),
        _initial(),
        _config(4, step_size=5.0, leapfrog=3),
    )
    result = runner.run(active_results=3, global_start_index=11)

    assert set(result.tensors) == set(TRANSITION_TENSOR_KEYS)
    assert tuple(result.tensors["pre_state"].shape) == (4, 3, 2)
    assert tuple(result.tensors["initial_energy"].shape) == (4, 3)
    assert tuple(result.tensors["metropolis_seed"].shape) == (4, 2)
    assert _active(result, "transition_index").tolist() == [11, 12, 13]
    np.testing.assert_allclose(
        _active(result, "delta_h"),
        -_active(result, "log_accept_ratio"),
        rtol=0.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        _active(result, "log_acceptance_correction"),
        _active(result, "initial_kinetic_energy")
        - _active(result, "final_kinetic_energy"),
        rtol=0.0,
        atol=2e-7,
    )
    rejected = ~_active(result, "is_accepted")
    assert np.any(rejected)
    pre = _active(result, "pre_state")
    proposal = _active(result, "proposed_state")
    post = _active(result, "post_state")
    np.testing.assert_array_equal(post[rejected], pre[rejected])
    assert np.any(proposal[rejected] != pre[rejected])
    assert np.any(_active(result, "initial_momentum")[rejected] != 0.0)
    assert bool(result.diagnostics["transition_state_continuity"].numpy())
    assert result.diagnostics["exact_mechanics_identities_passed"] is True
    assert result.diagnostics["hamiltonian_identity_diagnostic"]["passed"] is True
    assert (
        result.diagnostics["kinetic_correction_identity_diagnostic"]["passed"]
        is True
    )


@pytest.mark.parametrize("use_xla", [False, True])
def test_one_call_equals_two_continued_calls_by_absolute_seed(use_xla: bool) -> None:
    whole = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(6, use_xla=use_xla)
    ).run(active_results=6, global_start_index=0)
    runner = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(3, use_xla=use_xla)
    )
    first = runner.run(active_results=3, global_start_index=0)
    second = runner.run(
        active_results=3,
        global_start_index=3,
        current_state=first.final_state,
    )

    for key in TRANSITION_TENSOR_KEYS:
        continued = np.concatenate((_active(first, key), _active(second, key)), axis=0)
        np.testing.assert_array_equal(_active(whole, key), continued)
    np.testing.assert_array_equal(whole.final_state.numpy(), second.final_state.numpy())
    if use_xla:
        assert second.metadata["compile_trace_count"] == 1


def test_exact_512_call_equals_two_256_continuation_calls() -> None:
    whole = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(512, use_xla=True)
    ).run(active_results=512, global_start_index=0)
    runner = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(256, use_xla=True)
    )
    first = runner.run(active_results=256, global_start_index=0)
    second = runner.run(
        active_results=256,
        global_start_index=256,
        current_state=first.final_state,
    )

    for key in TRANSITION_TENSOR_KEYS:
        continued = np.concatenate((_active(first, key), _active(second, key)), axis=0)
        np.testing.assert_array_equal(_active(whole, key), continued)
    np.testing.assert_array_equal(whole.final_state.numpy(), second.final_state.numpy())
    assert second.metadata["compile_trace_count"] == 1


@pytest.mark.parametrize("use_xla", [False, True])
def test_archive_matches_signed_runner_within_each_runtime(use_xla: bool) -> None:
    mode = "tf_function" if use_xla else "eager"
    seed = (20260712, 29)
    adapter = ReviewedGaussianAdapter()
    candidate = build_hmc_transition_archive_runner(
        adapter, _initial(), _config(8, use_xla=use_xla)
    ).run(active_results=8, global_start_index=0)
    reference = build_fixed_size_hmc_chunk_runner(
        adapter,
        _initial(),
        FixedSizeHMCChunkConfig(
            max_results=8,
            num_burnin_steps=0,
            step_size=0.2,
            num_leapfrog_steps=3,
            seed=seed,
            use_xla=use_xla,
            trace_policy="standard",
            target_scope="hmc_transition_archive_gaussian",
            chain_execution_mode=mode,
        ),
    ).run(active_results=8, seed=seed)

    np.testing.assert_array_equal(
        reference.samples.numpy(), _active(candidate, "post_state")
    )
    np.testing.assert_array_equal(
        reference.trace["is_accepted"].numpy(), _active(candidate, "is_accepted")
    )
    np.testing.assert_array_equal(
        reference.trace["target_log_prob"].numpy(),
        _active(candidate, "post_target_log_prob"),
    )
    np.testing.assert_array_equal(
        reference.trace["log_accept_ratio"].numpy(),
        _active(candidate, "log_accept_ratio"),
    )


def test_instrumented_outputs_match_signed_fixed_size_runner_exactly() -> None:
    adapter = ReviewedGaussianAdapter()
    initial = _initial()
    seed = (20260712, 2900)
    reference = build_fixed_size_hmc_chunk_runner(
        adapter,
        initial,
        FixedSizeHMCChunkConfig(
            max_results=1,
            num_burnin_steps=0,
            step_size=0.2,
            num_leapfrog_steps=3,
            seed=seed,
            use_xla=True,
            trace_policy="standard",
            target_scope="hmc_transition_archive_gaussian",
            chain_execution_mode="tf_function",
        ),
    ).run(active_results=1, seed=seed)
    candidate = build_hmc_transition_archive_runner(
        adapter,
        initial,
        HMCTransitionArchiveConfig(
            max_results=1,
            step_size=0.2,
            num_leapfrog_steps=3,
            master_seed=seed,
            use_xla=True,
            target_scope="hmc_transition_archive_gaussian",
        ),
    ).run(active_results=1, global_start_index=0)

    np.testing.assert_array_equal(reference.samples.numpy(), candidate.tensors["post_state"].numpy())
    np.testing.assert_array_equal(
        reference.trace["is_accepted"].numpy(), candidate.tensors["is_accepted"].numpy()
    )
    np.testing.assert_array_equal(
        reference.trace["target_log_prob"].numpy(),
        candidate.tensors["post_target_log_prob"].numpy(),
    )
    np.testing.assert_array_equal(
        reference.trace["log_accept_ratio"].numpy(),
        candidate.tensors["log_accept_ratio"].numpy(),
    )


def test_atomic_shard_round_trip_and_schema(tmp_path: Path) -> None:
    result = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(4)
    ).run(active_results=3, global_start_index=20)
    path = tmp_path / "block_000.npz"
    record = write_hmc_transition_shard(
        result,
        path=path,
        role="warmup_diagnostic",
        block_index=0,
        metadata={"kernel_hash": "a" * 64},
    )
    readback = read_hmc_transition_shard(path)

    assert record["readback_verified"] is True
    assert record["global_start_index"] == 20
    assert record["global_end_index_exclusive"] == 23
    assert readback.sha256 == record["sha256"]
    for key in TRANSITION_TENSOR_KEYS:
        np.testing.assert_array_equal(readback.tensors[key], _active(result, key))
    with pytest.raises(FileExistsError):
        write_hmc_transition_shard(
            result,
            path=path,
            role="warmup_diagnostic",
            block_index=0,
        )


def test_atomic_shard_accepts_phase30_posterior_role(tmp_path: Path) -> None:
    result = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(4)
    ).run(active_results=2, global_start_index=0)
    record = write_hmc_transition_shard(
        result,
        path=tmp_path / "posterior.npz",
        role="posterior",
        block_index=0,
    )

    assert record["role"] == "posterior"
    assert read_hmc_transition_shard(record["path"]).metadata["role"] == "posterior"


def test_archive_configuration_and_source_fail_closed() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        build_hmc_transition_archive_runner(
            ReviewedGaussianAdapter(), tf.zeros((2,), tf.float64), _config(2)
        )
    runner = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(2)
    )
    with pytest.raises(ValueError, match="active_results"):
        runner.run(active_results=3, global_start_index=0)
    with pytest.raises(ValueError, match="global_start_index"):
        runner.run(active_results=1, global_start_index=-1)
    with pytest.raises(ValueError, match="integer scalar"):
        runner.run(active_results=1.5, global_start_index=0)
    with pytest.raises(ValueError, match="integer scalar"):
        runner.run(active_results=1, global_start_index=np.array([0]))
    source = inspect.getsource(type(runner))
    assert "tf.while_loop" in source
    assert "sample_chain(" not in source
    assert "tf.map_fn" not in source


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_results", 2.5, "integer scalar"),
        ("num_leapfrog_steps", True, "integer scalar"),
        ("master_seed", (20260712.5, 29), "integer scalar"),
        ("use_xla", "false", "must be boolean"),
    ],
)
def test_archive_config_rejects_coerced_authority_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "max_results": 2,
        "step_size": 0.2,
        "num_leapfrog_steps": 3,
        "master_seed": (20260712, 29),
        "use_xla": False,
        "target_scope": "hmc_transition_archive_gaussian",
        "chain_execution_mode": "eager",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        HMCTransitionArchiveConfig(**kwargs)


def test_no_overwrite_publish_is_race_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(2)
    ).run(active_results=1, global_start_index=0)
    path = tmp_path / "raced.npz"
    real_link = os.link

    def racing_link(source, destination):
        Path(destination).write_bytes(b"historical")
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        write_hmc_transition_shard(
            result,
            path=path,
            role="instrumentation_canary",
            block_index=0,
        )

    assert path.read_bytes() == b"historical"
    assert tuple(tmp_path.glob("*.tmp.*.npz")) == ()


def test_transition_writer_requires_explicit_boolean_overwrite(tmp_path: Path) -> None:
    result = build_hmc_transition_archive_runner(
        ReviewedGaussianAdapter(), _initial(), _config(2)
    ).run(active_results=1, global_start_index=0)

    with pytest.raises(ValueError, match="must be boolean"):
        write_hmc_transition_shard(
            result,
            path=tmp_path / "block.npz",
            role="instrumentation_canary",
            block_index=0,
            overwrite="false",
        )
