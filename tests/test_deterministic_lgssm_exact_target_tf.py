from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.runtime import stable_config_hash
from bayesfilter.inference.neutra_batching import (
    InvalidNeuTraBatchTarget,
    bind_batch_native_neutra_target,
)
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    InvalidDeterministicLGSSMTarget,
    load_deterministic_lgssm_exact_target,
)
from bayesfilter.testing import deterministic_lgssm_exact_target_tf as exact_module
from docs.benchmarks import run_multidim_lgssm_serious_hmc_tuning_2026_07_09 as legacy


def _legacy_adapter(bundle):
    return legacy.DeterministicLGSSMPosteriorAdapter(
        observations=bundle.fixture["observations"],
        contract=bundle.contract,
        parameter_names=bundle.parameter_names,
        evidence_path="phase0-parity-test",
    )


def test_exact_target_identity_is_fixture_and_source_bound() -> None:
    bundle = load_deterministic_lgssm_exact_target()

    assert len(bundle.target_signature) == 64
    assert len(bundle.adapter.adapter_signature()) == 64
    assert bundle.adapter.target_signature == bundle.target_signature
    assert bundle.raw_truth.shape == (18,)
    assert bundle.parameter_names == tuple(bundle.fixture["parameter_names"])
    assert bundle.target_signature_payload["fixture_artifact_hash"] == (
        bundle.fixture["artifact_hash"]
    )
    assert set(bundle.target_signature_payload["target_source_files"]) == {
        "bayesfilter/testing/multidim_triangular_lgssm_tf.py",
        "bayesfilter/linear/kalman_svd_derivatives_tf.py",
    }


def test_exact_adapter_matches_completed_campaign_adapter() -> None:
    bundle = load_deterministic_lgssm_exact_target()
    old = _legacy_adapter(bundle)
    perturbation = tf.random.stateless_normal(
        (5, 18), seed=(20260713, 901), stddev=1.0e-3, dtype=tf.float64
    )
    points = bundle.raw_truth[tf.newaxis, :] + perturbation

    new_value, new_score = bundle.adapter.log_prob_and_grad(points)
    old_value, old_score = old.log_prob_and_grad(points)
    np.testing.assert_allclose(new_value.numpy(), old_value.numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_score.numpy(), old_score.numpy(), rtol=0.0, atol=0.0)

    new_status = bundle.adapter.target_status_telemetry(points)
    old_status = old.target_status_telemetry(points)
    assert set(new_status) == set(old_status)
    for name in new_status:
        np.testing.assert_allclose(
            new_status[name].numpy(), old_status[name].numpy(), rtol=0.0, atol=0.0
        )


def test_exact_adapter_batch_value_score_xla_compiles() -> None:
    bundle = load_deterministic_lgssm_exact_target()
    points = tf.stack(
        [bundle.raw_truth, bundle.raw_truth + tf.constant(1.0e-4, tf.float64)],
        axis=0,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(theta):
        return bundle.adapter.log_prob_and_grad(theta)

    value, score = compiled(points)
    assert value.shape == (2,)
    assert score.shape == (2, 18)
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_exact_adapter_batch_native_binding_and_same_regime_parity() -> None:
    bundle = load_deterministic_lgssm_exact_target()
    points = tf.stack(
        [bundle.raw_truth, bundle.raw_truth + tf.constant(1.0e-4, tf.float64)],
        axis=0,
    )
    binding = bind_batch_native_neutra_target(
        bundle.adapter,
        target_signature=bundle.target_signature,
    )
    batch_value, batch_score, batch_status = binding.invoke(points)

    @tf.function(input_signature=[tf.TensorSpec((18,), tf.float64)], jit_compile=True)
    def scalar_compiled(theta):
        return bundle.adapter._single_log_prob_grad_status(theta)

    scalar = tuple(scalar_compiled(points[index]) for index in range(2))
    np.testing.assert_allclose(
        batch_value.numpy(),
        [item[0].numpy() for item in scalar],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        batch_score.numpy(),
        tf.stack([item[1] for item in scalar]).numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    for name in batch_status:
        expected = tf.stack([item[2][name] for item in scalar])
        if batch_status[name].dtype.is_floating:
            np.testing.assert_allclose(
                batch_status[name].numpy(),
                expected.numpy(),
                rtol=2e-10,
                atol=2e-12,
            )
        else:
            np.testing.assert_array_equal(
                batch_status[name].numpy(), expected.numpy()
            )

    payload = binding.payload()
    assert payload["method_name"] == "neutra_batch_log_prob_and_grad_status"
    assert payload["scalar_fallback_used"] is False
    assert payload["row_mapped_scalar_target_used"] is False
    assert payload["sample_axis_python_loop_used"] is False
    assert {item["module"] for item in payload["dependency_module_sources"]} == {
        "bayesfilter.linear.batched_kalman_svd_derivatives_tf",
        "bayesfilter.testing.multidim_triangular_lgssm_batched_tf",
    }


def test_exact_adapter_binding_rejects_live_helper_replacement(monkeypatch) -> None:
    bundle = load_deterministic_lgssm_exact_target()
    binding = bind_batch_native_neutra_target(
        bundle.adapter,
        target_signature=bundle.target_signature,
    )
    called = False

    def replacement(*args, **kwargs):
        nonlocal called
        called = True
        return exact_module.materialize_lower_triangular_lgssm_batch(*args, **kwargs)

    monkeypatch.setattr(
        exact_module,
        "materialize_lower_triangular_lgssm_batch",
        replacement,
    )
    with pytest.raises(InvalidNeuTraBatchTarget, match="dependency closure"):
        binding.invoke(tf.stack((bundle.raw_truth, bundle.raw_truth), axis=0))
    assert called is False


def test_exact_adapter_batch_invalid_row_is_nan_gated_and_isolated() -> None:
    bundle = load_deterministic_lgssm_exact_target()
    points = tf.stack(
        (bundle.raw_truth, bundle.raw_truth, bundle.raw_truth), axis=0
    )
    invalid = tf.tensor_scatter_nd_update(
        points,
        tf.constant([[1, 0]], tf.int32),
        tf.constant([float("nan")], tf.float64),
    )
    mixed_value, mixed_score, mixed_status = (
        bundle.adapter.neutra_batch_log_prob_and_grad_status(invalid)
    )
    regular_value, regular_score, _regular_status = (
        bundle.adapter.neutra_batch_log_prob_and_grad_status(points)
    )

    assert mixed_status["status_code"].numpy().tolist() == [0, 2, 0]
    assert mixed_status["valid_pre_regularized_score"].numpy().tolist() == [
        True,
        False,
        True,
    ]
    assert bool(tf.math.is_nan(mixed_value[1]).numpy())
    assert bool(tf.reduce_all(tf.math.is_nan(mixed_score[1])).numpy())
    np.testing.assert_allclose(
        tf.gather(mixed_value, [0, 2]).numpy(),
        tf.gather(regular_value, [0, 2]).numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tf.gather(mixed_score, [0, 2]).numpy(),
        tf.gather(regular_score, [0, 2]).numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_exact_target_rejects_tampered_fixture(tmp_path: Path) -> None:
    bundle = load_deterministic_lgssm_exact_target()
    tampered = dict(bundle.fixture)
    tampered["observations"] = [list(row) for row in tampered["observations"]]
    tampered["observations"][0][0] += 1.0e-6
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(InvalidDeterministicLGSSMTarget, match="artifact hash mismatch"):
        load_deterministic_lgssm_exact_target(fixture_path=path)


def test_exact_target_recomputed_fixture_changes_signature(tmp_path: Path) -> None:
    bundle = load_deterministic_lgssm_exact_target()
    changed = dict(bundle.fixture)
    changed["observations"] = [list(row) for row in changed["observations"]]
    changed["observations"][0][0] += 1.0e-6
    changed.pop("artifact_hash")
    changed["artifact_hash"] = f"sha256:{stable_config_hash(changed)}"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(InvalidDeterministicLGSSMTarget, match="target signature mismatch"):
        load_deterministic_lgssm_exact_target(
            fixture_path=path,
            expected_target_signature=bundle.target_signature,
        )
