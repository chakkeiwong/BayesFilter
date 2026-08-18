"""Diagnostic source-parity tests for the smooth varying-Hessian target."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_varying_hessian_target import (
    FrozenAffineLiftWeightedTransport,
    VaryingHessianValueScoreAdapter,
    affine_ridge_tangent_mixture_proposal,
    affine_scale_mixture_proposal,
    fit_defensive_branch_mixture_proposal,
    fit_reflected_positive_branch_mixture_proposal,
    load_varying_hessian_target_spec,
    reflect_first_local_coordinate,
    varying_hessian_log_prob_and_score_batch,
)
from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)
from bayesfilter.testing.importance_sampling_tf import validate_gaussian_mixture


ROOT = Path(__file__).resolve().parents[1]
DSGE_ROOT = Path("/home/ubuntu/python/dsge_hmc")
SOURCE = DSGE_ROOT / "src/dsge_hmc/benchmarks/nk_like_mild.py"
STRONG = DSGE_ROOT / (
    "results/neutra/gate3/nk_strong_smooth_bridge_20260604/frozen_constants/"
    "strong_smooth_from_seed42_affine_lift.json"
)


def _spec():
    return load_varying_hessian_target_spec(STRONG, expected_name="nk_like_strong_smooth")


def _source_module():
    spec = importlib.util.spec_from_file_location("source_nk_like_mild", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_source_constants_hash_is_bound_and_target_is_strong_smooth() -> None:
    spec = _spec()
    assert spec.name == "nk_like_strong_smooth"
    assert spec.dimension == 9
    assert spec.constants_sha256 == hashlib.sha256(STRONG.read_bytes()).hexdigest()
    assert spec.rot_alpha == pytest.approx(0.70)
    assert spec.weak_collapse == pytest.approx(0.9)
    assert spec.stiff_growth == pytest.approx(0.45)


def test_bayesfilter_batch_value_matches_source_tensorflow_formula() -> None:
    spec = _spec()
    source = _source_module()
    source_constants = source.NKLikeMildConstants.from_json_dict(
        {
            "name": spec.name,
            "dim": spec.dimension,
            "mu": list(spec.mu),
            "lchol": [list(row) for row in spec.lchol],
            "rot_alpha": spec.rot_alpha,
            "weak_collapse": spec.weak_collapse,
            "stiff_growth": spec.stiff_growth,
            "smooth": True,
        }
    )
    rows = tf.random.stateless_normal((17, 9), seed=(20260812, 15001), dtype=tf.float64)
    actual, _score = varying_hessian_log_prob_and_score_batch(spec, rows)
    expected = source.log_prob_batch_tf(source_constants, rows)
    tf.debugging.assert_near(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_explicit_score_matches_gradient_tape_reference() -> None:
    spec = _spec()
    rows = tf.random.stateless_normal((11, 9), seed=(20260812, 15002), dtype=tf.float64)
    actual_value, actual_score = varying_hessian_log_prob_and_score_batch(spec, rows)
    with tf.GradientTape() as tape:
        tape.watch(rows)
        expected_value, _ignored = varying_hessian_log_prob_and_score_batch(spec, rows)
        total = tf.reduce_sum(expected_value)
    expected_score = tape.gradient(total, rows)
    tf.debugging.assert_near(actual_value, expected_value, atol=1.0e-14, rtol=1.0e-14)
    tf.debugging.assert_near(actual_score, expected_score, atol=3.0e-11, rtol=3.0e-11)


def test_adapter_status_and_affine_scale_proposal_are_batch_native_and_finite() -> None:
    spec = _spec()
    adapter = VaryingHessianValueScoreAdapter(spec)
    proposal = affine_scale_mixture_proposal(spec)
    probabilities, means, covariances, _ = validate_gaussian_mixture(
        proposal["probabilities"], proposal["means"], proposal["covariances"]
    )
    assert probabilities.shape == (4,)
    assert means.shape == (4, 9)
    assert covariances.shape == (4, 9, 9)
    value, score, status = adapter.log_prob_and_grad_status(
        tf.zeros((5, 9), tf.float64)
    )
    tf.debugging.assert_all_finite(value, "value")
    tf.debugging.assert_all_finite(score, "score")
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())


def test_ridge_tangent_proposal_is_finite_full_support_mixture() -> None:
    spec = _spec()
    proposal = affine_ridge_tangent_mixture_proposal(spec)
    probabilities, means, covariances, _ = validate_gaussian_mixture(
        proposal["probabilities"], proposal["means"], proposal["covariances"]
    )
    assert proposal["identity"] == "affine_lift_smooth_ridge_tangent_mixture_v1"
    assert probabilities.shape == (5,)
    assert means.shape == (5, 9)
    assert covariances.shape == (5, 9, 9)


def test_ridge_tangent_proposal_accepts_a_broad_tail_profile() -> None:
    spec = _spec()
    proposal = affine_ridge_tangent_mixture_proposal(
        spec,
        radii=(0.0, 8.0, 8.0),
        signs=(0.0, 1.0, -1.0),
        weak_scales=(3.0, 25.0, 25.0),
        stiff_scales=(1.5, 1.0, 1.0),
        probabilities=(0.2, 0.4, 0.4),
    )
    probabilities, means, covariances, _ = validate_gaussian_mixture(
        proposal["probabilities"], proposal["means"], proposal["covariances"]
    )
    assert probabilities.shape == (3,)
    assert means.shape == (3, 9)
    assert covariances.shape == (3, 9, 9)


def test_pilot_branch_proposal_is_finite_and_records_branch_ess() -> None:
    spec = _spec()
    defensive = affine_ridge_tangent_mixture_proposal(spec)
    from bayesfilter.testing.importance_sampling_tf import gaussian_mixture_log_prob, sample_gaussian_mixture

    rows, _ = sample_gaussian_mixture(
        8_192,
        defensive["probabilities"],
        defensive["means"],
        defensive["covariances"],
        seed=(20260812, 15003),
    )
    del gaussian_mixture_log_prob
    fitted = fit_defensive_branch_mixture_proposal(
        spec,
        rows,
        tf.zeros((8_192,), tf.float64),
        defensive_proposal=defensive,
    )
    probabilities, means, covariances, _ = validate_gaussian_mixture(
        fitted["probabilities"], fitted["means"], fitted["covariances"]
    )
    assert probabilities.shape == (7,)
    assert means.shape == (7, 9)
    assert covariances.shape == (7, 9, 9)
    assert bool(tf.reduce_all(fitted["pilot_branch_effective_sample_size"] >= 20.0).numpy())


def test_source_bound_target_is_exactly_symmetric_under_first_local_reflection() -> None:
    spec = _spec()
    rows = tf.random.stateless_normal((29, 9), seed=(20260812, 15004), dtype=tf.float64)
    reflected = reflect_first_local_coordinate(spec, rows)
    value, score = varying_hessian_log_prob_and_score_batch(spec, rows)
    reflected_value, reflected_score = varying_hessian_log_prob_and_score_batch(spec, reflected)
    lchol = tf.constant(spec.lchol, tf.float64)
    local_score = tf.linalg.matvec(lchol, score, transpose_a=True)
    reflected_local_score = tf.linalg.matvec(lchol, reflected_score, transpose_a=True)
    expected_reflected_local_score = tf.concat(
        (-local_score[:, :1], local_score[:, 1:]), axis=1
    )
    value_difference = tf.abs(value - reflected_value)
    value_scale = tf.maximum(tf.abs(value), tf.abs(reflected_value))
    score_difference = tf.abs(reflected_local_score - expected_reflected_local_score)
    score_scale = tf.maximum(
        tf.abs(reflected_local_score), tf.abs(expected_reflected_local_score)
    )
    tf.debugging.assert_less_equal(
        value_difference, 1.0e-9 + 1.0e-10 * value_scale
    )
    tf.debugging.assert_less_equal(
        score_difference, 1.0e-10 + 1.0e-10 * score_scale
    )


def test_reflected_positive_pilot_proposal_has_equal_learned_branch_mass() -> None:
    spec = _spec()
    defensive = affine_ridge_tangent_mixture_proposal(spec)
    from bayesfilter.testing.importance_sampling_tf import sample_gaussian_mixture

    rows, _ = sample_gaussian_mixture(
        8_192,
        defensive["probabilities"],
        defensive["means"],
        defensive["covariances"],
        seed=(20260812, 15005),
    )
    fitted = fit_reflected_positive_branch_mixture_proposal(
        spec,
        rows,
        tf.zeros((8_192,), tf.float64),
        defensive_proposal=defensive,
    )
    probabilities, means, covariances, _ = validate_gaussian_mixture(
        fitted["probabilities"], fitted["means"], fitted["covariances"]
    )
    assert fitted["identity"] == "affine_lift_reflected_positive_pilot_defensive_mixture_v1"
    assert probabilities.shape == (7,)
    assert means.shape == (7, 9)
    assert covariances.shape == (7, 9, 9)
    assert float(probabilities[-1].numpy()) == pytest.approx(float(probabilities[-2].numpy()))
    assert float(fitted["pilot_positive_branch_effective_sample_size"].numpy()) >= 20.0


def test_affine_lift_weighted_transport_score_and_logdet_match_gradient_tape() -> None:
    spec = _spec()
    local_transport = WeightedDenseIAFTransport(
        WeightedNeuTraConfig(
            dimension=9,
            hidden_layers=(8, 8),
            stages=2,
            initialization_seed=(20260812, 15006),
            jit_compile=True,
        )
    )
    local_transport.bind_frozen_identity(
        {
            "checkpoint_sha256": "0" * 64,
            "training_state_hash": "1" * 64,
            "transport_tensor_hash": "2" * 64,
        }
    )
    transport = FrozenAffineLiftWeightedTransport(spec, local_transport)
    latent = tf.random.stateless_normal((13, 9), seed=(20260812, 15007), dtype=tf.float64)
    physical, logdet = transport.forward_and_logdet(latent)
    value, score = varying_hessian_log_prob_and_score_batch(spec, physical)
    expected = transport.pullback_score_batch(latent, score) + transport.log_abs_det_jacobian_score_batch(latent)
    with tf.GradientTape() as tape:
        tape.watch(latent)
        physical_tape, logdet_tape = transport.forward_and_logdet(latent)
        value_tape, _ = varying_hessian_log_prob_and_score_batch(spec, physical_tape)
        total = tf.reduce_sum(value_tape + logdet_tape)
    actual = tape.gradient(total, latent)
    tf.debugging.assert_near(logdet, logdet_tape, atol=1.0e-12, rtol=1.0e-12)
    tf.debugging.assert_near(expected, actual, atol=7.0e-10, rtol=7.0e-10)
    recovered = transport.inverse_physical_to_latent_batch(physical)
    tf.debugging.assert_near(recovered, latent, atol=1.0e-11, rtol=1.0e-11)
