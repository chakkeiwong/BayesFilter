from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import evaluate_batch_native_value_score
from bayesfilter.ssm import (
    stable_ssm_posterior_adapter_signature,
    stable_ssm_target_signature,
)
from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
    LGSSM_GENERIC_TARGET_NONCLAIMS,
    LGSSM_GENERIC_TARGET_XLA_HMC_NONCLAIMS,
    lgssm_gaussian_prior_log_prob_and_grad,
    lgssm_qr_log_likelihood_and_grad,
    make_lgssm_generic_target_contract,
    make_lgssm_generic_target_fixture,
)


def test_lgssm_generic_target_admitted_score_route_uses_no_gradient_tape() -> None:
    source = inspect.getsource(lgssm_qr_log_likelihood_and_grad)
    assert "GradientTape" not in source
    assert "tape.gradient" not in source


def test_lgssm_generic_target_contract_is_exact_stable_and_untransported() -> None:
    left = make_lgssm_generic_target_contract()
    right = make_lgssm_generic_target_contract()
    changed = make_lgssm_generic_target_contract(model_hash="sha256:changed-model")

    assert left.problem.static_shape.manifest_payload() == {
        "horizon": 4,
        "state_dim": 1,
        "observation_dim": 1,
        "innovation_dim": 1,
        "parameter_dim": 2,
    }
    assert left.filter_program.approximation_semantics == "exact"
    assert left.filter_program.deterministic_target_policy == "deterministic"
    assert "GradientTape" not in left.filter_program.filter_manifest["score_source"]
    assert "analytical QR Kalman score" in left.filter_program.filter_manifest["score_source"]
    assert left.frozen_transport is None
    assert stable_ssm_target_signature(left) == stable_ssm_target_signature(right)
    assert stable_ssm_target_signature(left) != stable_ssm_target_signature(changed)


def test_lgssm_generic_target_adapter_emits_finite_batch_values_scores() -> None:
    fixture = make_lgssm_generic_target_fixture()
    theta = tf.constant([[0.20, -1.05], [0.05, -0.80]], dtype=tf.float64)

    value, score = fixture.adapter.log_prob_and_grad(theta)
    result = evaluate_batch_native_value_score(fixture.adapter, theta, dtype=tf.float64)

    assert value.shape == (2,)
    assert score.shape == (2, 2)
    assert result.metadata.rank == "batch"
    assert result.metadata.parameter_dim == 2
    assert tf.reduce_all(tf.math.is_finite(value))
    assert tf.reduce_all(tf.math.is_finite(score))
    assert fixture.adapter.metadata().batch_rank_policy == "rank2_required"
    assert fixture.adapter.metadata().nonclaims == LGSSM_GENERIC_TARGET_XLA_HMC_NONCLAIMS
    capability = fixture.adapter.value_score_capability()
    assert capability.value_score_authority == "graph_native"
    assert capability.xla_hmc_ready is True
    assert capability.is_accepted_xla_hmc_authority is True
    assert capability.full_chain_xla_diagnostic_ready is False
    assert capability.nonclaims == LGSSM_GENERIC_TARGET_XLA_HMC_NONCLAIMS


def test_lgssm_generic_target_matches_prior_plus_qr_likelihood() -> None:
    fixture = make_lgssm_generic_target_fixture()
    theta = tf.constant([[0.20, -1.05], [0.05, -0.80]], dtype=tf.float64)

    value, score = fixture.adapter.log_prob_and_grad(theta)
    prior_value, prior_score = lgssm_gaussian_prior_log_prob_and_grad(
        theta,
        prior_scale=fixture.source_target.prior_scale,
    )
    likelihood_value, likelihood_score = lgssm_qr_log_likelihood_and_grad(
        theta,
        source_target=fixture.source_target,
    )

    tf.debugging.assert_near(value, prior_value + likelihood_value)
    tf.debugging.assert_near(score, prior_score + likelihood_score)


def test_lgssm_generic_target_batch_of_one_matches_source_rank1_target() -> None:
    fixture = make_lgssm_generic_target_fixture()
    theta = fixture.initial_batch

    value, score = fixture.adapter.log_prob_and_grad(theta)
    source_value, source_score = fixture.source_target.target_log_prob_and_grad(theta[0])

    np.testing.assert_allclose(value.numpy(), [source_value.numpy()], atol=1.0e-12)
    np.testing.assert_allclose(score.numpy(), source_score[tf.newaxis, :].numpy(), atol=1.0e-10)


def test_lgssm_generic_target_rejects_rank1_position() -> None:
    fixture = make_lgssm_generic_target_fixture()

    with pytest.raises(ValueError, match="rank 2 theta"):
        fixture.adapter.log_prob_and_grad(
            tf.constant([0.20, -1.05], dtype=tf.float64)
        )


def test_lgssm_generic_target_signature_and_manifest_are_stable() -> None:
    fixture = make_lgssm_generic_target_fixture()
    same = make_lgssm_generic_target_fixture()
    changed_contract = make_lgssm_generic_target_contract(prior_hash="sha256:changed-prior")
    changed_fixture = make_lgssm_generic_target_fixture()
    changed_adapter = type(fixture.adapter)(
        contract=changed_contract,
        prior_log_prob_and_grad=lambda theta: lgssm_gaussian_prior_log_prob_and_grad(
            theta,
            prior_scale=changed_fixture.source_target.prior_scale,
        ),
        filter_log_likelihood_and_grad=lambda theta: lgssm_qr_log_likelihood_and_grad(
            theta,
            source_target=changed_fixture.source_target,
        ),
        dtype=tf.float64,
        target_scope="lgssm-static-qr-generic-target-fixture",
        evidence_path="bayesfilter/testing/lgssm_generic_target_adapter_tf.py",
        nonclaims=LGSSM_GENERIC_TARGET_NONCLAIMS,
    )

    assert fixture.adapter.manifest_payload()["target_signature"] == stable_ssm_target_signature(
        fixture.contract
    )
    assert stable_ssm_posterior_adapter_signature(
        fixture.adapter
    ) == stable_ssm_posterior_adapter_signature(same.adapter)
    assert stable_ssm_posterior_adapter_signature(
        fixture.adapter
    ) != stable_ssm_posterior_adapter_signature(changed_adapter)


def test_lgssm_generic_target_score_matches_finite_difference_reference() -> None:
    fixture = make_lgssm_generic_target_fixture()
    theta = fixture.initial_batch
    _value, score = fixture.adapter.log_prob_and_grad(theta)
    eps = tf.constant(1.0e-5, dtype=tf.float64)
    columns = []
    for dim in range(2):
        basis = tf.one_hot(dim, 2, dtype=tf.float64)[tf.newaxis, :]
        plus = fixture.adapter.log_prob(theta + eps * basis)[0]
        minus = fixture.adapter.log_prob(theta - eps * basis)[0]
        columns.append((plus - minus) / (2.0 * eps))
    finite_difference = tf.stack(columns, axis=0)[tf.newaxis, :]

    tf.debugging.assert_near(score, finite_difference, rtol=1.0e-5, atol=1.0e-5)


def test_lgssm_generic_target_likelihood_score_matches_analytic_qr_diagnostic() -> None:
    fixture = make_lgssm_generic_target_fixture()
    theta = fixture.initial_batch
    likelihood_value, likelihood_score = lgssm_qr_log_likelihood_and_grad(
        theta,
        source_target=fixture.source_target,
    )
    analytic = fixture.source_target.analytic_score_hessian(theta[0])

    np.testing.assert_allclose(
        likelihood_value.numpy(),
        [analytic.log_likelihood.numpy()],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        likelihood_score.numpy(),
        analytic.score[tf.newaxis, :].numpy(),
        rtol=1.0e-8,
        atol=1.0e-8,
    )
