from __future__ import annotations

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
from bayesfilter.testing.simple_nonlinear_generic_target_adapter_tf import (
    MODEL_B_INITIAL_BATCH,
    SIMPLE_NONLINEAR_GENERIC_TARGET_NONCLAIMS,
    SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID,
    SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID,
    SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
    SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID,
    simple_nonlinear_admitted_filter_ids,
    simple_nonlinear_filter_route_inventory,
    make_simple_nonlinear_generic_target_contract,
    make_simple_nonlinear_generic_target_fixture,
    simple_nonlinear_filter_diagnostics,
    simple_nonlinear_gaussian_prior_log_prob_and_grad,
    simple_nonlinear_sigma_point_log_likelihood_and_grad,
    simple_nonlinear_svd_cubature_log_likelihood_and_grad,
    simple_nonlinear_svd_ukf_log_likelihood_and_grad,
)


def _theta_batch() -> tf.Tensor:
    return tf.constant(
        [
            [0.68, 0.24, 0.78],
            [0.70, 0.25, 0.80],
        ],
        dtype=tf.float64,
    )


def _likelihood_for_filter_id(filter_id: str):
    if filter_id == SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID:
        return simple_nonlinear_svd_ukf_log_likelihood_and_grad
    if filter_id == SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID:
        return simple_nonlinear_svd_cubature_log_likelihood_and_grad
    raise AssertionError(f"unexpected admitted filter id: {filter_id}")


def test_simple_nonlinear_contract_is_stable_deterministic_and_untransported() -> None:
    left = make_simple_nonlinear_generic_target_contract()
    right = make_simple_nonlinear_generic_target_contract()
    changed = make_simple_nonlinear_generic_target_contract(
        filter_hash="sha256:changed-filter",
    )

    assert left.problem.static_shape.manifest_payload() == {
        "horizon": 3,
        "state_dim": 2,
        "observation_dim": 1,
        "innovation_dim": 1,
        "parameter_dim": 3,
    }
    assert left.problem.problem_id == "model-b-nonlinear-accumulation-generic-target-fixture"
    assert left.filter_program.approximation_semantics == "deterministic_approximation"
    assert left.filter_program.deterministic_target_policy == "deterministic"
    assert left.frozen_transport is None
    assert stable_ssm_target_signature(left) == stable_ssm_target_signature(right)
    assert stable_ssm_target_signature(left) != stable_ssm_target_signature(changed)


def test_simple_nonlinear_filter_route_inventory_is_explicit_and_fail_closed() -> None:
    inventory = simple_nonlinear_filter_route_inventory()
    by_id = {entry["filter_id"]: entry for entry in inventory}

    assert simple_nonlinear_admitted_filter_ids() == (
        SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID,
        SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID,
    )
    assert by_id[SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID]["status"] == "admitted"
    assert by_id[SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID]["status"] == "admitted"
    assert by_id[SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID]["status"] == "deferred"
    assert by_id[SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID]["status"] == "deferred"
    assert "dedicated generic-adapter" in by_id[SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID]["reason"]
    assert "principal-sqrt" in by_id[SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID]["reason"]


def test_simple_nonlinear_adapter_emits_finite_batch_values_scores() -> None:
    fixture = make_simple_nonlinear_generic_target_fixture()
    theta = _theta_batch()

    value, score = fixture.adapter.log_prob_and_grad(theta)
    result = evaluate_batch_native_value_score(fixture.adapter, theta, dtype=tf.float64)

    assert value.shape == (2,)
    assert score.shape == (2, 3)
    assert result.metadata.rank == "batch"
    assert result.metadata.parameter_dim == 3
    assert tf.reduce_all(tf.math.is_finite(value))
    assert tf.reduce_all(tf.math.is_finite(score))
    assert fixture.adapter.metadata().batch_rank_policy == "rank2_required"
    assert fixture.adapter.metadata().nonclaims == SIMPLE_NONLINEAR_GENERIC_TARGET_NONCLAIMS
    capability = fixture.adapter.value_score_capability()
    assert capability.xla_hmc_ready is True
    assert capability.full_chain_xla_diagnostic_ready is True


@pytest.mark.parametrize("filter_id", simple_nonlinear_admitted_filter_ids())
@pytest.mark.parametrize("jit_compile", [False, True])
def test_simple_nonlinear_admitted_filter_adapters_emit_finite_batch_values_scores(
    filter_id: str,
    jit_compile: bool,
) -> None:
    fixture = make_simple_nonlinear_generic_target_fixture(
        filter_id=filter_id, jit_compile=jit_compile,
    )
    theta = _theta_batch()

    value, score = fixture.adapter.log_prob_and_grad(theta)
    result = evaluate_batch_native_value_score(fixture.adapter, theta, dtype=tf.float64)

    assert fixture.contract.filter_program.filter_id == filter_id
    assert fixture.contract.filter_program.approximation_semantics == (
        "deterministic_approximation"
    )
    assert value.shape == (2,)
    assert score.shape == (2, 3)
    assert result.metadata.rank == "batch"
    assert tf.reduce_all(tf.math.is_finite(value))
    assert tf.reduce_all(tf.math.is_finite(score))
    capability = fixture.adapter.value_score_capability()
    assert capability.xla_hmc_ready is True
    # The explicit non-XLA diagnostic configuration cannot claim a compiled
    # full chain. Both admitted filters have numerical compilation regressions.
    assert capability.full_chain_xla_diagnostic_ready is jit_compile


def test_simple_nonlinear_adapter_matches_prior_plus_svd_ukf_likelihood() -> None:
    fixture = make_simple_nonlinear_generic_target_fixture()
    theta = _theta_batch()

    value, score = fixture.adapter.log_prob_and_grad(theta)
    prior_value, prior_score = simple_nonlinear_gaussian_prior_log_prob_and_grad(theta)
    likelihood_value, likelihood_score = simple_nonlinear_svd_ukf_log_likelihood_and_grad(theta)

    tf.debugging.assert_near(value, prior_value + likelihood_value)
    tf.debugging.assert_near(score, prior_score + likelihood_score)


@pytest.mark.parametrize("filter_id", simple_nonlinear_admitted_filter_ids())
def test_simple_nonlinear_admitted_adapter_matches_prior_plus_filter_likelihood(
    filter_id: str,
) -> None:
    fixture = make_simple_nonlinear_generic_target_fixture(filter_id=filter_id)
    theta = _theta_batch()

    value, score = fixture.adapter.log_prob_and_grad(theta)
    prior_value, prior_score = simple_nonlinear_gaussian_prior_log_prob_and_grad(theta)
    likelihood_value, likelihood_score = _likelihood_for_filter_id(filter_id)(theta)

    tf.debugging.assert_near(value, prior_value + likelihood_value)
    tf.debugging.assert_near(score, prior_score + likelihood_score)


def test_simple_nonlinear_initial_batch_is_finite_and_residual_free() -> None:
    fixture = make_simple_nonlinear_generic_target_fixture()

    value, score = fixture.adapter.log_prob_and_grad(MODEL_B_INITIAL_BATCH)
    diagnostics = simple_nonlinear_filter_diagnostics(MODEL_B_INITIAL_BATCH)

    assert value.shape == (1,)
    assert score.shape == (1, 3)
    assert np.isfinite(value.numpy()).all()
    assert np.isfinite(score.numpy()).all()
    np.testing.assert_allclose(
        diagnostics["deterministic_residual"].numpy(),
        np.zeros([1]),
        atol=1.0e-12,
    )


@pytest.mark.parametrize("filter_id", simple_nonlinear_admitted_filter_ids())
def test_simple_nonlinear_admitted_filter_initial_batch_is_residual_free(
    filter_id: str,
) -> None:
    fixture = make_simple_nonlinear_generic_target_fixture(filter_id=filter_id)
    backend = fixture.contract.filter_program.filter_manifest["sigma_point_backend"]

    value, score = fixture.adapter.log_prob_and_grad(MODEL_B_INITIAL_BATCH)
    diagnostics = simple_nonlinear_filter_diagnostics(
        MODEL_B_INITIAL_BATCH,
        backend=backend,
    )

    assert value.shape == (1,)
    assert score.shape == (1, 3)
    assert np.isfinite(value.numpy()).all()
    assert np.isfinite(score.numpy()).all()
    np.testing.assert_allclose(
        diagnostics["deterministic_residual"].numpy(),
        np.zeros([1]),
        atol=1.0e-12,
    )


def test_simple_nonlinear_adapter_rejects_rank1_position() -> None:
    fixture = make_simple_nonlinear_generic_target_fixture()

    with pytest.raises(ValueError, match="rank 2 theta"):
        fixture.adapter.log_prob_and_grad(
            tf.constant([0.70, 0.25, 0.80], dtype=tf.float64)
        )


@pytest.mark.parametrize(
    "filter_id",
    (SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID, SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID),
)
def test_simple_nonlinear_deferred_filter_routes_are_rejected(filter_id: str) -> None:
    with pytest.raises(ValueError, match="not admitted"):
        make_simple_nonlinear_generic_target_contract(filter_id=filter_id)
    with pytest.raises(ValueError, match="not admitted"):
        make_simple_nonlinear_generic_target_fixture(filter_id=filter_id)


def test_simple_nonlinear_unknown_filter_route_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown simple nonlinear filter route"):
        make_simple_nonlinear_generic_target_contract(filter_id="not-a-filter")


def test_simple_nonlinear_adapter_signature_and_manifest_are_stable() -> None:
    fixture = make_simple_nonlinear_generic_target_fixture()
    same = make_simple_nonlinear_generic_target_fixture()
    changed_contract = make_simple_nonlinear_generic_target_contract(
        prior_hash="sha256:changed-prior",
    )
    changed_fixture = type(fixture.adapter)(
        contract=changed_contract,
        prior_log_prob_and_grad=simple_nonlinear_gaussian_prior_log_prob_and_grad,
        filter_log_likelihood_and_grad=simple_nonlinear_svd_ukf_log_likelihood_and_grad,
        dtype=tf.float64,
        target_scope="model-b-nonlinear-accumulation-generic-target-fixture",
        evidence_path="bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py",
        nonclaims=SIMPLE_NONLINEAR_GENERIC_TARGET_NONCLAIMS,
    )

    assert fixture.adapter.manifest_payload()["target_signature"] == stable_ssm_target_signature(
        fixture.contract
    )
    assert stable_ssm_posterior_adapter_signature(
        fixture.adapter
    ) == stable_ssm_posterior_adapter_signature(same.adapter)
    assert stable_ssm_posterior_adapter_signature(
        fixture.adapter
    ) != stable_ssm_posterior_adapter_signature(changed_fixture)


def test_simple_nonlinear_default_phase7_signatures_are_preserved() -> None:
    fixture = make_simple_nonlinear_generic_target_fixture()

    assert stable_ssm_target_signature(fixture.contract) == (
        "c6a942c251e08f111b5647f814c1815535f931fcd13a09d337a74b8fb5eacaa0"
    )
    assert stable_ssm_posterior_adapter_signature(fixture.adapter) == (
        "9fdc2ef475992711dd1ed5aadc0b47aeed235d7ccea9e9567740b57aaf2a04dd"
    )


def test_simple_nonlinear_admitted_filter_target_signatures_are_distinct() -> None:
    ukf = make_simple_nonlinear_generic_target_fixture(
        filter_id=SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID,
    )
    cubature = make_simple_nonlinear_generic_target_fixture(
        filter_id=SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID,
    )

    assert stable_ssm_target_signature(ukf.contract) != stable_ssm_target_signature(
        cubature.contract
    )
    assert stable_ssm_posterior_adapter_signature(
        ukf.adapter
    ) != stable_ssm_posterior_adapter_signature(cubature.adapter)


def test_simple_nonlinear_score_matches_finite_difference_reference() -> None:
    fixture = make_simple_nonlinear_generic_target_fixture()
    theta = MODEL_B_INITIAL_BATCH
    _value, score = fixture.adapter.log_prob_and_grad(theta)
    eps = tf.constant(1.0e-5, dtype=tf.float64)
    columns = []
    for dim in range(3):
        basis = tf.one_hot(dim, 3, dtype=tf.float64)[tf.newaxis, :]
        plus = fixture.adapter.log_prob(theta + eps * basis)[0]
        minus = fixture.adapter.log_prob(theta - eps * basis)[0]
        columns.append((plus - minus) / (2.0 * eps))
    finite_difference = tf.stack(columns, axis=0)[tf.newaxis, :]

    tf.debugging.assert_near(score, finite_difference, rtol=1.0e-4, atol=1.0e-4)


@pytest.mark.parametrize("filter_id", simple_nonlinear_admitted_filter_ids())
def test_simple_nonlinear_admitted_filter_score_matches_finite_difference_reference(
    filter_id: str,
) -> None:
    fixture = make_simple_nonlinear_generic_target_fixture(filter_id=filter_id)
    theta = MODEL_B_INITIAL_BATCH
    _value, score = fixture.adapter.log_prob_and_grad(theta)
    eps = tf.constant(1.0e-5, dtype=tf.float64)
    columns = []
    for dim in range(3):
        basis = tf.one_hot(dim, 3, dtype=tf.float64)[tf.newaxis, :]
        plus = fixture.adapter.log_prob(theta + eps * basis)[0]
        minus = fixture.adapter.log_prob(theta - eps * basis)[0]
        columns.append((plus - minus) / (2.0 * eps))
    finite_difference = tf.stack(columns, axis=0)[tf.newaxis, :]

    tf.debugging.assert_near(score, finite_difference, rtol=1.0e-4, atol=1.0e-4)


def test_simple_nonlinear_sigma_point_backend_rejects_unadmitted_backend() -> None:
    with pytest.raises(ValueError, match="not admitted"):
        simple_nonlinear_sigma_point_log_likelihood_and_grad(
            MODEL_B_INITIAL_BATCH,
            backend="tf_svd_cut4",
        )


def test_simple_nonlinear_generic_target_cpu_only_hides_gpu() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert tf.config.list_physical_devices("GPU") == []
