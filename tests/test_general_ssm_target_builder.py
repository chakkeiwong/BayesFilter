from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference import evaluate_batch_native_value_score, stable_adapter_signature
from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    FrozenTransportBinding,
    InvalidSSMContract,
    InvalidSSMTargetBuilderContract,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    build_ssm_posterior_adapter,
    stable_ssm_posterior_adapter_signature,
    stable_ssm_target_signature,
)


def _static_shape(**overrides):
    values = {
        "horizon": 4,
        "state_dim": 2,
        "observation_dim": 1,
        "innovation_dim": 1,
        "parameter_dim": 3,
    }
    values.update(overrides)
    return SSMStaticShape(**values)


def _problem(**overrides):
    values = {
        "problem_id": "toy-nonlinear-ssm",
        "static_shape": _static_shape(),
        "data_signature": SSMDataSignature(
            dataset_id="toy-nonlinear-data",
            observation_shape=(4, 1),
            data_hash="sha256:data-v1",
        ),
        "target_coordinate_convention": "unconstrained",
        "model_manifest": {
            "model_id": "toy-nonlinear-model",
            "model_hash": "sha256:model-v1",
            "capabilities": ("transition_mean", "observation_mean"),
        },
    }
    values.update(overrides)
    return BayesianSSMProblem(**values)


def _chart(**overrides):
    values = {
        "parameter_names": ("rho", "sigma", "beta"),
        "unconstrained_dim": 3,
        "constrained_shape": (3,),
        "transform_manifest": {
            "transform_id": "identity-chart",
            "transform_hash": "sha256:chart-v1",
        },
        "log_jacobian_convention": "not_included",
    }
    values.update(overrides)
    return ParameterChart(**values)


def _prior(**overrides):
    values = {
        "prior_manifest": {
            "prior_id": "toy-gaussian-prior",
            "prior_hash": "sha256:prior-v1",
        },
        "support_policy": "unbounded",
        "log_density_authority": "graph_native",
    }
    values.update(overrides)
    return ParameterPrior(**values)


def _filter_program(**overrides):
    values = {
        "filter_id": "toy-deterministic-filter",
        "required_model_capabilities": ("transition_mean", "observation_mean"),
        "deterministic_target_policy": "deterministic",
        "approximation_semantics": "deterministic_approximation",
        "filter_manifest": {
            "filter_id": "toy-deterministic-filter",
            "filter_hash": "sha256:filter-v1",
            "backend": "tensorflow",
        },
    }
    values.update(overrides)
    return FilterProgram(**values)


def _contract(**overrides):
    values = {
        "problem": _problem(),
        "chart": _chart(),
        "prior": _prior(),
        "filter_program": _filter_program(),
        "frozen_transport": None,
    }
    values.update(overrides)
    return SSMTargetContract(**values)


def _prior_log_prob_and_grad(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    value = -0.5 * tf.reduce_sum(tf.square(theta), axis=-1)
    score = -theta
    return value, score


def _toy_filter_log_likelihood_and_grad(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    dtype = theta.dtype
    rho = theta[:, 0]
    sigma = theta[:, 1]
    beta = theta[:, 2]
    t = tf.constant([0.0, 0.5, 1.0, 1.5], dtype=dtype)
    curvature = tf.constant([0.0, 0.25, 0.5, 0.75], dtype=dtype)
    y = tf.constant([0.1, 0.4, 0.8, 1.3], dtype=dtype)
    model = rho[:, tf.newaxis] + beta[:, tf.newaxis] * t + (
        tf.square(sigma)[:, tf.newaxis] * curvature
    )
    residual = y - model
    value = -0.5 * tf.reduce_sum(tf.square(residual), axis=-1)
    score_rho = tf.reduce_sum(residual, axis=-1)
    score_sigma = tf.reduce_sum(residual * (2.0 * sigma[:, tf.newaxis] * curvature), axis=-1)
    score_beta = tf.reduce_sum(residual * t, axis=-1)
    score = tf.stack([score_rho, score_sigma, score_beta], axis=-1)
    return value, score


def _adapter(contract: SSMTargetContract | None = None):
    return build_ssm_posterior_adapter(
        contract=_contract() if contract is None else contract,
        prior_log_prob_and_grad=_prior_log_prob_and_grad,
        filter_log_likelihood_and_grad=_toy_filter_log_likelihood_and_grad,
        dtype=tf.float64,
        evidence_path="tests/test_general_ssm_target_builder.py",
    )


def test_ssm_import_does_not_eagerly_import_tensorflow_from_package_init() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    code = (
        "import json, sys; "
        "import bayesfilter.ssm; "
        "print(json.dumps({'tensorflow': 'tensorflow' in sys.modules}, sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == '{"tensorflow": false}'


def test_target_builder_emits_batch_native_values_scores_and_metadata() -> None:
    adapter = _adapter()
    theta = tf.constant(
        [[0.2, 0.1, 0.3], [0.4, -0.2, 0.1]],
        dtype=tf.float64,
    )

    value, score = adapter.log_prob_and_grad(theta)
    result = evaluate_batch_native_value_score(adapter, theta, dtype=tf.float64)

    assert value.shape == (2,)
    assert score.shape == (2, 3)
    assert result.metadata.rank == "batch"
    assert result.metadata.parameter_dim == 3
    assert tf.reduce_all(tf.math.is_finite(value))
    assert tf.reduce_all(tf.math.is_finite(score))
    assert adapter.metadata().batch_rank_policy == "rank2_required"
    assert adapter.manifest_payload()["non_batch_static_shape"] == {
        "horizon": 4,
        "state_dim": 2,
        "observation_dim": 1,
        "innovation_dim": 1,
        "parameter_dim": 3,
    }
    assert adapter.manifest_payload()["target_signature"] == stable_ssm_target_signature(
        adapter.contract
    )
    assert adapter.value_score_capability().value_score_authority == "graph_native"
    assert adapter.value_score_capability().xla_hmc_ready is False


def test_target_builder_xla_hmc_ready_is_explicit_evidence_backed_opt_in() -> None:
    with pytest.raises(InvalidSSMTargetBuilderContract, match="evidence_path"):
        build_ssm_posterior_adapter(
            contract=_contract(),
            prior_log_prob_and_grad=_prior_log_prob_and_grad,
            filter_log_likelihood_and_grad=_toy_filter_log_likelihood_and_grad,
            dtype=tf.float64,
            xla_hmc_ready=True,
        )

    adapter = build_ssm_posterior_adapter(
        contract=_contract(),
        prior_log_prob_and_grad=_prior_log_prob_and_grad,
        filter_log_likelihood_and_grad=_toy_filter_log_likelihood_and_grad,
        dtype=tf.float64,
        evidence_path="tests/test_general_ssm_target_builder.py",
        xla_hmc_ready=True,
    )
    capability = adapter.value_score_capability()

    assert capability.value_score_authority == "graph_native"
    assert capability.xla_hmc_ready is True
    assert capability.is_accepted_xla_hmc_authority is True
    assert capability.full_chain_xla_diagnostic_ready is False
    assert adapter.manifest_payload()["xla_hmc_ready"] is True


def test_target_builder_full_chain_xla_requires_target_xla_authority() -> None:
    with pytest.raises(InvalidSSMTargetBuilderContract, match="requires xla_hmc_ready"):
        build_ssm_posterior_adapter(
            contract=_contract(),
            prior_log_prob_and_grad=_prior_log_prob_and_grad,
            filter_log_likelihood_and_grad=_toy_filter_log_likelihood_and_grad,
            dtype=tf.float64,
            evidence_path="tests/test_general_ssm_target_builder.py",
            full_chain_xla_diagnostic_ready=True,
        )


def test_target_builder_matches_direct_prior_plus_filter_likelihood() -> None:
    adapter = _adapter()
    theta = tf.constant(
        [[0.2, 0.1, 0.3], [0.4, -0.2, 0.1]],
        dtype=tf.float64,
    )

    value, score = adapter.log_prob_and_grad(theta)
    prior_value, prior_score = _prior_log_prob_and_grad(theta)
    filter_value, filter_score = _toy_filter_log_likelihood_and_grad(theta)

    tf.debugging.assert_near(value, prior_value + filter_value)
    tf.debugging.assert_near(score, prior_score + filter_score)


def test_target_builder_score_matches_finite_difference_reference() -> None:
    adapter = _adapter()
    theta = tf.constant([[0.2, 0.1, 0.3]], dtype=tf.float64)
    _value, score = adapter.log_prob_and_grad(theta)
    eps = tf.constant(1.0e-5, dtype=tf.float64)
    columns = []
    for dim in range(3):
        basis = tf.one_hot(dim, 3, dtype=tf.float64)[tf.newaxis, :]
        plus = adapter.log_prob(theta + eps * basis)[0]
        minus = adapter.log_prob(theta - eps * basis)[0]
        columns.append((plus - minus) / (2.0 * eps))
    finite_difference = tf.stack(columns, axis=0)[tf.newaxis, :]

    tf.debugging.assert_near(score, finite_difference, rtol=1.0e-5, atol=1.0e-5)


def test_target_builder_rejects_rank1_scalar_position_but_allows_batch_of_one() -> None:
    adapter = _adapter()

    with pytest.raises(ValueError, match=r"rank 2 theta"):
        adapter.log_prob_and_grad(tf.constant([0.2, 0.1, 0.3], dtype=tf.float64))

    value, score = adapter.log_prob_and_grad(
        tf.constant([[0.2, 0.1, 0.3]], dtype=tf.float64)
    )
    assert value.shape == (1,)
    assert score.shape == (1, 3)


def test_target_builder_signature_is_stable_and_changes_with_contract() -> None:
    adapter = _adapter()
    same = _adapter()
    changed = _adapter(
        _contract(
            prior=_prior(
                prior_manifest={
                    "prior_id": "toy-gaussian-prior",
                    "prior_hash": "sha256:prior-v2",
                }
            )
        )
    )

    assert stable_ssm_posterior_adapter_signature(adapter) == stable_ssm_posterior_adapter_signature(
        same
    )
    assert stable_adapter_signature(adapter) == stable_ssm_posterior_adapter_signature(adapter)
    assert stable_ssm_posterior_adapter_signature(adapter) != stable_ssm_posterior_adapter_signature(
        changed
    )


def test_target_builder_tf_function_cpu_only_compile_check() -> None:
    adapter = _adapter()

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float64)])
    def compiled(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return adapter.log_prob_and_grad(values)

    value, score = compiled(tf.constant([[0.2, 0.1, 0.3]], dtype=tf.float64))

    assert value.shape == (1,)
    assert score.shape == (1, 3)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"


def test_target_builder_rejects_nonready_filter_and_frozen_transport_contract() -> None:
    with pytest.raises(InvalidSSMContract, match="filter program must be deterministic"):
        _adapter(
            _contract(
                filter_program=_filter_program(
                    deterministic_target_policy="stochastic_not_hmc_ready",
                    approximation_semantics="fixed_randomness_approximation",
                )
            )
        )

    base = _contract()
    target_signature = stable_ssm_target_signature(base)
    transport = FrozenTransportBinding(
        transport_id="toy-frozen-neutra",
        dimension=3,
        target_signature=target_signature,
        log_jacobian_available=True,
        transport_manifest={
            "transport_id": "toy-frozen-neutra",
            "transport_hash": "sha256:transport-v1",
            "target_signature": target_signature,
        },
    )
    transported = SSMTargetContract(
        problem=base.problem,
        chart=base.chart,
        prior=base.prior,
        filter_program=base.filter_program,
        frozen_transport=transport,
    )
    with pytest.raises(InvalidSSMTargetBuilderContract, match="untransported"):
        _adapter(transported)


def test_target_builder_rejects_bad_value_score_shapes_and_unknown_policy() -> None:
    with pytest.raises(InvalidSSMTargetBuilderContract, match="unknown batch_rank_policy"):
        build_ssm_posterior_adapter(
            contract=_contract(),
            prior_log_prob_and_grad=_prior_log_prob_and_grad,
            filter_log_likelihood_and_grad=_toy_filter_log_likelihood_and_grad,
            batch_rank_policy="scalar_allowed",  # type: ignore[arg-type]
        )

    def bad_prior(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.reduce_sum(theta, axis=-1), tf.reduce_sum(theta, axis=0)

    adapter = build_ssm_posterior_adapter(
        contract=_contract(),
        prior_log_prob_and_grad=bad_prior,
        filter_log_likelihood_and_grad=_toy_filter_log_likelihood_and_grad,
        dtype=tf.float64,
    )
    with pytest.raises(ValueError, match=r"prior score must have rank 2"):
        adapter.log_prob_and_grad(tf.constant([[0.2, 0.1, 0.3]], dtype=tf.float64))


def test_target_builder_log_prob_and_grad_has_no_python_row_loop() -> None:
    import inspect
    import textwrap

    from bayesfilter.ssm.target_builder import GenericSSMPosteriorAdapter

    source = textwrap.dedent(inspect.getsource(GenericSSMPosteriorAdapter.log_prob_and_grad))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        assert not (
            isinstance(node, ast.Attribute)
            and node.attr in {"map_fn", "vectorized_map"}
        )
