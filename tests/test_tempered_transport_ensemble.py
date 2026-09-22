"""Focused mechanics fixtures for the tempered transport ensemble.

These tests use an analytic TensorFlow target.  They establish the algebraic
contracts without making a q=20 or posterior-convergence claim.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.inference.tempered_target_tf import (
    GaussianLikelihoodBridge,
    TemperedBridgeError,
    build_q20_properness_receipt,
)
from bayesfilter.inference.tempered_transport_ensemble_tf import (
    AffineDiagonalTransport,
    IndependentTemperedReverseKLTrainer,
    JointTemperedMixtureReverseKLTrainer,
    TemperedEnsembleError,
    TransportBank,
    capture_trainable_transport_checkpoint,
    chunked_pullback_gaussianization_diagnostic,
    mixture_reverse_kl_terms,
    paired_reverse_kl_improvement,
    prepare_transport_initialization,
    preflight_transport_initialization,
    pullback_gaussianization_diagnostic,
    restore_trainable_transport_checkpoint,
)


class _AnalyticComponentTarget:
    """A proper two-dimensional Gaussian likelihood fixture."""

    def target_signature(self) -> str:
        return "analytic-tempered-target-v1"

    @property
    def target_scope(self) -> str:
        return "analytic-tempered-target"

    def adapter_signature(self) -> str:
        return self.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="analytic_tensorflow_fixture",
            target_scope=self.target_scope,
            nonclaims=("analytic mechanics fixture only",),
        )

    def batch_prior_likelihood_value_score_status(self, theta):
        theta = tf.convert_to_tensor(theta, tf.float64)
        likelihood_center = tf.constant([2.0, -1.0], tf.float64)
        prior = -0.5 * tf.reduce_sum(tf.square(theta), axis=1)
        likelihood_delta = theta - likelihood_center
        likelihood = -0.5 * tf.reduce_sum(tf.square(likelihood_delta), axis=1)
        prior_score = -theta
        likelihood_score = -likelihood_delta
        valid = tf.reduce_all(tf.math.is_finite(theta), axis=1)
        status = {
            "status_code": tf.where(valid, 0, 1),
            "valid_pre_regularized_score": valid,
        }
        return likelihood, likelihood_score, prior, prior_score, status


def _facts() -> dict[str, object]:
    return {
        "horizon": 2,
        "observation_dim": 1,
        "augmented_state_dim": 3,
        "parameter_dim": 2,
        "prior_variance": 1.0,
        "observation_variance": 0.5,
        "sigma_rule": "unscented",
        "sigma_alpha": 1.0,
        "sigma_beta": 2.0,
        "sigma_kappa": 0.0,
        "covariance_weights": [2.0, *(1.0 / 6.0 for _ in range(6))],
        "covariance_weights_nonnegative": True,
        "covariance_weight_sum": 3.0,
        "gaussian_innovation_factorization": True,
        "likelihood_strictly_positive": True,
    }


def _bridge(*, jit_compile: bool = False) -> GaussianLikelihoodBridge:
    return GaussianLikelihoodBridge(
        _AnalyticComponentTarget(),
        prior_center=tf.zeros([2], tf.float64),
        prior_variance=1.0,
        source_facts=_facts(),
        jit_compile=jit_compile,
    )


def _prepared_iaf(
    config: WeightedNeuTraConfig,
    bridge: GaussianLikelihoodBridge,
    *,
    component_id: str,
    beta: float,
    seed: tuple[int, int],
):
    return prepare_transport_initialization(
        WeightedDenseIAFTransport(config),
        bridge,
        component_id=component_id,
        seed=seed,
        batch_size=4,
        repair_scales=(1.0,),
        beta=beta,
    )


def _checkpoint_scope() -> dict[str, object]:
    return {
        "data_identity": "analytic-tempered-target-v1",
        "dtype": "float64",
        "backend": "tensorflow_fixture",
        "jit_compile": False,
        "training_seed_derivation": {"root": [29, 3], "fold_order": []},
        "validation_bank_ids": ["analytic-checkpoint-bank-v1"],
    }


def test_properness_receipt_rejects_missing_positive_variance() -> None:
    facts = _facts()
    facts["observation_variance"] = 0.0
    with pytest.raises(TemperedBridgeError, match="strictly positive"):
        build_q20_properness_receipt(
            facts, target_signature="fixture", bridge_id="fixture-bridge"
        )


def test_bridge_endpoint_and_interior_score_decomposition() -> None:
    bridge = _bridge()
    theta = tf.constant([[0.0, 0.0], [1.0, -2.0]], tf.float64)
    likelihood, likelihood_score, prior, prior_score, _ = bridge.component_terms(theta)
    for beta in (0.0, 0.37, 1.0):
        value, score, status = bridge.value_score_status(theta, beta)
        np.testing.assert_allclose(
            value.numpy(), (prior + beta * likelihood).numpy(), rtol=0.0, atol=1e-13
        )
        np.testing.assert_allclose(
            score.numpy(), (prior_score + beta * likelihood_score).numpy(),
            rtol=0.0,
            atol=1e-13,
        )
        np.testing.assert_array_equal(status["status_code"].numpy(), [0, 0])
    invalid = bridge.value_score_status(theta, 1.1)[2]
    np.testing.assert_array_equal(invalid["status_code"].numpy(), [1, 1])


def test_affine_bank_is_not_an_averaged_map_and_is_permutation_invariant() -> None:
    first = AffineDiagonalTransport([0.0, 0.0], [1.0, 1.0], component_id="a")
    second = AffineDiagonalTransport([4.0, 0.0], [1.0, 2.0], component_id="b")
    bank = TransportBank([first, second], component_ids=("a", "b"))
    swapped = TransportBank([second, first], component_ids=("b", "a"))
    points = tf.constant([[0.0, 0.0], [4.0, 1.0]], tf.float64)
    np.testing.assert_allclose(
        bank.mixture_log_prob(points).numpy(),
        swapped.mixture_log_prob(points).numpy(),
        rtol=0.0,
        atol=1e-13,
    )
    latent = tf.constant([[1.0, 1.0]], tf.float64)
    physical, _ = bank.forward_bank(tf.stack([latent, latent], axis=0))
    averaged_map = 0.5 * (physical[0] + physical[1])
    assert not np.allclose(averaged_map.numpy(), physical[0].numpy())
    cross = bank.cross_component_log_prob(physical)
    assert tuple(cross.shape) == (2, 2, 1)


def test_joint_mixture_loss_has_quadratic_transport_work_and_one_target_batch() -> None:
    bridge = _bridge()
    bank = TransportBank(
        [
            AffineDiagonalTransport([0.0, 0.0], [1.0, 1.0], component_id="a"),
            AffineDiagonalTransport([2.0, 0.0], [1.0, 1.0], component_id="b"),
        ],
        component_ids=("a", "b"),
    )
    latent_bank = tf.constant(
        [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]], tf.float64
    )
    physical, logdet = bank.forward_bank(latent_bank)
    flattened = tf.reshape(physical, [4, 2])
    target, _, status = bridge.value_score_status(flattened, 0.5)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
    loss, per_sample, mixture, cross = mixture_reverse_kl_terms(
        bank, physical, tf.reshape(target, [2, 2]), logdet
    )
    assert tuple(per_sample.shape) == (2, 2)
    assert tuple(mixture.shape) == (2, 2)
    assert tuple(cross.shape) == (2, 2, 2)
    assert int(loss.shape.rank) == 0
    assert float(loss.numpy()) == pytest.approx(float(loss.numpy()))


def test_independent_reverse_kl_uses_fresh_batch_and_updates_once() -> None:
    bridge = _bridge()
    config = WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(4,),
        stages=1,
        initialization_scale=0.01,
        learning_rate=1.0e-3,
        jit_compile=False,
    )
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id="chart-a",
        batch_size=4,
        prepared_initialization=_prepared_iaf(
            config,
            bridge,
            component_id="chart-a",
            beta=0.5,
            seed=(5, 7),
        ),
    )
    first = trainer.train_step((11, 17))
    second = trainer.train_step((11, 18))
    assert bool(first.valid.numpy())
    assert bool(second.valid.numpy())
    assert int(second.step.numpy()) == 2
    assert int(first.target_call_count.numpy()) == 1
    assert int(first.cross_density_work.numpy()) == 0


def test_pullback_gaussianization_diagnostic_is_exact_for_gaussian_bridge() -> None:
    bridge = _bridge()
    beta = 0.5
    likelihood_center = tf.constant([2.0, -1.0], tf.float64)
    precision = 1.0 + beta
    exact = AffineDiagonalTransport(
        beta * likelihood_center / precision,
        tf.fill([2], tf.constant(precision ** -0.5, tf.float64)),
        component_id="exact-beta-chart",
    )
    latent = tf.constant(
        [[-1.0, 0.5], [0.0, 0.0], [0.75, -0.25], [1.5, 1.0]],
        tf.float64,
    )
    result = pullback_gaussianization_diagnostic(
        exact, bridge, beta=beta, latent=latent
    )
    assert bool(result.finite.numpy())
    assert int(result.valid_row_count.numpy()) == 4
    assert float(result.centered_log_density_rms.numpy()) < 1.0e-12
    assert float(result.pullback_score_maximum_row_norm.numpy()) < 1.0e-12


def test_chunked_pullback_diagnostic_matches_direct_rows() -> None:
    bridge = _bridge()
    beta = 0.5
    likelihood_center = tf.constant([2.0, -1.0], tf.float64)
    precision = 1.0 + beta
    exact = AffineDiagonalTransport(
        beta * likelihood_center / precision,
        tf.fill([2], tf.constant(precision ** -0.5, tf.float64)),
        component_id="exact-beta-chart-chunked",
    )
    latent = tf.random.stateless_normal(
        [32, 2], tf.constant([91, 7], tf.int32), dtype=tf.float64
    )
    direct = pullback_gaussianization_diagnostic(
        exact, bridge, beta=beta, latent=latent
    )
    chunked = chunked_pullback_gaussianization_diagnostic(
        exact, bridge, beta=beta, latent=latent, chunk_size=8
    )
    np.testing.assert_allclose(
        chunked.reverse_kl_per_sample.numpy(),
        direct.reverse_kl_per_sample.numpy(),
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        chunked.pullback_log_density_residual.numpy(),
        direct.pullback_log_density_residual.numpy(),
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        chunked.pullback_score_residual.numpy(),
        direct.pullback_score_residual.numpy(),
        rtol=0.0,
        atol=1.0e-13,
    )
    assert int(chunked.batch_size.numpy()) == 32
    assert int(chunked.valid_row_count.numpy()) == 32
    assert bool(chunked.finite.numpy())


def test_chunked_pullback_diagnostic_rejects_singleton_and_nondivisor() -> None:
    bridge = _bridge()
    chart = AffineDiagonalTransport([0.0, 0.0], [1.0, 1.0])
    latent = tf.zeros([8, 2], tf.float64)
    with pytest.raises(TemperedEnsembleError, match="greater than one"):
        chunked_pullback_gaussianization_diagnostic(
            chart, bridge, beta=0.5, latent=latent, chunk_size=1
        )
    with pytest.raises(TemperedEnsembleError, match="divide"):
        chunked_pullback_gaussianization_diagnostic(
            chart, bridge, beta=0.5, latent=latent, chunk_size=3
        )


def test_paired_reverse_kl_improvement_detects_worse_chart() -> None:
    bridge = _bridge()
    beta = 0.5
    latent = tf.constant(
        [
            [-1.5, -0.5],
            [-1.0, 0.25],
            [-0.5, 1.0],
            [0.0, -1.0],
            [0.5, -0.25],
            [1.0, 0.5],
            [1.5, 1.25],
            [2.0, -1.5],
        ],
        tf.float64,
    )
    exact = AffineDiagonalTransport(
        tf.constant([2.0 / 3.0, -1.0 / 3.0], tf.float64),
        tf.fill([2], tf.constant((1.5) ** -0.5, tf.float64)),
    )
    worse = AffineDiagonalTransport([8.0, -7.0], [1.0, 1.0])
    exact_diagnostic = pullback_gaussianization_diagnostic(
        exact, bridge, beta=beta, latent=latent
    )
    worse_diagnostic = pullback_gaussianization_diagnostic(
        worse, bridge, beta=beta, latent=latent
    )
    degradation = paired_reverse_kl_improvement(exact_diagnostic, worse_diagnostic)
    assert bool(degradation["finite"].numpy())
    assert float(degradation["mean_final_minus_start"].numpy()) > 0.0
    assert bool(degradation["training_viable"].numpy()) is False


def test_joint_trainer_reports_k_squared_work() -> None:
    bridge = _bridge()
    config = WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(3,),
        stages=1,
        initialization_scale=0.01,
        learning_rate=1.0e-3,
        jit_compile=False,
    )
    first = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id="a",
        batch_size=3,
        prepared_initialization=_prepared_iaf(
            config, bridge, component_id="a", beta=0.5, seed=(21, 1)
        ),
    )
    second = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id="b",
        batch_size=3,
        prepared_initialization=_prepared_iaf(
            config, bridge, component_id="b", beta=0.5, seed=(21, 2)
        ),
    )
    bank = TransportBank(
        [first.transport, second.transport], component_ids=("a", "b")
    )
    joint_preflights = tuple(
        prepare_transport_initialization(
            transport,
            bridge,
            component_id=component_id,
            seed=(23, index),
            batch_size=4,
            repair_scales=(1.0,),
            beta=0.5,
        ).receipt
        for index, (component_id, transport) in enumerate(
            zip(bank.component_ids, bank.transports, strict=True)
        )
    )
    trainer = JointTemperedMixtureReverseKLTrainer(
        bank,
        bridge,
        beta=0.5,
        batch_size=3,
        preflight_receipts=joint_preflights,
        jit_compile=False,
    )
    result = trainer.train_step((31, 41))
    assert bool(result.valid.numpy())
    assert int(result.target_call_count.numpy()) == 1
    assert int(result.cross_density_work.numpy()) == 12


def test_temperature_checkpoint_restores_fresh_exact_transport() -> None:
    bridge = _bridge()
    config = WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(4,),
        stages=1,
        initialization_scale=0.01,
        initialization_seed=(29, 1),
        learning_rate=1.0e-3,
        jit_compile=False,
    )
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id="chart-checkpoint",
        batch_size=4,
        prepared_initialization=_prepared_iaf(
            config,
            bridge,
            component_id="chart-checkpoint",
            beta=0.5,
            seed=(29, 2),
        ),
    )
    trainer.train_step((29, 3))
    latent = tf.constant([[0.25, -0.5], [1.0, 0.75]], tf.float64)
    physical, logdet = trainer.transport.forward_and_logdet(latent)
    checkpoint = capture_trainable_transport_checkpoint(
        trainer.transport,
        component_id="chart-checkpoint",
        beta=0.5,
        bridge_signature=bridge.signature,
        target_signature=bridge.target_signature,
        parent_checkpoint_hash=None,
        update_count=1,
        checkpoint_scope=_checkpoint_scope(),
    )
    restored = restore_trainable_transport_checkpoint(
        checkpoint,
        expected_context={
            "component_id": "chart-checkpoint",
            "beta": 0.5,
            "bridge_signature": bridge.signature,
            "target_signature": bridge.target_signature,
            "checkpoint_scope": _checkpoint_scope(),
        },
    )
    replay_physical, replay_logdet = restored.forward_and_logdet(latent)
    replay_latent, replay_inverse_logdet = restored.inverse_and_forward_logdet(
        physical
    )
    assert restored is not trainer.transport
    np.testing.assert_allclose(replay_physical.numpy(), physical.numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay_logdet.numpy(), logdet.numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay_latent.numpy(), latent.numpy(), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        replay_inverse_logdet.numpy(), logdet.numpy(), rtol=1.0e-12, atol=1.0e-12
    )


def test_temperature_checkpoint_rejects_tamper_and_context_drift() -> None:
    bridge = _bridge()
    config = WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(3,),
        stages=1,
        initialization_seed=(31, 1),
        jit_compile=False,
    )
    prepared = _prepared_iaf(
        config,
        bridge,
        component_id="chart-checkpoint",
        beta=0.5,
        seed=(31, 2),
    )
    checkpoint = capture_trainable_transport_checkpoint(
        prepared.transport,
        component_id="chart-checkpoint",
        beta=0.5,
        bridge_signature=bridge.signature,
        target_signature=bridge.target_signature,
        parent_checkpoint_hash="parent-hash",
        update_count=0,
        checkpoint_scope=_checkpoint_scope(),
    )
    with pytest.raises(TemperedEnsembleError, match="context mismatch"):
        restore_trainable_transport_checkpoint(
            checkpoint, expected_context={"beta": 1.0}
        )
    tampered = dict(checkpoint)
    tampered["update_count"] = 7
    with pytest.raises(TemperedEnsembleError, match="hash mismatch"):
        restore_trainable_transport_checkpoint(tampered)


def test_temperature_checkpoint_requires_complete_execution_scope() -> None:
    bridge = _bridge()
    config = WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(3,),
        stages=1,
        initialization_seed=(37, 1),
        jit_compile=False,
    )
    prepared = _prepared_iaf(
        config,
        bridge,
        component_id="chart-scope",
        beta=0.5,
        seed=(37, 2),
    )
    incomplete = _checkpoint_scope()
    del incomplete["validation_bank_ids"]
    with pytest.raises(TemperedEnsembleError, match="validation_bank_ids"):
        capture_trainable_transport_checkpoint(
            prepared.transport,
            component_id="chart-scope",
            beta=0.5,
            bridge_signature=bridge.signature,
            target_signature=bridge.target_signature,
            parent_checkpoint_hash=None,
            update_count=0,
            checkpoint_scope=incomplete,
        )


def test_preflight_uses_fixed_bank_and_rejects_single_row_batches() -> None:
    bridge = _bridge()
    transport = AffineDiagonalTransport([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(TemperedEnsembleError, match="exceed one"):
        preflight_transport_initialization(
            transport,
            bridge,
            component_id="a",
            seed=(1, 2),
            batch_size=1,
        )
    result = preflight_transport_initialization(
        transport,
        bridge,
        component_id="a",
        seed=(1, 2),
        batch_size=4,
        beta=0.5,
    )
    assert result.valid is True
    assert result.finite_rows == 4


def test_runtime_sources_have_no_forbidden_row_mapping() -> None:
    from bayesfilter.inference import tempered_target_tf, tempered_transport_ensemble_tf

    source = inspect.getsource(tempered_target_tf) + inspect.getsource(
        tempered_transport_ensemble_tf
    )
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "pfor" not in source.lower()
