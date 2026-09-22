"""Analytic Phase 3--6 fixtures for tempered transport mechanics.

These tests establish implementation identities only.  Stochastic summaries
are diagnostic and make no convergence, discovery, or sampler-ranking claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
    build_fixed_transport_value_score_adapter,
)
from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
    WeightedNeuTraTrainingError,
)
from bayesfilter.inference.neutra_hmc import (
    SequentialExactTransitionConfig,
    run_sequential_exact_transition,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.inference.tempered_lineage_tf import (
    TemperedLineageConfig,
    TemperedLineageController,
)
from bayesfilter.inference.tempered_target_tf import GaussianLikelihoodBridge
from bayesfilter.inference.tempered_transitions_tf import (
    BoundWithinTemperatureKernel,
    FixedChartKernelMixture,
    FixedChartSelection,
    ProperBridgeReplicaExchange,
    ProperReplicaExchangeTransitionProgram,
    ProperReplicaExchangeConfig,
    TemperedTransitionError,
    apply_proper_adjacent_swaps,
    build_fixed_transport_hmc_kernel,
    proper_swap_log_ratio,
    screen_transport_reliability,
)
from bayesfilter.inference.tempered_transport_ensemble_tf import (
    AffineDiagonalTransport,
    IndependentTemperedReverseKLTrainer,
    ReferenceAffineTransport,
    TemperedEnsembleError,
    TransportBank,
    mixture_reverse_kl_terms,
    prepare_transport_initialization,
    preflight_transport_initialization,
)


class _AnalyticComponentTarget:
    parameter_dim = 2
    parameter_names = ("theta.0", "theta.1")
    target_scope = "analytic-tempered-target"

    def target_signature(self) -> str:
        return "analytic-tempered-target-v2"

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
        delta = theta - likelihood_center
        likelihood = -0.5 * tf.reduce_sum(tf.square(delta), axis=1)
        valid = tf.reduce_all(tf.math.is_finite(theta), axis=1)
        return (
            likelihood,
            -delta,
            prior,
            -theta,
            {
                "status_code": tf.where(valid, 0, 1),
                "valid_pre_regularized_score": valid,
            },
        )


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


def _bridge() -> GaussianLikelihoodBridge:
    return GaussianLikelihoodBridge(
        _AnalyticComponentTarget(),
        prior_center=tf.zeros([2], tf.float64),
        prior_variance=1.0,
        source_facts=_facts(),
        jit_compile=False,
    )


def _config(seed: tuple[int, int]) -> WeightedNeuTraConfig:
    return WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(4,),
        stages=1,
        initialization_scale=0.03,
        initialization_seed=seed,
        learning_rate=2.0e-3,
        jit_compile=False,
    )


def _trained_chart(
    bridge: GaussianLikelihoodBridge,
    *,
    component_id: str,
    initialization_seed: tuple[int, int],
    preflight_seed: tuple[int, int],
    update_seed: tuple[int, int],
):
    config = _config(initialization_seed)
    prepared = prepare_transport_initialization(
        WeightedDenseIAFTransport(config),
        bridge,
        component_id=component_id,
        seed=preflight_seed,
        batch_size=6,
        repair_scales=(1.0,),
        beta=0.5,
    )
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id=component_id,
        batch_size=6,
        prepared_initialization=prepared,
    )
    trainer.train_step(update_seed)
    trainer.transport.bind_frozen_identity(
        {
            "checkpoint_sha256": component_id + "-checkpoint",
            "training_state_hash": component_id + "-training-state",
            "transport_tensor_hash": component_id + "-tensor-state",
        }
    )
    return trainer.transport


class _FiniteBoxBridge:
    parameter_dim = 1
    signature = "finite-box-bridge-v1"

    def value_score_status(self, theta, beta):
        del beta
        values = tf.convert_to_tensor(theta, tf.float64)
        valid = tf.reduce_all(tf.abs(values) < 2.0, axis=1)
        value = -0.5 * tf.reduce_sum(tf.square(values), axis=1)
        return value, -values, {
            "status_code": tf.where(valid, 0, 1),
            "valid_pre_regularized_score": valid,
            "bridge_valid": valid,
        }


def test_scale_repair_is_persisted_in_the_admitted_map() -> None:
    bridge = _FiniteBoxBridge()
    raw = AffineDiagonalTransport([100.0], [1.0], component_id="raw")
    prepared = prepare_transport_initialization(
        raw,
        bridge,
        component_id="raw",
        seed=(7, 9),
        batch_size=8,
        repair_scales=(1.0, 0.01),
        beta=0.0,
    )
    assert prepared.receipt.valid is True
    assert prepared.receipt.repair_index == 1
    assert prepared.receipt.actual_map_repaired is True
    assert isinstance(prepared.transport, ReferenceAffineTransport)
    physical = prepared.transport.forward_batch(tf.zeros([2, 1], tf.float64))
    np.testing.assert_allclose(physical.numpy(), [[1.0], [1.0]], atol=1e-14)
    with pytest.raises(TemperedEnsembleError, match="persistent map repair"):
        preflight_transport_initialization(
            raw,
            bridge,
            component_id="raw",
            seed=(7, 9),
            batch_size=8,
            repair_scales=(1.0, 0.01),
            beta=0.0,
        )


def test_lineage_records_true_restarts_unique_seeds_and_preoptimizer_receipts() -> None:
    bridge = _bridge()
    config = TemperedLineageConfig(
        betas=(0.0, 0.5, 1.0),
        component_ids=("a", "b"),
        root_seed=(101, 303),
        discovery_arm="positive_temperature_branching",
        positive_branch_betas=(0.5,),
        restart_component_indices=(1,),
        preflight_batch_size=6,
        repair_scales=(1.0,),
    )
    controller = TemperedLineageController(config, bridge)
    raw = (
        WeightedDenseIAFTransport(_config((1, 11))),
        WeightedDenseIAFTransport(_config((1, 12))),
    )
    receipts = controller.preflight_components(raw, beta_index=0)
    assert all(row.valid and row.optimizer_state_absent for row in receipts)
    assert all(
        isinstance(transport, ReferenceAffineTransport)
        for transport in controller.admitted_transports(0)
    )
    checkpoint = controller.checkpoint(1)
    assert checkpoint.parent_indices == (0, -1)
    assert controller.checkpoint(1) == checkpoint
    assert controller.seed_ledger()["all_seeds_unique"] is True
    prepared = controller.prepared_initializations(0)[0]
    trainer = IndependentTemperedReverseKLTrainer(
        _config((1, 11)),
        bridge,
        beta=0.0,
        component_id="a",
        batch_size=4,
        prepared_initialization=prepared,
    )
    assert int(trainer.optimizer.iterations.numpy()) == 0


def test_bank_checkpoint_roundtrip_and_categorical_moment_diagnostic() -> None:
    first = AffineDiagonalTransport([0.0, 0.0], [1.0, 1.0], component_id="a")
    second = AffineDiagonalTransport([4.0, -2.0], [1.0, 2.0], component_id="b")
    bank = TransportBank(
        [first, second],
        component_ids=("a", "b"),
        alpha_logits=tf.Variable(
            [math.log(0.25), math.log(0.75)], dtype=tf.float64
        ),
    )
    sample_count = 20_000
    samples, indices, _latent = bank.sample(sample_count, (31, 41))
    empirical_mean = tf.reduce_mean(samples, axis=0)
    analytic_mean = tf.constant([3.0, -1.5], tf.float64)
    analytic_variance = tf.constant([4.0, 4.75], tf.float64)
    mean_tolerance = 6.0 * tf.sqrt(analytic_variance / sample_count)
    assert bool(tf.reduce_all(tf.abs(empirical_mean - analytic_mean) < mean_tolerance).numpy())
    empirical_weight = tf.reduce_mean(tf.cast(indices == 1, tf.float64))
    bernoulli_se = math.sqrt(0.75 * 0.25 / sample_count)
    assert abs(float(empirical_weight.numpy()) - 0.75) < 6.0 * bernoulli_se

    extreme = TransportBank(
        [first, second],
        component_ids=("a", "b"),
        alpha_logits=tf.Variable([-1000.0, 1000.0], dtype=tf.float64),
    )
    tail = extreme.mixture_log_prob(tf.constant([[1.0e3, -1.0e3]], tf.float64))
    assert bool(tf.reduce_all(tf.math.is_finite(tail)).numpy())

    config = _config((8, 8))
    trainable = WeightedDenseIAFTransport(config)
    checkpoint_bank = TransportBank(
        [trainable],
        component_ids=("trainable",),
        alpha_logits=tf.Variable([0.0], dtype=tf.float64),
    )
    checkpoint = checkpoint_bank.state_payload()
    trainable.trainable_variables[0].assign_add(
        tf.ones_like(trainable.trainable_variables[0])
    )
    checkpoint_bank.restore_state_payload(checkpoint)
    assert checkpoint_bank.state_payload()["state_hash"] == checkpoint["state_hash"]


def test_invalid_training_row_rejects_the_whole_update_without_mutation() -> None:
    bridge = _bridge()

    class ToggleBridge:
        parameter_dim = 2
        signature = bridge.signature
        invalid = False

        def value_score_status(self, theta, beta):
            value, score, status = bridge.value_score_status(theta, beta)
            if self.invalid:
                invalid = tf.zeros_like(status["bridge_valid"], tf.bool)
                status = dict(status)
                status["bridge_valid"] = invalid
                status["valid_pre_regularized_score"] = invalid
                status["status_code"] = tf.ones_like(status["status_code"])
            return value, score, status

    toggled = ToggleBridge()
    config = _config((3, 5))
    prepared = prepare_transport_initialization(
        WeightedDenseIAFTransport(config),
        toggled,
        component_id="toggle",
        seed=(5, 6),
        batch_size=4,
        repair_scales=(1.0,),
        beta=0.5,
    )
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        toggled,
        beta=0.5,
        component_id="toggle",
        batch_size=4,
        prepared_initialization=prepared,
    )
    before = tuple(variable.numpy().copy() for variable in trainer.variables)
    toggled.invalid = True
    with pytest.raises(WeightedNeuTraTrainingError, match="rejected"):
        trainer.train_step((19, 23))
    for expected, variable in zip(before, trainer.variables, strict=True):
        np.testing.assert_array_equal(variable.numpy(), expected)
    assert int(trainer.optimizer.iterations.numpy()) == 0
    assert int(trainer.step.numpy()) == 0


def test_nonlinear_chart_reliability_transformed_parity_and_physical_replay() -> None:
    bridge = _bridge()
    charts = (
        _trained_chart(
            bridge,
            component_id="a",
            initialization_seed=(11, 1),
            preflight_seed=(11, 2),
            update_seed=(11, 3),
        ),
        _trained_chart(
            bridge,
            component_id="b",
            initialization_seed=(17, 1),
            preflight_seed=(17, 2),
            update_seed=(17, 3),
        ),
    )
    self_latent = tf.constant(
        [
            [[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]],
            [[-0.5, -1.0], [0.25, 0.75], [1.25, 0.25]],
        ],
        tf.float64,
    )
    cross_physical = tf.stack(
        [chart.forward_batch(self_latent[index]) for index, chart in enumerate(charts)],
        axis=0,
    )
    fixed_beta = bridge.fixed_beta_adapter(0.5)
    reliability = screen_transport_reliability(
        charts,
        component_ids=("a", "b"),
        self_latent_bank=self_latent,
        cross_physical_bank=cross_physical,
        reference_points=tf.constant([[0.0, 0.0], [1.0, -1.0]], tf.float64),
        declared_points=tf.constant([[2.0, -1.0], [-2.0, 1.0]], tf.float64),
        physical_score_fn=lambda value: fixed_beta.log_prob_and_grad(value)[1],
        maximum_condition_number=1.0e6,
        tolerance=1.0e-8,
    )
    assert reliability.passed is True, reliability.failures
    assert reliability.physical_score_checked is True

    adapters = tuple(
        build_fixed_transport_value_score_adapter(
            base_adapter=fixed_beta,
            fixed_transport=chart,
            target_scope=f"analytic:beta=0.5:chart={component_id}",
            evidence_path=None,
            xla_hmc_ready=False,
            full_chain_xla_diagnostic_ready=False,
        )
        for component_id, chart in zip(("a", "b"), charts, strict=True)
    )
    latent = tf.constant([[0.2, -0.4], [0.0, 0.0]], tf.float64)
    value_z, score_z = adapters[0].log_prob_and_grad(latent)
    physical, logdet = charts[0].forward_and_logdet(latent)
    value_theta, score_theta = fixed_beta.log_prob_and_grad(physical)
    expected_score = charts[0].pullback_score_batch(latent, score_theta) + charts[
        0
    ].log_abs_det_jacobian_score_batch(latent)
    np.testing.assert_allclose(value_z.numpy(), (value_theta + logdet).numpy(), atol=1e-11)
    np.testing.assert_allclose(score_z.numpy(), expected_score.numpy(), atol=1e-10)

    kernels = tuple(
        build_fixed_transport_hmc_kernel(
            adapter,
            state_shape=(4, 2),
            step_size=0.03,
            num_leapfrog_steps=2,
            jit_compile=False,
        )
        for adapter in adapters
    )
    current_physical = tf.constant(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.5]], tf.float64
    )
    replay_a = kernels[0](current_physical, tf.constant([71, 73], tf.int32))
    replay_b = kernels[0](current_physical, tf.constant([71, 73], tf.int32))
    np.testing.assert_array_equal(replay_a.numpy(), replay_b.numpy())
    mixture = FixedChartKernelMixture(
        kernels, gamma=(0.3, 0.7), chart_ids=("a", "b")
    )
    result = mixture.transition(current_physical, (79, 83))
    assert bool(tf.reduce_all(tf.math.is_finite(result["state"])).numpy())
    assert result["state_independent_selection"] is True


def test_proper_replica_exchange_cache_coherence_and_beta_one_boundary() -> None:
    bridge = _bridge()
    exchange = ProperBridgeReplicaExchange(bridge, (0.0, 0.5, 1.0))
    state = tf.constant(
        [
            [[-1.0, 0.0], [0.1, 0.2]],
            [[1.0, -0.5], [2.0, 1.0]],
            [[3.0, -1.0], [-2.0, 1.0]],
        ],
        tf.float64,
    )
    evaluated = exchange.evaluate(state)
    ratio = proper_swap_log_ratio(evaluated["cross_values"], 0, 1)
    direct = (
        evaluated["cross_values"][0, 1]
        + evaluated["cross_values"][1, 0]
        - evaluated["cross_values"][0, 0]
        - evaluated["cross_values"][1, 1]
    )
    np.testing.assert_array_equal(ratio.numpy(), direct.numpy())
    identities = exchange.initial_identities(2)
    result = exchange.transition(state, identities, seed=(89, 97), parity=0)
    reevaluated = exchange.evaluate(result["state"])
    np.testing.assert_allclose(
        result["values_at_temperature"].numpy(),
        reevaluated["values_at_temperature"].numpy(),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result["scores_at_temperature"].numpy(),
        reevaluated["scores_at_temperature"].numpy(),
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        result["status_at_temperature"]["status_code"].numpy(),
        reevaluated["status_at_temperature"]["status_code"].numpy(),
    )
    cold = exchange.posterior_state(result)
    np.testing.assert_array_equal(cold["state"].numpy(), result["state"][-1].numpy())
    assert cold["beta"] == 1.0 and cold["posterior_stream_only"] is True
    stream = exchange.posterior_stream(
        tf.stack((state, result["state"]), axis=0),
        tf.stack((identities, result["identities_at_temperature"]), axis=0),
    )
    assert tuple(stream["samples"].shape) == (2, 2, 2)
    with pytest.raises(TemperedTransitionError, match="start at beta=0"):
        ProperReplicaExchangeConfig((0.1, 1.0), bridge.signature)


def test_swap_rejection_preserves_all_caches_and_pair_schedules_do_not_overlap() -> None:
    state = tf.reshape(tf.range(12, dtype=tf.float64), [4, 1, 3])
    prior = tf.reshape(tf.range(4, dtype=tf.float64), [4, 1])
    likelihood = prior + 10.0
    identities = tf.reshape(tf.range(4, dtype=tf.int32), [4, 1])
    cross = tf.zeros([4, 4, 1], tf.float64)
    cross = tf.tensor_scatter_nd_update(cross, [[0, 1, 0]], [-1.0e6])
    status = {"status_code": tf.reshape(tf.range(10, 14, dtype=tf.int32), [4, 1])}
    rejected = apply_proper_adjacent_swaps(
        state,
        prior,
        likelihood,
        identities,
        cross,
        seed=(101, 103),
        parity=0,
        status_at_temperature=status,
    )
    assert bool(rejected["swap_is_accepted_adjacent"][0, 0].numpy()) is False
    np.testing.assert_array_equal(rejected["state"][0].numpy(), state[0].numpy())
    np.testing.assert_array_equal(
        rejected["status_at_temperature"]["status_code"][0].numpy(),
        status["status_code"][0].numpy(),
    )
    even = rejected["swap_is_proposed_adjacent"][:, 0].numpy().tolist()
    odd = apply_proper_adjacent_swaps(
        state,
        prior,
        likelihood,
        identities,
        tf.zeros_like(cross),
        seed=(107, 109),
        parity=1,
    )["swap_is_proposed_adjacent"][:, 0].numpy().tolist()
    assert even == [True, False, True]
    assert odd == [False, True, False]


def test_shared_exact_transition_controller_archives_only_beta_one_and_continues() -> None:
    transition_signature = "analytic-exact-refresh-v1"
    archive_calls = []

    def transition_program(state, *, num_results, seed, stage):
        del stage
        samples = tf.random.stateless_normal(
            [num_results, 4, 2], seed=seed, dtype=tf.float64
        )
        identities = tf.tile(
            tf.range(4, dtype=tf.int32)[tf.newaxis, :], [num_results, 1]
        )
        return {
            "transition_signature": transition_signature,
            "posterior_stream_only": True,
            "posterior_temperature": 1.0,
            "posterior_samples": samples,
            "posterior_replica_identities": identities,
            "final_transition_state": {
                "cold": samples[-1],
                "chunk_count": int(state["chunk_count"]) + 1,
            },
            "health": {"passed": True, "hard_vetoes": ()},
        }

    def archive(**kwargs):
        assert "posterior_samples" in kwargs
        assert "final_transition_state" not in kwargs
        assert kwargs["posterior_temperature"] == 1.0
        archive_calls.append(
            (kwargs["stage"], kwargs["chunk_index"], kwargs["cumulative"])
        )
        return {"archived": True, "stage": kwargs["stage"]}

    config = SequentialExactTransitionConfig(
        transition_signature=transition_signature,
        warmup_seed=(401, 409),
        retained_seed=(419, 421),
        warmup_chunk_results=32,
        warmup_min_results=32,
        warmup_check_window_results=32,
        warmup_max_results=32,
        warmup_rhat_max=10.0,
        retained_chunk_results=8,
        retained_min_results=8,
        retained_max_results=8,
        retained_rhat_max=10.0,
    )
    initial = {"cold": tf.zeros([4, 2], tf.float64), "chunk_count": 0}
    result = run_sequential_exact_transition(
        transition_program=transition_program,
        initial_transition_state=initial,
        posterior_state_fn=lambda value: value["cold"],
        parameter_names=("theta.0", "theta.1"),
        config=config,
        retained_diagnostic_fn=lambda _samples: {
            "passed": True,
            "hard_vetoes": (),
        },
        archive_callback=archive,
    )
    assert result["passed"] is True
    assert result["warmup_excluded_from_posterior"] is True
    assert result["posterior_temperature"] == 1.0
    assert tuple(result["private_warmup_beta_one"].shape) == (32, 4, 2)
    assert tuple(result["private_retained_beta_one"].shape) == (8, 4, 2)
    assert result["private_final_transition_state"]["chunk_count"] == 2
    np.testing.assert_array_equal(
        result["private_final_transition_state"]["cold"].numpy(),
        result["private_retained_beta_one"][-1].numpy(),
    )
    assert archive_calls == [
        ("warmup", 0, False),
        ("retained", 0, False),
        ("warmup", None, True),
        ("retained", None, True),
    ]


def test_end_to_end_exact_replica_program_has_identity_travel_and_cold_moments() -> None:
    bridge = _bridge()
    betas = (0.0, 0.5, 1.0)
    exchange = ProperBridgeReplicaExchange(bridge, betas)

    def identity_kernel(state, _seed):
        return tf.identity(state)

    identity_program = ProperReplicaExchangeTransitionProgram(
        exchange,
        tuple(
            BoundWithinTemperatureKernel(
                beta=beta,
                bridge_signature=bridge.signature,
                kernel_signature=f"identity-beta-{beta:g}",
                kernel=identity_kernel,
                mechanics_role="analytic_identity_travel_fixture",
            )
            for beta in betas
        ),
        jit_compile=False,
    )
    identical = tf.zeros([3, 4, 2], tf.float64)
    identity_result = identity_program(
        identity_program.initial_state(identical),
        num_results=2,
        seed=(431, 433),
        stage="warmup",
    )
    np.testing.assert_array_equal(
        identity_result["posterior_replica_identities"].numpy(),
        [[2, 2, 2, 2], [0, 0, 0, 0]],
    )

    likelihood_center = tf.constant([2.0, -1.0], tf.float64)

    def exact_refresh(beta):
        mean = beta / (1.0 + beta) * likelihood_center
        scale = math.sqrt(1.0 / (1.0 + beta))

        @tf.function(
            input_signature=(
                tf.TensorSpec([4, 2], tf.float64),
                tf.TensorSpec([2], tf.int32),
            ),
            jit_compile=False,
            reduce_retracing=False,
        )
        def kernel(state, seed):
            return mean + tf.constant(scale, tf.float64) * tf.random.stateless_normal(
                tf.shape(state), seed=seed, dtype=tf.float64
            )

        return kernel

    refresh_program = ProperReplicaExchangeTransitionProgram(
        exchange,
        tuple(
            BoundWithinTemperatureKernel(
                beta=beta,
                bridge_signature=bridge.signature,
                kernel_signature=f"exact-gaussian-refresh-beta-{beta:g}",
                kernel=exact_refresh(beta),
                mechanics_role="analytic_exact_invariant_refresh_fixture",
            )
            for beta in betas
        ),
        jit_compile=False,
    )
    sequential = run_sequential_exact_transition(
        transition_program=refresh_program,
        initial_transition_state=refresh_program.initial_state(identical),
        posterior_state_fn=refresh_program.posterior_state,
        parameter_names=("theta.0", "theta.1"),
        config=SequentialExactTransitionConfig(
            transition_signature=refresh_program.transition_signature,
            warmup_seed=(439, 443),
            retained_seed=(449, 457),
            warmup_chunk_results=64,
            warmup_min_results=64,
            warmup_check_window_results=64,
            warmup_max_results=64,
            warmup_rhat_max=1.2,
            retained_chunk_results=512,
            retained_min_results=512,
            retained_max_results=512,
            retained_rhat_max=1.2,
        ),
    )
    assert sequential["passed"] is True
    retained = tf.reshape(sequential["private_retained_beta_one"], [-1, 2])
    sample_count = int(retained.shape[0])
    empirical_mean = tf.reduce_mean(retained, axis=0)
    centered = retained - empirical_mean
    empirical_variance = tf.reduce_sum(tf.square(centered), axis=0) / float(
        sample_count - 1
    )
    expected_mean = likelihood_center / 2.0
    expected_variance = tf.constant([0.5, 0.5], tf.float64)
    mean_half_width = 6.0 * tf.sqrt(expected_variance / float(sample_count))
    variance_half_width = 8.0 * expected_variance * math.sqrt(
        2.0 / float(sample_count - 1)
    )
    assert bool(
        tf.reduce_all(tf.abs(empirical_mean - expected_mean) < mean_half_width).numpy()
    )
    assert bool(
        tf.reduce_all(
            tf.abs(empirical_variance - expected_variance) < variance_half_width
        ).numpy()
    )


def test_fixed_gamma_invariance_and_state_dependent_counterexample_are_exact() -> None:
    pi = tf.constant([0.25, 0.75], tf.float64)
    kernel_a = tf.constant([[0.7, 0.3], [0.1, 0.9]], tf.float64)
    kernel_b = tf.eye(2, dtype=tf.float64)
    for gamma in ((0.5, 0.5), (0.2, 0.8)):
        combined = gamma[0] * kernel_a + gamma[1] * kernel_b
        np.testing.assert_allclose(
            tf.linalg.matvec(combined, pi, transpose_a=True).numpy(),
            pi.numpy(),
            atol=1e-15,
        )
    state_dependent = tf.stack((kernel_a[0], kernel_b[1]), axis=0)
    propagated = tf.linalg.matvec(state_dependent, pi, transpose_a=True)
    assert not np.allclose(propagated.numpy(), pi.numpy())
    with pytest.raises(TemperedTransitionError, match="state-dependent"):
        FixedChartSelection((lambda _state: 1.0,), ("bad",))


def test_unequal_component_error_biases_alpha_away_from_regional_mass() -> None:
    regional_mass = tf.constant([0.25, 0.75], tf.float64)
    approximation_error = tf.constant([0.0, math.log(2.0)], tf.float64)
    unnormalized = regional_mass * tf.exp(-approximation_error)
    alpha_star = unnormalized / tf.reduce_sum(unnormalized)
    np.testing.assert_allclose(alpha_star.numpy(), [0.4, 0.6], atol=1e-15)
    assert not np.allclose(alpha_star.numpy(), regional_mass.numpy())


def test_joint_loss_gradient_matches_central_difference() -> None:
    bridge = _bridge()
    bank = TransportBank(
        [
            WeightedDenseIAFTransport(_config((121, 1))),
            WeightedDenseIAFTransport(_config((121, 2))),
        ],
        component_ids=("a", "b"),
    )
    latent = tf.constant(
        [
            [[-0.5, 0.25], [0.75, -0.25]],
            [[0.2, -0.8], [1.0, 0.5]],
        ],
        tf.float64,
    )

    def objective():
        physical, logdet = bank.forward_bank(latent)
        target, _score, _status = bridge.value_score_status(
            tf.reshape(physical, [4, 2]), 0.5
        )
        return mixture_reverse_kl_terms(
            bank, physical, tf.reshape(target, [2, 2]), logdet
        )[0]

    variables = bank.trainable_variables
    with tf.GradientTape() as tape:
        loss = objective()
    gradients = tape.gradient(loss, variables)
    candidates = [
        (float(tf.reduce_max(tf.abs(gradient)).numpy()), index, gradient)
        for index, gradient in enumerate(gradients)
        if gradient is not None
    ]
    _, variable_index, gradient = max(candidates, key=lambda row: row[0])
    variable = variables[variable_index]
    flat_index = int(tf.argmax(tf.reshape(tf.abs(gradient), [-1])).numpy())
    original = variable.numpy().copy()
    step = 1.0e-5
    plus = original.copy().reshape(-1)
    plus[flat_index] += step
    variable.assign(plus.reshape(original.shape))
    loss_plus = float(objective().numpy())
    minus = original.copy().reshape(-1)
    minus[flat_index] -= step
    variable.assign(minus.reshape(original.shape))
    loss_minus = float(objective().numpy())
    variable.assign(original)
    finite_difference = (loss_plus - loss_minus) / (2.0 * step)
    analytic = float(tf.reshape(gradient, [-1])[flat_index].numpy())
    assert finite_difference == pytest.approx(analytic, rel=2.0e-4, abs=2.0e-6)
