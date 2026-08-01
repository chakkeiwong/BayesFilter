"""Identity, seed, transformed-target, and dry-run tests for P2."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.neural_force_campaign import (
    NeuralForceCampaignError,
    NeuralForceTuningCandidate,
    bind_transformed_neural_force_target,
    dry_run_tier_a_registry,
    generate_neural_force_supervision,
    require_force_target_binding,
    select_health_aware_tuning_candidate,
    validate_disjoint_seed_domains,
    validate_value_only_endpoint_parity,
)
from bayesfilter.inference.neural_force_training import (
    FrozenScalarResidualForce,
    ScalarResidualPotentialNetwork,
)


TARGET = "a" * 64
TRANSPORT = "b" * 64


class NonlinearChartAdapter:
    def log_prob_and_grad_batch(self, positions):
        z = tf.convert_to_tensor(positions, tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(z)
            theta = z + 0.2 * tf.pow(z, 3)
            logdet = tf.reduce_sum(tf.math.log1p(0.6 * tf.square(z)), axis=-1)
            log_prob = -0.5 * tf.reduce_sum(tf.square(theta), axis=-1) + logdet
        score = tape.gradient(
            log_prob, z, output_gradients=tf.ones_like(log_prob)
        )
        return log_prob, score


def _frozen(target=TARGET, transport=TRANSPORT):
    network = ScalarResidualPotentialNetwork(
        dimension=2,
        hidden_layers=(3,),
        position_mean=tf.zeros([2], tf.float64),
        position_scale=tf.ones([2], tf.float64),
        seed=(2, 3),
        trainable=False,
    )
    return FrozenScalarResidualForce(network, target, transport, "c" * 64, "d" * 64)


def test_complete_transformed_target_value_force_and_endpoint_parity():
    adapter = NonlinearChartAdapter()

    def value_only(z):
        value, _score = adapter.log_prob_and_grad_batch(z)
        return -value

    binding = bind_transformed_neural_force_target(
        adapter=adapter,
        endpoint_potential_function=value_only,
        target_signature=TARGET,
        transport_signature=TRANSPORT,
        dimension=2,
    )
    z = tf.constant([[0.4, -0.2], [-0.8, 0.6]], tf.float64)
    supervision = generate_neural_force_supervision(binding, z)
    theta = z + 0.2 * tf.pow(z, 3)
    logdet = tf.reduce_sum(tf.math.log1p(0.6 * tf.square(z)), axis=-1)
    complete = 0.5 * tf.reduce_sum(tf.square(theta), axis=-1) - logdet
    raw_only = 0.5 * tf.reduce_sum(tf.square(theta), axis=-1)
    tf.debugging.assert_near(supervision.potentials, complete, atol=1e-14)
    tf.debugging.assert_near(binding.hmc_target().function(z), complete, atol=1e-14)
    assert validate_value_only_endpoint_parity(binding, z)["passed"] is True
    assert bool(tf.reduce_any(tf.abs(raw_only - complete) > 0.03))
    with tf.GradientTape() as tape:
        tape.watch(z)
        direct = binding.potential(z)
    tf.debugging.assert_near(
        supervision.forces,
        tape.gradient(direct, z, output_gradients=tf.ones_like(direct)),
        atol=1e-12,
    )


def test_target_chart_force_substitution_is_rejected():
    adapter = NonlinearChartAdapter()
    binding = bind_transformed_neural_force_target(
        adapter=adapter,
        endpoint_potential_function=lambda z: -adapter.log_prob_and_grad_batch(z)[0],
        target_signature=TARGET,
        transport_signature=TRANSPORT,
        dimension=2,
    )
    require_force_target_binding(force=_frozen(), target=binding)
    with pytest.raises(NeuralForceCampaignError, match="target signature"):
        require_force_target_binding(force=_frozen(target="e" * 64), target=binding)
    with pytest.raises(NeuralForceCampaignError, match="transport signature"):
        require_force_target_binding(force=_frozen(transport="e" * 64), target=binding)


def test_tuning_cannot_select_by_acceptance_alone_and_fails_closed():
    high_acceptance_unhealthy = NeuralForceTuningCandidate(
        "unhealthy", 0.01, 1, False, 1.001, 0.01, 1.0
    )
    healthy = NeuralForceTuningCandidate("healthy", 0.1, 8, True, 1.02, 2.0, 0.72)
    selected = select_health_aware_tuning_candidate([high_acceptance_unhealthy, healthy])
    assert selected.candidate_id == "healthy"
    with pytest.raises(NeuralForceCampaignError, match="no tuning candidate"):
        select_health_aware_tuning_candidate([high_acceptance_unhealthy])


def test_training_tuning_warmup_retained_seeds_are_disjoint():
    domains = {
        "training_screen": (20260717, 51000),
        "fresh_training": (20260717, 52000),
        "tuning": (20260717, 53000),
        "warmup": (20260717, 54000),
        "retained": (20260717, 55000),
    }
    assert validate_disjoint_seed_domains(domains)["passed"] is True
    domains["retained"] = domains["tuning"]
    with pytest.raises(NeuralForceCampaignError, match="disjoint"):
        validate_disjoint_seed_domains(domains)


def test_all_five_tier_a_cells_resolve_target_and_transport_identity():
    result = dry_run_tier_a_registry(
        Path("docs/plans/artifacts/corrected-neural-force-hmc-20260717/target_registry.json")
    )
    assert result["passed"] is True
    assert result["cell_count"] == 5
    assert {row["cell_id"] for row in result["cells"]} == {
        "LGSSM-KF", "PP-UKF", "PP-SGQF", "SIR-SGQF", "STR-UKF"
    }


def test_supervision_is_one_batch_call_without_row_loop():
    adapter = NonlinearChartAdapter()
    binding = bind_transformed_neural_force_target(
        adapter=adapter,
        endpoint_potential_function=lambda z: -adapter.log_prob_and_grad_batch(z)[0],
        target_signature=TARGET,
        transport_signature=TRANSPORT,
        dimension=2,
    )
    result = generate_neural_force_supervision(binding, tf.zeros([32, 2], tf.float64))
    assert result.positions.shape == (32, 2)
    source = inspect.getsource(generate_neural_force_supervision)
    assert "for " not in source
    assert "numpy" not in source
