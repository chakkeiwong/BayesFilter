"""CPU-only mechanics checks for the defensive weighted NeuTra GPU runner."""

from __future__ import annotations

import importlib.util
import os
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    WeightedForwardKLNeuTraTrainer,
    WeightedNeuTraConfig,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/benchmarks/run_defensive_weighted_neutra_analytic_2026_08_11.py"
)
SPEC = importlib.util.spec_from_file_location("weighted_neutra_analytic_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

SUMMARY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/benchmarks/summarize_defensive_weighted_neutra_replications_2026_08_11.py"
)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "weighted_neutra_replication_summary", SUMMARY_SCRIPT
)
assert SUMMARY_SPEC is not None and SUMMARY_SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY)


def test_two_mode_target_analytic_moments_match_direct_mixture_formula() -> None:
    target = RUNNER._target("two-mode-canary", tf, tf.float64)
    probabilities = target["target_probabilities"]
    means = target["means"]
    covariances = target["covariances"]
    expected_mean = tf.reduce_sum(probabilities[:, tf.newaxis] * means, axis=0)
    centered = means - expected_mean
    expected_covariance = tf.reduce_sum(
        probabilities[:, tf.newaxis, tf.newaxis]
        * (covariances + centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :]),
        axis=0,
    )
    tf.debugging.assert_near(target["true_mean"], expected_mean, atol=1.0e-14)
    tf.debugging.assert_near(
        target["true_covariance"], expected_covariance, atol=1.0e-14
    )


def test_three_mode_target_is_noncollinear_and_has_exact_moments() -> None:
    target = RUNNER._target("three-mode-canary", tf, tf.float64)
    assert target["identity"] == "separated_three_mode_unequal_weight_d4_v1"
    tf.debugging.assert_near(
        target["target_probabilities"],
        tf.constant((0.5, 0.3, 0.2), tf.float64),
        atol=1.0e-15,
    )
    edges = target["means"][1:, :2] - target["means"][:1, :2]
    assert abs(float(tf.linalg.det(edges).numpy())) > 1.0
    expected_mean = tf.reduce_sum(
        target["target_probabilities"][:, tf.newaxis] * target["means"], axis=0
    )
    centered = target["means"] - expected_mean
    expected_covariance = tf.reduce_sum(
        target["target_probabilities"][:, tf.newaxis, tf.newaxis]
        * (
            target["covariances"]
            + centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :]
        ),
        axis=0,
    )
    tf.debugging.assert_near(target["true_mean"], expected_mean, atol=1.0e-14)
    tf.debugging.assert_near(
        target["true_covariance"], expected_covariance, atol=1.0e-14
    )


def test_runner_accepts_an_explicit_plan_binding() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--plan-file", type=Path, default=None)' in source
    assert 'raise FileNotFoundError(f"active plan is missing: {active_plan}")' in source
    assert '"active_plan_file": active_plan.as_posix()' in source
    assert source.count("cwd=ROOT,") >= 2


def test_two_mode_coverage_keeps_historical_alias() -> None:
    target = RUNNER._target("two-mode-canary", tf, tf.float64)
    config = WeightedNeuTraConfig(
        dimension=4,
        hidden_layers=(4,),
        stages=1,
        initialization_seed=(20260812, 12100),
        jit_compile=False,
    )
    weighted = WeightedForwardKLNeuTraTrainer(config)
    reverse = WeightedForwardKLNeuTraTrainer(config)
    decision = RUNNER._mixture_canary_decision(
        tf,
        weighted,
        reverse,
        target,
        256,
        seed_root=20260812,
    )
    coverage = decision["base_component_coverage"]["weighted"]
    assert coverage["both_components_observed"] == coverage["all_components_observed"]


def test_snapshot_restores_transport_optimizer_and_step() -> None:
    config = WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(4,),
        stages=1,
        initialization_seed=(20260811, 7101),
        jit_compile=False,
    )
    trainer = WeightedForwardKLNeuTraTrainer(config)
    rows = tf.constant(((0.0, 0.0), (1.0, -1.0), (0.5, 0.25)), tf.float64)
    weights = tf.zeros(3, tf.float64)
    trainer.train_step(rows, weights)
    snapshot = RUNNER._snapshot(trainer)
    expected_transport = [value.numpy().copy() for value in snapshot["transport"]]
    expected_optimizer = [value.numpy().copy() for value in snapshot["optimizer"]]
    trainer.train_step(rows + 1.0, weights)
    RUNNER._restore(trainer, snapshot)
    assert int(trainer.step.numpy()) == 1
    for actual, expected in zip(trainer.variables, expected_transport):
        tf.debugging.assert_equal(actual, expected)
    for actual, expected in zip(trainer.optimizer.variables, expected_optimizer):
        tf.debugging.assert_equal(actual, expected)


def test_gaussian_decision_uses_same_audit_cloud_baseline() -> None:
    audit = {
        "initial": {"weighted_nll": 10.0},
        "weighted": {
            "all_finite": True,
            "weighted_nll": 9.0,
            "latent_weighted_mean_norm": 0.01,
            "latent_covariance_error_frobenius": 0.02,
            "base_pushforward_relative_covariance_error": 0.03,
        },
        "reverse_kl": {"weighted_nll": 9.5},
    }
    decision = RUNNER._gaussian_canary_decision(audit)
    assert decision["candidate_passed"]
    assert decision["gates"]["heldout_nll_improved_from_initial"]


def test_replication_interval_contains_constant_truth() -> None:
    interval = SUMMARY._interval([0.2] * 8)
    assert interval["mean"] == 0.2
    assert interval["lower"] == 0.2
    assert interval["upper"] == 0.2


def test_replication_interval_requires_exactly_eight_runs() -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        SUMMARY._interval([0.2] * 7)


def test_replication_summary_accepts_explicit_mixed_version_names() -> None:
    names = tuple(f"replication-{index}-v{2 if index == 4 else 1}" for index in range(8))
    args = Namespace(
        replication_names=names,
        replication_zero_name="unused",
        replication_prefix="unused",
    )
    roots = SUMMARY._replication_roots(Path("/tmp/input"), args)
    assert [path.name for path in roots] == list(names)


def test_confirmation_replication_ids_can_exclude_selection_seed() -> None:
    args = Namespace(
        expected_replications=tuple(range(1, 9)),
        replication_names=tuple(f"replication-{index}" for index in range(1, 9)),
    )
    assert SUMMARY._expected_replication_ids(args) == list(range(1, 9))


def test_explicit_replication_ids_require_explicit_paths() -> None:
    args = Namespace(expected_replications=tuple(range(1, 9)), replication_names=None)
    with pytest.raises(ValueError, match="requires --replication-names"):
        SUMMARY._expected_replication_ids(args)
