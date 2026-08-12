"""Focused checks for the frozen analytic HMC replication harness."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "weighted_neutra_analytic_hmc_replication_runner",
    "docs/benchmarks/run_defensive_weighted_neutra_analytic_hmc_replication_2026_08_12.py",
)
SUMMARY = _load(
    "weighted_neutra_analytic_hmc_replication_summary",
    "docs/benchmarks/summarize_defensive_weighted_neutra_analytic_hmc_replications_2026_08_12.py",
)


def test_frozen_inputs_match_reviewed_checkpoint_and_kernel() -> None:
    frozen = RUNNER._validate_frozen_inputs()
    assert frozen["sha256"] == RUNNER.EXPECTED_TUNING_SHA256
    assert frozen["kernel"]["num_leapfrog_steps"] == 20
    assert frozen["kernel"]["step_size"] == RUNNER.EXPECTED_STEP_SIZE
    assert frozen["kernel"]["mass_policy"] == "fixed_identity_z"


def test_target_signature_migration_is_one_ulp_and_value_score_identical() -> None:
    target = RUNNER.analytic_two_mode_target() if hasattr(RUNNER, "analytic_two_mode_target") else None
    if target is None:
        from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import analytic_two_mode_target

        target = analytic_two_mode_target()
    compatibility = RUNNER._target_signature_compatibility(tf, target)
    assert compatibility["passed"] is True
    assert compatibility["comparison_point_count"] == 4110
    assert compatibility["maximum_covariance_absolute_delta"] <= 2.0**-54
    assert set(compatibility["maximum_output_absolute_deltas"].values()) == {0.0}


def test_predeclared_root_seeds_are_distinct() -> None:
    seeds = [(20260812, 91011 + replication) for replication in range(4)]
    assert len(set(seeds)) == 4
    assert seeds == [
        (20260812, 91011),
        (20260812, 91012),
        (20260812, 91013),
        (20260812, 91014),
    ]


def test_summary_stats_are_descriptive_only() -> None:
    summary = SUMMARY._summary_stats([0.18, 0.20, 0.22, 0.20])
    assert summary["mean"] == 0.2
    assert summary["minimum"] == 0.18
    assert summary["maximum"] == 0.22
    assert summary["role"] == "descriptive_between_root_seed_variability_only"


def test_mode_transition_diagnostics_count_each_chain() -> None:
    physical = tf.constant(
        [
            [[-4.0, -0.5, 0.75, -0.25], [4.0, 0.5, -0.75, 0.25]],
            [[4.0, 0.5, -0.75, 0.25], [-4.0, -0.5, 0.75, -0.25]],
            [[-4.0, -0.5, 0.75, -0.25], [4.0, 0.5, -0.75, 0.25]],
        ],
        tf.float64,
    )
    diagnostics = RUNNER._mode_transition_diagnostics(tf, physical)
    assert diagnostics["hard_assignment_transition_count_by_chain"].numpy().tolist() == [2, 2]
    assert diagnostics["all_chains_transitioned"] is True
