from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_phase8_power_repair_2026_07_17.py"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_power_repair", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bindings_preserve_failed_nomination_and_no_confirmation(harness: ModuleType) -> None:
    bindings = harness._validate_bindings()
    assert bindings["failed_nomination"]["decision"] == (
        "PHASE8_CONTROLLED_NOMINATION_UNDERPOWERED_REPAIR_REQUIRED"
    )
    assert bindings["pilot"]["split_contract"]["confirmation_forecast_bank_opened"] is False
    assert harness._sha256(harness.FAILED_RUNNER_PATH) == harness.FAILED_RUNNER_SHA256


def test_nomination_binds_exact_power_repair_smoke(harness: ModuleType) -> None:
    bindings = harness._validate_bindings(require_power_repair_smoke=True)
    smoke = bindings["power_repair_smoke"]
    assert smoke["decision"] == "PHASE8_POWER_REPAIR_SMOKE_PASSED_NOMINATION_REQUIRED"
    assert smoke["configuration"]["selected_arm"] is None
    assert smoke["configuration"]["selected_mmd_tolerance"] is None
    assert harness._sha256(harness.POWER_REPAIR_SMOKE_PATH) == (
        harness.POWER_REPAIR_SMOKE_SHA256
    )


def test_candidate_ladder_uses_prospective_2048_checkpoint_minus_pilot(
    harness: ModuleType,
) -> None:
    assert harness.DRAW_COUNT == 2048 - 64
    assert harness.DRAW_COUNT % harness.BLOCK_LENGTH == 0
    assert harness.ARM_ORDER == ("B", "C", "D")
    arms = {arm.name: arm for arm in harness.ARMS}
    assert arms["B"].mean_margin == pytest.approx(0.15)
    assert arms["C"].mean_margin == pytest.approx(0.10)
    assert arms["D"].equivalence_rule == "iut_tost"
    assert arms["C"].log_variance_margin == pytest.approx(0.5 * math.log(1.25))


def test_iut_and_union_error_control_roles_are_explicit(harness: ModuleType) -> None:
    source = harness.PLAN_PATH.read_text(encoding="utf-8")
    assert "null is the union" in source
    assert "rejection requires every component TOST to reject" in source
    assert "any-feature material claim is a" in source
    assert "retains the Bonferroni simultaneous interval" in source
    assert "previously planned\n  but not yet acquired 2048-draw" in source


def test_tost_bounds_use_one_sided_alpha_and_are_narrower_than_bonferroni(
    harness: ModuleType,
) -> None:
    estimate = tf.zeros([20], tf.float64)
    standard_error = tf.ones([20], tf.float64)
    lower, upper, critical = harness._tost_bounds(
        estimate, standard_error, tf.constant(harness.FEATURE_ALPHA, tf.float64)
    )
    expected = 1.8807936081512509
    assert float(critical) == pytest.approx(expected, rel=1e-12)
    tf.debugging.assert_near(lower, tf.fill([20], tf.constant(-expected, tf.float64)))
    tf.debugging.assert_near(upper, tf.fill([20], tf.constant(expected, tf.float64)))


def test_tost_equivalence_and_bonferroni_material_roles_are_separate(
    harness: ModuleType,
) -> None:
    margins = tf.fill([20], tf.constant(0.10, tf.float64))
    bonf_lower = tf.fill([20], tf.constant(-0.12, tf.float64))
    bonf_upper = tf.fill([20], tf.constant(0.12, tf.float64))
    tost_lower = tf.fill([20], tf.constant(-0.08, tf.float64))
    tost_upper = tf.fill([20], tf.constant(0.08, tf.float64))
    assert harness._feature_status(
        bonf_lower,
        bonf_upper,
        tost_lower,
        tost_upper,
        margins,
        equivalence_rule="iut_tost",
    ) == "PASS"
    assert harness._feature_status(
        bonf_lower,
        bonf_upper,
        tost_lower,
        tost_upper,
        margins,
        equivalence_rule="symmetric_bonferroni",
    ) == "INCONCLUSIVE_UNDERPOWERED"

    material_lower = tf.tensor_scatter_nd_update(bonf_lower, [[0]], [0.11])
    assert harness._feature_status(
        material_lower,
        bonf_upper,
        tost_lower,
        tost_upper,
        margins,
        equivalence_rule="iut_tost",
    ) == "MATERIAL_DIFFERENCE"


def _passing_aggregates(harness: ModuleType, count: int) -> dict:
    result = {}
    for arm in harness.ARMS:
        result[arm.name] = {}
        for family in harness.FAMILIES:
            result[arm.name][family.name] = {
                "replication_count": count,
                "coverage_count": count,
                "by_tolerance": {
                    str(value): {
                        "pass_count": count if family.role == "equivalence" else 0,
                        "material_difference_count": count if family.role == "material" else 0,
                        "inconclusive_count": 0,
                    }
                    for value in harness.MMD_TOLERANCES
                },
            }
    return result


def test_nomination_requires_complete_count_and_uses_prospective_order(
    harness: ModuleType,
) -> None:
    complete = _passing_aggregates(harness, harness.NOMINATION_COUNT)
    assert harness._candidate_pass(complete, "B", harness.MMD_TOLERANCES[0])
    incomplete = _passing_aggregates(harness, 10)
    assert not harness._candidate_pass(incomplete, "B", harness.MMD_TOLERANCES[0])
    assert harness._viable_candidates(incomplete, 10)[0] == (
        "B",
        harness.MMD_TOLERANCES[0],
    )


def test_futility_requires_mathematical_impossibility(harness: ModuleType) -> None:
    partial = _passing_aggregates(harness, 10)
    for arm in harness.ARMS:
        partial[arm.name]["material_mean_local_h1_pos0p20"]["coverage_count"] = 7
    assert harness._viable_candidates(partial, 10) == ()


def test_runner_cannot_open_validation_hmc_or_retained_inputs(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'mode not in {"smoke", "nomination"}' in source
    assert "validation and HMC acquisition remain closed" in source
    assert "retained-acquisition.json" not in source
    assert "segment-001_retained_samples" not in source
    with pytest.raises(harness.PowerRepairError, match="remain closed"):
        harness.run(
            mode="validation",
            output=Path("/tmp/must-not-exist.json"),
            wall_cap_seconds=1.0,
        )


def test_smoke_cannot_select_an_arm_or_tolerance(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if mode == "smoke":' in source
    assert 'decision = "PHASE8_POWER_REPAIR_SMOKE_PASSED_NOMINATION_REQUIRED"' in source
    assert source.index("selected_arm = None") < source.index('if mode == "smoke":')
    assert source.index("selected_tolerance = None") < source.index('if mode == "smoke":')


def test_candidate_decisions_share_one_evidence_surface(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    one_evidence = source[source.index("def _one_evidence(") : source.index("def _aggregate(")]
    assert one_evidence.count("predictive.mean_log_variance_influence(") == 2
    assert one_evidence.count("predictive.chain_batch_long_run_covariance(") == 1
    assert one_evidence.count("predictive.simultaneous_feature_intervals(") == 1
    assert one_evidence.count("predictive.cross_chain_linear_mmd(") == 1
    assert "for arm in ARMS:" in one_evidence
