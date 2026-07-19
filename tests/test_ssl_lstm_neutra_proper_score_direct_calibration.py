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
SCRIPT = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_proper_score_direct_calibration_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("proper_score_direct_calibration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_thresholds_separate_declared_anchors_on_loss_scale(harness: ModuleType) -> None:
    negligible = max(0.5 * 0.05**2, 0.25 * math.log(1.05) ** 2)
    material = min(
        0.5 * 0.20**2,
        0.25 * math.log(1.25) ** 2,
        0.25 * math.log(0.80) ** 2,
    )
    assert harness.NEGLIGIBLE_ANCHOR_LOSS == pytest.approx(negligible)
    assert harness.MATERIAL_ANCHOR_LOSS == pytest.approx(material)
    assert negligible < harness.ACCEPTABLE_AVERAGE_LOSS < material
    assert harness.ACCEPTABLE_HORIZON_LOSS == harness.ACCEPTABLE_AVERAGE_LOSS


def test_average_only_contract_is_impossible_for_local_material_variance(
    harness: ModuleType,
) -> None:
    persistent_negligible = 0.5 * 0.05**2
    local_material_variance = 0.25 * math.log(1.25) ** 2 / harness.HORIZON
    assert local_material_variance < persistent_negligible
    assert (
        0.25 * math.log(1.25) ** 2
        > harness.ACCEPTABLE_HORIZON_LOSS
    )


def test_loss_matrices_encode_average_then_ten_horizons(harness: ModuleType) -> None:
    assert harness.LOSS_MATRICES.shape == (11, 20, 20)
    average_diagonal = tf.linalg.diag_part(harness.LOSS_MATRICES[0])
    tf.debugging.assert_near(
        average_diagonal,
        tf.concat(
            (
                tf.fill([10], tf.constant(0.05, tf.float64)),
                tf.fill([10], tf.constant(0.025, tf.float64)),
            ),
            axis=0,
        ),
    )
    for horizon in range(10):
        diagonal = tf.linalg.diag_part(harness.LOSS_MATRICES[horizon + 1])
        assert int(tf.math.count_nonzero(diagonal)) == 2
        assert float(diagonal[horizon]) == pytest.approx(0.5)
        assert float(diagonal[horizon + 10]) == pytest.approx(0.25)


def test_required_families_and_truth_cover_persistent_and_local_cases(
    harness: ModuleType,
) -> None:
    required = [family for family in harness.FAMILIES if family.role != "explanatory"]
    explanatory = [family for family in harness.FAMILIES if family.role == "explanatory"]
    assert len(required) == harness.REQUIRED_FAMILY_COUNT == 11
    assert len(explanatory) == 2
    local = next(
        family
        for family in required
        if family.name == "material_variance_local_h1_1p25"
    )
    truth = harness._family_truth(local)
    assert int(tf.math.count_nonzero(truth)) == 1
    assert float(truth[10]) == pytest.approx(-math.log(1.25))


def test_fresh_seeds_and_two_rung_sequential_contract(harness: ModuleType) -> None:
    historical = {
        (14001, 14002),
        (15501, 15502),
        (16001, 16002),
        (17001, 17002),
        (18001, 18002),
        (20260717, 1901),
    }
    assert harness.DRAW_LADDER == (4096, 8192)
    assert harness.SMOKE_SEED not in historical
    assert len(set(harness.MATERIAL_SEEDS)) == 2
    assert not set(harness.MATERIAL_SEEDS) & historical
    assert harness.SMOKE_SEED not in set(harness.MATERIAL_SEEDS)


def test_simultaneous_exact_binomial_allocation_and_edge_cases(
    harness: ModuleType,
) -> None:
    assert harness.SIMULTANEOUS_CLAIM_COUNT == 11 * 4 * 2 == 88
    assert harness.BINOMIAL_TAIL_ALPHA == pytest.approx(0.05 / 88)
    lower_all, upper_all = harness._exact_one_sided_interval(256, 256)
    lower_zero, upper_zero = harness._exact_one_sided_interval(0, 256)
    assert lower_all > 0.90
    assert upper_all == 1.0
    assert lower_zero == 0.0
    assert upper_zero < 0.05
    _, upper_two = harness._exact_one_sided_interval(2, 256)
    _, upper_three = harness._exact_one_sided_interval(3, 256)
    assert upper_two < 0.05 < upper_three


def test_gate_requires_coverage_power_false_and_invalid_targets(
    harness: ModuleType,
) -> None:
    family = harness.FAMILIES[0]
    count = 256
    valid = tf.ones([count], tf.bool)
    passing = tf.ones([count], tf.bool)
    material = tf.zeros([count], tf.bool)
    covered = tf.ones([count], tf.bool)
    zeros = tf.zeros([count, 11], tf.float64)
    batch = {
        "estimate": tf.zeros([count, 20], tf.float64),
        "valid": valid,
        "pass": passing,
        "material": material,
        "covered": covered,
        "condition_number": tf.ones([count], tf.float64),
        "point_loss": zeros,
        "lower_bound": zeros,
        "upper_bound": zeros,
        "lower_kkt": zeros,
        "upper_kkt": zeros,
    }
    row = harness._summarize_family(family, [batch], count)
    assert row["gate"]["passed"]
    invalid = dict(batch)
    invalid["valid"] = tf.concat(
        (tf.zeros([3], tf.bool), tf.ones([253], tf.bool)), axis=0
    )
    invalid["pass"] = invalid["valid"]
    invalid["covered"] = invalid["valid"]
    failed = harness._summarize_family(family, [invalid], count)
    assert not failed["gate"]["invalid_procedure"]
    assert not failed["gate"]["passed"]


def test_strict_json_rejects_duplicate_and_nonfinite_values(
    harness: ModuleType, tmp_path: Path
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="ascii")
    with pytest.raises(harness.CalibrationError, match="duplicate"):
        harness._strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="ascii")
    with pytest.raises(harness.CalibrationError, match="nonfinite"):
        harness._strict_json(nonfinite)


def test_runner_has_no_hmc_private_archive_or_confirmation_input(
    harness: ModuleType,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retained-private" not in source
    assert "retained_samples.tftensor" not in source
    assert "segment-001_retained_samples" not in source
    assert "target-pilot-repair" not in source
    assert 'mode not in {"smoke", "material"}' in source
    assert "HMC acquisition and confirmation remain closed" in source
    with pytest.raises(harness.CalibrationError, match="remain closed"):
        harness.run(
            mode="hmc",
            output=Path("/tmp/direct-calibration-must-not-exist.json"),
            wall_cap_seconds=1.0,
        )


def test_smoke_receipt_binding_rejects_claim_drift(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness, "_sha256", lambda path: harness.SMOKE_RECEIPT_SHA256)
    monkeypatch.setattr(
        harness,
        "_strict_json",
        lambda path: {
            "decision": "DIRECT_CALIBRATION_SMOKE_PASSED_MATERIAL_REQUIRED",
            "configuration": {"mode": "smoke"},
            "claim_boundary": {"statistical_evidence": True},
        },
    )
    with pytest.raises(harness.CalibrationError, match="contract drift"):
        harness._validate_smoke_receipt()


def test_smoke_receipt_identity_is_frozen(harness: ModuleType) -> None:
    assert harness._sha256(harness.SMOKE_RECEIPT_PATH) == harness.SMOKE_RECEIPT_SHA256
    receipt = harness._validate_smoke_receipt()
    assert receipt["decision"] == "DIRECT_CALIBRATION_SMOKE_PASSED_MATERIAL_REQUIRED"


def test_plan_records_skeptical_audit_and_nonclaims(harness: ModuleType) -> None:
    text = (ROOT / harness.PLAN_PATH).read_text(encoding="utf-8")
    assert "PASS_FOR_CONTROLLED_IMPLEMENTATION_AND_SMOKE" in text
    assert "MMD remains explanatory" in text
    assert "No HMC" in text
    assert "0.05/88" in text
    assert "256 replications" in text
