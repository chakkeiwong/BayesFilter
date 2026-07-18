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
    / "docs/benchmarks/run_ssl_lstm_neutra_directional_region_remedy_2026_07_18.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("directional_region_remedy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_region_and_loss_contract_is_frozen(harness: ModuleType) -> None:
    assert harness.AVERAGE_ALPHA == pytest.approx(0.025)
    assert harness.HORIZON_ALPHA == pytest.approx(0.0025)
    assert harness.AVERAGE_ALPHA + 10 * harness.HORIZON_ALPHA == pytest.approx(0.05)
    expected = 0.5 * (
        max(0.5 * 0.05**2, 0.25 * math.log(1.05) ** 2)
        + min(
            0.5 * 0.20**2,
            0.25 * math.log(1.25) ** 2,
            0.25 * math.log(0.80) ** 2,
        )
    )
    assert harness.ACCEPTABLE_LOSS == pytest.approx(expected)
    assert harness.RIDGE_LADDER == (0.0,)


def test_boundary_families_separate_leakage_from_power(harness: ModuleType) -> None:
    roles = {family.name: family.role for family in harness.FAMILIES}
    assert roles["boundary_mean_local_h1_exact"] == "boundary"
    assert roles["boundary_mean_local_h1_inside"] == "guard_equivalence"
    assert roles["boundary_mean_local_h1_outside"] == "guard_material"
    assert roles["boundary_variance_local_h1_exact_up"] == "boundary"
    assert roles["boundary_mean_persistent_exact"] == "boundary"
    assert len(harness.PRIMARY_FAMILIES) == 11
    assert harness.AUDIT_OPERATING_CLAIMS <= 120
    assert harness.AUDIT_REPLICATION_COUNT == 1536


def test_truth_constants_match_exact_boundary(harness: ModuleType) -> None:
    mean_family = next(
        family for family in harness.FAMILIES
        if family.name == "boundary_mean_local_h1_exact"
    )
    variance_family = next(
        family for family in harness.FAMILIES
        if family.name == "boundary_variance_local_h1_exact_up"
    )
    mean_truth = harness._family_truth(mean_family)
    variance_truth = harness._family_truth(variance_family)
    mean_loss = 0.5 * float(mean_truth[0]) ** 2
    variance_loss = 0.25 * float(variance_truth[10]) ** 2
    assert mean_loss == pytest.approx(harness.ACCEPTABLE_LOSS)
    assert variance_loss == pytest.approx(harness.ACCEPTABLE_LOSS)


def test_variance_stress_transforms_paths_and_conditional_moments_consistently(
    harness: ModuleType,
) -> None:
    paths = tf.ones([1, 2, 3, 2, 10], tf.float64) * 2.0
    means = tf.ones_like(paths) * 1.5
    variances = tf.ones_like(paths) * 0.4
    family = harness.Family(
        "variance", "material", variance_ratio=1.25, local_horizon=0
    )
    transformed_paths, transformed_means, transformed_variances = (
        harness._apply_family(paths, means, variances, family)
    )
    assert float(transformed_paths[0, 0, 0, 0, 0]) == pytest.approx(
        2.0 * math.sqrt(1.25)
    )
    assert float(transformed_means[0, 0, 0, 0, 0]) == pytest.approx(
        1.5 * math.sqrt(1.25)
    )
    assert float(transformed_variances[0, 0, 0, 0, 0]) == pytest.approx(0.5)
    tf.debugging.assert_equal(transformed_paths[..., 1:], paths[..., 1:])
    tf.debugging.assert_equal(transformed_means[..., 1:], means[..., 1:])
    tf.debugging.assert_equal(transformed_variances[..., 1:], variances[..., 1:])


def test_development_ladder_and_seed_domains_are_prospective(harness: ModuleType) -> None:
    candidates = harness._development_candidates()
    assert candidates[0] == ("baseline_full_path_k1", "full", "path", 1.0)
    assert {row[3] for row in candidates if row[2] == "path"} == set(
        harness.HAC_CANDIDATES
    )
    assert {row[3] for row in candidates if row[2] == "rao_blackwell"} == set(
        harness.HAC_CANDIDATES
    )
    assert len({harness.SMOKE_SEED, harness.DEVELOPMENT_SEED, harness.AUDIT_SEED}) == 3


def test_nomination_prioritizes_validity_and_coverage_before_power(
    harness: ModuleType,
) -> None:
    def row(role: str, coverage: float, required: float, false: float, invalid: float):
        return {
            "family": {"role": role},
            "coverage": {"estimate": coverage},
            "required_decision": {"estimate": required},
            "false_or_boundary_decision": {"estimate": false},
            "invalid_procedure": {"estimate": invalid},
        }

    candidates = [
        {
            "name": "high_power_low_coverage",
            "geometry": "split",
            "estimator": "path",
            "hac_multiplier": 1.0,
            "families": [row("material", 0.80, 1.0, 0.0, 0.0)],
        },
        {
            "name": "valid_coverage",
            "geometry": "split",
            "estimator": "rao_blackwell",
            "hac_multiplier": 2.0,
            "families": [row("material", 0.95, 0.85, 0.0, 0.0)],
        },
    ]
    selected = harness._nominate(candidates)
    assert selected["candidate_name"] == "valid_coverage"
    assert selected["statistical_ranking_supported"] is False


def test_audit_receipt_validation_rejects_promotion_or_geometry_drift(
    harness: ModuleType, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"schema":"bayesfilter.ssl_lstm.directional_region_remedy.v1",'
        '"configuration":{"mode":"development"},'
        '"claim_boundary":{"statistical_promotion_evidence":true},'
        '"nomination":{"geometry":"split"}}\n',
        encoding="ascii",
    )
    with pytest.raises(harness.RemedyError, match="contract drift"):
        harness._validate_development_receipt(bad)


def test_runner_has_no_hmc_retained_or_confirmation_input(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retained-private" not in source
    assert "retained_samples.tftensor" not in source
    assert "target-pilot-repair" not in source
    assert 'mode not in {"smoke", "development", "capacity", "audit"}' in source
    with pytest.raises(harness.RemedyError, match="remain closed"):
        harness.run(
            mode="hmc",
            output=Path("/tmp/directional-remedy-must-not-exist.json"),
            wall_cap_seconds=1.0,
        )


def test_plan_records_evidence_contract_and_audit(harness: ModuleType) -> None:
    text = (ROOT / harness.PLAN_PATH).read_text(encoding="utf-8")
    assert "PASS_FOR_IMPLEMENTATION_AND_BOUNDED_CONTROLLED_EXECUTION" in text
    assert "select one locked candidate but is not promotion evidence" in text
    assert "1,536 replications" in text
    assert "No HMC or NeuTra training is authorized" in text


def test_capacity_preflight_requires_margin_above_audit_threshold(
    harness: ModuleType,
) -> None:
    assert harness.CAPACITY_DRAW_LADDER == (12288, 16384)
    assert harness.CAPACITY_MINIMUM_POOLED_COVERAGE == pytest.approx(0.93)
    assert harness.CAPACITY_MAXIMUM_POOLED_COVERAGE == pytest.approx(0.97)
    assert harness.CAPACITY_MINIMUM_REQUIRED_DECISION == pytest.approx(0.85)
    assert harness.CAPACITY_MAXIMUM_FALSE_DECISION == pytest.approx(0.02)


def test_immutable_capacity_receipt_supports_reviewed_pooled_nomination(
    harness: ModuleType,
) -> None:
    nomination = harness._nominate_capacity_from_receipt(harness.CAPACITY_RECEIPT)
    assert nomination["audit_authorized"] is True
    assert nomination["draw_count"] == 12288
    assert nomination["geometry"] == "split"
    assert nomination["estimator"] == "rao_blackwell"
    assert nomination["hac_multiplier"] == pytest.approx(3.0)
    assert "minimum-across-96" in nomination["repair_note"]
