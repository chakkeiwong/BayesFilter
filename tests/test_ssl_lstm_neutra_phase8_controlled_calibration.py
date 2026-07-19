from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_phase8_controlled_calibration_2026_07_17.py"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_controlled_calibration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configuration_respects_materiality_and_confirmation_shape(harness: ModuleType) -> None:
    assert harness.MEAN_MARGIN < 0.20
    assert harness.LOG_VARIANCE_MARGIN < abs(harness.math.log(1.25))
    assert harness.DRAW_COUNT == 448
    assert harness.BLOCK_LENGTH == 16
    assert harness.DRAW_COUNT % harness.BLOCK_LENGTH == 0
    assert harness.FEATURE_ALPHA + harness.MMD_ALPHA == pytest.approx(harness.TOTAL_ALPHA)
    assert harness.NOMINATION_SEED != harness.VALIDATION_SEED


def test_feature_margin_vector_matches_classifier_shape(harness: ModuleType) -> None:
    margins = harness._feature_margins()
    assert tuple(margins.shape) == (20,)
    tf.debugging.assert_equal(
        margins[: harness.HORIZON],
        tf.fill([harness.HORIZON], tf.constant(harness.MEAN_MARGIN, tf.float64)),
    )
    tf.debugging.assert_equal(
        margins[harness.HORIZON :],
        tf.fill(
            [harness.HORIZON],
            tf.constant(harness.LOG_VARIANCE_MARGIN, tf.float64),
        ),
    )


def test_required_and_explanatory_families_are_separated(harness: ModuleType) -> None:
    roles = {family.name: family.role for family in harness.FAMILIES}
    assert roles["material_mean_persistent_pos0p20"] == "material"
    assert roles["material_mean_persistent_neg0p20"] == "material"
    assert roles["material_variance_persistent_1p25"] == "material"
    assert roles["material_variance_persistent_0p80"] == "material"
    assert roles["skew_explanatory"] == "explanatory"
    assert roles["dependence_explanatory"] == "explanatory"


def test_fixture_shapes_truth_and_seed_replay(harness: ModuleType) -> None:
    left, right, truth = harness._family_paths(harness.NOMINATION_SEED, 2, 0)
    replay = harness._family_paths(harness.NOMINATION_SEED, 2, 0)
    assert tuple(left.shape) == tuple(right.shape) == (4, 448, 2, 10)
    assert tuple(truth.shape) == (20,)
    tf.debugging.assert_equal(left, replay[0])
    tf.debugging.assert_equal(right, replay[1])
    tf.debugging.assert_near(truth[:10], tf.fill([10], tf.constant(-0.05, tf.float64)))


def test_nomination_rule_requires_every_required_family(harness: ModuleType) -> None:
    passing = {}
    for family in harness.FAMILIES:
        passing[family.name] = {
            "replication_count": 20,
            "coverage_count": 20,
            "by_tolerance": {
                str(value): {
                    "pass_count": 20 if family.role == "equivalence" else 0,
                    "material_difference_count": 20 if family.role == "material" else 0,
                    "inconclusive_count": 0,
                }
                for value in harness.MMD_TOLERANCES
            },
        }
    assert harness._nomination_pass(passing, harness.MMD_TOLERANCES[0])
    passing["material_mean_local_h1_pos0p20"]["by_tolerance"]["0.005"][
        "material_difference_count"
    ] = 15
    assert not harness._nomination_pass(passing, 0.005)


def test_nomination_requires_complete_count_and_futility_is_mathematical(
    harness: ModuleType,
) -> None:
    partial = {}
    for family in harness.FAMILIES:
        partial[family.name] = {
            "replication_count": 10,
            "coverage_count": 10,
            "by_tolerance": {
                str(value): {
                    "pass_count": 10 if family.role == "equivalence" else 0,
                    "material_difference_count": 10 if family.role == "material" else 0,
                    "inconclusive_count": 0,
                }
                for value in harness.MMD_TOLERANCES
            },
        }
    assert not harness._nomination_pass(partial, harness.MMD_TOLERANCES[0])
    assert harness._viable_nomination_tolerances(partial, 10) == harness.MMD_TOLERANCES

    partial["material_mean_local_h1_pos0p20"]["coverage_count"] = 7
    assert harness._viable_nomination_tolerances(partial, 10) == ()


def test_nomination_sequential_stop_is_strictly_before_final_replication(
    harness: ModuleType,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "if replication + 1 < count and not viable:" in source


def test_exact_binomial_thresholds_match_prospective_validation_count(
    harness: ModuleType,
) -> None:
    coverage_lower, _ = harness._exact_bounds(56, 60)
    power_lower, _ = harness._exact_bounds(49, 60)
    _, one_event_upper = harness._exact_bounds(1, 60)
    _, two_event_upper = harness._exact_bounds(2, 60)
    assert coverage_lower >= 0.85
    assert power_lower >= 0.70
    assert one_event_upper <= 0.10
    assert two_event_upper > 0.10


def test_smoke_uses_only_null_and_material_mean_without_nomination(
    harness: ModuleType,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '(0, 4) if mode == "smoke"' in source
    assert "PHASE8_CONTROLLED_CALIBRATION_SMOKE_PASSED_NOMINATION_REQUIRED" in source


def test_nomination_binds_exact_passing_smoke(harness: ModuleType) -> None:
    smoke = harness._validate_smoke()
    assert smoke["decision"] == (
        "PHASE8_CONTROLLED_CALIBRATION_SMOKE_PASSED_NOMINATION_REQUIRED"
    )
    assert harness._sha256(harness.SMOKE_RECEIPT_PATH) == harness.SMOKE_RECEIPT_SHA256


def test_validation_is_closed_until_nomination_is_hard_bound(harness: ModuleType) -> None:
    with pytest.raises(harness.CalibrationError, match="nomination receipt"):
        harness.run(
            mode="validation",
            output=Path("/tmp/must-not-exist.json"),
            wall_cap_seconds=1.0,
            selected_tolerance=harness.MMD_TOLERANCES[0],
        )


def test_serious_run_requires_gpu_and_single_trace(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_require_gpu(left, right, truth" in source
    assert '"compile_trace_counts": trace_counts' in source
    assert "if any(value != 1 for value in trace_counts.values())" in source


def test_runner_has_no_confirmation_or_retained_archive_input(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retained-acquisition.json" not in source
    assert "segment-001_retained_samples" not in source
    assert "target-pilot-repair-03.json" in source
