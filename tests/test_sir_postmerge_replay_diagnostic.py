"""CPU-only reporting integrity checks; no scientific kernels or GPU imports."""

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_sir_postmerge_replay.py"
SPECIFICATION = importlib.util.spec_from_file_location("postmerge_reporting", MODULE_PATH)
REPORTING = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(REPORTING)


def test_input_paths_normalize_without_weakening_hash_comparison(tmp_path):
    absolute = str(tmp_path / "data.json")
    normalize = REPORTING.normalized_input_hashes
    assert normalize({"data.json": "hash"}, tmp_path) == normalize({absolute: "hash"}, tmp_path)
    assert normalize({"data.json": "hash"}, tmp_path) != normalize({absolute: "changed"}, tmp_path)
    assert normalize({"other.json": "hash"}, tmp_path) != normalize({absolute: "hash"}, tmp_path)
    with pytest.raises(ValueError, match="Duplicate input path alias"):
        normalize({"data.json": "hash", absolute: "hash"}, tmp_path)


@pytest.mark.parametrize("change", ["metadata", "computation", "side_effect"])
def test_driver_equivalence_is_limited_to_plan_metadata(tmp_path, change):
    baseline = tmp_path / "before.py"
    candidate = tmp_path / "after.py"
    baseline.write_text('manifest = dict(plan="old.md", seed=123)\nvalue = kernel(1)\n')
    candidate_text = ('parser.add_argument("--plan", default="old.md")\n'
                      'manifest = dict(plan=arguments.plan, seed=123)\nvalue = kernel(1)\n')
    if change == "computation":
        candidate_text = candidate_text.replace("kernel(1)", "kernel(2)")
    if change == "side_effect":
        candidate_text = candidate_text.replace("plan=arguments.plan", "plan=modify_kernel()")
    candidate.write_text(candidate_text)
    manifests = {lane: dict(script_sha256=REPORTING.digest(path))
                 for lane, path in (("before", baseline), ("after", candidate))}
    if change == "metadata":
        result = REPORTING.verify_replay_drivers(manifests, baseline, candidate)
        assert result["mode"] == "identical_computation_after_plan_metadata_normalization"
    else:
        with pytest.raises(ValueError, match="beyond plan metadata|Unsupported plan metadata"):
            REPORTING.verify_replay_drivers(manifests, baseline, candidate)


def test_driver_differences_require_both_hash_verified_sources(tmp_path):
    baseline = tmp_path / "before.py"
    candidate = tmp_path / "after.py"
    baseline.write_text("value = 1\n")
    candidate.write_text("value = 2\n")
    manifests = {lane: dict(script_sha256=REPORTING.digest(path))
                 for lane, path in (("before", baseline), ("after", candidate))}
    with pytest.raises(ValueError, match="provide both frozen drivers"):
        REPORTING.verify_replay_drivers(manifests)
    with pytest.raises(ValueError, match="Both frozen replay drivers"):
        REPORTING.verify_replay_drivers(manifests, baseline)
    candidate.write_text("value = 1\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        REPORTING.verify_replay_drivers(manifests, baseline, candidate)


def test_numerical_comparison_rejects_shape_and_nonfinite_errors():
    with pytest.raises(ValueError, match="shapes differ"):
        REPORTING.compare_numeric([[1., 2.]], [1., 2.])
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Non-finite"):
            REPORTING.compare_numeric([1.], [value])


def test_numerical_comparison_obeys_absolute_and_relative_tolerance():
    assert REPORTING.compare_numeric(0., 9e-9)["passed"]
    assert not REPORTING.compare_numeric(0., 2e-8)["passed"]
    assert REPORTING.compare_numeric(1e8, 1e8 + .5)["passed"]
    assert not REPORTING.compare_numeric(1e8, 1e8 + 2.)["passed"]


def test_duplicate_rows_are_rejected(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {"dataset": 1, "arm": "trained", "replicate": 0}
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        REPORTING.read_rows(path)


def test_complete_grid_counts():
    manifest = {"dataset_seeds": list(range(64)), "score_datasets": [0, 3, 20, 51]}
    assert len(REPORTING.expected_keys(manifest)) == 2496
    assert len(REPORTING.expected_keys(manifest, directional=True)) == 432


def test_measurement_flag_is_distinct_from_validity():
    row = {"valid": True, "checks": {"program_valid": True,
                                      "guide_conditioning_margin_measured": False}}
    assert REPORTING.validity(row) == (True, [])
    row["checks"]["program_valid"] = False
    assert REPORTING.validity(row) == (False, ["program_valid"])


def test_added_passing_monitor_is_allowed_but_missing_validity_is_not():
    key = (1, "prediction", None)
    row = {"arm": "prediction", "valid": True, "loss": 1., "blocks": [1., 1.],
           "checks": {"program_valid": True}}
    added = {**row, "checks": {**row["checks"], "extra_valid": True}}
    assert REPORTING.compare_rows({key: row}, {key: added})["passed"]
    missing = {**row, "checks": {}}
    assert not REPORTING.compare_rows({key: row}, {key: missing})["passed"]


def test_runtime_excludes_either_cold_call_and_averages_within_dataset():
    before, after = {}, {}
    for replicate, seconds in enumerate((1., 3., 100.)):
        key = (1, "trained", replicate)
        before[key] = {"dataset": 1, "arm": "trained", "evaluation_seconds": seconds,
                       "cold": False}
        after[key] = {**before[key], "evaluation_seconds": 2. * seconds,
                      "cold": replicate == 2}
    result = REPORTING.runtime_report(before, after)
    summary = result["warm_paired_dataset_summaries"]["trained"]
    assert summary["datasets"] == 1
    assert summary["before_mean"] == 2.
    assert summary["after_mean"] == 4.
    assert summary["difference_percentile_95"] == [2., 2.]
    assert len(result["cold_calls"]["after"]) == 1


def test_row_pairing_and_validity_changes_cannot_pass():
    key = (1, "prediction", None)
    row = {"arm": "prediction", "valid": True, "loss": 1., "blocks": [1., 1.],
           "observation_regime": "low", "compartment_regime": "interior"}
    assert REPORTING.compare_rows({key: row}, {key: row})["passed"]
    invalid = {**row, "valid": False}
    assert not REPORTING.compare_rows({key: row}, {key: invalid})["passed"]
    mismatched = {**row, "observation_regime": "high"}
    with pytest.raises(ValueError, match="Pairing"):
        REPORTING.compare_rows({key: row}, {key: mismatched})


def test_per_arm_numerical_summary_localizes_failures():
    before, after = {}, {}
    for arm in ("plain", "trained"):
        for replicate in range(2):
            key = (1, arm, replicate)
            before[key] = {field: 1. for field in REPORTING.FIELDS}
            before[key].update(arm=arm, valid=True)
            after[key] = dict(before[key])
    failed_key = (1, "plain", 1)
    after[failed_key]["score"] = 2.
    result = REPORTING.compare_rows(before, after)
    assert not result["passed"]
    assert result["fields"]["score"]["rows"] == 4
    assert result["fields"]["score"]["failed_rows"] == 1
    assert result["by_arm"]["trained"]["score"]["passed"]
    plain = result["by_arm"]["plain"]["score"]
    assert plain["rows"] == 2
    assert plain["failed_rows"] == 1
    assert plain["worst_key"] == failed_key
    assert plain["max_absolute_key"] == failed_key


def test_loss_report_preserves_frozen_regimes_and_requires_coverage():
    rows = {}
    for dataset, regime in enumerate((("low", "nearer_boundary"), ("low", "interior"),
                                     ("high", "nearer_boundary"), ("high", "interior"))):
        for arm in REPORTING.PARTICLE_ARMS + REPORTING.HEURISTIC_ARMS:
            rows[(dataset, arm)] = {
                "dataset": dataset, "arm": arm,
                "loss": 2. if arm in REPORTING.PARTICLE_ARMS else 1.,
                "observation_regime": regime[0], "compartment_regime": regime[1]}
    report = REPORTING.loss_report(rows)
    assert report["conditional"]["all"]["datasets"] == 4
    assert report["conditional"]["compartment_nearer_boundary"]["datasets"] == 2
    assert report["conditional"]["compartment_interior"]["datasets"] == 2
    assert report["conditional"]["low_nearer_boundary"]["datasets"] == 1
    assert report["conditional"]["high_nearer_boundary"]["datasets"] == 1
    assert report["heuristic_dominance_verdict"] == "promotion_veto"
    interior = {key: row for key, row in rows.items()
                if row["compartment_regime"] == "interior"}
    with pytest.raises(ValueError, match="compartment_nearer_boundary"):
        REPORTING.loss_report(interior)
