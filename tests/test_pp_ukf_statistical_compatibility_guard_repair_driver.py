from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/run_pp_ukf_statistical_compatibility_guard_repair_20260721.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pp_ukf_statistical_compatibility_guard_driver", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def _source_payloads():
    private = json.loads((ROOT / driver.SOURCE_PRIVATE).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / driver.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    return private, manifest


def test_source_reclassification_rejects_only_l3():
    private, _ = _source_payloads()
    rows = driver._source_classification_rows(private)
    dispositions = {
        int(row["num_leapfrog_steps"]): row["corrected_evidence"]["disposition"]
        for row in rows
    }
    assert dispositions[3] == "needs_lower_epsilon"
    assert all(
        dispositions[leapfrog] == "provisional_viable"
        for leapfrog in (5, 9, 13, 18, 25)
    )


def test_projection_fits_unchanged_budget_and_binds_nine_guards():
    private, manifest = _source_payloads()
    projection = driver.prospective_guard_projection(private, manifest)
    assert projection["compatible_primary_l_values"] == (5, 9, 13, 18, 25)
    assert projection["coverage_probe_l_values"] == (
        4,
        6,
        8,
        10,
        12,
        14,
        17,
        19,
        24,
    )
    assert projection["guard_l_values"] == (4, 6, 8, 10, 12, 14, 17, 19, 24)
    assert projection["guard_count"] == 9
    assert projection["guard_campaign_authorized"] is True
    assert projection["projected_cumulative_seconds"] < driver.CAMPAIGN_CAP_SECONDS
    assert projection["prior_charged_seconds"] == pytest.approx(
        9931.762329853023
    )
