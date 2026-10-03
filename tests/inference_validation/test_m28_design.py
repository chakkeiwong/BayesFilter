"""Prevent plan/JSON count drift before the supplied-map campaign launch."""
import copy
from pathlib import Path

import pytest

from docs.benchmarks.prepare_hmc_m28_2026_09_23 import PLAN, build_suite, check_counts
from bayesfilter.testing.inference_validation.execution import plan_suite


def test_four_resolved_map_seed_cells_match_written_count_table():
    suite = build_suite()
    plan = plan_suite(suite)
    designs = [job["design"] for job in plan["jobs"]]
    assert check_counts(designs, Path(PLAN).read_text())["checked_designs"] == 4
    assert {(d["options"]["transport_payload"]["construction"]["kind"], d["seed"])
            for d in designs} == {(k, s) for k in ("exact", "residual") for s in (2026092381, 2026092382)}
    assert plan["maximum_worker_seconds"] == 4000
    assert all(tuple(d["l_grid"]) == (3, 5, 9, 13, 18, 25) for d in designs)


def test_count_check_rejects_warmup_cap_drift_and_missing_reason():
    designs = build_suite()["designs"]
    changed = copy.deepcopy(designs)
    changed[0]["options"]["posterior_settings"]["warmup_max_results"] = 10000
    with pytest.raises(ValueError, match="warmup_max_results"):
        check_counts(changed, Path(PLAN).read_text())
    changed = copy.deepcopy(designs)
    del changed[-1]["options"]["posterior_count_budget"]["count_budget_reason"]
    with pytest.raises(ValueError, match="reason"):
        check_counts(changed, Path(PLAN).read_text())
    with pytest.raises(ValueError, match="table"):
        check_counts(designs, Path(PLAN).read_text().replace(
            "| posterior_settings.warmup_max_results | 30000 |", ""))
