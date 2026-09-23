"""A replay comparison must preserve the complete model-specific design."""
import copy
from pathlib import Path

import pytest

from docs.benchmarks.prepare_hmc_m29_2026_09_23 import baselines, build_suite, check_designs, NOTE


@pytest.fixture
def inputs():
    root = Path(__file__).resolve().parents[2]
    baseline = baselines(root / "docs/plans/artifacts/hmc-repair-master-2026-09-16/m27-r1")
    return baseline, build_suite(baseline), (root / NOTE).read_text()


def test_m29_preserves_both_model_allocations(inputs):
    baseline, suite, note = inputs
    assert check_designs(suite, baseline, note)["passed"]
    assert "profile_execution" not in suite["designs"][0]["options"]


@pytest.mark.parametrize("field", ["seed", "counts", "tolerance", "written_table"])
def test_m29_rejects_scientific_drift(inputs, field):
    baseline, suite, note = inputs
    suite = copy.deepcopy(suite)
    if field == "seed":
        suite["designs"][0]["seed"] += 1
    elif field == "counts":
        suite["designs"][0]["options"]["posterior_settings"]["warmup_max_results"] = 10000
    elif field == "tolerance":
        suite["designs"][0]["mcse_tolerance"] *= 2
    else:
        note = note.replace("| warmup_max_results | 60000 |", "| warmup_max_results | 10000 |")
    with pytest.raises(ValueError):
        check_designs(suite, baseline, note)
