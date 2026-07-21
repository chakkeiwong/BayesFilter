from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_pp_ukf_state_continuing_epsilon_repair_20260721.py"
SPEC = importlib.util.spec_from_file_location("pp_ukf_state_continuing_driver", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_repair_direction_and_unbracketed_factor():
    increased, low, high, action = driver.next_repair_epsilon(
        epsilon=0.8,
        acceptance_mean=0.81,
        lower_epsilon=None,
        upper_epsilon=None,
    )
    assert increased == pytest.approx(0.96)
    assert low == pytest.approx(0.8)
    assert high is None
    assert action == "increase_epsilon"

    decreased, low, high, action = driver.next_repair_epsilon(
        epsilon=0.8,
        acceptance_mean=0.60,
        lower_epsilon=None,
        upper_epsilon=None,
    )
    assert decreased == pytest.approx(0.8 / 1.2)
    assert low is None
    assert high == pytest.approx(0.8)
    assert action == "decrease_epsilon"


def test_repair_uses_geometric_midpoint_after_bracketing():
    proposal, low, high, action = driver.next_repair_epsilon(
        epsilon=1.2,
        acceptance_mean=0.60,
        lower_epsilon=0.8,
        upper_epsilon=None,
    )
    assert low == pytest.approx(0.8)
    assert high == pytest.approx(1.2)
    assert proposal == pytest.approx(math.sqrt(0.8 * 1.2))
    assert action == "geometric_bracket_midpoint"


def test_in_region_epsilon_is_unchanged():
    proposal, low, high, action = driver.next_repair_epsilon(
        epsilon=0.9,
        acceptance_mean=0.70,
        lower_epsilon=0.8,
        upper_epsilon=1.0,
    )
    assert proposal == pytest.approx(0.9)
    assert low == pytest.approx(0.8)
    assert high == pytest.approx(1.0)
    assert action == "calibration_region_reached"


def test_primary_projection_fits_unchanged_remaining_budget():
    payload = json.loads((ROOT / driver.PRIOR_RESULT).read_text(encoding="utf-8"))
    projection = driver.prospective_primary_projection(payload)
    assert projection["primary_barrier_authorized"] is True
    assert projection["prior_charged_seconds"] == pytest.approx(
        6994.005394253036
    )
    assert projection["projected_cumulative_seconds"] < driver.CAMPAIGN_CAP_SECONDS
    assert len(projection["rows"]) == 6
    assert all(float(item["projected_seconds"]) > 0.0 for item in projection["rows"])


def test_frozen_protocol_constants_preserve_required_roles():
    assert driver.PRIMARY_L_GRID == (3, 5, 9, 13, 18, 25)
    assert driver.ADAPTATION_STEPS == 96
    assert driver.POST_ADAPTATION_RESULTS == 32
    assert driver.CALIBRATION_REGION == (0.68, 0.72)
    assert driver.FINAL_SCREEN_RESULTS == 96
    assert driver.FINAL_SCREEN_BURNIN == 8
    assert driver.REPLICATION_COUNT == 3


def test_active_plan_is_distinct_from_withdrawn_broad_grid_plan():
    assert driver.PLAN != driver.base.PLAN
