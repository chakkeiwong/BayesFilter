from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_pp_ukf_operational_broad_grid_20260721.py"
SPEC = importlib.util.spec_from_file_location("pp_ukf_operational_broad_grid_driver", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_exact_primary_grid_and_work_contract():
    assert driver.PRIMARY_L_GRID == (3, 5, 9, 13, 18, 25)
    assert driver.INITIAL_STEP_SIZE == 0.9853849721883557
    assert driver.primary_transition_leapfrog_work(3) == 3 * (65 + 3 * 129)
    assert driver.worst_case_guard_transition_leapfrog_work() == 120 * 3 * 129


def test_resource_projection_charges_prior_work_and_worst_case_guards():
    projection = driver.project_remaining_campaign_seconds(
        warm_seconds_per_transition_leapfrog=0.01,
        canary_wall_seconds=100.0,
        prior_charged_seconds=200.0,
    )
    assert projection["remaining_primary_transition_leapfrogs"] == 70 * 452
    assert projection["guard_transition_leapfrogs_charged"] == 0
    assert projection["projected_cumulative_seconds"] == 300.0 + 1.5 * 0.01 * (70 * 452)
    assert projection["full_grid_authorized"] is True


def test_resource_projection_fails_closed_above_cap():
    projection = driver.project_remaining_campaign_seconds(
        warm_seconds_per_transition_leapfrog=1.0,
        canary_wall_seconds=1000.0,
        prior_charged_seconds=driver.CAMPAIGN_CAP_SECONDS - 1.0,
    )
    assert projection["full_grid_authorized"] is False


def test_guard_projection_charges_only_actual_guard_requests():
    projection = driver.project_guard_barrier_seconds(
        warm_seconds_per_transition_leapfrog=0.01,
        current_attempt_wall_seconds=500.0,
        prior_charged_seconds=200.0,
        guard_l_values=(8, 10),
    )
    assert projection["actual_guard_count"] == 2
    assert projection["guard_transition_leapfrogs"] == 18 * 3 * 129
    assert projection["projected_cumulative_seconds"] == 700.0 + 1.5 * 0.01 * (18 * 3 * 129)
    assert projection["guard_barrier_authorized"] is True
