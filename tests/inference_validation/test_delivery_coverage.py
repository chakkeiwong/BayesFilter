"""Independent-fit reporting: caps, missing slots and siblings are distinct."""
from copy import deepcopy

import pytest

from bayesfilter.testing.inference_validation.engines.delivery_coverage import summarize_delivery_coverage


def row(index, *, delivered=True, covered=True, available=True):
    return {"replication": index, "inventory": {"failures": []}, "members": [{
        "candidate_id": "a", "L": 3, "runtime_checks_passed": delivered,
        "warmup_exclusion_matches": True, "duplicate_chains": False,
        "stopped_intervals": {"quantities": [{"name": "x", "kind": "mean",
            "available": available, "covered": covered}]}}]}


def summarize(rows, planned=256, **kwargs):
    return summarize_delivery_coverage(rows, planned, declared_names=("x:mean", "y:quantile"),
                                       member_rule="first_verified", member_l=3, **kwargs)


def test_delivery_coverage_and_joint_events_do_not_substitute_for_each_other():
    rows = [row(0), row(1, delivered=False), row(2, covered=False),
            {"replication": 3, "execution_failure": "timeout"}]
    # A second successful sibling cannot rescue the selected capped member.
    sibling = deepcopy(rows[1]["members"][0])
    sibling.update(candidate_id="z", runtime_checks_passed=True)
    rows[1]["members"].append(sibling)
    result = summarize(rows, planned=5, coverage_floor=.90)
    quantity = result["quantities"]["x:mean"]
    assert result["delivery"]["count"] == quantity["coverage"]["count"] == 2
    assert quantity["delivery_and_coverage"]["count"] == 1
    assert quantity["delivery_and_coverage"]["total"] == 5
    assert quantity["conditional_coverage"]["total"] == 3
    assert "screen_passed" not in quantity["conditional_coverage"]
    assert result["quantities"]["y:quantile"]["unavailable"] == 5
    assert result["unstarted"] == 1


@pytest.mark.parametrize("count,passes", [(239, False), (240, True)])
def test_original_256_fit_exact_screen_boundary(count, passes):
    result = summarize([row(i) for i in range(count)], coverage_floor=.90)
    assert result["delivery"]["screen_passed"] is passes
    assert result["quantities"]["x:mean"]["delivery_and_coverage"]["screen_passed"] is passes
    assert result["planned"] == 256


def test_no_implicit_promotion_threshold_and_empty_inventory():
    result = summarize([])
    assert result["delivery"]["count"] == 0
    assert result["delivery"]["screen_passed"] is None
    assert result["quantities"]["x:mean"]["conditional_coverage"]["interval"] is None


@pytest.mark.parametrize("rows", [[row(0), row(0)], [row(-1)], [row(256)]])
def test_invalid_fit_inventory_rejected(rows):
    with pytest.raises(ValueError, match="indices"):
        summarize(rows)


@pytest.mark.parametrize("key,value", [("duplicate_chains", True), ("warmup_exclusion_matches", False)])
def test_readiness_cannot_override_numerical_veto(key, value):
    record = row(0)
    record["members"][0][key] = value
    assert summarize([record])["delivery"]["count"] == 0


def test_available_interval_required_for_coverage():
    with pytest.raises(ValueError, match="availability"):
        summarize([row(0, available=False)])


def test_numpy_flags_and_serialized_flags_give_the_same_report():
    import numpy as np
    record = row(0)
    expected = summarize([record], coverage_floor=.90)
    record["members"][0]["stopped_intervals"]["quantities"][0]["covered"] = np.bool_(True)
    assert summarize([record], coverage_floor=.90) == expected


def test_legacy_implicit_order_cannot_supply_a_confirmation_screen():
    record = row(0)
    del record["replication"]
    assert summarize([record])["replication_identity"] == "implicit_diagnostic_order"
    with pytest.raises(ValueError, match="explicit replication"):
        summarize([record], coverage_floor=.90)
