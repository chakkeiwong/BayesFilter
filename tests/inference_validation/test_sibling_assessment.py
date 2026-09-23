"""Predeclared sibling assessments preserve fit denominators and tuning members."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from bayesfilter.testing.inference_validation.procedures import selected_member_ids, execute_pipeline
from bayesfilter.testing.inference_validation.engines.pipeline import run_replication, summarize_replications
from bayesfilter.testing.inference_validation.storage import read_json, file_hash, json_ready


def sibling_options(**extra):
    return {"member_rule": "shortest_verified_l", "posterior_member_count": 2,
            "posterior_members": "selected", **extra}


def test_selection_is_distinct_length_ordered_and_never_uses_posterior(design):
    d = design("stopping", route="prepared", options=sibling_options())
    candidates = [SimpleNamespace(candidate_id=cid, leapfrog_steps=steps)
                  for cid, steps in (("z", 3), ("a", 25), ("b", 5), ("c", 3))]
    assert selected_member_ids(d, candidates) == ("c", "b")
    assert selected_member_ids(d, reversed(candidates)) == ("c", "b")
    assert len(candidates) == 4
    assert selected_member_ids(d, candidates[:1]) == ("z",)
    assert selected_member_ids(d, []) == ()


@pytest.mark.parametrize("options", [
    sibling_options(posterior_member_count=None), sibling_options(posterior_member_count=0),
    sibling_options(posterior_member_count=True), sibling_options(posterior_member_count=1.5),
    sibling_options(posterior_members="all"),
    {"member_rule": "first_verified", "posterior_member_count": 2},
])
def test_invalid_or_implicit_sibling_allocations_fail_before_execution(design, options):
    with pytest.raises(ValueError):
        design("stopping", route="prepared", options=options)


def assessed(cid, steps):
    return {"candidate_id": cid, "L": steps, "assessment": {"finding": "within_descriptive_tolerance"},
            "warmup_exclusion_matches": True, "duplicate_chains": False,
            "runtime_checks_passed": True, "stopped_intervals": {"quantities": [
                {"name": "x", "kind": "mean", "available": True, "covered": True}]}}


def test_slot_denominators_keep_missing_fits_shortages_and_failed_processes(design):
    d = design("stopping", route="prepared", replications=4, options=sibling_options())
    base = {"inventory": {"failures": []}, "tuning_completion": "complete"}
    records = [
        dict(base, replication=0, members=[assessed("a", 3), assessed("b", 5),
            {"candidate_id": "c", "L": 9, "status": "unassessed_by_design"}],
            selection={"candidate_ids": ["a", "b"]}),
        dict(base, replication=1, members=[assessed("d", 5)], selection={"candidate_ids": ["d"]}),
        dict(base, replication=2, members=[assessed("e", 3), assessed("f", 5)],
            selection={"candidate_ids": ["e", "f"]}, execution_failure={"status": "failed"}),
    ]
    result = summarize_replications(d, records)
    assert result["completed"] == 2 and result["planned"] == 4
    assert result["verified_members"] == 6 and result["unassessed_by_design_members"] == 1
    assert result["declared_member_slots"] == 8 and result["selection_shortfall"] == 3
    assert not result["assessment_complete"]
    assert "interval_coverage_at_stop" not in result  # No arbitrary representative or pooled estimate.
    first, second = (result["member_slot_assessments"][str(i)] for i in (1, 2))
    assert first["interval_coverage_at_stop"]["x:mean"]["covered"] == 2
    assert second["interval_coverage_at_stop"]["x:mean"]["covered"] == 1
    assert first["interval_coverage_at_stop"]["x:mean"]["unavailable"] == 2
    assert second["interval_coverage_at_stop"]["x:mean"]["unavailable"] == 3
    assert first["posterior_checks_passed"] == 2 and second["posterior_checks_passed"] == 1
    assert first["posterior_available_replications"] == 2
    assert second["posterior_unavailable_slots"] == 3
    assert all(s["planned"] == 4 for s in (first, second))
    with pytest.raises(ValueError, match="missing predeclared"):
        summarize_replications(d, [dict(base, replication=0, members=[])])


def test_unassessed_siblings_do_not_hide_complete_selected_comparator(design):
    d = design("stopping", route="prepared", replications=1,
        options=sibling_options(fixed_comparator={"warmup_results": 64, "retained_results": 128}))
    first, second = assessed("first", 3), assessed("second", 5)
    for row in (first, second):
        row["stopping_pair"] = {"fixed_status": "assessed", "fixed": {}, "stopped": {}}
    record = {"replication": 0, "inventory": {"failures": []},
              "members": [first, second, {"candidate_id": "aaa-unassessed", "L": 9,
                                          "status": "unassessed_by_design"}],
              "selection": {"candidate_ids": ["first", "second"]},
              "tuning_completion": "complete"}
    result = summarize_replications(d, [record])
    assert result["assessment_complete"] and result["comparison_complete"]
    assert result["finding"] == "pipeline_assessed"


@pytest.mark.parametrize("target", ["gaussian", "rotated_gaussian"])
def test_real_public_pipeline_continues_after_failed_first_sibling_and_resumes(
        design, tmp_path, monkeypatch, target):
    import bayesfilter.inference as public
    original = public.run_hmc_posterior
    calls = []
    def first_budget_stop(**kwargs):
        calls.append(kwargs["member"].num_leapfrog_steps)
        if len(calls) == 1:
            kwargs["budget_check"] = lambda _: False
        return original(**kwargs)
    monkeypatch.setattr(public, "run_hmc_posterior", first_budget_stop)
    d = design("stopping", target, "prepared", replications=1, draws=64, posterior_cap=128,
        step_size=.9, l_grid=(3, 5), options=sibling_options(
            acceptance_policy={"practical_region": (.41, .99), "repair_region": (.405, .995)},
            search={"pilot_enabled": False, "refinement_rounds": 0, "total_budget_units": 48,
                    "repair_reserve_units": 8, "evidence_rungs": (1,)},
            fixed_comparator={"warmup_results": 64, "retained_results": 128}))
    record = run_replication(d, tmp_path, 0)
    path = tmp_path / "replication-0000"
    native = read_json(path / "pipeline.json")
    selected = native["selection"]["candidate_ids"]
    assert len(selected) == 2 and calls == [3, 5]
    assert native["selection"]["selection_shortfall"] == 0
    members = {m["candidate_id"]: m for m in native["members"]}
    assert not members[selected[0]]["posterior"]["passed"]
    assert members[selected[0]]["recorded_retained_count"] == 0
    assert members[selected[1]]["status"] == "assessed"
    assert members[selected[1]]["posterior"]["warmup_results_per_chain"] > 0
    assert set(members) == set(native["verified_candidate_ids"])
    summary = summarize_replications(d, [record])
    assert summary["member_slot_assessments"]["1"]["interval_coverage_at_stop"]["x:mean"]["unavailable"] == 1
    assert summary["member_slot_assessments"]["2"]["stopped_versus_fixed"]["planned"] == 1
    hashes = {cid: file_hash(members[cid]["warmup_path"]) for cid in selected}
    again = execute_pipeline(d, path)
    assert calls == [3, 5] and json_ready(again["selection"]) == native["selection"]
    assert hashes == {m["candidate_id"]: file_hash(m["warmup_path"])
                      for m in again["members"] if m["candidate_id"] in selected}
    changed = replace(d, options={**d.options, "posterior_member_count": 1})
    with pytest.raises(ValueError, match="selection changed"):
        execute_pipeline(changed, path)
