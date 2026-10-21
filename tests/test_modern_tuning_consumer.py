"""Public numerical consumer regressions; explicitly CPU mechanics evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest

from bayesfilter.testing.inference_validation.modern_tuning_consumer import (
    candidate_inventory, run_frozen_trajectory_search, validate_frozen_search)
from tests.test_hmc_candidate_set_execution import make_binding
from tests.test_hmc_duration_preserving_repair import repaired_result
from tests.test_modern_hmc_pipeline import setup


@pytest.mark.parametrize("family", ["hmc", "mams"])
@pytest.mark.parametrize("transport", [False, True])
def test_above_one_proposals_use_both_public_authorities_and_replay(tmp_path, family, transport):
    binding, template = setup(family=family, transport=transport)
    search = replace(template, primary_l_grid=(2,), epsilon_by_l=((2, (1.1, 1.2)),))
    run, members = run_frozen_trajectory_search(binding, search, output_dir=tmp_path/"search")
    assert len(members) == 2, run.result.candidate_states
    assert set(members) == set(run.result.verified_candidate_ids)
    report = json.loads((tmp_path/"search/result.json").read_text())
    assert report["proposal_count"] == report["candidate_count"] == 2
    assert report["uncreated_primary_pairs"] == []
    assert report["posterior_assessment"] == "not_run"
    for i, member in enumerate(members.values()):
        continued = member.run(num_results=4, seed=(413, 500+i), output_dir=tmp_path/f"member-{i}")
        direct = binding._run(member.candidate, member.initial_active_state, 4, (413, 500+i))
        np.testing.assert_array_equal(continued["samples"], direct.samples)


def test_actual_acceptance_repair_child_is_exported_and_parent_preserved(tmp_path):
    base, template = setup()
    policy = replace(base.config.acceptance_policy, base_repetitions=32,
                     max_repetitions=32, qualification_region=(.5, 1.))
    binding = make_binding(config=replace(base.config, acceptance_policy=policy),
        max_repairs_per_family=1, repair_factor=2.)
    search = replace(template, primary_l_grid=(3,), epsilon_by_l=((3, (1.95,)),),
        replicated_acceptance_policy=policy, total_budget_units=16, repair_reserve_units=6,
        repair_trajectory_policy="preserve_mean_duration", repair_mean_duration_max_steps=16)
    run, members = run_frozen_trajectory_search(binding, search, output_dir=tmp_path/"search")
    assert len(run.result.repair_actions) == 1, run.result.candidate_states
    root, child = run.result.candidates
    assert child.leapfrog_steps != root.leapfrog_steps
    assert child.parent_candidate_id == root.candidate_id
    assert child.candidate_id in members, run.result.candidate_states
    assert root.candidate_id not in members
    summary = json.loads((tmp_path/"search/result.json").read_text())
    assert summary["proposal_count"] == 1 and summary["candidate_count"] == 2
    assert summary["repair_descendant_count"] == 1
    assert {row["candidate_id"] for row in summary["rows"]} == {root.candidate_id, child.candidate_id}
    assert summary["rows"][1]["primary_candidate_id"] == root.candidate_id
    member = members[child.candidate_id]
    replay = member.run(num_results=4, seed=(413, 699), output_dir=tmp_path/"replay")
    direct = binding._run(child, member.initial_active_state, 4, (413, 699))
    np.testing.assert_array_equal(replay["samples"], direct.samples)


def test_paused_inventory_keeps_uncreated_roots_and_actual_descendants():
    result = repaired_result()
    inventory = candidate_inventory(result, [(9, .2), (3, 1.2)])
    assert inventory["uncreated_primary_pairs"] == [(3, 1.2)]
    assert inventory["created_primary_count"] == 1
    assert inventory["repair_descendant_count"] == 1
    assert inventory["rows"][1]["status"] == "posterior_not_started"
    # Numerical equality with another requested root must not erase the child.
    child = result.candidates[1]
    inventory = candidate_inventory(result, [(9, .2), (child.leapfrog_steps, child.epsilon)])
    assert len(inventory["rows"]) == 2
    assert inventory["uncreated_primary_pairs"] == [(child.leapfrog_steps, child.epsilon)]


def test_actual_budget_stop_preserves_cohort_without_false_exports(tmp_path, monkeypatch):
    from bayesfilter.runtime.execution_budget import execution_budget
    binding, search = setup()
    monkeypatch.setattr(binding, "_run", lambda *a, **k: pytest.fail("work after resource stop"))
    with execution_budget(check=lambda: False):
        run, members = run_frozen_trajectory_search(binding, search, output_dir=tmp_path/"search")
    assert run.result.completion_status == "partial_budget"
    assert not members
    summary = json.loads((tmp_path/"search/result.json").read_text())
    assert summary["proposal_count"] == 2
    assert summary["candidate_count"] + len(summary["uncreated_primary_pairs"]) == 2
    assert summary["verified_candidate_ids"] == []
    assert all(row["status"] == "not_verified" for row in summary["rows"])


def test_fixed_hmc_uses_same_frozen_consumer(tmp_path):
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    base, search = setup()
    binding = make_binding(config=replace(base.config,
        transition_profile=HMCTransitionProfile("hmc", "fixed")))
    run, members = run_frozen_trajectory_search(binding, search, output_dir=tmp_path/"search")
    assert len(members) == 2, run.result.candidate_states
    assert all(member.candidate.transition_spec.profile.trajectory == "fixed"
               for member in members.values())


@pytest.mark.parametrize("defect", ["missing_parent", "family", "cycle", "unknown_verified", "extra_root"])
def test_corrupt_inventory_is_rejected(defect):
    result = repaired_result()
    root, child = result.candidates
    pairs = [(9, .2)]
    if defect == "missing_parent":
        child = replace(child, parent_candidate_id="absent")
    elif defect == "family":
        child = replace(child, candidate_family_id="different")
    elif defect == "cycle":
        child = replace(child, parent_candidate_id=child.candidate_id)
    elif defect == "unknown_verified":
        result = replace(result, verified_candidate_ids=("absent",))
    elif defect == "extra_root":
        pairs = [(3, .2)]
    result = replace(result, candidates=(root, child))
    with pytest.raises(ValueError):
        candidate_inventory(result, pairs)


@pytest.mark.parametrize("defect", ["epsilon_domain", "policy", "repair_cap", "legacy_pilot"])
def test_invalid_configuration_fails_before_output_or_numerical_work(tmp_path, monkeypatch, defect):
    binding, search = setup()
    if defect == "epsilon_domain":
        search = replace(search, primary_l_grid=(3,), epsilon_by_l=((3, (2.,)),))
    elif defect == "policy":
        search = replace(search, replicated_acceptance_policy=replace(
            search.replicated_acceptance_policy, temporal_tolerance=.2))
    elif defect == "repair_cap":
        binding = make_binding(config=binding.config, max_repairs_per_family=1)
    else:
        search = replace(search, pilot_enabled=True)
    monkeypatch.setattr(binding, "_run", lambda *a, **k: pytest.fail("numerical work before preflight"))
    with pytest.raises(ValueError):
        run_frozen_trajectory_search(binding, search, output_dir=tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()


def test_existing_qualification_is_not_overwritten(tmp_path, monkeypatch):
    binding, search = setup()
    marker = tmp_path/"search-design.json"
    marker.write_text("previous attempt")
    monkeypatch.setattr(binding, "_run", lambda *a, **k: pytest.fail("old output reused"))
    with pytest.raises(FileExistsError):
        run_frozen_trajectory_search(binding, search, output_dir=tmp_path)
    assert marker.read_text() == "previous attempt"


def test_missing_export_cannot_silently_shrink_verified_set(tmp_path, monkeypatch):
    from bayesfilter.inference import hmc_candidate_set_retained as retained
    binding, search = setup()
    monkeypatch.setattr(retained, "export_hmc_candidate_retained_runners", lambda **kwargs: {})
    with pytest.raises(ValueError, match="export omitted"):
        run_frozen_trajectory_search(binding, search, output_dir=tmp_path/"search")
    payload = json.loads((tmp_path/"search/tuning/candidate_set_result.json").read_text())
    assert len(payload["verified_candidate_ids"]) == 2
