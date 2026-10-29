"""Read-only settlement mechanics; fixtures establish no sampler validity."""
from copy import deepcopy
import json

import pytest

from scripts import settle_modern_hmc_frozen_run as close


def records():
    design = dict(search=dict(epsilon_by_l=[[1, [1.]], [3, [.5]]]),
        max_repairs_per_family=1, execution=dict(acceptance_policy=dict(max_candidates=4), use_xla=True),
        posterior=dict(sequential=dict(warmup_min_results=2000, retained_max_results=10000),
            assessment=dict(retained_bulk_ess_min=400.), precision=dict(method="lugsail"),
            seed=[4, 10], mcse_sd_ratio_max=.02, reference_mcse_multiplier=3.,
            reference_normalized_tolerance=.1))
    candidates = [dict(candidate_id=f"c{i}", parent_candidate_id=None, creation_ordinal=i+1,
        candidate_family_id=f"f{i}", leapfrog_steps=l, epsilon=e, candidate_record_hash=f"h{i}")
        for i, (l, e) in enumerate([(1, 1.), (3, .5)])]
    controller = dict(candidates=candidates, verified_candidate_ids=["c0"], completion_status="complete")
    report = dict(status="completed", verified_candidate_ids=["c0"], delivered_count=1,
        tuning_membership_unchanged=True, rows=[dict(candidate_id="c0", status="posterior_checks_passed",
            delivered=True, posterior_hard_vetoes=[], reference=dict(passed=True), result_file="result.json"),
        dict(candidate_id="c1", status="not_verified")])
    allocation = dict(available_after_existing_held_local_reserve_seconds=1000., numerical_cap_seconds=900.,
        unallocated_after_numerical_reservation_seconds=100., existing_local_reserve_still_held_seconds=200.,
        prior_balance_carried_separately_seconds=17.)
    supervisor = dict(status="completed", elapsed_seconds=150., max_wall_seconds=900., exit_code=0)
    return design, controller, report, allocation, supervisor


def test_full_and_partial_cohorts_preserve_denominator_and_failures():
    design, controller, report, _, _ = records()
    result = close.inspect_inventory(design, controller, report)
    assert result["created_count"] == 2 and result["verified_count"] == 1
    report["rows"][0].update(delivered=False, status="posterior_incomplete")
    report["delivered_count"] = 0
    result = close.inspect_inventory(design, controller, report)
    assert result["failed_assessment_ids"] == ["c0"]
    assert result["verified_count"] == 1 and result["all_verified_assessed"]
    controller["completion_status"] = "partial_budget"
    controller["candidates"].pop()
    report["rows"].pop()
    report["rows"][0].update(status="posterior_not_started_resource_stopped")
    result = close.inspect_inventory(design, controller, report)
    assert result["uncreated_primary_pairs"] == [[3, .5]]
    assert result["unfinished_verified_ids"] == ["c0"]
    assert not result["tuning_complete"] and not result["all_verified_assessed"]


def test_repair_children_stay_in_inventory():
    design, controller, report, _, _ = records()
    controller["candidates"].append(dict(controller["candidates"][0], candidate_id="c2",
        parent_candidate_id="c0", creation_ordinal=3, epsilon=.5))
    report["rows"].append(dict(candidate_id="c2", status="not_verified"))
    assert close.inspect_inventory(design, controller, report)["repair_descendant_count"] == 1


@pytest.mark.parametrize("change", [
    lambda d, c, r: r["rows"].pop(),
    lambda d, c, r: r["rows"].reverse(),
    lambda d, c, r: c["verified_candidate_ids"].append("missing"),
    lambda d, c, r: c["candidates"][1].update(candidate_id="c0"),
    lambda d, c, r: c["candidates"][1].update(parent_candidate_id="missing"),
    lambda d, c, r: c["candidates"][1].update(parent_candidate_id="c0"),
    lambda d, c, r: c["candidates"][1].update(leapfrog_steps=9),
    lambda d, c, r: r["rows"][0].update(delivered=1),
    lambda d, c, r: r["rows"][0].update(posterior_hard_vetoes=["nonfinite"]),
    lambda d, c, r: r["rows"][0]["reference"].update(passed=False),
    lambda d, c, r: r["rows"][1].update(delivered=True),
    lambda d, c, r: r.update(delivered_count=2),
    lambda d, c, r: d["execution"]["acceptance_policy"].update(max_candidates=1),
])
def test_bad_candidate_reports_rejected(change):
    design, controller, report, _, _ = records()
    change(design, controller, report)
    with pytest.raises(ValueError):
        close.inspect_inventory(design, controller, report)


@pytest.mark.parametrize("spent,balance,shortfall", [(150., 850., 0.), (950., 50., 0.), (1100., 0., 100.)])
def test_charge_once_without_resubtracting_reserves_or_new_grant(spent, balance, shortfall):
    _, _, _, allocation, supervisor = records()
    supervisor.update(elapsed_seconds=spent, child_elapsed_seconds=spent-1)
    result = close.settle_budget(allocation, supervisor)
    assert result["numerical_charged_seconds"] == spent
    assert result["unallocated_worker_seconds"] == balance
    assert result["shortfall_seconds"] == shortfall
    assert result["existing_local_reserve_still_held_seconds"] == 200.
    assert result["prior_balance_carried_separately_seconds"] == 17.
    assert not result["additional_72h_grant_included"]


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf")])
def test_invalid_cost_rejected(bad):
    _, _, _, allocation, supervisor = records()
    supervisor["elapsed_seconds"] = bad
    with pytest.raises(ValueError):
        close.settle_budget(allocation, supervisor)


def test_combined_funding_is_reported_without_crediting_grants_again():
    _, _, _, allocation, supervisor = records()
    allocation["funding_sources"] = [
        dict(allocation_id="settled-old-pool", available_seconds=300.),
        dict(allocation_id="new-grant", available_seconds=700.)]
    result = close.settle_budget(allocation, supervisor)
    assert result["unallocated_worker_seconds"] == 850.
    assert result["numerical_charged_seconds"] == 150.
    assert result["funding_sources"] == allocation["funding_sources"]
    assert "additional_72h_grant_included" not in result


@pytest.mark.parametrize("sources", [
    [], "invalid", [None],
    [dict(allocation_id="", available_seconds=1000.)],
    [dict(allocation_id="a", available_seconds=999.)],
    [dict(allocation_id="a", available_seconds=True)],
    [dict(allocation_id="a", available_seconds=float("nan"))],
    [dict(allocation_id="a", available_seconds=-1.)],
    [dict(allocation_id="a", available_seconds=500.),
     dict(allocation_id="a", available_seconds=500.)],
])
def test_invalid_combined_funding_rejected(sources):
    _, _, _, allocation, supervisor = records()
    allocation["funding_sources"] = sources
    with pytest.raises(ValueError):
        close.settle_budget(allocation, supervisor)


@pytest.mark.parametrize("state", sorted(close.ACTIVE))
def test_active_service_never_settles_even_with_terminal_file(state):
    assert close.readiness(dict(ActiveState=state), records()[-1]) == "waiting"


def test_terminal_readiness_and_missing_supervisor():
    supervisor = records()[-1]
    assert close.readiness(dict(ActiveState="inactive", ExecMainStatus="0"), supervisor) == "ready"
    with pytest.raises(ValueError):
        close.readiness(dict(ActiveState="failed", ExecMainStatus="1"), supervisor)
    with pytest.raises(ValueError):
        close.readiness(dict(ActiveState="inactive"), None)
    supervisor.update(status="supervisor_budget_exhausted", exit_code=-9)
    assert close.readiness(dict(ActiveState="failed", ExecMainStatus="247"), supervisor) == "ready"


def dump(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True, allow_nan=False))


def sealed(body):
    return dict(body, content_hash=close.payload_hash(body))


def file_fixture(tmp_path):
    phase = tmp_path/"phase"
    run = phase/"run"
    design, controller, report, allocation, supervisor = records()
    design_path = tmp_path/"design.json"
    dump(design_path, design)
    source = tmp_path/"frozen.py"
    source.write_text("# Numerical-source fixture; never executed.\n")
    spec = dict(example="fixture")
    binding = close.payload_hash(spec)
    command = ["python", str(source), "--design", str(design_path), "--output", str(run),
               "--max-wall-seconds", "900"]
    launch = dict(command=command, numerical_cap_seconds=900, source_commit="commit",
        expected_binding_hash=binding, design_sha256=close.digest(design_path), design_file=str(design_path),
        expected_data_sha256="data", expected_geometry_sha256="geometry",
        environment=dict(CUDA_VISIBLE_DEVICES="1"))
    worker_command = command[:2]+["--worker"]+command[2:-1]+["900.0"]
    supervisor["command"] = worker_command
    manifest = dict(status="completed", command=worker_command, max_wall_seconds=900., git_commit="commit",
        binding_hash=binding, design=design, design_sha256=close.digest(design_path),
        started_utc="2026-10-15T00:00:00+00:00", finished_utc="2026-10-15T00:02:30+00:00",
        target=dict(data_sha256="data", geometry_sha256="geometry"),
        memory_policy=dict(all_physical_devices_memory_growth=True, configured_before_logical_device_initialization=True),
        use_xla=True, tf32_enabled=True, cpu_reference=False, cuda_visible_devices="1",
        source_and_input_sha256={str(source):close.digest(source), str(design_path):close.digest(design_path)},
        reference=dict(quantities=["x"]))
    directory = run/"posterior/member-000"
    policy = dict(design["posterior"]["sequential"], step_size=1., num_leapfrog_steps=1,
        jit_compile=True, warmup_seed=[4,10], retained_seed=[4,11], assessment_policy=dict(
            retained_bulk_ess_min=400., precision=dict(method="lugsail", targets=[
                dict(name="x", kind="mean", mcse_sd_ratio_max=.02)])))
    identity = dict(member_hash="member", policy=policy, quantities=["x"])
    inputs = dict(config=policy, binding_hash=binding, candidate_record_hash="h0")
    chunk = directory/"chunks/committed/chunk-000000"
    attempt = directory/"chunks/attempt-000000"
    dump(attempt/"call-start.json", dict(inputs=inputs))
    chunk.mkdir(parents=True)
    tensor = chunk/"values.tensor"
    tensor.write_bytes(b"fixture tensor bytes")
    bundle = dict(identity_hash=close.payload_hash(identity), inputs_hash=close.payload_hash(inputs),
        metadata=dict(attempt=str(attempt)), tree=dict(tensor=dict(file=tensor.name, sha256=close.digest(tensor))))
    dump(chunk/"bundle.json", bundle)
    (chunk/"bundle.sha256").write_text(close.digest(chunk/"bundle.json"))
    dump(directory/"policy.json", policy)
    dump(directory/"chunks/identity.json", identity)
    dump(directory/"result.json", dict(passed=True, config=policy, warmup_excluded_from_posterior=True,
        hard_vetoes=[], warmup_results_per_chain=2000, retained_results_per_chain=1000))
    report["rows"][0].update(result_file=str(directory/"result.json"), reference=dict(passed=True,
        precision_policy=policy["assessment_policy"]["precision"], normalized_error_screen=[.05],
        normalized_tolerance=.1, mcse_multiplier=3.))
    member_bundle = sealed(dict(fixture=True))
    dump(run/"search/members/bundle.json", member_bundle)
    dump(run/"search/members/c0-member.json", sealed(dict(candidate_id="c0", member_hash="member",
        evidence_bundle=dict(path="bundle.json", content_hash=member_bundle["content_hash"]))))
    dump(run/"search/tuning/tuning_checkpoint.json", sealed(dict(binding_hash=binding,
        result=controller, partial_chunks={}, numerical_evidence_hashes=[])))
    dump(run/"search/tuning/execution_spec.json", dict(binding_hash=binding, execution=spec))
    dump(run/"posterior/result.json", report)
    dump(run/"result.json", dict(status="completed", posterior=report))
    dump(run/"manifest.json", manifest)
    dump(run/"supervisor.json", supervisor)
    dump(phase/"allocation.json", allocation)
    dump(phase/"launch.json", launch)
    return phase


def test_full_artifact_audit_is_read_only(tmp_path):
    phase = file_fixture(tmp_path)
    before = {str(p):close.digest(p) for p in phase.rglob("*") if p.is_file()}
    result, preserved = close.audit(phase)
    assert result["continuation_route"] == "next_frozen_comparator"
    assert result["disposition"]["verified_count"] == 1
    assert result["checked_tensor_files"] == 1
    assert result["numerical_charged_seconds"] == 150.
    assert all(close.digest(p) == h for p,h in {**before, **preserved}.items())


@pytest.mark.parametrize("file,key,value", [
    ("run/manifest.json", "git_commit", "changed"),
    ("run/manifest.json", "tf32_enabled", False),
    ("run/manifest.json", "status", "qualifying"),
    ("run/supervisor.json", "max_wall_seconds", 901.),
    ("run/posterior/member-000/result.json", "warmup_excluded_from_posterior", False),
    ("run/posterior/member-000/policy.json", "retained_seed", [4,13]),
])
def test_file_audit_rejects_changed_identity_policy_or_decision(tmp_path, file, key, value):
    phase = file_fixture(tmp_path)
    path = phase/file
    body = close.read(path)
    body[key] = value
    dump(path, body)
    with pytest.raises(ValueError):
        close.audit(phase)


def test_corrupt_saved_tensor_rejected(tmp_path):
    phase = file_fixture(tmp_path)
    (phase/"run/posterior/member-000/chunks/committed/chunk-000000/values.tensor").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="tensor checksum"):
        close.audit(phase)


def test_default_count_budget_omission_matches_frozen_serializer():
    design, controller, _, _, _ = records()
    design["posterior"]["sequential"].update(max_results_per_chain=10000, count_budget_reason=None)
    policy = dict(warmup_min_results=2000, retained_max_results=10000,
        step_size=1., num_leapfrog_steps=1, jit_compile=True, warmup_seed=[4,10],
        retained_seed=[4,11], assessment_policy=dict(retained_bulk_ess_min=400.,
            precision=dict(method="lugsail", targets=[dict(name="x", kind="mean", mcse_sd_ratio_max=.02)])))
    close.check_policy(policy, design, controller["candidates"][0], 0, ["x"])
    policy["max_results_per_chain"] = 20000
    with pytest.raises(ValueError, match="policy"):
        close.check_policy(policy, design, controller["candidates"][0], 0, ["x"])
    policy.pop("max_results_per_chain")
    design["posterior"]["sequential"].update(max_results_per_chain=20000, count_budget_reason="declared")
    with pytest.raises(ValueError, match="policy"):
        close.check_policy(policy, design, controller["candidates"][0], 0, ["x"])


def test_partial_qualification_without_exports_can_settle_but_cannot_skip_recovery(tmp_path):
    phase = file_fixture(tmp_path)
    # Represent a stop after public qualification but before posterior execution.
    import shutil
    shutil.rmtree(phase/"run/posterior")
    shutil.rmtree(phase/"run/search/members")
    checkpoint_path = phase/"run/search/tuning/tuning_checkpoint.json"
    checkpoint = close.read(checkpoint_path)
    checkpoint.pop("content_hash")
    checkpoint["result"]["completion_status"] = "partial_budget"
    dump(checkpoint_path, sealed(checkpoint))
    for name in ("manifest.json", "supervisor.json"):
        path = phase/"run"/name
        body = close.read(path)
        body.update(status="supervisor_budget_exhausted")
        if name == "supervisor.json":
            body.update(exit_code=-9, elapsed_seconds=901.)
        dump(path,body)
    dump(phase/"run/result.json", dict(status="supervisor_budget_exhausted", delivered_count=0,
        verified_candidate_ids=["c0"], rows=[dict(candidate_id=f"c{i}", status="not_assessed") for i in range(2)]))
    result, _ = close.audit(phase)
    assert result["continuation_route"] == "qualification_recovery_requires_validation"
    assert result["disposition"]["unfinished_verified_ids"] == ["c0"]
    assert result["unallocated_worker_seconds"] == 99.


def test_waiter_writes_one_settlement_and_cannot_restart_same_destination(tmp_path, monkeypatch):
    phase = file_fixture(tmp_path)
    monkeypatch.setattr(close, "service_state", lambda _:dict(ActiveState="inactive", ExecMainStatus="0"))
    monkeypatch.setattr(close.sys, "argv", ["settle", "--phase", str(phase), "--service", "fixture",
                                          "--max-wait-seconds", "1"])
    close.main()
    assert close.read(phase/"terminal-audit-01/observer.json")["status"] == "closed_out"
    assert close.read(phase/"terminal-audit-01/settlement.json")["numerical_charged_seconds"] == 150.
    with pytest.raises(FileExistsError):
        close.main()


def test_waiter_uses_current_allocation_plan(tmp_path, monkeypatch):
    phase = file_fixture(tmp_path)
    allocation = close.read(phase/"allocation.json")
    allocation["phase_plan"] = "docs/plans/current-phase.md"
    dump(phase/"allocation.json", allocation)
    monkeypatch.setattr(close, "service_state", lambda _:dict(ActiveState="inactive", ExecMainStatus="0"))
    monkeypatch.setattr(close.sys, "argv", ["settle", "--phase", str(phase), "--service", "fixture",
                                          "--max-wait-seconds", "1"])
    close.main()
    for name in ("observer.json", "settlement.json"):
        assert close.read(phase/"terminal-audit-01"/name)["plan"] == allocation["phase_plan"]
