"""Scientific decision separation and execution prerequisites."""
import json

import pytest

from bayesfilter.inference.surrogate_hmc_closure_program import (
    ARCHIVES, archived_judgments, coverage_register, jobs, readiness, scope_judgment,
)


def test_practical_acceptance_preserves_discrepant_finite_target_and_inconclusive_sampling():
    verdicts = {"B/R": "INCONCLUSIVE", "D/R": "DISCREPANCY"}
    row = scope_judgment(scope="LG1", comparisons=verdicts, health="PASSED", accepted=True)
    assert row["practical_judgment"] == "ACCEPTED_PILOT"
    assert row["formal_comparisons"] == verdicts
    assert row["next_action"] == "CLOSED_NO_RERUN"
    verdicts["B/R"] = "EQUIVALENT"
    assert row["formal_comparisons"]["B/R"] == "INCONCLUSIVE"


@pytest.mark.parametrize("health,expected", [("PASSED", "VIABLE_PILOT_REQUIRES_SCOPE_JUDGMENT"),
    ("FAILED", "NUMERICAL_FAILURE"), ("NOT_CHECKED", "NOT_ASSESSED")])
def test_health_is_not_formal_equivalence(health, expected):
    row = scope_judgment(scope="KSC", comparisons={"B/R": "INCONCLUSIVE"}, health=health)
    assert row["practical_judgment"] == expected
    assert row["formal_comparisons"]["B/R"] == "INCONCLUSIVE"


def test_acceptance_cannot_hide_missing_or_bad_evidence():
    for scope, health, valid in (("LG1", "FAILED", True), ("LG1", "NOT_CHECKED", True),
                                 ("LG1", "PASSED", False), ("KSC", "PASSED", True)):
        with pytest.raises(ValueError):
            scope_judgment(scope=scope, comparisons={}, health=health, artifact_valid=valid, accepted=True)


def test_ready_requires_all_prerequisites_and_never_reopens_lg1():
    missing = readiness("AR2", {})["missing"]
    complete = dict.fromkeys(missing, True)
    assert readiness("AR2", complete)["ready"]
    for key in missing:
        assert not readiness("AR2", complete | {key: False})["ready"]
    assert readiness("LG1", complete)["status"] == "CLOSED_NO_RERUN"


def test_archived_pilots_and_catalog_remain_complete():
    rows = archived_judgments()
    assert set(rows) == set(ARCHIVES)
    assert all(row["sampling_health"] == "PASSED" for row in rows.values())
    assert rows["LG1"]["practical_judgment"] == "ACCEPTED_PILOT"
    assert rows["LG1"]["formal_comparisons"]["D/R"] == "DISCREPANCY"
    assert rows["KSC"]["formal_comparisons"]["D/R"] == "EQUIVALENT"
    assert len(coverage_register()) == 17
    assert all(not row["inference_launched"] for row in rows.values())


def test_independent_repairs_do_not_depend_on_structural_success():
    stage = {row["name"]: row for row in jobs()}
    assert stage["uncertainty"]["dependencies"] == ["review"]
    assert stage["wrapped"]["dependencies"] == ["review"]
    assert all("LG1" not in json.dumps(row["command"]) for row in stage.values())


def test_selected_confirmation_does_not_repeat_old_calibration_or_skip_review():
    assert [row["name"] for row in jobs("uncertainty-confirmation")] == ["review", "uncertainty-confirmation"]
    with pytest.raises(ValueError):
        jobs("uncertainty,uncertainty-confirmation")
    with pytest.raises(ValueError):
        jobs("invented-stage")
    from bayesfilter.testing.surrogate_hmc_closure_diagnostics import uncertainty_protocol
    initial, fresh = uncertainty_protocol(), uncertainty_protocol(True)
    old_seeds = {initial["seed_base"]+10000*g+i for g, row in enumerate(initial["groups"]) for i in range(row[3])}
    new_seeds = {fresh["seed_base"]+10000*g+i for g, row in enumerate(fresh["groups"]) for i in range(row[3])}
    assert not old_seeds.intersection(new_seeds)
    assert all(initial[key] == fresh[key] for key in initial if key not in ("groups", "seed_base"))


def test_existing_master_dispatches_closure(monkeypatch, tmp_path):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    from bayesfilter.inference import surrogate_hmc_program as master
    seen = []
    monkeypatch.setattr(closure, "main", lambda args: seen.extend(args) or 0)
    assert master.main(["--stage", "closure", "--output", str(tmp_path/"new"), "--dry-run"]) == 0
    assert seen == ["--output", str(tmp_path/"new"), "--dry-run"]


def test_retry_budget_counts_failed_attempts_and_rejects_missing_accounting(monkeypatch, tmp_path):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    previous = tmp_path/"surrogate-hmc-closure-20260922-01"
    previous.mkdir()
    with pytest.raises(ValueError, match="lacks accounting"):
        closure.preceding_charges(tmp_path/"surrogate-hmc-closure-20260922-02")
    charge = dict(cpu_worker_seconds=120., gpu_seconds=80.)
    (previous/"accounting.json").write_text(json.dumps(dict(charged_this_run=charge)))
    total, records = closure.preceding_charges(tmp_path/"surrogate-hmc-closure-20260922-02")
    assert len(records) == 1
    assert total == {key: closure.BASE_CHARGE[key]+charge[key] for key in charge}
    assert closure.remaining_pass_reservation(records) == {key: closure.PASS_CAP[key]-charge[key] for key in charge}


def test_interleaved_campaign_work_is_charged_outside_closure_reservation(monkeypatch, tmp_path):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    closure_charge = dict(cpu_worker_seconds=120., gpu_seconds=80.)
    original_charge = {key: value+321. for key, value in closure.BASE_CHARGE.items()}
    for name, charge in (("surrogate-hmc-all-models-interleaved", original_charge),
                         ("surrogate-hmc-closure-20260922-01", closure_charge)):
        path = tmp_path/name/"accounting.json"
        path.parent.mkdir()
        path.write_text(json.dumps(dict(charged_this_run=charge)))
    total, records = closure.preceding_charges(tmp_path/"surrogate-hmc-closure-20260922-02")
    assert total == {key: original_charge[key]+closure_charge[key] for key in total}
    assert closure.remaining_pass_reservation(records) == {key: closure.PASS_CAP[key]-closure_charge[key] for key in total}


@pytest.mark.parametrize("failed_stage", ["review", "structural"])
def test_execution_failure_blocks_only_dependent_stages(monkeypatch, tmp_path, failed_stage):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(closure, "snapshot_sources", lambda output: {})
    launched = []
    def fake_run(command, directory, ledger, **options):
        name = directory.name
        launched.append(name)
        directory.mkdir(parents=True)
        record = dict(status="FAILED" if name == failed_stage else "COMPLETED",
                      cpu_worker_seconds=.01, gpu_seconds=0.)
        ledger["jobs"].append(record)
        closure.write_json(directory/"report.json", dict(status="DIAGNOSTIC_RESULT", qualified=False))
        return record
    monkeypatch.setattr(closure, "run_job", fake_run)
    monkeypatch.setattr(closure, "archived_judgments", lambda: {})
    output = tmp_path/"surrogate-hmc-closure-20260922-01"
    assert closure.main(["--output", str(output)]) == 1
    report = json.loads((output/"report.json").read_text())
    assert not report["all_models_validated"] and not report["inference_launched"]
    if failed_stage == "review":
        assert launched == ["review"]
        assert report["stages"]["uncertainty"]["execution"] == "BLOCKED_DEPENDENCY"
    else:
        assert "uncertainty" in launched
    assert (output/"accounting.json").is_file()


def test_setup_failure_is_preserved_and_charged(monkeypatch, tmp_path):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    def broken_snapshot(output):
        raise OSError("fixture: source unavailable")
    monkeypatch.setattr(closure, "snapshot_sources", broken_snapshot)
    output = tmp_path/"surrogate-hmc-closure-20260922-01"
    assert closure.main(["--output", str(output)]) == 1
    assert "source unavailable" in json.loads((output/"report.json").read_text())["error"]
    assert json.loads((output/"accounting.json").read_text())["charged_this_run"]["cpu_worker_seconds"] >= 0


def test_review_repair_preserves_scientific_attempt_limit(monkeypatch, tmp_path):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(closure, "snapshot_sources", lambda output: {})
    monkeypatch.setattr(closure, "archived_judgments", lambda: {})
    for attempt in range(1, 4):
        root = tmp_path/f"surrogate-hmc-closure-20260922-{attempt:02d}"
        root.mkdir()
        closure.write_json(root/"accounting.json", dict(charged_this_run=dict(cpu_worker_seconds=1., gpu_seconds=0.)))
        for stage in ("review", "structural"):
            (root/stage).mkdir()
            closure.write_json(root/stage/"run-manifest.json", {})
    launched = []
    def fake_run(command, directory, ledger, **options):
        launched.append(directory.name)
        directory.mkdir()
        record = dict(status="COMPLETED", cpu_worker_seconds=.01, gpu_seconds=0.)
        ledger["jobs"].append(record)
        return record
    monkeypatch.setattr(closure, "run_job", fake_run)
    output = tmp_path/"surrogate-hmc-closure-20260922-04"
    assert closure.main(["--output", str(output), "--stages", "structural"]) == 1
    assert launched == ["review"]
    report = json.loads((output/"report.json").read_text())
    assert report["stages"]["structural"]["execution"] == "ATTEMPT_LIMIT"
    accounting = json.loads((output/"accounting.json").read_text())
    assert accounting["previous_charge"]["cpu_worker_seconds"] == closure.BASE_CHARGE["cpu_worker_seconds"]+3.
