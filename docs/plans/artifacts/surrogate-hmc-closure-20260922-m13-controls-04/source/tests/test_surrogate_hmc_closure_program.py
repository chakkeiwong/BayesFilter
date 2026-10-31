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


def test_structural_pass_uses_separate_reservation_and_runs_actual_integration():
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    records = [dict(path=f"/tmp/surrogate-hmc-closure-20260922-{name}/accounting.json",
                    charged_this_run=dict(cpu_worker_seconds=cpu, gpu_seconds=gpu))
               for name, cpu, gpu in (("01", 100., 200.), ("structural-01", 30., 40.))]
    assert closure.remaining_pass_reservation(records)["cpu_worker_seconds"] == closure.PASS_CAP["cpu_worker_seconds"]-100.
    assert closure.remaining_pass_reservation(records, "structural")["cpu_worker_seconds"] == closure.PASS_CAP["cpu_worker_seconds"]-30.
    selected = jobs("structural-integration")
    assert [row["name"] for row in selected] == ["review", "structural-integration"]
    assert "tests/test_ledh_structural_innovation.py" in selected[0]["command"]
    with pytest.raises(ValueError, match="separate bounded reservation"):
        jobs("structural-integration,uncertainty")


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


def test_ar2_reference_has_focused_checks_and_preserves_refinement_boundary():
    selected = jobs("ar2-reference")
    assert [row["name"] for row in selected] == ["review", "ar2-reference"]
    assert "tests/test_surrogate_hmc_ar2_reference.py" in selected[0]["command"]
    assert selected[1]["device"] == "gpu"
    assert [row["name"] for row in jobs("ar2-reference-extended")] == ["review", "ar2-reference-extended"]
    for selection in ("ar2-reference,ar2-reference-extended", "ar2-reference,structural-integration"):
        with pytest.raises(ValueError, match="separate bounded refinement"):
            jobs(selection)
    assert [row["name"] for row in jobs("ar2-controls")] == ["review", "ar2-controls"]
    with pytest.raises(ValueError, match="separate inspected reference"):
        jobs("ar2-controls,structural-integration")


def test_each_subpass_charges_only_its_own_jobs():
    from bayesfilter.inference.surrogate_hmc_closure_program import remaining_pass_reservation, PASS_CAPS
    records = [dict(path=f"/tmp/surrogate-hmc-closure-20260922-{suffix}/accounting.json",
                    charged_this_run=dict(cpu_worker_seconds=cpu, gpu_seconds=gpu))
               for suffix, cpu, gpu in (("01", 100., 200.), ("structural-01", 300., 400.), ("ar2-validation-01", 500., 600.),
                                        ("ar2-inference-01", 700., 800.))]
    for kind, cpu, gpu in (("initial", 100., 200.), ("structural", 300., 400.), ("ar2-validation", 500., 600.),
                           ("ar2-inference", 700., 800.)):
        assert remaining_pass_reservation(records, kind) == dict(cpu_worker_seconds=PASS_CAPS[kind]["cpu_worker_seconds"]-cpu,
            gpu_seconds=PASS_CAPS[kind]["gpu_seconds"]-gpu)


def test_inference_jobs_have_separate_review_and_bounded_reservations():
    from bayesfilter.inference.surrogate_hmc_program import reserve_job, validated_budget
    from bayesfilter.inference.surrogate_hmc_closure_program import PASS_CAPS
    for arm in ("a", "b"):
        selected = jobs("ar2-inference-"+arm)
        assert [row["name"] for row in selected] == ["review", "ar2-inference-"+arm]
        assert "tests/test_surrogate_hmc_ar2_inference.py" in selected[0]["command"]
        cap = PASS_CAPS["ar2-inference"]
        ledger = validated_budget(cap["gpu_seconds"], cap["cpu_worker_seconds"])
        ledger["charged"]["cpu_worker_seconds"] = selected[0]["cpu"]
        job = selected[1]
        reserve_job(ledger, job["device"], job["timeout"], job["cpu"],
                    max_timeout_seconds=job.get("max_timeout_seconds", 4*3600))
    with pytest.raises(ValueError, match="separate focused prerequisite"):
        jobs("ar2-inference-a,ar2-inference-b")


@pytest.mark.parametrize("launched", [False, True])
def test_inference_master_uses_explicit_stage_and_preserves_launch_status(monkeypatch, tmp_path, launched):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(closure, "snapshot_sources", lambda output: {})
    monkeypatch.setattr(closure, "archived_judgments", lambda: {})
    seen = []
    def run(command, directory, ledger, **options):
        directory.mkdir()
        seen.append((command, options))
        record = dict(status="COMPLETED", cpu_worker_seconds=.01, gpu_seconds=0.)
        ledger["jobs"].append(record)
        closure.write_json(directory/"report.json", dict(status="FIXTURE", inference_launched=launched))
        return record
    monkeypatch.setattr(closure, "run_job", run)
    output = tmp_path/"surrogate-hmc-closure-20260922-ar2-inference-01"
    with pytest.raises(SystemExit):
        closure.main(["--output", str(output), "--stages", "review", "--dry-run"])
    with pytest.raises(SystemExit):
        closure.main(["--output", str(tmp_path/"surrogate-hmc-closure-20260922-01"),
                      "--stages", "ar2-inference-b", "--dry-run"])
    assert closure.main(["--output", str(output), "--stages", "ar2-inference-b"]) == 0
    result = json.loads((output/"report.json").read_text())
    manifest = json.loads((output/"run-manifest.json").read_text())
    assert manifest["pass_kind"] == "ar2-inference"
    assert result["inference_launched"] is launched and not result["all_models_validated"]
    assert seen[1][0][seen[1][0].index("--stage")+1] == "ar2-inference-b"
    assert seen[1][1]["max_timeout_seconds"] == 22500.


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


def test_finite_target_preflight_requires_separate_stage_and_focused_review():
    selected = jobs("ar2-finite-preflight")
    assert [row["name"] for row in selected] == ["review", "ar2-finite-preflight"]
    assert "tests/test_surrogate_hmc_ar2_finite.py" in selected[0]["command"]
    with pytest.raises(ValueError, match="finite-target"):
        jobs("ar2-finite-preflight,wrapped")


def test_finite_target_pass_deducts_prior_attempts_without_reusing_inference_budget():
    from bayesfilter.inference.surrogate_hmc_closure_program import remaining_pass_reservation
    rows = [
        dict(path="/tmp/surrogate-hmc-closure-20260922-ar2-finite-01/accounting.json",
             charged_this_run=dict(cpu_worker_seconds=600., gpu_seconds=800.)),
        dict(path="/tmp/surrogate-hmc-closure-20260922-ar2-inference-01/accounting.json",
             charged_this_run=dict(cpu_worker_seconds=2000., gpu_seconds=2200.))]
    assert remaining_pass_reservation(rows, "ar2-finite") == dict(cpu_worker_seconds=5400., gpu_seconds=6700.)


def test_m13_repair_is_one_additional_attempt_without_resetting_campaign_budget(monkeypatch, tmp_path):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(closure, "snapshot_sources", lambda output: {})
    monkeypatch.setattr(closure, "archived_judgments", lambda: {})
    for number, stage in enumerate(closure.M13_CONSUMER_STAGES, 1):
        root = tmp_path/f"surrogate-hmc-closure-20260922-m13-consumer-{number:02d}"
        (root/stage).mkdir(parents=True)
        closure.write_json(root/"accounting.json",
                           dict(charged_this_run=dict(cpu_worker_seconds=1., gpu_seconds=2.)))
        closure.write_json(root/stage/"run-manifest.json", {})
    launched = []
    def fake_run(command, directory, ledger, **options):
        launched.append(directory.name)
        directory.mkdir()
        record = dict(status="COMPLETED", cpu_worker_seconds=.01, gpu_seconds=0.)
        closure.write_json(directory/"run-manifest.json", record)
        ledger["jobs"].append(record)
        if directory.name != "review":
            closure.write_json(directory/"report.json", dict(qualified=False))
        return record
    monkeypatch.setattr(closure, "run_job", fake_run)
    output = tmp_path/"surrogate-hmc-closure-20260922-m13-repair-01"
    assert closure.main(["--output", str(output), "--stages", closure.M13_CONSUMER_REPAIR]) == 0
    assert launched == ["review", closure.M13_CONSUMER_REPAIR]
    accounting = json.loads((output/"accounting.json").read_text())
    assert accounting["previous_charge"]["cpu_worker_seconds"] == closure.BASE_CHARGE["cpu_worker_seconds"]+3.
    manifest = json.loads((output/"run-manifest.json").read_text())
    assert manifest["reservation"] == dict(cpu_worker_seconds=1260., gpu_seconds=1200.)

    # The repair name cannot restart either the repair counter or the original counter.
    for number, stage in ((5, closure.M13_CONSUMER_REPAIR), (6, "m13-consumer-parity")):
        launched.clear()
        kind = "m13-repair" if stage == closure.M13_CONSUMER_REPAIR else "m13-consumer"
        blocked = tmp_path/f"surrogate-hmc-closure-20260922-{kind}-{number:02d}"
        assert closure.main(["--output", str(blocked), "--stages", stage]) == 1
        assert launched == ["review"]
        report = json.loads((blocked/"report.json").read_text())
        assert report["stages"][stage]["execution"] == "ATTEMPT_LIMIT"
