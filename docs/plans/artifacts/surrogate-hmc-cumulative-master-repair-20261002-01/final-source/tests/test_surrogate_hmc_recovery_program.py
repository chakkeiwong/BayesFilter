"""Master orchestration tests: phases refresh and never bypass qualification."""
import json
from pathlib import Path
import pytest
from bayesfilter.inference import surrogate_hmc_recovery_program as master

def gpu_pass_report():
    return dict(jit_compile=True, memory_policy=dict(all_physical_devices_memory_growth=True),
        arms={name:dict(passed=True,checks={"health":True},parity_max=0.,
                       artifact_and_continuation_checks=True,endpoint_flag_archive_wiring="PASS")
              for name in ("exact","imperfect_rotational")})


def test_budget_reserves_retention_and_stops_before_overspending():
    device,wall,cpu=master.phase_budget("m13-public-tuning",dict(cpu_worker_seconds=60000.,gpu_seconds=60000.))
    assert device=="gpu" and wall==cpu==47980.
    with pytest.raises(ValueError):
        master.phase_budget("m13-public-tuning",dict(cpu_worker_seconds=12020.,gpu_seconds=60000.))

def test_decisions_distinguish_failed_worker_candidate_and_posterior():
    available=dict(cpu_worker_seconds=20000.,gpu_seconds=20000.)
    assert master.phase_decision("m13-public-tuning","COMPLETED",dict(qualified=True),available)["automatic"]
    fail=master.phase_decision("m13-public-tuning","COMPLETED",dict(qualified=False,status="WARMUP_MAXIMUM_INCONCLUSIVE"),available)
    assert fail["terminal"] and not fail["research_direction_rejected"]
    worker=master.phase_decision("m13-public-tuning","OWNER_LOST",{},available)
    assert worker["phase"]=="repair_execution" and not worker["candidate_rejected"]
    done=master.phase_decision("m13-retained","COMPLETED",dict(sampling_qualified=True, posterior_reference_qualified=False),available)
    assert done["sampling_qualified"] and not done["pilot_qualified"] and not done["all_model_complete"]

@pytest.mark.parametrize("qualified",[True,False])
def test_executable_master_refreshes_and_assesses_cumulative_samples(tmp_path,monkeypatch,qualified):
    monkeypatch.setattr(master,"prior_recovery_charges",lambda *a,**k:(dict(cpu_worker_seconds=0.,gpu_seconds=0.),[]))
    monkeypatch.setattr(master,"prerequisite_check",lambda allocation="legacy":dict(mechanics_only=True))
    monkeypatch.setattr(master,"snapshot",lambda output:dict(mechanics_only="fixture"))
    monkeypatch.setattr(master,"check_snapshot",lambda hashes:None)
    launched=[]
    def run(command,directory,ledger,**kwargs):
        directory.mkdir()
        phase=directory.name
        launched.append(phase)
        result=dict(status="PASS" if phase in ("gpu-smoke","m13-preflight") else "mechanics",qualified=qualified)
        path=directory/"report.json"
        if phase=="gpu-smoke":
            result=gpu_pass_report()
            path=directory/"qualification/report.json"
        if phase=="m13-retained":
            result.update(sampling_qualified=True, posterior_reference_qualified=False)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result))
        receipt=dict(status="COMPLETED",cpu_worker_seconds=1.,gpu_seconds=1.,wall_seconds=1.)
        ledger["jobs"].append(receipt)
        for key in ledger["charged"]:
            ledger["charged"][key]+=1.
        return receipt
    monkeypatch.setattr(master,"run_job",run)
    code=master.execute(tmp_path/"run",research=True,limits=master.LIMITS)
    expected=["tests","gpu-smoke","m13-preflight","m13-public-tuning"]+(["m13-retained"] if qualified else [])
    assert launched==expected
    report=json.loads((tmp_path/"run/report.json").read_text())
    assert report["sampling_qualified"]==qualified
    assert not report["pilot_qualified"]
    assert report["master_complete"] is False
    assert report["active_phase"] is None and report["last_phase"] == expected[-1]
    assert len(json.loads((tmp_path/"run/phase-history.json").read_text()))==len(expected)
    plan=json.loads((tmp_path/"run/next-phase-plan.json").read_text())
    assert plan["part_of_existing_master"] and plan["terminal"]
    assert plan["independent_validation_required"] is False
    assert (tmp_path/"run/result.md").is_file()
    assert code==(0 if qualified else 1)

def test_check_mode_does_not_invoke_research(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"prerequisite_check",lambda allocation="legacy":{})
    monkeypatch.setattr(master,"snapshot",lambda output:{})
    monkeypatch.setattr(master,"check_snapshot",lambda hashes:None)
    phases=[]
    def run(command,directory,ledger,**kwargs):
        phases.append(directory.name)
        directory.mkdir()
        if directory.name=="gpu-smoke":
            (directory/"qualification").mkdir()
            (directory/"qualification/report.json").write_text(json.dumps(gpu_pass_report()))
        if directory.name=="m13-preflight":
            (directory/"report.json").write_text('{"status":"PASS"}')
        receipt=dict(status="COMPLETED",cpu_worker_seconds=1.,gpu_seconds=1.)
        ledger["jobs"].append(receipt)
        return receipt
    monkeypatch.setattr(master,"run_job",run)
    assert master.execute(tmp_path/"run",research=False)==0
    assert phases==["tests","gpu-smoke","m13-preflight"]


def test_new_output_root_cannot_reset_existing_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"ARTIFACTS",tmp_path)
    old=tmp_path/"surrogate-hmc-m13-sequential-recovery-01"
    (old/"m13-public-tuning").mkdir(parents=True)
    (old/"run-manifest.json").write_text(json.dumps(dict(allocation_id=master.ALLOCATION_ID,research=True)))
    (old/"m13-public-tuning/run-manifest.json").write_text("{}")
    (old/"m13-public-tuning/guard-receipt.json").write_text('{"accounting_complete":true}')
    with pytest.raises(ValueError,match="one scientific attempt"):
        master.prior_recovery_charges(tmp_path/"surrogate-hmc-m13-sequential-recovery-02",research=True)

def test_prior_check_charges_are_carried_into_scientific_allowance(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"ARTIFACTS",tmp_path)
    old=tmp_path/"surrogate-hmc-sequential-check-01"
    old.mkdir()
    (old/"run-manifest.json").write_text(json.dumps(dict(allocation_id=master.ALLOCATION_ID,research=False)))
    (old/"accounting.json").write_text(json.dumps(dict(charged_this_run=dict(cpu_worker_seconds=11.,gpu_seconds=7.))))
    charged,_=master.prior_recovery_charges(tmp_path/"new",research=True)
    assert charged==dict(cpu_worker_seconds=11.,gpu_seconds=7.)


@pytest.mark.parametrize("receipt_present",[True,False])
def test_crash_receipt_overrides_stale_parent_summary(tmp_path,monkeypatch,receipt_present):
    monkeypatch.setattr(master,"ARTIFACTS",tmp_path)
    old=tmp_path/"surrogate-hmc-sequential-check-crashed"
    worker=old/"gpu-smoke"
    worker.mkdir(parents=True)
    (old/"run-manifest.json").write_text(json.dumps(dict(
        allocation_id=master.ALLOCATION_ID,research=False,status="RUNNING")))
    (old/"accounting.json").write_text(json.dumps(dict(
        charged_this_run=dict(cpu_worker_seconds=0.,gpu_seconds=0.))))
    (worker/"run-manifest.json").write_text("{}")
    if receipt_present:
        (worker/"guard-receipt.json").write_text(json.dumps(dict(
            accounting_complete=True,cpu_worker_seconds=17.,gpu_seconds=11.)))
        charged,_=master.prior_recovery_charges(tmp_path/"new",research=True)
        assert charged==dict(cpu_worker_seconds=17.,gpu_seconds=11.)
    else:
        with pytest.raises(ValueError,match="unresolved recovery worker accounting"):
            master.prior_recovery_charges(tmp_path/"new",research=True)


@pytest.mark.parametrize("started", [False, True])
def test_preparation_failure_retry_carries_all_costs(tmp_path,monkeypatch,started):
    monkeypatch.setattr(master,"ARTIFACTS",tmp_path)
    old=tmp_path/"surrogate-hmc-m13-sequential-recovery-preflight"
    worker=old/"m13-public-tuning"
    worker.mkdir(parents=True)
    (old/"run-manifest.json").write_text(json.dumps(dict(allocation_id=master.ALLOCATION_ID,research=True)))
    (worker/"run-manifest.json").write_text("{}")
    (worker/"guard-receipt.json").write_text(json.dumps(dict(
        accounting_complete=True,cpu_worker_seconds=13.,gpu_seconds=7.)))
    (worker/"sampling-state.json").write_text(json.dumps(dict(
        status="SAMPLING_STARTED" if started else "PREPARING",inference_launched=started)))
    if started:
        with pytest.raises(ValueError,match="one scientific attempt"):
            master.prior_recovery_charges(tmp_path/"new",research=True)
    else:
        charged,_=master.prior_recovery_charges(tmp_path/"new",research=True)
        assert charged==dict(cpu_worker_seconds=13.,gpu_seconds=7.)
        (worker/"tuning").mkdir()
        assert master.scientific_attempt_started(worker)


def test_reviewed_legacy_failure_requires_intact_evidence(tmp_path):
    worker=tmp_path/"m13-public-tuning"
    worker.mkdir()
    (worker/"run-manifest.json").write_text("{}")
    (worker/"guard-receipt.json").write_text('{"accounting_complete":true}')
    (worker/"command.log").write_text("selected_components\nValueError: wrong or stale materiality selection")
    source="../source_snapshot/bayesfilter/testing/surrogate_hmc_m13_public_tuning.py"
    (worker/source).parent.mkdir(parents=True)
    (worker/source).write_text("# preserved source")
    proof=dict(classification="SELECTION_LOADING_BEFORE_PUBLIC_TUNER",inference_launched=False,
        evidence={name:master.file_hash(worker/name) for name in ("command.log","guard-receipt.json",source)})
    (worker/"pre-sampling-review.json").write_text(json.dumps(proof))
    assert not master.scientific_attempt_started(worker)
    (worker/"command.log").write_text("changed")
    with pytest.raises(ValueError,match="invalid pre-sampling"):
        master.scientific_attempt_started(worker)


def test_failed_preflight_cannot_launch_research():
    decision=master.phase_decision("m13-preflight","COMPLETED",dict(status="FAIL"),master.LIMITS)
    assert decision["terminal"] and decision["phase"]=="repair_execution"


def test_saved_real_gpu_qualification_schema_advances():
    path=master.ARTIFACTS/"surrogate-hmc-m13-sequential-recovery-02/gpu-smoke/qualification/report.json"
    result=json.loads(path.read_text())
    assert "status" not in result
    assert master.gpu_qualification_passed(result)
    assert master.phase_decision("gpu-smoke","COMPLETED",result,master.LIMITS)["phase"]=="m13-preflight"


@pytest.mark.parametrize("damage",["missing_arm","failed","incomplete","check","parity","jit"])
def test_gpu_qualification_rejects_missing_or_failed_evidence(damage):
    result=gpu_pass_report()
    if damage=="missing_arm":
        del result["arms"]["imperfect_rotational"]
    elif damage=="jit":
        result["jit_compile"]=False
    else:
        arm=result["arms"]["imperfect_rotational"]
        if damage=="failed":arm["passed"]=False
        elif damage=="incomplete":del arm["artifact_and_continuation_checks"]
        elif damage=="check":arm["checks"]["health"]=False
        elif damage=="parity":arm["parity_max"]=float("nan")
    assert not master.gpu_qualification_passed(result)
    decision=master.phase_decision("gpu-smoke","COMPLETED",result,master.LIMITS)
    assert decision["phase"]=="repair_execution" and not decision["candidate_rejected"]
