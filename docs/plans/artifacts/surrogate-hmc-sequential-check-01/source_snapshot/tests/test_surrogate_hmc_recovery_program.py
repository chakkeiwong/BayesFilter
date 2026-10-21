"""Master orchestration tests: phases refresh and never bypass qualification."""
import json
from pathlib import Path
import pytest
from bayesfilter.inference import surrogate_hmc_recovery_program as master

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
    done=master.phase_decision("m13-retained","COMPLETED",dict(posterior_qualified=True),available)
    assert done["pilot_qualified"] and not done["all_model_complete"]

@pytest.mark.parametrize("qualified",[True,False])
def test_executable_master_refreshes_and_runs_retained_only_after_verification(tmp_path,monkeypatch,qualified):
    monkeypatch.setattr(master,"prior_recovery_charges",lambda *a,**k:(dict(cpu_worker_seconds=0.,gpu_seconds=0.),[]))
    monkeypatch.setattr(master,"prerequisite_check",lambda:dict(mechanics_only=True))
    monkeypatch.setattr(master,"snapshot",lambda output:dict(mechanics_only="fixture"))
    monkeypatch.setattr(master,"check_snapshot",lambda hashes:None)
    launched=[]
    def run(command,directory,ledger,**kwargs):
        directory.mkdir()
        phase=directory.name
        launched.append(phase)
        result=dict(status="mechanics",qualified=qualified)
        path=directory/"report.json"
        if phase=="gpu-smoke":
            path=directory/"qualification/report.json"
        if phase=="m13-retained":
            result["posterior_qualified"]=True
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result))
        receipt=dict(status="COMPLETED",cpu_worker_seconds=1.,gpu_seconds=1.,wall_seconds=1.)
        ledger["jobs"].append(receipt)
        for key in ledger["charged"]:
            ledger["charged"][key]+=1.
        return receipt
    monkeypatch.setattr(master,"run_job",run)
    code=master.execute(tmp_path/"run",research=True,limits=master.LIMITS)
    expected=["tests","gpu-smoke","m13-public-tuning"]+(["m13-retained"] if qualified else [])
    assert launched==expected
    report=json.loads((tmp_path/"run/report.json").read_text())
    assert report["pilot_qualified"]==qualified
    assert report["master_complete"] is False
    assert len(json.loads((tmp_path/"run/phase-history.json").read_text()))==len(expected)
    plan=json.loads((tmp_path/"run/next-phase-plan.json").read_text())
    assert plan["part_of_existing_master"] and plan["terminal"]
    assert (tmp_path/"run/result.md").is_file()
    assert code==(0 if qualified else 1)

def test_check_mode_does_not_invoke_research(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"prerequisite_check",lambda:{})
    monkeypatch.setattr(master,"snapshot",lambda output:{})
    monkeypatch.setattr(master,"check_snapshot",lambda hashes:None)
    phases=[]
    def run(command,directory,ledger,**kwargs):
        phases.append(directory.name)
        directory.mkdir()
        if directory.name=="gpu-smoke":
            (directory/"qualification").mkdir()
            (directory/"qualification/report.json").write_text('{"status":"PASS"}')
        receipt=dict(status="COMPLETED",cpu_worker_seconds=1.,gpu_seconds=1.)
        ledger["jobs"].append(receipt)
        return receipt
    monkeypatch.setattr(master,"run_job",run)
    assert master.execute(tmp_path/"run",research=False)==0
    assert phases==["tests","gpu-smoke"]


def test_new_output_root_cannot_reset_existing_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"ARTIFACTS",tmp_path)
    old=tmp_path/"surrogate-hmc-m13-sequential-recovery-01"
    (old/"m13-public-tuning").mkdir(parents=True)
    (old/"run-manifest.json").write_text(json.dumps(dict(allocation_id=master.ALLOCATION_ID,research=True)))
    (old/"m13-public-tuning/run-manifest.json").write_text("{}")
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
