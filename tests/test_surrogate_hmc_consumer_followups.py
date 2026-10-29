"""CPU-only master-continuation orchestration checks."""
import json
from pathlib import Path
from types import SimpleNamespace
from scripts import run_surrogate_hmc_consumer_followups as follow

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

def campaign(tmp_path):
    root=tmp_path/"campaign"
    write(root/"final-scope-advance-01/advance.json", {"status":"REPAIR_OR_BUDGET_REQUIRED"})
    write(root/"budget.json", {"attempts":[]})
    return root

def test_waits_for_terminal_parent_and_running_reservation(tmp_path):
    root=campaign(tmp_path)
    assert follow.ready(root)
    write(root/"budget.json", {"attempts":[{"status":"RUNNING"}]})
    assert not follow.ready(root)
    write(root/"budget.json", {"attempts":[]})
    write(root/"final-scope-advance-01/advance.json", {"status":"IN_PROGRESS"})
    assert not follow.ready(root)

def test_command_uses_fresh_scope_and_no_repeated_cost_localization(tmp_path):
    root=campaign(tmp_path)
    scope={"contraction":"matrix","runtime_precision":"fp64_recursion_fp32_pairs","tuning_seed_offset":40}
    assert follow.comparison_command(root,tmp_path/"out","sir_d18",scope,321) is None
    write(root/"final-scope-advance-01/sir_d18/filter-tuning/sir_d18/filter-selection.json",{"selected":{"index":1}})
    command=follow.comparison_command(root,tmp_path/"out","sir_d18",scope,321)
    assert "--consumer-paired-only" in command
    for option,value in (("--shared-gaussian-contraction","matrix"),
            ("--runtime-precision","fp64_recursion_fp32_pairs"),("--tuning-seed-offset","40"),
            ("--per-model-timeout","321")):
        assert command[command.index(option)+1]==value
    assert command[command.index("--selection")+1].startswith(str(root/"final-scope-advance-01"))

def test_failed_comparison_continues_independent_model(tmp_path,monkeypatch):
    root=campaign(tmp_path); out=tmp_path/"out"
    monkeypatch.setattr(follow.master,"MODELS",("lgssm","ksc"))
    scopes={name:{"contraction":"direct","runtime_precision":"uniform","tuning_seed_offset":40} for name in follow.master.MODELS}
    monkeypatch.setattr(follow.master,"execution_scopes",lambda _:scopes)
    monkeypatch.setattr(follow.master,"campaign_remaining",lambda _:({"attempts":[],"max_attempts":10},1000))
    monkeypatch.setattr(follow,"snapshot",lambda _:None)
    for name in follow.master.MODELS:
        write(root/f"final-scope-advance-01/{name}/filter-tuning/{name}/filter-selection.json",{"selected":{"index":0}})
    calls=[]
    def run(command,**kwargs):
        name=command[command.index("--model")+1]; calls.append(name)
        write(out/name/name/"consumer-paired.json",{"status":follow.PASS if name=="ksc" else "VETO"})
        return SimpleNamespace(returncode=0 if name=="ksc" else 1)
    monkeypatch.setattr(follow.subprocess,"run",run)
    assert follow.execute(root,out)==1
    assert calls==["lgssm","ksc"]
    state=follow.read(out/"next-phase-plan.json")
    assert state["status"]=="LOCAL_REPAIR_REQUIRED"
    assert state["models"][1]["status"]==follow.PASS

def test_exhausted_budget_does_not_launch_gpu_worker(tmp_path,monkeypatch):
    root=campaign(tmp_path)
    monkeypatch.setattr(follow.master,"execution_scopes",lambda _: {})
    monkeypatch.setattr(follow.master,"campaign_remaining",lambda _:({"attempts":[],"max_attempts":10},0))
    monkeypatch.setattr(follow.subprocess,"run",lambda *a,**k:(_ for _ in ()).throw(AssertionError("worker launched")))
    assert follow.execute(root,tmp_path/"out")==1
    assert follow.read(tmp_path/"out/next-phase-plan.json")["status"]=="BUDGET_OR_ATTEMPT_LIMIT_REACHED"


def test_explicit_continuation_never_uses_stale_selection(tmp_path, monkeypatch):
    root=campaign(tmp_path); advance=root/"corrected-advance"; plan=root/"corrected-scope.json"
    out=tmp_path/"out"
    scope={"contraction":"direct","runtime_precision":"uniform","tuning_seed_offset":60}
    stale=root/"final-scope-advance-01/lgssm/filter-tuning/lgssm/filter-selection.json"
    write(stale,{"selected":{"index":0}})
    assert not follow.ready(root,advance)
    assert follow.comparison_command(root,out,"lgssm",scope,100,advance) is None
    write(advance/"advance.json",{"status":"REPAIR_OR_BUDGET_REQUIRED"})
    selected=advance/"lgssm/filter-tuning/lgssm/filter-selection.json"
    write(selected,{"selected":{"index":1}})
    assert follow.ready(root,advance)
    write(root/"budget.json",{"attempts":[{"status":"RUNNING"}]})
    assert not follow.ready(root,advance)
    write(root/"budget.json",{"attempts":[]})
    monkeypatch.setattr(follow.master,"MODELS",("lgssm",))
    def scopes(path):
        assert path==plan
        return {"lgssm":scope}
    monkeypatch.setattr(follow.master,"execution_scopes",scopes)
    monkeypatch.setattr(follow.master,"campaign_remaining",lambda _:({"attempts":[],"max_attempts":10},1000))
    monkeypatch.setattr(follow,"snapshot",lambda _:None)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        assert command[command.index("--selection")+1]==str(selected)
        assert command[command.index("--tuning-seed-offset")+1]=="60"
        write(out/"lgssm/lgssm/consumer-paired.json",{"status":follow.PASS})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(follow.subprocess,"run",run)
    assert follow.execute(root,out,advance_dir=advance,execution_plan=plan)==0
    state=follow.read(out/"next-phase-plan.json")
    assert len(calls)==1
    assert state["source_advance_dir"]==str(advance)
    assert state["execution_plan"]==str(plan)
    assert state["status"]=="PAIRED_CONSUMER_CHECKS_COMPLETE"
