"""Supervisor mechanics only: fake child/clock cannot provide numerical evidence."""
from copy import deepcopy
import json
import subprocess

import pytest

from scripts import run_hmc_v7_confirmation as campaign


def design():
    return dict(schema=campaign.SCHEMA, source_manifest_sha256="a"*64, gpu_uuid="GPU-fixture",
        budget_seconds=2000, search_cap_seconds=10, closeout_cap_seconds=10,
        settlement_reserve_seconds=100, prices={f:"/prices/"+f for f in campaign.FAMILIES},
        readiness=dict(wait_cap_seconds=2, poll_seconds=1, minimum_free_mib=4096,
                       provenance="fake readiness fixture, not a runtime default"),
        provenance="mechanics fixture; no campaign authorization or numerical evidence",
        slots=[dict(slot_id=f"s-{i}-{f}", family=f, seed=[20261004,50000+3*i+j])
               for i in range(32) for j,f in enumerate(campaign.FAMILIES)])


class Clock:
    value = 0.
    def now(self): return self.value


def test_all_planned_slots_execute_and_child_flags_cannot_replace_identity():
    d=design(); clock=Clock(); calls=[]; saved=[]
    def attempt(slot,cap):
        calls.append((slot,cap));clock.value+=2
        return dict(exit_code=0,seed=[0,0],slot_id="wrong",source_manifest_sha256="wrong")
    result=campaign.execute_slots(d,attempt,lambda rows:saved.append(deepcopy(rows)),now=clock.now)
    assert len(calls)==len(result["outcomes"])==result["original_denominator"]==96
    assert len(saved[-1])==96
    assert all(cap==22 for _,cap in calls)
    for expected,actual in zip(d["slots"],result["outcomes"]):
        assert all(actual[k]==v for k,v in expected.items())
        assert actual["source_manifest_sha256"]==d["source_manifest_sha256"]
    assert result["wall_seconds"]==192
    assert not result["release_ready"] and not result["default_promoted"]


def test_budget_exhaustion_keeps_every_unstarted_slot_on_original_denominator():
    d=design();d["budget_seconds"]=104;clock=Clock();caps=[]
    def attempt(slot,cap):
        caps.append(cap);clock.value+=2;return dict(exit_code=2)
    result=campaign.execute_slots(d,attempt,lambda rows:None,now=clock.now)
    assert caps==[4,2]
    assert len(result["outcomes"])==96
    assert all(r["disposition"]=="budget_deferred" and r["exit_code"]==3 for r in result["outcomes"][2:])


@pytest.mark.parametrize("failure",[1,-9,True,None,"exception"])
def test_harness_failure_stops_execution_without_erasing_later_slots(failure):
    calls=[]
    def attempt(slot,cap):
        calls.append(slot)
        if failure=="exception":raise RuntimeError("injected bad harness")
        return dict(exit_code=failure)
    result=campaign.execute_slots(design(),attempt,lambda rows:None,now=lambda:0.)
    assert len(calls)==1 and result["status"]=="harness_failure"
    assert result["outcomes"][0]["exit_code"]==1
    assert len(result["outcomes"])==96
    assert all(r["disposition"]=="unstarted_after_harness_failure" for r in result["outcomes"][1:])


def test_timeout_and_resource_failures_remain_outcomes_and_do_not_shrink_inventory():
    calls=[]
    def attempt(slot,cap):
        calls.append(slot)
        if len(calls)==1:raise subprocess.TimeoutExpired("fixture",cap)
        return dict(exit_code=3)
    result=campaign.execute_slots(design(),attempt,lambda rows:None,now=lambda:0.)
    assert len(calls)==96 and result["outcomes"][0]["exit_code"]==124
    assert all(r["exit_code"]==3 for r in result["outcomes"][1:])


@pytest.mark.parametrize("damage",["duplicate_seed","duplicate_slot","development_seed","missing_slot","missing_family","nonfinite_budget","changed_source","relative_price","readiness_budget","readiness_poll","readiness_memory"])
def test_invalid_design_fails_before_any_child(damage):
    d=design()
    if damage=="duplicate_seed":d["slots"][1]["seed"]=d["slots"][0]["seed"]
    if damage=="duplicate_slot":d["slots"][1]["slot_id"]=d["slots"][0]["slot_id"]
    if damage=="development_seed":d["slots"][0]["seed"]=[20261002,2501]
    if damage=="missing_slot":d["slots"].pop()
    if damage=="missing_family":d["prices"].pop(campaign.FAMILIES[0])
    if damage=="nonfinite_budget":d["budget_seconds"]=float("nan")
    if damage=="changed_source":d["source_manifest_sha256"]="missing"
    if damage=="relative_price":d["prices"][campaign.FAMILIES[0]]="relative"
    if damage=="readiness_budget":d["readiness"]["wait_cap_seconds"]=float("inf")
    if damage=="readiness_poll":d["readiness"]["poll_seconds"]=61
    if damage=="readiness_memory":d["readiness"]["minimum_free_mib"]=True
    with pytest.raises(ValueError):
        campaign.execute_slots(d,lambda *a:pytest.fail("must not launch"),lambda r:None)


def test_confirmation_only_changes_seed_and_declared_role_from_priced_profile():
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    slot=design()["slots"][0]
    profile=full_search_configuration(slot["family"],seed=(20261002,2501),wall_seconds=1800,replicated_trial_batch_size=32)
    cfg=campaign.slot_configuration(profile,slot,1800)
    for key in set(profile)-{"seed","case_id","classification","provenance"}:
        assert cfg[key]==profile[key]
    assert cfg["classification"]=="confirmation" and cfg["seed"]==slot["seed"]
    with pytest.raises(ValueError):campaign.slot_configuration(profile,slot,1801)
    with pytest.raises(ValueError):campaign.slot_configuration(profile,{**slot,"family":"nonlinear"},1800)


def test_source_mismatch_prevents_output_creation_and_worker_launch(tmp_path,monkeypatch):
    import os,sys
    from scripts import run_hmc_v7_release_prices as prices
    d=design();path=tmp_path/"design.json";path.write_text(json.dumps(d))
    monkeypatch.setattr(sys,"argv",["confirm","--source",str(tmp_path/"source"),"--output",str(tmp_path/"run"),"--design",str(path)])
    monkeypatch.setattr(sys,"path",list(sys.path))
    monkeypatch.setattr(os,"environ",dict(os.environ))
    monkeypatch.setattr(os,"sched_setaffinity",lambda *args:None)
    monkeypatch.setattr(prices,"check_source",lambda source:"b"*64)
    with pytest.raises(ValueError,match="source differ"):
        campaign.main()
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("damage", [None, "profile", "source", "result_json", "missing_result", "wrong_result_seed", "late_source", "late_result_corruption", "wait_budget", "short_process_cap"])
def test_main_reads_real_price_files_and_checked_profiles_before_fake_children(tmp_path, monkeypatch, damage):
    import hashlib, os, sys
    from pathlib import Path
    from types import SimpleNamespace
    from scripts import run_hmc_v7_release_prices as prices_driver
    from tests.test_hmc_release_confirmation import prices, complete_model
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    d=design();d["search_cap_seconds"]=1800
    if damage=="wait_budget":d["readiness"]["wait_cap_seconds"]=20
    if damage=="short_process_cap":d["budget_seconds"]=100000
    price_files=prices(tmp_path/"prices")
    for index,path in enumerate(price_files):
        family=campaign.FAMILIES[index]
        d["prices"][family]=str(path.parent)
        outer=json.loads(path.read_text())
        outer.update(source_manifest_sha256=d["source_manifest_sha256"],gpu_uuid=d["gpu_uuid"])
        if damage=="short_process_cap" and index==0:
            outer["wall_seconds"]=1900;outer["attempts"][0]["wall_seconds"]=1899
        path.write_text(json.dumps(outer))
        cfg=full_search_configuration(family,seed=(20261002,2501+index),wall_seconds=1800,replicated_trial_batch_size=32)
        config_path=path.parent/(family+"-config.json");config_path.write_text(json.dumps(cfg))
        (path.parent/family/"model/configuration.json").write_text(json.dumps(cfg))
        manifest=dict(configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      source_manifest_sha256=d["source_manifest_sha256"])
        if damage=="source" and index==0:manifest["source_manifest_sha256"]="b"*64
        (path.parent/family/"manifest.json").write_text(json.dumps(manifest))
        if damage=="profile" and index==0:
            cfg["active_starts"][0][0]+=.1;config_path.write_text(json.dumps(cfg))
    design_path=tmp_path/"design.json";design_path.write_text(json.dumps(d))
    root=tmp_path/"run"
    monkeypatch.setattr(sys,"argv",["confirm","--source",str(tmp_path/"source"),"--output",str(root),"--design",str(design_path)])
    monkeypatch.setattr(sys,"path",list(sys.path));monkeypatch.setattr(os,"environ",dict(os.environ))
    monkeypatch.setattr(os,"sched_setaffinity",lambda *args:None)
    source_checks=[]
    def checked_source(source):
        source_checks.append(source)
        if damage=="late_source" and len(source_checks)>1:raise ValueError("injected source change")
        return d["source_manifest_sha256"]
    monkeypatch.setattr(prices_driver,"check_source",checked_source)
    launched=[]
    def fake_child(command, **kwargs):
        config=json.loads(Path(command[command.index("--config")+1]).read_text())
        family=config["case_id"].removeprefix("confirmation-")
        launched.append(config)
        directory=Path(command[command.index("--output")+1])/"model";directory.mkdir()
        result=complete_model(family,config["seed"])
        if damage=="wrong_result_seed" and len(launched)==1:result["sampling_streams"]=[0,0]
        if damage!="missing_result" or len(launched)!=1:
            (directory/"result.json").write_text("broken" if damage=="result_json" and len(launched)==1 else json.dumps(result))
        if damage=="late_result_corruption" and len(launched)==96:
            (root/d["slots"][0]["slot_id"]/"model/result.json").write_text("broken after child")
        assert kwargs["timeout"] <= d["search_cap_seconds"]+d["closeout_cap_seconds"]+d["readiness"]["wait_cap_seconds"]
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,"run",fake_child)
    if damage in {"profile", "source"}:
        with pytest.raises(ValueError,match="profile or source changed"):campaign.main()
        assert not root.exists() and not launched
    elif damage in {"wait_budget", "short_process_cap"}:
        message="fund readiness" if damage=="wait_budget" else "process cap"
        with pytest.raises(ValueError,match=message):campaign.main()
        assert not root.exists() and not launched
    elif damage is None:
        assert campaign.main()==0 and len(launched)==96
        result=json.loads((root/"result.json").read_text())
        assert result["delivery"]["original_denominator"]==96
        assert result["delivery"]["delivery_criterion_passed"]
        assert not result["release_ready"]
    else:
        assert campaign.main()==1
        result=json.loads((root/"result.json").read_text())
        assert result["status"]=="harness_failure" and "delivery" not in result
        assert result["original_denominator"]==96 and len(result["outcomes"])==96
        assert len(json.loads((root/"outcomes.json").read_text()))==96
        assert not result["release_ready"]
        assert len(launched)==(96 if damage in {"late_source", "late_result_corruption"} else 1)


def test_memory_contention_recovers_before_sampling_with_all_waits_recorded():
    clock=Clock();saved=[];probe_caps=[];inventories=iter(["GPU-fixture, 512, 90", "GPU-fixture, 8000, 70"])
    def probe(cap):probe_caps.append(cap);return next(inventories)
    def sleep(seconds):clock.value+=seconds
    result=campaign.wait_for_memory("GPU-fixture",design()["readiness"],probe,
        lambda row:saved.append(deepcopy(row)),now=clock.now,sleep=sleep)
    assert result["ready"] and result["wall_seconds"]==1 and not result["gpu_initialized"]
    assert [r["ready"] for r in result["observations"]]==[False,True]
    assert probe_caps==[2,1] and len(saved)==2


def test_persistent_memory_contention_exhausts_only_the_declared_wait():
    clock=Clock();sleeps=[]
    def sleep(seconds):sleeps.append(seconds);clock.value+=seconds
    result=campaign.wait_for_memory("GPU-fixture",design()["readiness"],lambda cap:"GPU-fixture, 1, 99",
        lambda row:None,now=clock.now,sleep=sleep)
    assert not result["ready"] and result["wall_seconds"]==2
    assert result["disposition"]=="resource_deferred" and sleeps==[1,1]


@pytest.mark.parametrize("error",["missing_gpu","malformed_memory","probe_exception"])
def test_readiness_probe_errors_remain_harness_failures(error):
    saved=[]
    def probe(cap):
        if error=="probe_exception":raise RuntimeError("probe failed")
        return "GPU-other, 9000, 0" if error=="missing_gpu" else "GPU-fixture, unknown, 0"
    with pytest.raises((ValueError,RuntimeError)):
        campaign.wait_for_memory("GPU-fixture",design()["readiness"],probe,saved.append,now=lambda:0.)
    assert saved[-1]["disposition"]=="harness_failure" and not saved[-1]["gpu_initialized"]


def test_readiness_probe_cannot_return_success_after_wait_deadline():
    clock=Clock()
    def probe(cap):clock.value+=3;return "GPU-fixture, 9000, 0"
    result=campaign.wait_for_memory("GPU-fixture",design()["readiness"],probe,lambda row:None,now=clock.now)
    assert not result["ready"] and result["disposition"]=="resource_deferred"


def test_ample_memory_compute_contention_recovers_within_original_wait():
    clock=Clock();saved=[];caps=[]
    policy={**design()["readiness"],"require_no_foreign_compute":True}
    processes=iter(["GPU-fixture, 41", "GPU-other, 42"])
    def process_probe(cap):caps.append(cap);return next(processes)
    result=campaign.wait_for_memory("GPU-fixture",policy,lambda cap:"GPU-fixture, 8000, 99",
        lambda row:saved.append(deepcopy(row)),process_probe=process_probe,
        now=clock.now,sleep=lambda seconds:setattr(clock,"value",clock.value+seconds))
    assert result["ready"] and result["wall_seconds"]==1 and not result["gpu_initialized"]
    assert [o["foreign_pids"] for o in result["observations"]]==[[41],[]]
    assert caps==[2,1] and len(saved)==2


def test_compute_contention_exhaustion_is_charged_resource_deferral():
    clock=Clock();policy={**design()["readiness"],"require_no_foreign_compute":True}
    result=campaign.wait_for_memory("GPU-fixture",policy,lambda cap:"GPU-fixture, 32000, 50",
        lambda row:None,process_probe=lambda cap:"GPU-fixture, 41",now=clock.now,
        sleep=lambda seconds:setattr(clock,"value",clock.value+seconds))
    assert not result["ready"] and result["wall_seconds"]==2
    assert result["disposition"]=="resource_deferred" and not result["gpu_initialized"]


@pytest.mark.parametrize("inventory",["GPU-fixture, unknown","GPU-fixture, 0",
    "GPU-fixture, 41\nGPU-fixture, 41","missing identity","", "GPU-other, 7"])
def test_process_inventory_validation_fails_closed_or_accepts_clear_device(inventory):
    policy={**design()["readiness"],"require_no_foreign_compute":True};saved=[]
    def check():
        return campaign.wait_for_memory("GPU-fixture",policy,lambda cap:"GPU-fixture, 8000, 0",
            saved.append,process_probe=lambda cap:inventory,now=lambda:0.)
    if inventory in ("", "GPU-other, 7"):
        assert check()["ready"]
    else:
        with pytest.raises(ValueError):check()
        assert saved[-1]["disposition"]=="harness_failure"


def test_compute_probe_cannot_authorize_sampling_after_deadline():
    policy={**design()["readiness"],"require_no_foreign_compute":True};clock=Clock()
    def processes(cap):clock.value+=3;return ""
    result=campaign.wait_for_memory("GPU-fixture",policy,lambda cap:"GPU-fixture, 8000, 0",
        lambda row:None,process_probe=processes,now=clock.now)
    assert not result["ready"] and result["wall_seconds"]==3


def test_explicit_compute_policy_requires_probe_and_boolean_design_field():
    d=design();d["readiness"]["require_no_foreign_compute"]=True
    campaign.validate_design(d)
    with pytest.raises(ValueError,match="requires a compute-process probe"):
        campaign.wait_for_memory("GPU-fixture",d["readiness"],lambda cap:"GPU-fixture, 8000, 0",lambda row:None)
    for value in (1,"true",None):
        d["readiness"]["require_no_foreign_compute"]=value
        with pytest.raises(ValueError,match="Boolean"):campaign.validate_design(d)


def test_legacy_memory_only_policy_does_not_query_compute_processes():
    result=campaign.wait_for_memory("GPU-fixture",design()["readiness"],
        lambda cap:"GPU-fixture, 8000, 99",lambda row:None,
        process_probe=lambda cap:pytest.fail("legacy policy must not query processes"),now=lambda:0.)
    assert result["ready"]
    assert not result["observations"][0]["process_inventory_checked"]


@pytest.mark.parametrize("clears",[False,True])
def test_worker_checks_process_contention_before_tensorflow_import(tmp_path,monkeypatch,clears):
    import builtins,os,sys
    from types import SimpleNamespace
    from scripts import run_hmc_v7_release_prices as prices
    d=design();d["readiness"]["require_no_foreign_compute"]=True
    config=tmp_path/"config.json";config.write_text(json.dumps({"classification":"confirmation"}))
    frozen=tmp_path/"design.json";frozen.write_text(json.dumps(d))
    root=tmp_path/"output";root.mkdir()
    monkeypatch.setattr(sys,"path",list(sys.path));monkeypatch.setattr(os,"chdir",lambda path:None)
    monkeypatch.setattr(prices,"check_source",lambda source:d["source_manifest_sha256"])
    clock=Clock();original_wait=campaign.wait_for_memory;queries=[];imports=[]
    def bounded_wait(*args,**kwargs):
        return original_wait(*args,**kwargs,now=clock.now,
            sleep=lambda seconds:setattr(clock,"value",clock.value+seconds))
    monkeypatch.setattr(campaign,"wait_for_memory",bounded_wait)
    def query(command,**kwargs):
        assert kwargs["timeout"] in (1,2)
        if "--query-compute-apps=gpu_uuid,pid" in command:
            queries.append(clock.value)
            return "" if clears and clock.value==1 else "GPU-fixture, 41"
        return "GPU-fixture, 32000, 99"
    monkeypatch.setattr(subprocess,"check_output",query)
    original_import=builtins.__import__
    class ImportBoundaryReached(Exception):pass
    def guarded_import(name,*args,**kwargs):
        if name=="tensorflow":
            imports.append(clock.value)
            raise ImportBoundaryReached()
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded_import)
    args=SimpleNamespace(source=tmp_path/"source",output=root,config=config,design=frozen,gpu="GPU-fixture")
    if clears:
        with pytest.raises(ImportBoundaryReached):campaign.worker(args)
        assert imports==[1]
    else:
        assert campaign.worker(args)==3 and not imports
    resource=json.loads((root/"resource.json").read_text())
    assert resource["ready"] is clears and queries==[0,1]
