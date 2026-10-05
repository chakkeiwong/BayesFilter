"""Framework-free checks for the executable K0--K7 campaign plan."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesfilter.testing.inference_validation import ssm_campaign as campaign
from bayesfilter.testing.inference_validation import timeout_policy
from bayesfilter.testing.inference_validation.ssm_campaign import build_suites
from bayesfilter.testing.inference_validation.ssm_campaign_profiles import PROFILES
from bayesfilter.testing.inference_validation.designs import digest
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash


def _datasets():
    return {
        target: [{
            "dataset_id": f"{profile.case}-data-A",
            "data_seed": [20260926, 1],
            "data": ([0.1] * profile.horizon if profile.family != "multivariate"
                      else [[0.1] * 4] * 120),
            "data_sha256": "frozen-test-data",
            "reference_settings": {"resolution": 161, "sensitivity": [0.002, 0.002]},
            "reference_checked": profile.reference != "unavailable",
        }, *([] if profile.case == "K6" else [{
            "dataset_id": f"{profile.case}-data-B",
            "data_seed": [20260926, 2],
            "data": ([0.2] * profile.horizon),
            "data_sha256": "frozen-test-data-b",
            "reference_settings": {"resolution": 161, "sensitivity": [0.002, 0.002]},
            "reference_checked": True,
        }])]
        for target, profile in PROFILES.items()
    }


def test_campaign_has_original_k0_k7_inventory_and_frozen_posterior_policy():
    suites = build_suites(_datasets())
    main = suites["main-unpriced"]
    assert len(main["designs"]) == 32
    assert {d["options"]["campaign_case"] for d in main["designs"]} == set("K0 K1 K2 K3 K4 K5 K6 K7".split())
    assert all(tuple(d["l_grid"]) == (3, 5, 9, 13, 18, 25) for d in main["designs"])
    assert all(d["options"]["posterior_settings"]["warmup_min_results"] == 2000 for d in main["designs"])
    assert all(d["options"]["posterior_settings"]["warmup_check_window_results"] == 1000 for d in main["designs"])
    assert all(d["options"]["posterior_settings"]["retained_max_results"] == 10000 for d in main["designs"])
    assert all(d["options"]["posterior_members"] == "selected" for d in main["designs"])
    assert {d["options"]["posterior_member_count"] for d in main["designs"]
            if d["options"]["campaign_case"] in {"K0", "K7"}} == {2}
    assert {d["options"]["posterior_member_count"] for d in main["designs"]
            if d["options"]["campaign_case"] not in {"K0", "K7"}} == {1}


def test_campaign_has_complete_price_and_preflight_shapes():
    suites = build_suites(_datasets())
    assert len(suites["pricing"]["designs"]) == 8
    assert len(suites["preflight-mechanics"]["designs"]) == 8
    assert len(suites["preflight-pipeline"]["designs"]) == 4
    assert len(suites["stationarity"]["designs"]) == 3
    assert all(d["device"] == "gpu" for d in suites["preflight-pipeline"]["designs"])
    assert all(d["options"]["posterior_members"] == "selected"
               for d in suites["preflight-pipeline"]["designs"])


def _ledger():
    return {"grant_gpu_seconds":180000.,"remaining_gpu_seconds":36000.,
        "allocations":{"canonical_neutra_pricing_reserved_only":1200.},
        "records":[{"receipt":campaign.C1_RECEIPT,"elapsed_seconds":144000.}]}


def test_c1_and_its_reservation_cannot_be_double_spent():
    with pytest.raises(ValueError,match="active reservation"):
        campaign.free_gpu_seconds(_ledger(),c1_active=True)
    ledger = _ledger()
    ledger["active_reservation"] = {"service":"another-service"}
    with pytest.raises(ValueError,match="active reservation"):
        campaign.free_gpu_seconds(ledger,c1_active=False)
    assert campaign.free_gpu_seconds(_ledger(),c1_active=False) == 34800.


@pytest.mark.parametrize("mutation",["missing_c1","duplicate_receipt","bad_balance","nonfinite"])
def test_unsettled_or_invalid_accounting_is_refused(mutation):
    ledger = _ledger()
    if mutation == "missing_c1": ledger["records"][0]["receipt"]="other.json"
    elif mutation == "duplicate_receipt": ledger["records"] *= 2
    elif mutation == "bad_balance": ledger["remaining_gpu_seconds"] += 1
    else: ledger["records"][0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError): campaign.free_gpu_seconds(ledger,c1_active=False)


def _pilot(tmp_path, *, shared_contention=False):
    datasets = _datasets()
    suites = build_suites(datasets,shared_contention=shared_contention)
    index = {"source":{"identity":"checked-source"},"suite_identity":digest(suites["pricing"]),
             "execution_options":{"reuse_leapfrog_graphs":True,"share_unused_budget":False},"jobs":{}}
    for design in suites["pricing"]["designs"]:
        root = tmp_path/design["design_id"]
        result = write_json(root/"result.json",{"execution_status":"complete","assessment":{}})
        # A failed posterior check is a complete workload, not a cheap failed
        # process. Pricing never requires posterior success or chooses seeds.
        write_json(root/"replication-0000/pipeline.json",{
            "completion":"complete","members":[{"status":"assessed","posterior":{"passed":False},
                                                     "recorded_retained_count":1000}]
            * design["options"]["posterior_member_count"]})
        index["jobs"][design["design_id"]] = {"status":"complete","result":str(result),
            "attempts":[{"elapsed_seconds":100.,"exit_code":0,"status":"complete"}]}
    return suites["main-unpriced"], index, datasets


def test_measured_price_preserves_counts_and_all_32_slots(tmp_path):
    template,index,datasets = _pilot(tmp_path)
    priced,costs = campaign.freeze_priced_main(template,index,datasets,available_seconds=4800.)
    assert len(priced["designs"]) == 32
    assert len(priced["required_coverage"]) == 32
    assert sum(d["budget_seconds"] for d in priced["designs"]) == 4800.
    assert all(d["options"]["posterior_settings"] == campaign.COUNTS for d in priced["designs"])
    assert all(p["pilot_seconds"] == 100. for p in costs["prices"].values())
    assert costs["original_denominator"] == 32
    assert costs["runtime_tail_guarantee"] is False


@pytest.mark.parametrize("failure",["timeout","empty_search","missing_member","warmup_only","child_failed","retry","changed_workload"])
def test_failed_or_incomplete_pilot_cannot_supply_a_cheap_price(tmp_path,failure):
    template,index,datasets = _pilot(tmp_path)
    job = index["jobs"]["price-K0"]
    if failure == "timeout": job["attempts"][0]["timed_out"] = True
    if failure == "retry": job["attempts"] *= 2
    if failure == "changed_workload":
        index["suite_identity"] = "different"
        with pytest.raises(ValueError,match="workload"):
            campaign.freeze_priced_main(template,index,datasets,available_seconds=5000.)
        return
    if failure == "child_failed": write_json(job["result"],{"execution_status":"failed"})
    if failure in {"empty_search","missing_member"}:
        write_json(Path(job["result"]).parent/"replication-0000/pipeline.json",{
            "completion":"complete","members":([] if failure=="empty_search" else [{"status":"assessed"}])})
    if failure == "warmup_only":
        write_json(Path(job["result"]).parent/"replication-0000/pipeline.json",{
            "completion":"complete","members":[{"status":"assessed","recorded_retained_count":0}]*2})
    priced,costs = campaign.freeze_priced_main(template,index,datasets,available_seconds=5000.)
    assert len(priced["designs"]) == 28
    assert len(priced["required_coverage"]) == len(costs["dispositions"]) == 32
    assert all(d["options"]["campaign_case"] != "K0" for d in priced["designs"])


def test_unaffordable_or_unchecked_lanes_remain_explicit(tmp_path):
    template,index,datasets = _pilot(tmp_path)
    datasets["ssm_campaign_interior"][0]["reference_checked"] = False
    priced,costs = campaign.freeze_priced_main(template,index,datasets,available_seconds=600.)
    assert len(priced["designs"]) == 4
    assert len(priced["required_coverage"]) == 32
    assert {d["disposition"] for d in costs["dispositions"]} == {
        "funded at measured price","reference unavailable","insufficient remaining grant"}


def test_shared_campaign_changes_only_scheduling_and_prices_the_full_enclosing_work(tmp_path):
    original=build_suites(_datasets())
    shared=build_suites(_datasets(),shared_contention=True)
    for stage in original:
        for old,new in zip(original[stage]['designs'],shared[stage]['designs']):
            assert {k:v for k,v in old.items() if k not in {'budget_seconds','options'}} == {
                k:v for k,v in new.items() if k not in {'budget_seconds','options'}}
            assert {k:v for k,v in old['options'].items() if k!='timeout_policy'} == {
                k:v for k,v in new['options'].items() if k!='timeout_policy'}
    assert sum(d['budget_seconds'] for d in shared['pricing']['designs'])==8*1785.
    assert sum(d['budget_seconds'] for d in shared['preflight-pipeline']['designs'])==4*350.
    template,index,datasets=_pilot(tmp_path,shared_contention=True)
    priced,costs=campaign.freeze_priced_main(template,index,datasets,available_seconds=4800.)
    assert len(priced['designs'])==32
    assert all(d['options']['timeout_policy']['gpu_admission_mode']=='shared' for d in priced['designs'])
    assert all(d['options']['timeout_policy']['max_contention_retries']==1 for d in priced['designs'])


def test_main_pricing_intersects_original_wall_and_cumulative_campaign_caps(tmp_path,monkeypatch):
    template,index,datasets=_pilot(tmp_path)
    write_json(tmp_path/'source.json',index['source'])
    write_json(tmp_path/'pricing/run_index.json',index)
    write_json(tmp_path/'main-unpriced.json',template)
    write_json(tmp_path/'datasets.json',datasets)
    write_json(tmp_path/'checkpoint.json',{'started_epoch':1000.})
    monkeypatch.setattr(campaign,'time',SimpleNamespace(time=lambda:1000.+46*3600.-630.))
    monkeypatch.setattr(campaign,'check_preflight',lambda root:None)
    monkeypatch.setattr(campaign,'c1_is_active',lambda:False)
    costs=campaign.price(tmp_path,_ledger(),allocation_cap_seconds=1200.)
    assert costs['allocation_cap_seconds']==600.
    assert len(read_json(tmp_path/'main.json')['designs'])==4
    costs=campaign.price(tmp_path,_ledger(),allocation_cap_seconds=300.)
    assert costs['allocation_cap_seconds']==300.
    assert not read_json(tmp_path/'main.json')['designs']


def test_preflight_validates_the_final_recovered_child(tmp_path):
    assessment=write_json(tmp_path/'independent_assessment.json',{'members':[]})
    write_json(tmp_path/'process-attempt-001-exit.json',{'status':'budget_exhausted','exit_code':75})
    write_json(tmp_path/'process-attempt-001-manifest.json',{'runtime':{'original':True}})
    write_json(tmp_path/'process-attempt-002-exit.json',{'status':'complete','exit_code':0,
        'assessment_sha256':file_hash(assessment)})
    write_json(tmp_path/'process-attempt-002-manifest.json',{'runtime':{'recovered':True}})
    assert campaign.completed_fit_runtime(tmp_path)=={'recovered':True}
    write_json(assessment,{'changed':True})
    with pytest.raises(ValueError,match='final child or assessment'):
        campaign.completed_fit_runtime(tmp_path)


def test_launch_cannot_start_while_c1_is_active(tmp_path,monkeypatch):
    ledger_path = write_json(tmp_path/"grant.json",_ledger())
    monkeypatch.setattr(campaign,"c1_is_active",lambda:True)
    monkeypatch.setattr(campaign.subprocess,"run",lambda *a,**kw:pytest.fail("worker must not launch"))
    with pytest.raises(ValueError,match="active reservation"):
        campaign.run_stage(tmp_path,"pricing",ledger_path)


def test_failed_stage_still_charges_one_enclosing_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(timeout_policy, "wait_for_gpu_admission", lambda **kw: {"admitted":True,"wait_seconds":0.})
    ledger_path = write_json(tmp_path/"grant.json",_ledger())
    write_json(tmp_path/"preflight-mechanics.json",build_suites(_datasets())["preflight-mechanics"])
    write_json(tmp_path/"source.json",{"files":{},"identity":"test"})
    write_json(tmp_path/"dependencies.json",{})
    write_json(tmp_path/"prepared-inputs.json",{})
    monkeypatch.setattr(campaign,"c1_is_active",lambda:False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-fixture")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH","true")
    launches = []
    def fake_run(command,**kwargs):
        launches.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(campaign.subprocess,"run",fake_run)
    receipt = campaign.run_stage(tmp_path,"preflight-mechanics",ledger_path)
    ledger = read_json(ledger_path)
    assert receipt["exit_code"] == 1
    assert len(ledger["records"]) == 2
    assert "active_reservation" not in ledger
    assert "--property=KillMode=control-group" in launches[0]
    assert ledger["charged_gpu_seconds"] == sum(r["elapsed_seconds"] for r in ledger["records"])
    with pytest.raises(ValueError,match="already attempted"):
        campaign.run_stage(tmp_path,"preflight-mechanics",ledger_path)


def test_unconfirmed_shutdown_preserves_reservation_and_conservative_charge(tmp_path,monkeypatch):
    monkeypatch.setattr(timeout_policy, "wait_for_gpu_admission", lambda **kw: {"admitted":True,"wait_seconds":0.})
    ledger_path = write_json(tmp_path/"grant.json",_ledger())
    write_json(tmp_path/"preflight-mechanics.json",build_suites(_datasets())["preflight-mechanics"])
    for name,payload in (("source.json",{"files":{},"identity":"test"}),
                         ("dependencies.json",{}),("prepared-inputs.json",{})):
        write_json(tmp_path/name,payload)
    monkeypatch.setattr(campaign,"c1_is_active",lambda:False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-fixture")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH","true")
    def interrupted(command,**kwargs):
        if command[0] == "systemd-run": raise KeyboardInterrupt()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(campaign.subprocess,"run",interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_stage(tmp_path,"preflight-mechanics",ledger_path)
    ledger = read_json(ledger_path)
    assert ledger["active_reservation"]
    receipt = ledger["records"][-1]
    assert receipt["shutdown_confirmed"] is False
    assert receipt["elapsed_seconds"] >= 510
    with pytest.raises(ValueError,match="active reservation"):
        campaign.free_gpu_seconds(ledger,c1_active=False)


def test_gpu_change_cannot_reuse_prices_or_preflight(tmp_path, monkeypatch):
    ledger_path = write_json(tmp_path/"grant.json", _ledger())
    write_json(tmp_path/"preflight-mechanics.json", build_suites(_datasets())["preflight-mechanics"])
    write_json(tmp_path/"checkpoint.json", {
        "started_epoch": campaign.time.time(), "gpu_uuid": "GPU-original", "stages": {}})
    monkeypatch.setattr(campaign, "c1_is_active", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-different")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **kw: pytest.fail("worker must not launch"))
    with pytest.raises(ValueError, match="same selected GPU UUID"):
        campaign.run_stage(tmp_path, "preflight-mechanics", ledger_path)
    assert read_json(ledger_path) == _ledger()


def test_failed_stage_stops_cli_with_nonzero_exit_and_preserves_report(tmp_path, monkeypatch):
    calls = []
    def failed_stage(root, stage, ledger):
        calls.append(stage)
        return {"exit_code": 7}
    monkeypatch.setattr(campaign, "run_stage", failed_stage)
    monkeypatch.setattr(campaign, "summarize", lambda root: calls.append("report"))
    with pytest.raises(SystemExit) as outcome:
        campaign.main(["run", str(tmp_path), "--stage", "all"])
    assert outcome.value.code == 1
    assert calls == ["preflight-mechanics", "report"]


def test_busy_stage_is_charged_and_deferred_without_launch(tmp_path, monkeypatch):
    ledger_path = write_json(tmp_path/"grant.json", _ledger())
    write_json(tmp_path/"preflight-mechanics.json", build_suites(_datasets())["preflight-mechanics"])
    for name, payload in (("source.json", {"files":{},"identity":"test"}),
                          ("dependencies.json",{}), ("prepared-inputs.json",{})):
        write_json(tmp_path/name, payload)
    monkeypatch.setattr(campaign, "c1_is_active", lambda:False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setattr(timeout_policy, "wait_for_gpu_admission", lambda **kw: {"admitted":False,"wait_seconds":0.})
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **kw:pytest.fail("busy stage launched"))
    receipt = campaign.run_stage(tmp_path, "preflight-mechanics", ledger_path)
    assert receipt["exit_code"] == 75 and receipt["shutdown_confirmed"]
    ledger = read_json(ledger_path)
    assert "active_reservation" not in ledger and len(ledger["records"]) == 2
    assert read_json(tmp_path/"preflight-mechanics-admission.json")["admitted"] is False
