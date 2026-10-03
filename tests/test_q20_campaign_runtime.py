"""Process deadlines, durable accounting and replay integrity; no GPU work."""
import json
from pathlib import Path
import sys
import time

import pytest

from bayesfilter.inference.q20_campaign_runtime import Campaign, CampaignBudgetError
from bayesfilter.inference.q20_production_config import protocol_template


def campaign(tmp_path, *, cap=10.):
    config = protocol_template()
    config["budget"]["arm_cap_seconds"] = cap
    config["execution"].update(termination_grace_seconds=.1, poll_seconds=.02)
    return Campaign(tmp_path, repo=Path(__file__).resolve().parents[1], config=config,
        allowance={"campaign_remaining_seconds":30.,"diagnostic_remaining_seconds":20.,
                   "unsettled_holds":[{"seconds":2.,"reason":"unsettled previous work"}]})


def test_process_tree_timeout_and_accounting_once(tmp_path):
    c = campaign(tmp_path, cap=.5)
    code = ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
            "print(p.pid,flush=True); time.sleep(20)")
    with c.locked():
        attempt=c.execute("hang",[sys.executable,"-c",code],cap_seconds=.5,diagnostic=True)
        assert attempt["status"]=="timed_out"
        assert .35 <= attempt["elapsed_seconds"] < 2.
        pid=int((Path(attempt["directory"])/"console.log").read_text())
        from bayesfilter.inference.q20_campaign_runtime import _process_start
        for _ in range(20):
            if _process_start(pid) is None:
                break
            time.sleep(.02)
        assert _process_start(pid) is None
        spent=c.state["spent_seconds"]
        c.recover()
        assert c.state["spent_seconds"]==spent==c.state["diagnostic_spent_seconds"]
        assert c.state["campaign_limit"]==28.
        with pytest.raises(CampaignBudgetError,match="cumulative"):
            c.execute("hang",[sys.executable,"-c","pass"],cap_seconds=1.,diagnostic=True)
    resumed=Campaign(tmp_path,repo=c.repo,config=c.config,
        allowance={"campaign_remaining_seconds":99999.,"diagnostic_remaining_seconds":99999.})
    with resumed.locked():
        assert resumed.state["campaign_limit"]==28.
        assert resumed.state["spent_seconds"]==spent


def test_normal_exit_cleans_leftover_descendant(tmp_path):
    c=campaign(tmp_path)
    code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True)"
    with c.locked():
        attempt=c.execute("exit",[sys.executable,"-c",code],cap_seconds=2.,diagnostic=False)
        assert attempt["status"]=="completed"
        from bayesfilter.inference.q20_campaign_runtime import _process_start
        pid=int((Path(attempt["directory"])/"console.log").read_text())
        for _ in range(20):
            if _process_start(pid) is None:
                break
            time.sleep(.02)
        assert _process_start(pid) is None


def test_orphan_upper_charge_and_unresolved_creation(tmp_path):
    c=campaign(tmp_path)
    with c.locked():
        item={"stage":"lost","status":"running","pid":None,"process_start":None,
              "started_epoch":time.time(),"cap_seconds":2.,"diagnostic":True,"directory":str(tmp_path/"missing")}
        c.state["attempts"].append(item)
        with pytest.raises(RuntimeError,match="creation"):
            c.recover()
        item["started_epoch"]-=3.
        c.recover()
        assert c.state["spent_seconds"]==2.
        c.recover()
        assert c.state["spent_seconds"]==2.


def test_cached_artifact_integrity_and_request_identity(tmp_path):
    from bayesfilter.inference.q20_production_config import digest
    import hashlib
    c=campaign(tmp_path)
    result=tmp_path/"result.json"
    result.write_text('{"completed":true}')
    dependency=tmp_path/"map.json"
    dependency.write_text('{}')
    request={"stage":"train"}
    c.state["stages"]["train"]={"request_hash":digest(request),"result_path":str(result),
        "supervisor_seconds":1.,
        "result_sha256":hashlib.sha256(result.read_bytes()).hexdigest(),
        "artifact_hashes":{str(dependency):hashlib.sha256(dependency.read_bytes()).hexdigest()}}
    with c.locked():
        assert c.numerical_stage("train",request,cap_seconds=0,diagnostic=False)["completed"]
        with pytest.raises(ValueError,match="request changed"):
            c.numerical_stage("train",{"stage":"train","different":True},cap_seconds=0,diagnostic=False)
        dependency.write_text('{"changed":true}')
        with pytest.raises(ValueError,match="artifact changed"):
            c.numerical_stage("train",request,cap_seconds=0,diagnostic=False)


def test_failed_json_publication_keeps_last_checkpoint(tmp_path,monkeypatch):
    from bayesfilter.inference.q20_production_config import write_json
    path=tmp_path/"cohort.json"
    write_json(path,{"iteration":1})
    original=Path.replace
    def interrupted(source,destination):
        if source.name=="cohort.json.tmp":
            raise InterruptedError("serialization interrupted")
        return original(source,destination)
    monkeypatch.setattr(Path,"replace",interrupted)
    with pytest.raises(InterruptedError):
        write_json(path,{"iteration":2},exclusive=False)
    assert json.loads(path.read_text())=={"iteration":1}
    with pytest.raises(FileExistsError):
        write_json(path,{"iteration":3})


def test_exhausted_campaign_reports_without_probing_gpu(tmp_path):
    from bayesfilter.inference.q20_master_program import execute_master
    c=campaign(tmp_path)
    with c.locked():
        c.state["spent_seconds"]=c.state["campaign_limit"]
        c.save()
    result=execute_master(c.config,tmp_path,repo=c.repo)
    assert result["status"]=="BUDGET_EXHAUSTED_BEFORE_GPU_PROBE"
