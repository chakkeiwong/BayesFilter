"""Orchestration invariants; no TensorFlow or GPU work is launched."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[1]/"scripts/run_surrogate_hmc_marginal_expanded.py"
spec=importlib.util.spec_from_file_location("expanded_master_under_test",PATH)
master=importlib.util.module_from_spec(spec)
spec.loader.exec_module(master)


def setup_previous(tmp_path,monkeypatch):
    previous=tmp_path/"previous";previous.mkdir()
    args=argparse.Namespace(output=tmp_path/"new",prior_campaign=previous,
        budget_seconds=1.,gpu="diagnostic-no-gpu",scopes="lg1-t10,lg3-t10",
        particle_ladder="1024",stage="all")
    config=master.configuration(args)
    rows=[]
    for scope,valid in (("lg1-t10",True),("lg3-t10",False)):
        result=previous/(scope+".json")
        result.write_text(json.dumps(dict(datasets=4,summaries=[dict(valid=valid)])))
        rows.append(dict(kind="scope",scope=scope,particles=1024,
            key="scope-"+scope+"-n1024",status="COMPLETE",wall_seconds=2.,
            result=str(result)))
    record=dict(configuration=dict(config,harness_revision="previous_version"),
        status="ARCHIVED_FOR_REPAIRED_SOURCE",attempts=rows,
        prior_consumed_seconds=3.,migration_additional_seconds=4.)
    (previous/"campaign.json").write_text(json.dumps(record))
    monkeypatch.setattr(master,"identity",lambda:{"fixture":"new_source"})
    def forbid_launch(*a,**kw):
        raise AssertionError("exhausted campaign must not launch a worker")
    monkeypatch.setattr(master.subprocess,"Popen",forbid_launch)
    return args,record


def test_continuation_charges_every_attempt_and_preserves_source(tmp_path,monkeypatch):
    args,previous=setup_previous(tmp_path,monkeypatch)
    master.master(args)
    result=json.loads((args.output/"campaign.json").read_text())
    assert result["prior_consumed_seconds"]==11.
    assert len(result["attempts"])==1
    old=result["attempts"][0]
    assert old["scope"]=="lg1-t10" and old["wall_seconds"]==0
    assert old["source_revision"]=="previous_version"
    assert old["evidence_status"]=="retained_previous_source_result_not_rerun"
    assert old["result_sha256"]==hashlib.sha256(Path(old["result"]).read_bytes()).hexdigest()
    assert result["status"]=="INCOMPLETE"


def test_continuation_rejects_a_live_old_worker(tmp_path,monkeypatch):
    args,previous=setup_previous(tmp_path,monkeypatch)
    previous["attempts"][0]["status"]="RUNNING"
    (args.prior_campaign/"campaign.json").write_text(json.dumps(previous))
    with pytest.raises(ValueError,match="worker still running"):
        master.master(args)


def test_continuation_cannot_silently_expand_budget(tmp_path,monkeypatch):
    args,previous=setup_previous(tmp_path,monkeypatch)
    args.budget_seconds=2.
    with pytest.raises(ValueError,match="preserve the declared grid and budget"):
        master.master(args)
