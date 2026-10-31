"""Mechanical observer checks; fake processes do not establish GPU readiness."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location("hmc_price_observer", Path(__file__).with_name("observe_complete_price.py"))
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@pytest.mark.parametrize("mode", ["complete", "busy_then_ready", "always_busy", "observer_error", "foreign_after_launch"])
def test_observer_waits_and_preserves_owned_process_boundary(tmp_path, monkeypatch, mode):
    root = tmp_path/"output"
    monkeypatch.setattr(sys,"argv",["observer", "--source",str(tmp_path/"source"),"--output",str(root),"--gpu","GPU-fixture"])
    monkeypatch.setattr(os,"sched_setaffinity",lambda *args:None)
    now=[0.];monkeypatch.setattr(observer.time,"monotonic",lambda:now[0])
    monkeypatch.setattr(observer.time,"sleep",lambda delay:now.__setitem__(0,now[0]+delay))
    probes=[];launched=[];signals=[]
    class Child:
        pid=999999;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=0;return 0
    child=Child()
    def launch(command,**kwargs):
        assert kwargs["start_new_session"]
        assert kwargs["env"]["TF_FORCE_GPU_ALLOW_GROWTH"]=="true"
        assert not (root/"prices").exists()
        launched.append(command);return child
    monkeypatch.setattr(observer.subprocess,"Popen",launch)
    monkeypatch.setattr(observer.os,"killpg",lambda *args:signals.append(args))
    def probe(gpu,owner=None):
        probes.append(owner)
        if mode=="observer_error" and owner is not None:raise RuntimeError("lost telemetry")
        busy=(mode=="always_busy" or (mode=="busy_then_ready" and len(probes)==1)
              or (mode=="foreign_after_launch" and owner is not None))
        return dict(free_mib=8000,foreign_pids=[123] if busy else [],unknown_pids=[])
    monkeypatch.setattr(observer,"probe",probe)
    code=observer.main()
    receipt=json.loads((root/"receipt.json").read_text())
    if mode=="always_busy":
        assert code==3 and not launched and now[0]==60 and not signals
    elif mode=="observer_error":
        assert code==1 and receipt["status"]=="observer_or_price_failure"
        assert signals and all(pid==child.pid for pid,_ in signals)
    else:
        assert code==0 and len(launched)==1 and not signals
        assert receipt["wall_seconds"]==(10 if mode=="busy_then_ready" else 0)
        assert receipt["foreign_or_unknown_during_price"] is (mode=="foreign_after_launch")
    assert receipt["price_result"]==str(root/"prices/result.json")


def test_ownership_distinguishes_current_ancestor_and_unknown_pid():
    assert observer.descendant(os.getpid(),os.getpid()) is True
    assert observer.descendant(1,os.getpid()) is False
    assert observer.descendant(2**31-1,os.getpid()) is None
