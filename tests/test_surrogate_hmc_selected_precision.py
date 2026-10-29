"""CPU-only guards for selected-consumer arithmetic diagnostics."""
import json
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_selected_precision_diagnostic import compare
from scripts import run_surrogate_hmc_selected_precision as controller


class SelectedPrecisionTest(unittest.TestCase):
    def test_owner_scaled_threshold_and_binding_mismatch(self):
        reference=tf.constant([[10., 0.],[-2., 3.]],tf.float64)
        near=tf.constant([[10.09, .009],[-2.015,3.029]],tf.float64)
        far=tf.constant([[10.09, .011],[-2.015,3.029]],tf.float64)
        self.assertTrue(compare(near,reference)["passed"])
        self.assertFalse(compare(far,reference)["passed"])
        self.assertFalse(compare(tf.fill([2,2],float("nan")),reference)["passed"])
        self.assertFalse(compare(near,reference,tolerance=1e-6)["passed"])

    def test_controller_waits_for_terminal_pair_and_free_ledger(self):
        with TemporaryDirectory() as root:
            r=Path(root)
            paired=r/"consumer-paired-followups-01"/"next-phase-plan.json"
            paired.parent.mkdir(parents=True)
            (r/"budget.json").write_text(json.dumps({"attempts":[{"status":"RUNNING"}]}))
            paired.write_text(json.dumps({"status":"PAIRED_CONSUMER_CHECKS_COMPLETE"}))
            self.assertFalse(controller.paired_finished(r))
            (r/"budget.json").write_text(json.dumps({"attempts":[{"status":"FINISHED"}]}))
            self.assertTrue(controller.paired_finished(r))
            paired.write_text(json.dumps({"status":"WAIT_LIMIT_REACHED"}))
            self.assertFalse(controller.paired_finished(r))

    def test_snapshot_mismatch_and_targeted_queue(self):
        self.assertEqual(controller.models_to_run("lgssm"),("lgssm",))
        with TemporaryDirectory() as root:
            r=Path(root); source=r/"src.py"; source.write_text("first")
            snapshot=r/"out/source/src.py"; snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(source.read_bytes())
            result={"source_sha256":{str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
                    "scope":{"source_sha256":{}}}
            self.assertEqual(controller.snapshot_mismatches(result,r/"out",r),[])
            snapshot.write_text("changed")
            self.assertEqual(controller.snapshot_mismatches(result,r/"out",r),["src.py"])


if __name__ == "__main__":
    unittest.main()


def test_explicit_followup_consumes_corrected_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root=tmp_path/"campaign"; root.mkdir()
    advance=root/"corrected-advance"; paired=root/"corrected-paired"
    plan=root/"corrected-scope.json"; out=tmp_path/"out"
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    write(root/"budget.json",{"attempts":[],"max_attempts":10,
          "authorized_gpu_wall_seconds":1000})
    write(root/"consumer-paired-followups-01/next-phase-plan.json",
          {"status":"PAIRED_CONSUMER_CHECKS_COMPLETE"})
    assert not controller.paired_finished(root,paired)
    write(paired/"next-phase-plan.json",{"status":"LOCAL_REPAIR_REQUIRED"})
    assert controller.paired_finished(root,paired)
    selected=advance/"lgssm/filter-tuning/lgssm/filter-selection.json"
    write(selected,{"selected":{"index":1}})
    monkeypatch.setattr(controller.master,"MODELS",("lgssm",))
    def scopes(path):
        assert path==plan
        return {"lgssm":{"contraction":"direct","runtime_precision":"uniform",
                         "tuning_seed_offset":60}}
    monkeypatch.setattr(controller.master,"execution_scopes",scopes)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        assert command[command.index("--selection")+1]==str(selected)
        assert command[command.index("--scope-plan")+1]==str(plan)
        assert not controller.paired_finished(root,paired)
        write(out/"lgssm/selected-consumer-precision.json",{"status":controller.PASS})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controller.subprocess,"run",run)
    assert controller.execute(root,out,advance_dir=advance,paired_dir=paired,
                              execution_plan=plan)==0
    state=controller.read(out/"next-phase-plan.json")
    assert len(calls)==1
    assert state["source_advance_dir"]==str(advance)
    assert state["paired_dir"]==str(paired)
    assert state["execution_plan"]==str(plan)
    assert state["status"]=="SELECTED_CONSUMER_PRECISION_COMPLETE"
    ledger=controller.read(root/"budget.json")
    assert ledger["attempts"][0]["status"]==controller.PASS
    assert ledger["used_seconds"]>=0
