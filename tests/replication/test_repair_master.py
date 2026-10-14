"""Focused dispatcher regression: no scientific or GPU work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import signal
import os
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("master",ROOT/"scripts/run_replication_master.py")
master=importlib.util.module_from_spec(spec);spec.loader.exec_module(master)

class MasterTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="replication-master-test-")
        self.root=Path(self.temp.name)
        master.atomic_json(self.root/"opening-budget.json",dict(opening_cpu_seconds=0,
            opening_gpu_seconds=0,cpu_limit_seconds=10,gpu_limit_seconds=10,new_attempt_limit=3))
        master.atomic_json(self.root/"source-iapf.json",{"status":"TEST"})
        (self.root/"execution-spec.md").write_text("Harness fixture only")
        self.script=self.root/"fixture.R"
        self.script.write_text('args<-commandArgs(TRUE);if(args[1]=="fail")quit(status=2);'
            'if(args[1]=="timeout")Sys.sleep(3);'
            'write.csv(data.frame(status="PASS"),file.path(args[2],"decision.csv"),row.names=FALSE)')
        self.phase=dict(id="iapf_controls",stage="TEST",resource="cpu",limit=2,
            depends=[],script=str(self.script),argument="controls",implemented=True)
        self.patches=[patch.object(master,"SOURCES",[str(self.script)]),
                      patch.object(master,"PHASES",[self.phase])]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in self.patches:p.stop()
        self.temp.cleanup()
    def test_value_parity_does_not_complete_replications(self):
        for i,phase in enumerate(["iapf_value_cpu","iapf_value_gpu_graph","iapf_value_gpu_xla"]):
            folder=self.root/f"attempt{i+1:02d}";folder.mkdir()
            master.atomic_json(folder/"manifest.json",dict(phase=phase,resource="cpu",
                status="COMPLETE",charged_seconds=0,decision={"status":"PASS"}))
        result=master.snapshot(self.root)
        self.assertEqual(result["paper_target_frozen_twist_value_parity"],"PASS")
        self.assertEqual(result["gpu_port"],"PARTIAL_VALUE_PARITY_ONLY")
        self.assertEqual(result["iapf_replication"],"INCOMPLETE")
        self.assertEqual(result["mdpf_replication"],"INCOMPLETE")
    def test_finite_and_adaptive_parity_cannot_complete_replication(self):
        for i,phase in enumerate(["iapf_finite_cpu","iapf_finite_gpu_graph",
                                  "iapf_finite_gpu_xla","iapf_paper_adaptive_gpu_xla_v2"]):
            folder=self.root/f"attempt{i+1:02d}";folder.mkdir()
            master.atomic_json(folder/"manifest.json",dict(phase=phase,resource="cpu",
                status="COMPLETE",charged_seconds=0,decision={"status":"PASS"}))
        result=master.snapshot(self.root)
        self.assertEqual(result["finite_fitter_controller_parity"],"PASS")
        self.assertEqual(result["paper_horizon_adaptive_parity"],"PASS")
        self.assertEqual(result["iapf_replication"],"INCOMPLETE")
        self.assertEqual(result["mdpf_replication"],"INCOMPLETE")
        self.assertIn("MDPF_PORT_OPEN",result["gpu_port"])
    def test_diagnostic_completion_preserves_failed_paper_parity(self):
        records=[("iapf_paper_adaptive_gpu_xla_v2","FAILED","FAIL"),
                 ("iapf_finite_impact","COMPLETE","PASS")]
        for i,(phase,status,verdict) in enumerate(records):
            folder=self.root/f"attempt{i+1:02d}";folder.mkdir()
            master.atomic_json(folder/"manifest.json",dict(phase=phase,resource="cpu",
                status=status,charged_seconds=0,decision={"status":verdict}))
        result=master.snapshot(self.root)
        self.assertEqual(result["paper_horizon_adaptive_parity"],"FAIL")
        self.assertEqual(result["paper_horizon_impact_diagnostic"],"COMPLETE_NOT_PROMOTION")
        self.assertIn("iapf_paper_adaptive_gpu_xla_v2",result["failed_outcomes"])
        self.assertEqual(result["iapf_replication"],"INCOMPLETE")

    def test_repaired_source_status_preserves_evidence_and_nonpromotion(self):
        records=[("iapf_vmmin_cpu","COMPLETE","PASS"),
                 ("iapf_vmmin_gpu_graph","COMPLETE","PASS"),
                 ("iapf_vmmin_gpu_xla_regression","COMPLETE","PASS"),
                 ("iapf_repaired_small_cpu","COMPLETE","PASS"),
                 ("iapf_repaired_small_gpu_xla","COMPLETE","PASS"),
                 ("iapf_paper_adaptive_gpu_xla_v2","FAILED","FAIL"),
                 ("iapf_repaired_paper_gpu_xla","FAILED","FAIL")]
        for i,(phase,status,verdict) in enumerate(records):
            folder=self.root/f"attempt{i+1:02d}";folder.mkdir()
            master.atomic_json(folder/"manifest.json",dict(phase=phase,resource="cpu",
                status=status,charged_seconds=0,decision={"status":verdict}))
        result=master.snapshot(self.root)
        self.assertEqual(result["finite_solver_source_controls"],"PASS")
        self.assertEqual(result["repaired_fitter_controller_parity"],"PASS")
        self.assertEqual(result["repaired_paper_horizon_parity"],"FAIL")
        self.assertIn("iapf_repaired_paper_gpu_xla",result["failed_outcomes"])
        # Even a future repaired-source pass cannot erase a historical failure
        # or promote the complete scientific replication.
        master.atomic_json(folder/"manifest.json",dict(phase=phase,resource="cpu",
            status="COMPLETE",charged_seconds=0,decision={"status":"PASS"}))
        result=master.snapshot(self.root)
        self.assertEqual(result["repaired_paper_horizon_parity"],"PASS")
        self.assertEqual(result["paper_horizon_adaptive_parity"],"FAIL")
        self.assertIn("iapf_paper_adaptive_gpu_xla_v2",result["failed_outcomes"])
        self.assertEqual(result["iapf_replication"],"INCOMPLETE")
        self.assertEqual(result["mdpf_replication"],"INCOMPLETE")

    def test_cli_reports_child_failure(self):
        with patch.object(sys,"argv",["master","run-ready"]), \
             patch.object(master,"validate"), \
             patch.object(master,"snapshot",return_value={"next_phase":"iapf_controls"}), \
             patch.object(master,"execute",return_value={"status":"FAILED","returncode":7}), \
             patch("builtins.print"):
            self.assertEqual(master.main(),1)
    def test_preflight_rejects_syntax_before_launch(self):
        script=self.root/"broken.py";script.write_text('value="unterminated')
        self.phase["script"]=str(script)
        with self.assertRaises(SyntaxError):master.validate(self.root)
        self.assertEqual(master.load_attempts(self.root),[])
    def test_unfinished_dependency_blocks_launch(self):
        self.phase["depends"]=["missing_phase"]
        self.assertEqual(master.snapshot(self.root)["phases"][0]["state"],"WAITING_FOR_DEPENDENCY")
        with self.assertRaises(RuntimeError):master.execute(self.phase,self.root)
    def test_success_and_no_duplicate_completion(self):
        result=master.execute(self.phase,self.root)
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(master.snapshot(self.root)["phases"][0]["state"],"COMPLETE")
        self.assertIsNone(master.snapshot(self.root)["next_phase"])
        with self.assertRaises(RuntimeError):master.execute(self.phase,self.root)
    def test_failure_preserves_log_and_charge(self):
        self.phase["argument"]="fail"
        result=master.execute(self.phase,self.root)
        self.assertEqual(result["status"],"FAILED")
        self.assertGreater(result["charged_seconds"],0)
        self.assertTrue((Path(result["output_root"])/"command.log").exists())
    def test_timeout_is_bounded(self):
        self.phase.update(argument="timeout",limit=1)
        result=master.execute(self.phase,self.root)
        self.assertEqual(result["status"],"TIMED_OUT")
        self.assertLess(result["charged_seconds"],2)
    def test_duplicate_lock(self):
        with master.acquire_lock(self.root):
            with self.assertRaises(RuntimeError):master.acquire_lock(self.root)
    def test_budget_blocks(self):
        p=self.root/"opening-budget.json";b=json.loads(p.read_text())
        b["opening_cpu_seconds"]=10;master.atomic_json(p,b)
        self.assertEqual(master.snapshot(self.root)["phases"][0]["state"],"BUDGET_EXHAUSTED")
        with self.assertRaises(RuntimeError):master.execute(self.phase,self.root)
    def test_lost_supervisor_does_not_invent_success(self):
        d=self.root/"attempt01";d.mkdir()
        master.atomic_json(d/"manifest.json",dict(phase="iapf_controls",resource="cpu",
            status="RUNNING",pid=999999999,process_identity="never",
            supervisor_pid=999999998,supervisor_identity="never",
            started_epoch=time.time()-2,timeout_seconds=1))
        master.recover(self.root)
        m=json.loads((d/"manifest.json").read_text())
        self.assertEqual(m["status"],"INTERRUPTED_EXIT_UNOBSERVED")
        self.assertIsNone(m["returncode"])
        self.assertEqual(m["charged_seconds"],1)
    def test_running_charge_blocks_concurrent_launch(self):
        d=self.root/"attempt01";d.mkdir()
        master.atomic_json(d/"manifest.json",dict(phase="iapf_controls",resource="cpu",
            status="RUNNING",pid=999999999,process_identity="never",
            supervisor_pid=999999998,supervisor_identity="never",
            started_epoch=time.time()-2,timeout_seconds=1))
        self.assertIsNone(master.snapshot(self.root)["next_phase"])


    def test_killed_supervisor_keeps_worker_lock_until_timeout(self):
        self.phase.update(argument="timeout",limit=1.5)
        setup=("import importlib.util,json;from pathlib import Path;"
               f"spec=importlib.util.spec_from_file_location('master',{str(ROOT/'scripts/run_replication_master.py')!r});"
               "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
               f"m.SOURCES=[{str(self.script)!r}];m.PHASES=[{self.phase!r}];"
               f"m.execute(m.PHASES[0],Path({str(self.root)!r}))")
        proc=subprocess.Popen([sys.executable,"-c",setup],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                manifests=list(self.root.glob("attempt*/manifest.json"))
                if manifests:
                    data=json.loads(manifests[0].read_text())
                    if data.get("pid"):break
                time.sleep(.01)
            else:self.fail("worker did not start")
            os.kill(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
            with self.assertRaises(RuntimeError):master.acquire_lock(self.root)
            while time.monotonic()<deadline+2:
                try:
                    lock=master.acquire_lock(self.root);lock.close();break
                except RuntimeError:time.sleep(.02)
            else:self.fail("inherited lock outlived timeout")
            master.recover(self.root)
            self.assertEqual(json.loads(manifests[0].read_text())["status"],"INTERRUPTED_EXIT_UNOBSERVED")
        finally:
            if proc.poll() is None:proc.kill();proc.wait()

if __name__=="__main__":
    unittest.main()
