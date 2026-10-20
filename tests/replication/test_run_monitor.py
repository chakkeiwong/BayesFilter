"""Host-only tests of unattended decisions; never touch live jobs or notifications."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import base64
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import monitor_controlled_smc_kdm as monitor


def fixture():
    queues = {phase: dict(status="WAITING", supervisor_alive=True, deadline_epoch=10000)
              for phase in monitor.QUEUES}
    source = dict(status="RUNNING", started_epoch=0, timeout_seconds=10000,
                  worker_alive=True, supervisor_alive=True, manifest="attempt145/manifest.json")
    drivers = [dict(driver=key, completed_cases=count, expected_cases=total,
                    nonfinite_cases=0, partial_or_invalid_rows=0)
               for key, count, total in (("lorenz_csmc",3537,4000), ("lorenz_apf",0,4000),
                    ("neuroscience_csmc",2000,2000), ("neuroscience_bpf",855,2000))]
    failures = [dict(driver=key, status=status, completed_cases=count)
                for key, status, count in monitor.KNOWN_FAILURES]
    return dict(at=1000, progress=dict(drivers=drivers, source_driver_failures=failures),
                attempts={"c1_source_full":source}, queues=queues,
                memory_bytes=dict(MemAvailable=4*1024**3, MemTotal=8*1024**3))


def config():
    return dict(queue_deadlines={phase:10000 for phase in monitor.QUEUES})


class RunMonitorTests(unittest.TestCase):
    def test_healthy_progress_and_known_failure_do_not_alert(self):
        s, state = fixture(), {}
        self.assertEqual(monitor.evaluate(s,state,config()), ([],[],False))
        s["at"] += 1800
        s["progress"]["drivers"][-1]["completed_cases"] += 1
        self.assertEqual(monitor.evaluate(s,state,config()), ([],[],False))

    def test_stall_warns_without_numerical_relaunch(self):
        s, state = fixture(), {}
        monitor.evaluate(s,state,config())
        s["at"] += 1260
        self.assertEqual(monitor.evaluate(s,state,config())[0], [])
        s["at"] += 60
        events, repairs, done = monitor.evaluate(s,state,config())
        self.assertTrue(any("stalled" in e[2] for e in events))
        self.assertEqual(repairs, [])
        self.assertFalse(done)

    def test_partial_writer_row_does_not_alert_if_transient(self):
        s, state = fixture(), {}
        s["progress"]["drivers"][-1]["partial_or_invalid_rows"] = 1
        self.assertEqual(monitor.evaluate(s,state,config())[0], [])
        s["progress"]["drivers"][-1]["partial_or_invalid_rows"] = 0
        self.assertEqual(monitor.evaluate(s,state,config())[0], [])
        s["progress"]["drivers"][-1]["nonfinite_cases"] = 1
        monitor.evaluate(s,state,config())
        self.assertTrue(monitor.evaluate(s,state,config())[0])

    def test_dead_waiting_queue_can_recover_only_after_confirmation(self):
        s, state = fixture(), {}
        phase = monitor.QUEUES[0]
        s["queues"][phase]["supervisor_alive"] = False
        self.assertEqual(monitor.evaluate(s,state,config())[1], [])
        self.assertEqual(monitor.evaluate(s,state,config())[1], [phase])
        state["repair_counts"] = {phase:2}
        self.assertEqual(monitor.evaluate(s,state,config())[1], [])

    def test_executing_or_attempted_phase_cannot_restart(self):
        for status, attempted in (("EXECUTING",False), ("WAITING",True), ("FAILED",False)):
            with self.subTest(status=status, attempted=attempted):
                s, state = fixture(), {}
                phase = monitor.QUEUES[0]
                s["queues"][phase].update(status=status, supervisor_alive=False)
                if attempted:
                    s["attempts"][phase] = copy.deepcopy(s["attempts"]["c1_source_full"])
                monitor.evaluate(s,state,config())
                self.assertEqual(monitor.evaluate(s,state,config())[1], [])

    def test_dead_worker_alerts_without_restart(self):
        s, state = fixture(), {}
        s["attempts"]["c1_source_full"]["worker_alive"] = False
        monitor.evaluate(s,state,config())
        events, repairs, _ = monitor.evaluate(s,state,config())
        self.assertTrue(any("worker missing" in e[2] for e in events))
        self.assertEqual(repairs, [])

    def test_known_partial_terminal_requires_complete_neuroscience(self):
        s = fixture()
        s["attempts"]["c1_source_full"].update(status="FAILED",returncode=1,decision={})
        self.assertEqual(monitor.evaluate(s,{},config())[0][0][1], "ERROR")
        s["progress"]["drivers"][-1]["completed_cases"] = 2000
        event = monitor.evaluate(s,{},config())[0][0]
        self.assertEqual(event[1], "INFO")
        self.assertIn("Replication incomplete", event[2])
        s["progress"]["drivers"][-1]["nonfinite_cases"] = 1
        self.assertEqual(monitor.evaluate(s,{},config())[0][0][1], "ERROR")

    def test_diagnostic_pass_with_failed_comparison_is_an_error(self):
        s = fixture()
        s["attempts"]["p0_forward_compare"] = dict(status="COMPLETE", manifest="attempt148/manifest.json",
            decision=dict(status="PASS",forward_screen_passed="False",comparison_verdict="MISMATCH"))
        event = monitor.evaluate(s,{},config())[0][0]
        self.assertEqual(event[1], "ERROR")
        self.assertIn("requires investigation",event[2])

    def test_repair_calls_real_dispatcher_entry_with_fresh_state_and_cap(self):
        s, state, phase = fixture(), {}, monitor.QUEUES[0]
        s["queues"][phase]["supervisor_alive"] = False
        with patch.object(monitor,"sample",return_value=s), \
             patch.object(monitor.runner,"status",return_value={}), \
             patch.object(monitor.runner,"queue_wait_state",return_value=(False,10000)), \
             patch.object(monitor.runner,"start_detached",return_value={"launch":{"status":"WAITING"}}) as launch:
            self.assertEqual(monitor.repair_queue(phase,state,config())["status"],"RESTART_REQUESTED")
            launch.assert_called_once_with(phase)
            s["queues"][phase]["supervisor_alive"] = True
            self.assertEqual(monitor.repair_queue(phase,state,config())["status"],"SKIPPED_STATE_CHANGED")
            self.assertEqual(launch.call_count,1)

    def test_repair_refuses_deadline_extension_or_expiry(self):
        s, phase = fixture(), monitor.QUEUES[0]
        s["queues"][phase]["supervisor_alive"] = False
        with patch.object(monitor,"sample",return_value=s), \
             patch.object(monitor.runner,"status",return_value={}), \
             patch.object(monitor.runner,"queue_wait_state",return_value=(False,10001)), \
             patch.object(monitor.runner,"start_detached") as launch:
            self.assertEqual(monitor.repair_queue(phase,{},config())["status"],"REPAIR_FAILED")
            s["at"] = 10001
            self.assertEqual(monitor.repair_queue(phase,{},config())["status"],"SKIPPED_STATE_CHANGED")
            launch.assert_not_called()

    def test_monitor_deadline_includes_final_gpu_execution(self):
        deadlines = config()["queue_deadlines"]
        self.assertGreaterEqual(monitor.monitoring_deadline(deadlines),
                                deadlines["p0_forward_compare"] + 600 + 120)

    def test_notifications_are_data_not_shell_interpolation(self):
        title, body = "Status <&>", "$(touch /tmp/not-executed) `echo nope` ' quoted"
        with patch.object(monitor.subprocess,"run",return_value=subprocess.CompletedProcess([],0,b"SUBMITTED; setting=Enabled",b"")) as run:
            result = monitor.notify(title,body)
        command = run.call_args.args[0]
        script = base64.b64decode(command[-1]).decode("utf-16le")
        encoded = script.split("FromBase64String('",1)[1].split("'",1)[0]
        xml = ET.fromstring(base64.b64decode(encoded))
        self.assertEqual([n.text for n in xml.iter("text")],[title,body])
        self.assertNotIn(body,script)
        self.assertFalse(run.call_args.kwargs.get("shell",False))
        self.assertTrue(result["submitted"])
        self.assertTrue(result["notifications_enabled"])


if __name__ == "__main__":
    unittest.main()
