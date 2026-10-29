"""Bounded repair-screen phase with automatic continuation of the existing master."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from bayesfilter.inference import surrogate_hmc_recovery_program as master
from bayesfilter.inference.surrogate_hmc_program import run_job, write_json, file_hash
from bayesfilter.inference.surrogate_hmc_worker_guard import process_identity

DEV = Path(__file__).resolve().parent
STATE = DEV / "energy-repair-continuation.json"
OUTPUT = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-03"
PROBE = DEV / "energy-step-probe-03"
PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
COMMAND = [PYTHON, str(ROOT / "scripts/run_surrogate_hmc_recovery.py"), "start",
           "--allocation", "48h", "--cpu-budget", "172800", "--gpu-budget", "172800",
           "--output", str(OUTPUT)]


def main():
    state = dict(status="CHECKING", part_of_existing_master=True,
        scientific_goal="testing surrogate HMC on frozen M13/data1704",
        owner=process_identity(os.getpid()),
        started=datetime.now(timezone.utc).isoformat(),
        plan=master.CONTINUATION_PLAN, output=str(OUTPUT),
        probe=str(PROBE), next_master_command=COMMAND,
        posterior_qualified=False, master_complete=False)
    write_json(STATE, state)
    try:
        if OUTPUT.exists() or PROBE.exists():
            raise ValueError("fresh probe and master output directories required")
        charges, receipts = master.continuation_charges(OUTPUT)
        history = next(r for r in receipts if r.get("kind") == "allocation history")
        if history["scientific_attempts_started"] != 1:
            raise ValueError("this repair requires exactly one remaining full attempt")
        development = dict(cpu_worker_seconds=0., gpu_seconds=0.)
        for receipt in DEV.glob("*/guard-receipt.json"):
            record = json.loads(receipt.read_text())
            if record.get("accounting_complete") is not True:
                raise ValueError("incomplete development accounting")
            for key in development:
                development[key] += record[key]
        reservation = dict(cpu_worker_seconds=3600., gpu_seconds=3000.)
        if any(development[k]+reservation[k] > 21600. for k in development):
            raise ValueError("six-hour development envelope cannot cover this screen")
        profile = master.allocation_profile("48h")
        if any(charges[k]+reservation[k]+profile["retained_reserve"] >= profile["limits"][k] for k in charges):
            raise ValueError("screen would consume the retained reserve")
        if master.seconds_to_deadline(profile) <= 3000.+profile["retained_reserve"]:
            raise ValueError("deadline cannot preserve the retained reserve")
        frozen_names = set(master.NEW_SOURCES) | {
            "bayesfilter/inference/hmc_tensorflow_tuning.py",
            str(Path(__file__).relative_to(ROOT)),
            str((DEV / "energy_step_probe_03.py").relative_to(ROOT))}
        hashes = {name:file_hash(ROOT / name) for name in frozen_names}
        state.update(status="RUNNING_ENERGY_REPAIR_SCREEN",
            active_phase="m13-energy-repair-screen", source_hashes=hashes,
            prior_charges=charges, prior_development_charges=development,
            reservation=reservation, automatic_next_phase="master tests and attempt2 preflight")
        write_json(STATE, state)
        ledger = dict(limits=reservation, charged=dict(cpu_worker_seconds=0., gpu_seconds=0.), jobs=[])
        job = run_job([PYTHON, str(DEV / "energy_step_probe_03.py"), str(PROBE)],
            PROBE, ledger, device="gpu", timeout=3000., cpu_allowance=3600.)
        state["probe_receipt"] = job
        report_path = PROBE / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if (job.get("status") != "COMPLETED" or not job.get("accounting_complete")
                or report.get("status") != "DIAGNOSTIC_COMPLETE"
                or report.get("repair_screen_passed") is not True):
            state.update(status="REPAIR_SCREEN_NOT_QUALIFIED", active_phase=None,
                automatic_next_phase=None, reason="preserve probe result; do not launch full attempt")
            write_json(STATE, state)
            return 1
        master.check_snapshot(hashes)
        # The master repeats actual-binding checks and counts every prior receipt.
        launch = subprocess.run(COMMAND, cwd=ROOT, text=True, capture_output=True, check=False)
        state.update(master_launch_returncode=launch.returncode, master_launch_stdout=launch.stdout,
                     master_launch_stderr=launch.stderr)
        if launch.returncode:
            raise RuntimeError("master launch failed; see preserved launch output")
        state.update(status="MASTER_STARTED", active_phase="master",
                     automatic_next_phase="owned by master next-phase-plan.json",
                     master_supervisor=json.loads(launch.stdout))
        write_json(STATE, state)
        return 0
    except Exception as exc:
        state.update(status="REPAIR_CONTINUATION_FAILED", reason=repr(exc), active_phase=None)
        write_json(STATE, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
