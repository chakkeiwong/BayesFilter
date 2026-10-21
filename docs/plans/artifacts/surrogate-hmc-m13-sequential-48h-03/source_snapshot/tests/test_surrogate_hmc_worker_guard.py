"""Independent watchdog regression, deliberately CPU-only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from bayesfilter.inference.surrogate_hmc_program import ROOT, run_job
from bayesfilter.inference.surrogate_hmc_worker_guard import process_identity

def ledger():
    return dict(limits=dict(cpu_worker_seconds=30.,gpu_seconds=30.),
                charged=dict(cpu_worker_seconds=0.,gpu_seconds=0.), jobs=[])

def test_receipt_is_complete_for_failed_work_and_reaped_children(tmp_path):
    result = run_job([sys.executable,"-c",
        "import subprocess,sys; subprocess.run([sys.executable,'-c','sum(range(1000000))']); raise SystemExit(7)"],
        tmp_path/"job",ledger(),device="cpu",timeout=5.,cpu_allowance=5.)
    receipt=json.loads((tmp_path/"job/guard-receipt.json").read_text())
    assert result==receipt and result["accounting_complete"]
    assert result["exit_code"]==7 and result["status"]=="FAILED"
    assert result["cpu_worker_seconds"]>0 and result["gpu_seconds"]==0

def test_owner_loss_stops_worker_and_preserves_receipt(tmp_path):
    directory=tmp_path/"orphan-test"
    code=("from pathlib import Path; from bayesfilter.inference.surrogate_hmc_program import run_job; "
          f"run_job([{sys.executable!r},'-c','import time; time.sleep(30)'],Path({str(directory)!r}),"
          f"{ledger()!r},device='cpu',timeout=15.,cpu_allowance=10.)")
    owner=subprocess.Popen([sys.executable,"-c",code],cwd=ROOT,
                           env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES="-1"))
    try:
        deadline=time.monotonic()+8
        heartbeat=directory/"heartbeat.json"
        while not heartbeat.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert heartbeat.exists()
        worker=json.loads(heartbeat.read_text())["worker_pid"]
        owner.kill()
        owner.wait()
        receipt=directory/"guard-receipt.json"
        while not receipt.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert receipt.exists(), (directory/"guard.log").read_text()
        result=json.loads(receipt.read_text())
        assert result["status"]=="OWNER_LOST" and result["accounting_complete"]
        assert result["cpu_worker_seconds"]>0 and process_identity(worker) is None
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait()

def test_health_progress_callback_does_not_replace_watchdog(tmp_path):
    events=[]
    result=run_job([sys.executable,"-c","import time; time.sleep(8)"],
        tmp_path/"job",ledger(),device="cpu",timeout=.8,cpu_allowance=2.,
        progress_callback=lambda directory:events.append(directory))
    assert events and result["status"]=="WALL_TIME_LIMIT"
    assert result["wall_seconds"]<4 and result["accounting_complete"]
