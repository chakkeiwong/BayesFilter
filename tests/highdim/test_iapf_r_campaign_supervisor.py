"""CPU diagnostic harness tests; no R filtering or production imports."""
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"docs/benchmarks"))
driver=importlib.import_module("run_iapf_r_campaign_phase")
report=importlib.import_module("summarize_iapf_r_campaign_phase")
old_report=importlib.import_module("summarize_iapf_r_fitting_comparison")
supervisor=importlib.import_module("supervise_iapf_r_24hour_campaign")


def test_bootstrap_matches_previous_report_and_retains_pairing():
    values=[.71,.96,1.13,1.04,.99,1.12,1.08,.87]
    for sd in (False,True):
        expected=old_report.ci(values,old_report.stats.stdev if sd else old_report.stats.mean)
        assert report.interval(values,sd)==pytest.approx(expected,abs=1e-12)
    assert report.interval([-1000]*8)==[-1000,-1000]


def test_timeout_repairs_only_missing_pairs_and_charges_attempts(tmp_path,monkeypatch):
    out=tmp_path/"campaign";out.mkdir()
    monkeypatch.setattr(driver,"OUT",out)
    monkeypatch.setattr(driver,"ROOT",tmp_path)
    monkeypatch.setattr(driver,"snapshot_sources",lambda *a:{})
    ledger=dict(plan="test",deadline_utc="2099-01-01T00:00:00+00:00",budget_seconds=1000,
                phases={},worker_seconds=0,attempt_count=0)
    monkeypatch.setattr(driver,"initialize",lambda:ledger)
    monkeypatch.setattr(driver,"r_worker_command",lambda snap,worker,output,*args:[str(output),*map(str,args)])
    monkeypatch.setattr(driver.checks,"inspect",lambda spec:dict(records_ok=True,eligible=spec["exit_code"]==0))
    calls=[]
    lock=threading.Lock()
    def fake_worker(command,**kwargs):
        output=Path(command[0]);output.mkdir()
        first=command[4]
        with Path(command[7]).open() as stream:
            skip={(r["replication"],r["method"]) for r in csv.DictReader(stream)}
        with lock:
            calls.append((first,skip))
        for name in ("observations.csv","kalman.csv"):
            (output/name).write_text("fixed data\n")
        methods=["qr","short_qr","bpf","fully_adapted","sis"]
        pending=[m for m in methods if (first,m) not in skip]
        with (output/"replicates.csv").open("w") as stream:
            writer=csv.writer(stream);writer.writerow(["replication","method"])
            for m in pending if skip else pending[:1]:
                writer.writerow([first,m])
        if not skip:
            raise subprocess.TimeoutExpired(command,kwargs["timeout"])
        return 0
    monkeypatch.setattr(driver,"run_bounded_worker",fake_worker)
    phase=tmp_path/"phase.json"
    phase.write_text(json.dumps(dict(phase="timeout-test",plan="test",cells=[dict(
        cell_id="cell",stage="controller",arm="controller",dimension=80,data_seed=4,
        first=3001,repeats=1,chunk_size=1,timeout_seconds=1)])))
    monkeypatch.setattr(sys,"argv",["driver",str(phase)])
    driver.main()
    attempts=[json.loads(line) for line in (out/"attempts.jsonl").read_text().splitlines()]
    assert len(attempts)==2
    assert [a["exit_code"] for a in attempts]==[124,0]
    assert attempts[1]["completed_before"]==[["3001","qr"]]
    assert calls==[("3001",set()),("3001",{("3001","qr")})]
    assert ledger["worker_seconds"]==sum(a["worker_seconds"] for a in attempts)>0
    assert ledger["inflight"]==[]
    assert all((out/a["name"]/"attempt-result.json").exists() for a in attempts)
    # Restart sees all five committed pairs and performs no favorable rerun.
    driver.main()
    assert len(calls)==2


def test_report_does_not_accept_duplicate_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(report,"OUT",tmp_path)
    path=tmp_path/"attempt/results";path.mkdir(parents=True)
    for name in ("observations.csv","kalman.csv"):
        (path/name).write_text("same")
    # Error rows need no success diagnostics but still have unique identities.
    with (path/"replicates.csv").open("w") as stream:
        w=csv.writer(stream);w.writerow(["replication","method","seed","status"])
        w.writerow([3001,"qr",61030011,"error"])
    a=dict(name="attempt",phase="p",cell_id="c",exit_code=2,
        **{name+"_sha256":report.digest(path/name) for name in ("observations.csv","kalman.csv")})
    cell=dict(cell_id="c",first=3001,repeats=1,dimension=80,arm="qr")
    with pytest.raises(RuntimeError,match="duplicate"):
        report.load_cell(dict(phase="p"),cell,[a,a])


@pytest.mark.parametrize("coverage",[True,False])
def test_phase_progression_distinguishes_candidate_rejection_from_missing_evidence(tmp_path,monkeypatch,coverage):
    monkeypatch.setattr(supervisor,"OUT",tmp_path)
    phase="phase02-controller-64"
    manifest=dict(inflight=[],worker_seconds=1,phases={phase:dict(status="workers finished; analysis pending")})
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys,"argv",["supervisor"])
    rejected=dict(complete_coverage=coverage,cells=[dict(methods={"qr":dict(reference_screen_pass=False)})])
    monkeypatch.setattr(supervisor,"analyze",lambda *args:rejected)
    launches=[];outcomes=[]
    def fake_run(command,*args):
        launches.append(command)
        manifest["phases"]["phase03-first-study-1000"]=dict(status="workers finished; analysis pending")
        (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(supervisor,"run",fake_run)
    monkeypatch.setattr(supervisor,"terminal",lambda state,reason:outcomes.append(reason))
    supervisor.main()
    assert bool(launches)==coverage
    assert len(outcomes)==1
    if coverage:
        assert str(launches[0][-1]).endswith("phase03.json")
    else:
        assert "continuation boundary" in outcomes[0]
