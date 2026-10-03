"""Local unattended phase progression for the authorized independent R campaign.

This supervisor does not select numerical settings from results. Failed method
screens continue to the frozen study; incomplete/corrupt evidence is explicit.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"docs/plans/artifacts/iapf-r-24hour-campaign-20260921-01"
DRIVER=ROOT/"docs/benchmarks/run_iapf_r_campaign_phase.py"
REPORTER=ROOT/"docs/benchmarks/summarize_iapf_r_campaign_phase.py"
CHECKPOINT=ROOT/"docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md"
MASTER=ROOT/"docs/plans/younis-kdm-score-master-program-2026-09-14.md"


def save(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def heartbeat(state):
    state["heartbeat_utc"]=datetime.now(timezone.utc).isoformat()
    save(OUT/"supervisor.json",state)


def run(command,log_path,state,label):
    state.update(status=label,command=[str(c) for c in command])
    with log_path.open("w") as log:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",
            OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"),stdout=log,stderr=subprocess.STDOUT)
        state["child_pid"]=child.pid
        heartbeat(state)
        while child.poll() is None:
            heartbeat(state)
            time.sleep(20)
        state["last_exit_code"]=child.returncode
        state.pop("child_pid",None)
        heartbeat(state)
    if child.returncode:
        raise RuntimeError(f"{label} exited {child.returncode}; see {log_path}")


def analyze(phase_file,state):
    phase=read(phase_file)
    directory=OUT/phase["phase"]
    # Each invocation preserves prior reports, including any failed partial one.
    version=1
    while (directory/f"analysis-v{version}").exists():
        version+=1
    run([sys.executable,str(REPORTER),str(phase_file),"--output-name",f"analysis-v{version}"],
        OUT/f"{phase['phase']}-analysis-v{version}.log",state,f"analyzing {phase['phase']}")
    summary=directory/f"analysis-v{version}/summary.json"
    state.setdefault("reports",{})[phase["phase"]]=str(summary)
    heartbeat(state)
    return read(summary)


def terminal(state,reason):
    manifest=read(OUT/"manifest.json")
    summaries=[read(Path(path)) for path in state.get("reports",{}).values()]
    rows=[]
    for summary in summaries:
        for cell in summary["cells"]:
            for method in ("qr","short_qr"):
                value=cell["methods"].get(method,{})
                rows.append(dict(phase=summary["phase"],dimension=cell["dimension"],
                    data_seed=cell["data_seed"],method=method,n=value.get("n",0),planned=cell["repeats"],
                    practical=value.get("practical_primary_pass",False),
                    heuristic=value.get("heuristic_dominance","unchecked"),
                    literal=value.get("literal_pattern_pass",False),
                    reference_screen=value.get("reference_screen_pass",False)))
    full=[r for r in rows if r["phase"]=="phase03-first-study-1000"]
    decision=dict(reason=reason,study_complete=len(full)==10 and all(r["n"]==1000 for r in full),
        method_screen={m:len([r for r in full if r["method"]==m])==5 and
            all(r["reference_screen"] for r in full if r["method"]==m) for m in ("qr","short_qr")},
        worker_seconds=manifest["worker_seconds"],budget_seconds=manifest["budget_seconds"],
        deadline_utc=manifest["deadline_utc"],reports=state.get("reports",{}),rows=rows,
        default_changed=False,author_replication_claim=False)
    save(OUT/"terminal-decision.json",decision)
    lines=["# Independent R 24-hour campaign result","",f"Status: {reason}.","",
        f"Charged R worker time: {manifest['worker_seconds']:.3f}/{manifest['budget_seconds']} seconds.",
        f"Full five-dimension study complete: {decision['study_complete']}.","",
        "| Phase | d | Data seed | Arm | Complete | Practical | Heuristic | Literal |",
        "|---|---:|---:|---|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['phase']} | {row['dimension']} | {row['data_seed']} | {row['method']} | "
            f"{row['n']}/{row['planned']} | {row['practical']} | {row['heuristic']} | {row['literal']} |")
    lines += ["", "| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |",
        "|---|---|---|---|---|---|",
        f"| Preserve both frozen reconstructions and all failures | All-dimension screen: {decision['method_screen']} | "
        "Per-cell numerical and conditional heuristic results in linked reports | Unknown author settings; one fixed dataset per dimension; rare weights | "
        "Review completed study against paper tables and the remaining R/TF implementation gaps | Literal author replication, superiority, production or HMC readiness |", "",
        "| Inference status | Finding |","|---|---|",
        "| Hard veto screen | Numerical failures, missing records and heuristic vetoes are retained per cell. |",
        "| Statistically supported ranking | No global ranking; only reported paired pointwise contrasts can support a conditional difference. |",
        "| Descriptive-only differences | Unmatched-cost timings, extreme ratios and raw paper-table differences. |",
        "| Default-readiness | No default changed. QR is not paper Equation15. |",
        "| Next evidence needed | Source-setting resolution and multivariate full-filter R/TF parity, guided by complete study evidence. |", "",
        "Reports preserve intervals, conditional errors, log-ratio underflow counts and exact provenance.",
        "The strongest alternative explanation is data dependence or missed rare large weights. A new",
        "independent-data replication could overturn a fixed-data comparison. Literal equivalence to",
        "unavailable author settings/data remains the weakest part of any replication claim.", ""]
    for phase,path in state.get("reports",{}).items():
        lines.append(f"- {phase}: {Path(path).parent.relative_to(OUT)}/result.md")
    (OUT/"result.md").write_text("\n".join(lines)+"\n")
    CHECKPOINT.write_text("# Active independent R iAPF checkpoint\n\n"
        f"24-hour campaign status: {reason}.\n"
        "Authoritative result: docs/plans/artifacts/iapf-r-24hour-campaign-20260921-01/result.md.\n"
        f"R worker use {manifest['worker_seconds']:.3f}/{manifest['budget_seconds']} seconds.\n"
        f"Five-dimension1000-replica study complete: {decision['study_complete']}.\n"
        f"All-dimension reference screens: {decision['method_screen']}. No default changed.\n"
        "Next: inspect terminal decisions and conditional uncertainty before choosing further repairs;\n"
        "remaining source settings and multivariate R/TF parity remain separate gaps. QR differs from\n"
        "Equation15. No LEDH/KDM/GPU/HMC validity claim follows. Preserve prior attempts and budgets.\n")
    text=MASTER.read_text()
    marker="<!-- IAPF_24H_AUTOMATIC_STATUS -->"
    end="<!-- END_IAPF_24H_AUTOMATIC_STATUS -->"
    block=(f"{marker}\nLatest unattended campaign status: {reason}. "
        f"Five-dimension study complete: {decision['study_complete']}. "
        f"All-dimension reference screens: {decision['method_screen']}. "
        "No algorithm default changed. "
        "[Terminal report](artifacts/iapf-r-24hour-campaign-20260921-01/result.md).\n"+end+"\n\n")
    if marker in text:
        start=text.index(marker);stop=text.index(end,start)+len(end)
        text=text[:start]+block.rstrip()+text[stop:]
    else:
        split=text.index("\n\n")+2;text=text[:split]+block+text[split:]
    MASTER.write_text(text)
    state.update(status=reason,terminal_decision=str(OUT/"terminal-decision.json"))
    heartbeat(state)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait-for",default="phase02-controller-64")
    args=parser.parse_args()
    state=dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
               status="waiting for current frozen phase",wait_for=args.wait_for,reports={})
    previous=OUT/"phase01-controller-pending/analysis-v1/summary.json"
    if previous.exists():
        state["reports"]["phase01-controller-pending"]=str(previous)
    heartbeat(state)
    try:
        while True:
            manifest=read(OUT/"manifest.json")
            status=manifest["phases"].get(args.wait_for,{}).get("status","")
            if not manifest.get("inflight") and status not in ("", "prepared", "running"):
                break
            state["observed_worker_seconds"]=manifest["worker_seconds"]
            heartbeat(state)
            time.sleep(20)
        phase2=analyze(OUT/"phase02.json",state)
        if not phase2["complete_coverage"] or status in ("budget boundary","validity boundary"):
            terminal(state,"continuation boundary after phase2: "+status)
            return
        run([sys.executable,str(DRIVER),str(OUT/"phase03.json")],OUT/"phase03-supervisor.log",state,
            "executing frozen first-study1000 phase")
        study=analyze(OUT/"phase03.json",state)
        manifest=read(OUT/"manifest.json")
        phase_status=manifest["phases"]["phase03-first-study-1000"]["status"]
        terminal(state,"frozen campaign completed" if phase_status=="workers finished; analysis pending" and
                 study["complete_coverage"] else "incomplete frozen campaign: "+phase_status)
    except Exception as error:
        state["error"]=repr(error)
        terminal(state,"supervisor needs localized repair: "+repr(error))
        raise


if __name__=="__main__":
    main()
