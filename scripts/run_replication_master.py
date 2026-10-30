"""Authoritative bounded dispatcher for the active iAPF/MDPF repair sequence.

Reference CPU diagnostics are explicit. Scientific replication is never inferred
from process success. Run status/validate without launching numerical work.
"""
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "docs/plans/artifacts/iapf-mdpf-whole-repair-20260928-01"
MASTER = "docs/plans/iapf-kdm-replication-master-program-2026-09-24.md"
PLAN = "docs/plans/iapf-mdpf-whole-repair-plan-20260928.md"
SOURCES = ["docs/benchmarks/diagnose_iapf_finite_repair_reference.R",
           "docs/benchmarks/reference_iapf_paper.R",
           "docs/benchmarks/reference_iapf_scale_aware_fit.R"]
PHASES = [
    dict(id="iapf_controls", stage="R2", resource="cpu", limit=300, depends=[],
         script=SOURCES[0], argument="controls", implemented=True),
    dict(id="iapf_pilot", stage="R2", resource="cpu", limit=900,
         depends=["iapf_controls"], script=SOURCES[0], argument="pilot", implemented=True),
    dict(id="mdpf_saved_screen", stage="R3a", resource="cpu", limit=300, max_attempts=3, depends=[],
         script="docs/benchmarks/diagnose_mdpf_saved_reference.py", implemented=True,
         interpreter=str(ROOT/".localresources/envs/mdpf-reference-20260924/bin/python")),
    dict(id="iapf_value_fixture_v2", stage="R4a", resource="cpu", limit=60,
         depends=["iapf_controls","iapf_pilot"], argument="export", implemented=True,
         script="docs/benchmarks/export_iapf_paper_value_fixture.R",
         specification="value-port-spec.md"),
    *[dict(id="iapf_value_"+name, stage="R4a", resource=resource, limit=limit,
         depends=[dependency], implemented=True, max_attempts=3 if name=="cpu" else 2,
         script="docs/benchmarks/diagnose_iapf_paper_value_port.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         mode=mode, fixture_phase="iapf_value_fixture_v2", specification="value-port-spec.md",
         backend="TensorFlow FP64 paper-target candidate value parity",
         sources=["bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py",
                  "bayesfilter/score_study/fitted_twist_tf.py",
                  "bayesfilter/score_study/gaussian_tf.py",
                  "bayesfilter/score_study/iapf_adapter.py",
                  "bayesfilter/score_study/iapf_fit_tf.py",
                  "bayesfilter/score_study/conditional_means_tf.py",
                  "bayesfilter/score_study/complete_data_score_tf.py",
                  "bayesfilter/score_study/iapf_scope.py"])
       for name,resource,limit,dependency,mode in [
         ("cpu","cpu",120,"iapf_value_fixture_v2","cpu"),
         ("gpu_graph","gpu",180,"iapf_value_cpu","gpu-graph"),
         ("gpu_xla","gpu",360,"iapf_value_gpu_graph","gpu-xla")]],
    dict(id="legacy_graph_localization", stage="R4b", resource="cpu", limit=60,
         depends=["iapf_value_cpu"], implemented=True,
         script="docs/benchmarks/diagnose_iapf_legacy_graph.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         specification="legacy-graph-spec.md",
         backend="CPU FP64 graph/eager debugging exception; not replication",
         sources=["bayesfilter/score_study/fitted_twist_tf.py",
                  "bayesfilter/score_study/gaussian_tf.py",
                  "bayesfilter/score_study/conditional_means_tf.py",
                  "bayesfilter/score_study/complete_data_score_tf.py"]),
    dict(id="legacy_optimizer_localization", stage="R4b", resource="cpu", limit=60,
         depends=["legacy_graph_localization"], implemented=True,
         script="docs/benchmarks/diagnose_iapf_legacy_graph.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         specification="legacy-graph-spec.md",
         backend="CPU FP64 graph optimizer localization; no runtime default change",
         sources=["bayesfilter/score_study/fitted_twist_tf.py",
                  "bayesfilter/score_study/gaussian_tf.py",
                  "bayesfilter/score_study/conditional_means_tf.py",
                  "bayesfilter/score_study/complete_data_score_tf.py"]),
    dict(id="legacy_optimizer_isolated", stage="R4b", resource="cpu", limit=90,
         depends=["legacy_optimizer_localization"], implemented=True,
         script="docs/benchmarks/diagnose_iapf_optimizer_isolation.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         specification="legacy-graph-spec.md",
         backend="Fresh-process CPU graph optimizer diagnostic only",
         sources=["bayesfilter/score_study/fitted_twist_tf.py",
                  "bayesfilter/score_study/gaussian_tf.py",
                  "bayesfilter/score_study/conditional_means_tf.py",
                  "bayesfilter/score_study/complete_data_score_tf.py"]),
    dict(id="iapf_finite_fixture",stage="R4c",resource="cpu",limit=90,max_attempts=3,
         depends=["iapf_controls"],implemented=True,argument="export",
         script="docs/benchmarks/export_iapf_finite_fixture.R",
         specification="finite-port-spec.md", seeds="9286001-9286003, 9286100, 9286200+filter_call; arrays preserved"),
    *[dict(id="iapf_finite_"+name, stage="R4c", resource=resource, limit=limit,
         depends=[dependency], implemented=True, max_attempts=3,
         script="docs/benchmarks/diagnose_iapf_finite_port.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         mode=mode, fixture_phase="iapf_finite_fixture", specification="finite-port-spec.md",
         backend="TensorFlow FP64 finite iAPF reconstruction parity",
         seeds="R fixture 9286001-9286003,9286100,9286200+call; paired preserved arrays",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"])
       for name,resource,limit,dependency,mode in [
         ("cpu","cpu",240,"iapf_finite_fixture","cpu"),
         ("gpu_graph","gpu",300,"iapf_finite_cpu","gpu-graph"),
         ("gpu_xla","gpu",900,"iapf_finite_gpu_graph","gpu-xla")]],
    dict(id="iapf_paper_adaptive_fixture",stage="R4c",resource="cpu",limit=120,max_attempts=2,
         depends=["iapf_finite_gpu_xla"],implemented=True,argument="paper",
         script="docs/benchmarks/export_iapf_finite_fixture.R",
         specification="finite-port-spec.md",seeds="9286300,9286400+filter_call"),
    dict(id="iapf_paper_adaptive_gpu_xla",stage="R4c",resource="gpu",limit=600,max_attempts=1,
         depends=["iapf_paper_adaptive_fixture"],implemented=True,mode="gpu-xla",
         script="docs/benchmarks/diagnose_iapf_finite_port.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_paper_adaptive_fixture",specification="finite-port-spec.md",
         backend="TensorFlow FP64 paper-horizon adaptive finite-reconstruction diagnostic",
         seeds="R reference data9286300, entropy9286400+call; preserved arrays",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"]),
    dict(id="iapf_finite_divergence",stage="R4c",resource="gpu",limit=180,max_attempts=1,
         depends=["iapf_paper_adaptive_fixture","iapf_finite_gpu_xla"],implemented=True,
         script="docs/benchmarks/diagnose_iapf_finite_divergence.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_paper_adaptive_fixture",specification="finite-port-spec.md",
         backend="CPU/GPU graph/XLA deterministic reference replay; no promotion",
         seeds="Saved call2 times55/54/53; no new random draws",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"]),
    dict(id="iapf_paper_adaptive_fixture_v2",stage="R4c",resource="cpu",limit=120,max_attempts=1,
         depends=["iapf_finite_divergence"],implemented=True,argument="paper",
         script="docs/benchmarks/export_iapf_finite_fixture.R",
         specification="finite-port-spec.md",seeds="9286001-3,9286300,9286400+call; exact float64 roundtrip"),
    dict(id="iapf_paper_adaptive_gpu_xla_v2",stage="R4c",resource="gpu",limit=600,max_attempts=1,
         depends=["iapf_paper_adaptive_fixture_v2"],implemented=True,mode="gpu-xla",
         script="docs/benchmarks/diagnose_iapf_finite_port.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_paper_adaptive_fixture_v2",specification="finite-port-spec.md",
         backend="TensorFlow FP64 exact-fixture adaptive reconstruction diagnostic",
         seeds="9286001-3,data9286300,entropy9286400+call; preserved roundtrip-exact matrices",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"]),
    dict(id="iapf_finite_impact",stage="R4c",resource="gpu",limit=600,max_attempts=1,
         depends=["iapf_paper_adaptive_fixture_v2"],implemented=True,
         script="docs/benchmarks/diagnose_iapf_finite_impact.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_paper_adaptive_fixture_v2",specification="finite-port-spec.md",
         backend="GPU/XLA sensitivity and actual consumer diagnostic; no parity promotion",
         seeds="Exact fixture data9286300 entropy9286400+call; deterministic perturbations",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"]),
    dict(id="iapf_vmmin_fixture",stage="R4d",resource="cpu",limit=30,max_attempts=1,
         depends=["iapf_finite_impact"],implemented=True,argument="controls",
         script="docs/benchmarks/export_iapf_vmmin_controls.R",
         specification="r4d-finite-semantics-spec.md",seeds="N/A: seven deterministic analytic controls"),
    *[dict(id="iapf_vmmin_"+name,stage="R4d",resource=resource,limit=limit,max_attempts=2,
         depends=[dependency],implemented=True,mode=mode,
         script="docs/benchmarks/diagnose_iapf_vmmin_semantics.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_vmmin_fixture",specification="r4d-finite-semantics-spec.md",
         backend="Shared TF finite optimizer semantic diagnostic; not replication",
         seeds="Deterministic controls and exact saved d5 T100 fixture",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"])
      for name,mode,resource,limit,dependency in [
          ("cpu","cpu","cpu",90,"iapf_vmmin_fixture"),
          ("gpu_graph","gpu-graph","gpu",300,"iapf_vmmin_cpu"),
          ("gpu_xla","gpu-xla","gpu",300,"iapf_vmmin_gpu_graph")]],
    dict(id="iapf_vmmin_gpu_xla_regression",stage="R4d",resource="gpu",limit=300,max_attempts=1,
         depends=["iapf_vmmin_gpu_xla"],implemented=True,mode="gpu-xla",
         script="docs/benchmarks/diagnose_iapf_vmmin_semantics.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_vmmin_fixture",specification="r4d-finite-semantics-spec.md",
         backend="Semantic and observability regression after call-chain state repair",
         seeds="N/A: deterministic controls; exact preserved paper arrays",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"]),
    *[dict(id="iapf_repaired_small_"+name,stage="R4d",resource=resource,limit=limit,max_attempts=2,
         depends=[dependency],implemented=True,mode=mode,
         script="docs/benchmarks/diagnose_iapf_finite_port.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_finite_fixture",specification="r4d-finite-semantics-spec.md",
         backend="Repaired shared fitter/backward/controller regression",
         seeds="Unchanged exact small fixture; recorded saved arrays",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"])
      for name,mode,resource,limit,dependency in [
          ("cpu","cpu","cpu",90,"iapf_vmmin_gpu_xla_regression"),
          ("gpu_xla","gpu-xla","gpu",300,"iapf_repaired_small_cpu")]],
    dict(id="iapf_repaired_paper_gpu_xla",stage="R4d",resource="gpu",limit=600,max_attempts=1,
         depends=["iapf_repaired_small_gpu_xla"],implemented=True,mode="gpu-xla",
         script="docs/benchmarks/diagnose_iapf_finite_port.py",
         interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python",
         fixture_phase="iapf_paper_adaptive_fixture_v2",specification="r4d-finite-semantics-spec.md",
         backend="Unchanged paper-horizon finite parameter gate after semantic repair",
         seeds="Exact d5 T100 fixture data9286300 entropy9286400+call",
         sources=["bayesfilter/score_study/iapf_finite_tf.py",
                  "bayesfilter/score_study/iapf_paper_tf.py",
                  "bayesfilter/runtime/gpu_memory_policy.py"]),
    dict(id="iapf_finite_numerics", stage="R4d", resource="gpu", implemented=False,
         gap="R4d source-semantic controls pass but unchanged paper-horizon parameter parity still fails; freeze a fresh paired downstream/distributional evidence contract and recover or label the original recipe"),
    dict(id="mdpf_density_comparators", stage="R3b", resource="gpu", implemented=False,
         gap="Implement normalized UKF/BPF density comparisons with matched source initialization, circular observation likelihood and conditional strata"),
    dict(id="mdpf_appendix_b", stage="R3c", resource="gpu", implemented=False,
         gap="Recover Appendix B coefficients and initialization or freeze an explicitly labeled reconstruction; implement actual linear-Gaussian/bimodal learning experiment"),
    dict(id="mdpf_tf_gpu_port", stage="R4e", resource="gpu", implemented=False,
         gap="Implement batch-native TF/TFP GPU/XLA IWSG learning and filtering consumers with author-reference parity"),
    dict(id="iapf_scoped_replication", stage="R5a", resource="gpu", implemented=False,
         gap="Freeze finite recipe, measure costs, implement Kalman/BPF/fully adapted comparisons and independent uncertainty analysis over all declared paper cells"),
    dict(id="mdpf_scoped_replication", stage="R5b", resource="gpu", implemented=False,
         gap="Refinement, independent training seeds and declared comparators; full repeated suite exceeds current budget at measured source cost"),
    dict(id="terminal_review", stage="R6", resource="cpu", implemented=False,
         gap="Both scoped replications and port evidence remain incomplete"),
]

def stamp():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def proc_identity(pid):
    try:
        # /proc field 22 is starttime; comm may contain spaces.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None

def alive(pid, identity):
    return bool(identity and proc_identity(pid) == identity)

def load_attempts(root=CAMPAIGN):
    return [(p, json.loads(p.read_text())) for p in sorted(root.glob("attempt*/manifest.json"))]

def charge(m):
    if m["status"] == "RUNNING":
        return min(m["timeout_seconds"], max(0, time.time() - m["started_epoch"]))
    return m.get("charged_seconds", m.get("wall_seconds", 0))

def snapshot(root=CAMPAIGN):
    opening = json.loads((root/"opening-budget.json").read_text())
    attempts = load_attempts(root)
    spending = {resource: opening[f"opening_{resource}_seconds"] +
                sum(charge(m) for _, m in attempts if m["resource"] == resource)
                for resource in ("cpu", "gpu")}
    running = [dict(phase=m["phase"], manifest=str(p),
                    worker_alive=alive(m.get("pid"), m.get("process_identity")),
                    supervisor_alive=alive(m.get("supervisor_pid"), m.get("supervisor_identity")))
               for p,m in attempts if m["status"] == "RUNNING"]
    completed = {m["phase"] for _,m in attempts if m["status"] == "COMPLETE"}
    successful_controls = any(m["phase"] == "iapf_controls" and m["status"]=="COMPLETE"
                             and m.get("decision",{}).get("status")=="PASS" for _,m in attempts)
    phases = []
    eligible = []
    for phase in PHASES:
        identifier=phase["id"]
        previous=[m for _,m in attempts if m["phase"]==identifier]
        used=sum(charge(m) for m in previous)
        state="NOT_IMPLEMENTED"
        if phase["implemented"]:
            state="READY"
            if identifier in completed:state="COMPLETE"
            elif any(x["phase"]==identifier for x in running):state="RUNNING"
            elif "iapf_controls" in phase["depends"] and not successful_controls:state="WAITING_FOR_CONTROLS"
            elif any(dep not in completed for dep in phase["depends"]):state="WAITING_FOR_DEPENDENCY"
            elif len(previous)>=phase.get("max_attempts",2):state="ATTEMPT_LIMIT_REQUIRES_REVIEW"
            elif used>=phase["limit"] or spending[phase["resource"]]>=opening[f'{phase["resource"]}_limit_seconds']:
                state="BUDGET_EXHAUSTED"
            elif len(attempts)>=opening["new_attempt_limit"]:state="TRANCHE_ATTEMPT_LIMIT"
            if state=="READY":eligible.append(identifier)
        phases.append(dict(id=identifier,stage=phase["stage"],state=state,
                           gap=phase.get("gap"),attempts=len(previous),charged_seconds=used))
    value_phases={"iapf_value_cpu","iapf_value_gpu_graph","iapf_value_gpu_xla"}
    passed={m["phase"] for _,m in attempts if m["status"]=="COMPLETE"
            and m.get("decision",{}).get("status")=="PASS"}
    value_passed=value_phases.issubset(passed)
    finite_passed={"iapf_finite_cpu","iapf_finite_gpu_graph","iapf_finite_gpu_xla"}.issubset(passed)
    paper_adaptive_passed="iapf_paper_adaptive_gpu_xla_v2" in passed
    paper_adaptive_failed=any(m["phase"]=="iapf_paper_adaptive_gpu_xla_v2"
        and m.get("decision",{}).get("status")=="FAIL" for _,m in attempts)
    finite_impact_complete="iapf_finite_impact" in passed
    semantics_passed={"iapf_vmmin_cpu","iapf_vmmin_gpu_graph",
                      "iapf_vmmin_gpu_xla_regression"}.issubset(passed)
    repaired_small_passed={"iapf_repaired_small_cpu","iapf_repaired_small_gpu_xla"}.issubset(passed)
    repaired_paper=[m for _,m in attempts if m["phase"]=="iapf_repaired_paper_gpu_xla"]
    repaired_paper_verdict=(repaired_paper[-1].get("decision",{}).get("status","NOT_EVALUATED")
                            if repaired_paper else "INCOMPLETE")
    work=[dict(phase=p["id"],stage=p["stage"],required=p["gap"])
          for p in PHASES if not p["implemented"]]
    return dict(master=MASTER, plan=PLAN, campaign=str(root),
                execution_state="RUNNING" if running else "READY" if eligible else
                    "IMPLEMENTATION_OR_SPECIFICATION_REQUIRED" if work else "NO_READY_PHASE",
                next_required_work=work,
                scientific_outcomes={m["phase"]:m.get("decision",{}) for _,m in attempts
                                     if m["status"]=="COMPLETE"},
                failed_outcomes={m["phase"]:dict(execution_status=m["status"],
                    decision=m.get("decision",{}),manifest=str(p))
                    for p,m in attempts if m["status"] in ("FAILED","TIMED_OUT","INTERRUPTED_EXIT_UNOBSERVED")},
                source_iapf=json.loads((root/"source-iapf.json").read_text())["status"],
                phases=phases, running=running,
                next_phase=eligible[0] if eligible and not running else None,
                resources={r:dict(charged_seconds=spending[r],
                    remaining_seconds=max(0,opening[f"{r}_limit_seconds"]-spending[r]))
                    for r in spending},
                new_attempts=len(attempts),new_attempt_limit=opening["new_attempt_limit"],
                iapf_replication="INCOMPLETE",mdpf_replication="INCOMPLETE",
                paper_target_frozen_twist_value_parity="PASS" if value_passed else "INCOMPLETE",
                finite_fitter_controller_parity="PASS" if finite_passed else "INCOMPLETE",
                paper_horizon_adaptive_parity=("PASS" if paper_adaptive_passed else
                    "FAIL" if paper_adaptive_failed else "INCOMPLETE"),
                finite_parity_scope="recorded small-fixture source snapshots; not full scientific replication",
                finite_solver_source_controls="PASS" if semantics_passed else "INCOMPLETE",
                repaired_fitter_controller_parity="PASS" if repaired_small_passed else "INCOMPLETE",
                repaired_paper_horizon_parity=repaired_paper_verdict,
                repaired_parity_scope="R4d repaired source snapshots; historical R4c failures preserved separately",
                paper_horizon_impact_diagnostic="COMPLETE_NOT_PROMOTION" if finite_impact_complete else "INCOMPLETE",
                gpu_port=("IAPF_SMALL_FINITE_PARITY_PASSED_MDPF_PORT_OPEN"
                          if finite_passed else "PARTIAL_VALUE_PARITY_ONLY" if value_passed
                          else "NOT_VALIDATED"))

def acquire_lock(root):
    stream=(root/"runner.lock").open("a+")
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError("A repair supervisor or inherited worker holds the campaign lock")
    return stream

def recover(root=CAMPAIGN):
    with acquire_lock(root):
        repaired=[]
        for path,m in load_attempts(root):
            if m["status"]!="RUNNING":continue
            if alive(m.get("pid"),m.get("process_identity")):
                raise RuntimeError("Worker is alive; preserve it and inspect its log")
            # No witnessed exit: do not invent success from output presence.
            m.update(status="INTERRUPTED_EXIT_UNOBSERVED",ended_utc=stamp(),
                     charged_seconds=charge(m),returncode=None,
                     recovery="Worker absent; preserve partial evidence and charge bounded elapsed time")
            atomic_json(path,m);repaired.append(str(path))
        return repaired

def execute(phase, root=CAMPAIGN):
    with acquire_lock(root) as lock:
        current=snapshot(root)
        if current["running"]:raise RuntimeError("Reconcile the existing attempt before launching")
        state=next(x for x in current["phases"] if x["id"]==phase["id"])
        if state["state"]!="READY":raise RuntimeError(f'Phase is not ready: {state["state"]}')
        remaining=current["resources"][phase["resource"]]["remaining_seconds"]
        timeout=min(phase["limit"]-state["charged_seconds"],remaining)
        if timeout<1:raise RuntimeError("Insufficient remaining budget")
        output=root/f'attempt{current["new_attempts"]+1:02d}_{phase["id"]}'
        output.mkdir()
        sources=output/"sources";sources.mkdir()
        source_hashes={}
        for relative in dict.fromkeys([*SOURCES,str(Path(__file__).resolve()),phase["script"],*phase.get("sources",[])]):
            source=ROOT/relative;target=sources/source.name
            target.write_bytes(source.read_bytes());source_hashes[relative]=sha(target)
        env=os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1",TF_FORCE_GPU_ALLOW_GROWTH="true",
                   BAYESFILTER_PRELOAD_CUSTOM_OP="0")
        if phase["resource"]=="cpu":env["CUDA_VISIBLE_DEVICES"]="-1"
        else:env.pop("CUDA_VISIBLE_DEVICES",None)
        if "interpreter" in phase:
            command=[phase["interpreter"],"-I","-B",str(sources/Path(phase["script"]).name),
                     "--output",str(output)]
            if "mode" in phase:command+=["--mode",phase["mode"]]
            if "fixture_phase" in phase:
                fixtures=[m for _,m in load_attempts(root) if m["phase"]==phase["fixture_phase"]
                          and m["status"]=="COMPLETE"]
                command+=["--fixture",fixtures[-1]["output_root"]]
        else:
            command=["/usr/bin/Rscript","--vanilla",str(sources/Path(phase["script"]).name),
                     phase["argument"],str(output),str(sources)]
        wrapped=["/usr/bin/timeout","--signal=TERM","--kill-after=5s",f"{timeout}s",*command]
        m=dict(schema="replication_repair_attempt_v1",phase=phase["id"],status="RUNNING",
               resource=phase["resource"],cpu_only=phase["resource"]=="cpu",
               gpu_intentionally_hidden=phase["resource"]=="cpu",
               backend=phase.get("backend",("independent author PyTorch reference analysis" if "interpreter" in phase
                        else "independent R FP64 reference; not BayesFilter runtime")),
               started_utc=stamp(),started_epoch=time.time(),timeout_seconds=timeout,
               command=command,supervised_command=wrapped,cwd=str(ROOT),
               supervisor_pid=os.getpid(),supervisor_identity=proc_identity(os.getpid()),
               plan=PLAN,specification=str(root/phase.get("specification",
                       "mdpf-source-and-screen-spec.md" if "interpreter" in phase else "execution-spec.md")),output_root=str(output),
               git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               source_sha256=source_hashes,seeds=phase.get("seeds",("N/A: fixed deterministic tensors; no random draws" if phase["id"].startswith("legacy_") else "Fixture seeds and preserved arrays in index.csv" if phase["id"].startswith("iapf_value_") else
                      "Saved trial dataset seed 925231; source evaluation effective seed 0; no new stochastic work"
                      if "interpreter" in phase else "See frozen specification, histories.csv and RDS streams")),
               supervisor_python=sys.version,interpreter=command[0],
               environment={k:env.get(k) for k in ["CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","TF_FORCE_GPU_ALLOW_GROWTH"]})
        manifest=output/"manifest.json"
        atomic_json(manifest,m)
        start=time.monotonic()
        with (output/"command.log").open("w") as log:
            child=subprocess.Popen(wrapped,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,pass_fds=(lock.fileno(),))
            m.update(pid=child.pid,process_identity=proc_identity(child.pid))
            atomic_json(manifest,m)
            try:
                rc=child.wait(timeout=timeout+8)
            except BaseException:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                m.update(status="INTERRUPTED",ended_utc=stamp(),returncode=child.returncode,
                         charged_seconds=time.monotonic()-start)
                atomic_json(manifest,m)
                raise
        decision={}
        if (output/"decision.csv").exists():
            with (output/"decision.csv").open() as stream:decision=next(csv.DictReader(stream),{})
        status="COMPLETE" if rc==0 and decision else "TIMED_OUT" if rc in (124,137) else "FAILED"
        m.update(status=status,ended_utc=stamp(),returncode=rc,
                 wall_seconds=time.monotonic()-start,charged_seconds=time.monotonic()-start,
                 decision=decision,output_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob("*")
                    if p.is_file() and p!=manifest})
        atomic_json(manifest,m)
        return {k:m[k] for k in ["phase","status","returncode","charged_seconds","decision","output_root"]}

def validate(root=CAMPAIGN):
    required=set(SOURCES)|{p["script"] for p in PHASES if p["implemented"]}
    missing=[p for p in required if not (ROOT/p).is_file()]
    if missing:raise RuntimeError(f"Missing sources: {missing}")
    for name in ["opening-budget.json","source-iapf.json","execution-spec.md"]:
        if not (root/name).is_file():raise RuntimeError(f"Missing contract: {name}")
    if not Path("/usr/bin/Rscript").is_file():raise RuntimeError("Rscript unavailable")
    for phase in PHASES:
        if not phase["implemented"]:continue
        if "specification" in phase and not (root/phase["specification"]).is_file():
            raise RuntimeError("Missing phase specification")
        for relative in phase.get("sources",[]):
            if not (ROOT/relative).is_file():raise RuntimeError("Missing dependency: "+relative)
        path=ROOT/phase["script"]
        if path.suffix==".py":ast.parse(path.read_text(),filename=str(path))
        if "interpreter" in phase and not Path(phase["interpreter"]).is_file():
            raise RuntimeError("Missing interpreter: "+phase["interpreter"])
    if any(p["id"]=="mdpf_saved_screen" and p["implemented"] for p in PHASES):
        if not (root/"mdpf-source-and-screen-spec.md").is_file():
            raise RuntimeError("Missing MDPF screen specification")
    return dict(status="VALID",launch=False,implemented=[p["id"] for p in PHASES if p["implemented"]],
                warning="Validation is structural; regression and numerical controls are separate")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["status","validate","recover","run-next","run-ready"])
    args=parser.parse_args()
    if args.action=="validate":result=validate()
    elif args.action=="recover":result={"recovered":recover(),"status":snapshot()}
    elif args.action=="status":result=snapshot()
    else:
        validate();done=[]
        while True:
            current=snapshot()
            if not current["next_phase"]:break
            phase=next(p for p in PHASES if p["id"]==current["next_phase"])
            result=execute(phase);done.append(result)
            print(json.dumps(result),flush=True)
            # Failed numerical work needs inspection, not blind retry.
            if args.action=="run-next" or result["status"]!="COMPLETE":break
        result={"executed":done,"status":snapshot()}
    print(json.dumps(result,indent=2))
    return int(any(x["status"]!="COMPLETE" for x in result.get("executed",[])))

if __name__=="__main__":
    sys.exit(main())
