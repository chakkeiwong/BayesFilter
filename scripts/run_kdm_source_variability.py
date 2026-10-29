"""Bounded source-only KDM training-variability calibration supervisor.

Independent author-reference workers only. Does not change either algorithm or
qualify a TensorFlow implementation. See the fixed phase plan for inference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PLAN = "docs/plans/controlled-smc-kdm-source-variability-20261008.md"
RESULT = "docs/plans/controlled-smc-kdm-source-variability-result-20261008.md"
PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
SEEDS = (202610101, 202610111)
CAP = 131400.0
ROUTINE = 240.0
SOURCE_RUN = BASE / "kdm-complete-training-20261007-03"
EVALUATION = BASE / "kdm-development-evaluation-20261008-03/diagnostic"
TRAIN = "docs/benchmarks/reference_kdm_fresh_training_cost.py"
AUDIT = "docs/benchmarks/reference_kdm_author_terminal_audit.py"
EVAL = "docs/benchmarks/reference_kdm_development_evaluation.py"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1",
               LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
               MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    return env


def hashes():
    paths = [Path(__file__), ROOT/PLAN, ROOT/TRAIN, ROOT/AUDIT, ROOT/EVAL,
             ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py"]
    old = ROOT/".localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks"
    paths += [old/"diagnose_replication_training.py", old/"diagnose_replication_stage0.py"]
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}


def charge(out):
    total = ROUTINE
    steps = []
    for path in sorted(out.glob("unit*-*/manifest.json")):
        item = read(path)
        elapsed = (max(0, time.time()-item["started_unix_seconds"])
                   if item["status"] == "RUNNING" else item.get("wall_seconds", 0))
        total += elapsed
        steps.append(dict(path=str(path.parent.relative_to(ROOT)), kind=item["kind"],
                          status=item["status"], elapsed_seconds=elapsed,
                          child_pid=item.get("child_pid"), seed=item["seed"]))
    return total, steps


def run_step(out, unit, seed, kind, limit, reference=None):
    directory = out/("unit%02d-%s" % (unit, kind))
    directory.mkdir()
    diagnostic = directory/"diagnostic"
    if kind in ("cost", "full"):
        command = [PYTHON, "-I", "-B", str(ROOT/TRAIN), "--output", str(diagnostic),
                   "--epochs", "cost" if kind == "cost" else "full",
                   "--constructor-seed-base", str(seed)]
        if reference is not None:
            command += ["--reference", str(reference)]
        result_path = diagnostic/"result.json"
    elif kind == "audit":
        command = [PYTHON, "-I", "-B", str(ROOT/AUDIT), "--run",
                   str(out/("unit%02d-full" % unit)), "--output", str(directory/"result.json")]
        result_path = directory/"result.json"
    elif kind == "evaluation":
        command = [PYTHON, "-I", "-B", str(ROOT/EVAL), "--worker", "--output", str(diagnostic),
                   "--source-run", str(out/("unit%02d-full" % unit)),
                   "--source-audit", str(out/("unit%02d-audit" % unit)/"result.json")]
        result_path = diagnostic/"result.json"
    else:
        raise ValueError(kind)
    env = environment()
    manifest = dict(schema="kdm_source_variability_step_v1", status="RUNNING",
        mode="source", resource="cpu", epoch_mode=kind, kind=kind, seed=seed,
        command=command, cwd=str(ROOT), plan=PLAN, result=RESULT,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        started_utc=stamp(), started_unix_seconds=time.time(), timeout_seconds=limit,
        heartbeat_timeout_seconds=1800 if kind in ("cost","full") else None,
        supervisor_pid=os.getpid(), hashes=hashes(),
        seeds=dict(data=202610071, constructors=[seed+i for i in range(3)],
                   evaluation_data=202610081, evaluation_filter=[202610082,202610083]),
        data_version="fixed source development train5000/validation1000",
        cpu_gpu_status="CPU independent author reference; GPU intentionally hidden",
        environment={k:env.get(k) for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH",
            "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS")})
    if reference is not None:
        manifest.update(reference=str(reference), reference_sha256=sha(reference))
    save(directory/"manifest.json",manifest)
    started = time.monotonic()
    process = None
    code = 1
    try:
        with (directory/"command.log").open("w") as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
            manifest["child_pid"] = process.pid
            save(directory/"manifest.json",manifest)
            while True:
                elapsed = time.monotonic()-started
                remaining = limit-elapsed
                progress = diagnostic/"progress.json"
                last = progress.stat().st_mtime if progress.exists() else manifest["started_unix_seconds"]
                stalled = kind in ("cost","full") and time.time()-last > 1800
                if remaining <= 0 or stalled:
                    manifest["termination_reason"] = "HEARTBEAT_STALL" if stalled else "STEP_TIME_LIMIT"
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                    code = 124
                    break
                try:
                    code = process.wait(timeout=min(30,remaining))
                    break
                except subprocess.TimeoutExpired:
                    manifest.update(elapsed_running_seconds=time.monotonic()-started,
                                    last_progress_unix_seconds=last)
                    save(directory/"manifest.json",manifest)
    except BaseException:
        manifest["error"] = traceback.format_exc()
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
    manifest.update(status="COMPLETED" if code == 0 else "FAILED",returncode=code,
                    wall_seconds=time.monotonic()-started,ended_utc=stamp())
    try:
        if result_path.exists():
            payload = read(result_path)
            manifest.update(result_path=str(result_path),result_sha256=sha(result_path))
            expected = payload.get("verdict",payload.get("status",payload.get("decision",{}).get("status")))
            if expected != "PASS":
                manifest.update(status="FAILED",termination_reason="FAILED_RESULT",returncode=1)
        else:
            manifest.update(status="FAILED",termination_reason="MISSING_RESULT",returncode=1)
    except Exception:
        manifest.update(status="FAILED",termination_reason="INVALID_RESULT",returncode=1,
                        result_error=traceback.format_exc())
    save(directory/"manifest.json",manifest)
    if manifest["status"] != "COMPLETED":
        raise RuntimeError(str(directory)+": "+manifest.get("termination_reason","worker failed"))
    return result_path


def verify_cost(path, seed, baseline, other=None):
    current = read(path)
    if current["constructor_seed_base"] != seed or current["data_seed"] != 202610071:
        raise ValueError("Wrong calibration seed")
    if current["source_training_configs"] != baseline["source_training_configs"]:
        raise ValueError("Source hyperparameters changed")
    for split, meta in current["datasets"].items():
        if sha(meta["path"]) != meta["sha256"] or meta["sha256"] != baseline["datasets"][split]["sha256"]:
            raise ValueError("Frozen data mismatch: "+split)
    for name, meta in current["fixtures"].items():
        if sha(meta["path"]) != meta["sha256"] or meta["sha256"] == baseline["fixtures"][name]["sha256"]:
            raise ValueError("Constructor was not independently generated: "+name)
        if other is not None and meta["sha256"] == other["fixtures"][name]["sha256"]:
            raise ValueError("Duplicate new constructor: "+name)


def verify_evaluation(path):
    current, previous = read(path), read(EVALUATION/"result.json")
    if current["dataset_sha256"] != previous["dataset_sha256"]:
        control = current.get("data_value_control") or {}
        if (control.get("exact_tensor_equality") is not True
                or control.get("reference_file_sha256") != previous["dataset_sha256"]
                or control.get("reference_path") != str(EVALUATION/"evaluation-dataset.pt")
                or not control.get("tensor_sha256")
                or control.get("tensor_sha256") != control.get("reference_tensor_sha256")):
            raise ValueError("Development evaluation data changed or unchecked")
    checks = 0
    for variant in ("mdpf","am_dpf"):
        for seed in (202610082,202610083):
            filename = variant+"-tf_trained-"+str(seed)+".json"
            left, right = read(path.parent/filename), read(EVALUATION/filename)
            for key in ("metrics","conditional","per_step_nll","per_step_squared_error"):
                if left[key] != right[key]:
                    raise ValueError("Fixed TF evaluator replay changed: "+filename+"/"+key)
                checks += 1
    return checks


def execute_unit(out, unit, seed, reference):
    try:
        run_step(out,unit,seed,"full",64800,reference)
        run_step(out,unit,seed,"audit",120,reference)
        evaluation = run_step(out,unit,seed,"evaluation",300,reference)
        controls = verify_evaluation(evaluation)
        return dict(unit=unit,seed=seed,status="COMPLETED",evaluation=str(evaluation),
                    evaluation_sha256=sha(evaluation),fixed_tf_replay_controls=controls)
    except Exception:
        return dict(unit=unit,seed=seed,status="FAILED",error=traceback.format_exc())


def calibration(completed):
    records = [("original",202610072,read(EVALUATION/"result.json"))]
    records += [("unit%02d"%r["unit"],r["seed"],read(r["evaluation"])) for r in completed]
    rows, planning = [], []
    fixed = {}
    for variant in ("mdpf","am_dpf"):
        for label,seed,report in records:
            cells = [r for r in report["comparisons"] if r["variant"] == variant]
            if len(cells) != 2:
                raise ValueError("Missing filter-seed cells")
            value = {metric:statistics.mean(c["source"][metric] for c in cells)
                     for metric in ("nll","rmse")}
            rows.append(dict(unit=label,seed=seed,variant=variant,**value))
            fixed[variant] = {metric:statistics.mean(c["tf_trained"][metric] for c in cells)
                              for metric in ("nll","rmse")}
        variant_rows = [r for r in rows if r["variant"] == variant]
        for metric in ("nll","log_rmse"):
            values = [r["nll"] if metric == "nll" else math.log(r["rmse"]) for r in variant_rows]
            sd = statistics.stdev(values)
            delta = .01 if metric == "nll" else math.log(1.01)
            estimate = max(2,math.ceil(2*(1.6448536269514722+.8416212335729143)**2*sd**2/delta**2))
            planning.append(dict(variant=variant,metric=metric,n_source=len(values),
                mean=statistics.mean(values),standard_deviation=sd,
                minimum=min(values),maximum=max(values),
                approximate_independent_units_per_arm=estimate,
                inference="Descriptive planning only; three source units and unknown TF variance"))
    return dict(source_units=rows,fixed_tf= fixed,planning=planning,training_units_per_variant=3,
                statistical_ranking="NOT_ESTABLISHED",qualification="INCOMPLETE")


def refresh():
    import run_controlled_smc_kdm as master
    current = master.status()
    master.shared.atomic_json(master.CAMPAIGN/"status.json",current)
    master.refresh_checkpoint(current)


def finish(out, manifest, units, error=None):
    total, steps = charge(out)
    okay = error is None and len(units) == 2 and all(row["status"] == "COMPLETED" for row in units)
    calibrated = None
    if okay:
        try:
            calibrated = calibration(units)
        except Exception:
            okay = False
            error = traceback.format_exc()
    summary = dict(schema="kdm_source_variability_summary_v1",
        decision=dict(status="PASS" if okay else "FAIL",
                      scope="SOURCE_ONLY_TRAINING_VARIABILITY_CALIBRATION",
                      replication="INCOMPLETE",
                      scientific_verdict="SOURCE_VARIABILITY_CALIBRATED_QUALIFICATION_OPEN" if okay
                                         else "CALIBRATION_INCOMPLETE_INSPECT_FAILURE"),
        units=units,steps=steps,charged_cpu_seconds=total,plan_file=PLAN,result_file=RESULT,
        error=error,nonclaims="No noninferiority, numerical acceptance, ranking or replication completion.")
    if okay:
        summary["calibration"] = calibrated
    save(out/"summary.json",summary)
    manifest.update(status="COMPLETED" if okay else "FAILED",ended_utc=stamp(),
        wall_seconds=time.time()-manifest["started_unix_seconds"],
        charged_seconds=total,resource_wall_seconds={"cpu":total-ROUTINE,"gpu":0.0},
        units=units,summary=str(out/"summary.json"),summary_sha256=sha(out/"summary.json"))
    save(out/"manifest.json",manifest)
    # This is the only phase charge; active status removes its live accounting
    # once the output root appears in the terminal reference ledger.
    ledger_path = BASE/"reference-diagnostics.json"
    ledger = read(ledger_path)
    record_id = out.name+"_cpu"
    if any(r["id"] == record_id for r in ledger):
        raise RuntimeError("Phase already charged")
    next_action = ("Review source-only variability and freeze an adequately powered untouched "
        "qualification design; preserve float32 screens and controlled-SMC fitting gaps." if okay else
        "Inspect the preserved calibration failure; repair only the harness within the remaining "
        "campaign budget. Do not replace an unfavorable source seed.")
    ledger.append(dict(id=record_id,resource="cpu",charged_seconds=total,
        summary=str((out/"summary.json").relative_to(ROOT)),summary_sha256=sha(out/"summary.json"),
        result_file=RESULT,plan_file=PLAN,stage=summary["decision"]["scientific_verdict"],
        next_action=next_action,after_attempt=160,
        executions=[dict(path=str(out.relative_to(ROOT)),resource="cpu",wall_seconds=total-ROUTINE)]))
    save(ledger_path,ledger)
    lines = ["# KDM source-training variability result — 8 October 2026","",
             "Phase "+manifest["status"]+". Both replications remain incomplete.",
             "","Plan: ["+PLAN.split("/")[-1]+"]("+PLAN.split("/")[-1]+").",
             "Summary: [structured result]("+str((out/"summary.json").relative_to(ROOT/"docs/plans"))+").",
             "","The reference uses unchanged author training, two predeclared new seed units,",
             "the same training/validation data and the same development author evaluator.",
             "Each complete unit must pass constructor/data checks, the independent terminal",
             "audit and exact replay of the fixed TF checkpoint evaluation.",""]
    if okay:
        info=summary["calibration"]
        lines += ["| Source unit | Stage seed base | Variant | NLL | RMSE |",
                  "|---|---:|---|---:|---:|"]
        lines += ["| {unit} | {seed} | {variant} | {nll:.9g} | {rmse:.9g} |".format(**r)
                  for r in info["source_units"]]
        lines += ["","Each value averages two filter seeds within one trained unit.",
                  "The independently trained sample size is three per variant. Conditional",
                  "metrics and per-step values remain in the linked unit evaluations.","",
                  "| Variant / metric | Source SD | Planning units per arm |",
                  "|---|---:|---:|"]
        lines += ["| "+r["variant"]+"/"+r["metric"]+" | "+format(r["standard_deviation"],".6g")+
                  " | "+str(r["approximate_independent_units_per_arm"])+" |" for r in info["planning"]]
        lines += ["","The sample-size figures use the predeclared equal-variance normal",
                  "approximation and are uncertain planning estimates, not validated power",
                  "calculations. TF training variance and paired covariance remain unknown."]
    else:
        lines += ["The phase is incomplete. Inspect the unit errors and preserved command logs;",
                  "failed or unexecuted units are not successful baseline outcomes."]
    lines += ["","| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
              "|---|---|---|---|---|---|",
              "| "+("Calibration complete" if okay else "Calibration incomplete")+
              " | "+("Two full audited source units evaluated" if okay else "Not satisfied")+
              " | Float32 screen remains open | Training-level precision and TF variability | "+
              next_action+" | Replication or noninferiority |","",
              "| Inference class | Status |","|---|---|",
              "| Hard veto screen | Preserve all failed numerical checks; see per-unit validity/audit status |",
              "| Statistically supported ranking | None |",
              "| Descriptive-only differences | Source training variation and fixed TF comparison |",
              "| Default-readiness | Not established |",
              "| Next evidence needed | Reviewed untouched trained-unit qualification and numerical accuracy checks |","",
              "Post-run review: the strongest alternative explanation is development-data-specific",
              "or arithmetic-dependent variability. Three source units provide an unstable",
              "variance estimate; more independent training units or new data could change it.",
              "This phase does not explain the existing TF gap causally or repair the author.",
              "", "Charged CPU seconds: "+format(total,".9f")+
              " (summed child wall time plus 240 routine seconds). GPU charge: zero.",
              "Exact environments, seeds, code hashes, commands and wall times are in the manifests.",
              "The live checkpoint carries the remaining allocation.",""]
    (ROOT/RESULT).write_text("\n".join(lines))
    master_path=ROOT/"docs/plans/controlled-smc-kdm-replication-master-program-20260928.md"
    text=master_path.read_text()
    running="Source-only variability calibration is now running under the bounded phase plan."
    terminal="Source-only variability calibration "+manifest["status"].lower()+             "; see the [phase result](controlled-smc-kdm-source-variability-result-20261008.md)."
    if running in text:
        master_path.write_text(text.replace(running,terminal,1))
    refresh()
    print(json.dumps(dict(status=manifest["status"],charged_cpu_seconds=total,
                          summary=str(out/"summary.json"))),flush=True)
    return 0 if okay else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.parent!=BASE or not out.name.startswith("kdm-source-variability-20261008-"):
        raise ValueError("Use the versioned phase output root")
    if list(BASE.glob("kdm-source-variability-20261008-*/manifest.json")):
        raise RuntimeError("Fixed two-unit phase already launched; inspect preserved state")
    import run_controlled_smc_kdm as master
    current=master.status()
    available=min(current["remaining_seconds"]["cpu"],current["tranche_remaining_seconds"]["cpu"])
    if available<CAP or current["running"]:
        raise RuntimeError("Insufficient phase budget or an existing campaign run")
    original_manifest=read(SOURCE_RUN/"manifest.json")
    baseline_path=Path(original_manifest["reference"])
    baseline=read(baseline_path)
    if sha(baseline_path)!=original_manifest["reference_sha256"]:
        raise ValueError("Original constructor/data reference changed")
    out.mkdir()
    manifest=dict(schema="kdm_source_variability_campaign_v1",status="RUNNING",
        resource="cpu",mode="source_only_variability",plan=PLAN,result=RESULT,
        command=[sys.executable]+sys.argv,cwd=str(ROOT),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        hashes=hashes(),started_utc=stamp(),started_unix_seconds=time.time(),
        supervisor_pid=os.getpid(),phase_cpu_cap_seconds=CAP,phase_gpu_cap_seconds=0,
        routine_cpu_seconds=ROUTINE,seed_bases=list(SEEDS),units=[],
        opening_active_cpu_seconds=available,replication="INCOMPLETE")
    save(out/"manifest.json",manifest)
    (ROOT/RESULT).write_text(
        "# KDM source-training variability — running\n\n"
        "The fixed two-unit source-only calibration is running. Both replications remain incomplete.\n\n"
        "Plan: [source variability plan](controlled-smc-kdm-source-variability-20261008.md).\n"
        "Stage seed bases: 202610101 and 202610111; frozen data seed 202610071.\n"
        "Cap: 131400 CPU seconds including all workers and routine checks; GPU use zero.\n"
        "The supervisor audits and evaluates each full unit, preserves errors and refreshes the live checkpoint.\n"
        "Exact commands, code hashes, environment and child progress are in " +
        str(out.relative_to(ROOT)) + "/manifest.json.\n")
    master_path=ROOT/"docs/plans/controlled-smc-kdm-replication-master-program-20260928.md"
    text=master_path.read_text()
    anchor="Updated 8 October 2026. **ACTIVE; BOTH REPLICATIONS INCOMPLETE.**"
    master_path.write_text(text.replace(anchor,anchor+"\nSource-only variability calibration is now running under the bounded phase plan.",1))
    refresh()
    units=[]
    try:
        references=[]
        for unit,seed in enumerate(SEEDS,1):
            path=run_step(out,unit,seed,"cost",300)
            verify_cost(path,seed,baseline,read(references[0]) if references else None)
            references.append(path)
            print(json.dumps(dict(unit=unit,stage="COST_REFERENCE_VERIFIED",seed=seed)),flush=True)
            refresh()
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(execute_unit,out,i,seed,ref):i
                     for i,(seed,ref) in enumerate(zip(SEEDS,references),1)}
            while pending:
                for future in list(pending):
                    if future.done():
                        record=future.result()
                        units.append(record)
                        del pending[future]
                        print(json.dumps(record),flush=True)
                total,steps=charge(out)
                manifest.update(units=units,charged_seconds=total,steps=steps,
                                last_update_utc=stamp())
                save(out/"manifest.json",manifest)
                refresh()
                if pending:
                    time.sleep(30)
        units.sort(key=lambda r:r["unit"])
        return finish(out,manifest,units)
    except Exception:
        return finish(out,manifest,units,error=traceback.format_exc())


if __name__=="__main__":
    raise SystemExit(main())
