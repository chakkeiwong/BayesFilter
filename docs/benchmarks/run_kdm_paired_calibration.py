"""Bounded development calibration of repaired KDM training variability.

Standard-library orchestration only. Every numerical worker is an existing
checked TF GPU/XLA route or an explicitly independent author CPU reference.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE = BASE / "paired-calibration-20261011"
SOURCE_UNITS = BASE / "kdm-source-variability-20261008-01"
PILOT = BASE / "repaired-training-20261011/training-pilot-02"
PROOF = BASE / "stage-refresh-20261011/stage-refresh-01/verification/diagnostic/result.json"
PLAN = "docs/plans/controlled-smc-kdm-paired-calibration-20261011.md"
RESULT = "docs/plans/controlled-smc-kdm-paired-calibration-result-20261011.md"
TF = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
SOURCE = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
GPU = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
CAP = {"gpu": 30000., "cpu": 1800.}
SEEDS = (1100000000, 1200000000)
DRIVER = ROOT / "docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py"
EVALUATOR = ROOT / "docs/benchmarks/reference_kdm_development_evaluation.py"
AUDITOR = ROOT / "docs/benchmarks/reference_kdm_training_terminal_audit.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    from run_replication_gap_localization import campaign
    from reference_kdm_development_evaluation import source_completion_valid
    balance = campaign()
    require(not balance["running"], "Another replication campaign is running")
    prior = sorted(PHASE.glob("calibration-*/manifest.json"))
    require(len(prior) < 2, "Two-attempt phase cap exhausted")
    require(not any(read(p)["status"] == "RUNNING" for p in prior), "Calibration already running")
    spent = {r: sum(read(p)["resource_wall_seconds"][r] for p in prior) for r in CAP}
    remaining = {r: min(CAP[r]-spent[r], balance["remaining_seconds"][r],
                        balance["tranche_remaining_seconds"][r]) for r in CAP}
    require(remaining["gpu"] >= 19000 and remaining["cpu"] >= 300,
            "Insufficient phase/campaign allowance for two complete pairs")
    runtime = [ROOT/p for p in read(PILOT/"manifest.json")["source_hashes"]
               if p.startswith("bayesfilter/") or Path(p).name in
               ("diagnose_kdm_fresh_training_cost_tf.py", "diagnose_kdm_source_epoch_tf.py")]
    paths = list(dict.fromkeys(runtime + [Path(__file__).resolve(), EVALUATOR, AUDITOR,
        ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py", ROOT/PLAN]))
    inputs = [PROOF, PROOF.parent.parent/"manifest.json", PILOT/"manifest.json",
              PILOT/"decision.json", PILOT/"evaluation/diagnostic/result.json"]
    proof = read(PROOF)
    require(proof["status"] == "PASS" and not proof["checks_failed"], "Source-stage proof failed")
    require(sha(PROOF) == read(PROOF.parent.parent/"manifest.json")["result_sha256"], "Proof hash")
    for name, digest in read(PROOF.parent.parent/"manifest.json")["hashes"].items():
        if name.startswith("bayesfilter/replication/"):
            require(sha(ROOT/name) == digest, "Unverified runtime change: "+name)
    pilot_manifest = read(PILOT/"manifest.json")
    require(pilot_manifest["status"] == "COMPLETED", "Original repaired partner incomplete")
    require(pilot_manifest["evaluation_sha256"] == sha(inputs[-1]), "Original pair evaluation hash")
    units = []
    pilot_reference = Path(read(PILOT/"training/manifest.json")["reference"])
    inputs.append(pilot_reference)
    fixed_data = {k:v["sha256"] for k,v in read(pilot_reference)["datasets"].items()}
    for index, seed in enumerate(SEEDS, 1):
        source = SOURCE_UNITS/f"unit{index:02d}-full"
        reference = SOURCE_UNITS/f"unit{index:02d}-cost/diagnostic/result.json"
        audit = SOURCE_UNITS/f"unit{index:02d}-audit/result.json"
        source_result = read(source/"diagnostic/result.json")
        ref = read(reference)
        source_manifest = read(source/"manifest.json")
        checked = read(audit)
        require(checked["verdict"] == "PASS" and not checked["failures"], "Source terminal audit")
        require(Path(checked["run"]).resolve() == source, "Source audit unit")
        require(checked["run_manifest_sha256"] == sha(source/"manifest.json"), "Source audit manifest hash")
        require(source_completion_valid(source_manifest, source/"diagnostic", checked),
                "Source completion evidence")
        require(source_manifest["result_sha256"] == sha(source/"diagnostic/result.json"), "Source result hash")
        require(read(reference.parent.parent/"manifest.json")["result_sha256"] == sha(reference), "Cost hash")
        require(source_result["status"] == ref["status"] == "PASS", "Source cost/full status")
        require(source_result["source_pin"] == ref["source_pin"] == PIN, "Source pin")
        require(source_result["source_training_protocol_complete"], "Source full protocol")
        require(source_result["source_training_configs"] == ref["source_training_configs"], "Source config pairing")
        for section in ("datasets", "fixtures"):
            require(set(source_result[section]) == set(ref[section]), "Pair section "+section)
            for name, item in ref[section].items():
                other = source_result[section][name]
                require(item["sha256"] == other["sha256"], "Pair hash "+section+"/"+name)
                for value in (item, other):
                    require(sha(value["path"]) == value["sha256"], "Saved paired input changed")
                    inputs.append(Path(value["path"]))
        data_hashes = {k:v["sha256"] for k,v in ref["datasets"].items()}
        require(data_hashes == fixed_data, "Author units must share fixed data")
        inputs += [reference, reference.parent.parent/"manifest.json", source/"manifest.json",
                   source/"diagnostic/result.json", audit]
        units.append(dict(unit=index, seed_base=seed, source=str(source), reference=str(reference),
                          source_audit=str(audit), constructor_seed_base=ref["constructor_seed_base"]))
    return balance, spent, remaining, paths, list(dict.fromkeys(inputs)), units


def training_check(directory, seed, reference, full):
    result = read(directory/"result.json")
    require(result["status"] == "PASS" and result["checks_failed"] == 0, "Training checks")
    require(result["seed_base"] == seed, "CLI seed did not reach training result")
    require(result["reference_sha256"] == sha(reference), "Training used wrong unit reference")
    require(result["verification"]["sha256"] == sha(PROOF), "Training proof changed")
    require(result["tf32"] is False and result["jit_compile"] is True, "Training XLA/TF32")
    require(result["visible_device"] == GPU, "Wrong training GPU")
    require(result["memory_policy"]["all_physical_devices_memory_growth"], "GPU memory policy")
    require([x["stage"] for x in result["completed_stages"]] == ["weights","mdpf","am_dpf"],
            "Missing training stage")
    require(bool(result["source_training_protocol_complete"]) == full, "Wrong training completion scope")
    for name in ("weights", "mdpf", "am_dpf"):
        checkpoint = read(directory/(name+"-latest.json"))
        require(checkpoint["seed_policy"]["shuffle_base"] == seed and
                checkpoint["seed_policy"]["native_base"] == seed+1, "Kernel/checkpoint seed mismatch")
    return result


def summarize(evaluations):
    units = []
    for unit, path in enumerate(evaluations):
        report = read(path)
        require(report["decision"]["status"] == "PASS" and not report["failures"], "Evaluation validity")
        require(len(report["comparisons"]) == 4, "Incomplete variant/seed coverage")
        row = {"unit":unit, "evaluation":str(path), "evaluation_sha256":sha(path), "variants":{}}
        for variant in ("mdpf", "am_dpf"):
            values = [x for x in report["comparisons"] if x["variant"] == variant]
            require({v["seed"] for v in values} == {202610082,202610083}, "Evaluation seed mismatch")
            row["variants"][variant] = dict(
                source_nll=statistics.mean(v["source"]["nll"] for v in values),
                tf_nll=statistics.mean(v["tf_trained"]["nll"] for v in values),
                nll_difference=statistics.mean(v["nll_difference"] for v in values),
                log_rmse_ratio=statistics.mean(math.log(v["rmse_ratio"]) for v in values))
        units.append(row)
    statistics_rows = {}
    for variant in ("mdpf", "am_dpf"):
        values = [u["variants"][variant] for u in units]
        nll = [v["nll_difference"] for v in values]
        ratio = [v["log_rmse_ratio"] for v in values]
        source = [v["source_nll"] for v in values]
        tf = [v["tf_nll"] for v in values]
        covariance = sum((s-statistics.mean(source))*(t-statistics.mean(tf))
                         for s,t in zip(source,tf))/(len(source)-1)
        # Precision at true equality, Bonferroni 95% across four endpoints.
        # A planning illustration only: n=3 cannot validate normality or SD.
        z = statistics.NormalDist().inv_cdf(1-.05/4)
        statistics_rows[variant] = dict(n=len(values), nll_mean=statistics.mean(nll),
            nll_sd=statistics.stdev(nll), log_rmse_ratio_mean=statistics.mean(ratio),
            log_rmse_ratio_sd=statistics.stdev(ratio), source_tf_nll_covariance=covariance,
            illustrative_units_at_true_equality=dict(
                nll=math.ceil(((z+statistics.NormalDist().inv_cdf(.8))*statistics.stdev(nll)/.01)**2),
                rmse=math.ceil(((z+statistics.NormalDist().inv_cdf(.8))*statistics.stdev(ratio)/math.log(1.01))**2)),
            planning_assumptions="Normal paired means, plug-in n=3 SD, 80% power; NOT validated or a qualification design")
    return dict(status="PAIRED_DEVELOPMENT_CALIBRATION_COMPLETED", replication="INCOMPLETE",
                independent_training_pairs=len(units), units=units, descriptive_statistics=statistics_rows,
                inference="DESCRIPTIVE_ONLY; no supported ranking or population noninferiority verdict")


def main(preflight=False):
    balance, spent, remaining, paths, inputs, units = prepare()
    previous = sorted(PHASE.glob("calibration-*/manifest.json"))
    out = PHASE/f"calibration-{len(previous)+1:02d}"
    if preflight:
        print(json.dumps(dict(status="PREFLIGHT_PASS", output=str(out), remaining_seconds=remaining,
                              units=units, source_files=len(paths), input_files=len(inputs))))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    for p in paths:
        dest = out/"sources"/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_CPP_MIN_LOG_LEVEL="2", OMP_NUM_THREADS="1", TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1")
    gpu_env.pop("LD_LIBRARY_PATH", None)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    manifest = dict(schema="kdm_paired_calibration_v1", status="RUNNING", command=[sys.executable,*sys.argv],
        cwd=str(ROOT), git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan_file=PLAN, result_file=RESULT, started_utc=stamp(), supervisor_pid=os.getpid(),
        phase_cap_seconds=CAP, prior_phase_seconds=spent, campaign_remaining_before=remaining,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}, source_pin=PIN, units=units,
        data_version="Shared source training5000/validation1000, data seed 202610071",
        environment_gpu={k:gpu_env[k] for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
            "TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS","OMP_NUM_THREADS")},
        environment_cpu={k:cpu_env[k] for k in ("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","LD_LIBRARY_PATH")},
        interpreters=dict(tf=TF, source=SOURCE), gpu_trust_basis="escalated_local_gpu_access",
        cpu_gpu_status="TF GPU/XLA; author reference CPU with GPU intentionally hidden",
        resource_wall_seconds={"gpu":0., "cpu":0.}, steps=[], completed_units=[], replication="INCOMPLETE")
    def save(): write(out/"manifest.json", manifest)
    def run(label, command, resource, limit, progress=None):
        allowance = min(limit, remaining[resource]-manifest["resource_wall_seconds"][resource])
        require(allowance > 0, "Phase/campaign budget exhausted")
        step = dict(label=label, command=command, resource=resource, timeout_seconds=allowance,
                    status="RUNNING", log=label+".log", started_unix_seconds=time.time(), started_utc=stamp())
        manifest["steps"].append(step)
        save()
        start = time.monotonic()
        child = None
        try:
            with (out/step["log"]).open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=gpu_env if resource=="gpu" else cpu_env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                step["pid"] = child.pid
                save()
                while True:
                    try:
                        code = child.wait(timeout=max(.1,min(30,allowance-(time.monotonic()-start))))
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = time.monotonic()-start
                        age = time.time()-progress.stat().st_mtime if progress and progress.exists() else elapsed
                        step.update(elapsed_seconds=elapsed, progress_age_seconds=age)
                        save()
                        if elapsed>=allowance or age>1800:
                            raise TimeoutError("STEP_TIME_LIMIT" if elapsed>=allowance else "HEARTBEAT_STALL")
            step.update(status="COMPLETED" if code==0 else "FAILED", returncode=code)
            require(code == 0, label+" failed; see "+str(out/step["log"]))
        except BaseException:
            step["status"] = "FAILED"
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            raise
        finally:
            step["wall_seconds"] = time.monotonic()-start
            manifest["resource_wall_seconds"][resource] += step["wall_seconds"]
            save()
        return step
    save()
    decision = dict(status="INCOMPLETE_EXECUTION", replication="INCOMPLETE")
    try:
        run("nvidia", ["nvidia-smi","--query-gpu=uuid,name,memory.used,memory.total","--format=csv"],"gpu",30)
        # Regression of the changed reference binding on the completed first pair.
        run("original-pair-evaluation", [SOURCE,"-I","-B",str(EVALUATOR),"--worker",
            "--output",str(out/"original-pair-evaluation"),"--tf-run",str(PILOT/"training"),
            "--tf-audit",str(PILOT/"audit.json")],"cpu",300)
        original = read(PILOT/"evaluation/diagnostic/result.json")
        replay = read(out/"original-pair-evaluation/result.json")
        require(replay["comparisons"] == original["comparisons"], "Original-pair evaluation regression")
        evaluations = [out/"original-pair-evaluation/result.json"]
        for unit in units:
            name = f"unit{unit['unit']:02d}"
            directory = out/name
            directory.mkdir()
            reference = Path(unit["reference"])
            command = [TF,"-I","-B",str(DRIVER),"--reference",str(reference),
                       "--verification",str(PROOF),"--seed-base",str(unit["seed_base"])]
            run(name+"-smoke",command+["--epochs","smoke","--output",str(directory/"smoke")],
                "gpu",600,directory/"smoke/progress.json")
            training_check(directory/"smoke",unit["seed_base"],reference,False)
            train = directory/"training"
            train.mkdir()
            training_manifest = dict(schema="kdm_complete_training_manifest_v1", mode="tf", epoch_mode="full",
                status="RUNNING", cwd=str(ROOT), git_commit=manifest["git_commit"], plan=PLAN, result=RESULT,
                reference=str(reference), reference_sha256=sha(reference), verification=str(PROOF),
                verification_sha256=sha(PROOF), source_pin=PIN,
                hashes={str(p.relative_to(ROOT)):sha(p) for p in paths if p != ROOT/PLAN},
                seeds=dict(data=202610071,constructor_base=unit["constructor_seed_base"],
                           tf_seed_base=unit["seed_base"]), environment=manifest["environment_gpu"],
                cpu_gpu_status=manifest["cpu_gpu_status"], replication="INCOMPLETE")
            training_command = command+["--epochs","full","--output",str(train/"diagnostic")]
            training_manifest["command"] = training_command
            write(train/"manifest.json",training_manifest)
            try:
                step = run(name+"-full",training_command,"gpu",14400,train/"diagnostic/progress.json")
                result = training_check(train/"diagnostic",unit["seed_base"],reference,True)
                training_manifest.update(status="COMPLETED", returncode=0, wall_seconds=step["wall_seconds"],
                    result_sha256=sha(train/"diagnostic/result.json"), memory_policy=result["memory_policy"],
                    jit_compile=result["jit_compile"],tf32=result["tf32"],visible_device=result["visible_device"])
            except BaseException:
                training_manifest.update(status="FAILED",returncode=1)
                raise
            finally:
                write(train/"manifest.json",training_manifest)
            audit = directory/"audit.json"
            run(name+"-audit",[SOURCE,"-I","-B",str(AUDITOR),"--run",str(train),"--output",str(audit)],"cpu",300)
            require(read(audit)["verdict"]=="PASS" and not read(audit)["failures"], "TF terminal audit")
            evaluation = directory/"evaluation"
            run(name+"-evaluation",[SOURCE,"-I","-B",str(EVALUATOR),"--worker","--output",str(evaluation),
                "--source-run",unit["source"],"--source-audit",unit["source_audit"],
                "--tf-run",str(train),"--tf-audit",str(audit)],"cpu",300)
            report = read(evaluation/"result.json")
            require(report["source_training_run"]==unit["source"] and report["tf_training_run"]==str(train),
                    "Evaluation loaded the wrong training pair")
            require(report["decision"]["status"]=="PASS" and not report["failures"], "Evaluation validity")
            evaluations.append(evaluation/"result.json")
            manifest["completed_units"].append(dict(**unit, training_run=str(train), audit=str(audit),
                evaluation=str(evaluation/"result.json"), evaluation_sha256=sha(evaluation/"result.json")))
            save()
        for field in ("source_hashes","input_hashes"):
            for name,digest in manifest[field].items():
                require(sha(ROOT/name)==digest, "Code/input changed during calibration: "+name)
        decision = summarize(evaluations)
        manifest["status"] = "COMPLETED"
    except BaseException:
        manifest["status"] = "FAILED"
        decision["error"] = traceback.format_exc()
        (out/"error.txt").write_text(decision["error"])
    finally:
        write(out/"decision.json",decision)
        manifest.update(decision_sha256=sha(out/"decision.json"),ended_utc=stamp())
        save()
        ledger = BASE/"reference-diagnostics.json"
        records = read(ledger)
        for resource in CAP:
            record = dict(id="kdm-paired-"+out.name+"_"+resource,resource=resource,
                charged_seconds=manifest["resource_wall_seconds"][resource],
                summary=str((out/"decision.json").relative_to(ROOT)),summary_sha256=sha(out/"decision.json"),
                manifest=str((out/"manifest.json").relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage="KDM_PAIRED_DEVELOPMENT_CALIBRATION",after_attempt=balance["attempts"],
                next_action="Review paired variability and qualification feasibility; no automatic promotion",
                executions=[dict(path=str(out.relative_to(ROOT)),resource=resource,
                                 wall_seconds=manifest["resource_wall_seconds"][resource])])
            require(not any(x["id"]==record["id"] for x in records), "Duplicate ledger ID")
            records.append(record)
        write(ledger,records)
    print(json.dumps(dict(status=manifest["status"],output=str(out),
                          resource_wall_seconds=manifest["resource_wall_seconds"],decision=decision["status"])))
    return 0 if manifest["status"]=="COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(preflight=args.preflight))

