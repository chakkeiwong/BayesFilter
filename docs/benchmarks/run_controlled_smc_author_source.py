"""Independent Octave author-source reference runner; no BayesFilter numerical kernel.

Only case scheduling, RNG checkpointing, and reporting change source driver text.
Author algorithm files are byte-for-byte copies of the C0 verified snapshot.
SciPy is used solely to read the independent reference MAT output.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import resource

DRIVERS = (
    ("lorenz_csmc", "Lorenz", "cSMC_VaryF.m", "cSMC_LogNormConst", [1,20,40], 4000, 202610031),
    ("lorenz_apf", "Lorenz", "APF_VaryF.m", "APF_LogNormConst", [1,20,40], 4000, 202610032),
    ("neuroscience_csmc", "NeuroScience", "NormConst_cSMC.m", "cSMC_LogNormConst", [1,10,20], 2000, 202610033),
    ("neuroscience_bpf", "NeuroScience", "NormConst_BPF.m", "BPF_LogNormConst", [1,10,20], 2000, 202610034),
)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def instrument(original, key, lane, variable, indices, mode):
    # No model, sample size, horizon, fit, solver, or numerical safeguard edit.
    result = original
    if mode == "pilot":
        result, count = re.subn(r"(?m)^nrep = 100;", "nrep = 1;", result)
        assert count == 1, "source repetition declaration changed"
        result, count = re.subn(r"(?m)^for igrid = 1:ngrid",
            "for igrid = [" + " ".join(map(str, indices)) + "]", result)
        assert count == 1, "source grid loop changed"
    result, count = re.subn(r"(?m)^(\s*)for irep = 1:nrep[ \t]*$", r"\1for irep = 1:nrep\n\1    source_case_clock = tic;\n\1    source_case_cpu_clock = cputime;", result)
    assert count == 1
    # Save only after all source statistics for this case have been computed.
    if lane == "Lorenz":
        marker = ("cSMC_Iterations(irep,igrid) = size(cSMC,1) - 1;" if key.endswith("csmc")
                  else "APF_ESS(irep,igrid) = mean(cSMC{1,2}.ESS);")
    else:
        marker = next(line.strip() for line in original.splitlines()
                      if f"{variable}(irep,igrid) = cSMC_Resample" in line)
    assert result.count(marker) == 1
    ess = ("cSMC_ESS(irep,igrid)" if key == "lorenz_csmc" else
           "APF_ESS(irep,igrid)" if key == "lorenz_apf" else "NaN")
    refinements = "cSMC_Iterations(irep,igrid)" if lane == "Lorenz" else "I"
    report = (
        "\n        source_case_seconds = toc(source_case_clock);\n"
        "        source_case_cpu_seconds = cputime - source_case_cpu_clock;\n"
        f"        source_case_value = {variable}(irep,igrid);\n"
        f"        source_case_file = fopen('{key}-cases.csv','a');\n"
        f"        fprintf(source_case_file,'%d,%d,%.17g,%.17g,%d,%d,%d,%d,%.17g,%d,%.17g\\n',igrid,irep,source_case_seconds,real(source_case_value),isreal(source_case_value)&&isfinite(source_case_value),N,I,T,{ess},{refinements},source_case_cpu_seconds);\n"
        "        fclose(source_case_file);\n"
        f"        save('-mat7-binary','{key}-progress.mat','{variable}');"
    )
    result = result.replace(marker, marker + report)
    result += f"\nsave('-mat7-binary','{key}-result.mat','{variable}','N','I','T','nrep','ngrid');\n"
    return result

def validate_native_matrix(data, variable, rows, nrep, ngrid):
    """Check independently saved source MAT values against the case ledger."""
    values = data[variable]
    assert values.shape == (nrep, ngrid), "native source result matrix has wrong shape"
    for row in rows:
        native = complex(values[row["repetition"] - 1, row["grid_index"] - 1])
        finite_real = native.imag == 0 and math.isfinite(native.real)
        assert row["finite_real"] == finite_real, "MAT/CSV validity classification differs"
        assert row["logZ"] == (native.real if math.isfinite(native.real) else None), "MAT/CSV logZ differs"
    return dict(shape=list(values.shape), checked_cases=len(rows), csv_values_match=True)


def add_case_checkpoint(text, key, variable, *, isolated):
    """Instrument only case scheduling, output retention, and random-state transfer."""
    arrays = [variable]
    if key == "lorenz_csmc":
        arrays += ["cSMC_ESS", "cSMC_Iterations"]
    elif key == "lorenz_apf":
        arrays += ["APF_ESS"]
    marker = f"save('-mat7-binary','{key}-progress.mat','{variable}');"
    state = ("source_rand_state=rand('state'); source_randn_state=randn('state');\n"
             f"save('-mat7-binary','{key}-checkpoint-next.mat',"
             + ",".join("'" + x + "'" for x in arrays + ["source_rand_state","source_randn_state"])
             + f");\nmovefile('{key}-checkpoint-next.mat','{key}-checkpoint.mat','f');")
    assert text.count(marker) == 1
    text = text.replace(marker, marker + "\n" + state)
    if isolated:
        restore = (f"if exist('{key}-checkpoint.mat','file')\n"
                   f" load('{key}-checkpoint.mat');\n"
                   " rand('state',source_rand_state); randn('state',source_randn_state);\nend\n")
        text, count = re.subn(r"(?m)^for igrid = 1:ngrid",
            restore + "for igrid = str2double(getenv('BAYESFILTER_SOURCE_GRID'))", text)
        assert count == 1
        text, count = re.subn(r"(?m)^(\s*)for irep = 1:nrep",
            r"\1for irep = str2double(getenv('BAYESFILTER_SOURCE_REPETITION'))", text)
        assert count == 1
    return text


def octave_command(stats, dest, instrumented, seed):
    quote = lambda x: str(x).replace("'", "''")
    expression = (f"addpath('{quote(stats)}');addpath('{quote(stats/'distributions')}');"
                  f"cd('{quote(dest)}');save_default_options('-mat7-binary');"
                  f"rand('state',{seed});randn('state',{seed});run('{quote(instrumented.name)}');")
    return ["octave", "--no-gui", "--quiet", "--eval", expression]


def reference_memory_limit():
    # CPU-only reference process; reject excessive address space rather than OOM the host.
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))


def run_isolated_cases(command, dest, key, cases, seconds, log_path, process_path,
                       prefix_rows=None):
    started = time.monotonic()
    completed = 0
    with log_path.open("a") as log, process_path.open("a") as record:
        for grid, repetition in cases:
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                return dict(returncode=124,completed_cases=completed,reason="driver deadline")
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="-1", BAYESFILTER_SOURCE_GRID=str(grid),
                       BAYESFILTER_SOURCE_REPETITION=str(repetition))
            case_start = time.monotonic()
            child = subprocess.Popen(command,cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     preexec_fn=reference_memory_limit)
            try:
                rc = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.terminate()
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                rc = 124
            info = dict(grid_index=grid,repetition=repetition,pid=child.pid,returncode=rc,
                        wall_seconds=time.monotonic()-case_start,
                        cumulative_peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                        address_space_limit_bytes=8 * 1024**3,
                        rng_transfer="rand(state), randn(state); prior completed case",
                        gpu_intentionally_hidden=True)
            record.write(json.dumps(info)+"\n"); record.flush()
            if rc:
                return dict(returncode=rc,completed_cases=completed,reason="isolated source process failed")
            with (dest/(key+"-cases.csv")).open("rb") as ledger:
                ledger.seek(0,2)
                ledger.seek(max(0,ledger.tell()-4096))
                line = ledger.read().decode().splitlines()[-1]
            row = next(csv.reader([line]))
            assert (int(row[0]),int(row[1])) == (grid,repetition)
            assert (dest/(key+"-checkpoint.mat")).is_file()
            if prefix_rows and (grid,repetition) in prefix_rows:
                old=prefix_rows[grid,repetition]
                # Compare exact source outputs and settings; CPU/wall clocks are explanatory.
                for index in (3,4,5,6,7,8,9):
                    x,y=float(row[index]),float(old[index])
                    assert x==y or (math.isnan(x) and math.isnan(y)), "interrupted-source prefix changed"
            completed += 1
    return dict(returncode=0,completed_cases=completed,reason="complete",
                cumulative_peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--source-lock", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=("pilot","full"), default="pilot")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--isolate-cases", action="store_true")
    ap.add_argument("--compare-prefix", type=Path)
    ap.add_argument("--full-spec", type=Path)
    ap.add_argument("--only-driver", choices=tuple(x[0] for x in DRIVERS))
    ap.add_argument("--resume-pilot", type=Path)
    ap.add_argument("--driver-seconds", type=float, default=45)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/"reference-work").exists():
        raise RuntimeError("Use a fresh output directory; reference evidence must not be overwritten")
    if a.driver_seconds <= 0:
        raise ValueError("driver-seconds must be positive")
    lock = json.loads((a.source_lock/"source-lock.json").read_text())
    assert lock["status"] == "PASS"
    full_spec=None
    if a.mode=="full":
        assert a.full_spec, "full mode requires the reviewed full-run specification"
        full_spec=json.loads(a.full_spec.read_text())
        assert full_spec["source_lock_sha256"]==sha(a.source_lock/"source-lock.json")
        assert set(full_spec["driver_limits_seconds"])=={x[0] for x in DRIVERS}
        assert full_spec["expected_cases"]==sum(x[5] for x in DRIVERS)
        assert all(math.isfinite(x) and x > 0 for x in full_spec["driver_limits_seconds"].values())
        assert sum(full_spec["driver_limits_seconds"].values())<full_spec["aggregate_cpu_wall_seconds"]

    observed = []
    reused = {}
    if a.resume_pilot:
        assert a.mode=="pilot" and a.only_driver, "resume is restricted to a named pilot driver"
        previous=json.loads((a.resume_pilot/"summary.json").read_text())
        assert previous["source_lock_sha256"]==sha(a.source_lock/"source-lock.json")
        assert previous["scope"]=="AUTHOR_SOURCE_EXECUTABILITY_AND_COST_PILOT"
        reused={x["driver"]:x for x in previous["drivers"] if x["status"]=="COMPLETE"}
    for entry in lock["verified_snapshot_files"]:
        rel = Path(entry["path"]).relative_to(".localresources/code/heng-controlled-smc")
        p = a.source_lock/"author-source"/rel
        assert sha(p) == entry["sha256"], f"source snapshot changed: {rel}"
    for entry in lock["runtime_dependencies"]:
        assert sha(a.root/entry["path"]) == entry["sha256"], "reference dependency changed"
    stats = a.root/".localresources/code/octave-statistics-1.5.0/inst"
    completed = set()
    for key,lane,driver,variable,indices,full_cases,seed in DRIVERS:
        if a.only_driver and key != a.only_driver:
            if key in reused:
                info=dict(reused[key],reused_from=str(a.resume_pilot),
                          reused_summary_sha256=sha(a.resume_pilot/"summary.json"))
                observed.append(info)
                completed.add(key)
            continue
        dest = a.output/"reference-work"/lane
        dest.mkdir(parents=True,exist_ok=True)
        source_dir = a.source_lock/"author-source/StateSpaceModels"/lane
        for p in source_dir.iterdir():
            if p.is_file():
                target = dest/p.name
                if not target.exists():
                    shutil.copyfile(p,target)
        original = (source_dir/driver).read_text()
        text = instrument(original,key,lane,variable,indices,a.mode)
        if a.isolate_cases:
            assert a.mode == "full", "case isolation is the reviewed full-source repair"
            text = add_case_checkpoint(text,key,variable,isolated=True)
        instrumented = dest/(Path(driver).stem+"_"+a.mode+".m")
        instrumented.write_text(text)
        (a.output/(key+"-instrumentation.diff")).write_text("".join(
            difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=driver,tofile=instrumented.name)))
        actual_seed = seed if a.mode == "pilot" else full_spec["seeds"][key]
        assert actual_seed == (seed if a.mode=="pilot" else seed+100)
        driver_limit=a.driver_seconds if a.mode=="pilot" else full_spec["driver_limits_seconds"][key]
        if key == "lorenz_apf" and "lorenz_csmc" not in completed and not a.prepare_only:
            observed.append(dict(driver=key,status="BLOCKED_CSMC_ITERATIONS",source_failure=False))
            continue
        quote = lambda x: str(x).replace("'","''")
        eval_text = (
            f"addpath('{quote(stats)}');addpath('{quote(stats/'distributions')}');"
            f"cd('{quote(dest)}');"
            "save_default_options('-mat7-binary');"
            f"rand('state',{actual_seed});randn('state',{actual_seed});"
            f"run('{quote(instrumented.name)}');"
        )
        command = ["octave","--no-gui","--quiet","--eval",eval_text]
        info = dict(driver=key,lane=lane,command=command,mode=a.mode,seed=actual_seed,
                    source_driver_sha256=sha(source_dir/driver),instrumented_sha256=sha(instrumented),
                    source_algorithm_sha256=sha(dest/"cSMC_Resample.m"),full_cases=full_cases,
                    timeout_seconds=driver_limit)
        if a.prepare_only:
            info["status"]="PREPARED_NOT_EXECUTED";observed.append(info);continue
        start=time.monotonic()
        if a.isolate_cases:
            prefix_rows={}
            if a.compare_prefix:
                prefix_path=a.compare_prefix/"reference-work"/lane/(key+"-cases.csv")
                if prefix_path.exists():
                    with prefix_path.open() as stream:
                        prefix_rows={(int(row[0]),int(row[1])):row for row in csv.reader(stream) if len(row)==11}
            cases=[(g,r) for g in range(1,41 if lane=="Lorenz" else 21) for r in range(1,101)]
            isolated=run_isolated_cases(command,dest,key,cases,driver_limit,
                a.output/(key+".log"),a.output/(key+"-processes.jsonl"),prefix_rows)
            info.update(process_isolation=isolated,compared_interrupted_prefix_cases=len(prefix_rows))
            rc=isolated["returncode"]
        else:
            with (a.output/(key+".log")).open("w") as log:
                child=subprocess.Popen(command,cwd=dest,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
                try:
                    rc=child.wait(timeout=driver_limit)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    rc=124
        info.update(returncode=rc,wall_seconds=time.monotonic()-start)
        rows=[]
        csv_path=dest/(key+"-cases.csv")
        if csv_path.exists():
            for row in csv.reader(csv_path.open()):
                value=float(row[3])
                rows.append(dict(grid_index=int(row[0]),repetition=int(row[1]),seconds=float(row[2]),
                                 logZ=value if math.isfinite(value) else None,finite_real=bool(int(row[4])),
                                 N=int(row[5]),I=int(row[6]),T=int(row[7]),
                                 ESS=float(row[8]) if math.isfinite(float(row[8])) else None,
                                 refinements=int(row[9]),cpu_seconds=float(row[10]) if len(row)>10 else None))
        expected=len(indices) if a.mode=="pilot" else full_cases
        grid_indices=indices if a.mode=="pilot" else range(1,41 if lane=="Lorenz" else 21)
        expected_keys={(g,r) for g in grid_indices for r in range(1,2 if a.mode=="pilot" else 101)}
        case_keys=[(row["grid_index"],row["repetition"]) for row in rows]
        coverage_ok=len(case_keys)==len(set(case_keys)) and set(case_keys)==expected_keys
        expected_ni={"lorenz_csmc":(512,4),"neuroscience_csmc":(128,3),"neuroscience_bpf":(5529,0)}
        for row in rows:
            assert row["seconds"] >= 0 and math.isfinite(row["seconds"])
            if key in expected_ni:
                assert (row["N"],row["I"])==expected_ni[key]
            else:
                assert row["I"]==0 and row["N"]=={1:1382,2:2867,3:4505,4:6144}[row["refinements"]]
        info.update(cases=rows,expected_cases=expected,exact_case_coverage=coverage_ok)
        info["status"]=("COMPLETE" if rc==0 and coverage_ok else
                        "TIMEOUT" if rc==124 else "SOURCE_OR_RUNTIME_FAILURE")
        info["source_numerical_failure"]=any(not row["finite_real"] for row in rows)
        if info["status"]=="COMPLETE":
            completed.add(key)
            from scipy.io import loadmat  # Independent source MAT inspection only.
            data=loadmat(dest/(key+"-result.mat"))
            # Check the exact paper-scale inputs survived driver instrumentation.
            info["source_settings"]={name:int(data[name].item()) for name in ("N","I","T","nrep","ngrid")}
            assert info["source_settings"]["nrep"]==(1 if a.mode=="pilot" else 100)
            assert info["source_settings"]["ngrid"]==(40 if lane=="Lorenz" else 20)
            info["native_matrix_validation"] = validate_native_matrix(
                data, variable, rows, info["source_settings"]["nrep"], info["source_settings"]["ngrid"])
            source_data=loadmat(source_dir/("SimulatedData.mat" if lane=="Lorenz" else "Data_NeuroScience.mat"))
            expected_t=100 if lane=="Lorenz" else source_data["y"].size-1
            assert info["source_settings"]["T"]==expected_t
            assert all(row["T"]==expected_t for row in rows)
            if key=="lorenz_apf":
                csmc_rows=next(x["cases"] for x in observed if x["driver"]=="lorenz_csmc")
                iterations={(x["grid_index"],x["repetition"]):x["refinements"] for x in csmc_rows}
                assert all(row["refinements"]==iterations[(row["grid_index"],row["repetition"])] for row in rows)
            if key in expected_ni:
                assert (info["source_settings"]["N"],info["source_settings"]["I"])==expected_ni[key]
        if rows:
            times=[row["seconds"] for row in rows]
            info["descriptive_full_matrix_seconds"]=dict(
                mean_projection=sum(times)/len(times)*full_cases,
                conservative_reservation=2*max(times)*full_cases+60,
                observed_case_count=len(rows),
                evidence="selected cells; scheduling estimate, not a confidence interval")
            info["clock_consistency"] = ("INCONSISTENT_OCTAVE_TIMER_EXCEEDS_PROCESS_WALL"
                if sum(times)>info["wall_seconds"]+0.5 else "NO_OBSERVED_EXCESS")
            info["cost_limitations"]="Sparse cells, runtime noise and separate Octave/Python clocks; no precise cost claim"
        observed.append(info)
        (a.output/"progress.json").write_text(json.dumps(observed,indent=2,allow_nan=False)+"\n")
    status=("PREPARED" if a.prepare_only else "PASS" if all(
        x["status"]=="COMPLETE" for x in observed) and len(observed)==4 else "FAIL")
    summary=dict(status=status,scope="AUTHOR_SOURCE_EXECUTABILITY_AND_COST_PILOT" if a.mode=="pilot" else "AUTHOR_SOURCE_MATRIX",
        replication="INCOMPLETE",cpu_only=True,gpu_intentionally_hidden=True,
        author_algorithm_modified=False,case_process_isolation=a.isolate_cases,
        interrupted_prefix=str(a.compare_prefix) if a.compare_prefix else None,
        driver_changes=("Pilot grid/repetition scheduling and observation-only reporting" if a.mode=="pilot" else "Case process isolation with RNG/output state transfer; unchanged source cases" if a.isolate_cases else "Observation-only reporting; source grid/repetitions unchanged"),
        runtime_equivalence="Author MATLAB source executed in Octave; MATLAB/Octave equivalence not yet certified",
        source_validity=("PREPARED_NOT_EXECUTED" if a.prepare_only else
            "Per-case finite-real logZ screen recorded; full covariance/domain validity is not certified by this screen"),
        source_lock=str(a.source_lock),source_lock_sha256=sha(a.source_lock/"source-lock.json"),drivers=observed,
        completed_driver_count=sum(x["status"]=="COMPLETE" for x in observed),
        resume_pilot=str(a.resume_pilot) if a.resume_pilot else None,
        scientific_verdict="No replication, statistical ranking, or port promotion from cost/executability alone",
        checks=[dict(name="all_four_driver_cases_accounted_for",passed=status=="PASS")])
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=status,drivers=[(x["driver"],x["status"]) for x in observed])))
    if status not in ("PASS","PREPARED"):raise SystemExit(1)

if __name__=="__main__":main()
