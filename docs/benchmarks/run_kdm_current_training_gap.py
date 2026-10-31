"""Bounded supervisor for current-state replication diagnostics; no training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE = BASE / "training-gap-20261012"
PLAN = "docs/plans/controlled-smc-kdm-training-gap-investigation-20261012.md"
RESULT = "docs/plans/controlled-smc-kdm-training-gap-result-20261012.md"
CPU = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
GPU = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(p)


def main():
    assert sys.argv[1:] in ([], ["--saved-optimizer"])
    saved_optimizer = sys.argv[1:] == ["--saved-optimizer"]
    from run_replication_gap_localization import campaign
    balance = campaign()
    assert not balance["running"]
    prior = list(PHASE.glob("attempt*/manifest.json"))
    assert len(prior) < 3 and all(read(p)["status"] != "RUNNING" for p in prior)
    remaining = {r: min(1800-sum(read(p)["resource_wall_seconds"][r] for p in prior),
                        balance["remaining_seconds"][r], balance["tranche_remaining_seconds"][r]) for r in ("cpu", "gpu")}
    assert remaining["cpu"] > 60 and remaining["gpu"] > 180
    out = PHASE / ("attempt%02d" % (len(prior)+1))
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", command=[sys.executable]+sys.argv,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    plan_file=PLAN, result_file=RESULT, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    seed=202610121, cpu_gpu_status="CPU author explicitly hides GPU; escalated TF GPU/XLA, TF32 disabled",
                    environment={"cpu":CPU, "gpu":GPU}, campaign_before=remaining,
                    resource_wall_seconds={"cpu":0., "gpu":0.}, steps=[],
                    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                                   [Path(__file__), ROOT/PLAN, ROOT/"docs/benchmarks/reference_kdm_current_training_gap.py",
                                    ROOT/"docs/benchmarks/reference_kdm_coupled_updates.py",
                                    ROOT/"docs/benchmarks/diagnose_kdm_neural_precision_tf.py"]})
    decision = dict(status="ERROR", replication="INCOMPLETE")
    write(out/"manifest.json", manifest)
    def run(label, cmd, resource):
        env = dict(os.environ, OMP_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
                   TF_NUM_INTEROP_THREADS="1", TF_FORCE_GPU_ALLOW_GROWTH="true")
        env["CUDA_VISIBLE_DEVICES"] = "-1" if resource == "cpu" else "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
        env["LD_LIBRARY_PATH"] = "/home/chakwong/anaconda3/envs/tftwogpu/lib:" + env.get("LD_LIBRARY_PATH", "")
        record = dict(label=label, command=cmd, resource=resource, log=str(out/(label+".log")))
        manifest["steps"].append(record)
        started = time.monotonic()
        try:
            with (out/(label+".log")).open("w") as log:
                process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=remaining[resource]-manifest["resource_wall_seconds"][resource])
                record["returncode"] = process.returncode
                assert process.returncode == 0, label
        finally:
            record["wall_seconds"] = time.monotonic()-started
            manifest["resource_wall_seconds"][resource] += record["wall_seconds"]
            write(out/"manifest.json", manifest)
    try:
        run("nvidia", ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"], "gpu")
        run("author", [CPU, "-I", "-B", str(ROOT/"docs/benchmarks/reference_kdm_current_training_gap.py"),
                       "--output", str(out/"reference")] + (["--saved-optimizer"] if saved_optimizer else []), "cpu")
        run("tf", [GPU, "-I", "-B", str(ROOT/"docs/benchmarks/diagnose_kdm_neural_precision_tf.py"),
                   "--reference", str(out/"reference/author.json"), "--output", str(out/"tf"),
                   "--plan", PLAN, "--result-note", RESULT], "gpu")
        decision = read(out/"tf/decision.json")
        manifest["status"] = "COMPLETED"
    except BaseException:
        manifest["status"] = "FAILED"
        decision["error"] = traceback.format_exc()
    finally:
        manifest["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out/"decision.json", decision)
        write(out/"manifest.json", manifest)
        ledger = BASE/"reference-diagnostics.json"
        records = read(ledger)
        for r in ("cpu", "gpu"):
            identity = "kdm-current-training-gap-"+out.name+"_"+r
            assert not any(x["id"] == identity for x in records)
            records.append(dict(id=identity, resource=r, charged_seconds=manifest["resource_wall_seconds"][r],
                                summary=str((out/"decision.json").relative_to(ROOT)), summary_sha256=sha(out/"decision.json"),
                                manifest=str((out/"manifest.json").relative_to(ROOT)), plan_file=PLAN, result_file=RESULT,
                                stage="KDM_CURRENT_TRAINING_GAP_DIAGNOSTIC", after_attempt=balance["attempts"],
                                next_action="Inspect current-state source/TF differences; no full training",
                                executions=[dict(path=str(out.relative_to(ROOT)), resource=r,
                                                 wall_seconds=manifest["resource_wall_seconds"][r])]))
        write(ledger, records)
    print(json.dumps(dict(output=str(out), status=manifest["status"], decision=decision,
                          wall_seconds=manifest["resource_wall_seconds"])))
    return 0 if manifest["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
