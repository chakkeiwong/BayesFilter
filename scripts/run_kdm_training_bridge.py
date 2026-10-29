"""Bounded, logged launcher for the KDM source-controller/native bridge diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/controlled-smc-kdm-training-bridge-20261007.md"
RESULT = "docs/plans/controlled-smc-kdm-training-bridge-result-20261007.md"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("controller", "native", "guards"), required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", TF_NUM_INTEROP_THREADS="1",
               TF_NUM_INTRAOP_THREADS="1", TF_FORCE_GPU_ALLOW_GROWTH="true")
    if args.mode == "controller":
        interpreter = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
        env.update(CUDA_VISIBLE_DEVICES="-1",
                   LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
        driver = "docs/benchmarks/diagnose_kdm_training_controller.py"
        timeout = 1800
    else:
        interpreter = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
        driver = "docs/benchmarks/diagnose_kdm_training_bridge_guards_tf.py" if args.mode == "guards" else "docs/benchmarks/diagnose_kdm_training_bridge_tf.py"
        timeout = 3600
    command = [interpreter, "-B", str(ROOT/driver), "--output", str(out/"diagnostic")]
    files = [Path(__file__), ROOT/PLAN, ROOT/driver]
    files += sorted((ROOT/"bayesfilter/replication").glob("kdm*tf.py"))
    files += [ROOT/"bayesfilter/replication/kdm_training_controller.py"]
    manifest = dict(schema="kdm_training_bridge_manifest_v1", command=command,
        launcher_command=[sys.executable]+sys.argv, cwd=str(ROOT), mode=args.mode,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"],cwd=ROOT,text=True).strip(),
        plan=PLAN, result=RESULT, timeout_seconds=timeout,
        device="CPU deliberately hidden GPU" if args.mode=="controller" else "trusted GPU/XLA",
        environment={k:env.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
            "OMP_NUM_THREADS","TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS","LD_LIBRARY_PATH")},
        hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        data="author source loss streams" if args.mode=="controller" else "frozen neural B64 source fixture; first update",
        seeds="N/A deterministic loss streams" if args.mode=="controller" else [731,1],
        status="RUNNING")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    started = time.monotonic()
    with (out/"command.log").open("w") as log:
        try:
            run = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=timeout)
            code = run.returncode
        except subprocess.TimeoutExpired:
            code = 124
    manifest.update(returncode=code, wall_seconds=time.monotonic()-started,
                    status="COMPLETED" if code==0 else "FAILED")
    result = out/"diagnostic/result.json"
    if result.exists():
        manifest["result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("status","returncode","wall_seconds","mode")}))
    print(str(out))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
