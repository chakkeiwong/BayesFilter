"""Bounded native replay under two CPU affinities; no tuning or price authority."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.model) != 3:
        raise ValueError("all three predeclared model families required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = Path(__file__).with_name("probe_current_native_cost.py").read_text()
    marker = 'p.add_argument("--gpu", required=True)'
    insertion = 'assert os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu'
    assert original.count(marker) == original.count(insertion) == 1
    worker = original.replace(marker, marker+'\np.add_argument("--cpus", type=int, nargs=4, required=True)')
    worker = worker.replace(insertion, 'os.sched_setaffinity(0, set(args.cpus))\n'+insertion)
    worker_path = root/"worker.py"
    worker_path.write_text(worker)
    (root/"runner.py").write_bytes(Path(__file__).read_bytes())
    env = os.environ.copy()
    env.update(TF_FORCE_GPU_ALLOW_GROWTH="true", CUDA_VISIBLE_DEVICES=args.gpu,
        TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    env.pop("PYTHONPATH", None)
    arms = [("original-01", [8,9,10,11]), ("candidate-01", [32,33,34,35]),
            ("candidate-02", [32,33,34,35]), ("original-02", [8,9,10,11])]
    receipt = dict(command=sys.argv, source=str(args.source.resolve()),
        environment=sys.executable, resource="gpu", gpu_uuid=args.gpu,
        started_utc=datetime.now(timezone.utc).isoformat(), budget_seconds=400,
        per_worker_cap_seconds=90, arms=arms, attempts=[], release_ready=False,
        plan="docs/plans/artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-04.md",
        original_worker_sha256=hashlib.sha256(original.encode()).hexdigest(),
        worker_sha256=hashlib.sha256(worker_path.read_bytes()).hexdigest())
    write(root/"manifest.json", receipt)
    code = 1
    try:
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
        receipt["preflight_compute_processes"] = processes
        if any(line.split(",")[0].strip() == args.gpu for line in processes):
            raise RuntimeError("selected GPU has another compute process; diagnostic deferred")
        for label, cpus in arms:
            output = root/label
            output.mkdir()
            command = [sys.executable, str(worker_path), "--source", str(args.source.resolve()),
                "--output", str(output), "--gpu", args.gpu, "--cpus", *map(str, cpus)]
            for model in args.model:
                command.extend(["--model", str(model.resolve())])
            remaining = 390-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("diagnostic allocation exhausted")
            before = time.monotonic()
            with (output/"run.log").open("x") as log:
                try:
                    arm_code = subprocess.run(command, env=env, stdout=log,
                        stderr=subprocess.STDOUT, timeout=min(90,remaining)).returncode
                except subprocess.TimeoutExpired:
                    arm_code = 124
            record = dict(arm=label, cpu_affinity=cpus, command=command,
                exit_code=arm_code, wall_seconds=time.monotonic()-before)
            receipt["attempts"].append(record)
            write(root/"progress.json", receipt)
            if arm_code:
                raise RuntimeError(f"{label} failed with code {arm_code}")
            result = json.loads((output/"result.json").read_text())
            if result["status"] != "exact_original_native_batches_replayed" or len(result["rows"]) != 9:
                raise ValueError("missing exact full-record replay for three models")
        code = 0
    except Exception as error:
        receipt["error"] = type(error).__name__+": "+str(error)
    finally:
        receipt.update(exit_code=code, wall_seconds=time.monotonic()-started,
            status="exact_four_arm_affinity_replay" if code == 0 else "diagnostic_incomplete")
        write(root/"receipt.json", receipt)
        print(json.dumps({k:receipt[k] for k in ("status","exit_code","wall_seconds")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
