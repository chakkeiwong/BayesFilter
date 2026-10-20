"""Observe resource conditions around the unchanged frozen complete-price driver."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            fields = Path(f"/proc/{pid}/status").read_text().splitlines()
            pid = int(next(line for line in fields if line.startswith("PPid:")).split()[1])
        except (OSError, StopIteration, ValueError):
            return None
    return False


def probe(gpu, owner=None):
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
    selected = [line.split(",") for line in devices if line.split(",")[0].strip() == gpu]
    if len(selected) != 1:
        raise ValueError("selected GPU missing or duplicated")
    foreign, unknown = [], []
    for line in processes:
        parts = [part.strip() for part in line.split(",")]
        if parts[0] != gpu:
            continue
        pid = int(parts[1])
        owned = False if owner is None else descendant(pid, owner)
        if owned is None:
            unknown.append(pid)
        elif not owned:
            foreign.append(pid)
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), devices=devices,
        processes=processes, foreign_pids=foreign, unknown_pids=unknown,
        free_mib=int(selected[0][1]), utilization_percent=int(selected[0][2]),
        cpu_pressure=Path("/proc/pressure/cpu").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    args = p.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {8, 9, 10, 11})
    env = os.environ.copy()
    env.update(TF_FORCE_GPU_ALLOW_GROWTH="true", CUDA_VISIBLE_DEVICES=args.gpu,
        TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    env.pop("PYTHONPATH", None)
    started = time.monotonic()
    command = [sys.executable, str(args.source/"scripts/run_hmc_v7_release_prices.py"),
        "--source", str(args.source), "--output", str(args.output/"prices"),
        "--gpu", args.gpu, "--cases", "lgssm_qr", "nonlinear", "funnel_residual",
        "--wall-seconds", "1800", "--closeout-seconds", "1200",
        "--budget-seconds", "9300", "--trial-batch-size", "32"]
    manifest = dict(command=command, source=str(args.source), gpu_uuid=args.gpu,
        environment=sys.executable, started_utc=datetime.now(timezone.utc).isoformat(),
        resource="gpu", cpu_affinity=sorted(os.sched_getaffinity(0)),
        prelaunch_wait_cap_seconds=60, resource_poll_seconds=10,
        enclosing_cap_seconds=9370, minimum_free_mib=4096,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan="docs/plans/artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-04.md")
    write(args.output/"manifest.json", manifest)
    (args.output/"runner.py").write_bytes(Path(__file__).read_bytes())
    observations = []
    child = None
    status, code, error = "resource_deferred", 3, None
    try:
        while time.monotonic()-started < 60:
            observation = probe(args.gpu)
            observations.append({**observation, "stage":"before_launch"})
            write(args.output/"resources.json", observations)
            if (observation["free_mib"] >= 4096 and not observation["foreign_pids"]
                    and not observation["unknown_pids"]):
                break
            time.sleep(min(10, max(0,60-(time.monotonic()-started))))
        else:
            return code
        with (args.output/"run.log").open("x") as log:
            child = subprocess.Popen(command, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                observation = probe(args.gpu, child.pid)
                observations.append({**observation, "stage":"price_running"})
                write(args.output/"resources.json", observations)
                remaining = 9370-(time.monotonic()-started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command,9370)
                try:
                    child.wait(timeout=min(10,remaining))
                except subprocess.TimeoutExpired:
                    pass
        code = child.returncode
        status = "complete" if code == 0 else "price_failed"
    except Exception as exc:
        error = type(exc).__name__+": "+str(exc)
        status, code = "observer_or_price_failure", 1
        if child is not None and child.poll() is None:
            # Only this owned process group; no foreign GPU process is touched.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
    finally:
        receipt = dict(manifest, status=status, exit_code=code, error=error,
            wall_seconds=time.monotonic()-started,
            price_result=str(args.output/"prices/result.json"),
            foreign_or_unknown_process_observed=any(o["foreign_pids"] or o["unknown_pids"] for o in observations),
            foreign_or_unknown_during_price=any(o["stage"] == "price_running" and
                (o["foreign_pids"] or o["unknown_pids"]) for o in observations),
            resource_observations=len(observations),
            interpretation="Sampled resource observations cannot prove continuous exclusivity or future throughput. Complete price and all earlier prices remain separate evidence.")
        write(args.output/"receipt.json",receipt)
        print(json.dumps(receipt))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
