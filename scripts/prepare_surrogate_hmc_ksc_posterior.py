#!/usr/bin/env python3
"""Qualify the untouched KSC T2 parameter posterior after frozen control selection."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
from scripts.prepare_surrogate_hmc_ksc import PLAN, budgets, save, sha, sources, check_selection, run_bounded_worker


def worker(out):
    manifest = json.loads((out / "run-manifest.json").read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if len(tf.config.list_physical_devices("GPU")) != 1:
        raise ValueError("exactly one designated GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    selection = check_selection(Path(manifest["controls_root"]), validate_controls=True)
    from bayesfilter.testing.surrogate_hmc_ksc_reference import make_ksc_dataset
    from bayesfilter.testing.surrogate_hmc_ksc_posterior_reference import prepare_ksc_posterior
    data = make_ksc_dataset(partition="claim")
    save(out / "data.json", data)
    manifest.update(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        gpu_status=[d.name for d in tf.config.list_logical_devices("GPU")], seeds=data["seeds"],
        data_sha256=sha(out / "data.json"), selected_multiplier=selection["selected_multiplier"])
    save(out / "run-manifest.json", manifest)

    def check():
        if time.process_time() > manifest["cpu_budget"] - 2:
            raise TimeoutError("KSC posterior CPU allowance exhausted")

    report, grids = prepare_ksc_posterior(data, resource_check=check)
    files = []
    for intervals, grid in zip(report["parameter_intervals"], grids):
        directory = out / f"grid-{intervals}"
        directory.mkdir()
        row = {}
        for name, tensor in zip(("gamma", "beta", "density", "predictive"), grid):
            path = directory / (name + ".tensor")
            tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
            row[name] = dict(path=str(path), sha256=sha(path))
        files.append(row)
    report.update(grids=files, data_sha256=manifest["data_sha256"], selection_sha256=manifest["selection_sha256"],
                  posterior_comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")})
    save(out / "report.json", report)
    manifest["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    save(out / "run-manifest.json", manifest)
    return 0 if report["qualified"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.controls_root or not args.output:
        parser.error("--controls-root and --output required")
    started, parent_cpu = time.monotonic(), time.process_time()
    out = args.output.resolve()
    if out.parent != ROOT / "docs/plans/artifacts" or not out.name.startswith("surrogate-hmc-all-models-20260921-ksc-t2-posterior-"):
        raise ValueError("fresh versioned KSC posterior output root required")
    total, family, _ = budgets()
    prior = dict(cpu_worker_seconds=0., gpu_seconds=0.)
    for path in out.parent.glob("surrogate-hmc-all-models-*-ksc-t2-posterior-*/accounting.json"):
        payload = json.loads(path.read_text())
        for key, value in payload.get("charged_this_run", payload.get("charged")).items():
            prior[key] += value
    cpu = min(150 - prior["cpu_worker_seconds"], 1800 - family["cpu_worker_seconds"], 54000 - total["cpu_worker_seconds"])
    wall = min(250 - prior["gpu_seconds"], 7200 - family["gpu_seconds"], 226800 - total["gpu_seconds"])
    if min(cpu, wall) < 5:
        raise ValueError("posterior/family/campaign allocation exhausted")
    # Read-only selection checks before importing TensorFlow in the GPU worker.
    check_selection(args.controls_root)
    provenance = sources()
    for name in ("scripts/prepare_surrogate_hmc_ksc_posterior.py",
                 "bayesfilter/testing/surrogate_hmc_ksc_posterior_reference.py",
                 "bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py"):
        provenance[name] = sha(ROOT / name)
    out.mkdir(exist_ok=False)
    for name in provenance:
        dest = out / "executed-source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    save(out / "executed-source/sha256.json", provenance)
    manifest = dict(command=[sys.executable, *sys.argv], environment=sys.prefix,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes=provenance, plan=PLAN, result_file=str(out / "report.json"),
        controls_root=str(args.controls_root.resolve()), selection_sha256=sha(args.controls_root / "selection.json"),
        cpu_budget=cpu, wall_budget=wall, memory_policy="pending trusted GPU configuration",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        data_version="untouched nominal KSC T2 claim partition; first reference qualification")
    save(out / "run-manifest.json", manifest)
    save(out / "report.json", dict(status="RUNNING", posterior_comparisons="NOT_RUN"))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = cpu - (time.process_time() - parent_cpu) - 1

    code, watchdog = run_bounded_worker([sys.executable, str(Path(__file__).resolve()), "--worker", str(out)],
        log_path=out / "worker.log", cpu_budget=child_cpu, wall_budget=max(1, wall - (time.monotonic() - started)))
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    charge = dict(cpu_worker_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
                 + time.process_time() - parent_cpu, gpu_seconds=time.monotonic() - started)
    save(out / "accounting.json", dict(charged=charge, charged_this_run=charge, exit_code=code,
                                      wall_seconds=charge["gpu_seconds"], watchdog=watchdog))
    report = json.loads((out / "report.json").read_text())
    if code and report.get("status") == "RUNNING":
        report.update(status="INFRASTRUCTURE_OR_RESOURCE_FAILURE", exit_code=code)
        save(out / "report.json", report)
    manifest = json.loads((out / "run-manifest.json").read_text())
    manifest.update(wall_seconds=charge["gpu_seconds"], cpu_worker_seconds=charge["cpu_worker_seconds"], exit_code=code)
    save(out / "run-manifest.json", manifest)
    if any(sha(ROOT / name) != digest for name, digest in provenance.items()):
        raise ValueError("source drift during posterior preparation")
    print(json.dumps(dict(status=report["status"], accounting=charge, output=str(out))), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
