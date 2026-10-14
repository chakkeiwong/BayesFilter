"""Bounded supervisor for diagnostic author/TF neural-angle compatibility."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PLAN = ROOT / "docs/plans/controlled-smc-kdm-neural-angle-conditioning-20261007.md"
RESULT = "docs/plans/controlled-smc-kdm-neural-angle-conditioning-result-20261007.md"
AUTHOR = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
SOURCE_PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
TF_PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
GPU = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
CAPS = {"cpu": 600., "gpu": 900.}
FIXTURES = [("fresh_a", 202610171, 256), ("fresh_b", 202610181, 288)]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.parent != BASE or not out.name.startswith("kdm-neural-angle-20261007-"):
        raise ValueError("Use the phase's unique versioned output root")
    attempts = sorted(BASE.glob("kdm-neural-angle-20261007-*/manifest.json"))
    if len(attempts) >= 3:
        raise RuntimeError("Three-attempt phase budget exhausted")
    used = dict.fromkeys(CAPS, 0.)
    for path in attempts:
        previous = json.loads(path.read_text())
        if previous["status"] == "RUNNING":
            raise RuntimeError("An angle diagnostic is already active")
        for resource in CAPS:
            used[resource] += previous["resource_wall_seconds"][resource]
    full_tf = json.loads((BASE / "kdm-complete-training-20261007-04/manifest.json").read_text())
    if full_tf["status"] != "COMPLETED":
        raise RuntimeError("The full TF training must finish before this GPU diagnostic")
    if any(CAPS[key] - used[key] < 15 for key in CAPS):
        raise RuntimeError("Insufficient phase budget")
    out.mkdir(exist_ok=False)
    source_helper = ROOT / "docs/benchmarks/reference_kdm_neural_conditioning.py"
    gpu_helper = ROOT / "docs/benchmarks/diagnose_kdm_neural_angle_tf.py"
    files = [Path(__file__), source_helper, gpu_helper, PLAN,
             ROOT / "bayesfilter/replication/kdm_neural_replay_tf.py",
             ROOT / "bayesfilter/replication/kdm_normal_von_mises_tf.py"]
    snapshots = out / "sources"
    snapshots.mkdir()
    for path in files:
        shutil.copyfile(path, snapshots / path.name)
    start = time.monotonic()
    manifest = dict(schema="kdm_neural_angle_supervisor_v1", status="RUNNING",
        command=[sys.executable] + sys.argv, supervisor_pid=os.getpid(),
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        plan_file=str(PLAN.relative_to(ROOT)), result_file=RESULT,
        source_hashes={str(path.relative_to(ROOT)): sha(path) for path in files},
        cpu_gpu_status="GPU-hidden one-thread author reference; trusted TF GPU/XLA",
        random_seeds=[20261007, 20261008, 202610171, 202610172, 202610181, 202610182],
        fixtures=FIXTURES, data_version="development replay; not untouched qualification",
        phase_budget_seconds=CAPS, prior_charged_seconds=used,
        resource_wall_seconds=dict.fromkeys(CAPS, 0.), executions=[], replication="INCOMPLETE")
    def save():
        write(out / "manifest.json", manifest)
    save()
    def run(command, resource, name, env):
        remaining = CAPS[resource] - used[resource] - manifest["resource_wall_seconds"][resource]
        if remaining < 5:
            raise RuntimeError("Phase budget exhausted for " + resource)
        execution = dict(command=command, resource=resource, timeout_seconds=remaining,
            environment={k: env.get(k) for k in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
              "OMP_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_FORCE_GPU_ALLOW_GROWTH"]},
            log=str((out / (name + ".log")).relative_to(ROOT)), status="RUNNING")
        manifest["executions"].append(execution)
        save()
        tick = time.monotonic()
        process = None
        try:
            with (out / (name + ".log")).open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                execution["pid"] = process.pid
                save()
                try:
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    execution["termination_reason"] = "RESOURCE_BUDGET"
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 124
            execution["returncode"] = code
            execution["status"] = "COMPLETED" if code == 0 else "FAILED"
            if code:
                raise RuntimeError(name + " failed with return code " + str(code))
        finally:
            elapsed = time.monotonic() - tick
            execution["wall_seconds"] = elapsed
            manifest["resource_wall_seconds"][resource] += elapsed
            save()

    try:
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1")
        source_env = dict(env, CUDA_VISIBLE_DEVICES="-1",
                          LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
        components = [("preserved", BASE / "kdm-neural-20261006-06/author.json")]
        for name, seed, offset in FIXTURES:
            path = out / (name + ".json")
            run([SOURCE_PYTHON, "-I", "-B", str(source_helper), "--output", str(path),
                 "--batch-size", "4", "--updates", "2", "--seed", str(seed),
                 "--data-offset", str(offset)], "cpu", name, source_env)
            components.append((name, path))
        merged = None
        all_cases = []
        provenance = []
        for name, path in components:
            payload = json.loads(path.read_text())
            if not payload["gpu_intentionally_hidden"]:
                raise RuntimeError("CPU reference did not hide GPUs")
            for field in ["checkpoint", "dataset"]:
                if sha(payload[field]) != payload[field + "_sha256"]:
                    raise RuntimeError("Changed " + field)
            for relative, expected in payload["author_hashes"].items():
                if sha(AUTHOR / relative) != expected:
                    raise RuntimeError("Changed author source " + relative)
            if merged is None:
                merged = {key: value for key, value in payload.items() if key != "cases"}
            else:
                for field in ["author_commit", "checkpoint_sha256", "dataset_sha256", "training_config"]:
                    if payload[field] != merged[field]:
                        raise RuntimeError("Mismatched replay baseline " + field)
            for case in payload["cases"]:
                case["fixture_id"] = name
                if len(case["updates"]) != 2:
                    raise RuntimeError("Missing actual optimizer updates")
                for update in case["updates"]:
                    capture = update["captured"]
                    if len(capture["proposal_before_normalization"]) != len(capture["outputs"]):
                        raise RuntimeError("Raw/normalized proposal time alignment mismatch")
                all_cases.append(case)
            provenance.append(dict(fixture_id=name, path=str(path.relative_to(ROOT)), sha256=sha(path),
                seed=payload.get("seed", 20261007), data_offset=payload.get("data_offset", 0),
                batch_size=payload.get("batch_size", 2)))
        merged.update(cases=all_cases, diagnostic_seeds=manifest["random_seeds"],
                      reference_components=provenance)
        reference = out / "reference.json"
        write(reference, merged)
        manifest.update(reference_components=provenance, reference_sha256=sha(reference),
                        checkpoint_sha256=merged["checkpoint_sha256"], data_sha256=merged["dataset_sha256"])
        save()
        gpu_env = dict(env, CUDA_VISIBLE_DEVICES=GPU, TF_FORCE_GPU_ALLOW_GROWTH="true")
        run([TF_PYTHON, "-B", str(gpu_helper), "--output", str(out / "gpu"),
             "--reference", str(reference)], "gpu", "gpu", gpu_env)
        decision = json.loads((out / "gpu/decision.json").read_text())
        child_manifest = json.loads((out / "gpu/manifest.json").read_text())
        comparison = json.loads((out / "gpu/comparison.json").read_text())
        manifest.update(status="COMPLETED", decision=decision,
            checks_total=comparison["checks_total"], failed_checks=len(comparison["failed_checks"]),
            memory_policy=comparison["memory_policy"], physical_devices=comparison["physical_devices"],
            jit_compile=comparison["jit_compile"], tf32=comparison["tf32"],
            gpu_manifest_sha256=sha(out / "gpu/manifest.json"))
        if decision["status"] != "PASS":
            raise RuntimeError("Diagnostic failed to complete")
        for path in files:
            if sha(path) != manifest["source_hashes"][str(path.relative_to(ROOT))]:
                raise RuntimeError("Source/helper changed during diagnostic")
        print(json.dumps(dict(status=manifest["status"], decision=decision,
                              checks=manifest["checks_total"], failed=manifest["failed_checks"])))
        return 0
    except Exception:
        manifest.update(status="FAILED", error=traceback.format_exc())
        (out / "error.txt").write_text(manifest["error"])
        print(manifest["error"])
        return 1
    finally:
        manifest["wall_seconds"] = time.monotonic() - start
        manifest["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["outputs"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                              if p.is_file() and p.name != "manifest.json"}
        save()

if __name__ == "__main__":
    raise SystemExit(main())
