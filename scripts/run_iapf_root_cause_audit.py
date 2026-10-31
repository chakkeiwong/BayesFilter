"""Bounded CPU-only diagnostic of saved iAPF fitting failures; stdlib only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03"
OUT = ROOT / "docs/plans/artifacts/iapf-root-cause-audit-20260926-01"
PLAN = ROOT / "docs/plans/iapf-root-cause-audit-20260926.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = [json.loads(p.read_text()) for p in OUT.glob("attempt*/manifest.json")]
    if len(prior) >= 2 or any(m["status"] == "RUNNING" for m in prior):
        raise SystemExit("Audit attempt limit or unfinished diagnostic")
    remaining = 600 - 60 - sum(m["wall_seconds"] for m in prior)
    if remaining <= 0:
        raise SystemExit("CPU diagnostic budget exhausted")
    attempt = OUT / f"attempt{len(prior)+1:02d}"
    attempt.mkdir(parents=True, exist_ok=False)
    sources = attempt / "sources"
    sources.mkdir()
    selected = [OLD / "sources/reference_iapf_paper.R",
                OLD / "sources/reference_iapf_scale_aware_fit.R",
                ROOT / "docs/benchmarks/diagnose_iapf_current_fit_root_causes.R",
                Path(__file__).resolve(), PLAN]
    for path in selected:
        shutil.copy2(path, sources / path.name)
    old_manifest = json.loads((OLD / "manifest.json").read_text())
    inputs = sorted(OLD.glob("*.rds"))
    historical_sources = {Path(k).name: v for k, v in old_manifest["source_sha256"].items()}
    for path in selected[:2]:
        if sha(path) != historical_sources[path.name]:
            raise SystemExit(f"Historical source hash mismatch: {path}")
    for path in inputs:
        if sha(path) != old_manifest["output_sha256"][path.name]:
            raise SystemExit(f"Historical input hash mismatch: {path}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = ["Rscript", "--vanilla", str(sources / "diagnose_iapf_current_fit_root_causes.R"),
               str(sources), str(OLD), str(attempt)]
    manifest = dict(status="RUNNING", start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    command=command, plan=str(PLAN), result=str(OUT / "result.md"),
                    environment={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                    python_executable=os.sys.executable, R_version=subprocess.check_output(["Rscript", "--version"], text=True, stderr=subprocess.STDOUT).strip(),
                    hardware="CPU-only independent R/FP64; GPU deliberately hidden",
                    data_version="Frozen attempt03 paper Section 5.2 model d5 T100 N1000",
                    seeds=[9262001, 9262002], gpu_seconds=0,
                    input_sha256={str(p): sha(p) for p in inputs},
                    source_sha256={str(p): sha(p) for p in selected},
                    historical_manifest_sha256=sha(OLD / "manifest.json"),
                    historical_status=old_manifest.get("status"),
                    reconnaissance_wall_charge_seconds=60, timeout_seconds=min(240, remaining))
    manifest_path = attempt / "manifest.json"
    def write():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write()
    started = time.monotonic()
    try:
        with (attempt / "run.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=manifest["timeout_seconds"], check=False)
        manifest.update(status="COMPLETE" if completed.returncode == 0 else "FAILED", exit_code=completed.returncode)
    except subprocess.TimeoutExpired:
        manifest.update(status="TIMEOUT", exit_code=None)
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["output_sha256"] = {str(p.relative_to(attempt)): sha(p) for p in attempt.rglob("*")
                                    if p.is_file() and p != manifest_path}
        write()
    print(json.dumps({k: manifest[k] for k in ("status", "exit_code", "wall_seconds")}))
    print(str(manifest_path))
    if manifest["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
