"""Bounded CPU reference checks of existing structural AR(2) filtering."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TEST_FILE = "tests/test_structural_svd_sigma_point_tf.py"
TESTS = [
    TEST_FILE + "::test_structural_svd_cubature_matches_linear_qr_on_affine_lgssm",
    TEST_FILE + "::test_structural_svd_ukf_matches_linear_qr_on_affine_lgssm",
    TEST_FILE + "::test_structural_svd_sigma_points_preserve_deterministic_completion_support",
]
SOURCES = [TEST_FILE, "bayesfilter/structural_tf.py",
           "bayesfilter/nonlinear/sigma_points_tf.py",
           "bayesfilter/highdim/ledh_canonical_score_tf.py"]


def child_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (110, 110))


def cpu_seconds(kind):
    usage = resource.getrusage(kind)
    return usage.ru_utime + usage.ru_stime


def main():
    if (OUT / "checks.json").exists() or (OUT / "checks.log").exists():
        raise SystemExit("Preserve completed evidence; use a new output directory.")
    resource.setrlimit(resource.RLIMIT_CPU, (10, 120))
    started = time.monotonic()
    environment = dict(os.environ)
    settings = dict(CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTRAOP_THREADS="2",
                    TF_NUM_INTEROP_THREADS="2", OMP_NUM_THREADS="2",
                    TF_CPP_MIN_LOG_LEVEL="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    environment.update(settings)
    command = [sys.executable, "-m", "pytest", "-q", *TESTS]
    manifest = {
        "role": "CPU-only existing structural AR2 reference tests; no DCTR or HMC execution",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": command, "python": sys.executable, "environment": settings,
        "gpu_status": "intentionally hidden before framework import",
        "jit_compile": "not requested by this reference-test command; no GPU/XLA claim",
        "data_version": "deterministic in-file affine AR2 fixtures, bound by test-file SHA256",
        "random_seeds": "N/A: deterministic fixed fixtures",
        "budget": {"child_cpu_seconds": 110, "parent_cpu_soft_seconds": 10,
                   "child_wall_seconds": 150},
        "plan_file": "docs/plans/surrogate-hmc-algorithm-code-literature-audit-2026-09-22.md",
        "result_file": "docs/plans/surrogate-hmc-algorithm-code-literature-audit-2026-09-22.md",
        "artifact_paths": [str(OUT / "checks.log"), str(OUT / "checks.json")],
        "selected_source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in SOURCES},
    }
    with (OUT / "checks.log").open("x") as log:
        try:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=150,
                                    preexec_fn=child_limits, check=False)
            manifest["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            manifest["exit_code"] = 124
    manifest["wall_seconds"] = time.monotonic() - started
    manifest["cpu_seconds"] = cpu_seconds(resource.RUSAGE_SELF) + cpu_seconds(resource.RUSAGE_CHILDREN)
    manifest["gpu_wall_seconds"] = 0.0
    manifest["selected_sources_unchanged"] = all(
        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        for path, digest in manifest["selected_source_sha256"].items())
    with (OUT / "checks.json").open("x") as result_file:
        result_file.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in
                     ("exit_code", "wall_seconds", "cpu_seconds", "gpu_wall_seconds",
                      "selected_sources_unchanged")}))
    return 0 if manifest["exit_code"] == 0 and manifest["selected_sources_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
