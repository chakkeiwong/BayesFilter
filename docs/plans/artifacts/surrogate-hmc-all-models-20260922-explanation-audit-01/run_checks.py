"""Bounded CPU mechanics/wiring checks for the explanatory audit, not sampling."""

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
TESTS = [
    "tests/test_neural_force_hmc.py::test_proposal_with_momentum_flip_is_an_involution",
    "tests/test_neural_force_hmc.py::test_full_energy_trace_and_cached_endpoint_call_count",
    "tests/test_neural_force_hmc.py::test_omitted_kinetic_rule_fails_known_swap_counterexample",
    "tests/test_neural_force_hmc.py::test_batch_permutation_replay_shape_and_dtype",
    "tests/test_neural_force_hmc.py::test_nonfinite_force_uses_finite_fallback_and_records_telemetry",
    "tests/test_neural_force_hmc.py::test_nonlinear_chart_requires_and_includes_log_jacobian",
    "tests/test_neural_force_hmc.py::test_kinetic_energy_uses_configured_inverse_mass",
    "tests/test_surrogate_hmc_linear_consumer.py::test_consumer_reaches_canonical_batch_and_preserves_sign_chart_and_status",
    "tests/test_surrogate_hmc_sv_consumer.py::test_consumer_probit_prior_status_and_actual_canonical_call",
    "tests/test_surrogate_hmc_sv_force_guard.py",
    str(OUT / "test_one_step_identity.py"),
]


def cpu_used(kind):
    usage = resource.getrusage(kind)
    return usage.ru_utime + usage.ru_stime


def child_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (110, 110))


def main():
    if (OUT / "checks.json").exists() or (OUT / "checks.log").exists():
        raise SystemExit("Preserve the completed attempt; use a new artifact directory.")
    resource.setrlimit(resource.RLIMIT_CPU, (10, 130))
    started = time.monotonic()
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2",
        OMP_NUM_THREADS="2",
        TF_CPP_MIN_LOG_LEVEL="2",
        BAYESFILTER_PRELOAD_CUSTOM_OP="0",
    )
    command = [sys.executable, "-m", "pytest", "-q", *TESTS]
    manifest = {
        "role": "focused CPU-only mechanics and mocked consumer wiring; no posterior experiment",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": command,
        "python": sys.executable,
        "gpu_status": "intentionally hidden before framework import (CUDA_VISIBLE_DEVICES=-1)",
        "threads": {"intraop": 2, "interop": 2, "omp": 2},
        "budget": {"child_cpu_seconds": 110, "parent_cpu_soft_seconds": 10, "child_wall_seconds": 150},
        "seed_policy": "existing deterministic test fixtures and explicit fixed tensors in the new identity check",
        "plan_and_result": "docs/plans/surrogate-hmc-algorithm-code-literature-audit-2026-09-22.md",
        "data_version": "N/A: mathematical and mocked wiring fixtures",
        "sources_sha256": {},
    }
    for relative in [
        "bayesfilter/inference/neural_force_hmc.py",
        "bayesfilter/inference/surrogate_hmc_equivalence.py",
        "scripts/run_surrogate_hmc_sv.py",
        "bayesfilter/testing/surrogate_hmc_linear_consumer.py",
        "bayesfilter/testing/surrogate_hmc_sv_consumer.py",
        "bayesfilter/testing/surrogate_hmc_sv_inference_consumer.py",
    ]:
        manifest["sources_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    with (OUT / "checks.log").open("w") as log:
        try:
            result = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                timeout=150, preexec_fn=child_limits, check=False,
            )
            manifest["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            manifest["exit_code"] = 124
    manifest["wall_seconds"] = time.monotonic() - started
    manifest["cpu_seconds"] = cpu_used(resource.RUSAGE_SELF) + cpu_used(resource.RUSAGE_CHILDREN)
    manifest["gpu_wall_seconds"] = 0.0
    (OUT / "checks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("exit_code", "wall_seconds", "cpu_seconds", "gpu_wall_seconds")}))
    return 0 if manifest["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
