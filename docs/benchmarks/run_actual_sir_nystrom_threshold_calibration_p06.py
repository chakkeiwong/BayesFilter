"""Run P06 actual-SIR Nystrom SVD fresh validation.

This runner implements the gated one-seed-at-a-time protocol in
docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-threshold-calibration-p06-svd-fresh-validation-subplan-2026-06-24.md.
It treats legacy paired-threshold benchmark exits as stochastic exceedances
when deterministic artifact checks pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
BENCHMARK = Path("docs/benchmarks/benchmark_actual_sir_nystrom_compiled_redo.py")
SUBPLAN = Path(
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-threshold-calibration-p06-svd-fresh-validation-subplan-2026-06-24.md"
)
P05_SUMMARY = Path(
    "docs/benchmarks/actual-sir-nystrom-threshold-calibration-p05-svd-core-tuning-summary-2026-06-24.json"
)
SUMMARY_PATH = Path(
    "docs/benchmarks/actual-sir-nystrom-threshold-calibration-p06-svd-validation-summary-2026-06-24.json"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-threshold-calibration-p06-svd-fresh-validation-result-2026-06-24.md"
)
P07_SUBPLAN_PASS = Path(
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-threshold-calibration-p07-evidence-package-subplan-2026-06-24.md"
)
P07_SUBPLAN_REPAIR = Path(
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-threshold-calibration-p07-repair-closeout-subplan-2026-06-24.md"
)
P07_SUBPLAN_BLOCKER = Path(
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-threshold-calibration-p07-blocker-repair-subplan-2026-06-24.md"
)

INITIAL_SEEDS = list(range(82968, 82982))
EXTENSION_SEEDS = list(range(82982, 82998))
FORBIDDEN_SEEDS = set(range(82920, 82951)) | set(range(82962, 82968))
TAU_COMPONENT = 0.03
T = 20
M = 9
TAU_TOTAL = 5.4
CP_PASS_GATE = 0.20
ALLOWED_LEGACY_PAIR_VETOES = {
    "paired:paired_log_likelihood_max_abs_delta",
    "paired:paired_log_likelihood_mean_abs_delta",
}


def _run_text(command: list[str], *, timeout: float = 10.0) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt_seed_stem(kind: str, seed: int) -> str:
    return (
        "actual-sir-nystrom-threshold-calibration-p06-svd-"
        f"{kind}-seed{seed}-r32-eps0p5-2026-06-24"
    )


def _paths_for(kind: str, seed: int) -> dict[str, Path]:
    stem = _fmt_seed_stem(kind, seed)
    return {
        "json": Path("docs/benchmarks") / f"{stem}.json",
        "markdown": Path("docs/benchmarks") / f"{stem}.md",
        "log": Path("docs/plans/logs") / f"{stem}.log",
    }


def _normal_approx_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "sample_sd": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def _binomial_cdf(n: int, k: int, p: float) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def _cp_upper_one_sided_95(n: int, k: int) -> float | None:
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _binomial_cdf(n, k, mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _selected_gpu_preflight() -> list[dict[str, str]]:
    text = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=10.0,
    )
    rows = []
    if text == "unavailable":
        return rows
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mib": parts[2],
                    "memory_total_mib": parts[3],
                    "utilization_gpu_percent": parts[4],
                }
            )
    return rows


def _benchmark_command(
    *,
    seed: int,
    kind: str,
    selected_physical_gpu: str,
    gpu_note: str,
    paths: dict[str, Path],
    row_timeout_seconds: int,
) -> list[str]:
    phase_kind = "VALIDATION" if kind == "validation" else "EXTENSION"
    return [
        "timeout",
        str(row_timeout_seconds),
        str(PYTHON),
        str(BENCHMARK),
        "--route",
        "both",
        "--batch-seeds",
        str(seed),
        "--time-steps",
        str(T),
        "--num-particles",
        "8192",
        "--transport-policy",
        "active-all",
        "--sinkhorn-iterations",
        "10",
        "--sinkhorn-epsilon",
        "1.0",
        "--annealed-scaling",
        "0.9",
        "--annealed-convergence-threshold",
        "0.001",
        "--row-chunk-size",
        "1024",
        "--col-chunk-size",
        "1024",
        "--particle-chunk-size",
        "1024",
        "--nystrom-diagnostics",
        "--nystrom-rank",
        "32",
        "--nystrom-epsilon",
        "0.5",
        "--nystrom-max-iterations",
        "160",
        "--nystrom-convergence-threshold",
        "0.0001",
        "--nystrom-core-solver",
        "svd_truncated",
        "--nystrom-core-rcond",
        "1e-6",
        "--nystrom-kernel-mode",
        "raw",
        "--nystrom-scaling-normalization",
        "none",
        "--history-mode",
        "value-only",
        "--warmups",
        "0",
        "--repeats",
        "1",
        "--dtype",
        "float32",
        "--tf32-mode",
        "enabled",
        "--jit-compile",
        "--device-scope",
        "visible",
        "--cuda-visible-devices",
        selected_physical_gpu,
        "--device",
        "/GPU:0",
        "--expect-device-kind",
        "gpu",
        "--selected-physical-gpu",
        selected_physical_gpu,
        "--gpu-selection-note",
        gpu_note,
        "--phase-id",
        f"ACTUAL-SIR-NYSTROM-THRESHOLD-CALIBRATION-P06-SVD-{phase_kind}-SEED{seed}",
        "--quiet",
        "--output",
        str(paths["json"]),
        "--markdown-output",
        str(paths["markdown"]),
    ]


def _row_by_route(artifact: dict[str, Any], route: str) -> dict[str, Any] | None:
    for row in artifact.get("rows", []):
        if row.get("route") == route:
            return row
    return None


def _classify_artifact(
    *,
    artifact: dict[str, Any],
    seed: int,
    kind: str,
    paths: dict[str, Path],
    returncode: int | None,
    elapsed_seconds: float | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    hard_vetoes = list(artifact.get("hard_vetoes") or [])
    nonlegacy_vetoes = [veto for veto in hard_vetoes if veto not in ALLOWED_LEGACY_PAIR_VETOES]
    if nonlegacy_vetoes:
        reasons.append(f"nonlegacy_hard_vetoes:{nonlegacy_vetoes}")
    if artifact.get("batch_seeds") != [seed]:
        reasons.append(f"seed_mismatch:{artifact.get('batch_seeds')}")
    if artifact.get("route_request") != "both":
        reasons.append(f"route_request_mismatch:{artifact.get('route_request')}")
    if artifact.get("routes_executed") != ["streaming", "nystrom"]:
        reasons.append(f"routes_executed_mismatch:{artifact.get('routes_executed')}")
    shape = artifact.get("shape") or {}
    expected_shape = {
        "batch_size": 1,
        "time_steps": T,
        "num_particles": 8192,
        "state_dim": 18,
        "obs_dim": M,
    }
    for key, expected in expected_shape.items():
        if shape.get(key) != expected:
            reasons.append(f"shape_{key}_mismatch:{shape.get(key)}")
    precision = artifact.get("precision") or {}
    if precision.get("active_dtype") != "float32" and precision.get("dtype") != "float32":
        reasons.append(f"dtype_mismatch:{precision}")
    if precision.get("tf32_execution_enabled") is not True:
        reasons.append("tf32_not_enabled")
    transport = artifact.get("transport") or {}
    expected_transport = {
        "transport_policy": "active-all",
        "nystrom_rank": 32,
        "nystrom_epsilon": 0.5,
        "nystrom_max_iterations": 160,
        "nystrom_convergence_threshold": 0.0001,
        "nystrom_core_solver": "svd_truncated",
        "nystrom_core_rcond": 1e-6,
        "nystrom_kernel_mode": "raw",
        "nystrom_scaling_normalization": "none",
    }
    for key, expected in expected_transport.items():
        observed = transport.get(key)
        if isinstance(expected, float):
            if observed is None or abs(float(observed) - expected) > 1.0e-12:
                reasons.append(f"transport_{key}_mismatch:{observed}")
        elif observed != expected:
            reasons.append(f"transport_{key}_mismatch:{observed}")
    manifest = artifact.get("run_manifest") or {}
    if manifest.get("selected_physical_gpu_argument") != "0":
        reasons.append(f"selected_physical_gpu_argument_mismatch:{manifest.get('selected_physical_gpu_argument')}")
    if manifest.get("cuda_visible_devices") != "0":
        reasons.append(f"cuda_visible_devices_mismatch:{manifest.get('cuda_visible_devices')}")
    if not manifest.get("logical_gpus"):
        reasons.append("logical_gpu_missing")
    selected = manifest.get("selected_physical_gpu") or {}
    if selected.get("index") != "0":
        reasons.append(f"selected_physical_gpu_metadata_mismatch:{selected}")
    streaming_row = _row_by_route(artifact, "streaming")
    nystrom_row = _row_by_route(artifact, "nystrom")
    if streaming_row is None or nystrom_row is None:
        reasons.append("missing_streaming_or_nystrom_row")
    for row_name, row in (("streaming", streaming_row), ("nystrom", nystrom_row)):
        if row is None:
            continue
        if row.get("status") != "PASS":
            reasons.append(f"{row_name}_row_status:{row.get('status')}")
        if row.get("hard_vetoes"):
            reasons.append(f"{row_name}_row_hard_vetoes:{row.get('hard_vetoes')}")
        devices = " ".join(str(device) for device in row.get("output_devices", []))
        if "GPU" not in devices.upper():
            reasons.append(f"{row_name}_gpu_output_missing:{row.get('output_devices')}")
    if nystrom_row is not None:
        if nystrom_row.get("finite_factors") is not True:
            reasons.append("nonfinite_nystrom_factors")
        if nystrom_row.get("finite_particles") is not True:
            reasons.append("nonfinite_nystrom_particles")
        if nystrom_row.get("nystrom_core_solver") != "svd_truncated":
            reasons.append(f"nystrom_core_solver_mismatch:{nystrom_row.get('nystrom_core_solver')}")
        if nystrom_row.get("nystrom_diagnostics_enabled") is not True:
            reasons.append("nystrom_diagnostics_missing")
        if nystrom_row.get("landmark_core_effective_rank_min") is None:
            reasons.append("svd_effective_rank_missing")
    paired = artifact.get("paired_comparability") or {}
    delta_by_seed = paired.get("log_likelihood_delta_by_seed")
    paired_total_abs_delta = paired.get("log_likelihood_max_abs_delta")
    if not isinstance(delta_by_seed, list) or len(delta_by_seed) != 1:
        reasons.append(f"paired_delta_missing_or_wrong_length:{delta_by_seed}")
        normalized_abs_delta = None
        exceeds = None
    elif paired_total_abs_delta is None:
        reasons.append("paired_total_abs_delta_missing")
        normalized_abs_delta = None
        exceeds = None
    else:
        normalized_abs_delta = abs(float(paired_total_abs_delta)) / float(T * M)
        exceeds = normalized_abs_delta > TAU_COMPONENT
    deterministic_valid = not reasons
    return {
        "seed": seed,
        "kind": kind,
        "json": str(paths["json"]),
        "markdown": str(paths["markdown"]),
        "log": str(paths["log"]),
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "benchmark_status": artifact.get("status"),
        "benchmark_hard_vetoes": hard_vetoes,
        "legacy_pair_vetoes_only": bool(hard_vetoes)
        and all(veto in ALLOWED_LEGACY_PAIR_VETOES for veto in hard_vetoes),
        "deterministic_valid": deterministic_valid,
        "deterministic_invalid_reasons": reasons,
        "paired_total_abs_delta": paired_total_abs_delta,
        "normalized_abs_delta": normalized_abs_delta,
        "exceeds_tau_component": exceeds,
        "max_row_residual": None if nystrom_row is None else nystrom_row.get("max_row_residual"),
        "max_column_residual": None if nystrom_row is None else nystrom_row.get("max_column_residual"),
        "finite_factors": None if nystrom_row is None else nystrom_row.get("finite_factors"),
        "finite_particles": None if nystrom_row is None else nystrom_row.get("finite_particles"),
        "nystrom_effective_rank_min": None
        if nystrom_row is None
        else nystrom_row.get("landmark_core_effective_rank_min"),
        "core_matrix_min": None if nystrom_row is None else nystrom_row.get("core_matrix_min"),
        "core_matrix_max": None if nystrom_row is None else nystrom_row.get("core_matrix_max"),
        "left_factor_min": None if nystrom_row is None else nystrom_row.get("left_factor_min"),
        "left_factor_max": None if nystrom_row is None else nystrom_row.get("left_factor_max"),
    }


def _run_or_reuse_row(
    *,
    seed: int,
    kind: str,
    selected_physical_gpu: str,
    gpu_note: str,
    row_timeout_seconds: int,
) -> dict[str, Any]:
    paths = _paths_for(kind, seed)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    command = _benchmark_command(
        seed=seed,
        kind=kind,
        selected_physical_gpu=selected_physical_gpu,
        gpu_note=gpu_note,
        paths=paths,
        row_timeout_seconds=row_timeout_seconds,
    )
    returncode: int | None = None
    elapsed: float | None = None
    if not paths["json"].exists():
        start = time.perf_counter()
        with paths["log"].open("w", encoding="utf-8") as log:
            log.write(f"# Command\n{' '.join(command)}\n\n")
            log.flush()
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        returncode = completed.returncode
    try:
        artifact = _load_json(paths["json"])
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "seed": seed,
            "kind": kind,
            "json": str(paths["json"]),
            "markdown": str(paths["markdown"]),
            "log": str(paths["log"]),
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "benchmark_status": None,
            "benchmark_hard_vetoes": [],
            "legacy_pair_vetoes_only": False,
            "deterministic_valid": False,
            "deterministic_invalid_reasons": [f"artifact_unparseable:{exc}"],
            "paired_total_abs_delta": None,
            "normalized_abs_delta": None,
            "exceeds_tau_component": None,
        }
    return _classify_artifact(
        artifact=artifact,
        seed=seed,
        kind=kind,
        paths=paths,
        returncode=returncode,
        elapsed_seconds=elapsed,
    )


def _status_for(rows: list[dict[str, Any]], *, stopped_for_futility: bool) -> tuple[str, str]:
    invalid = [row for row in rows if not row.get("deterministic_valid")]
    if invalid:
        return "P06_DETERMINISTIC_BLOCKER", "deterministic/artifact/GPU/policy validity failed"
    n_valid = len(rows)
    n_exceed = sum(1 for row in rows if row.get("exceeds_tau_component") is True)
    cp_upper = _cp_upper_one_sided_95(n_valid, n_exceed)
    if stopped_for_futility or n_exceed >= 3:
        return "P06_FAIL_TO_P07_REPAIR_OR_CLOSEOUT", "third exceedance futility stop"
    if n_valid == len(INITIAL_SEEDS) and n_exceed == 0 and cp_upper is not None and cp_upper <= CP_PASS_GATE:
        return "P06_PASS_TO_P07_EVIDENCE_PACKAGE", "initial 0/14 CP validation gate passed"
    if n_valid == len(INITIAL_SEEDS) and n_exceed in (1, 2):
        return "P06_CONTINUE_EXTENSION", "initial panel inconclusive; extension required"
    if n_valid == len(INITIAL_SEEDS) + len(EXTENSION_SEEDS):
        if cp_upper is not None and cp_upper <= CP_PASS_GATE:
            return "P06_PASS_TO_P07_EVIDENCE_PACKAGE", "30-row CP validation gate passed"
        return "P06_FAIL_TO_P07_REPAIR_OR_CLOSEOUT", "30-row CP validation gate failed"
    return "P06_RUNNING", "panel still running"


def _summary_payload(
    *,
    rows: list[dict[str, Any]],
    selected_physical_gpu: str,
    gpu_note: str,
    gpu_preflight_rows: list[dict[str, str]],
    status: str,
    status_reason: str,
    started_at: str,
    ended_at: str | None,
    row_timeout_seconds: int,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("deterministic_valid")]
    n_valid = len(valid_rows)
    n_exceed = sum(1 for row in valid_rows if row.get("exceeds_tau_component") is True)
    values = [
        float(row["normalized_abs_delta"])
        for row in valid_rows
        if row.get("normalized_abs_delta") is not None
    ]
    return {
        "schema_version": "actual_sir_nystrom_threshold_calibration_p06.v1",
        "status": status,
        "status_reason": status_reason,
        "date": "2026-06-24",
        "subplan": str(SUBPLAN),
        "p05_summary": str(P05_SUMMARY),
        "result": str(RESULT_PATH),
        "evidence_contract": {
            "question": "Do fresh disjoint validation seeds support SVD policy tau_component=0.03 under the exact one-sided 95% CP exceedance rule?",
            "fixed_harness": "same-artifact compiled streaming TF32 actual-SIR value-route comparator",
            "candidate": "rank=32, epsilon=0.5, kernel_mode=raw, scaling_normalization=none, core_solver=svd_truncated, core_rcond=1e-6",
            "primary_pass": "deterministic validity and one-sided 95% CP upper bound for Pr(abs(delta)/(T*M)>0.03) <= 0.20",
            "not_concluded": "no default readiness, posterior correctness, HMC readiness, statistical superiority, or broad Nystrom rejection",
        },
        "threshold": {
            "tau_component": TAU_COMPONENT,
            "T": T,
            "M": M,
            "tau_total": TAU_TOTAL,
            "cp_upper_gate": CP_PASS_GATE,
        },
        "seeds": {
            "initial": INITIAL_SEEDS,
            "extension": EXTENSION_SEEDS,
            "forbidden_prior_manifest": {
                "p1_p3": [82920, 82950],
                "p5": [82962, 82967],
            },
        },
        "counts": {
            "n_rows": len(rows),
            "n_deterministic_valid": n_valid,
            "n_deterministic_invalid": len(rows) - n_valid,
            "n_exceed_tau_component": n_exceed,
            "one_sided_95_cp_upper_exceedance_probability": _cp_upper_one_sided_95(n_valid, n_exceed),
        },
        "normalized_abs_delta_summary": _normal_approx_summary(values),
        "exceedance_seeds": [
            row["seed"] for row in valid_rows if row.get("exceeds_tau_component") is True
        ],
        "deterministic_invalid_rows": [
            row for row in rows if not row.get("deterministic_valid")
        ],
        "rows": rows,
        "run_manifest": {
            "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": _run_text(["git", "status", "--short"]),
            "python_executable": str(PYTHON),
            "runner_python": sys.executable,
            "python_version": platform.python_version(),
            "working_directory": str(ROOT),
            "selected_physical_gpu": selected_physical_gpu,
            "gpu_selection_note": gpu_note,
            "gpu_preflight_rows": gpu_preflight_rows,
            "row_timeout_seconds": row_timeout_seconds,
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "inference_status": {
            "hard_veto_screen": "PASS" if len(rows) == n_valid else "FAIL",
            "statistically_supported_ranking": "NO",
            "descriptive_only_differences": "normalized deltas, runtime, and residual magnitudes are descriptive except for the predeclared exceedance count screen",
            "default_readiness": "NO",
            "next_evidence_needed": "P07 evidence packaging if pass, otherwise repair/closeout planning",
        },
        "nonclaims": [
            "no default readiness claim",
            "no posterior correctness claim",
            "no HMC readiness claim",
            "no statistical superiority claim",
            "no cholesky-vs-SVD ranking claim",
            "no broad Nystrom rejection claim",
        ],
    }


def _write_result(summary: dict[str, Any], next_subplan: Path) -> None:
    counts = summary["counts"]
    rows = summary["rows"]
    status = summary["status"]
    cp_upper = counts.get("one_sided_95_cp_upper_exceedance_probability")
    lines = [
        "# P06 Result: SVD Fresh Validation",
        "",
        "Date: 2026-06-24",
        "",
        f"Status: `{status}`",
        "",
        "## Decision Table",
        "",
        "| Field | Decision |",
        "| --- | --- |",
        f"| Decision | `{summary['status_reason']}` |",
        (
            "| Primary criterion status | "
            f"`n_valid={counts['n_deterministic_valid']}`, "
            f"`n_exceed={counts['n_exceed_tau_component']}`, "
            f"`CP_upper={cp_upper}` |"
        ),
        (
            "| Veto diagnostic status | "
            f"`{summary['inference_status']['hard_veto_screen']}`: "
            f"{counts['n_deterministic_invalid']} deterministic-invalid rows |"
        ),
        (
            "| Main uncertainty | Fresh validation supports only the bounded value-route screen; "
            "it does not establish posterior correctness, HMC readiness, default readiness, or a ranking. |"
        ),
        f"| Next justified action | Drafted `{next_subplan}`. |",
        "| What is not being concluded | No default readiness, no posterior correctness, no HMC readiness, no statistical superiority, no cholesky-vs-SVD ranking, and no broad Nystrom rejection. |",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {summary['evidence_contract']['question']} |",
        f"| Fixed harness | {summary['evidence_contract']['fixed_harness']} |",
        f"| Candidate | {summary['evidence_contract']['candidate']} |",
        f"| Primary pass criterion | {summary['evidence_contract']['primary_pass']} |",
        "| Veto diagnostics | Deterministic invalidity, malformed/missing artifact, GPU/TF32/shape/policy mismatch, missing SVD metadata, seed overlap, or third-exceedance futility. |",
        "| Explanatory diagnostics | Runtime, residual magnitudes, SVD factor/core diagnostics, and observed normalized deltas. |",
        f"| Artifact | `{SUMMARY_PATH}` |",
        "",
        "## Validation Results",
        "",
        "| Seed | Panel | Normalized abs delta | Exceeds `0.03` | Deterministic status |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        status_text = "PASS" if row.get("deterministic_valid") else "FAIL"
        exceeds = row.get("exceeds_tau_component")
        if exceeds is True:
            exceeds_text = "YES"
        elif exceeds is False:
            exceeds_text = "NO"
        else:
            exceeds_text = "N/A"
        lines.append(
            f"| `{row['seed']}` | `{row['kind']}` | `{row.get('normalized_abs_delta')}` | "
            f"`{exceeds_text}` | `{status_text}` |"
        )
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| Row | Status |",
            "| --- | --- |",
            f"| Hard veto screen | `{summary['inference_status']['hard_veto_screen']}` |",
            "| Statistically supported ranking | `NO` |",
            "| Descriptive-only differences | Runtime, residuals, and non-gated delta magnitudes are descriptive only. |",
            "| Default-readiness | `NO` |",
            f"| Next evidence needed | {summary['inference_status']['next_evidence_needed']} |",
            "",
            "## Run Manifest",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Git commit | `{summary['run_manifest']['git_commit']}` |",
            f"| Python | `{summary['run_manifest']['python_executable']}` |",
            "| Shape | `T=20`, `N=8192`, `state_dim=18`, `obs_dim=9` |",
            "| Fixed policy | `rank=32`, `epsilon=0.5`, `kernel_mode=raw`, `scaling_normalization=none`, `core_solver=svd_truncated`, `core_rcond=1e-6` |",
            "| Dtype/precision | `float32`, TF32 enabled |",
            f"| Actual GPU | Physical GPU{summary['run_manifest']['selected_physical_gpu']} selected; `CUDA_VISIBLE_DEVICES={summary['run_manifest']['selected_physical_gpu']}` remapped it to TensorFlow `/GPU:0` |",
            f"| Summary artifact | `{SUMMARY_PATH}` |",
            f"| Started | `{summary['run_manifest']['started_at']}` |",
            f"| Ended | `{summary['run_manifest']['ended_at']}` |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: the fixed paired value-route screen can pass while later posterior, HMC, model-suite, or default-policy gates fail.",
            "",
            "What would overturn this result: a deterministic artifact/GPU/policy validity problem in the row artifacts, a seed-overlap discovery, or a later fresh validation panel with a predeclared statistical veto.",
            "",
            "Weakest evidence: P06 is scoped to the bounded actual-SIR value-route screen and does not test end-to-end posterior correctness.",
            "",
            "## Handoff",
            "",
            f"Next subplan: `{next_subplan}`",
            "",
        ]
    )
    RESULT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_p07_subplan(summary: dict[str, Any]) -> Path:
    status = summary["status"]
    if status == "P06_PASS_TO_P07_EVIDENCE_PACKAGE":
        path = P07_SUBPLAN_PASS
        objective = "Package the P06 bounded value-route validation evidence and decide the next non-promotion evidence gap."
        entry = "P06 passed the frozen SVD bounded value-route CP screen."
        handoff = "P07_CLOSEOUT_READY_FOR_NEXT_MODEL_SUITE_OR_DEFAULT_GAP_PLAN"
        stop = "Stop before any default promotion, posterior correctness, HMC readiness, or superiority claim."
    elif status == "P06_FAIL_TO_P07_REPAIR_OR_CLOSEOUT":
        path = P07_SUBPLAN_REPAIR
        objective = "Close out the failed P06 validation screen and plan the smallest repair or alternative-policy diagnostic."
        entry = "P06 deterministic validity passed but the frozen CP exceedance screen failed or hit futility."
        handoff = "P07_REPAIR_PLAN_REQUIRED_BEFORE_MORE_GPU_SWEEPS"
        stop = "Stop before changing thresholds or tuning a new policy without a reviewed repair subplan."
    else:
        path = P07_SUBPLAN_BLOCKER
        objective = "Analyze the P06 deterministic blocker and specify a repair-only continuation."
        entry = "P06 hit a deterministic/artifact/GPU/policy blocker."
        handoff = "P07_BLOCKER_REPAIR_REQUIRED"
        stop = "Stop before treating blocked rows as stochastic non-exceedances or validation evidence."
    lines = [
        "# P07 Subplan: Evidence Package Or Repair Closeout",
        "",
        "Date: 2026-06-24",
        "",
        "Status: `DRAFT_LOCAL_REVIEWED`",
        "",
        "## Phase Objective",
        "",
        objective,
        "",
        "## Entry Conditions Inherited From Previous Phase",
        "",
        f"- {entry}",
        f"- P06 status: `{status}`.",
        f"- P06 summary artifact: `{SUMMARY_PATH}`.",
        f"- P06 result artifact: `{RESULT_PATH}`.",
        "",
        "## Required Artifacts",
        "",
        "- P06 aggregate summary JSON.",
        "- P06 result markdown.",
        "- This P07 subplan.",
        "- Any later closeout or repair result must be a separate reviewed artifact.",
        "",
        "## Required Checks, Tests, And Reviews",
        "",
        "- Parse the P06 summary JSON.",
        "- Verify deterministic validity, exceedance counts, CP upper bound, and status agree between summary and result.",
        "- Verify no default, posterior, HMC, superiority, or broad Nystrom claim is made.",
        "- Review boundary safety before any further experiment or promotion action.",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        "| Question | What is the next justified action after P06 under the statistical evidence discipline? |",
        "| Baseline/comparator | P06 uses the fixed same-artifact compiled streaming TF32 value-route comparator; P07 does not add a new numerical comparison. |",
        "| Primary criterion | Consistency of P06 summary/result with the predeclared P06 gate. |",
        "| Veto diagnostics | Missing/malformed P06 artifacts, inconsistent counts/status, unsupported promotion claims, or any proposed threshold/default change without a reviewed plan. |",
        "| Explanatory diagnostics | P06 row deltas, CP upper bound, runtime, and residual diagnostics. |",
        "| Not concluded | No default readiness, no posterior correctness, no HMC readiness, no statistical superiority, and no broad Nystrom rejection. |",
        "",
        "## Forbidden Claims And Actions",
        "",
        "- Do not promote SVD/Nystrom to a new default from P07 alone.",
        "- Do not claim posterior correctness, HMC readiness, or statistical superiority.",
        "- Do not change `tau_component=0.03` or the CP gate in P07.",
        "- Do not launch another tuning or validation sweep without a new reviewed subplan.",
        "",
        "## Exact Next-Phase Handoff Conditions",
        "",
        f"- `{handoff}`: P07 local checks confirm P06 artifacts and boundary claims are internally consistent.",
        "",
        "## Stop Conditions",
        "",
        f"- {stop}",
        "- P06 summary/result mismatch.",
        "- Required artifact missing or malformed.",
        "- Any next step would require human product/scientific authorization.",
        "",
        "## Local Consistency Review",
        "",
        "- Consistency: `PASS`; P07 scope is closeout/repair planning, not promotion.",
        "- Correctness: `PASS`; it preserves the P06 statistical interpretation boundary.",
        "- Feasibility: `PASS`; checks are artifact parsing and boundary review.",
        "- Artifact coverage: `PASS`; P06 summary, result, row artifacts, and logs are named.",
        "- Boundary safety: `PASS`; forbidden claims/actions block default and scientific overclaiming.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _preflight(args: argparse.Namespace) -> list[dict[str, str]]:
    if not P05_SUMMARY.exists():
        raise SystemExit(f"missing P05 summary: {P05_SUMMARY}")
    p05 = _load_json(P05_SUMMARY)
    if p05.get("status") != "P05_NOMINATE_SVD_TO_P06":
        raise SystemExit(f"P05 status mismatch: {p05.get('status')}")
    planned = set(INITIAL_SEEDS) | set(EXTENSION_SEEDS)
    overlap = sorted(planned & FORBIDDEN_SEEDS)
    if overlap:
        raise SystemExit(f"P06 seed overlap with forbidden manifest: {overlap}")
    if args.selected_physical_gpu != "0":
        raise SystemExit("P06 subplan froze physical GPU0 after trusted preflight; use --selected-physical-gpu 0")
    gpu_rows = _selected_gpu_preflight()
    if not gpu_rows:
        raise SystemExit("trusted nvidia-smi preflight did not return GPU rows")
    return gpu_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--selected-physical-gpu", default="0")
    parser.add_argument("--row-timeout-seconds", type=int, default=900)
    parser.add_argument("--stop-after", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = parse_args()
    started_at = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    gpu_rows = _preflight(args)
    gpu_note = (
        "GPU0 selected because trusted preflight showed GPU1 unsuitable/saturated; "
        "CUDA_VISIBLE_DEVICES=0 remaps selected physical GPU0 to TensorFlow /GPU:0 "
        "for the frozen P06 panel."
    )
    rows: list[dict[str, Any]] = []
    stopped_for_futility = False
    ended_at: str | None = None

    def checkpoint(status: str, reason: str) -> None:
        payload = _summary_payload(
            rows=rows,
            selected_physical_gpu=args.selected_physical_gpu,
            gpu_note=gpu_note,
            gpu_preflight_rows=gpu_rows,
            status=status,
            status_reason=reason,
            started_at=started_at,
            ended_at=ended_at,
            row_timeout_seconds=args.row_timeout_seconds,
        )
        _write_json(SUMMARY_PATH, payload)

    print("P06 preflight passed; launching one row at a time on physical GPU0.", flush=True)
    for seed in INITIAL_SEEDS:
        row = _run_or_reuse_row(
            seed=seed,
            kind="validation",
            selected_physical_gpu=args.selected_physical_gpu,
            gpu_note=gpu_note,
            row_timeout_seconds=args.row_timeout_seconds,
        )
        rows.append(row)
        status, reason = _status_for(rows, stopped_for_futility=stopped_for_futility)
        checkpoint(status, reason)
        print(
            f"seed={seed} valid={row.get('deterministic_valid')} "
            f"norm={row.get('normalized_abs_delta')} exceed={row.get('exceeds_tau_component')} "
            f"status={status}",
            flush=True,
        )
        if not row.get("deterministic_valid"):
            break
        if sum(1 for item in rows if item.get("exceeds_tau_component") is True) >= 3:
            stopped_for_futility = True
            break
        if args.stop_after is not None and len(rows) >= args.stop_after:
            break

    status, reason = _status_for(rows, stopped_for_futility=stopped_for_futility)
    if status == "P06_CONTINUE_EXTENSION" and not stopped_for_futility:
        for seed in EXTENSION_SEEDS:
            row = _run_or_reuse_row(
                seed=seed,
                kind="extension",
                selected_physical_gpu=args.selected_physical_gpu,
                gpu_note=gpu_note,
                row_timeout_seconds=args.row_timeout_seconds,
            )
            rows.append(row)
            status, reason = _status_for(rows, stopped_for_futility=stopped_for_futility)
            checkpoint(status, reason)
            print(
                f"seed={seed} valid={row.get('deterministic_valid')} "
                f"norm={row.get('normalized_abs_delta')} exceed={row.get('exceeds_tau_component')} "
                f"status={status}",
                flush=True,
            )
            if not row.get("deterministic_valid"):
                break
            if sum(1 for item in rows if item.get("exceeds_tau_component") is True) >= 3:
                stopped_for_futility = True
                status, reason = _status_for(rows, stopped_for_futility=stopped_for_futility)
                break
            if args.stop_after is not None and len(rows) >= args.stop_after:
                break

    ended_at = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    status, reason = _status_for(rows, stopped_for_futility=stopped_for_futility)
    summary = _summary_payload(
        rows=rows,
        selected_physical_gpu=args.selected_physical_gpu,
        gpu_note=gpu_note,
        gpu_preflight_rows=gpu_rows,
        status=status,
        status_reason=reason,
        started_at=started_at,
        ended_at=ended_at,
        row_timeout_seconds=args.row_timeout_seconds,
    )
    _write_json(SUMMARY_PATH, summary)
    if status.startswith("P06_") and status not in {"P06_RUNNING", "P06_CONTINUE_EXTENSION"}:
        next_subplan = _write_p07_subplan(summary)
        _write_result(summary, next_subplan)
    print(f"P06 final status: {status} ({reason})", flush=True)
    if status in {"P06_DETERMINISTIC_BLOCKER", "P06_BLOCKED"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
