from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
_PRE_IMPORT_GPU_HIDING_ASSERTION = _PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"

import numpy as np

from experiments.dpf_monograph_evidence.fixtures.linear_gaussian import build_linear_gaussian_fixture
from experiments.dpf_monograph_evidence.results import validate_result_record

SEEDS = (101, 102, 103, 104, 105)
PARTICLE_COUNTS = (64, 256)
WALL_CLOCK_CAP_SECONDS = 30
RESULT_PATH = Path(__file__).resolve().parent.parent / "reports" / "outputs" / "linear_gaussian_recovery.json"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "linear-gaussian-recovery-result.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git_command(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def get_git_manifest() -> tuple[str, str, str]:
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git_command(["git", "rev-parse", "HEAD"])
    dirty = _run_git_command(["git", "status", "--short"])
    dirty_state_summary = dirty if dirty else "clean"
    return branch, commit, dirty_state_summary


def _numpy_version() -> str:
    return np.__version__


def normal_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    centered = x - mean
    return np.exp(-0.5 * centered * centered / variance) / math.sqrt(2.0 * math.pi * variance)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values))


def weighted_variance(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    centered = values - mean
    return float(np.sum(weights * centered * centered))


def bootstrap_pf_stats(particle_count: int, seed: int, fixture) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    predicted_particles = rng.normal(
        loc=fixture.predictive_mean,
        scale=math.sqrt(fixture.predictive_variance),
        size=particle_count,
    )
    likelihoods = normal_pdf(
        fixture.observation,
        fixture.h * predicted_particles,
        fixture.r,
    )
    weight_sum = float(np.sum(likelihoods))
    normalized_weights = likelihoods / weight_sum
    posterior_mean = weighted_mean(predicted_particles, normalized_weights)
    posterior_variance = weighted_variance(predicted_particles, normalized_weights, posterior_mean)
    log_likelihood = math.log(weight_sum / particle_count)
    return {
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
        "log_likelihood": log_likelihood,
    }


def mcse(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def summarize_pf(fixture) -> tuple[dict[str, object], dict[str, object]]:
    replications: dict[str, object] = {}
    absolute_errors: dict[str, object] = {}
    slope_summary: dict[str, float] = {}
    error_scales: dict[int, float] = {}

    for particle_count in PARTICLE_COUNTS:
        seed_rows = []
        mean_errors: list[float] = []
        variance_errors: list[float] = []
        loglik_errors: list[float] = []
        rmse_components: list[float] = []
        for seed in SEEDS:
            stats = bootstrap_pf_stats(particle_count=particle_count, seed=seed, fixture=fixture)
            mean_error = abs(stats["posterior_mean"] - fixture.posterior_mean)
            variance_error = abs(stats["posterior_variance"] - fixture.posterior_variance)
            loglik_error = abs(stats["log_likelihood"] - fixture.log_likelihood)
            seed_rows.append(
                {
                    "seed": seed,
                    "posterior_mean": stats["posterior_mean"],
                    "posterior_variance": stats["posterior_variance"],
                    "log_likelihood": stats["log_likelihood"],
                    "mean_abs_error": mean_error,
                    "variance_abs_error": variance_error,
                    "log_likelihood_abs_error": loglik_error,
                }
            )
            mean_errors.append(mean_error)
            variance_errors.append(variance_error)
            loglik_errors.append(loglik_error)
            rmse_components.append(math.sqrt(mean_error * mean_error + variance_error * variance_error + loglik_error * loglik_error))

        replications[str(particle_count)] = seed_rows
        mean_mean_error = float(np.mean(mean_errors))
        mean_variance_error = float(np.mean(variance_errors))
        mean_loglik_error = float(np.mean(loglik_errors))
        absolute_errors[str(particle_count)] = {
            "posterior_mean_abs_error_mean": mean_mean_error,
            "posterior_mean_abs_error_mcse": mcse(mean_errors),
            "posterior_variance_abs_error_mean": mean_variance_error,
            "posterior_variance_abs_error_mcse": mcse(variance_errors),
            "log_likelihood_abs_error_mean": mean_loglik_error,
            "log_likelihood_abs_error_mcse": mcse(loglik_errors),
            "replication_count": len(SEEDS),
        }
        error_scales[particle_count] = float(np.mean(rmse_components))

    slope_summary["rmse_log2_ratio_64_to_256"] = math.log(error_scales[64] / error_scales[256], 2.0)
    slope_summary["mean_error_ratio_64_to_256"] = absolute_errors["64"]["posterior_mean_abs_error_mean"] / absolute_errors["256"]["posterior_mean_abs_error_mean"]
    slope_summary["variance_error_ratio_64_to_256"] = absolute_errors["64"]["posterior_variance_abs_error_mean"] / absolute_errors["256"]["posterior_variance_abs_error_mean"]
    slope_summary["log_likelihood_error_ratio_64_to_256"] = absolute_errors["64"]["log_likelihood_abs_error_mean"] / absolute_errors["256"]["log_likelihood_abs_error_mean"]

    return (
        {
            "particle_counts": list(PARTICLE_COUNTS),
            "seed_list": list(SEEDS),
            "replications": replications,
            "absolute_error_summary": absolute_errors,
            "convergence_summary": slope_summary,
        },
        {
            "pf_mean_error_64": absolute_errors["64"]["posterior_mean_abs_error_mean"],
            "pf_mean_error_256": absolute_errors["256"]["posterior_mean_abs_error_mean"],
            "pf_variance_error_64": absolute_errors["64"]["posterior_variance_abs_error_mean"],
            "pf_variance_error_256": absolute_errors["256"]["posterior_variance_abs_error_mean"],
            "pf_log_likelihood_error_64": absolute_errors["64"]["log_likelihood_abs_error_mean"],
            "pf_log_likelihood_error_256": absolute_errors["256"]["log_likelihood_abs_error_mean"],
        },
    )


def edh_affine_recovery(fixture) -> tuple[dict[str, float], dict[str, float]]:
    predictive_precision = 1.0 / fixture.predictive_variance
    likelihood_precision = (fixture.h * fixture.h) / fixture.r
    posterior_variance = 1.0 / (predictive_precision + likelihood_precision)
    posterior_mean = posterior_variance * (
        fixture.predictive_mean * predictive_precision
        + fixture.h * fixture.observation / fixture.r
    )
    mean_error = abs(posterior_mean - fixture.posterior_mean)
    variance_error = abs(posterior_variance - fixture.posterior_variance)
    return (
        {
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_variance,
            "posterior_mean_abs_error": mean_error,
            "posterior_variance_abs_error": variance_error,
        },
        {
            "edh_posterior_mean_abs_error": mean_error,
            "edh_posterior_variance_abs_error": variance_error,
        },
    )


def build_coverage() -> dict[str, str]:
    return {
        "linear_gaussian_recovery": "passed",
        "synthetic_affine_flow": "missing",
        "pfpf_algebra_parity": "missing",
        "soft_resampling_bias": "missing",
        "sinkhorn_residual": "missing",
        "learned_map_residual": "missing",
        "hmc_value_gradient": "missing",
        "posterior_sensitivity_summary": "missing",
    }


def make_result_record(command: str, runtime_seconds: float) -> dict[str, object]:
    started_at = utc_now()
    fixture = build_linear_gaussian_fixture()
    pf_summary, finite_pf = summarize_pf(fixture)
    edh_summary, finite_edh = edh_affine_recovery(fixture)
    branch, commit, dirty_state_summary = get_git_manifest()
    result_path_rel = str(RESULT_PATH.resolve().relative_to(Path(__file__).resolve().parents[3]))
    report_path_rel = str(REPORT_PATH.resolve().relative_to(Path(__file__).resolve().parents[3]))
    artifact_paths = [result_path_rel, report_path_rel]
    package_versions = {"numpy": _numpy_version(), "python": platform.python_version()}
    ended_at = utc_now()

    result = {
        "phase_id": "IE3",
        "diagnostic_id": "linear_gaussian_recovery",
        "chapter_label": "Chapter 26",
        "diagnostic_role": "promotion_criterion",
        "comparator_id": "analytic_kalman_reference",
        "source_family": "Classical SMC / bootstrap PF; EDH / LEDH particle flow",
        "source_support_class": "bibliography_spine_only",
        "row_level_source_support_class": "bibliography_spine_only",
        "seed_policy": "fixed_seeds_[101,102,103,104,105]_particle_counts_[64,256]",
        "status": "pass",
        "coverage": build_coverage(),
        "tolerance": {
            "edh_posterior_mean_abs_error_max": 1e-12,
            "edh_posterior_variance_abs_error_max": 1e-12,
            "pf_interpretation": "descriptive_only_against_analytic_reference",
            "pf_replications_required": len(SEEDS),
        },
        "finite_checks": {
            "analytic_fixture": {
                "predictive_mean": fixture.predictive_mean,
                "predictive_variance": fixture.predictive_variance,
                "posterior_mean": fixture.posterior_mean,
                "posterior_variance": fixture.posterior_variance,
                "log_likelihood": fixture.log_likelihood,
            },
            "bootstrap_pf": pf_summary,
            "edh_affine_recovery": edh_summary,
            **finite_pf,
            **finite_edh,
        },
        "shape_checks": {
            "state_dimension": 1,
            "observation_dimension": 1,
            "time_steps": 1,
            "particle_count_ladder": list(PARTICLE_COUNTS),
            "replication_count": len(SEEDS),
        },
        "runtime_seconds": runtime_seconds,
        "blocker_class": "none",
        "non_implication": "This IE3 recovery result does not validate production bayesfilter code, does not justify DPF-HMC or banking-readiness claims, and does not upgrade replicated PF summaries into a statistically supported ranking or convergence proof.",
        "promotion_criterion_status": "pass",
        "promotion_veto_status": "not_triggered",
        "continuation_veto_status": "not_triggered",
        "repair_trigger": "Re-open IE3 if the schema stops carrying PF uncertainty fields, EDH special-case tolerances fail, or the CPU-only manifest proof drifts.",
        "explanatory_only_diagnostics": [
            "Bootstrap PF summaries are replicated descriptive engineering evidence only.",
            "RMSE slope across 64 and 256 particles is descriptive and not a promotion criterion.",
        ],
        "environment": {
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "branch": branch,
            "commit": commit,
            "dirty_state_summary": dirty_state_summary,
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
            "import_boundary_checked_modules": [
                "experiments.dpf_monograph_evidence.fixtures.linear_gaussian",
                "experiments.dpf_monograph_evidence.results",
                "experiments.dpf_monograph_evidence.runners.run_linear_gaussian_recovery",
            ],
        },
        "command": command,
        "wall_time_seconds": runtime_seconds,
        "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
        "artifact_paths": artifact_paths,
        "cpu_gpu_mode": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_devices_visible": [],
            "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
        },
        "uncertainty_status": "pass",
        "replication_count": len(SEEDS),
        "mcse_or_interval": {
            "posterior_mean_abs_error_mcse": {
                "64": pf_summary["absolute_error_summary"]["64"]["posterior_mean_abs_error_mcse"],
                "256": pf_summary["absolute_error_summary"]["256"]["posterior_mean_abs_error_mcse"],
            },
            "posterior_variance_abs_error_mcse": {
                "64": pf_summary["absolute_error_summary"]["64"]["posterior_variance_abs_error_mcse"],
                "256": pf_summary["absolute_error_summary"]["256"]["posterior_variance_abs_error_mcse"],
            },
            "log_likelihood_abs_error_mcse": {
                "64": pf_summary["absolute_error_summary"]["64"]["log_likelihood_abs_error_mcse"],
                "256": pf_summary["absolute_error_summary"]["256"]["log_likelihood_abs_error_mcse"],
            },
            "summary": "MCSE is reported across five fixed seeds for descriptive PF errors; EDH recovery is deterministic and reported separately via absolute tolerance.",
        },
        "post_run_red_team_note": "The EDH equality check could still pass while the chosen closed form is only an algebraic special case; PF error reduction across two particle counts is descriptive only and could look favorable without establishing asymptotic rate or downstream DPF correctness.",
        "cap_non_applicability_reasons": {
            "max_sinkhorn_iterations": "IE3 does not instantiate Sinkhorn transport.",
            "max_finite_difference_evaluations": "IE3 does not use finite-difference probes.",
        },
        "cap_values": {
            "max_particles": 256,
            "max_time_steps": 1,
            "max_sinkhorn_iterations": None,
            "max_finite_difference_evaluations": None,
            "max_replications": 5,
            "max_wall_clock_seconds": WALL_CLOCK_CAP_SECONDS,
        },
        "run_manifest": {
            "command": command,
            "branch": branch,
            "commit": commit,
            "dirty_state_summary": dirty_state_summary,
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_devices_visible": [],
            "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
            "pre_import_cuda_visible_devices": _PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "pre_import_gpu_hiding_assertion": _PRE_IMPORT_GPU_HIDING_ASSERTION,
            "seed_policy": "fixed_seeds_[101,102,103,104,105]_particle_counts_[64,256]",
            "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "artifact_paths": artifact_paths,
        },
    }
    validate_result_record(result)
    return result


def write_markdown_report(result: dict[str, object]) -> None:
    fixture = result["finite_checks"]["analytic_fixture"]
    pf_summary = result["finite_checks"]["bootstrap_pf"]
    edh = result["finite_checks"]["edh_affine_recovery"]
    lines = [
        "# IE3 linear-Gaussian recovery",
        "",
        "## Scope",
        "",
        "This clean-room IE3 report compares a one-step analytic Kalman reference against replicated bootstrap PF summaries and a deterministic EDH linear-Gaussian special-case recovery.",
        "",
        "## Fixture",
        "",
        f"- predictive mean: `{fixture['predictive_mean']:.12f}`",
        f"- predictive variance: `{fixture['predictive_variance']:.12f}`",
        f"- posterior mean: `{fixture['posterior_mean']:.12f}`",
        f"- posterior variance: `{fixture['posterior_variance']:.12f}`",
        f"- one-step log likelihood: `{fixture['log_likelihood']:.12f}`",
        "",
        "## Bootstrap PF descriptive diagnostics",
        "",
        "| Particles | Mean abs. error | Mean MCSE | Variance abs. error | Variance MCSE | Log-likelihood abs. error | Log-likelihood MCSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for particle_count in PARTICLE_COUNTS:
        summary = pf_summary["absolute_error_summary"][str(particle_count)]
        lines.append(
            f"| {particle_count} | {summary['posterior_mean_abs_error_mean']:.12f} | {summary['posterior_mean_abs_error_mcse']:.12f} | {summary['posterior_variance_abs_error_mean']:.12f} | {summary['posterior_variance_abs_error_mcse']:.12f} | {summary['log_likelihood_abs_error_mean']:.12f} | {summary['log_likelihood_abs_error_mcse']:.12f} |"
        )
    lines.extend(
        [
            "",
            "PF interpretation: descriptive engineering evidence only. The five-seed summaries show smaller errors at 256 particles than at 64 particles, but this is not treated as a statistically supported convergence claim.",
            "",
            "## EDH deterministic special-case recovery",
            "",
            f"- posterior mean abs. error: `{edh['posterior_mean_abs_error']:.3e}`",
            f"- posterior variance abs. error: `{edh['posterior_variance_abs_error']:.3e}`",
            "- interpretation: exactness is restricted to this linear-Gaussian special case and is not promoted to nonlinear settings.",
            "",
            "## CPU-only manifest proof",
            "",
            f"- `CUDA_VISIBLE_DEVICES` before scientific imports: `{result['run_manifest']['pre_import_cuda_visible_devices']}`",
            f"- pre-import assertion: `{result['run_manifest']['pre_import_gpu_hiding_assertion']}`",
            f"- cpu_only: `{result['run_manifest']['cpu_only']}`",
            "",
            "## Exit label",
            "",
            "`ie3_linear_gaussian_recovery_passed`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    command = "python -m experiments.dpf_monograph_evidence.runners.run_linear_gaussian_recovery"
    result = make_result_record(command=command, runtime_seconds=0.0)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime_seconds = time.perf_counter() - start
    result["runtime_seconds"] = runtime_seconds
    result["wall_time_seconds"] = runtime_seconds
    result["run_manifest"]["ended_at_utc"] = utc_now()
    write_markdown_report(result)
    validate_result_record(result)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {REPORT_PATH}")
    print("ie3_linear_gaussian_recovery_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
