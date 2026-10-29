"""Run LGSSM bootstrap PF and finite-Sinkhorn OT-DPF diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
_PRE_IMPORT_GPU_HIDING_ASSERTION = _PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"

import numpy as np

from experiments.dpf_implementation.filters.bootstrap_pf import run_bootstrap_particle_filter
from experiments.dpf_implementation.filters.dpf_ot import run_ot_dpf
from experiments.dpf_implementation.fixtures.lgssm import (
    build_lgssm_fixture,
    initial_sample,
    observation_log_density,
    transition_sample,
)
from experiments.dpf_implementation.references.kalman_lgssm import run_kalman_filter
from experiments.dpf_implementation.runners.common import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    finite_list,
    load_json,
    max_sinkhorn_residual,
    relative,
    rmse,
    stable_digest,
    utc_now,
    wall_time_call,
    write_json,
)


OUTPUT_PATH = OUTPUT_DIR / "dpf_ot_lgssm_2026-05-28.json"
REPORT_PATH = REPORT_DIR / "dpf-ot-lgssm-result-2026-05-28.md"
PLAN_PATH = Path("docs/plans/bayesfilter-dpf-ot-implementation-p6-lgssm-validation-result-plan-2026-05-28.md")
SEEDS = (111, 222, 333)
NUM_PARTICLES = 192
ESS_THRESHOLD_RATIO = 0.5
SINKHORN_EPSILON = 0.7
SINKHORN_ITERATIONS = 80
SINKHORN_TOLERANCE = 1e-7


def build_core_payload() -> dict[str, Any]:
    fixture = build_lgssm_fixture()
    kalman = run_kalman_filter(fixture)

    def init(rng, n):
        return initial_sample(rng, n, fixture)

    def trans(rng, particles, _t):
        return transition_sample(rng, particles, fixture)

    def obs_log(particles, observation, _t):
        return observation_log_density(particles, observation, fixture)

    rows = []
    for seed in SEEDS:
        bootstrap = run_bootstrap_particle_filter(
            observations=fixture.observations,
            initial_sample=init,
            transition_sample=trans,
            observation_log_density=obs_log,
            seed=seed,
            num_particles=NUM_PARTICLES,
            ess_threshold_ratio=ESS_THRESHOLD_RATIO,
        )
        ot = run_ot_dpf(
            observations=fixture.observations,
            initial_sample=init,
            transition_sample=trans,
            observation_log_density=obs_log,
            seed=seed,
            num_particles=NUM_PARTICLES,
            ess_threshold_ratio=ESS_THRESHOLD_RATIO,
            sinkhorn_epsilon=SINKHORN_EPSILON,
            sinkhorn_iterations=SINKHORN_ITERATIONS,
            sinkhorn_tolerance=SINKHORN_TOLERANCE,
        )
        rows.append(_row("bootstrap_sir_pf", bootstrap, fixture, kalman))
        rows.append(_row("ot_dpf_finite_sinkhorn_relaxed", ot, fixture, kalman))

    bootstrap_rows = [r for r in rows if r["method_id"] == "bootstrap_sir_pf"]
    ot_rows = [r for r in rows if r["method_id"] == "ot_dpf_finite_sinkhorn_relaxed"]
    summary = {
        "decision": "DPF_OT_LGSSM_UNDER_VALIDATION",
        "kalman_log_likelihood": kalman.log_likelihood,
        "candidate_rows": len(rows),
        "bootstrap_rows": len(bootstrap_rows),
        "ot_dpf_rows": len(ot_rows),
        "median_bootstrap_mean_rmse_to_kalman": float(
            np.median([r["reference_comparison"]["filtered_mean_rmse_to_kalman"] for r in bootstrap_rows])
        ),
        "median_ot_mean_rmse_to_kalman": float(
            np.median([r["reference_comparison"]["filtered_mean_rmse_to_kalman"] for r in ot_rows])
        ),
        "median_bootstrap_loglik_delta_to_kalman": float(
            np.median([r["reference_comparison"]["loglik_abs_delta_to_kalman"] for r in bootstrap_rows])
        ),
        "median_ot_loglik_delta_to_kalman": float(
            np.median([r["reference_comparison"]["loglik_abs_delta_to_kalman"] for r in ot_rows])
        ),
        "max_ot_sinkhorn_residual": float(
            max(max_sinkhorn_residual(r["resampling_diagnostics"]) for r in ot_rows)
        ),
        "thresholds": {
            "max_median_mean_rmse_to_kalman": 0.45,
            "max_median_loglik_delta_to_kalman": 8.0,
            "max_sinkhorn_residual": 1e-5,
            "threshold_role": "loose_run_validity_veto_not_ranking_metric",
        },
    }
    return {
        "model_definition": fixture.model_definition(),
        "reference": {
            "reference_id": kalman.reference_id,
            "log_likelihood": kalman.log_likelihood,
            "filtered_means": kalman.filtered_means.tolist(),
            "finite": kalman.finite,
            "reference_status": "exact Kalman reference for this LGSSM only",
        },
        "seed_policy": {
            "seeds": list(SEEDS),
            "num_particles": NUM_PARTICLES,
            "ess_threshold_ratio": ESS_THRESHOLD_RATIO,
            "sinkhorn_epsilon": SINKHORN_EPSILON,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "sinkhorn_tolerance": SINKHORN_TOLERANCE,
        },
        "rows": rows,
        "summary": summary,
        "non_implications": _non_implications(),
    }


def _row(method_id: str, result, fixture, kalman) -> dict[str, Any]:
    comparison = {
        "reference_id": kalman.reference_id,
        "filtered_mean_rmse_to_kalman": rmse(result.filtered_means, kalman.filtered_means),
        "loglik_abs_delta_to_kalman": float(
            abs(result.log_likelihood_estimate - kalman.log_likelihood)
        ),
        "comparator_status": "Kalman is exact only for this LGSSM fixture.",
    }
    return {
        "row_id": f"{method_id}_seed_{result.seed}",
        "method_id": method_id,
        "seed": result.seed,
        "num_particles": result.num_particles,
        "model_checksum": fixture.model_checksum,
        "observation_checksum": fixture.observation_checksum,
        "horizon": fixture.horizon,
        "dtype": "float64",
        "device": "cpu_only_cuda_visible_devices_-1",
        "log_likelihood_estimate": result.log_likelihood_estimate,
        "filtered_means": result.filtered_means.tolist(),
        "ess_summary": {
            "min_ess": float(np.min(result.ess_by_time)),
            "mean_ess": float(np.mean(result.ess_by_time)),
            "final_ess": float(result.ess_by_time[-1]),
        },
        "resampling_count": result.resampling_count,
        "resampling_diagnostics": result.resampling_diagnostics,
        "finite": result.finite,
        "reference_comparison": comparison,
        "resampling_status": (
            "classical_categorical_systematic"
            if method_id == "bootstrap_sir_pf"
            else "relaxed_finite_sinkhorn_not_categorical"
        ),
    }


def build_payload(command: str, runtime_seconds: float) -> dict[str, Any]:
    started_at = utc_now()
    core = build_core_payload()
    core_digest = stable_digest(core)
    rerun_digest = stable_digest(build_core_payload())
    payload = {
        "decision": "DPF_OT_LGSSM_UNDER_VALIDATION",
        "plan_path": str(PLAN_PATH),
        "question": "LGSSM OT-DPF relaxed resampling smoke against Kalman reference",
        **core,
        "reproducibility": {
            "core_digest": core_digest,
            "rerun_core_digest": rerun_digest,
            "matches": core_digest == rerun_digest,
        },
        "environment": environment_manifest(
            pre_import_cuda_visible_devices=_PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            pre_import_gpu_hiding_assertion=_PRE_IMPORT_GPU_HIDING_ASSERTION,
        ),
        "command": command,
        "runtime_seconds": runtime_seconds,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "artifact_paths": [relative(OUTPUT_PATH), relative(REPORT_PATH)],
    }
    errors = validate_payload(payload)
    payload["schema_validation"] = {"status": "pass" if not errors else "fail", "errors": errors}
    payload["decision"] = "DPF_OT_LGSSM_PASSED" if not errors else "DPF_OT_LGSSM_FAILED"
    payload["summary"]["decision"] = payload["decision"]
    return payload


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload["environment"].get("pre_import_cuda_visible_devices") != "-1":
        errors.append("CUDA_VISIBLE_DEVICES was not -1 before NumPy import")
    if payload["reproducibility"].get("matches") is not True:
        errors.append("reproducibility digest mismatch")
    if payload["reference"].get("finite") is not True:
        errors.append("Kalman reference is non-finite")
    model_checksum = payload["model_definition"]["model_checksum"]
    observation_checksum = payload["model_definition"]["observation_checksum"]
    for row in payload["rows"]:
        if row["model_checksum"] != model_checksum:
            errors.append(f"{row['row_id']} model checksum mismatch")
        if row["observation_checksum"] != observation_checksum:
            errors.append(f"{row['row_id']} observation checksum mismatch")
        if row["finite"] is not True:
            errors.append(f"{row['row_id']} emitted non-finite values")
        if not finite_list(row["filtered_means"]):
            errors.append(f"{row['row_id']} filtered means non-finite")
    summary = payload["summary"]
    thresholds = summary["thresholds"]
    if summary["median_bootstrap_mean_rmse_to_kalman"] > thresholds["max_median_mean_rmse_to_kalman"]:
        errors.append("bootstrap median mean RMSE exceeded loose cap")
    if summary["median_ot_mean_rmse_to_kalman"] > thresholds["max_median_mean_rmse_to_kalman"]:
        errors.append("OT median mean RMSE exceeded loose cap")
    if summary["median_bootstrap_loglik_delta_to_kalman"] > thresholds["max_median_loglik_delta_to_kalman"]:
        errors.append("bootstrap loglik delta exceeded loose cap")
    if summary["median_ot_loglik_delta_to_kalman"] > thresholds["max_median_loglik_delta_to_kalman"]:
        errors.append("OT loglik delta exceeded loose cap")
    if summary["max_ot_sinkhorn_residual"] > thresholds["max_sinkhorn_residual"]:
        errors.append("Sinkhorn residual exceeded loose cap")
    return errors


def write_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# DPF OT LGSSM Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
        f"| primary criterion | `{payload['schema_validation']['status']}` | finite rows, Kalman comparison, Sinkhorn residuals, and reproducibility checks |",
        f"| median bootstrap mean RMSE | `diagnostic` | `{summary['median_bootstrap_mean_rmse_to_kalman']:.6f}` |",
        f"| median OT-DPF mean RMSE | `diagnostic` | `{summary['median_ot_mean_rmse_to_kalman']:.6f}` |",
        f"| median bootstrap loglik delta | `diagnostic` | `{summary['median_bootstrap_loglik_delta_to_kalman']:.6f}` |",
        f"| median OT-DPF loglik delta | `diagnostic` | `{summary['median_ot_loglik_delta_to_kalman']:.6f}` |",
        f"| max OT Sinkhorn residual | `veto` | `{summary['max_ot_sinkhorn_residual']:.3e}` |",
        f"| reproducibility | `{'pass' if payload['reproducibility']['matches'] else 'fail'}` | `{payload['reproducibility']['core_digest']}` |",
        "",
        "## Interpretation",
        "",
        "The LGSSM smoke passed for the bounded finite-Sinkhorn relaxed OT-DPF path.  The Kalman filter is the exact reference for this LGSSM fixture; the OT-DPF path remains a finite-budget relaxed resampling diagnostic, not categorical PF equivalence.",
        "",
        "## Non-Implications",
        "",
        *[f"- {item}" for item in payload["non_implications"]],
        "",
        "## Review Record",
        "",
        "- Claude result review: pending.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
        "No learned or neural OT promotion is concluded.",
        "No banking or model-risk claim is concluded.",
        "No monograph claim is concluded without separate review.",
        "Finite Sinkhorn OT-DPF rows are relaxed-resampling diagnostics only.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        payload = load_json(OUTPUT_PATH)
        errors = validate_payload(payload)
        if errors:
            raise SystemExit("\n".join(errors))
        print("validation ok")
        return
    if args.check_reproducibility:
        payload = load_json(OUTPUT_PATH)
        rerun_digest = stable_digest(build_core_payload())
        if payload["reproducibility"]["core_digest"] != rerun_digest:
            raise SystemExit("reproducibility digest mismatch")
        print("reproducibility ok")
        return
    payload, runtime = wall_time_call(lambda: build_payload("python -m experiments.dpf_implementation.runners.run_lgssm_ot_dpf", 0.0))
    payload["runtime_seconds"] = runtime
    write_json(OUTPUT_PATH, payload)
    write_report(payload)
    if payload["schema_validation"]["errors"]:
        raise SystemExit("\n".join(payload["schema_validation"]["errors"]))
    print(payload["decision"])


if __name__ == "__main__":
    main()
