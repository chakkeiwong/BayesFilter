"""Run range-bearing bootstrap PF and finite-Sinkhorn OT-DPF diagnostics."""

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
from experiments.dpf_implementation.fixtures.range_bearing import (
    make_fixture,
    initial_sample,
    observation_log_density,
    range_bearing_observation,
    transition_sample,
)
from experiments.dpf_implementation.references.ukf import run_range_bearing_ukf
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


OUTPUT_PATH = OUTPUT_DIR / "dpf_ot_range_bearing_2026-05-28.json"
REPORT_PATH = REPORT_DIR / "dpf-ot-range-bearing-result-2026-05-28.md"
PLAN_PATH = Path("docs/plans/bayesfilter-dpf-ot-implementation-p7-range-bearing-validation-result-plan-2026-05-28.md")
SEEDS = (31, 43, 59)
NUM_PARTICLES = 192
ESS_THRESHOLD_RATIO = 0.5
SINKHORN_EPSILON = 0.35
SINKHORN_ITERATIONS = 80
SINKHORN_TOLERANCE = 1e-7


def build_core_payload() -> dict[str, Any]:
    fixture = make_fixture("range_bearing_gaussian_moderate")
    ukf = run_range_bearing_ukf(fixture)

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
        rows.append(_row("bootstrap_sir_pf", bootstrap, fixture, ukf))
        rows.append(_row("ot_dpf_finite_sinkhorn_relaxed", ot, fixture, ukf))

    bootstrap_rows = [r for r in rows if r["method_id"] == "bootstrap_sir_pf"]
    ot_rows = [r for r in rows if r["method_id"] == "ot_dpf_finite_sinkhorn_relaxed"]
    summary = {
        "decision": "DPF_OT_RANGE_BEARING_UNDER_VALIDATION",
        "candidate_rows": len(rows),
        "bootstrap_rows": len(bootstrap_rows),
        "ot_dpf_rows": len(ot_rows),
        "median_bootstrap_state_rmse_to_ukf": float(
            np.median([r["reference_comparison"]["state_rmse_to_ukf"] for r in bootstrap_rows])
        ),
        "median_ot_state_rmse_to_ukf": float(
            np.median([r["reference_comparison"]["state_rmse_to_ukf"] for r in ot_rows])
        ),
        "median_bootstrap_latent_position_rmse": float(
            np.median([r["proxy_metrics"]["latent_position_rmse"] for r in bootstrap_rows])
        ),
        "median_ot_latent_position_rmse": float(
            np.median([r["proxy_metrics"]["latent_position_rmse"] for r in ot_rows])
        ),
        "median_bootstrap_observation_proxy_rmse": float(
            np.median([r["proxy_metrics"]["observation_proxy_rmse"] for r in bootstrap_rows])
        ),
        "median_ot_observation_proxy_rmse": float(
            np.median([r["proxy_metrics"]["observation_proxy_rmse"] for r in ot_rows])
        ),
        "max_ot_sinkhorn_residual": float(
            max(max_sinkhorn_residual(r["resampling_diagnostics"]) for r in ot_rows)
        ),
        "thresholds": {
            "max_median_state_rmse_to_ukf": 0.45,
            "max_median_latent_position_rmse": 0.45,
            "max_median_observation_proxy_rmse": 0.35,
            "max_sinkhorn_residual": 1e-5,
            "threshold_role": "loose_proxy_run_validity_veto_not_correctness_or_ranking",
        },
        "controlled_context": {
            "fixed_grid_moderate_position_rmse_range": [0.0578863, 0.0677156],
            "status": "comparison_only_proxy_context_not_acceptance_evidence",
        },
    }
    return {
        "model_definition": fixture.model_definition(),
        "reference": {
            "reference_id": ukf.reference_id,
            "filtered_means": ukf.filtered_means.tolist(),
            "finite": ukf.finite,
            "approximate_reference": ukf.approximate_reference,
            "reference_status": "UKF is approximate, not ground truth.",
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


def _row(method_id: str, result, fixture, ukf) -> dict[str, Any]:
    filtered_means = result.filtered_means
    ukf_means = ukf.filtered_means
    latent = fixture.states[1:]
    predicted_obs = range_bearing_observation(filtered_means)
    comparison = {
        "reference_id": ukf.reference_id,
        "state_rmse_to_ukf": rmse(filtered_means, ukf_means),
        "comparator_status": "UKF is approximate for the nonlinear fixture, not truth.",
    }
    proxy_metrics = {
        "latent_state_rmse_explanatory": rmse(filtered_means, latent),
        "latent_position_rmse": rmse(filtered_means[:, :2], latent[:, :2]),
        "observation_proxy_rmse": rmse(predicted_obs, fixture.observations),
        "proxy_status": "proxy_only_not_correctness_or_posterior_evidence",
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
        "filtered_means": filtered_means.tolist(),
        "ess_summary": {
            "min_ess": float(np.min(result.ess_by_time)),
            "mean_ess": float(np.mean(result.ess_by_time)),
            "final_ess": float(result.ess_by_time[-1]),
        },
        "resampling_count": result.resampling_count,
        "resampling_diagnostics": result.resampling_diagnostics,
        "finite": result.finite,
        "reference_comparison": comparison,
        "proxy_metrics": proxy_metrics,
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
        "decision": "DPF_OT_RANGE_BEARING_UNDER_VALIDATION",
        "plan_path": str(PLAN_PATH),
        "question": "Range-bearing OT-DPF relaxed resampling smoke with UKF approximate reference",
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
    payload["decision"] = "DPF_OT_RANGE_BEARING_PASSED" if not errors else "DPF_OT_RANGE_BEARING_FAILED"
    payload["summary"]["decision"] = payload["decision"]
    return payload


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload["environment"].get("pre_import_cuda_visible_devices") != "-1":
        errors.append("CUDA_VISIBLE_DEVICES was not -1 before NumPy import")
    if payload["reproducibility"].get("matches") is not True:
        errors.append("reproducibility digest mismatch")
    if payload["reference"].get("finite") is not True:
        errors.append("UKF reference is non-finite")
    if payload["reference"].get("approximate_reference") is not True:
        errors.append("UKF approximate-reference caveat missing")
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
    if summary["median_bootstrap_state_rmse_to_ukf"] > thresholds["max_median_state_rmse_to_ukf"]:
        errors.append("bootstrap state RMSE to UKF exceeded loose cap")
    if summary["median_ot_state_rmse_to_ukf"] > thresholds["max_median_state_rmse_to_ukf"]:
        errors.append("OT state RMSE to UKF exceeded loose cap")
    if summary["median_bootstrap_latent_position_rmse"] > thresholds["max_median_latent_position_rmse"]:
        errors.append("bootstrap latent position RMSE exceeded loose cap")
    if summary["median_ot_latent_position_rmse"] > thresholds["max_median_latent_position_rmse"]:
        errors.append("OT latent position RMSE exceeded loose cap")
    if summary["median_bootstrap_observation_proxy_rmse"] > thresholds["max_median_observation_proxy_rmse"]:
        errors.append("bootstrap observation proxy RMSE exceeded loose cap")
    if summary["median_ot_observation_proxy_rmse"] > thresholds["max_median_observation_proxy_rmse"]:
        errors.append("OT observation proxy RMSE exceeded loose cap")
    if summary["max_ot_sinkhorn_residual"] > thresholds["max_sinkhorn_residual"]:
        errors.append("Sinkhorn residual exceeded loose cap")
    return errors


def write_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# DPF OT Range-Bearing Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
        f"| primary criterion | `{payload['schema_validation']['status']}` | finite UKF/PF/OT rows, Sinkhorn residuals, reproducibility, and loose proxy caps |",
        f"| median bootstrap state RMSE to UKF | `proxy` | `{summary['median_bootstrap_state_rmse_to_ukf']:.6f}` |",
        f"| median OT-DPF state RMSE to UKF | `proxy` | `{summary['median_ot_state_rmse_to_ukf']:.6f}` |",
        f"| median bootstrap latent position RMSE | `proxy` | `{summary['median_bootstrap_latent_position_rmse']:.6f}` |",
        f"| median OT-DPF latent position RMSE | `proxy` | `{summary['median_ot_latent_position_rmse']:.6f}` |",
        f"| median bootstrap observation proxy RMSE | `proxy` | `{summary['median_bootstrap_observation_proxy_rmse']:.6f}` |",
        f"| median OT-DPF observation proxy RMSE | `proxy` | `{summary['median_ot_observation_proxy_rmse']:.6f}` |",
        f"| max OT Sinkhorn residual | `veto` | `{summary['max_ot_sinkhorn_residual']:.3e}` |",
        f"| reproducibility | `{'pass' if payload['reproducibility']['matches'] else 'fail'}` | `{payload['reproducibility']['core_digest']}` |",
        "",
        "## Interpretation",
        "",
        "The bounded range-bearing smoke passed with a UKF approximate reference and a classical bootstrap PF comparator.  All RMSE values are proxy diagnostics only; UKF is not ground truth and finite-Sinkhorn OT-DPF is a relaxed resampling path.",
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
        "UKF is approximate and proxy RMSE is not correctness evidence.",
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
    payload, runtime = wall_time_call(lambda: build_payload("python -m experiments.dpf_implementation.runners.run_range_bearing_ot_dpf", 0.0))
    payload["runtime_seconds"] = runtime
    write_json(OUTPUT_PATH, payload)
    write_report(payload)
    if payload["schema_validation"]["errors"]:
        raise SystemExit("\n".join(payload["schema_validation"]["errors"]))
    print(payload["decision"])


if __name__ == "__main__":
    main()
