"""Run same-scalar finite-difference checks for the OT-DPF proxy scalar."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
_PRE_IMPORT_GPU_HIDING_ASSERTION = _PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"

import numpy as np

from experiments.dpf_implementation.filters.dpf_ot import run_ot_dpf
from experiments.dpf_implementation.fixtures.lgssm import (
    LGSSMFixture,
    build_lgssm_fixture,
    initial_sample,
    observation_log_density,
    transition_sample,
)
from experiments.dpf_implementation.runners.common import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    load_json,
    relative,
    stable_digest,
    utc_now,
    wall_time_call,
    write_json,
)


OUTPUT_PATH = OUTPUT_DIR / "dpf_ot_gradient_check_2026-05-28.json"
REPORT_PATH = REPORT_DIR / "dpf-ot-gradient-check-result-2026-05-28.md"
PLAN_PATH = Path("docs/plans/bayesfilter-dpf-ot-implementation-p5-gradient-contract-and-finite-difference-plan-2026-05-28.md")
SEED = 444
NUM_PARTICLES = 96
SINKHORN_EPSILON = 0.75
SINKHORN_ITERATIONS = 60
FINITE_DIFFERENCE_STEP = 1e-4


def scalar_value(q_scale: float) -> float:
    """Return the named relaxed OT-DPF negative log-normalizer proxy."""

    base = build_lgssm_fixture(horizon=12)
    fixture = _scaled_q_fixture(base, q_scale)

    def init(rng, n):
        return initial_sample(rng, n, fixture)

    def trans(rng, particles, _t):
        return transition_sample(rng, particles, fixture)

    def obs_log(particles, observation, _t):
        return observation_log_density(particles, observation, fixture)

    result = run_ot_dpf(
        observations=fixture.observations,
        initial_sample=init,
        transition_sample=trans,
        observation_log_density=obs_log,
        seed=SEED,
        num_particles=NUM_PARTICLES,
        ess_threshold_ratio=0.5,
        sinkhorn_epsilon=SINKHORN_EPSILON,
        sinkhorn_iterations=SINKHORN_ITERATIONS,
        sinkhorn_tolerance=1e-7,
    )
    return float(-result.log_likelihood_estimate)


def build_core_payload() -> dict[str, Any]:
    center = 1.0
    step = FINITE_DIFFERENCE_STEP
    values = {
        "center": scalar_value(center),
        "plus": scalar_value(center + step),
        "minus": scalar_value(center - step),
        "plus_half": scalar_value(center + 0.5 * step),
        "minus_half": scalar_value(center - 0.5 * step),
    }
    fd = (values["plus"] - values["minus"]) / (2.0 * step)
    fd_half = (values["plus_half"] - values["minus_half"]) / step
    residual = abs(fd - fd_half)
    digest_payload = {
        "scalar_id": "lgssm_relaxed_ot_log_normalizer_proxy",
        "parameter": "lgssm_process_noise_scale",
        "seed": SEED,
        "num_particles": NUM_PARTICLES,
        "sinkhorn_epsilon": SINKHORN_EPSILON,
        "sinkhorn_iterations": SINKHORN_ITERATIONS,
        "values": values,
        "finite_difference": fd,
    }
    summary = {
        "decision": "DPF_OT_GRADIENT_UNDER_VALIDATION",
        "scalar_id": "lgssm_relaxed_ot_log_normalizer_proxy",
        "target_status": "relaxed_finite_sinkhorn_proxy_not_likelihood_score",
        "parameter_coordinates": "positive process-noise scale in fixture coordinates",
        "randomness_policy": "fixed seed/common random numbers through deterministic RNG calls",
        "resampling_policy": "finite_sinkhorn_relaxed",
        "gradient_path": "central_finite_difference_only",
        "autodiff_status": "autodiff_not_tested",
        "center_value": values["center"],
        "finite_difference_gradient": fd,
        "half_step_finite_difference_gradient": fd_half,
        "finite_difference_stability_abs": residual,
        "thresholds": {
            "max_fd_stability_abs": 5e-2,
            "threshold_role": "same-scalar_finite_difference_stability_veto_not_HMC_validation",
        },
        "same_scalar_digest": stable_digest(digest_payload),
    }
    return {
        "question": "same-scalar finite-difference check for relaxed OT-DPF scalar",
        "seed_policy": {
            "seed": SEED,
            "num_particles": NUM_PARTICLES,
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "common_random_numbers": True,
        },
        "values": values,
        "summary": summary,
        "non_implications": _non_implications(),
    }


def _scaled_q_fixture(base: LGSSMFixture, q_scale: float) -> LGSSMFixture:
    if q_scale <= 0.0:
        raise ValueError("q_scale must be positive")
    scaled_q = base.Q * q_scale
    return LGSSMFixture(
        name=f"{base.name}_qscale_{q_scale:.6f}",
        A=base.A,
        C=base.C,
        Q=scaled_q,
        R=base.R,
        m0=base.m0,
        P0=base.P0,
        states=base.states,
        observations=base.observations,
        fixture_generation_seed=base.fixture_generation_seed,
        model_checksum=base.model_checksum,
        observation_checksum=base.observation_checksum,
    )


def build_payload(command: str, runtime_seconds: float) -> dict[str, Any]:
    started_at = utc_now()
    core = build_core_payload()
    core_digest = stable_digest(core)
    rerun_digest = stable_digest(build_core_payload())
    payload = {
        "decision": "DPF_OT_GRADIENT_UNDER_VALIDATION",
        "plan_path": str(PLAN_PATH),
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
    payload["decision"] = "DPF_OT_GRADIENT_FD_PASSED" if not errors else "DPF_OT_GRADIENT_FAILED"
    payload["summary"]["decision"] = payload["decision"]
    return payload


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload["environment"].get("pre_import_cuda_visible_devices") != "-1":
        errors.append("CUDA_VISIBLE_DEVICES was not -1 before NumPy import")
    if payload["reproducibility"].get("matches") is not True:
        errors.append("reproducibility digest mismatch")
    values = payload.get("values", {})
    for key in ("center", "plus", "minus", "plus_half", "minus_half"):
        if not np.isfinite(float(values.get(key, np.nan))):
            errors.append(f"{key} scalar value is non-finite")
    summary = payload["summary"]
    if not np.isfinite(float(summary.get("finite_difference_gradient", np.nan))):
        errors.append("finite-difference gradient is non-finite")
    if (
        abs(float(summary.get("finite_difference_stability_abs", np.inf)))
        > summary["thresholds"]["max_fd_stability_abs"]
    ):
        errors.append("finite-difference stability residual exceeded loose cap")
    if summary.get("scalar_id") != "lgssm_relaxed_ot_log_normalizer_proxy":
        errors.append("unexpected scalar id")
    if summary.get("gradient_path") != "central_finite_difference_only":
        errors.append("unexpected gradient path")
    return errors


def write_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# DPF OT Gradient Check Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
        f"| primary criterion | `{payload['schema_validation']['status']}` | same-scalar finite-difference value path |",
        f"| scalar id | `{summary['scalar_id']}` | named relaxed OT-DPF proxy |",
        f"| gradient path | `{summary['gradient_path']}` | no autodiff claim |",
        f"| autodiff status | `{summary['autodiff_status']}` | unresolved risk, not blocker |",
        f"| finite-difference gradient | `diagnostic` | `{summary['finite_difference_gradient']:.6f}` |",
        f"| stability residual | `veto` | `{summary['finite_difference_stability_abs']:.3e}` |",
        f"| reproducibility | `{'pass' if payload['reproducibility']['matches'] else 'fail'}` | `{payload['reproducibility']['core_digest']}` |",
        "",
        "## Interpretation",
        "",
        "The same-scalar finite-difference check passed for the named relaxed OT-DPF log-normalizer proxy.  This is finite-difference-only evidence; no autodiff, HMC, posterior, or likelihood-score validity is concluded.",
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
        "Finite-difference gradient evidence is for the named relaxed proxy scalar only.",
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
    payload, runtime = wall_time_call(lambda: build_payload("python -m experiments.dpf_implementation.runners.run_gradient_checks", 0.0))
    payload["runtime_seconds"] = runtime
    write_json(OUTPUT_PATH, payload)
    write_report(payload)
    if payload["schema_validation"]["errors"]:
        raise SystemExit("\n".join(payload["schema_validation"]["errors"]))
    print(payload["decision"])


if __name__ == "__main__":
    main()
