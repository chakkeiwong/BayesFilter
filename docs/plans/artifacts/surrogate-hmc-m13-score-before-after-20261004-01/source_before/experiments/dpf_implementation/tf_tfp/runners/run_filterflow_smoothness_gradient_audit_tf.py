"""Smoothness and gradient scalar-contract audit for filterflow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_final_gaps_closure_tf as final_gaps,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-gap-closure-program-plan-2026-05-31.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_smoothness_gradient_audit_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-smoothness-gradient-audit-2026-05-31.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    smooth = final_gaps._run_smoothness_subprocess()
    scalar_ledger = _scalar_ledger(smooth)
    decision = _decision(smooth, scalar_ledger)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Audit filterflow smoothness gradient scalar normalization and scale diagnostics.",
        "smoothness_gradient": smooth,
        "scalar_normalization_ledger": scalar_ledger,
        "gradient_claim_status": _gradient_claim_status(smooth, scalar_ledger),
        "bayesfilter_gradient_status": {
            "status": "not_run_structured_scope_limit",
            "reason": (
                "Current matched BayesFilter table runner is likelihood-table oriented. "
                "A same-scalar BayesFilter GradientTape surface requires a separate "
                "differentiable scalar harness after the filterflow scalar contract is reconciled."
            ),
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_smoothness_gradient_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _scalar_ledger(smooth: dict[str, Any]) -> dict[str, Any]:
    if smooth.get("status") != "executed":
        return {"status": "blocked", "reason": smooth.get("status")}
    settings = smooth["settings"]
    horizon = float(settings["T"])
    dpf_ll = [float(value) for value in smooth["dpf_log_likelihoods"]]
    kalman_ll = [float(value) for value in smooth["kalman_log_likelihoods"]]
    total_diff = [a - b for a, b in zip(dpf_ll, kalman_ll)]
    per_time_diff = [value / horizon for value in total_diff]
    return {
        "status": "severe_unreconciled_gradient_magnitude_mismatch_risk_recorded",
        "filterflow_surface_scalar": "total_log_likelihood_from_filterflow_simple_linear_smoothness",
        "kalman_reference_scalar": "total_log_likelihood_from_filterflow_get_surface_kf_finite_difference",
        "negative_scalar_available": True,
        "per_time_normalization_available": True,
        "horizon": horizon,
        "total_likelihood_rmse": smooth["likelihood_rmse"],
        "per_time_likelihood_rmse": _rmse(per_time_diff),
        "total_gradient_rmse": smooth["gradient_rmse"],
        "per_time_gradient_rmse_proxy": smooth["gradient_rmse"] / horizon,
        "gradient_norm_ratio_dpf_to_kalman_fd": (
            None
            if smooth["kalman_fd_gradient_norm"] == 0.0
            else smooth["dpf_gradient_norm"] / smooth["kalman_fd_gradient_norm"]
        ),
        "gradient_cosine_vs_kalman_fd": smooth["gradient_cosine_vs_kalman_fd"],
        "gradient_sign_agreement": smooth["gradient_sign_agreement"],
        "interpretation": (
            "Finite gradients are present, but total/per-time normalization does not by itself "
            "explain the severe gradient magnitude mismatch. Gradient agreement is not concluded."
        ),
    }


def _decision(smooth: dict[str, Any], scalar_ledger: dict[str, Any]) -> str:
    if smooth.get("status") != "executed":
        return "smoothness_gradient_blocked"
    if not smooth.get("finite_likelihoods") or not smooth.get("finite_gradients"):
        return "smoothness_gradient_nonfinite_veto"
    if scalar_ledger["status"] == "severe_unreconciled_gradient_magnitude_mismatch_risk_recorded":
        return "smoothness_gradient_severe_unreconciled_magnitude_risk_recorded"
    return "smoothness_gradient_review_required"


def _gradient_claim_status(smooth: dict[str, Any], scalar_ledger: dict[str, Any]) -> str:
    if smooth.get("status") != "executed":
        return "blocked"
    if not smooth.get("finite_gradients"):
        return "nonfinite_veto"
    if scalar_ledger["status"] == "severe_unreconciled_gradient_magnitude_mismatch_risk_recorded":
        return "finite_gradient_smoke_not_agreement"
    return "review_required"


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != "smoothness_gradient_severe_unreconciled_magnitude_risk_recorded":
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if payload["gradient_claim_status"] != "finite_gradient_smoke_not_agreement":
        raise RuntimeError("gradient overclaim risk")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Smoothness Gradient Audit

## Decision

`{payload['decision']}`

## Scalar Normalization Ledger

{_key_value_table(payload['scalar_normalization_ledger'])}

## Gradient Claim Status

`{payload['gradient_claim_status']}`

## BayesFilter Gradient Status

{_key_value_table(payload['bayesfilter_gradient_status'])}

## Non-Implications

{_non_implications_markdown()}
"""


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


def _rmse(values: list[float]) -> float:
    return (sum(value * value for value in values) / max(len(values), 1)) ** 0.5


def _non_implications() -> list[str]:
    return [
        "No gradient agreement is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No production readiness is concluded.",
        "No full supplement figure or table reproduction is concluded.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


if __name__ == "__main__":
    raise SystemExit(main())
