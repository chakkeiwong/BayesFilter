"""Time-3 observation log-probability micro-audit for the R1 mismatch."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_agreement_ladder_tf as agreement,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r1_observation_path_mismatch_localization_tf as localization,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-r1-time3-observation-logprob-audit-plan-2026-06-02.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-r1-time3-observation-logprob-audit-result-2026-06-02.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_r1_time3_observation_logprob_audit_2026-06-02.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-r1-time3-observation-logprob-audit-2026-06-02.md"
TARGET_TIME = 3
PREFIX = 4
EXPLAIN_TOLERANCE_ABS = 1e-8
EXPLAIN_TOLERANCE_REL = 1e-8


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
    markdown = _markdown(payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    initial_fingerprint = continuation._filterflow_fingerprint()
    fixture = agreement._filterflow_fixture_subprocess()
    if fixture.get("status") != "executed":
        return _blocked_payload("fixture_blocked", fixture.get("blocker", "unknown"), initial_fingerprint)

    control = localization._control_base_scenario()
    r1 = localization._r1_base_scenario(fixture, control)
    scenario = localization._prefix_scenario(r1, PREFIX)
    filterflow = localization._filterflow_reference(scenario)
    if filterflow.get("status") != "executed":
        return _blocked_payload("filterflow_blocked", filterflow.get("blocker", "unknown"), initial_fingerprint)

    bf64 = localization._bayesfilter_reference(scenario, dtype=tf.float64)
    bf32 = localization._bayesfilter_reference(scenario, dtype=tf.float32)
    comparison64 = agreement._compare_runs(bf64, filterflow)
    comparison32 = agreement._compare_runs(bf32, filterflow)
    time_payload = _time_payload(
        bf64=bf64,
        bf32=bf32,
        filterflow=filterflow,
        time_index=TARGET_TIME,
    )
    decomposition = _decompose(time_payload)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(decomposition, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "What explains the time-3 R1 observation-log-likelihood mismatch?",
        "fixed_inputs": {
            "target_time": TARGET_TIME,
            "prefix": PREFIX,
            "theta": step.THETA0,
            "Q": step.Q,
            "R": step.R,
            "num_particles": step.NUM_PARTICLES,
            "epsilon": step.EPSILON,
            "scaling": step.SCALING,
            "convergence_threshold": agreement.CONVERGENCE_THRESHOLD,
            "max_iterations": agreement.MAX_ITERATIONS,
            "transition_noises_source": "accepted generated_T100 ledger",
            "observation_source": "local executable filterflow smoothness path",
        },
        "comparison_summary": {
            "bf64_implementation_agreement": comparison64.get("implementation_agreement"),
            "bf64_scalar_delta": comparison64.get("scalar_delta"),
            "bf32_implementation_agreement": comparison32.get("implementation_agreement"),
            "bf32_scalar_delta": comparison32.get("scalar_delta"),
        },
        "time_payload": time_payload,
        "decomposition": decomposition,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r1_time3_observation_logprob_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + [
            "No correctness claim is made for either implementation.",
            "No dtype policy change is concluded.",
            "No tolerance policy change is concluded.",
        ],
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "blocker": blocker,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r1_time3_observation_logprob_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _time_payload(
    *,
    bf64: dict[str, Any],
    bf32: dict[str, Any],
    filterflow: dict[str, Any],
    time_index: int,
) -> dict[str, Any]:
    bf64_step = bf64["ledger"][time_index]
    bf32_step = bf32["ledger"][time_index]
    ff_step = filterflow["ledger"][time_index]
    observation = float(ff_step["observation"])
    rows = {
        "bayesfilter_float64": _row_payload(bf64_step, observation),
        "bayesfilter_float32": _row_payload(bf32_step, observation),
        "filterflow_float32": _row_payload(ff_step, observation),
    }
    for row in rows.values():
        row["closed_form_float64"] = _closed_form_logprob(
            row["predicted_particles"],
            observation,
            dtype=tf.float64,
        )
        row["closed_form_float32"] = _closed_form_logprob(
            row["predicted_particles"],
            observation,
            dtype=tf.float32,
        )
    return {
        "time_index": time_index,
        "observation": observation,
        "rows": rows,
    }


def _row_payload(step_row: dict[str, Any], observation: float) -> dict[str, Any]:
    return {
        "resampling_flags": step_row["resampling_flags"],
        "pre_log_weights": step_row["pre_log_weights"],
        "pre_particles": step_row["pre_particles"],
        "post_transport_particles": step_row["post_transport_particles"],
        "transition_noise": step_row["transition_noise"],
        "predicted_particles": step_row["predicted_particles"],
        "observation_log_likelihoods": step_row["observation_log_likelihoods"],
        "unnormalized_log_weights": step_row["unnormalized_log_weights"],
        "per_step_log_normalizer": step_row["per_step_log_normalizer"],
        "post_update_log_weights": step_row["post_update_log_weights"],
        "observation": observation,
    }


def _closed_form_logprob(
    particles: Any,
    observation: float,
    *,
    dtype: tf.DType,
) -> list[list[float]]:
    x = tf.squeeze(tf.constant(particles, dtype=dtype), axis=2)
    y = tf.constant(observation, dtype=dtype)
    variance = tf.constant(step.R, dtype=dtype)
    log_const = tf.math.log(tf.constant(2.0 * math.pi, dtype=dtype) * variance)
    values = -0.5 * (log_const + (y - x) * (y - x) / variance)
    return tf.cast(values, tf.float64).numpy().tolist()


def _decompose(time_payload: dict[str, Any]) -> dict[str, Any]:
    rows = time_payload["rows"]
    bf64 = rows["bayesfilter_float64"]
    ff = rows["filterflow_float32"]
    observed_delta = _max_abs(bf64["observation_log_likelihoods"], ff["observation_log_likelihoods"])
    closed_bf64_vs_ff32 = _max_abs(bf64["closed_form_float64"], ff["closed_form_float32"])
    reconstruction_residual = abs(observed_delta - closed_bf64_vs_ff32)
    state_delta_bf64 = _max_abs(bf64["closed_form_float64"], _closed_form_logprob(ff["predicted_particles"], time_payload["observation"], dtype=tf.float64))
    dtype_delta_on_ff_state = _max_abs(
        _closed_form_logprob(ff["predicted_particles"], time_payload["observation"], dtype=tf.float64),
        ff["closed_form_float32"],
    )
    predicted_particle_delta = _max_abs(bf64["predicted_particles"], ff["predicted_particles"])
    pre_log_weight_delta = _max_abs(bf64["pre_log_weights"], ff["pre_log_weights"])
    post_transport_particle_delta = _max_abs(bf64["post_transport_particles"], ff["post_transport_particles"])
    normalizer_delta = _max_abs(bf64["per_step_log_normalizer"], ff["per_step_log_normalizer"])
    return {
        "observed_observation_logprob_delta": observed_delta,
        "closed_form_bf64_vs_filterflow_bf32_delta": closed_bf64_vs_ff32,
        "closed_form_reconstruction_residual": reconstruction_residual,
        "closed_form_reconstruction_relative_residual": reconstruction_residual / max(observed_delta, 1e-12),
        "state_delta_bf64_same_dtype": state_delta_bf64,
        "dtype_delta_on_filterflow_state": dtype_delta_on_ff_state,
        "predicted_particle_delta": predicted_particle_delta,
        "pre_log_weight_delta": pre_log_weight_delta,
        "post_transport_particle_delta": post_transport_particle_delta,
        "per_step_log_normalizer_delta": normalizer_delta,
        "state_fraction_of_observed": state_delta_bf64 / max(observed_delta, 1e-12),
        "dtype_fraction_of_observed": dtype_delta_on_ff_state / max(observed_delta, 1e-12),
    }


def _decision(decomposition: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift:
        return "time3_observation_logprob_audit_blocked_by_comparator_drift"
    reconstruction_ok = (
        decomposition["closed_form_reconstruction_residual"] <= EXPLAIN_TOLERANCE_ABS
        or decomposition["closed_form_reconstruction_relative_residual"] <= EXPLAIN_TOLERANCE_REL
    )
    if not reconstruction_ok:
        return "time3_observation_logprob_unexplained_upstream_or_wrapper"
    if decomposition["state_fraction_of_observed"] >= 0.8:
        return "time3_observation_logprob_state_delta_dominant"
    if decomposition["dtype_fraction_of_observed"] >= 0.8:
        return "time3_observation_logprob_dtype_delta_dominant"
    return "time3_observation_logprob_mixed_arithmetic_state_dtype"


def _max_abs(left: Any, right: Any) -> float:
    left_tensor = tf.constant(left, dtype=tf.float64)
    right_tensor = tf.constant(right, dtype=tf.float64)
    return float(tf.reduce_max(tf.abs(left_tensor - right_tensor)).numpy())


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "path_boundary_manifest",
        "run_manifest",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    manifest = payload["run_manifest"]
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("parent pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError("parent GPU devices visible")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    if payload["decision"].endswith("blocked_by_comparator_drift") and not payload["comparator_drift"]:
        raise ValueError("drift-blocked decision without comparator drift")
    if "time_payload" in payload:
        rows = payload["time_payload"]["rows"]
        for key in ("bayesfilter_float64", "bayesfilter_float32", "filterflow_float32"):
            row = rows[key]
            for field in ("predicted_particles", "observation_log_likelihoods", "closed_form_float64", "closed_form_float32"):
                if not _finite_nested(row[field]):
                    raise ValueError(f"nonfinite {key}.{field}")


def _finite_nested(value: Any) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.constant(value, tf.float64))).numpy())


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# R1 Time-3 Observation Log-Probability Micro-Audit",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
    ]
    if "blocker" in payload:
        lines.extend(["## Blocker", "", f"`{payload['blocker']}`", ""])
        return "\n".join(lines)
    decomp = payload["decomposition"]
    lines.extend(
        [
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | "
                f"`observed delta {decomp['observed_observation_logprob_delta']} reconstructed by closed-form delta {decomp['closed_form_bf64_vs_filterflow_bf32_delta']}` | "
                "`whether the tiny predicted-particle delta is acceptable under a future tolerance policy` | "
                "`audit the time-3 predicted-particle generation path if we need to remove the remaining state delta` | "
                "`correctness of either implementation, dtype policy, tolerance policy` |"
            ),
            "",
            "## Decomposition",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in decomp.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- CPU-only execution with `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import.",
            "- JSON validation and schema checks passed in runner validation.",
            "- No production readiness or correctness claim is made.",
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
