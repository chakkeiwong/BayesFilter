"""Matched-filterflow LGSSM LEDH-PF-PF-OT diagnostics."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_ot_tf import (
    run_ledh_pfpf_ot_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.jacobians_tf import (
    linear_observation_jacobian_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import (
    gaussian_logpdf_tf,
    ledh_flow_batch_tf,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_lgssm_matched_cross_audit_tf as matched,
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


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-gap-closure-program-plan-2026-05-31.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_matched_ledh_pfpf_ot_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-matched-ledh-pfpf-ot-2026-05-31.md"
THETAS = matched.THETAS
EPSILONS = matched.EPSILONS
NUM_PARTICLES = matched.NUM_PARTICLES
HORIZON = matched.HORIZON
REALIZATION_INDICES = (0, 1, 2)


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
    filterflow = matched._run_filterflow_subprocess()
    observations = tf.constant(filterflow["observations"], dtype=DTYPE)
    initial_particles = tf.constant(filterflow["initial_particles"], dtype=DTYPE)
    spec = matched.MatchedSpec()
    kalman_rows = matched._kalman_rows_tf(observations, spec)
    rows = []
    for theta in THETAS:
        kalman_ll = tf.constant(
            next(row["log_likelihood"] for row in kalman_rows if row["theta"] == theta),
            dtype=DTYPE,
        )
        for epsilon in EPSILONS:
            for realization_index in REALIZATION_INDICES:
                rows.append(
                    _run_one(
                        observations=observations,
                        initial_particles=initial_particles,
                        theta=float(theta),
                        epsilon=float(epsilon),
                        realization_index=int(realization_index),
                        kalman_ll=kalman_ll,
                    )
                )
    summary = _summary(rows)
    decision = "matched_ledh_pfpf_ot_finite_diagnostics"
    if summary["nonfinite_row_count"] > 0:
        decision = "matched_ledh_pfpf_ot_nonfinite_veto"
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Matched-filterflow LGSSM LEDH-PF-PF-OT finite diagnostic comparison.",
        "filterflow_command": filterflow["command"],
        "settings": {
            "horizon": HORIZON,
            "num_particles": NUM_PARTICLES,
            "theta_grid": list(THETAS),
            "epsilon_grid": list(EPSILONS),
            "realization_indices": list(REALIZATION_INDICES),
            "realization_count": len(REALIZATION_INDICES),
            "filterflow_total_realizations_available": len(filterflow["initial_particles"]),
            "transition_covariance": "I_2 executable filterflow convention",
            "observation_covariance": "0.1 I_2",
            "comparison_status": (
                "diagnostic bounded subset; LEDH is not required to match filterflow "
                "RegularisedTransform because it is a different proposal"
            ),
        },
        "kalman_rows": kalman_rows,
        "rows": rows,
        "summary": summary,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_matched_ledh_pfpf_ot_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _run_one(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    theta: float,
    epsilon: float,
    realization_index: int,
    kalman_ll: tf.Tensor,
) -> dict[str, Any]:
    transition_matrix = tf.eye(2, dtype=DTYPE) * tf.constant(theta, DTYPE)
    transition_covariance = tf.eye(2, dtype=DTYPE)
    observation_covariance = tf.eye(2, dtype=DTYPE) * tf.constant(0.1, DTYPE)
    observation_jacobian = linear_observation_jacobian_tf(tf.eye(2, dtype=DTYPE))

    def initial_sample(num_particles: int, seed: int) -> tf.Tensor:
        del seed
        selected = tf.cast(initial_particles[realization_index], DTYPE)
        if int(selected.shape[0]) != num_particles:
            raise ValueError("initial particle count mismatch")
        return selected

    def transition_sample(particles: tf.Tensor, seed: int, time_index: int) -> tf.Tensor:
        noise = tf.random.stateless_normal(
            tf.shape(particles),
            seed=_seed_pair(seed + realization_index, 2000 + time_index),
            dtype=DTYPE,
        )
        return tf.constant(theta, DTYPE) * tf.cast(particles, DTYPE) + noise

    def transition_log_density(particles: tf.Tensor, ancestors: tf.Tensor, time_index: int) -> tf.Tensor:
        del time_index
        mean = tf.linalg.matmul(tf.cast(ancestors, DTYPE), transition_matrix, transpose_b=True)
        return gaussian_logpdf_tf(tf.cast(particles, DTYPE) - mean, transition_covariance)

    def observation_log_density(particles: tf.Tensor, observation: tf.Tensor, time_index: int) -> tf.Tensor:
        del time_index
        return gaussian_logpdf_tf(tf.cast(particles, DTYPE) - tf.cast(observation, DTYPE), observation_covariance)

    def ledh_flow(pre_flow: tf.Tensor, ancestors: tf.Tensor, observation: tf.Tensor, time_index: int):
        del time_index
        return ledh_flow_batch_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=lambda x: tf.cast(x, DTYPE),
            observation_jacobian_fn=observation_jacobian,
            observation_residual_fn=lambda predicted, observed: tf.reshape(tf.cast(observed, DTYPE), [-1])
            - tf.reshape(tf.cast(predicted, DTYPE), [-1]),
        )

    try:
        result = run_ledh_pfpf_ot_tf(
            observations=observations,
            initial_sample=initial_sample,
            transition_sample=transition_sample,
            ledh_flow=ledh_flow,
            transition_log_density=transition_log_density,
            observation_log_density=observation_log_density,
            seed=matched.FILTER_SEED + 10000 + realization_index,
            num_particles=NUM_PARTICLES,
            ess_threshold_ratio=0.5,
            sinkhorn_epsilon=epsilon,
            sinkhorn_iterations=120,
            sinkhorn_tolerance=1e-7,
            transport_method="annealed_transport",
            annealed_scaling=0.9,
            annealed_convergence_threshold=1e-3,
            method_id="matched_filterflow_ledh_pfpf_annealed_transport_tf",
        )
        log_likelihood = tf.cast(result.log_likelihood_estimate, DTYPE)
        diagnostics = result.resampling_diagnostics
        return {
            "theta": theta,
            "epsilon": epsilon,
            "realization_index": realization_index,
            "row_status": "executed",
            "finite": bool(result.finite),
            "log_likelihood_estimate": _float(log_likelihood),
            "kalman_log_likelihood": _float(kalman_ll),
            "error_per_time": _float((log_likelihood - kalman_ll) / tf.cast(HORIZON, DTYPE)),
            "resampling_count": result.resampling_count,
            "min_ess": _min_diag(diagnostics, "ess"),
            "max_ess": _max_diag(diagnostics, "ess"),
            "max_sinkhorn_residual": _max_sinkhorn_residual(diagnostics),
            "max_abs_corrected_log_weight": _max_diag(diagnostics, "max_abs_corrected_log_weight"),
            "min_jacobian_singular_value": _min_diag(diagnostics, "min_jacobian_singular_value"),
            "max_abs_forward_log_det": _max_diag(diagnostics, "max_abs_forward_log_det"),
            "pfpf_correction": "log_target_transition_plus_observation_minus_q0_plus_forward_logdet",
            "proposal_density_accounting": "pre_flow_transition_density_plus_forward_log_det",
        }
    except Exception as exc:  # noqa: BLE001 - recorded as diagnostic row.
        return {
            "theta": theta,
            "epsilon": epsilon,
            "realization_index": realization_index,
            "row_status": "veto",
            "finite": False,
            "diagnostic": str(exc),
        }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    executed = [row for row in rows if row["row_status"] == "executed"]
    finite_rows = [row for row in executed if row["finite"]]
    return {
        "row_count": len(rows),
        "executed_row_count": len(executed),
        "finite_row_count": len(finite_rows),
        "nonfinite_row_count": len(rows) - len(finite_rows),
        "max_abs_error_per_time": _max_abs(row.get("error_per_time") for row in executed),
        "max_sinkhorn_residual": _max_abs(row.get("max_sinkhorn_residual") for row in executed),
        "max_abs_corrected_log_weight": _max_abs(
            row.get("max_abs_corrected_log_weight") for row in executed
        ),
        "min_jacobian_singular_value": _min_value(
            row.get("min_jacobian_singular_value") for row in executed
        ),
        "interpretation": (
            "finite bounded diagnostics on the matched filterflow LGSSM protocol; "
            "not a requirement that LEDH equal filterflow RegularisedTransform"
        ),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != "matched_ledh_pfpf_ot_finite_diagnostics":
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if payload["settings"]["transition_covariance"] != "I_2 executable filterflow convention":
        raise RuntimeError("wrong covariance convention")
    if payload["summary"]["nonfinite_row_count"] != 0:
        raise RuntimeError("non-finite LEDH rows")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Matched Filterflow LGSSM LEDH-PF-PF-OT Diagnostics

## Decision

`{payload['decision']}`

## Settings

- Horizon: `{payload['settings']['horizon']}`
- Particles: `{payload['settings']['num_particles']}`
- Theta grid: `{payload['settings']['theta_grid']}`
- Epsilon grid: `{payload['settings']['epsilon_grid']}`
- Realization indices: `{payload['settings']['realization_indices']}`
- Transition covariance: `{payload['settings']['transition_covariance']}`
- Observation covariance: `{payload['settings']['observation_covariance']}`

## Summary

{_key_value_table(payload['summary'])}

## Rows

{_row_table(payload['rows'])}

## Non-Implications

{_non_implications_markdown()}
"""


def _row_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| theta | eps | realization | status | finite | error/time | resamples | max residual | max abs weight |",
        "| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {theta} | {eps} | {realization} | `{status}` | {finite} | {err} | {resamples} | {resid} | {weight} |".format(
                theta=row["theta"],
                eps=row["epsilon"],
                realization=row["realization_index"],
                status=row["row_status"],
                finite=row.get("finite"),
                err=_fmt(row.get("error_per_time")),
                resamples=_fmt(row.get("resampling_count")),
                resid=_fmt(row.get("max_sinkhorn_residual")),
                weight=_fmt(row.get("max_abs_corrected_log_weight")),
            )
        )
    return "\n".join(lines)


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


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _max_diag(diagnostics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(diag[key]) for diag in diagnostics if key in diag]
    return max(values) if values else None


def _min_diag(diagnostics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(diag[key]) for diag in diagnostics if key in diag]
    return min(values) if values else None


def _max_sinkhorn_residual(diagnostics: list[dict[str, Any]]) -> float:
    residuals = []
    for diag in diagnostics:
        if diag.get("resampling_method") in {
            "finite_sinkhorn_relaxed_tf",
            "fixed_target_sinkhorn_local_comparator_tf",
        }:
            residuals.extend(
                [
                    float(diag.get("max_row_residual", 0.0)),
                    float(diag.get("max_column_residual", 0.0)),
                    float(diag.get("total_mass_residual", 0.0)),
                ]
            )
        elif diag.get("resampling_method") == "filterflow_style_annealed_transport_tf":
            residuals.append(float(diag.get("max_column_residual", 0.0)))
    return max(residuals) if residuals else 0.0


def _max_abs(values) -> float | None:
    clean = [abs(float(value)) for value in values if value is not None]
    return max(clean) if clean else None


def _min_value(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No claim that LEDH must match filterflow RegularisedTransform is concluded.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


if __name__ == "__main__":
    raise SystemExit(main())
