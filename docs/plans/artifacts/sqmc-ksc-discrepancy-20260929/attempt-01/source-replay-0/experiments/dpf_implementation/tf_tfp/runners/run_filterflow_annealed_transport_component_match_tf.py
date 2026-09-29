"""Match reusable annealed transport against executable filterflow LGSSM."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
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
PLAN_PATH = "docs/plans/bayesfilter-dpf-annealed-transport-reference-alignment-plan-2026-05-31.md"
JSON_PATH = OUTPUT_DIR / "dpf_annealed_transport_component_match_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-annealed-transport-component-match-2026-05-31.md"
THETAS = matched.THETAS
EPSILONS = matched.EPSILONS
HORIZON = matched.HORIZON
NUM_PARTICLES = matched.NUM_PARTICLES
NUM_REALIZATIONS = matched.NUM_REALIZATIONS


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
    spec = matched.MatchedSpec()
    observations = tf.constant(filterflow["observations"], dtype=DTYPE)
    initial_particles = tf.constant(filterflow["initial_particles"], dtype=DTYPE)
    kalman_by_theta = {theta: matched._kalman_log_likelihood(observations, theta, spec) for theta in THETAS}
    rows = []
    for epsilon in EPSILONS:
        for theta in THETAS:
            rows.append(
                _run_component_row(
                    observations=observations,
                    initial_particles=initial_particles,
                    spec=spec,
                    theta=float(theta),
                    epsilon=float(epsilon),
                    kalman_ll=kalman_by_theta[theta],
                    filterflow_rows=filterflow["runs"],
                )
            )
    summary = _summary(rows)
    decision = "annealed_transport_component_matched_filterflow"
    if summary["within_band_count"] != 9:
        decision = "annealed_transport_component_match_gap"
    if summary["nonfinite_count"] > 0:
        decision = "annealed_transport_component_nonfinite_veto"
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Reusable TF annealed transport component versus canonical executable filterflow RegularisedTransform.",
        "reference_hierarchy": _reference_hierarchy(),
        "settings": {
            "horizon": HORIZON,
            "num_particles": NUM_PARTICLES,
            "num_realizations": NUM_REALIZATIONS,
            "theta_grid": list(THETAS),
            "epsilon_grid": list(EPSILONS),
            "transition_covariance": "I_2 executable filterflow reproduction setting",
            "observation_covariance": "0.1 I_2",
            "resampling_threshold": "ESS <= 0.5 N",
            "annealed_scaling": 0.9,
            "convergence_threshold": 1e-3,
        },
        "filterflow_command": filterflow["command"],
        "rows": rows,
        "summary": summary,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_annealed_transport_component_match_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _run_component_row(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    spec: matched.MatchedSpec,
    theta: float,
    epsilon: float,
    kalman_ll: tf.Tensor,
    filterflow_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    particles = tf.cast(initial_particles, DTYPE)
    batch_size = tf.shape(particles)[0]
    log_weights = tf.fill([batch_size, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_likelihoods = tf.zeros([batch_size], DTYPE)
    resampling_counts = tf.zeros([batch_size], DTYPE)
    skipped_rows = tf.constant(0.0, DTYPE)
    triggered_rows = tf.constant(0.0, DTYPE)
    max_row_residual = tf.constant(0.0, DTYPE)
    max_column_residual = tf.constant(0.0, DTYPE)
    max_cost_scale = tf.constant(0.0, DTYPE)
    finite = True
    for time_index in range(HORIZON):
        weights = tf.exp(log_weights)
        ess = matched._ess_batched(weights)
        do_resample = ess <= tf.constant(0.5 * NUM_PARTICLES, DTYPE)
        resampled = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=epsilon,
            scaling=0.9,
            convergence_threshold=1e-3,
            max_iterations=100,
            ess_mask=do_resample,
        )
        particles = resampled.particles
        log_weights = resampled.log_weights
        diagnostics = resampled.diagnostics
        triggered_rows += tf.cast(diagnostics["triggered_rows"], DTYPE)
        skipped_rows += tf.cast(diagnostics["skipped_rows"], DTYPE)
        max_row_residual = tf.maximum(max_row_residual, tf.cast(diagnostics["max_row_residual"], DTYPE))
        max_column_residual = tf.maximum(max_column_residual, tf.cast(diagnostics["max_column_residual"], DTYPE))
        max_cost_scale = tf.maximum(max_cost_scale, tf.cast(diagnostics["max_cost_scale"], DTYPE))
        resampling_counts += tf.cast(do_resample, DTYPE)
        particles = matched._transition_particles(
            particles,
            theta,
            spec,
            "bayesfilter_filterflow_style_transport_ess",
            time_index,
        )
        obs_logp = matched._observation_log_prob(particles, observations[time_index], spec)
        unnormalized = log_weights + obs_logp
        normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods += normalizer
        log_weights = unnormalized - normalizer[:, None]
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(particles)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_weights)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_likelihoods)).numpy())
    errors = (log_likelihoods - kalman_ll) / tf.cast(HORIZON, DTYPE)
    filterflow_row = _find_filterflow_row(filterflow_rows, theta, epsilon)
    mean_error = matched._float(tf.reduce_mean(errors))
    std_error = matched._sample_std(errors)
    delta = mean_error - filterflow_row["mean_error_per_time"]
    within = abs(delta) <= filterflow_row["std_error_per_time"]
    return {
        "theta": theta,
        "epsilon": epsilon,
        "row_status": "executed",
        "finite": finite,
        "filterflow_mean_error_per_time": filterflow_row["mean_error_per_time"],
        "filterflow_std_error_per_time": filterflow_row["std_error_per_time"],
        "bayesfilter_mean_error_per_time": mean_error,
        "bayesfilter_std_error_per_time": std_error,
        "delta": delta,
        "within_one_filterflow_sd": within,
        "median_resampling_count": matched._float(matched._median_tensor(resampling_counts)),
        "triggered_rows": matched._float(triggered_rows),
        "skipped_rows": matched._float(skipped_rows),
        "max_row_residual": matched._float(max_row_residual),
        "max_column_residual": matched._float(max_column_residual),
        "max_cost_scale": matched._float(max_cost_scale),
        "algorithm_id": "filterflow_style_annealed_transport_tf",
    }


def _find_filterflow_row(rows: list[dict[str, Any]], theta: float, epsilon: float) -> dict[str, Any]:
    for row in rows:
        if (
            row["method_id"] == "filterflow_regularized"
            and row["theta"] == theta
            and row.get("epsilon") == epsilon
        ):
            return row
    raise KeyError((theta, epsilon))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "within_band_count": sum(1 for row in rows if row["within_one_filterflow_sd"]),
        "nonfinite_count": sum(1 for row in rows if not row["finite"]),
        "max_abs_delta": max(abs(row["delta"]) for row in rows),
        "max_row_residual": max(row["max_row_residual"] for row in rows),
        "max_column_residual": max(row["max_column_residual"] for row in rows),
        "min_triggered_rows": min(row["triggered_rows"] for row in rows),
        "fixed_target_sinkhorn_status": "not_used_local_comparator_only",
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != "annealed_transport_component_matched_filterflow":
        raise RuntimeError(payload["decision"])
    if payload["summary"]["within_band_count"] != 9:
        raise RuntimeError("not all nine epsilon/theta cells matched")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Annealed Transport Component Match

## Decision

`{payload['decision']}`

## Reference Hierarchy

{_key_value_table(payload['reference_hierarchy'])}

## Summary

{_key_value_table(payload['summary'])}

## Nine-Cell Comparison

{_rows_table(payload['rows'])}

## Non-Implications

{_bullets(payload['non_implications'])}
"""


def _rows_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| epsilon | theta | filterflow mean | component mean | delta | within 1 sd | max row residual | max column residual |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {epsilon} | {theta} | {fm:.6g} | {bm:.6g} | {delta:.6g} | {within} | {rr:.3e} | {cr:.3e} |".format(
                epsilon=row["epsilon"],
                theta=row["theta"],
                fm=row["filterflow_mean_error_per_time"],
                bm=row["bayesfilter_mean_error_per_time"],
                delta=row["delta"],
                within=row["within_one_filterflow_sd"],
                rr=row["max_row_residual"],
                cr=row["max_column_residual"],
            )
        )
    return "\n".join(lines)


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _reference_hierarchy() -> dict[str, str]:
    return {
        "canonical_executable_reference": "local filterflow bayesfilter-py311-float64-reference",
        "transition_covariance": "I_2 executable filterflow reproduction setting",
        "fixed_target_sinkhorn": "local BayesFilter diagnostic/comparator only",
        "component_under_test": "reusable TF/TFP annealed_transport_resample_tf",
    }


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No gradient correctness is concluded.",
    ]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
