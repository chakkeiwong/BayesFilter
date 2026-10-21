"""Run TF/TFP LGSSM LEDH-PF-PF-OT smoke diagnostics."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    run_bootstrap_particle_filter_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.dpf_ot_tf import run_ot_dpf_tf
from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_ot_tf import (
    run_ledh_pfpf_ot_tf,
)
from experiments.dpf_implementation.tf_tfp.fixtures.lgssm_tf import (
    build_lgssm_fixture_tf,
    gaussian_logpdf_zero_mean_tf,
    observation_log_density_tf,
    sample_initial_particles_tf,
    sample_transition_particles_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.jacobians_tf import (
    linear_observation_jacobian_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf
from experiments.dpf_implementation.tf_tfp.references.kalman_lgssm_tf import (
    run_kalman_filter_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    max_sinkhorn_residual,
    rmse_tf,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    wall_time_call,
    write_json,
    write_text,
)


JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_ot_tf_tfp_lgssm_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-ot-tf-tfp-lgssm-result-2026-05-29.md"
SEEDS = [111, 222, 333]
NUM_PARTICLES = 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        payload = _load_json_or_fail(JSON_PATH)
        _validate_payload(payload)
        return 0
    payload, runtime = wall_time_call(_run)
    payload["run_manifest"]["wall_time_seconds"] = runtime
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    if args.check_reproducibility:
        second = _run()
        second["run_manifest"]["wall_time_seconds"] = payload["run_manifest"]["wall_time_seconds"]
        second["reproducibility_digest"] = _digest_payload(second)
        if second["reproducibility_digest"] != payload["reproducibility_digest"]:
            raise RuntimeError("LGSSM LEDH-PF-PF-OT reproducibility digest mismatch")
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = build_lgssm_fixture_tf()
    kalman = run_kalman_filter_tf(fixture)
    observation_jacobian = linear_observation_jacobian_tf(fixture.C)

    def initial(num_particles: int, seed: int) -> tf.Tensor:
        return sample_initial_particles_tf(fixture, num_particles=num_particles, seed=seed)

    def transition(particles: tf.Tensor, seed: int, time_index: int) -> tf.Tensor:
        return sample_transition_particles_tf(
            fixture,
            particles,
            seed=seed,
            time_index=time_index,
        )

    def obs_log_density(particles: tf.Tensor, observation: tf.Tensor, time_index: int) -> tf.Tensor:
        del time_index
        return observation_log_density_tf(fixture, particles, observation)

    def transition_log_density(particles: tf.Tensor, ancestors: tf.Tensor, time_index: int) -> tf.Tensor:
        del time_index
        mean = tf.linalg.matmul(tf.cast(ancestors, fixture.A.dtype), fixture.A, transpose_b=True)
        return gaussian_logpdf_zero_mean_tf(tf.cast(particles, fixture.A.dtype) - mean, fixture.Q)

    def ledh_flow(pre_flow: tf.Tensor, ancestors: tf.Tensor, observation: tf.Tensor, time_index: int):
        del time_index
        return ledh_flow_batch_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=fixture.A,
            transition_covariance=fixture.Q,
            observation_covariance=fixture.R,
            observation_fn=lambda x: tf.linalg.matvec(fixture.C, x),
            observation_jacobian_fn=observation_jacobian,
            observation_residual_fn=lambda predicted, observed: tf.reshape(observed, [-1])
            - tf.reshape(predicted, [-1]),
        )

    rows = []
    for seed in SEEDS:
        bootstrap = run_bootstrap_particle_filter_tf(
            observations=fixture.observations,
            initial_sample=initial,
            transition_sample=transition,
            observation_log_density=obs_log_density,
            seed=seed,
            num_particles=NUM_PARTICLES,
        )
        ot = run_ot_dpf_tf(
            observations=fixture.observations,
            initial_sample=initial,
            transition_sample=transition,
            observation_log_density=obs_log_density,
            seed=seed,
            num_particles=NUM_PARTICLES,
            sinkhorn_epsilon=0.7,
            sinkhorn_iterations=80,
            sinkhorn_tolerance=1e-7,
        )
        ledh = run_ledh_pfpf_ot_tf(
            observations=fixture.observations,
            initial_sample=initial,
            transition_sample=transition,
            ledh_flow=ledh_flow,
            transition_log_density=transition_log_density,
            observation_log_density=obs_log_density,
            seed=seed,
            num_particles=NUM_PARTICLES,
            sinkhorn_epsilon=0.7,
            sinkhorn_iterations=80,
            sinkhorn_tolerance=1e-7,
        )
        rows.append(_row(bootstrap, kalman, fixture))
        rows.append(_row(ot, kalman, fixture))
        rows.append(_row(ledh, kalman, fixture))

    bootstrap_rows = [row for row in rows if row["method_id"].startswith("bootstrap")]
    ot_rows = [row for row in rows if row["method_id"].startswith("ot_dpf")]
    ledh_rows = [row for row in rows if row["method_id"].startswith("ledh")]
    metrics = {
        "median_bootstrap_filtered_mean_rmse_to_kalman": _median(
            [row["filtered_mean_rmse_to_kalman"] for row in bootstrap_rows]
        ),
        "median_ot_filtered_mean_rmse_to_kalman": _median(
            [row["filtered_mean_rmse_to_kalman"] for row in ot_rows]
        ),
        "median_ledh_filtered_mean_rmse_to_kalman": _median(
            [row["filtered_mean_rmse_to_kalman"] for row in ledh_rows]
        ),
        "median_ledh_abs_log_likelihood_delta_to_kalman": _median(
            [abs(row["log_likelihood_delta_to_kalman"]) for row in ledh_rows]
        ),
        "max_ledh_sinkhorn_residual": max(row["max_sinkhorn_residual"] for row in ledh_rows),
        "max_ledh_abs_corrected_log_weight": max(
            row["max_abs_corrected_log_weight"] for row in ledh_rows
        ),
        "min_ledh_jacobian_singular_value": min(
            row["min_jacobian_singular_value"] for row in ledh_rows
        ),
    }
    smoke_caps = {
        "median_ledh_filtered_mean_rmse_to_kalman": 0.75,
        "median_ledh_abs_log_likelihood_delta_to_kalman": 12.0,
        "max_ledh_sinkhorn_residual": 1e-5,
    }
    decision = "DPF_LEDH_PFPF_OT_TF_TFP_LGSSM_PASSED"
    if (
        metrics["median_ledh_filtered_mean_rmse_to_kalman"]
        > smoke_caps["median_ledh_filtered_mean_rmse_to_kalman"]
        or metrics["median_ledh_abs_log_likelihood_delta_to_kalman"]
        > smoke_caps["median_ledh_abs_log_likelihood_delta_to_kalman"]
        or metrics["max_ledh_sinkhorn_residual"] > smoke_caps["max_ledh_sinkhorn_residual"]
        or metrics["min_ledh_jacobian_singular_value"] <= 1e-12
    ):
        decision = "DPF_LEDH_PFPF_OT_TF_TFP_LGSSM_FAILED_SMOKE_CAP"
    return {
        "decision": decision,
        "question": "TF/TFP LGSSM LEDH-PF-PF-OT smoke against Kalman and comparator filters",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "model_definition": fixture.model_definition(),
        "reference": {
            "reference_id": kalman.reference_id,
            "reference_status": "exact Kalman reference for this LGSSM only",
            "finite": kalman.finite,
            "log_likelihood": scalar(kalman.log_likelihood),
        },
        "seed_list": SEEDS,
        "num_particles": NUM_PARTICLES,
        "default_experimental_dpf_architecture": "tf_tfp_ledh_pfpf_with_finite_sinkhorn_ot_resampling",
        "bootstrap_ot_dpf_status": "comparator_component_baseline_not_large_model_default",
        "primary_metrics": metrics,
        "smoke_caps": smoke_caps,
        "rows": rows,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_lgssm_ledh_pfpf_ot_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _row(result, kalman, fixture) -> dict:
    diagnostics = result.resampling_diagnostics
    return {
        "method_id": result.method_id,
        "seed": result.seed,
        "num_particles": result.num_particles,
        "finite": bool(result.finite),
        "log_likelihood_estimate": scalar(result.log_likelihood_estimate),
        "log_likelihood_delta_to_kalman": scalar(result.log_likelihood_estimate - kalman.log_likelihood),
        "filtered_mean_rmse_to_kalman": rmse_tf(result.filtered_means, kalman.filtered_means),
        "resampling_count": result.resampling_count,
        "max_sinkhorn_residual": max_sinkhorn_residual(diagnostics),
        "max_abs_corrected_log_weight": _max_diag(diagnostics, "max_abs_corrected_log_weight"),
        "min_jacobian_singular_value": _min_diag(diagnostics, "min_jacobian_singular_value"),
        "max_abs_forward_log_det": _max_diag(diagnostics, "max_abs_forward_log_det"),
        "model_checksum": fixture.model_checksum,
        "observation_checksum": fixture.observation_checksum,
        "filtered_means": tensor_to_json(result.filtered_means),
        "ess_by_time": tensor_to_json(result.ess_by_time),
    }


def _validate_payload(payload: dict) -> None:
    if payload["decision"] != "DPF_LEDH_PFPF_OT_TF_TFP_LGSSM_PASSED":
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["bootstrap_ot_dpf_status"] != "comparator_component_baseline_not_large_model_default":
        raise RuntimeError("bootstrap OT-DPF status drifted")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    for row in payload["rows"]:
        if not row["finite"]:
            raise RuntimeError("non-finite row")


def _markdown(payload: dict) -> str:
    metrics = payload["primary_metrics"]
    return f"""# TF/TFP LEDH-PF-PF-OT LGSSM Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| primary criterion | pass | finite TF/TFP LEDH-PF-PF-OT rows, Kalman comparison, corrected-weight diagnostics, Sinkhorn residuals, reproducibility digest |
| median bootstrap RMSE to Kalman | comparator | `{metrics['median_bootstrap_filtered_mean_rmse_to_kalman']:.6f}` |
| median OT-DPF RMSE to Kalman | comparator | `{metrics['median_ot_filtered_mean_rmse_to_kalman']:.6f}` |
| median LEDH-PF-PF-OT RMSE to Kalman | smoke | `{metrics['median_ledh_filtered_mean_rmse_to_kalman']:.6f}` |
| median LEDH abs loglik delta | smoke | `{metrics['median_ledh_abs_log_likelihood_delta_to_kalman']:.6f}` |
| max LEDH Sinkhorn residual | veto | `{metrics['max_ledh_sinkhorn_residual']:.3e}` |
| min LEDH Jacobian singular value | veto | `{metrics['min_ledh_jacobian_singular_value']:.3e}` |

## Interpretation

The LGSSM smoke checks the TF/TFP LEDH-PF-PF proposal-correction path against an
exact Kalman reference and keeps bootstrap PF / bootstrap OT-DPF as comparators.
This nominates LEDH-PF-PF-OT as the default experimental DPF architecture, not
as production or posterior validation.

## Non-Implications

{_non_implications_markdown()}
"""


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
        "No NAWM-scale readiness is concluded.",
        "No banking or model-risk claim is concluded.",
        "No monograph claim is concluded without separate review.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


def _max_diag(diagnostics: list[dict], key: str) -> float:
    values = [float(diag[key]) for diag in diagnostics if key in diag]
    return max(values) if values else 0.0


def _min_diag(diagnostics: list[dict], key: str) -> float:
    values = [float(diag[key]) for diag in diagnostics if key in diag]
    return min(values) if values else 0.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _digest_payload(payload: dict) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _load_json_or_fail(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
