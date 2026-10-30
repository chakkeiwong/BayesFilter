"""Run TF/TFP range-bearing LEDH-PF-PF-OT smoke diagnostics."""

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
from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import (
    build_range_bearing_fixture_tf,
    gaussian_logpdf_zero_mean_tf,
    observation_log_density_tf,
    observation_residual_tf,
    range_bearing_observation_tf,
    sample_initial_particles_tf,
    sample_transition_particles_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.jacobians_tf import (
    range_bearing_jacobian_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf
from experiments.dpf_implementation.tf_tfp.references.ukf_tf import (
    run_range_bearing_ukf_tf,
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


JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_ot_tf_tfp_range_bearing_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-ot-tf-tfp-range-bearing-result-2026-05-29.md"
SEEDS = [31, 43, 59]
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
            raise RuntimeError("range-bearing LEDH-PF-PF-OT reproducibility digest mismatch")
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = build_range_bearing_fixture_tf()
    ukf = run_range_bearing_ukf_tf(fixture)

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
            observation_fn=range_bearing_observation_tf,
            observation_jacobian_fn=range_bearing_jacobian_tf,
            observation_residual_fn=observation_residual_tf,
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
            sinkhorn_epsilon=0.35,
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
            sinkhorn_epsilon=0.35,
            sinkhorn_iterations=80,
            sinkhorn_tolerance=1e-7,
        )
        rows.append(_row(bootstrap, ukf, fixture))
        rows.append(_row(ot, ukf, fixture))
        rows.append(_row(ledh, ukf, fixture))

    bootstrap_rows = [row for row in rows if row["method_id"].startswith("bootstrap")]
    ot_rows = [row for row in rows if row["method_id"].startswith("ot_dpf")]
    ledh_rows = [row for row in rows if row["method_id"].startswith("ledh")]
    metrics = {
        "median_bootstrap_state_rmse_to_ukf": _median([row["state_rmse_to_ukf"] for row in bootstrap_rows]),
        "median_ot_state_rmse_to_ukf": _median([row["state_rmse_to_ukf"] for row in ot_rows]),
        "median_ledh_state_rmse_to_ukf": _median([row["state_rmse_to_ukf"] for row in ledh_rows]),
        "median_ledh_latent_position_rmse": _median([row["latent_position_rmse"] for row in ledh_rows]),
        "median_ledh_observation_proxy_rmse": _median([row["observation_proxy_rmse"] for row in ledh_rows]),
        "max_ledh_sinkhorn_residual": max(row["max_sinkhorn_residual"] for row in ledh_rows),
        "max_ledh_abs_corrected_log_weight": max(
            row["max_abs_corrected_log_weight"] for row in ledh_rows
        ),
        "min_ledh_jacobian_singular_value": min(
            row["min_jacobian_singular_value"] for row in ledh_rows
        ),
    }
    decision = "DPF_LEDH_PFPF_OT_TF_TFP_RANGE_BEARING_PASSED"
    if (
        not ukf.finite
        or metrics["max_ledh_sinkhorn_residual"] > 1e-5
        or metrics["min_ledh_jacobian_singular_value"] <= 1e-12
    ):
        decision = "DPF_LEDH_PFPF_OT_TF_TFP_RANGE_BEARING_FAILED_VETO"
    return {
        "decision": decision,
        "question": "TF/TFP range-bearing LEDH-PF-PF-OT proxy smoke with UKF and comparator filters",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "model_definition": fixture.model_definition(),
        "reference": {
            "reference_id": ukf.reference_id,
            "reference_status": "UKF is approximate for this nonlinear fixture, not ground truth.",
            "finite": ukf.finite,
        },
        "seed_list": SEEDS,
        "num_particles": NUM_PARTICLES,
        "default_experimental_dpf_architecture": "tf_tfp_ledh_pfpf_with_finite_sinkhorn_ot_resampling",
        "bootstrap_ot_dpf_status": "comparator_component_baseline_not_large_model_default",
        "primary_metrics": metrics,
        "rows": rows,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_range_bearing_ledh_pfpf_ot_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _row(result, ukf, fixture) -> dict:
    predicted_observations = range_bearing_observation_tf(result.filtered_means)
    diagnostics = result.resampling_diagnostics
    return {
        "method_id": result.method_id,
        "seed": result.seed,
        "num_particles": result.num_particles,
        "finite": bool(result.finite),
        "log_likelihood_estimate": scalar(result.log_likelihood_estimate),
        "state_rmse_to_ukf": rmse_tf(result.filtered_means, ukf.filtered_means),
        "latent_position_rmse": rmse_tf(result.filtered_means[:, :2], fixture.states[1:, :2]),
        "observation_proxy_rmse": rmse_tf(predicted_observations, fixture.observations),
        "resampling_count": result.resampling_count,
        "max_sinkhorn_residual": max_sinkhorn_residual(diagnostics),
        "max_abs_corrected_log_weight": _max_diag(diagnostics, "max_abs_corrected_log_weight"),
        "min_jacobian_singular_value": _min_diag(diagnostics, "min_jacobian_singular_value"),
        "max_abs_forward_log_det": _max_diag(diagnostics, "max_abs_forward_log_det"),
        "model_checksum": fixture.model_checksum,
        "observation_checksum": fixture.observation_checksum,
        "filtered_means": tensor_to_json(result.filtered_means),
        "ess_by_time": tensor_to_json(result.ess_by_time),
        "comparator_status": "UKF is approximate for the nonlinear fixture, not truth.",
    }


def _validate_payload(payload: dict) -> None:
    if payload["decision"] != "DPF_LEDH_PFPF_OT_TF_TFP_RANGE_BEARING_PASSED":
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if "UKF is approximate" not in payload["reference"]["reference_status"]:
        raise RuntimeError("missing UKF caveat")
    if payload["bootstrap_ot_dpf_status"] != "comparator_component_baseline_not_large_model_default":
        raise RuntimeError("bootstrap OT-DPF status drifted")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    for row in payload["rows"]:
        if not row["finite"]:
            raise RuntimeError("non-finite row")


def _markdown(payload: dict) -> str:
    metrics = payload["primary_metrics"]
    return f"""# TF/TFP LEDH-PF-PF-OT Range-Bearing Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| primary criterion | pass | finite TF/TFP UKF/PF/OT/LEDH rows, corrected-weight diagnostics, Sinkhorn residuals, reproducibility digest |
| median bootstrap state RMSE to UKF | comparator | `{metrics['median_bootstrap_state_rmse_to_ukf']:.6f}` |
| median OT-DPF state RMSE to UKF | comparator | `{metrics['median_ot_state_rmse_to_ukf']:.6f}` |
| median LEDH-PF-PF-OT state RMSE to UKF | proxy | `{metrics['median_ledh_state_rmse_to_ukf']:.6f}` |
| median LEDH latent position RMSE | proxy | `{metrics['median_ledh_latent_position_rmse']:.6f}` |
| median LEDH observation proxy RMSE | proxy | `{metrics['median_ledh_observation_proxy_rmse']:.6f}` |
| max LEDH Sinkhorn residual | veto | `{metrics['max_ledh_sinkhorn_residual']:.3e}` |
| min LEDH Jacobian singular value | veto | `{metrics['min_ledh_jacobian_singular_value']:.3e}` |

## Interpretation

The range-bearing smoke checks the TF/TFP LEDH-PF-PF proposal-correction path
against a UKF approximate reference and comparator filters.  UKF is not ground
truth and proxy RMSE does not establish nonlinear posterior correctness.

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
        "UKF is approximate and proxy RMSE is not correctness evidence.",
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
