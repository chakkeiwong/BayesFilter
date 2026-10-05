"""Run structural-interface linear AR(1) Kalman vs DPF policy ladder."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.structural_ar1_quadratic_tf import (
    DTYPE,
    build_structural_ar1_quadratic_fixture_tf,
)
from experiments.dpf_implementation.tf_tfp.references.kalman_structural_ar1_tf import (
    run_kalman_structural_ar1_linear_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    scalar,
    stable_digest,
    utc_now,
    wall_time_call,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.structural.models.structural_ar1_tf import (
    StructuralAR1ModelTF,
)
from experiments.dpf_implementation.tf_tfp.structural.resampling_policies_tf import (
    CATEGORICAL_ANCESTOR,
    NO_RESAMPLING,
    SINKHORN_CURRENT_Z,
    SINKHORN_FULL_CONTEXT,
)
from experiments.dpf_implementation.tf_tfp.structural.structural_filter_tf import (
    run_structural_ledh_pfpf_tf,
    summarize_structural_diagnostics,
)


JSON_PATH = OUTPUT_DIR / "dpf_structural_interface_linear_ar1_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-structural-interface-linear-ar1-result-2026-05-29.md"
SCALAR_ID = "structural_interface_linear_ar1_negative_log_likelihood_b_parameter_tf"
SEEDS = [111, 222, 333]
NUM_PARTICLES = 96
B_GRID = [0.35, 0.50, 0.65, 0.80, 0.95]
POLICIES = [NO_RESAMPLING, CATEGORICAL_ANCESTOR, SINKHORN_CURRENT_Z, SINKHORN_FULL_CONTEXT]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate(_load_json_or_fail(JSON_PATH))
        return 0
    payload, runtime = wall_time_call(_run)
    payload["run_manifest"]["wall_time_seconds"] = runtime
    payload["reproducibility_digest"] = _digest(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    if args.check_reproducibility:
        second = _run()
        second["run_manifest"]["wall_time_seconds"] = payload["run_manifest"]["wall_time_seconds"]
        second["reproducibility_digest"] = _digest(second)
        if second["reproducibility_digest"] != payload["reproducibility_digest"]:
            raise RuntimeError("structural interface linear AR(1) reproducibility digest mismatch")
    _validate(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = build_structural_ar1_quadratic_fixture_tf(
        name="structural_interface_linear_ar1_smoke",
        c_value=0.0,
        d_value=0.0,
        fixture_generation_seed=2026052903,
    )
    true_b = fixture.b
    kalman_true = _kalman_scalar_and_grad(fixture, true_b)
    kalman_grid = [_kalman_scalar(fixture, tf.constant(b, DTYPE)) for b in B_GRID]
    kalman_mle_b = B_GRID[_argmin(kalman_grid)]
    hessian = _second_difference_kalman(fixture, tf.constant(kalman_mle_b, DTYPE), tf.constant(0.05, DTYPE))
    se = tf.math.sqrt(1.0 / tf.maximum(hessian, tf.constant(1e-8, DTYPE)))
    policy_rows = []
    for policy in POLICIES:
        true_eval = _dpf_scalar_and_grad(fixture, true_b, policy, SEEDS[0])
        diag = _dpf_diagnostics(fixture, true_b, policy, SEEDS[0])
        grid_by_seed = []
        for seed in SEEDS:
            grid_by_seed.append([_dpf_scalar(fixture, tf.constant(b, DTYPE), policy, seed) for b in B_GRID])
        mle_by_seed = [B_GRID[_argmin(row)] for row in grid_by_seed]
        median_mle = _median(mle_by_seed)
        z_distance = abs(median_mle - kalman_mle_b) / float(se.numpy())
        policy_rows.append(
            {
                "policy_id": policy,
                "dpf_at_true": true_eval,
                "dpf_diagnostics_at_true": diag,
                "dpf_grid_values_by_seed": [[float(x) for x in row] for row in grid_by_seed],
                "dpf_mle_b_by_seed_grid": [float(x) for x in mle_by_seed],
                "dpf_median_mle_b_grid": float(median_mle),
                "mle_z_distance_b": float(z_distance),
                "matches_kalman_grid_mle": bool(median_mle == kalman_mle_b),
            }
        )
    best_policy = min(policy_rows, key=lambda row: row["mle_z_distance_b"])
    decision = "DPF_STRUCTURAL_INTERFACE_LINEAR_AR1_EXECUTED_WITH_POLICY_LADDER"
    if best_policy["matches_kalman_grid_mle"]:
        decision = "DPF_STRUCTURAL_INTERFACE_LINEAR_AR1_POLICY_MATCHED_KALMAN_GRID_MLE"
    if not kalman_true["finite"] or not bool(tf.math.is_finite(hessian).numpy()) or scalar(hessian) <= 0.0:
        decision = "DPF_STRUCTURAL_INTERFACE_LINEAR_AR1_STRUCTURED_BLOCKER"
    if any(row["dpf_diagnostics_at_true"]["max_completion_residual"] > 1e-9 for row in policy_rows):
        decision = "DPF_STRUCTURAL_INTERFACE_LINEAR_AR1_STRUCTURED_BLOCKER"
    return {
        "decision": decision,
        "question": "structural interface linear AR(1) exact Kalman vs DPF resampling policy ladder",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "same_scalar_contract": "Kalman and structural-interface DPF evaluate negative log-likelihood/log-normalizer as a function of b on the same c=d=0 observations.",
        "model_definition": fixture.model_definition(),
        "comparator": "Exact Kalman reference for this linear-Gaussian structural AR(1) fixture only",
        "true_b": scalar(true_b),
        "b_grid": B_GRID,
        "seed_list": SEEDS,
        "num_particles": NUM_PARTICLES,
        "resampling_policy_ladder": POLICIES,
        "kalman_at_true": kalman_true,
        "kalman_grid_values": [float(x) for x in kalman_grid],
        "kalman_mle_b_grid": float(kalman_mle_b),
        "kalman_observed_information_b": scalar(hessian),
        "kalman_se_b": scalar(se),
        "policy_rows": policy_rows,
        "best_policy_by_z_distance": best_policy["policy_id"],
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_structural_interface_linear_ar1_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _kalman_scalar_and_grad(fixture, b: tf.Tensor) -> dict:
    b_var = tf.Variable(b, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _kalman_scalar(fixture, b_var)
    grad = tape.gradient(value, b_var)
    return {"value": scalar(value), "gradient": scalar(grad), "finite": bool(tf.math.is_finite(grad).numpy())}


def _dpf_scalar_and_grad(fixture, b: tf.Tensor, policy: str, seed: int) -> dict:
    b_var = tf.Variable(b, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _dpf_scalar(fixture, b_var, policy, seed)
    grad = tape.gradient(value, b_var)
    return {"value": scalar(value), "gradient": scalar(grad), "finite": bool(tf.math.is_finite(grad).numpy())}


def _kalman_scalar(fixture, b: tf.Tensor) -> tf.Tensor:
    return run_kalman_structural_ar1_linear_tf(fixture, b=tf.cast(b, DTYPE)).scalar


def _dpf_scalar(fixture, b: tf.Tensor, policy: str, seed: int) -> tf.Tensor:
    model = StructuralAR1ModelTF(fixture=fixture, b=tf.cast(b, DTYPE))
    return run_structural_ledh_pfpf_tf(
        model=model,
        observations=fixture.observations,
        seed=seed,
        num_particles=NUM_PARTICLES,
        resampling_policy_id=policy,
    ).negative_log_likelihood


def _dpf_diagnostics(fixture, b: tf.Tensor, policy: str, seed: int) -> dict:
    model = StructuralAR1ModelTF(fixture=fixture, b=tf.cast(b, DTYPE))
    result = run_structural_ledh_pfpf_tf(
        model=model,
        observations=fixture.observations,
        seed=seed,
        num_particles=NUM_PARTICLES,
        resampling_policy_id=policy,
    )
    summary = summarize_structural_diagnostics(result.diagnostics)
    return {
        **summary,
        "finite": bool(result.finite),
        "resampling_count": result.resampling_count,
        "density_contract": result.diagnostics[0]["density_contract"],
        "flow_logdet_contract": result.diagnostics[0]["flow_logdet_contract"],
    }


def _second_difference_kalman(fixture, b: tf.Tensor, step: tf.Tensor) -> tf.Tensor:
    return (
        _kalman_scalar(fixture, b + step)
        - 2.0 * _kalman_scalar(fixture, b)
        + _kalman_scalar(fixture, b - step)
    ) / (step * step)


def _validate(payload: dict) -> None:
    if payload["decision"] not in {
        "DPF_STRUCTURAL_INTERFACE_LINEAR_AR1_EXECUTED_WITH_POLICY_LADDER",
        "DPF_STRUCTURAL_INTERFACE_LINEAR_AR1_POLICY_MATCHED_KALMAN_GRID_MLE",
        "DPF_STRUCTURAL_INTERFACE_LINEAR_AR1_STRUCTURED_BLOCKER",
    }:
        raise RuntimeError(payload["decision"])
    if payload["model_definition"]["c"] != 0.0 or payload["model_definition"]["d"] != 0.0:
        raise RuntimeError("linear structural fixture must set c=d=0")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if any(row["dpf_diagnostics_at_true"]["max_completion_residual"] > 1e-9 for row in payload["policy_rows"]):
        raise RuntimeError("deterministic residual veto failed")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict) -> str:
    rows = "\n".join(
        "| {policy} | {mle:.6f} | {z:.6f} | {grad:.6f} | {resid:.3e} |".format(
            policy=row["policy_id"],
            mle=row["dpf_median_mle_b_grid"],
            z=row["mle_z_distance_b"],
            grad=row["dpf_at_true"]["gradient"],
            resid=row["dpf_diagnostics_at_true"]["max_completion_residual"],
        )
        for row in payload["policy_rows"]
    )
    return f"""# DPF Structural Interface Linear AR(1) Result

## Decision

`{payload['decision']}`

Kalman grid MLE for `b`: `{payload['kalman_mle_b_grid']:.6f}`.

| Policy | DPF median grid MLE b | SE-scaled distance | DPF gradient at true b | max completion residual |
| --- | ---: | ---: | ---: | ---: |
{rows}

Exact Kalman validates only this `c=d=0` toy structural fixture.
"""


def _non_implications() -> list[str]:
    return [
        "No nonlinear structural equivalence is concluded.",
        "No DSGE or NAWM validation is concluded.",
        "No production readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
    ]


def _argmin(values: list[tf.Tensor]) -> int:
    floats = [float(v.numpy()) for v in values]
    return min(range(len(floats)), key=floats.__getitem__)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _digest(payload: dict) -> str:
    stable = {k: v for k, v in payload.items() if k not in {"created_at_utc", "run_manifest", "reproducibility_digest"}}
    return stable_digest(stable)


def _load_json_or_fail(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
