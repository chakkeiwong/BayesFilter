"""Run linear structural AR(1) Kalman vs LEDH-PF-PF-OT MLE smoke."""

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
    StructuralAR1QuadraticTFFixture,
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
from experiments.dpf_implementation.tf_tfp.runners.run_structural_ar1_cut4_ledh_gradient_mle_tf import (
    _dpf_diagnostics,
    _dpf_scalar,
    _dpf_scalar_and_grad,
)


JSON_PATH = OUTPUT_DIR / "dpf_structural_ar1_linear_mle_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-structural-ar1-linear-mle-result-2026-05-29.md"
SCALAR_ID = "structural_ar1_linear_negative_log_likelihood_b_parameter_tf"
SEEDS = [111, 222, 333]
NUM_PARTICLES = 96
B_GRID = [0.35, 0.50, 0.65, 0.80, 0.95]


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
            raise RuntimeError("linear structural AR(1) reproducibility digest mismatch")
    _validate(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = _linear_fixture()
    true_b = fixture.b
    base_particles = tf.random.stateless_normal(
        [NUM_PARTICLES],
        seed=tf.constant([929, 1], dtype=tf.int32),
        dtype=DTYPE,
    )
    transition_noise = tf.random.stateless_normal(
        [fixture.horizon, NUM_PARTICLES],
        seed=tf.constant([929, 2], dtype=tf.int32),
        dtype=DTYPE,
    )
    kalman_true = _kalman_scalar_and_grad(fixture, true_b)
    dpf_true = _dpf_scalar_and_grad(fixture, true_b, base_particles, transition_noise)
    dpf_diag = _dpf_diagnostics(fixture, true_b, base_particles, transition_noise)
    kalman_grid = [_kalman_scalar(fixture, tf.constant(b, DTYPE)) for b in B_GRID]
    dpf_grid_by_seed = []
    for seed in SEEDS:
        noise = tf.random.stateless_normal(
            [fixture.horizon, NUM_PARTICLES],
            seed=tf.constant([seed, 97], dtype=tf.int32),
            dtype=DTYPE,
        )
        base = tf.random.stateless_normal(
            [NUM_PARTICLES],
            seed=tf.constant([seed, 98], dtype=tf.int32),
            dtype=DTYPE,
        )
        dpf_grid_by_seed.append([_dpf_scalar(fixture, tf.constant(b, DTYPE), base, noise) for b in B_GRID])
    kalman_mle_b = B_GRID[_argmin(kalman_grid)]
    dpf_mle_by_seed = [B_GRID[_argmin(row)] for row in dpf_grid_by_seed]
    dpf_median_mle = _median(dpf_mle_by_seed)
    hessian = _second_difference_kalman(fixture, tf.constant(kalman_mle_b, DTYPE), tf.constant(0.05, DTYPE))
    se = tf.math.sqrt(1.0 / tf.maximum(hessian, tf.constant(1e-8, DTYPE)))
    z_distance = abs(dpf_median_mle - kalman_mle_b) / float(se.numpy())
    decision = "DPF_STRUCTURAL_AR1_LINEAR_MLE_PASSED_SMOKE"
    if dpf_median_mle != kalman_mle_b:
        decision = "DPF_STRUCTURAL_AR1_LINEAR_MLE_ESTIMATION_CALIBRATION_WARNING"
    if not kalman_true["finite"] or not dpf_true["finite"] or not bool(tf.math.is_finite(hessian).numpy()):
        decision = "DPF_STRUCTURAL_AR1_LINEAR_MLE_STRUCTURED_BLOCKER"
    if scalar(hessian) <= 0.0 or dpf_diag["max_deterministic_residual"] > 1e-9:
        decision = "DPF_STRUCTURAL_AR1_LINEAR_MLE_STRUCTURED_BLOCKER"
    return {
        "decision": decision,
        "question": "linear structural AR(1) exact Kalman vs LEDH-PF-PF-OT gradient and one-parameter MLE smoke",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "same_scalar_contract": "Kalman and DPF both evaluate negative log-likelihood/log-normalizer as a function of b on the same c=d=0 observations.",
        "model_definition": fixture.model_definition(),
        "comparator": "Exact Kalman reference for this linear-Gaussian structural AR(1) fixture only",
        "b_grid": B_GRID,
        "true_b": scalar(true_b),
        "kalman_at_true": kalman_true,
        "dpf_at_true": dpf_true,
        "dpf_diagnostics_at_true": dpf_diag,
        "kalman_grid_values": [float(x) for x in kalman_grid],
        "dpf_grid_values_by_seed": [[float(x) for x in row] for row in dpf_grid_by_seed],
        "kalman_mle_b_grid": float(kalman_mle_b),
        "dpf_mle_b_by_seed_grid": [float(x) for x in dpf_mle_by_seed],
        "dpf_median_mle_b_grid": float(dpf_median_mle),
        "kalman_observed_information_b": scalar(hessian),
        "kalman_se_b": scalar(se),
        "mle_z_distance_b": float(z_distance),
        "threshold_policy": "Smoke-scale SE distance for calibration; no final universal threshold.",
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_structural_ar1_linear_kalman_ledh_mle_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _linear_fixture() -> StructuralAR1QuadraticTFFixture:
    return build_structural_ar1_quadratic_fixture_tf(
        name="structural_ar1_linear_completion_smoke",
        c_value=0.0,
        d_value=0.0,
        fixture_generation_seed=2026052903,
    )


def _kalman_scalar_and_grad(fixture, b: tf.Tensor) -> dict:
    b_var = tf.Variable(b, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _kalman_scalar(fixture, b_var)
    grad = tape.gradient(value, b_var)
    return {"value": scalar(value), "gradient": scalar(grad), "finite": bool(tf.math.is_finite(grad).numpy())}


def _kalman_scalar(fixture, b: tf.Tensor) -> tf.Tensor:
    return run_kalman_structural_ar1_linear_tf(fixture, b=tf.cast(b, DTYPE)).scalar


def _second_difference_kalman(fixture, b: tf.Tensor, step: tf.Tensor) -> tf.Tensor:
    return (
        _kalman_scalar(fixture, b + step)
        - 2.0 * _kalman_scalar(fixture, b)
        + _kalman_scalar(fixture, b - step)
    ) / (step * step)


def _validate(payload: dict) -> None:
    if payload["decision"] not in {
        "DPF_STRUCTURAL_AR1_LINEAR_MLE_PASSED_SMOKE",
        "DPF_STRUCTURAL_AR1_LINEAR_MLE_ESTIMATION_CALIBRATION_WARNING",
        "DPF_STRUCTURAL_AR1_LINEAR_MLE_STRUCTURED_BLOCKER",
    }:
        raise RuntimeError(payload["decision"])
    if payload["model_definition"]["c"] != 0.0 or payload["model_definition"]["d"] != 0.0:
        raise RuntimeError("linear fixture must set c=d=0")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if payload["dpf_diagnostics_at_true"]["max_deterministic_residual"] > 1e-9:
        raise RuntimeError("structural deterministic residual veto failed")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict) -> str:
    return f"""# DPF Structural AR(1) Linear MLE Result

## Decision

`{payload['decision']}`

| Check | Kalman | DPF | Role |
| --- | ---: | ---: | --- |
| value at true b | `{payload['kalman_at_true']['value']:.6f}` | `{payload['dpf_at_true']['value']:.6f}` | same-scalar smoke |
| gradient at true b | `{payload['kalman_at_true']['gradient']:.6f}` | `{payload['dpf_at_true']['gradient']:.6f}` | gradient smoke |
| grid MLE b | `{payload['kalman_mle_b_grid']:.6f}` | `{payload['dpf_median_mle_b_grid']:.6f}` | estimation smoke |
| SE-scaled MLE distance | `{payload['mle_z_distance_b']:.6f}` | `{payload['mle_z_distance_b']:.6f}` | calibration |
| max deterministic residual | `{payload['dpf_diagnostics_at_true']['max_deterministic_residual']:.3e}` | `{payload['dpf_diagnostics_at_true']['max_deterministic_residual']:.3e}` | veto |

This is an exact-Kalman comparison only for the `c=d=0` linear structural toy
fixture.  It does not validate the nonlinear `c,d != 0` case or DSGE/NAWM.
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
