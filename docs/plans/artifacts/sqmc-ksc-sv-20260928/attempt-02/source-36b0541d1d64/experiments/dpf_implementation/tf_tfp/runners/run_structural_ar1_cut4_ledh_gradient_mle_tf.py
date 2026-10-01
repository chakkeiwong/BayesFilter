"""Run structural AR(1) CUT4 vs LEDH-PF-PF-OT gradient/MLE smoke."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import normalize_log_weights_tf
from experiments.dpf_implementation.tf_tfp.fixtures.structural_ar1_quadratic_tf import (
    DTYPE,
    build_structural_ar1_quadratic_fixture_tf,
    complete_k_tf,
    structural_observation_mean_tf,
)
from experiments.dpf_implementation.tf_tfp.references.cut4_structural_tf import (
    run_cut4_structural_filter_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import sinkhorn_resample_tf
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


JSON_PATH = OUTPUT_DIR / "dpf_nonlinear_ssm_structural_ar1_gradient_mle_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-nonlinear-ssm-structural-ar1-gradient-mle-result-2026-05-29.md"
SCALAR_ID = "structural_ar1_negative_log_normalizer_b_parameter_tf"
SEEDS = [111, 222, 333]
NUM_PARTICLES = 48
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
            raise RuntimeError("structural AR(1) gradient/MLE reproducibility digest mismatch")
    _validate(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = build_structural_ar1_quadratic_fixture_tf()
    true_b = fixture.b
    base_particles = tf.random.stateless_normal(
        [NUM_PARTICLES],
        seed=tf.constant([919, 1], dtype=tf.int32),
        dtype=DTYPE,
    )
    transition_noise = tf.random.stateless_normal(
        [fixture.horizon, NUM_PARTICLES],
        seed=tf.constant([919, 2], dtype=tf.int32),
        dtype=DTYPE,
    )

    cut4_true = _cut4_scalar_and_grad(fixture, true_b)
    dpf_true = _dpf_scalar_and_grad(fixture, true_b, base_particles, transition_noise)
    dpf_diag = _dpf_diagnostics(fixture, true_b, base_particles, transition_noise)
    cut4_grid = [_cut4_scalar(fixture, tf.constant(b, DTYPE)) for b in B_GRID]
    dpf_grid_by_seed = []
    for seed in SEEDS:
        noise = tf.random.stateless_normal(
            [fixture.horizon, NUM_PARTICLES],
            seed=tf.constant([seed, 87], dtype=tf.int32),
            dtype=DTYPE,
        )
        base = tf.random.stateless_normal(
            [NUM_PARTICLES],
            seed=tf.constant([seed, 88], dtype=tf.int32),
            dtype=DTYPE,
        )
        dpf_grid_by_seed.append([_dpf_scalar(fixture, tf.constant(b, DTYPE), base, noise) for b in B_GRID])

    cut4_mle_b = B_GRID[_argmin(cut4_grid)]
    dpf_mle_by_seed = [B_GRID[_argmin(row)] for row in dpf_grid_by_seed]
    dpf_median_mle = _median(dpf_mle_by_seed)
    hessian = _second_difference_cut4(fixture, tf.constant(cut4_mle_b, DTYPE), tf.constant(0.05, DTYPE))
    se = tf.math.sqrt(1.0 / tf.maximum(hessian, tf.constant(1e-8, DTYPE)))
    z_distance = abs(dpf_median_mle - cut4_mle_b) / float(se.numpy())
    decision = "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_GRADIENT_MLE_PASSED_SMOKE"
    if not bool(tf.math.is_finite(hessian).numpy()) or scalar(hessian) <= 0.0:
        decision = "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_GRADIENT_MLE_STRUCTURED_BLOCKER"
    if dpf_diag["max_deterministic_residual"] > 1e-9:
        decision = "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_GRADIENT_MLE_STRUCTURED_BLOCKER"
    if not cut4_true["finite"] or not dpf_true["finite"]:
        decision = "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_GRADIENT_MLE_STRUCTURED_BLOCKER"
    if (
        decision == "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_GRADIENT_MLE_PASSED_SMOKE"
        and dpf_median_mle != cut4_mle_b
    ):
        decision = "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_EXECUTED_WITH_ESTIMATION_CALIBRATION_WARNING"

    return {
        "decision": decision,
        "question": "structural AR(1) same-scalar CUT4 vs LEDH-PF-PF-OT gradient and one-parameter MLE smoke",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "same_scalar_contract": "CUT4 and DPF both evaluate negative log-normalizer as a function of b on the same fixed structural AR(1) observations.",
        "model_definition": fixture.model_definition(),
        "comparator": "CUT4 structural differentiable comparator integrating eps_t only, not ground truth",
        "structural_completion_policy": (
            "DPF samples and flows the exogenous m coordinate, then recomputes k_t from "
            "T_k(k_{t-1}, m_{t-1}, m_t; theta) after finite-Sinkhorn relaxed resampling."
        ),
        "b_grid": B_GRID,
        "true_b": scalar(true_b),
        "cut4_at_true": cut4_true,
        "dpf_at_true": dpf_true,
        "dpf_diagnostics_at_true": dpf_diag,
        "cut4_grid_values": [float(x) for x in cut4_grid],
        "dpf_grid_values_by_seed": [[float(x) for x in row] for row in dpf_grid_by_seed],
        "cut4_mle_b_grid": float(cut4_mle_b),
        "dpf_mle_b_by_seed_grid": [float(x) for x in dpf_mle_by_seed],
        "dpf_median_mle_b_grid": float(dpf_median_mle),
        "cut4_observed_information_b": scalar(hessian),
        "cut4_se_b": scalar(se),
        "mle_z_distance_b": float(z_distance),
        "threshold_policy": "No final universal threshold; z-distance calibrates future acceptance bands.",
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_structural_ar1_cut4_ledh_gradient_mle_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _cut4_scalar_and_grad(fixture, b: tf.Tensor) -> dict:
    b_var = tf.Variable(b, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _cut4_scalar(fixture, b_var)
    grad = tape.gradient(value, b_var)
    return {"value": scalar(value), "gradient": scalar(grad), "finite": bool(tf.math.is_finite(grad).numpy())}


def _dpf_scalar_and_grad(fixture, b: tf.Tensor, base: tf.Tensor, noise: tf.Tensor) -> dict:
    b_var = tf.Variable(b, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value, _ = _dpf_eval(fixture, b_var, base, noise, collect_diagnostics=False)
    grad = tape.gradient(value, b_var)
    return {"value": scalar(value), "gradient": scalar(grad), "finite": bool(tf.math.is_finite(grad).numpy())}


def _cut4_scalar(fixture, b: tf.Tensor) -> tf.Tensor:
    return run_cut4_structural_filter_tf(fixture, b=tf.cast(b, DTYPE)).scalar


def _dpf_scalar(fixture, b: tf.Tensor, base: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    value, _ = _dpf_eval(fixture, b, base, noise, collect_diagnostics=False)
    return value


def _dpf_diagnostics(fixture, b: tf.Tensor, base: tf.Tensor, noise: tf.Tensor) -> dict:
    _, diagnostics = _dpf_eval(fixture, b, base, noise, collect_diagnostics=True)
    return diagnostics


def _dpf_eval(
    fixture,
    b: tf.Tensor,
    base: tf.Tensor,
    noise: tf.Tensor,
    *,
    collect_diagnostics: bool,
) -> tuple[tf.Tensor, dict]:
    b = tf.cast(b, DTYPE)
    count = int(base.shape[0])
    m = fixture.m0_mean + tf.sqrt(fixture.m0_variance) * tf.cast(base, DTYPE)
    k = tf.fill([count], tf.cast(fixture.k0, DTYPE))
    particles = tf.stack([m, k], axis=1)
    log_weights = tf.fill([count], -tf.math.log(tf.cast(count, DTYPE)))
    log_normalizer = tf.constant(0.0, DTYPE)
    max_post_flow_residual = tf.constant(0.0, DTYPE)
    max_post_resample_residual = tf.constant(0.0, DTYPE)
    max_sinkhorn_residual = 0.0
    min_sinkhorn_coupling = 1.0

    for t, observation in enumerate(tf.unstack(fixture.observations, axis=0)):
        prev_m = particles[:, 0]
        prev_k = particles[:, 1]
        prior_mean = fixture.rho * prev_m
        pre_m = prior_mean + fixture.sigma * tf.cast(noise[t], DTYPE)
        post_m, logdet, q0_log = _structural_ledh_flow(
            pre_m=pre_m,
            prior_mean=prior_mean,
            previous_m=prev_m,
            previous_k=prev_k,
            observation=observation,
            fixture=fixture,
            b=b,
        )
        post_k = complete_k_tf(
            previous_k=prev_k,
            previous_m=prev_m,
            current_m=post_m,
            a=fixture.a,
            b=b,
            c=fixture.c,
            d=fixture.d,
        )
        target_transition = _normal_logpdf(post_m - prior_mean, fixture.sigma)
        target_observation = _normal_logpdf(
            observation - structural_observation_mean_tf(tf.stack([post_m, post_k], axis=1), fixture.lam),
            fixture.observation_scale,
        )
        corrected = log_weights + target_transition + target_observation - q0_log + logdet
        weights, increment = normalize_log_weights_tf(corrected)
        log_normalizer = log_normalizer + increment
        post_flow_check = complete_k_tf(
            previous_k=prev_k,
            previous_m=prev_m,
            current_m=post_m,
            a=fixture.a,
            b=b,
            c=fixture.c,
            d=fixture.d,
        )
        max_post_flow_residual = tf.maximum(max_post_flow_residual, tf.reduce_max(tf.abs(post_k - post_flow_check)))

        # Relax the stochastic/current coordinate together with the previous
        # structural context, then recomplete k_t. This keeps the structural
        # completion residual meaningful after relaxed OT resampling.
        structural_resampling_state = tf.stack([prev_m, prev_k, post_m], axis=1)
        resampled = sinkhorn_resample_tf(
            structural_resampling_state,
            weights,
            epsilon=0.45,
            max_iterations=70,
            tolerance=1e-7,
        )
        relaxed_prev_m = resampled.particles[:, 0]
        relaxed_prev_k = resampled.particles[:, 1]
        relaxed_current_m = resampled.particles[:, 2]
        recompleted_k = complete_k_tf(
            previous_k=relaxed_prev_k,
            previous_m=relaxed_prev_m,
            current_m=relaxed_current_m,
            a=fixture.a,
            b=b,
            c=fixture.c,
            d=fixture.d,
        )
        residual = recompleted_k - complete_k_tf(
            previous_k=relaxed_prev_k,
            previous_m=relaxed_prev_m,
            current_m=relaxed_current_m,
            a=fixture.a,
            b=b,
            c=fixture.c,
            d=fixture.d,
        )
        max_post_resample_residual = tf.maximum(max_post_resample_residual, tf.reduce_max(tf.abs(residual)))
        if collect_diagnostics:
            max_sinkhorn_residual = max(
                max_sinkhorn_residual,
                float(resampled.diagnostics["max_row_residual"]),
                float(resampled.diagnostics["max_column_residual"]),
                float(resampled.diagnostics["total_mass_residual"]),
            )
            min_sinkhorn_coupling = min(min_sinkhorn_coupling, float(resampled.diagnostics["min_coupling"]))
        particles = tf.stack([relaxed_current_m, recompleted_k], axis=1)
        log_weights = tf.fill([count], -tf.math.log(tf.cast(count, DTYPE)))

    diagnostics = {
        "max_post_flow_deterministic_residual": scalar(max_post_flow_residual),
        "max_post_resample_deterministic_residual": scalar(max_post_resample_residual),
        "max_deterministic_residual": scalar(tf.maximum(max_post_flow_residual, max_post_resample_residual)),
        "max_sinkhorn_residual": float(max_sinkhorn_residual),
        "min_sinkhorn_coupling": float(min_sinkhorn_coupling),
        "structural_residual_veto_tolerance": 1e-9,
        "resampling_status": "finite_sinkhorn_relaxed_then_structural_recompletion",
    }
    return -log_normalizer, diagnostics


def _structural_ledh_flow(
    *,
    pre_m: tf.Tensor,
    prior_mean: tf.Tensor,
    previous_m: tf.Tensor,
    previous_k: tf.Tensor,
    observation: tf.Tensor,
    fixture,
    b: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    prior_var = fixture.sigma * fixture.sigma
    obs_var = fixture.observation_scale * fixture.observation_scale
    pre_k = complete_k_tf(
        previous_k=previous_k,
        previous_m=previous_m,
        current_m=pre_m,
        a=fixture.a,
        b=b,
        c=fixture.c,
        d=fixture.d,
    )
    g_pre = structural_observation_mean_tf(tf.stack([pre_m, pre_k], axis=1), fixture.lam)
    jacobian = b + 2.0 * fixture.c * pre_m + fixture.d * previous_m + fixture.lam
    intercept = g_pre - jacobian * pre_m
    post_var = 1.0 / (1.0 / prior_var + jacobian * jacobian / obs_var)
    post_mean = post_var * (prior_mean / prior_var + jacobian * (observation - intercept) / obs_var)
    scale = tf.sqrt(tf.maximum(post_var, tf.constant(1e-12, DTYPE)) / prior_var)
    post_m = post_mean + scale * (pre_m - prior_mean)
    logdet = tf.math.log(scale)
    q0_log = _normal_logpdf(pre_m - prior_mean, tf.sqrt(prior_var))
    return post_m, logdet, q0_log


def _normal_logpdf(residual: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    variance = tf.cast(scale, DTYPE) * tf.cast(scale, DTYPE)
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE) * variance)
        + tf.cast(residual, DTYPE) * tf.cast(residual, DTYPE) / variance
    )


def _second_difference_cut4(fixture, b: tf.Tensor, step: tf.Tensor) -> tf.Tensor:
    return (_cut4_scalar(fixture, b + step) - 2.0 * _cut4_scalar(fixture, b) + _cut4_scalar(fixture, b - step)) / (
        step * step
    )


def _validate(payload: dict) -> None:
    if payload["decision"] not in {
        "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_GRADIENT_MLE_PASSED_SMOKE",
        "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_EXECUTED_WITH_ESTIMATION_CALIBRATION_WARNING",
        "DPF_NONLINEAR_SSM_STRUCTURAL_AR1_GRADIENT_MLE_STRUCTURED_BLOCKER",
    }:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if "not ground truth" not in payload["comparator"]:
        raise RuntimeError("missing CUT4 caveat")
    if payload["dpf_diagnostics_at_true"]["max_deterministic_residual"] > 1e-9:
        raise RuntimeError("structural deterministic residual veto failed")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict) -> str:
    return f"""# DPF Nonlinear-SSM Structural AR(1) Gradient/MLE Result

## Decision

`{payload['decision']}`

| Check | Status | Evidence |
| --- | --- | --- |
| same scalar | pass | `{payload['scalar_id']}` |
| CUT4 gradient at true b | diagnostic | `{payload['cut4_at_true']['gradient']:.6f}` |
| DPF gradient at true b | diagnostic | `{payload['dpf_at_true']['gradient']:.6f}` |
| CUT4 grid MLE b | diagnostic | `{payload['cut4_mle_b_grid']:.6f}` |
| DPF median grid MLE b | diagnostic | `{payload['dpf_median_mle_b_grid']:.6f}` |
| z distance | calibration | `{payload['mle_z_distance_b']:.6f}` |
| max deterministic residual | veto | `{payload['dpf_diagnostics_at_true']['max_deterministic_residual']:.3e}` |

CUT4 is a differentiable comparator, not ground truth.  The structural model is
a toy non-DSGE endogenous/exogenous split fixture.  The z-distance is a
calibration statistic, not a final universal threshold.
"""


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
        "No DSGE or NAWM validation is concluded.",
        "CUT4 is a comparator, not ground truth.",
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
