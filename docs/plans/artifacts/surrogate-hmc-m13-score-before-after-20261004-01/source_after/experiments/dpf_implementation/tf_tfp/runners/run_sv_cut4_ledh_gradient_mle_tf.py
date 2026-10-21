"""Run stochastic-volatility CUT4 vs LEDH-PF-PF-OT gradient/MLE smoke."""

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
from experiments.dpf_implementation.tf_tfp.fixtures.stochastic_volatility_tf import (
    DTYPE,
    build_stochastic_volatility_fixture_tf,
    sv_observation_log_density_tf,
    sv_transition_mean_tf,
)
from experiments.dpf_implementation.tf_tfp.references.cut4_sv_tf import run_cut4_sv_filter_tf
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


JSON_PATH = OUTPUT_DIR / "dpf_nonlinear_ssm_sv_gradient_mle_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-nonlinear-ssm-sv-gradient-mle-result-2026-05-29.md"
SCALAR_ID = "sv_negative_log_normalizer_mu_parameter_tf"
SEEDS = [101, 202, 303]
NUM_PARTICLES = 48
MU_GRID = [-1.0, -0.85, -0.70, -0.55, -0.40]


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
            raise RuntimeError("SV gradient/MLE reproducibility digest mismatch")
    _validate(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = build_stochastic_volatility_fixture_tf()
    true_mu = fixture.mu
    base_particles = tf.random.stateless_normal(
        [NUM_PARTICLES],
        seed=tf.constant([909, 1], dtype=tf.int32),
        dtype=DTYPE,
    )
    transition_noise = tf.random.stateless_normal(
        [fixture.horizon, NUM_PARTICLES],
        seed=tf.constant([909, 2], dtype=tf.int32),
        dtype=DTYPE,
    )

    cut4_true = _cut4_scalar_and_grad(fixture, true_mu)
    dpf_true = _dpf_scalar_and_grad(fixture, true_mu, base_particles, transition_noise)
    cut4_grid = [_cut4_scalar(fixture, tf.constant(mu, DTYPE)) for mu in MU_GRID]
    dpf_grid_by_seed = []
    for seed in SEEDS:
        noise = tf.random.stateless_normal(
            [fixture.horizon, NUM_PARTICLES],
            seed=tf.constant([seed, 77], dtype=tf.int32),
            dtype=DTYPE,
        )
        base = tf.random.stateless_normal(
            [NUM_PARTICLES],
            seed=tf.constant([seed, 78], dtype=tf.int32),
            dtype=DTYPE,
        )
        dpf_grid_by_seed.append([_dpf_scalar(fixture, tf.constant(mu, DTYPE), base, noise) for mu in MU_GRID])
    cut4_mle_mu = MU_GRID[_argmin(cut4_grid)]
    dpf_mle_by_seed = [MU_GRID[_argmin(row)] for row in dpf_grid_by_seed]
    dpf_median_mle = _median(dpf_mle_by_seed)
    hessian = _second_difference_cut4(fixture, tf.constant(cut4_mle_mu, DTYPE), tf.constant(0.05, DTYPE))
    se = tf.math.sqrt(1.0 / tf.maximum(hessian, tf.constant(1e-8, DTYPE)))
    z_distance = abs(dpf_median_mle - cut4_mle_mu) / float(se.numpy())
    decision = "DPF_NONLINEAR_SSM_SV_GRADIENT_MLE_PASSED_SMOKE"
    if not bool(tf.math.is_finite(hessian).numpy()) or scalar(hessian) <= 0.0:
        decision = "DPF_NONLINEAR_SSM_SV_GRADIENT_MLE_STRUCTURED_BLOCKER"
    return {
        "decision": decision,
        "question": "SV same-scalar CUT4 vs LEDH-PF-PF-OT gradient and one-parameter MLE smoke",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "same_scalar_contract": "CUT4 and DPF both evaluate negative log-normalizer as a function of mu on the same fixed SV observations.",
        "model_definition": fixture.model_definition(),
        "comparator": "CUT4 differentiable deterministic comparator, not ground truth",
        "mu_grid": MU_GRID,
        "true_mu": scalar(true_mu),
        "cut4_at_true": cut4_true,
        "dpf_at_true": dpf_true,
        "cut4_grid_values": [float(x) for x in cut4_grid],
        "dpf_grid_values_by_seed": [[float(x) for x in row] for row in dpf_grid_by_seed],
        "cut4_mle_mu_grid": float(cut4_mle_mu),
        "dpf_mle_mu_by_seed_grid": [float(x) for x in dpf_mle_by_seed],
        "dpf_median_mle_mu_grid": float(dpf_median_mle),
        "cut4_observed_information_mu": scalar(hessian),
        "cut4_se_mu": scalar(se),
        "mle_z_distance_mu": float(z_distance),
        "threshold_policy": "No final universal threshold; z-distance calibrates future acceptance bands.",
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_sv_cut4_ledh_gradient_mle_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _cut4_scalar_and_grad(fixture, mu: tf.Tensor) -> dict:
    mu_var = tf.Variable(mu, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _cut4_scalar(fixture, mu_var)
    grad = tape.gradient(value, mu_var)
    return {"value": scalar(value), "gradient": scalar(grad), "finite": bool(tf.math.is_finite(grad).numpy())}


def _dpf_scalar_and_grad(fixture, mu: tf.Tensor, base: tf.Tensor, noise: tf.Tensor) -> dict:
    mu_var = tf.Variable(mu, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _dpf_scalar(fixture, mu_var, base, noise)
    grad = tape.gradient(value, mu_var)
    return {"value": scalar(value), "gradient": scalar(grad), "finite": bool(tf.math.is_finite(grad).numpy())}


def _cut4_scalar(fixture, mu: tf.Tensor) -> tf.Tensor:
    return run_cut4_sv_filter_tf(fixture, mu=tf.cast(mu, DTYPE)).scalar


def _dpf_scalar(fixture, mu: tf.Tensor, base: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    mu = tf.cast(mu, DTYPE)
    h0_scale = tf.sqrt(fixture.h0_variance)
    particles = fixture.h0_mean + h0_scale * tf.cast(base, DTYPE)
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_normalizer = tf.constant(0.0, DTYPE)
    prior_var = fixture.sigma * fixture.sigma
    obs_var_for_flow = tf.constant(2.0, DTYPE)
    for t, observation in enumerate(tf.unstack(fixture.observations, axis=0)):
        prior_mean = sv_transition_mean_tf(particles, mu=mu, phi=fixture.phi)
        pre = prior_mean + fixture.sigma * tf.cast(noise[t], DTYPE)
        z = tf.math.log(observation * observation + tf.constant(1e-6, DTYPE))
        post, logdet, q0_log = _scalar_ledh_flow(pre, prior_mean, prior_var, z, obs_var_for_flow)
        target_transition = _normal_logpdf(post - prior_mean, tf.sqrt(prior_var))
        target_observation = sv_observation_log_density_tf(post, observation)
        corrected = log_weights + target_transition + target_observation - q0_log + logdet
        weights, increment = normalize_log_weights_tf(corrected)
        log_normalizer = log_normalizer + increment
        resampled = sinkhorn_resample_tf(post[:, None], weights, epsilon=0.6, max_iterations=60, tolerance=1e-7)
        particles = tf.reshape(resampled.particles, [-1])
        log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    return -log_normalizer


def _scalar_ledh_flow(pre: tf.Tensor, prior_mean: tf.Tensor, prior_var: tf.Tensor, z: tf.Tensor, obs_var: tf.Tensor):
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + z / obs_var)
    scale = tf.sqrt(post_var / prior_var)
    post = post_mean + scale * (pre - prior_mean)
    logdet = tf.math.log(scale)
    q0_log = _normal_logpdf(pre - prior_mean, tf.sqrt(prior_var))
    return post, logdet, q0_log


def _normal_logpdf(residual: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    variance = tf.cast(scale, DTYPE) * tf.cast(scale, DTYPE)
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE) * variance)
        + tf.cast(residual, DTYPE) * tf.cast(residual, DTYPE) / variance
    )


def _second_difference_cut4(fixture, mu: tf.Tensor, step: tf.Tensor) -> tf.Tensor:
    return (_cut4_scalar(fixture, mu + step) - 2.0 * _cut4_scalar(fixture, mu) + _cut4_scalar(fixture, mu - step)) / (step * step)


def _validate(payload: dict) -> None:
    if payload["decision"] not in {
        "DPF_NONLINEAR_SSM_SV_GRADIENT_MLE_PASSED_SMOKE",
        "DPF_NONLINEAR_SSM_SV_GRADIENT_MLE_STRUCTURED_BLOCKER",
    }:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if "not ground truth" not in payload["comparator"]:
        raise RuntimeError("missing CUT4 caveat")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict) -> str:
    return f"""# DPF Nonlinear-SSM SV Gradient/MLE Result

## Decision

`{payload['decision']}`

| Check | Status | Evidence |
| --- | --- | --- |
| same scalar | pass | `{payload['scalar_id']}` |
| CUT4 gradient at true mu | diagnostic | `{payload['cut4_at_true']['gradient']:.6f}` |
| DPF gradient at true mu | diagnostic | `{payload['dpf_at_true']['gradient']:.6f}` |
| CUT4 grid MLE mu | diagnostic | `{payload['cut4_mle_mu_grid']:.6f}` |
| DPF median grid MLE mu | diagnostic | `{payload['dpf_median_mle_mu_grid']:.6f}` |
| z distance | calibration | `{payload['mle_z_distance_mu']:.6f}` |

CUT4 is a differentiable comparator, not ground truth.  The z-distance is a
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
