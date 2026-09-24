"""Q2 Curves 2+3: trust-radius model-trust curve and LM-damping curve.

Contract: `docs/plans/bayesfilter-q2-calibration-campaign-plan-2026-08-24.md`
(Curves 2+3, finalized 2026-08-24). Two phases in one process:

1. CAPTURE: wrap `ledh_canonical_filter_tf._restore_cloud_primal`
   (diagnostic instrumentation), run the canonical filter on frozen
   Austria at the Curve-1 calibrated defaults (k=4, c=8), and keep the
   minimum-stage-ESS (takeoff) step's real (children, weights, design)
   correction input plus the exact reset kwargs the filter passed.
2. LADDERS: on that frozen fixture, drive ONE diagonal shape iteration
   per setting through a mirror of `_shape_iteration_jvp`'s primal math,
   parity-gated against the implementation itself (harness veto on
   mismatch). Radius ladder for the model-trust ratio rho; damping
   ladder for post-residual norm vs scaled-system condition.

The production call site casts the reset to float32, so the primary
ladder dtype is float32; a float64 duplicate is recorded as a
descriptive anchor.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402  (diagnostic/reporting only)
import tensorflow as tf  # noqa: E402

RADIUS_LADDER = [0.1, 0.2, 0.5, 1.0, 2.0, 0.0]  # 0.0 = uncapped arm
DAMPING_LADDER = [0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
STRENGTH = 0.2
FLOOR = 1.0e-5
SCALE_FLOOR = 1.0e-4


def _mirror_iteration(points, target_skew, target_kurt, *, damping, radius):
    """Primal mirror of one `_shape_iteration_jvp` step, returning the
    internals the curves need (r, J, capped displacement, predictions)."""

    from bayesfilter.highdim.higher_moment_contract_e import _right_solve
    from bayesfilter.highdim.genut_shape_lm_tf import (
        scaled_lm_coefficients_value,
        smooth_rms_cap_value,
    )

    dtype = points.dtype
    n_f = tf.cast(tf.shape(points)[0], dtype)
    mean = tf.reduce_mean(points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("ni,nj->ij", centered, centered) / n_f
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    chol = tf.linalg.cholesky(covariance)
    u = _right_solve(chol, centered)
    m3 = tf.reduce_mean(tf.pow(u, 3.0), axis=0)
    m4 = tf.reduce_mean(tf.pow(u, 4.0), axis=0)
    d3 = target_skew - m3
    d4 = target_kurt - m4
    direction3 = tf.square(u) - 1.0 - m3[None, :] * u
    direction4 = tf.pow(u, 3.0) - m3[None, :] - m4[None, :] * u
    j33 = tf.reduce_mean(3.0 * tf.square(u) * direction3, axis=0)
    j34 = tf.reduce_mean(3.0 * tf.square(u) * direction4, axis=0)
    j43 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction3, axis=0)
    j44 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction4, axis=0)
    jacobian = tf.stack(
        [tf.stack([j33, j34], axis=-1), tf.stack([j43, j44], axis=-1)],
        axis=-2,
    )
    residual = tf.stack([d3, d4], axis=-1)
    if damping > 0.0:
        lm = scaled_lm_coefficients_value(
            jacobian,
            residual,
            strength=STRENGTH,
            damping=damping,
            scale_floor=SCALE_FLOOR,
        )
        coefficient = lm["coefficient"]
        condition = float(
            tf.reduce_max(lm["scaled_system_condition"]).numpy()
        )
    else:
        normal = tf.linalg.matmul(jacobian, jacobian, transpose_a=True)
        normal += tf.cast(FLOOR, dtype) * tf.eye(
            2, batch_shape=[tf.shape(points)[1]], dtype=dtype
        )
        rhs = tf.linalg.matvec(jacobian, residual, transpose_a=True)
        coefficient = tf.cast(STRENGTH, dtype) * tf.linalg.solve(
            normal, rhs[:, :, None]
        )[:, :, 0]
        eigenvalues = tf.linalg.eigvalsh(normal)  # diagnostic (2x2)
        condition = float(
            tf.reduce_max(
                eigenvalues[:, 1] / tf.maximum(eigenvalues[:, 0], 1e-30)
            ).numpy()
        )
    displacement = (
        direction3 * coefficient[:, 0][None, :]
        + direction4 * coefficient[:, 1][None, :]
    )
    if radius > 0.0:
        displacement = smooth_rms_cap_value(displacement, radius=radius)[
            "displacement"
        ]
    # Linearized moment-map prediction on the ACTUAL (capped) step
    pred_d3 = d3 - tf.reduce_mean(3.0 * tf.square(u) * displacement, axis=0)
    pred_d4 = d4 - tf.reduce_mean(
        4.0 * tf.pow(u, 3.0) * displacement, axis=0
    )
    # Actual step: apply and re-standardize exactly as the implementation
    corrected = u + displacement
    c_mean = tf.reduce_mean(corrected, axis=0)
    c_centered = corrected - c_mean[None, :]
    c_cov = tf.einsum("ni,nj->ij", c_centered, c_centered) / n_f
    c_cov = 0.5 * (c_cov + tf.transpose(c_cov))
    c_chol = tf.linalg.cholesky(c_cov)
    output = _right_solve(c_chol, c_centered)
    act_d3 = target_skew - tf.reduce_mean(tf.pow(output, 3.0), axis=0)
    act_d4 = target_kurt - tf.reduce_mean(tf.pow(output, 4.0), axis=0)

    def _norm2(a, b):
        return float(
            (tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.square(b))).numpy()
        )

    before = _norm2(d3, d4)
    predicted = before - _norm2(pred_d3, pred_d4)
    actual = before - _norm2(act_d3, act_d4)
    return {
        "output": output,
        "act_d3": act_d3,
        "act_d4": act_d4,
        "residual_norm2_before": before,
        "residual_norm2_after": _norm2(act_d3, act_d4),
        "predicted_reduction": predicted,
        "actual_reduction": actual,
        "trust_ratio": actual / predicted if predicted > 0 else float("nan"),
        "condition": condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--capture-lane", choices=["f32tf32", "f64cpu"], default="f32tf32"
    )
    args = parser.parse_args()
    started = time.time()

    gpus = tf.config.list_physical_devices("GPU")
    memory_growth_verified = False
    if args.capture_lane == "f32tf32":
        if not gpus:
            raise SystemExit("FAIL-CLOSED: f32tf32 capture requires a GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        memory_growth_verified = all(
            tf.config.experimental.get_memory_growth(gpu) for gpu in gpus
        )
        if not memory_growth_verified:
            raise SystemExit("FAIL-CLOSED: memory growth not verified")
        tf.config.experimental.enable_tensor_float_32_execution(True)
        dtype = tf.float32
        device = "/GPU:0"
    else:
        if gpus:
            raise SystemExit("FAIL-CLOSED: f64cpu capture must hide GPUs")
        dtype = tf.float64
        device = "/CPU:0"

    from bayesfilter.highdim import ledh_canonical_filter_tf as filter_mod
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        make_canonical_neutra_target,
    )
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model
    from bayesfilter.highdim.higher_moment_contract_e import (
        weighted_shape_targets_jvp,
        _shape_iteration_jvp,
    )
    from bayesfilter.highdim.genut_guided_proposal_tf import (
        _restore_cloud_primal,
    )

    # --- Phase 1: capture
    captured = []
    original = filter_mod._restore_cloud_primal

    def capturing_restore(particles, weights, design, **kwargs):
        captured.append(
            {
                "particles": tf.identity(particles),
                "weights": tf.identity(weights),
                "design": tf.identity(design),
                "kwargs": dict(kwargs),
            }
        )
        return original(particles, weights, design, **kwargs)

    filter_mod._restore_cloud_primal = capturing_restore
    try:
        with tf.device("/CPU:0"):
            target = make_canonical_neutra_target(
                "austria_sir", particle_count=1008
            )
        theta0 = tf.constant([0.0, 0.0, 0.0], dtype)
        model, _sd = austria_sir_canonical_model(theta0, dtype=dtype)
        observations = tf.cast(target.observations, dtype)
        initial_mean = tf.cast(
            zhao_cui_sir_austria_model().initial_mean, dtype
        )
        variance = 100.0

        def transition_log_density_fn(points, ancestors, _t):
            mean = model.transition_mean_fn(theta0, ancestors)
            residual = points - mean
            return -0.5 * (
                tf.reduce_sum(tf.square(residual), axis=1)
                + 18.0 * tf.constant(np.log(2.0 * np.pi), dtype)
            )

        def observation_log_density_fn(points, observation, _t):
            observed = model.observation_fn(points)
            residual = observation[None, :] - observed
            return -0.5 * (
                tf.reduce_sum(tf.square(residual), axis=1) / variance
                + 9.0 * tf.constant(np.log(2.0 * np.pi * variance), dtype)
            )

        callbacks = CanonicalModelCallbacks(
            model_id=f"austria_sir_q2_curve23_seed{args.seed}",
            state_dim=18,
            observation_dim=9,
            transition_mean_fn=lambda p, t: model.transition_mean_fn(
                theta0, p
            ),
            transition_log_density_fn=transition_log_density_fn,
            process_noise_covariance=model.process_covariance,
            process_noise_covariance_provenance="model_exact",
            observation_fn=lambda p, t: model.observation_fn(p),
            observation_jacobian_fn=lambda p, t: (
                model.observation_jacobian_fn(p)
            ),
            observation_covariance=model.observation_covariance,
            observation_log_density_fn=observation_log_density_fn,
            initial_mean=initial_mean,
            initial_covariance=tf.eye(18, dtype=dtype),
            initial_covariance_provenance="model_exact",
        )
        with tf.device(device):
            result = filter_mod.canonical_value_and_diagnostics(
                callbacks,
                observations,
                particle_count=1008,
                seed=args.seed,
                flow_substeps=16,
                temper_stages=4,
                annealed_resampling=True,
                flow_prior_cap=8.0,
                resample_seed=args.seed,
            )
    finally:
        filter_mod._restore_cloud_primal = original

    ess = result["per_step_ess"].numpy()
    takeoff = int(np.argmin(ess))
    healthy = int(np.argmax(ess))
    fixtures = {
        "takeoff": captured[takeoff],
        "healthy": captured[healthy],
    }
    capture_info = {
        "takeoff_step": takeoff,
        "healthy_step": healthy,
        "per_step_ess": [round(float(e), 1) for e in ess],
        "program_valid": bool(result["program_valid"].numpy()),
        "reset_kwargs": {
            k: v
            for k, v in fixtures["takeoff"]["kwargs"].items()
            if isinstance(v, (int, float, str, bool))
        },
    }

    # --- Phase 2: ladders. Parity is GATED at f64 (math identity: the
    # mirror must be the same function as the implementation). The f32
    # lane repeats the ladder descriptively; its parity gap vs the
    # implementation is float32 accumulation-order noise and is recorded,
    # not gated (the f64 gate already proves the math).
    curves = {}
    for fixture_name, fixture in fixtures.items():
      for lane_dtype, lane_name in ((tf.float32, "f32"), (tf.float64, "f64")):
        children = tf.cast(fixture["particles"], lane_dtype)
        weights = tf.cast(fixture["weights"], lane_dtype)
        design = tf.cast(fixture["design"], lane_dtype)
        kwargs = dict(fixture["kwargs"])
        kwargs["dual_cap_enabled"] = False
        kwargs.pop("trust_region_enabled", None)
        kwargs.pop("trust_region_lm_damping", None)
        kwargs.pop("trust_region_lm_scale_floor", None)
        kwargs.pop("trust_region_radius", None)
        reset = _restore_cloud_primal(children, weights, design, **kwargs)
        points = tf.cast(reset["particles"], lane_dtype)
        zeros_p = tf.zeros(
            [tf.shape(children)[0], tf.shape(children)[1], 1], lane_dtype
        )
        zeros_w = tf.zeros([tf.shape(children)[0], 1], lane_dtype)
        targets = weighted_shape_targets_jvp(
            children, weights, zeros_p, zeros_w
        )
        ts, tk = targets["skew"], targets["kurtosis"]

        # Parity gate (harness veto): the mirror must reproduce the
        # implementation's own iteration at the default setting.
        mirror = _mirror_iteration(
            points, ts, tk, damping=1.0e-2, radius=0.5
        )
        impl = _shape_iteration_jvp(
            points,
            tf.zeros_like(zeros_p),
            ts,
            tk,
            tf.zeros([18, 1], lane_dtype),
            tf.zeros([18, 1], lane_dtype),
            strength=STRENGTH,
            floor=FLOOR,
            lm_damping=1.0e-2,
            lm_scale_floor=SCALE_FLOOR,
            trust_radius=0.5,
        )
        parity = float(
            tf.reduce_max(tf.abs(impl[0] - mirror["output"])).numpy()
        )
        # gate at f64 (math identity); f32 gap recorded as descriptive
        parity_ok = (
            parity < 1.0e-9 if lane_dtype == tf.float64 else None
        )

        radius_curve = {}
        for radius in RADIUS_LADDER:
            m = _mirror_iteration(
                points, ts, tk, damping=1.0e-2, radius=radius
            )
            radius_curve[f"radius_{radius:g}"] = {
                "trust_ratio": round(m["trust_ratio"], 4),
                "predicted_reduction": m["predicted_reduction"],
                "actual_reduction": m["actual_reduction"],
                "residual_norm2_after": m["residual_norm2_after"],
            }
        damping_curve = {}
        for damping in DAMPING_LADDER:
            m = _mirror_iteration(
                points, ts, tk, damping=damping, radius=0.5
            )
            damping_curve[f"damping_{damping:g}"] = {
                "residual_norm2_after": m["residual_norm2_after"],
                "max_scaled_system_condition": m["condition"],
                "trust_ratio": round(m["trust_ratio"], 4),
            }
        curves[f"{fixture_name}_{lane_name}"] = {
            "parity_max_abs": parity,
            "parity_ok": parity_ok,
            "residual_norm2_before": mirror["residual_norm2_before"],
            "radius_curve": radius_curve,
            "damping_curve": damping_curve,
        }

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema": "bayesfilter.q2_trust_damping_curves.v1",
        "plan": "docs/plans/bayesfilter-q2-calibration-campaign-plan-2026-08-24.md",
        "manifest": {
            "commit": commit,
            "command": " ".join(sys.argv),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "unset"
            ),
            "capture_lane": args.capture_lane,
            "memory_growth_verified": memory_growth_verified,
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "seed": args.seed,
            "strength": STRENGTH,
            "floor": FLOOR,
            "scale_floor": SCALE_FLOOR,
            "radius_ladder": RADIUS_LADDER,
            "damping_ladder": DAMPING_LADDER,
            "wall_seconds": round(time.time() - started, 1),
        },
        "capture": capture_info,
        "curves": curves,
    }
    out_dir = os.path.join(
        _ROOT, "docs", "benchmarks", "q2_calibration_20260824",
        f"trust_damping_seed{args.seed}_{int(started)}",
    )
    os.makedirs(out_dir, exist_ok=False)
    out_path = os.path.join(out_dir, "result.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(
        f"[done] seed={args.seed} takeoff={takeoff} healthy={healthy} "
        f"parity_f64_takeoff={curves['takeoff_f64']['parity_ok']} "
        f"parity_f64_healthy={curves['healthy_f64']['parity_ok']} "
        f"artifact={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
