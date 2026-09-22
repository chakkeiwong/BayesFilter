"""Part-5 battery, final arm: float32 + TF32 on the canonical fused lane.

The historical disease combo was float32/TF32 + compiled modes: value/score
program split (grappler), NaN escape (XLA T=20), cross-mode drift. This arm
runs the fused canonical lane (ONE program: analytical score, no autodiff
twin) at float32 on GPU under {TF32 on, TF32 off} x {eager, graph, XLA}:

  B4a finiteness + program_valid everywhere;
  B4b cross-mode drift at float32 (declared: rtol 1e-3 recorded, drift is
      descriptive beyond finiteness — float32 op-order class);
  B4c TF32-on vs TF32-off drift (descriptive: the TF32 arithmetic effect
      on a program with guarded factorizations and one score path);
  B4d poisoned-observation fail-closed under float32+TF32+XLA — the exact
      historical NaN-escape combo.
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)

import numpy as np
import tensorflow as tf

DTYPE = tf.float32


def build_model():
    from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
        PerPointScoreModel,
    )

    def transition_mean_fn(theta, points):
        return points + theta[:, 0:1] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points, d_theta):
        return (
            d_theta[:, 0:1] * tf.sin(points)
            + d_points
            + theta[:, 0:1] * tf.cos(points) * d_points
        )

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda p: p,
        observation_jacobian_fn=lambda p: tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(p)[0], 2, 2]
        ),
        observation_tangent_fn=lambda p, d: d,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def main() -> None:
    started = time.time()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = "/GPU:0" if gpus else "/CPU:0"

    from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
        canonical_batch_fused_value_score,
    )

    model = build_model()
    rng = np.random.default_rng(601)
    n, horizon = 64, 4
    initial = tf.constant(rng.standard_normal((n, 2)), DTYPE)
    covs = tf.constant(np.stack([np.eye(2)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, 2)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, 2)), DTYPE)
    theta = tf.constant([[0.6], [0.9], [1.2]], DTYPE)
    directions = tf.constant([[1.0], [1.0], [1.0]], DTYPE)

    def call(t, d, obs):
        return canonical_batch_fused_value_score(
            model, t, d, initial, covs, noises, obs, substeps=10,
            jitter=1.0e-6,
        )

    results: dict = {"device": device, "dtype": "float32"}
    with tf.device(device):
        for tf32_state in (True, False):
            tf.config.experimental.enable_tensor_float_32_execution(
                tf32_state
            )
            arm = {}
            value_e, score_e, diag_e = call(theta, directions, observations)
            graph_fn = tf.function(
                lambda t, d: call(t, d, observations), autograph=False
            )
            value_g, score_g, _ = graph_fn(theta, directions)
            xla_fn = tf.function(
                lambda t, d: call(t, d, observations),
                jit_compile=True,
                autograph=False,
            )
            value_x, score_x, _ = xla_fn(theta, directions)

            def rel(a, b):
                return float(
                    tf.reduce_max(
                        tf.abs(a - b)
                        / tf.maximum(tf.abs(b), tf.ones_like(b))
                    ).numpy()
                )

            arm["finite_all_modes"] = bool(
                tf.reduce_all(tf.math.is_finite(value_e)).numpy()
                and tf.reduce_all(tf.math.is_finite(score_e)).numpy()
                and tf.reduce_all(tf.math.is_finite(value_g)).numpy()
                and tf.reduce_all(tf.math.is_finite(score_g)).numpy()
                and tf.reduce_all(tf.math.is_finite(value_x)).numpy()
                and tf.reduce_all(tf.math.is_finite(score_x)).numpy()
            )
            arm["program_valid"] = bool(
                tf.reduce_all(diag_e["program_valid"]).numpy()
            )
            arm["graph_vs_eager"] = {
                "value_rel": rel(value_g, value_e),
                "score_rel": rel(score_g, score_e),
            }
            arm["xla_vs_eager"] = {
                "value_rel": rel(value_x, value_e),
                "score_rel": rel(score_x, score_e),
            }
            arm["values_eager"] = [float(v) for v in value_e.numpy()]
            results[f"tf32_{'on' if tf32_state else 'off'}"] = arm

        # cross-TF32 drift (descriptive)
        on = np.array(results["tf32_on"]["values_eager"])
        off = np.array(results["tf32_off"]["values_eager"])
        results["tf32_on_vs_off_value_rel"] = float(
            np.max(np.abs(on - off) / np.maximum(np.abs(off), 1.0))
        )

        # B4d: the historical disease combo — poisoned obs, float32, TF32
        # on, XLA-compiled. Must fail CLOSED (invalid + NaN-masked), never
        # raise, never silently accept.
        tf.config.experimental.enable_tensor_float_32_execution(True)
        poisoned = tf.tensor_scatter_nd_update(
            observations, [[2, 0]], [tf.cast(float("nan"), DTYPE)]
        )
        xla_poisoned = tf.function(
            lambda t, d: call(t, d, poisoned),
            jit_compile=True,
            autograph=False,
        )
        value_p, score_p, diag_p = xla_poisoned(theta, directions)
        results["b4d_fail_closed_f32_tf32_xla"] = {
            "all_rows_invalid": bool(
                tf.reduce_all(~diag_p["program_valid"]).numpy()
            ),
            "values_nan_masked": bool(
                tf.reduce_all(~tf.math.is_finite(value_p)).numpy()
            ),
            "no_exception": True,
        }

    payload = {
        "schema": "bayesfilter.ledh_canonical_battery_f32_tf32.v1",
        "results": results,
        "wall_seconds": time.time() - started,
    }
    out = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08", "battery_f32_tf32.json",
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
