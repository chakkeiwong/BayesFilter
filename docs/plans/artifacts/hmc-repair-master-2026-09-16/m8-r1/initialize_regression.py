"""CPU-only data-derived initialization diagnostic; no Stan draws used in solve."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(ROOT / "source-gpu-r6"))
import tensorflow as tf
from bayesfilter.testing.inference_validation.posteriordb_targets import PosteriordbTarget, read_zip_json


@tf.function(input_signature=[tf.TensorSpec([100, 5], tf.float64),
                             tf.TensorSpec([100], tf.float64)],
             autograph=False, jit_compile=False)
def initialize(x, y):
    gram = tf.transpose(x) @ x
    rhs = tf.linalg.matvec(x, y, transpose_a=True)
    eye = tf.eye(5, dtype=tf.float64)
    initial_s = tf.reduce_mean(tf.square(y))
    def update(i, s, beta, delta):
        beta = tf.linalg.solve(gram + s / 100. * eye, rhs[:, None])[:, 0]
        residual = y - tf.linalg.matvec(x, beta)
        rss = tf.reduce_sum(tf.square(residual))
        updated = 2. * rss / (99. + tf.sqrt(99.**2 + 4. * rss / 100.))
        return i + 1, updated, beta, tf.abs(updated - s) / s
    i, s, beta, delta = tf.while_loop(
        lambda i, s, beta, delta: tf.logical_and(i < 500, delta > 1.e-10), update,
        (tf.constant(0), initial_s, tf.zeros(5, tf.float64), tf.constant(math.inf, tf.float64)))
    # Recompute the final conditional beta after the final scale update.
    beta = tf.linalg.solve(gram + s / 100. * eye, rhs[:, None])[:, 0]
    return tf.concat([beta, tf.math.log(s)[None] / 2.], 0), i, delta


def main():
    path = REPO / ".localresources/posteriordb-20260918/posterior_database/data/data/sblrc.json.zip"
    data = read_zip_json(path)
    target = PosteriordbTarget("sblrc-blr", data, jit_compile=False)
    q, iterations, delta = initialize(tf.constant(data["X"], tf.float64), tf.constant(data["y"], tf.float64))
    value, gradient = target.log_prob_and_grad(q)
    zero, _ = target.log_prob_and_grad(tf.zeros(6, tf.float64))
    tolerance = 1.e-6
    valid = (bool(tf.reduce_all(tf.math.is_finite(q))) and bool(tf.math.is_finite(value))
             and float(value) > float(zero) and float(delta) <= 1.e-10
             and float(tf.reduce_max(tf.abs(gradient))) <= tolerance)
    result = {"initial_position": q.numpy().tolist(), "iterations": int(iterations),
        "relative_change": float(delta), "log_density": float(value), "zero_log_density": float(zero),
        "gradient": gradient.numpy().tolist(), "gradient_tolerance": tolerance,
        "gradient_tolerance_provenance": "FP64 absolute score check; compared with zero-point score in the same target units",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_signature": target.adapter_signature(), "reference_draws_used": False,
        "cpu_reference_only": True, "gpu_intentionally_hidden": True,
        "jit_compile": False, "non_xla_reason": "short independent initialization diagnostic; no sampler/performance claim",
        "passed": valid, "command": sys.argv,
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"}
    with (ROOT / "regression-data-initialization.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    if not valid:
        raise ValueError("initialization solve failed its finite/gradient check")


if __name__ == "__main__":
    main()
