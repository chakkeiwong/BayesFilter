"""CPU-only independent Kalman diagnostic; not a production sampler or gate on prior preference."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.testing import surrogate_hmc_literature_priors as priors

started = time.monotonic()
case = runtime.configure_prior(runtime.load_case("lgssm"), priors.PRIOR_ID)
f64 = tf.float64
constant = lambda x: tf.constant(x, f64)
# Explicit reference/diagnostic exception: TFP generates independent inputs
# outside the compiled likelihood. No training or runtime selection uses them.
phi = tfp.distributions.TruncatedNormal(constant(0.), tf.sqrt(constant(.5)),
    constant(-.55), constant(.55)).sample(64, seed=[26101517, 1])
variances = tfp.distributions.InverseGamma(constant(1.), constant(1.)).sample(
    [64, 2], seed=[26101517, 2])
location = tf.random.stateless_normal([64], [26101517, 3], dtype=f64)
theta = tf.stack([tf.math.atanh(phi/.55), tf.math.log(variances[:, 0]),
    tf.math.log(variances[:, 1]), location], axis=-1)
center = tf.cast(case.theta0, f64)
scales = tf.constant(priors.CHART_SCALES["lgssm"], f64)
q_draws = (theta-center)/scales
starts = tf.constant([[-.75]*4, [-.25]*4, [.25]*4, [.75]*4], f64)
positions = tf.concat([tf.zeros([1, 4], f64), starts, q_draws], axis=0)
log_prior, score = case.log_prior(positions)
reconstructed, _ = case.chart(q_draws)
chart_error = tf.reduce_max(tf.abs(reconstructed-theta))
components = runtime.bindings(case)
evaluation = components["target"].function(positions)
valid = evaluation.status["value_valid"]
passed = bool(tf.reduce_all(valid) & tf.reduce_all(tf.math.is_finite(log_prior))
    & tf.reduce_all(tf.math.is_finite(score)) & (chart_error < constant(1e-12)))
result = dict(status="PASS_CHECKED_INPUTS" if passed else "NUMERICAL_REPAIR_REQUIRED",
    cpu_only=True, gpu_status="intentionally hidden before TensorFlow import",
    jit_compile=True, likelihood="exact Kalman", prior_id=priors.PRIOR_ID,
    inputs="nominal, four actual HMC initial positions, 64 independent prior draws",
    seed=26101517, prior_draws=64, total_inputs=69, valid=valid,
    physical_phi=phi, variance_multipliers=variances, initial_mean_coefficient=location,
    q=positions, log_prior=log_prior, score=score, potential=evaluation.potential,
    chart_max_absolute_error=chart_error, scope=case.scope(),
    wall_seconds=time.monotonic()-started,
    command=[sys.executable, str(Path(__file__).resolve())],
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    plan="docs/plans/surrogate-hmc-literature-priors-2026-10-15.md",
    nonclaims=["No posterior samples or convergence assessment", "No global numerical support proof",
        "No prior selection or prior-predictive goodness-of-fit decision"])
runtime.save(Path(__file__).with_suffix(".json"), result)
print(result["status"], "valid", int(tf.reduce_sum(tf.cast(valid, tf.int32))), "/69",
      "seconds", result["wall_seconds"])
sys.exit(0 if passed else 1)
