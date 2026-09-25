"""Bounded review diagnostics; never an implementation or admission route.

The injected-emitter checks replace only the single-cloud result so that the
real batch wrapper remains under test. The actual-engine checks are unpatched.
Tiny fixtures and CPU exceptions establish mechanics only. No tuning, HMC
campaign, posterior inference, performance ranking, or production claim.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.device == "cpu" else "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
root = Path(__file__).resolve().parent
repo = root.parents[3]
sys.path.insert(0, str(repo))

import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

policy = (
    configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if args.device == "gpu"
    else {"mode": "cpu_reference", "gpu_intentionally_hidden": True}
)
if args.device == "gpu":
    tf.config.experimental.enable_tensor_float_32_execution(True)
    # Device probe precedes diagnostic kernels, with verified memory growth.
    with tf.device("/GPU:0"):
        device_probe = tf.reduce_sum(tf.eye(2))
    device_evidence = {"tensor_device": device_probe.device, "value": float(device_probe.numpy())}
else:
    device_evidence = {"gpu_intentionally_hidden": True}

from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_canonical_score_stages_tf import _sigma_points_with_tangent
import bayesfilter.highdim.ledh_canonical_batch_fused_tf as fused
from bayesfilter.inference.ledh_dual_parameter_target import DualParameterLEDHTarget

source_paths = [
    "docs/plans/artifacts/ledh-graceful-failure-codex-review-memo.md",
    "docs/plans/ledh-graceful-failure-comprehensive-testing-plan.md",
    "bayesfilter/highdim/ledh_numerical_safety_tf.py",
    "bayesfilter/highdim/ledh_unified_reset_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
    "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py",
    "bayesfilter/inference/ledh_dual_parameter_target.py",
]
record = {
    "purpose": "review diagnostics only",
    "command": [sys.executable, *sys.argv],
    "cwd": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    "python": sys.version,
    "tensorflow": tf.__version__,
    "tensorflow_probability": tfp.__version__,
    "device_mode": args.device,
    "device_probe": device_evidence,
    "memory_policy": policy,
    "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    "seeds": "N/A: deterministic fixtures, no random sampling",
    "source_sha256": {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in source_paths},
    "checks": {},
}
start = time.monotonic()


def clean(value):
    if isinstance(value, tf.Tensor):
        return clean(value.numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def save():
    record["wall_seconds"] = time.monotonic() - start
    (root / f"probe-{args.device}.json").write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")


def check(name, fn):
    begin = time.monotonic()
    try:
        output = clean(fn())
        record["checks"][name] = {"status": "returned", "output": output}
    except Exception as exc:
        record["checks"][name] = {"status": "raised", "exception": type(exc).__name__, "message": str(exc)}
        print(f"CHECK {name}: {type(exc).__name__}", flush=True)
        traceback.print_exc()
    record["checks"][name]["wall_seconds"] = time.monotonic() - begin
    save()


def matrices():
    f2 = tf.constant([[.7, .2], [.1, .6]], tf.float64)
    f4 = .8 * tf.eye(4, dtype=tf.float64) + .1 * tf.ones([4, 4], tf.float64)
    h4 = tf.eye(4, dtype=tf.float64)[:2]
    v = tf.constant([0., 0., 1., -1.], tf.float64)
    obs = tf.concat([h4 @ tf.linalg.matrix_power(f4, k) for k in range(4)], axis=0) if hasattr(tf.linalg, "matrix_power") else None
    if obs is None:
        powers, power = [], tf.eye(4, dtype=tf.float64)
        for _ in range(4):
            powers.append(h4 @ power)
            power = power @ f4
        obs = tf.concat(powers, axis=0)
    kron = tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(f4)] * 2).to_dense()
    p = tf.reshape(tf.linalg.solve(tf.eye(16, dtype=tf.float64) - kron, tf.reshape(tf.eye(4, dtype=tf.float64), [16, 1])), [4, 4])
    q = tf.eye(4, dtype=tf.float64)
    return {
        "case1_P0": 1 / (1 - .8**2),
        "case2_eigenvalues": tf.linalg.eigvalsh(tf.constant([[.8, 0.], [0., .5]], tf.float64)),
        "case2_characteristic_residual_at_0_8": tf.linalg.det(f2 - .8 * tf.eye(2, dtype=tf.float64)),
        "case2_characteristic_residual_at_0_5": tf.linalg.det(f2 - .5 * tf.eye(2, dtype=tf.float64)),
        "case3_eigenvalues": tf.linalg.eigvalsh(f4),
        "case3_observability_rank": tf.linalg.matrix_rank(obs),
        "case3_observability_singular_values": tf.linalg.svd(obs, compute_uv=False),
        "case3_O_times_unobservable_mode": tf.linalg.matvec(obs, v),
        "case3_lyapunov_solution_eigenvalues": tf.linalg.eigvalsh(p),
        "case3_lyapunov_residual_frobenius": tf.linalg.norm(p - f4 @ p @ tf.transpose(f4) - q),
        "tensorflow_has_solve_lyapunov": hasattr(tf.linalg, "solve_lyapunov"),
    }


if args.device == "cpu":
    check("plan_matrix_claims", matrices)

for dtype in (tf.float32, tf.float64):
    modes = ("eager", "graph", "xla")
    for mode in modes:
        signature = [tf.TensorSpec([2, 2], dtype)]
        fn = safe_cholesky if mode == "eager" else tf.function(safe_cholesky, input_signature=signature, jit_compile=mode == "xla")
        fixtures = {
            "healthy": [[4., 1.], [1., 3.]],
            "indefinite": [[1., 2.], [2., 1.]],
            "singular": [[1., 1.], [1., 1.]],
            "positive_infinity": [[float("inf"), 0.], [0., 1.]],
            "upper_nan_lower_finite": [[1., float("nan")], [0., 1.]],
            "tiny_well_conditioned": [[1e-20, 0.], [0., 1e-20]],
        }
        for fixture, values in fixtures.items():
            check(f"safe/{dtype.name}/{mode}/{fixture}", lambda fn=fn, values=values, dtype=dtype: fn(tf.constant(values, dtype)))

check("safe/batch_shape_2_1", lambda: {"valid_shape": safe_cholesky(tf.eye(2, batch_shape=[2, 1], dtype=tf.float64))[0].shape.as_list()})

def sigma_probe(cov):
    means = tf.zeros([2, 1], tf.float64)
    return _sigma_points_with_tangent(means, cov, tf.zeros_like(means), tf.zeros_like(cov), scale=1., jitter=1e-12)

for mode in ("eager", "graph", "xla"):
    fn = sigma_probe if mode == "eager" else tf.function(sigma_probe, input_signature=[tf.TensorSpec([2, 1, 1], tf.float64)], jit_compile=mode == "xla")
    check(f"sigma_invalid/{mode}", lambda fn=fn: fn(-tf.ones([2, 1, 1], tf.float64)))

dtype = tf.float64
model = fused.PerPointScoreModel(
    transition_mean_fn=lambda theta, x: .8 * x + theta[:, :1],
    transition_mean_tangent_fn=lambda theta, x, dx, direction: .8 * dx + direction[:, :1],
    observation_fn=lambda x: x,
    observation_jacobian_fn=lambda x: tf.ones([tf.shape(x)[0], 1, 1], dtype),
    observation_tangent_fn=lambda x, dx: dx,
    process_covariance=tf.constant([[.1]], dtype),
    observation_covariance=tf.constant([[.5]], dtype),
)
states = tf.constant([[-1.], [-.3], [.3], [1.]], dtype)
covs = tf.ones([4, 1, 1], dtype) * .2
noises = tf.zeros([1, 4, 1], dtype)
observations = tf.constant([[.1]], dtype)
settings = dict(substeps=1, reset_policy="contract_e", reset_design=tf.constant([[-1.], [-1.], [1.], [1.]], dtype), correction_steps=1, pairwise_steps=1, coordinate_cap=2.)
theta = tf.zeros([1, 1], dtype)
directions = tf.ones([1, 1, 1], dtype)

original_emitter = fused.canonical_value_and_analytical_score
try:
    # Injection is below the actual wrapper, not a replacement for it.
    def sentinel_emitter(model, theta, *args, **kwargs):
        del model, args, kwargs
        return tf.constant(float("-inf"), theta.dtype), tf.zeros([1], theta.dtype)

    fused.canonical_value_and_analytical_score = sentinel_emitter

    def bridge(t):
        return fused.canonical_batch_fused_value_score(model, t, directions, states, covs, noises, observations, **settings)

    for mode in ("eager", "graph", "xla"):
        fn = bridge if mode == "eager" else tf.function(bridge, input_signature=[tf.TensorSpec([1, 1], dtype)], jit_compile=mode == "xla")
        check(f"real_batch_wrapper_injected_sentinel/{mode}", lambda fn=fn: fn(theta))
finally:
    fused.canonical_value_and_analytical_score = original_emitter

def actual_engine(c):
    return fused.canonical_batch_fused_value_score(model, theta, directions, states, c, noises, observations, **settings)

for mode in ("eager", "graph", "xla"):
    fn = actual_engine if mode == "eager" else tf.function(actual_engine, input_signature=[tf.TensorSpec([4, 1, 1], dtype)], jit_compile=mode == "xla")
    for condition, input_cov in (("healthy", covs), ("indefinite_initial_cov", -tf.ones_like(covs))):
        check(f"actual_engine/{mode}/{condition}", lambda fn=fn, input_cov=input_cov: fn(input_cov))

target = DualParameterLEDHTarget(model, states, -tf.ones_like(covs), noises, observations, exact_params=settings, biased_params=settings)

def target_value_grad(t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        value = target(t)
    return value, tape.gradient(value, t)

for mode in ("eager", "xla"):
    fn = target_value_grad if mode == "eager" else tf.function(target_value_grad, input_signature=[tf.TensorSpec([1], dtype)], jit_compile=mode == "xla")
    check(f"actual_dual_target_invalid/{mode}", lambda fn=fn: fn(tf.zeros([1], dtype)))

save()
print(json.dumps({"result": str(root / f"probe-{args.device}.json"), "checks": len(record["checks"]), "wall_seconds": record["wall_seconds"]}), flush=True)
