"""CPU-only local-curvature diagnostic; no reference draws or sampler calls."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(ROOT / "source-gpu-r6"))
import tensorflow as tf
from bayesfilter.testing.inference_validation.posteriordb_targets import PosteriordbTarget, read_zip_json
from bayesfilter.inference.hmc_geometry import initialize_hmc_kernel_geometry


@tf.function(input_signature=[tf.TensorSpec([100, 5], tf.float64),
                             tf.TensorSpec([100], tf.float64),
                             tf.TensorSpec([6], tf.float64)],
             autograph=False, jit_compile=False)
def curvature(x, y, q):
    s = tf.exp(2. * q[-1])
    r = y - tf.linalg.matvec(x, q[:-1])
    bb = tf.transpose(x) @ x / s + tf.eye(5, dtype=tf.float64) / 100.
    cross = 2. * tf.linalg.matvec(x, r, transpose_a=True) / s
    scale = 2. * tf.reduce_sum(tf.square(r)) / s + 2. * s / 100.
    return tf.concat([tf.concat([bb, cross[:, None]], 1),
                      tf.concat([cross, scale[None]], 0)[None, :]], 0)


def main():
    initial_path = ROOT / "regression-data-initialization.json"
    initial = json.loads(initial_path.read_text())
    data_path = REPO / ".localresources/posteriordb-20260918/posterior_database/data/data/sblrc.json.zip"
    target = PosteriordbTarget("sblrc-blr", read_zip_json(data_path), jit_compile=False)
    assert initial["passed"] and not initial["reference_draws_used"]
    assert initial["data_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert initial["target_signature"] == target.adapter_signature()
    q = tf.constant(initial["initial_position"], tf.float64)
    data = read_zip_json(data_path)
    hessian = curvature(tf.constant(data["X"], tf.float64), tf.constant(data["y"], tf.float64), q)
    eigenvalues = tf.linalg.eigvalsh(hessian)
    checks = []
    for step in (1.e-4, 1.e-5):
        directions = step * tf.eye(6, dtype=tf.float64)
        _, plus = target.batch_log_prob_and_grad(q[None, :] + directions)
        _, minus = target.batch_log_prob_and_grad(q[None, :] - directions)
        estimate = -tf.transpose((plus - minus) / (2. * step))
        error = float(tf.reduce_max(tf.abs(estimate - hessian) / (1. + tf.abs(hessian))))
        checks.append({"step": step, "max_scaled_error": error, "tolerance": 1.e-5,
                       "passed": error <= 1.e-5})
    original = initialize_hmc_kernel_geometry(adapter=target, initial_position=q)
    local_limit = 2. / float(tf.sqrt(tf.reduce_max(eigenvalues)))
    passed = (bool(tf.reduce_all(tf.math.is_finite(hessian)))
              and bool(tf.reduce_all(eigenvalues > 0.)) and all(c["passed"] for c in checks))
    record = {"passed": passed, "negative_hessian": hessian.numpy().tolist(),
        "eigenvalues": eigenvalues.numpy().tolist(), "score_difference_checks": checks,
        "no_hint_epsilon": original.initial_step_size,
        "local_linear_stability_limit": local_limit,
        "epsilon_over_local_limit": original.initial_step_size / local_limit,
        "stability_interpretation": "local quadratic approximation only; no global guarantee",
        "initialization_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "data_sha256": initial["data_sha256"], "target_signature": initial["target_signature"],
        "reference_draws_used": False, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_intentionally_hidden": True, "jit_compile": False,
        "non_xla_reason": "short independent finite-difference diagnostic; not runtime or performance evidence",
        "command": sys.argv, "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"}
    with (ROOT / "regression-geometry-check.json").open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in record.items() if k != "negative_hessian"}, indent=2))
    if not passed:
        raise ValueError("local-curvature checks failed")


if __name__ == "__main__":
    main()
