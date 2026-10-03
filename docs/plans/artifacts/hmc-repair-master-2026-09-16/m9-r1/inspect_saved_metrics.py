"""Local affine-curvature diagnosis at the checked M8 center, not a tuning rule."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    started = time.monotonic()
    import tensorflow as tf
    m8 = ROOT.parent/"m8-r1"
    geometry_path = m8/"regression-geometry-check.json"
    geometry = read(geometry_path)
    assert geometry["passed"] and not geometry["reference_draws_used"]
    hessian = tf.constant(geometry["negative_hessian"], tf.float64)
    names = [ROOT/"regression-pilot-0-gpu-r1",
             *(ROOT/f"regression-combined-fresh-{i}-gpu-r1" for i in range(3)),
             *(m8/f"posteriordb-geometry-hint-regression-fresh-{i}-gpu-r1" for i in range(3))]
    rows = []
    for directory in names:
        path = directory/"tuning/execution_spec.json"
        spec = read(path)["execution"]
        if spec["scope"]["target_signature"] != geometry["target_signature"]:
            raise ValueError("curvature target mismatch")
        factor = tf.eye(6, dtype=tf.float64)
        for layer in spec["layers"]:
            artifact = layer["artifact"]
            if artifact["factor_orientation"] != "row_right_transpose":
                raise ValueError("unexpected affine convention")
            factor = tf.matmul(factor, tf.constant(artifact["factor"], tf.float64))
        curvature = tf.matmul(factor, tf.matmul(hessian, factor), transpose_a=True)
        eigenvalues = tf.linalg.eigvalsh(curvature)
        if not bool(tf.reduce_all(tf.math.is_finite(eigenvalues) & (eigenvalues > 0.))):
            raise ValueError("invalid transformed local curvature")
        progress = directory/("preparation" if "combined-fresh" in directory.name else "tuning")/"preparation_progress.json"
        update_event = next(e for e in read(progress)["events"] if e["phase"] == "windowed_mass_completed")
        rows.append({"run": directory.name, "execution_spec_sha256": sha(path),
            "metric_updates": update_event["details"]["operational_metric_update_count"],
            "composite_factor_identity": bool(tf.reduce_all(factor == tf.eye(6, dtype=tf.float64))),
            "local_curvature_eigenvalues": eigenvalues.numpy().tolist(),
            "local_curvature_condition_number": float(eigenvalues[-1]/eigenvalues[0]),
            "local_frequency_ratio": float(tf.sqrt(eigenvalues[-1]/eigenvalues[0])),
            "mass_signature": spec["scope"]["mass_signature"]})
    result = {"question": "Did preparation change the affine scaling of the local regression geometry?",
        "formula": "theta = center + A q; local final-coordinate negative Hessian = transpose(A) H A",
        "role": "explanatory diagnostic and possible metric-repair trigger; no candidate ranking/admission",
        "hessian_reference": str(geometry_path), "hessian_reference_sha256": sha(geometry_path),
        "rows": rows, "comparison_location": "same checked data/prior-only initial point in all arms",
        "limitation": "Local curvature only; not global posterior geometry or proof of nonconvergence cause.",
        "gpu_intentionally_hidden": True, "jit_compile": False,
        "non_xla_reason": "seven saved six-dimensional matrices; deterministic CPU debugging reference",
        "command": sys.argv, "environment": sys.executable, "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "elapsed_seconds": time.monotonic()-started, "returncode": 0, "device": "cpu_reference",
        "script_sha256": sha(Path(__file__))}
    with (ROOT/"saved-metrics-run.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
