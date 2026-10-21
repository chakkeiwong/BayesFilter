"""Read-only CPU diagnostics for the saved M13 affine whitening transform."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import hashlib, json, subprocess, sys, time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bayesfilter.testing import surrogate_hmc_m13_cumulative as cumulative

START = time.monotonic()
SOURCE = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-reclassified-20261002-02/report.json"
PLAN = ROOT / "docs/plans/surrogate-hmc-m13-whitening-check-2026-10-02.md"

def read_json(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if tf.is_tensor(x):
        return x.numpy().tolist()
    if hasattr(x, "item"):
        return x.item()
    return x

def load_tensor(manifest_path, manifest, name, dtype=tf.float64):
    row = manifest["tensors"][name]
    path = manifest_path.parent / row["path"]
    assert sha(path) == row["sha256"], "nomination checksum mismatch: " + name
    return tf.io.parse_tensor(tf.io.read_file(str(path)), dtype)

def lag1(x):
    centered = x - tf.reduce_mean(x, axis=0)
    num = tf.reduce_sum(centered[:-1] * centered[1:], axis=0)
    den = tf.sqrt(tf.reduce_sum(tf.square(centered[:-1]), axis=0) *
                  tf.reduce_sum(tf.square(centered[1:]), axis=0))
    return num / den

def geometry(x):
    flat = tf.reshape(x, [-1, 2])
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(flat)[0] - 1, tf.float64)
    sd = tf.sqrt(tf.linalg.diag_part(cov))
    corr = cov / tf.tensordot(sd, sd, axes=0)
    eig = tf.linalg.eigvalsh(cov)
    return {
        "draw_count": int(tf.shape(flat)[0].numpy()),
        "mean": serial(mean),
        "sd": serial(sd),
        "covariance": serial(cov),
        "correlation": serial(corr),
        "eigenvalues": serial(eig),
        "condition_number": float((eig[-1] / eig[0]).numpy()),
        "max_abs_covariance_minus_identity": float(
            tf.reduce_max(tf.abs(cov - tf.eye(2, dtype=tf.float64))).numpy()),
        "max_abs_sd_minus_one": float(tf.reduce_max(tf.abs(sd - 1.)).numpy()),
        "max_abs_offdiag_correlation": float(tf.abs(corr[0, 1]).numpy()),
        "lag1_by_chain" if x.shape.rank == 3 else "lag1": serial(lag1(x)),
    }

def segment_geometry(x):
    return {
        "pooled": geometry(x),
        **{f"chain_{c + 1}": geometry(x[:, c, :]) for c in range(4)}
    }

report = read_json(SOURCE)
sample_path = Path(report["sample_archive"]["path"])
assert sha(sample_path) == report["sample_archive"]["sha256"]
source_master = Path(report["input_report"])
assert sha(source_master) == report["input_report_sha256"]
tuning_path = Path(report["tuning_manifest"])
assert sha(tuning_path) == report["tuning_manifest_sha256"]
history = cumulative.load_history(
    tuning_path, binding_hash=report["binding_hash"],
    resume_report=source_master.parent / "worker/posterior/report.json")
samples = tf.io.parse_tensor(tf.io.read_file(str(sample_path)), tf.float64)
accepts = tf.concat(history.accepts, axis=0)
tf.debugging.assert_equal(samples, tf.concat(history.samples, axis=0))
tf.debugging.assert_equal(tf.shape(samples), [12750, 4, 2])
tf.debugging.assert_all_finite(samples, "nonfinite sample archive")

nomination_path = tuning_path.parent / "nomination" / "manifest.json"
nomination = read_json(nomination_path)
center = load_tensor(nomination_path, nomination, "center")
factor = load_tensor(nomination_path, nomination, "factor")
covariance = load_tensor(nomination_path, nomination, "covariance")
metric_eigenvalues = load_tensor(nomination_path, nomination, "metric_eigenvalues")
tf.debugging.assert_shapes([(center, (2,)), (factor, (2, 2)),
                            (covariance, (2, 2))])
tf.debugging.assert_near(factor @ tf.transpose(factor), covariance,
                         rtol=1e-11, atol=1e-11)
delta = samples - center
flat_delta = tf.reshape(delta, [-1, 2])
z_t = tf.linalg.triangular_solve(factor, tf.transpose(flat_delta), lower=True)
whitened = tf.reshape(tf.transpose(z_t), tf.shape(samples))
runtime_inverse = history.affine.position_to_latent(tf.reshape(samples, [-1, 2]))
tf.debugging.assert_near(runtime_inverse, tf.reshape(whitened, [-1, 2]),
                         rtol=1e-12, atol=1e-12)
tf.debugging.assert_near(history.affine.latent_to_position(whitened), samples,
                         rtol=1e-12, atol=1e-12)
reconstructed = tf.einsum("ncj,ij->nci", whitened, factor)
tf.debugging.assert_near(reconstructed, delta, rtol=1e-11, atol=1e-11)
tf.debugging.assert_all_finite(whitened, "nonfinite whitened coordinates")

segments = {}
for name, start, stop in [
    ("first_9000", 0, 9000),
    ("additional_3750", 9000, 12750),
    ("all_12750", 0, 12750),
]:
    segments[name] = {
        "start_inclusive": start,
        "end_exclusive": stop,
        "raw": segment_geometry(samples[start:stop]),
        "whitened": segment_geometry(whitened[start:stop]),
        "acceptance_per_chain": serial(tf.reduce_mean(
            tf.cast(accepts[start:stop], tf.float64), axis=0)),
    }

process = samples[:-1, :, 0]
transition_accepts = accepts[1:]
movement = tf.reduce_sum(tf.square(whitened[1:] - whitened[:-1]), axis=-1)
quantiles = tfp.stats.percentile(
    process, [0., 5., 25., 100.], axis=[0, 1],
    interpolation="linear")
region_names = ["lowest_5pct", "5_to_25pct", "upper_75pct"]
regions = {}
for i, name in enumerate(region_names):
    lower, upper = quantiles[i], quantiles[i + 1]
    if i == 0:
        mask = process < upper
    elif i == len(region_names) - 1:
        mask = process >= lower
    else:
        mask = tf.logical_and(process >= lower, process < upper)
    den = tf.reduce_sum(tf.cast(mask, tf.float64), axis=0)
    num = tf.reduce_sum(tf.cast(tf.logical_and(mask, transition_accepts), tf.float64), axis=0)
    rates = serial(tf.math.divide_no_nan(num, den))
    mean_squared_movement = serial(tf.math.divide_no_nan(
        tf.reduce_sum(tf.where(mask, movement, tf.zeros_like(movement)), axis=0), den))
    counts = serial(tf.reduce_sum(tf.cast(mask, tf.int32), axis=0))
    regions[name] = {
        "lower_bound": None if i == 0 else float(lower.numpy()),
        "upper_bound": None if i == len(region_names) - 1 else float(upper.numpy()),
        "transitions_per_chain": counts,
        "acceptance_per_chain": [value if count else None for value, count in zip(rates, counts)],
        "mean_squared_whitened_movement_per_chain": [value if count else None
            for value, count in zip(mean_squared_movement, counts)],
    }

raw_np, z_np = samples.numpy(), whitened.numpy()
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
for c in range(4):
    axes[0, 0].scatter(raw_np[:, c, 0], raw_np[:, c, 1], s=2, alpha=.18,
                       color=colors[c], label=f"Chain {c + 1}")
    axes[0, 1].scatter(z_np[:, c, 0], z_np[:, c, 1], s=2, alpha=.18,
                       color=colors[c])
    axes[1, 0].plot(raw_np[:, c, 0], color=colors[c], lw=.35, alpha=.65)
    axes[1, 1].plot(z_np[:, c, 0], color=colors[c], lw=.35, alpha=.65)
axes[0, 0].set(title="Raw coordinates", xlabel="log process scale",
               ylabel="log observation scale")
axes[0, 1].set(title="Affine-whitened coordinates", xlabel="z[0]", ylabel="z[1]")
axes[1, 0].set(title="Raw process-scale trace", xlabel="draw per chain")
axes[1, 1].set(title="Whitened process trace", xlabel="draw per chain")
for ax in axes.flat:
    ax.grid(alpha=.15)
for ax in axes[1, :]:
    ax.axvline(9000, color="black", ls="--", lw=1)
axes[0, 0].legend(ncol=2, fontsize=8)
fig.suptitle("M13 saved geometry; dashed line is the 9,000-draw stopping point")
fig.savefig(OUT / "raw-vs-whitened.png", dpi=155)
fig.savefig(OUT / "raw-vs-whitened.pdf")
plt.close(fig)

result = {
    "schema": "bayesfilter.m13.saved_whitening_diagnostic.v1",
    "role": "explanatory_post_run_diagnostics_only",
    "new_hmc_transitions": 0, "new_random_draws": 0,
    "stopping_rule_changed": False, "independent_validation_required": False,
    "gpu_intentionally_hidden": True, "jit_compile": False,
    "backend": "TensorFlow/TFP CPU; non-XLA post-run reporting exception",
    "target_scope": report["target_scope"],
    "parameter_order": report["parameter_order"],
    "transform": {
        "theta_equals_center_plus_factor_times_z": True,
        "center": serial(center), "factor": serial(factor),
        "covariance": serial(covariance),
        "metric_eigenvalues": serial(metric_eigenvalues),
        "factor_covariance_max_abs_error": float(
            tf.reduce_max(tf.abs(factor @ tf.transpose(factor) - covariance)).numpy()),
        "forward_inverse_max_abs_error": float(
            tf.reduce_max(tf.abs(reconstructed - delta)).numpy()),
        "inverse_definition": "z = solve_lower_triangular(factor, theta - center)",
        "actual_runtime_adapter_parity_passed": True,
    },
    "frozen_step_size": serial(history.step),
    "num_leapfrog_steps": serial(load_tensor(nomination_path, nomination, "num_leapfrog_steps", tf.int32)),
    "warmup_transitions_per_chain": history.warmup,
    "tuning_geometry_metadata": {
        key: nomination["config"].get(key) for key in [
            "geometry_provenance", "mass_window_results",
            "mass_window_leapfrog_steps", "mass_shrinkage", "covariance_jitter",
            "eigenvalue_floor", "max_condition_number", "target_accept_prob",
            "use_xla"
        ]
    },
    "segments": segments,
    "regional_acceptance": {
        "conditioning": "starting state samples[t-1]; outcome accepts[t]; first transition omitted",
        "quantile_boundaries": serial(quantiles),
        "fixed_full_history_process_scale_regions": regions,
    },
    "interpretation_limits": [
        "Identity covariance is exact only if the frozen factor equals target covariance.",
        "Finite MCMC samples estimate target covariance and do not prove it.",
        "A fixed affine transform removes linear scale and correlation only; it cannot remove nonlinear curvature.",
        "The 128/256 transition covariance windows were short and frozen before the 9,000-draw run.",
        "These are explanatory diagnostics, not new stopping or promotion criteria.",
    ],
}
(OUT / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
manifest = {
    "schema": "bayesfilter.m13.saved_whitening_diagnostic_manifest.v1",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=ROOT, text=True).strip(),
    "command": [sys.executable, str(Path(__file__).resolve())],
    "environment": sys.executable, "tensorflow": tf.__version__,
    "tensorflow_probability": tfp.__version__,
    "cpu_gpu": "CPU only; CUDA_VISIBLE_DEVICES=-1 before TensorFlow import",
    "source_report": str(SOURCE), "source_report_sha256": sha(SOURCE),
    "sample_archive": str(sample_path), "sample_archive_sha256": sha(sample_path),
    "tuning_manifest": str(tuning_path), "tuning_manifest_sha256": sha(tuning_path),
    "nomination_manifest": str(nomination_path),
    "nomination_manifest_sha256": sha(nomination_path),
    "analysis_source_sha256": sha(__file__), "new_hmc_transitions": 0,
    "new_random_draws": 0, "wall_seconds": time.monotonic() - START,
    "plan": str(PLAN), "result": str(OUT / "result.md"),
    "report": str(OUT / "report.json"),
    "plots": [str(OUT / "raw-vs-whitened.png"),
              str(OUT / "raw-vs-whitened.pdf")],
}
(OUT / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"status": "completed", "wall_seconds": manifest["wall_seconds"],
                  "report": str(OUT / "report.json"),
                  "new_hmc_transitions": 0}))

