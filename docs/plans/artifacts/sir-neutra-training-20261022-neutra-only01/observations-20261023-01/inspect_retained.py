"""CPU-only explanatory inspection of immutable, completed HMC chunks.

This diagnostic never changes a sampler, target, map, or promotion decision.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import hashlib
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

START = time.monotonic()
OUT = pathlib.Path(__file__).resolve().parent
CAMPAIGN = OUT.parent
ROOT = CAMPAIGN.parents[3]
RUN = CAMPAIGN / "019-hmc-neutra"
ARCHIVE = RUN / "public-tuning"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_tensor(base, record, default_dtype="float64"):
    raw = (base / record["path"]).read_bytes()
    if digest(raw) != record["sha256"]:
        raise ValueError(f"checksum mismatch: {record['path']}")
    return tf.io.parse_tensor(raw, tf.as_dtype(record.get("dtype", default_dtype)))


def plain(value):
    return value.numpy().tolist()


def geometry(states):
    flat = tf.reshape(states, [-1, 3])
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(flat)[0] - 1, tf.float64)
    sd = tf.sqrt(tf.linalg.diag_part(covariance))
    correlation = covariance / (sd[:, None] * sd[None, :])
    eigenvalues = tf.linalg.eigvalsh(covariance)
    return dict(mean=plain(mean), standard_deviation=plain(sd),
                covariance=plain(covariance), correlation=plain(correlation),
                covariance_eigenvalues=plain(eigenvalues),
                covariance_condition_number=plain(eigenvalues[-1]/eigenvalues[0]),
                maximum_absolute_off_diagonal_correlation=plain(tf.reduce_max(
                    tf.abs(correlation - tf.eye(3, dtype=tf.float64)))),
                chain_means=plain(tf.reduce_mean(states, axis=0)))


if (OUT / "report.json").exists():
    raise FileExistsError("Preserve prior evidence; use a fresh observation directory")
raw = (ARCHIVE / "fresh-progress.json").read_bytes()
(OUT / "progress-snapshot.json").write_bytes(raw)
progress = json.loads(raw)
nomination_raw = (ARCHIVE / "nomination/manifest.json").read_bytes()
(OUT / "nomination-snapshot.json").write_bytes(nomination_raw)
nomination = json.loads(nomination_raw)["tensors"]
center = read_tensor(ARCHIVE / "nomination", nomination["center"])
factor = read_tensor(ARCHIVE / "nomination", nomination["factor"])
retained = [c for c in progress["chunks"] if c["phase"] == "verification"]
if not retained:
    raise ValueError("No completed retained chunks")
arrays = {}
for key in ("states", "is_accepted", "log_accept_ratio", "delta_h", "divergence", "force_fallback"):
    pieces = [read_tensor(ARCHIVE, c["archive"]["tensors"][key]) for c in retained]
    for chunk, value in zip(retained, pieces):
        tf.debugging.assert_equal(tf.shape(value)[0], chunk["count"])
    arrays[key] = tf.concat(pieces, axis=0)
w = arrays["states"]
tf.debugging.assert_equal(tf.shape(w), [progress["counts"]["verification"], 4, 3])
tf.debugging.assert_all_finite(w, "Nonfinite archived state")
z = center + tf.linalg.matmul(w, factor, transpose_b=True)
# The consumer uses exactly this affine-to-NeuTra coordinate transformation.
tf.debugging.assert_all_finite(z, "Nonfinite NeuTra coordinate")
last = next(e for e in reversed(progress["events"])
            if e["phase"] == "verification" and e.get("rhat"))
state = json.loads((CAMPAIGN / "campaign.json").read_text())
now = time.time()
remaining = {key: max(0., cap - sum(
    a.get("wall_seconds", min(a["cap_seconds"], max(0., now-a["started"])))
    for a in state["attempts"] if a["budget"] == key)) / 3600
    for key, cap in state["budgets_seconds"].items()}
report = dict(schema="sir_neutra_retained_observation_v1", role="explanatory_only",
    status="SAMPLING_INCOMPLETE", timestamp_utc=datetime.now(timezone.utc).isoformat(),
    warmup_excluded=True, counts=progress["counts"],
    checked_coordinate_system="z (after learned map inverse; before tuner affine inverse)",
    transformation="z = center + w @ factor.T; w is the archived HMC coordinate",
    reference="Nominal standard normal N(0,I); covariance alone does not establish Gaussianity",
    neutra_z=geometry(z), additionally_preconditioned_hmc_w=geometry(w),
    acceptance_per_chain=plain(tf.reduce_mean(tf.cast(arrays["is_accepted"], tf.float64), axis=0)),
    mean_acceptance_probability_per_chain=plain(tf.reduce_mean(
        tf.exp(tf.minimum(arrays["log_accept_ratio"], 0.)), axis=0)),
    max_abs_delta_h=plain(tf.reduce_max(tf.abs(arrays["delta_h"]))),
    divergence_count=plain(tf.reduce_sum(tf.cast(arrays["divergence"], tf.int32))),
    force_fallback_count=plain(tf.reduce_sum(tf.cast(arrays["force_fallback"], tf.int32))),
    latest_cumulative_rhat_z=last["rhat"], latest_chunk_checks=last["checks"],
    live_remaining_gpu_hours=remaining,
    source_hashes_match=all(digest((ROOT / name).read_bytes()) == expected
                            for name, expected in state["source_sha256"].items()),
    inference_status=dict(hard_veto="No failed health check in the latest checkpoint",
        statistically_supported_ranking="Not a comparison; no ranking",
        descriptive_only="Acceptance and covariance from incomplete, correlated retained chains",
        default_readiness="Not established", next_evidence="Existing cumulative sampling and terminal posterior diagnostics"),
    decision=dict(action="CONTINUE_EXISTING_FROZEN_KERNEL", primary="R-hat criterion not yet met",
        veto="No new continuation veto", uncertainty="Autocorrelation and incomplete posterior coverage",
        nonclaim="Sufficient whitening, Gaussianity, convergence, or superiority are not established"))
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
manifest = dict(command=[sys.executable, str(pathlib.Path(__file__).resolve())],
    cwd=str(ROOT), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    environment=sys.executable, tensorflow_version=tf.__version__,
    cpu_only=True, gpu_intentionally_hidden=True, cuda_visible_devices="-1",
    jit_compile=False, execution_exception="Small CPU post-run diagnostic; no target or sampler execution",
    plan="docs/plans/sir-neutra-training-execution-2026-10-21.md",
    data_version=dict(progress_sha256=digest(raw), nomination_sha256=digest(nomination_raw)),
    seeds=[c["seed"] for c in retained], wall_seconds=time.monotonic()-START,
    script_sha256=digest(pathlib.Path(__file__).read_bytes()),
    artifacts=[str(OUT / name) for name in ("report.json", "progress-snapshot.json", "nomination-snapshot.json")])
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k:report[k] for k in ("counts", "acceptance_per_chain", "neutra_z", "live_remaining_gpu_hours", "source_hashes_match")}))
