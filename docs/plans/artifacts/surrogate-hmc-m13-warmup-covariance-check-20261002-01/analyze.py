"""CPU-only post-run diagnostic of covariance from archived M13 warmup."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

START = time.monotonic()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
LOG = (OUT / "command.log").open("x", buffering=1)
LOG.write("Command: " + sys.executable + " " + str(Path(__file__).resolve()) + "\n")
LOG.write("CPU-only: CUDA_VISIBLE_DEVICES=-1; saved data only; no new sampling.\n")
os.dup2(LOG.fileno(), 2)
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_m13_cumulative as cumulative
from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.inference import hmc_tensorflow_verified as verified

SOURCE = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-reclassified-20261002-02/report.json"
PLAN = ROOT / "docs/plans/surrogate-hmc-m13-warmup-covariance-check-2026-10-02.md"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def val(tensor):
    return tensor.numpy().tolist()

def empirical_covariance(x):
    flat = tf.reshape(x, [-1, 2])
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flat)[0]-1, tf.float64)
    return mean, covariance

def covariance_summary(covariance):
    tf.debugging.assert_all_finite(covariance, "nonfinite covariance")
    eigenvalues = tf.linalg.eigvalsh(covariance)
    tf.debugging.assert_positive(eigenvalues)
    sd = tf.sqrt(tf.linalg.diag_part(covariance))
    return dict(covariance=val(covariance), sd=val(sd),
        correlation=float((covariance[0, 1]/(sd[0]*sd[1])).numpy()),
        eigenvalues=val(eigenvalues), condition_number=float((eigenvalues[-1]/eigenvalues[0]).numpy()))

def transformed_geometry(samples, center, factor):
    flat = tf.reshape(samples, [-1, 2])
    z = tf.transpose(tf.linalg.triangular_solve(factor, tf.transpose(flat-center), lower=True))
    mean, covariance = empirical_covariance(z)
    return dict(mean=val(mean), **covariance_summary(covariance))

source = read(SOURCE)
tuning_path = Path(source["tuning_manifest"])
assert sha(tuning_path) == source["tuning_manifest_sha256"]
source_master = Path(source["input_report"])
assert sha(source_master) == source["input_report_sha256"]
history = cumulative.load_history(tuning_path, binding_hash=source["binding_hash"],
    resume_report=source_master.parent / "worker/posterior/report.json")
payload = history.payload
nomination_path = tuning_path.parent / "nomination/manifest.json"
assert sha(nomination_path) == payload["nomination_manifest_sha256"]
nomination = read(nomination_path)
config = stage.TensorFlowHMCKernelTuningConfig.from_payload(nomination["config"])
def nomination_tensor(name):
    return stage._read_tensor_record(nomination_path.parent, nomination["tensors"][name], tf.float64)
original_covariance = nomination_tensor("covariance")
original_factor = nomination_tensor("factor")
original_center = nomination_tensor("center")
prior_scales = nomination_tensor("parameter_scales")
prior_covariance = tf.linalg.diag(tf.square(prior_scales))
tf.debugging.assert_near(original_factor @ tf.transpose(original_factor), original_covariance,
    rtol=1e-12, atol=1e-12)
warmup_chunk_states = []
for row in payload["sequence"]["chunks"]:
    if row["phase"] == "warmup":
        chunk = verified._read_chunk(tuning_path.parent, row["archive"]["tensors"])
        warmup_chunk_states.append(chunk.states)
warmup_states = tf.concat(warmup_chunk_states, axis=0)
aggregate = verified._read_chunk(tuning_path.parent, payload["warmup"])
tf.debugging.assert_equal(warmup_states, aggregate.states)
tf.debugging.assert_equal(tf.shape(warmup_states), [9000, 4, 2])
warmup = history.affine.latent_to_position(warmup_states)
tf.debugging.assert_all_finite(warmup, "nonfinite raw warmup")
tf.debugging.assert_near(history.affine.position_to_latent(tf.reshape(warmup, [-1, 2])),
    tf.reshape(warmup_states, [-1, 2]), rtol=1e-12, atol=1e-12)
retained = tf.concat(history.samples, axis=0)
sample_path = Path(source["sample_archive"]["path"])
assert sha(sample_path) == source["sample_archive"]["sha256"]
tf.debugging.assert_equal(retained, tf.io.parse_tensor(tf.io.read_file(str(sample_path)), tf.float64))
tf.debugging.assert_equal(tf.shape(retained), [12750, 4, 2])

# Call the same covariance estimator once to verify the diagnostic formula.
welford = tf.function(stage._welford_covariance,
    input_signature=[tf.TensorSpec([None, 2], tf.float64)], autograph=False, jit_compile=False)
mean_all, covariance_all = empirical_covariance(warmup)
welford_mean, welford_covariance = welford(tf.reshape(warmup, [-1, 2]))
tf.debugging.assert_near(mean_all, welford_mean, rtol=1e-11, atol=1e-11)
tf.debugging.assert_near(covariance_all, welford_covariance, rtol=1e-11, atol=1e-11)

original = dict(center=val(original_center), factor=val(original_factor),
    **covariance_summary(original_covariance))
retained_segments = {"first_9000": retained[:9000], "all_12750": retained}
original["retained_geometry"] = {key: transformed_geometry(x, original_center, original_factor)
    for key, x in retained_segments.items()}
windows = {}
for name, start in [("all_9000", 0), ("last_4500", 4500), ("last_1000", 8000)]:
    x = warmup[start:]
    mean, empirical = empirical_covariance(x)
    covariance, factor, eigenvalues = stage._regularize_covariance(empirical,
        target=prior_covariance, config=config)
    shrink_only = (1-config.mass_shrinkage)*empirical + config.mass_shrinkage*prior_covariance
    tf.debugging.assert_near(covariance, shrink_only, rtol=1e-11, atol=1e-11,
        message="Eigenvalue floor or jitter materially changed the regularized estimate")
    empirical_factor = tf.linalg.cholesky(empirical)
    by_chain = []
    for chain in range(4):
        chain_mean, chain_covariance = empirical_covariance(x[:, chain, :])
        by_chain.append(dict(chain=chain+1, mean=val(chain_mean), **covariance_summary(chain_covariance)))
    windows[name] = dict(start_inclusive=start, end_exclusive=9000,
        draws_per_chain=9000-start, pooled_correlated_draws=4*(9000-start),
        mean=val(mean), empirical=covariance_summary(empirical),
        same_regularizer=dict(factor=val(factor), **covariance_summary(covariance)),
        variance_change_percent_vs_original=val(100*(tf.linalg.diag_part(covariance)/tf.linalg.diag_part(original_covariance)-1)),
        regularization_beyond_shrinkage_max_abs=float(tf.reduce_max(tf.abs(covariance-shrink_only)).numpy()),
        by_chain=by_chain,
        retained_geometry={policy: {key: transformed_geometry(samples, mean, candidate_factor)
            for key, samples in retained_segments.items()}
            for policy, candidate_factor in [("empirical_only", empirical_factor), ("same_regularizer", factor)]})

report = dict(schema="bayesfilter.m13.warmup_covariance_diagnostic.v1",
    role="explanatory_saved_data_only", target_scope=source["target_scope"],
    parameter_order=source["parameter_order"], new_hmc_transitions=0, new_random_draws=0,
    covariance_policy_changed=False, stopping_rule_changed=False,
    posterior_samples_used_to_fit_covariance=False,
    cpu_only=True, gpu_intentionally_hidden=True, jit_compile=False,
    warmup_transitions_per_chain=9000, number_of_chains=4,
    checks=dict(source_hashes=True, history_loader_continuity_and_health=True,
        warmup_aggregate_equals_chunks=True, warmup_shape=True, original_affine_parity=True,
        retained_archive_equality=True, finite_positive_covariances=True,
        covariance_estimator_parity=True, actual_repository_regularizer_used=True),
    covariance_estimator_max_abs_parity_error=float(tf.reduce_max(tf.abs(covariance_all-welford_covariance)).numpy()),
    shrinkage=config.mass_shrinkage, prior_scales=val(prior_scales),
    prior_covariance=val(prior_covariance), original=original, windows=windows,
    interpretation_limits=[
        "These matrices are estimates from the saved fixed-kernel warmup, not actual online adaptation outcomes.",
        "Warmup states are correlated and may be nonstationary; pooled covariance is not a known posterior covariance.",
        "Alternative transformations of saved retained draws do not predict acceptance, ESS, R-hat or sampling speed.",
        "Last-half and last-1000 windows are sensitivity checks, not a selected default.",
        "No mass-matrix, step-size, trajectory or master-program change was made."])
(OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
manifest = dict(schema="bayesfilter.m13.warmup_covariance_manifest.v1",
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    command=[sys.executable, str(Path(__file__).resolve())], environment=sys.executable,
    tensorflow=tf.__version__, cpu_gpu="CPU only; CUDA_VISIBLE_DEVICES=-1 before TensorFlow import",
    jit_compile=False, exception="offline post-run diagnostics",
    data_version=source["target_scope"], random_seeds="N/A: deterministic saved-data analysis; original seeds validated by history loader",
    source_report=str(SOURCE), source_report_sha256=sha(SOURCE),
    tuning_manifest=str(tuning_path), tuning_manifest_sha256=sha(tuning_path),
    nomination_manifest=str(nomination_path), nomination_manifest_sha256=sha(nomination_path),
    analysis_source_sha256=sha(__file__), plan=str(PLAN), plan_sha256=sha(PLAN),
    report=str(OUT / "report.json"), report_sha256=sha(OUT / "report.json"),
    result=str(OUT / "result.md"), command_log=str(OUT / "command.log"),
    wall_seconds=time.monotonic()-START, new_hmc_transitions=0, new_random_draws=0)
(OUT / "run-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
receipt=dict(exit_code=0, wall_seconds=manifest["wall_seconds"], new_hmc_transitions=0,
    new_random_draws=0, all_checks_passed=all(report["checks"].values()), report=str(OUT / "report.json"))
(OUT / "execution-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
LOG.write(json.dumps(receipt)+"\n")
print(json.dumps(receipt))
