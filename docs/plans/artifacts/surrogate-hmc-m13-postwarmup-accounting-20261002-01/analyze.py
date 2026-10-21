"""CPU-only diagnostic of saved M13 samples; generates no HMC transitions."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
RUN = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-midnight-20261001-01"
PLAN = "docs/plans/surrogate-hmc-m13-postwarmup-accounting-2026-10-02.md"
started = time.monotonic()
import tensorflow as tf
from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.inference import hmc_tensorflow_verified as verified
from bayesfilter.inference.hmc_sequential_verification import combined_chunks
from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_split_rhat, rank_normalized_bulk_tail_ess,
    posterior_mean_diagnostics,
)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(condition, message):
    if not condition:
        raise ValueError(message)

def tensor(root, record, dtype=tf.float64):
    return stage._read_tensor_record(root, record, tf.as_dtype(record.get("dtype", dtype)))

def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if tf.is_tensor(value):
        return value.numpy().tolist()
    return value

def diagnostics(samples, accepts):
    # Static per-count signatures bound tracing to the two requested archives.
    # Existing rank diagnostics deliberately use graph mode without XLA.
    @tf.function(input_signature=[tf.TensorSpec(samples.shape, tf.float64)], autograph=False)
    def calculate(x):
        chain_major = tf.transpose(x, [1, 0, 2])
        return dict(rhat=rank_normalized_split_rhat(chain_major),
                    ess=rank_normalized_bulk_tail_ess(chain_major),
                    means=posterior_mean_diagnostics(chain_major))
    result = serial(calculate(samples))
    result.update(draws_per_chain=int(samples.shape[0]), chains=4,
                  acceptance_per_chain=serial(tf.reduce_mean(tf.cast(accepts, tf.float64), 0)),
                  rhat_at_most_1_01=all(x <= 1.01 for x in result["rhat"]["maximum"]),
                  bulk_and_tail_ess_at_least_400=all(
                      x >= 400 for key in ("bulk", "tail") for x in result["ess"][key]))
    return result

master = read(RUN/"report.json")
w = master["worker"]
require(master["source_unchanged"] and master["execution"]["exit_code"] == 0,
        "invalid final worker execution")
tp = Path(w["tuning_manifest"])
require(sha(tp) == w["tuning_manifest_sha256"], "tuning checksum mismatch")
t = read(tp)
require(t["passed"] and t["handoff_eligible"] and all(t["checks"].values()),
        "unhealthy or failed tuning artifact")
np = tp.parent/"nomination/manifest.json"
require(sha(np) == t["nomination_manifest_sha256"], "nomination checksum mismatch")
nomination = read(np)
require(nomination["runner_binding"] == t["runner_binding"], "nomination binding changed")
affine = stage._TensorFlowAffineAdapter(
    stage._TensorFlowAffineTransform(
        tensor(np.parent, nomination["tensors"]["center"]),
        tensor(np.parent, nomination["tensors"]["factor"])),
    t["adapter_signature"], t["config"]["nomination"]["target_scope"])
step = tensor(np.parent, nomination["tensors"]["step_size"])
state = affine.position_to_latent(tf.constant(t["config"]["initial_chain_positions"], tf.float64))
used = []
phases = {"warmup": [], "verification": []}
counts = {"warmup": 0, "verification": 0}
for rec, event in zip(t["sequence"]["chunks"], t["sequence"]["events"], strict=True):
    phase = rec["phase"]
    require(rec["index"] == len(phases[phase]), "chunk order mismatch")
    require(rec["seed"] not in used, "verification seed reuse")
    expected_seed = tf.random.experimental.stateless_fold_in(
        tf.constant(t["config"]["fresh_seed"], tf.int32),
        (0 if phase == "warmup" else 100000) + rec["index"])
    require(rec["seed"] == serial(expected_seed), "verification seed schedule mismatch")
    initial = tensor(tp.parent, rec["archive"]["initial"])
    tf.debugging.assert_equal(initial, state, message="sequence state handoff")
    chunk = verified._read_chunk(tp.parent, rec["archive"]["tensors"])
    require(all(bool(v) for v in verified._health(chunk, state, t["config"]["energy_error_max"]).values()),
            "sequence health veto")
    tf.debugging.assert_equal(chunk.final_state, chunk.states[-1])
    tf.debugging.assert_equal(chunk.final_step_size, step, message="unfrozen step size")
    tf.debugging.assert_equal(tf.shape(chunk.states), [rec["count"], 4, 2])
    used.append(rec["seed"])
    phases[phase].append(chunk)
    counts[phase] += rec["count"]
    require(event["phase"] == phase and event["count"] == counts[phase]
            and all(event["checks"].values()), "sequence event mismatch")
    state = chunk.final_state
require(counts == t["sequence"]["counts"], "sequence counts mismatch")
require(used == t["sequence"]["used_seeds"], "sequence seed history mismatch")
aggregate = combined_chunks(phases["verification"])
archived = verified._read_chunk(tp.parent, t["verification"])
for name in aggregate._fields:
    tf.debugging.assert_equal(getattr(aggregate, name), getattr(archived, name),
                              message="aggregate mismatch: "+name)
verification_raw = affine.latent_to_position(aggregate.states)
raw_chunks = [verification_raw]
accept_chunks = [aggregate.is_accepted]
parent = None
used = [list(t["config"]["nomination"]["seed"]), list(t["config"]["fresh_seed"])] + used
for i, rec in enumerate(w["posterior"]["chunks"]):
    mp = Path(rec["manifest"])
    require(sha(mp) == rec["sha256"], "retained manifest checksum mismatch")
    m = read(mp)
    require(m["tuning_manifest_sha256"] == sha(tp) and m["binding_hash"] == w["binding_hash"],
            "retained frozen target/kernel binding changed")
    require(m["continuation_manifest"] == parent, "retained parent mismatch")
    seed = m["config"]["seed"]
    require(seed not in used, "retained seed reuse")
    require(m["used_seeds"] == used+[seed], "retained seed history mismatch")
    used.append(seed)
    initial = tensor(mp.parent, m["initial_state"])
    tf.debugging.assert_equal(initial, state, message="retained state handoff")
    chunk = verified._read_chunk(mp.parent, m["chunk"])
    health = {k: bool(v) for k, v in verified._health(chunk, state, t["config"]["energy_error_max"]).items()}
    require(all(health.values()) and m["health_passed"] and m["checks"] == health,
            "retained health veto")
    tf.debugging.assert_equal(chunk.final_state, chunk.states[-1])
    tf.debugging.assert_equal(chunk.final_step_size, step, message="retained step changed")
    raw = tensor(mp.parent, m["raw_samples"])
    tf.debugging.assert_equal(tf.shape(raw), [m["config"]["num_results"], 4, 2])
    # Different XLA/non-XLA rounding may change the last few bits of the affine transform.
    tf.debugging.assert_near(raw, affine.latent_to_position(chunk.states),
                            rtol=1e-12, atol=1e-12, message="raw-coordinate transform mismatch")
    raw_chunks.append(raw)
    accept_chunks.append(chunk.is_accepted)
    state = chunk.final_state
    parent = str(mp)
all_raw = tf.concat(raw_chunks, 0)
all_accept = tf.concat(accept_chunks, 0)
require(int(all_raw.shape[0]) == w["post_warmup_draws_per_chain"], "combined count mismatch")
require(int(all_raw.shape[0])-counts["verification"] == w["retained_draws_per_chain"], "suffix count mismatch")
print("Archive checks passed; computing 9000 and 12750 draw diagnostics.", flush=True)
prefix = diagnostics(verification_raw, aggregate.is_accepted)
saved_rhat = w["final_verification_rhat"]["maximum"]
tf.debugging.assert_near(tf.constant(prefix["rhat"]["maximum"], tf.float64),
                        tf.constant(saved_rhat, tf.float64), rtol=1e-10, atol=1e-10)
pooled = diagnostics(all_raw, all_accept)
report = dict(
    schema="bayesfilter.m13.postwarmup_accounting_diagnostic.v1",
    purpose="Correct exclusion of eligible fixed-kernel post-warmup samples",
    plan=PLAN, input_report=str(RUN/"report.json"), input_report_sha256=sha(RUN/"report.json"),
    tuning_manifest=str(tp), tuning_manifest_sha256=sha(tp),
    parameter_order=["log_process_scale", "log_observation_scale"],
    sample_layout="draw,chain,parameter", warmup_excluded_per_chain=counts["warmup"],
    archive_checks_passed=True, same_frozen_kernel=True, state_and_seed_continuity=True,
    saved_verification_rhat_reproduced=True, new_hmc_transitions=0,
    old_phase_names_do_not_determine_sample_eligibility=True,
    verification_prefix=prefix, all_postwarmup=pooled,
    target_scope=w["target_scope"], endpoint_role=w["posterior"]["endpoint_role"],
    exact_likelihood_claim=False, calibrated_equivalence_assessed=False,
    posterior_promotion_claim=False, ranking_supported=False, all_models_complete=False,
    gpu_intentionally_hidden=True, backend="TensorFlow CPU saved-sample diagnostic",
    jit_compile=False, jit_exception="Existing rank diagnostics use data-dependent segment counts; CPU diagnostic only",
)
samples_path = OUT/"all-postwarmup-samples.tfbin"
require(not samples_path.exists() and not (OUT/"report.json").exists(), "refuse artifact overwrite")
tf.io.write_file(str(samples_path), tf.io.serialize_tensor(all_raw))
report["pooled_sample_archive"] = dict(path=str(samples_path), sha256=sha(samples_path),
                                      dtype="float64", shape=serial(tf.shape(all_raw)))
report["wall_seconds"] = time.monotonic()-started
report["cpu_seconds"] = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime
(OUT/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
manifest = dict(
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    command=[sys.executable, str(Path(__file__).resolve())], script_sha256=sha(__file__),
    python=sys.version, tensorflow=tf.__version__, conda_environment=sys.prefix,
    cpu_only=True, gpu_intentionally_hidden=True, new_hmc_transitions=0,
    data_version=dict(seed=w["data_seed"], observation_checksum=w["observation_checksum"]),
    seeds=dict(verification=t["config"]["fresh_seed"], retained=w["retained_seed"]),
    wall_seconds=report["wall_seconds"], cpu_seconds=report["cpu_seconds"],
    plan=PLAN, result=str(OUT/"report.json"), samples=str(samples_path),
)
(OUT/"run-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
for name, item in (("verification_prefix", prefix), ("all_postwarmup", pooled)):
    print(json.dumps(dict(name=name, draws=item["draws_per_chain"],
          rhat=item["rhat"]["maximum"], ess=item["ess"],
          mean=item["means"]["pooled_mean"], mean_mcse=item["means"]["mean_mcse"],
          mcse_sd_ratio=item["means"]["mcse_sd_ratio"],
          acceptance=item["acceptance_per_chain"])), flush=True)
