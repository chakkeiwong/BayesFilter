"""One cumulative post-warmup sampling controller for the M13 master.

The tuner's historical ``verification`` label is provenance, not an exclusion
rule. Its fixed-kernel samples and any saved continuations form one posterior
history. Runtime and archive reclassification use the same loader and decision.
The public tuner remains the only tuning-artifact authority. Rank diagnostics
use the existing graph-mode, non-XLA exception; new HMC uses the public XLA runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.inference import hmc_tensorflow_verified as verified
from bayesfilter.inference.hmc_sequential_verification import combined_chunks
from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_split_rhat, rank_normalized_bulk_tail_ess, posterior_mean_diagnostics,
)
from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json

PLAN = "docs/plans/surrogate-hmc-cumulative-master-repair-2026-10-02.md"
SCHEMA = "bayesfilter.m13.cumulative_posterior.v1"
MAXIMUM = 20000
CHUNK = 250
CHECK_EVERY = 1000
RHAT_MAX = 1.01
SEED = (20260926, 2910)
COMPLETE = "M13_POSTERIOR_SAMPLING_COMPLETE"


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor(root, record):
    return stage._read_tensor_record(root, record, tf.as_dtype(record.get("dtype", "float64")))


def serial(value):
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    return value.numpy().tolist() if tf.is_tensor(value) else value


@lru_cache(maxsize=4)
def _diagnostic_graph(count):
    @tf.function(input_signature=[tf.TensorSpec([count, 4, 2], tf.float64)],
                 jit_compile=False, autograph=False)
    def calculate(samples):
        points = tf.transpose(samples, [1, 0, 2])
        return dict(rhat=rank_normalized_split_rhat(points),
                    ess=rank_normalized_bulk_tail_ess(points),
                    means=posterior_mean_diagnostics(points))
    return calculate


def diagnostics(samples, accepts):
    """Describe all supplied draws; no thinning, truncation or prefix selection."""
    count = int(samples.shape[0])
    tf.debugging.assert_equal(tf.shape(samples), [count, 4, 2])
    tf.debugging.assert_equal(tf.shape(accepts), [count, 4])
    tf.debugging.assert_all_finite(samples, "nonfinite posterior sample")
    result = serial(_diagnostic_graph(count)(samples))
    result.update(draws_per_chain=count, chains=4,
        acceptance_per_chain=serial(tf.reduce_mean(tf.cast(accepts, tf.float64), 0)),
        rhat_at_most_1_01=all(math.isfinite(x) and x <= RHAT_MAX
                            for x in result["rhat"]["maximum"]),
        bulk_and_tail_ess_at_least_400=all(math.isfinite(x) and x >= 400
                            for key in ("bulk", "tail") for x in result["ess"][key]))
    return result


def decision(metrics, *, minimum, maximum=MAXIMUM, health_passed=True):
    """R-hat decides sampling completion; precision metrics do not add a phase."""
    count = metrics["draws_per_chain"]
    require(type(maximum) is int and 0 < maximum <= MAXIMUM and maximum % CHUNK == 0,
            "invalid cumulative post-warmup maximum")
    require(0 < count <= maximum, "post-warmup count exceeds total cap")
    passed = (health_passed and count >= minimum and
              all(math.isfinite(x) and x <= RHAT_MAX for x in metrics["rhat"]["maximum"]))
    status = (COMPLETE if passed else "M13_POSTERIOR_HEALTH_VETO" if not health_passed
              else "M13_POSTERIOR_SAMPLE_CAP_INCONCLUSIVE" if count == maximum
              else "M13_POSTERIOR_MIXING_INCONCLUSIVE")
    return dict(status=status, qualified=passed, sampling_qualified=passed,
        qualification_scope="healthy cumulative post-warmup samples meet the R-hat stopping rule",
        posterior_reference_qualified=False, all_models_complete=False,
        warnings=[] if metrics["bulk_and_tail_ess_at_least_400"] else
            ["bulk_or_tail_ess_below_400_precision_warning_only"],
        independent_validation_required=False, continuation_veto=not health_passed)


@dataclass
class History:
    manifest: Path
    payload: dict
    affine: object
    step: object
    state: object
    used_seeds: list
    warmup: int
    prefix_count: int
    samples: list
    accepts: list
    probabilities: list
    max_energy: float
    initial_affine_error: float
    rows: list = field(default_factory=list)

    @property
    def count(self):
        return sum(int(x.shape[0]) for x in self.samples)

    @property
    def binding_hash(self):
        return self.payload["runner_binding"]["binding_hash"]

    def summarize(self):
        return diagnostics(tf.concat(self.samples, 0), tf.concat(self.accepts, 0))


def _check_chunk(chunk, initial, step, count, energy_limit):
    checks = {k: bool(v) for k, v in verified._health(chunk, initial, energy_limit).items()}
    require(all(checks.values()), "post-warmup archive health veto")
    tf.debugging.assert_equal(tf.shape(chunk.states), [count, 4, 2])
    tf.debugging.assert_equal(chunk.final_state, chunk.states[-1])
    tf.debugging.assert_equal(chunk.final_step_size, step, message="kernel step size changed")
    return checks


def append_archive(history, row):
    """Check an actual continuation handoff before adding its samples."""
    path = Path(row["manifest"]).resolve()
    require(file_hash(path) == row["sha256"], "continuation manifest checksum mismatch")
    saved = read(path)
    seed = [SEED[0], SEED[1] + len(history.rows)]
    parent = str(Path(history.rows[-1]["manifest"]).resolve()) if history.rows else None
    require(saved.get("schema") == verified.VERIFIED_TENSORFLOW_ARCHIVE_SCHEMA
        and saved["tuning_manifest_sha256"] == file_hash(history.manifest)
        and Path(saved["tuning_manifest"]).resolve() == history.manifest
        and saved["binding_hash"] == history.binding_hash
        and saved["continuation_manifest"] == parent
        and saved["config"]["num_results"] == CHUNK
        and saved["config"]["seed"] == seed and seed not in history.used_seeds
        and saved["used_seeds"] == history.used_seeds + [seed],
        "continuation target, kernel, parent or seed mismatch")
    initial = tensor(path.parent, saved["initial_state"])
    tf.debugging.assert_equal(initial, history.state, message="continuation state discontinuity")
    chunk = verified._read_chunk(path.parent, saved["chunk"])
    checks = _check_chunk(chunk, initial, history.step, CHUNK,
                          history.payload["config"]["energy_error_max"])
    require(saved.get("health_passed") is True and row.get("health_passed") is True
        and row.get("no_force_fallback") is True and checks == saved["checks"]
        and checks == row["checks"], "continuation health record mismatch")
    raw = tensor(path.parent, saved["raw_samples"])
    tf.debugging.assert_equal(tf.shape(raw), [CHUNK, 4, 2])
    # This reconstruction may run on CPU after a GPU/XLA run. Saved handoffs
    # above remain bit-exact; only recomputing the affine map allows rounding.
    tf.debugging.assert_near(raw, history.affine.latent_to_position(chunk.states),
                            rtol=1e-12, atol=1e-12, message="raw transform disagreement")
    history.samples.append(raw)
    history.accepts.append(chunk.is_accepted)
    history.probabilities.append(tf.exp(tf.minimum(tf.constant(0., tf.float64), chunk.log_accept_ratio)))
    history.max_energy = max(history.max_energy, float(tf.reduce_max(tf.abs(chunk.delta_h))))
    history.state = chunk.final_state
    history.used_seeds.append(seed)
    history.rows.append(dict(row, manifest=str(path)))


def load_history(manifest, *, binding_hash=None, resume_report=None):
    """Reconstruct the warmup boundary, then load every eligible saved draw."""
    path = Path(manifest).resolve()
    payload = read(path)
    require(payload.get("schema") == verified.VERIFIED_TENSORFLOW_HMC_SCHEMA
        and payload.get("artifact_authority") is True
        and payload["sequence"]["warmup_ready"] is True
        and all(value for key, value in payload["checks"].items() if key != "fresh_rhat"),
        "missing public tuning artifact or completed warmup")
    if binding_hash is not None:
        require(payload["runner_binding"]["binding_hash"] == binding_hash, "tuning binding mismatch")
    nomination_path = path.parent / "nomination/manifest.json"
    require(file_hash(nomination_path) == payload["nomination_manifest_sha256"], "nomination checksum mismatch")
    nomination = read(nomination_path)
    require(nomination["runner_binding"] == payload["runner_binding"], "nomination binding mismatch")
    affine = stage._TensorFlowAffineAdapter(stage._TensorFlowAffineTransform(
        tensor(nomination_path.parent, nomination["tensors"]["center"]),
        tensor(nomination_path.parent, nomination["tensors"]["factor"])),
        payload["adapter_signature"], payload["config"]["nomination"]["target_scope"])
    step = tensor(nomination_path.parent, nomination["tensors"]["step_size"])
    state = affine.position_to_latent(tf.constant(payload["config"]["initial_chain_positions"], tf.float64))
    used, phases = [], {"warmup": [], "verification": []}
    counts = dict(warmup=0, verification=0)
    initial_error = 0.
    for row, event in zip(payload["sequence"]["chunks"], payload["sequence"]["events"], strict=True):
        phase = row["phase"]
        require(phase in phases and not (phase == "warmup" and phases["verification"]),
                "warmup must precede every posterior sample")
        require(row["index"] == len(phases[phase]) and row["seed"] not in used,
                "sequence index or seed mismatch")
        expected_seed = tf.random.experimental.stateless_fold_in(
            tf.constant(payload["config"]["fresh_seed"], tf.int32),
            (0 if phase == "warmup" else 100000) + row["index"])
        require(row["seed"] == serial(expected_seed), "sequence seed schedule mismatch")
        initial = tensor(path.parent, row["archive"]["initial"])
        if not used:
            initial_error = float(tf.reduce_max(tf.abs(initial-state)))
            tf.debugging.assert_near(initial, state, rtol=7.2e-15, atol=7.2e-15,
                                     message="initial affine transform disagreement")
        else:
            tf.debugging.assert_equal(initial, state, message="sequence state discontinuity")
        chunk = verified._read_chunk(path.parent, row["archive"]["tensors"])
        checks = _check_chunk(chunk, initial, step, row["count"], payload["config"]["energy_error_max"])
        counts[phase] += row["count"]
        require(event["phase"] == phase and event["index"] == row["index"]
                and event["count"] == counts[phase]
                and event["checks"] == dict(checks, final_state_matches=True),
                "sequence health event mismatch")
        phases[phase].append(chunk)
        used.append(row["seed"])
        state = chunk.final_state
    require(counts == payload["sequence"]["counts"] and used == payload["sequence"]["used_seeds"],
            "sequence count or seed history mismatch")
    require(bool(phases["verification"]), "no post-warmup samples available")
    prefix = combined_chunks(phases["verification"])
    aggregate = verified._read_chunk(path.parent, payload["verification"])
    for name in prefix._fields:
        tf.debugging.assert_equal(getattr(prefix, name), getattr(aggregate, name), message="aggregate mismatch: "+name)
    history = History(path, payload, affine, step, state,
        [list(payload["config"]["nomination"]["seed"]), list(payload["config"]["fresh_seed"])] + used,
        counts["warmup"], counts["verification"], [affine.latent_to_position(prefix.states)],
        [prefix.is_accepted], [tf.exp(tf.minimum(tf.constant(0., tf.float64), prefix.log_accept_ratio))],
        float(tf.reduce_max(tf.abs(prefix.delta_h))), initial_error)
    if resume_report is not None:
        old = read(resume_report)
        require(Path(old["tuning_manifest"]).resolve() == path and old["binding_hash"] == history.binding_hash
            and old["warmup_excluded"] is True and old["seed"] == list(SEED),
            "saved posterior report identity mismatch")
        for row in old["chunks"]:
            append_archive(history, row)
        expected = history.count if old.get("schema") == SCHEMA else history.count-history.prefix_count
        require(old["draws_per_chain"] == expected, "saved posterior count mismatch")
    require(history.count <= MAXIMUM, "saved samples exceed the owner-approved total cap")
    return history


def run_history(history, output_dir, *, tuning=None, binding=None, maximum=MAXIMUM, allow_sampling=True):
    """Assess the entire saved history before even constructing a chain runner."""
    minimum = history.payload["config"]["verification_results"]
    metrics = history.summarize()
    verdict = decision(metrics, minimum=minimum, maximum=maximum)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original_count = history.count
    report = dict(schema=SCHEMA, plan=PLAN, **verdict,
        tuning_manifest=str(history.manifest), tuning_manifest_sha256=file_hash(history.manifest),
        binding_hash=history.binding_hash, target_scope=history.payload["config"]["nomination"]["target_scope"],
        parameter_order=["log_process_scale", "log_observation_scale"], warmup_excluded=True,
        warmup_excluded_per_chain=history.warmup, tuner_postwarmup_draws_per_chain=history.prefix_count,
        seed=list(SEED), chunk_size=CHUNK, total_postwarmup_maximum=maximum,
        minimum_draws_per_chain=minimum, check_every_results=CHECK_EVERY,
        rhat_threshold=RHAT_MAX, archive_checks_passed=True, same_frozen_kernel=True,
        state_and_seed_continuity=True, initial_affine_cpu_gpu_max_abs_error=history.initial_affine_error,
        exact_likelihood_claim=False, ranking_supported=False, calibrated_equivalence_assessed=False,
        reference_assessment_status="not_recalibrated_for_cumulative_samples",
        endpoint_role="separately frozen full-support importance potential for MH",
        force_semantics="deterministic position-only analytical DCTR proposal field",
        assessments=[metrics], chunks=history.rows, numerical_backend="tensorflow_only",
        diagnostic_jit_compile=False, diagnostic_exception="existing rank diagnostic non-XLA graph",
        new_hmc_jit_compile=True)

    def save():
        report.update(draws_per_chain=history.count, post_warmup_draws_per_chain=history.count,
            continuation_draws_per_chain=history.count-history.prefix_count,
            new_draws_per_chain=history.count-original_count, new_hmc_transitions=4*(history.count-original_count),
            diagnostics=metrics, acceptance_per_chain=serial(tf.reduce_mean(tf.cast(tf.concat(history.accepts, 0), tf.float64), 0)),
            acceptance_probability_per_chain=serial(tf.reduce_mean(tf.concat(history.probabilities, 0), 0)),
            maximum_absolute_energy_error=history.max_energy, wall_seconds=time.monotonic()-started)
        next_plan = dict(part_of_existing_master=True, phase="posterior_sampling", status=report["status"],
            next_look=min(maximum, (history.count//CHECK_EVERY+1)*CHECK_EVERY),
            maximum_per_chain=maximum, independent_validation_required=False,
            action=("report_completed_sampling_without_another_chain" if report["sampling_qualified"] else
                    "continue_same_cumulative_history_within_authorized_budget" if report["status"] == "M13_POSTERIOR_RUNNING" else
                    "preserve_result_and_refresh_remaining_campaign_plan"))
        report["next_phase_plan"] = next_plan
        write_json(out/"report.json", report)
        write_json(out/"next-phase-plan.json", next_plan)

    save()
    if report["sampling_qualified"] or history.count == maximum or not allow_sampling:
        return report
    require(tuning is not None and binding is not None and tuning.passed,
            "continuation requires the public tuner's eligible frozen kernel")
    from bayesfilter.inference import BoundRetainedHMCArchiveConfig, build_retained_bound_hmc_archive_runner_from_tuning_result
    from bayesfilter.inference.hmc_sequential_verification import runtime_budget_guard
    permit = runtime_budget_guard()
    runner = None
    while history.count < maximum:
        if permit is not None and not permit("retained", CHUNK):
            report.update(status="M13_POSTERIOR_BUDGET_EXHAUSTED")
            break
        if runner is None:
            runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=tuning, runner_binding=binding)
        index = len(history.rows)
        config = BoundRetainedHMCArchiveConfig(num_results=CHUNK, seed=(SEED[0], SEED[1]+index),
            output_dir=out/f"chunk-{index:02}", budget_provenance="single cumulative post-warmup allowance",
            continuation_manifest=Path(history.rows[-1]["manifest"]) if history.rows else None)
        result = runner.run(config)
        path = Path(result["archive_manifest_path"])
        saved = read(path)
        row = dict(manifest=str(path), sha256=file_hash(path), health_passed=result["health_passed"],
            no_force_fallback=saved["checks"]["no_force_fallback"], checks=saved["checks"],
            graph_trace_count=saved["chunk_graph_trace_count"])
        try:
            append_archive(history, row)
        except (ValueError, tf.errors.InvalidArgumentError) as error:
            report.update(status="M13_POSTERIOR_HEALTH_VETO", continuation_veto=True,
                          failed_chunk=row, failure=str(error))
            save()
            raise
        report.update(status="M13_POSTERIOR_RUNNING")
        if history.count % CHECK_EVERY == 0 or history.count == maximum:
            metrics = history.summarize()
            report["assessments"].append(metrics)
            report.update(decision(metrics, minimum=minimum, maximum=maximum))
            if not report["sampling_qualified"] and history.count < maximum:
                report["status"] = "M13_POSTERIOR_RUNNING"
        save()
        if report["sampling_qualified"]:
            return report
    # Assess any partial final checkpoint as well; no eligible draw disappears.
    if metrics["draws_per_chain"] != history.count:
        metrics = history.summarize()
        report["assessments"].append(metrics)
        reason = report["status"]
        report.update(decision(metrics, minimum=minimum, maximum=maximum))
        if not report["sampling_qualified"] and reason == "M13_POSTERIOR_BUDGET_EXHAUSTED":
            report["status"] = reason
    save()
    return report


def consume_chunks(tuning, binding, output_dir, *, max_results=MAXIMUM, resume_report=None, allow_sampling=True):
    history = load_history(tuning.artifact_manifest_path, binding_hash=binding.binding_hash, resume_report=resume_report)
    return run_history(history, output_dir, tuning=tuning, binding=binding,
                       maximum=max_results, allow_sampling=allow_sampling)


def reclassify_saved_run(run_root, output_dir):
    """Archive-only entry point: no target evaluation, tuning or HMC launch."""
    import os
    import resource
    import subprocess
    import sys
    from bayesfilter.inference.surrogate_hmc_program import ROOT
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "archive reclassification must intentionally hide GPU")
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    run_root, out = Path(run_root).resolve(), Path(output_dir).resolve()
    master = read(run_root/"report.json")
    execution = master.get("execution", {})
    require(master.get("source_unchanged") is True and execution.get("exit_code") == 0
        and execution.get("status") == "COMPLETED" and execution.get("accounting_complete") is True,
        "invalid saved master execution; cannot reclassify as a healthy posterior")
    original_manifest = read(run_root/"run-manifest.json")
    for name, digest in original_manifest["source_hashes"].items():
        require(file_hash(run_root/"source"/name) == digest, "saved source snapshot changed: "+name)
    worker = master["worker"]
    tuning_path = Path(worker["tuning_manifest"])
    require(file_hash(tuning_path) == worker["tuning_manifest_sha256"], "saved tuning checksum mismatch")
    resume_report = run_root/"worker/posterior/report.json" if worker.get("posterior") is not None else None
    if resume_report is not None:
        require(read(resume_report) == worker["posterior"], "master/posterior report mismatch")
    history = load_history(tuning_path, binding_hash=worker["binding_hash"], resume_report=resume_report)
    require(history.count == worker["post_warmup_draws_per_chain"], "master cumulative count mismatch")
    prefix = diagnostics(history.samples[0], history.accepts[0])
    saved_rhat = worker["final_verification_rhat"]["maximum"]
    tf.debugging.assert_near(tf.constant(prefix["rhat"]["maximum"], tf.float64),
        tf.constant(saved_rhat, tf.float64), rtol=1e-10, atol=1e-10, message="saved prefix R-hat not reproduced")
    report = run_history(history, out, allow_sampling=False)
    report.update(input_report=str(run_root/"report.json"), input_report_sha256=file_hash(run_root/"report.json"),
        original_status=master["status"], original_source_snapshots_checked=True,
        historical_prefix=prefix, historical_prefix_decision=decision(prefix,
            minimum=history.payload["config"]["verification_results"]),
        earlier_passing_prefix_does_not_override_later_samples=True,
        saved_prefix_rhat_reproduced=True, gpu_intentionally_hidden=True,
        data_seed=worker["data_seed"], observation_checksum=worker["observation_checksum"],
        sampling_deadline=original_manifest["deadline"], new_sampling_authorized=False)
    report["next_phase_plan"].update(
        automatic_continuation=False, new_sampling_authorized=False,
        sampling_deadline=original_manifest["deadline"],
        remaining_to_total_cap_per_chain=MAXIMUM-history.count,
        reason="archive reclassification only; prior sampling allocation has expired",
        action=("report_completed_sampling_without_another_chain" if report["sampling_qualified"] else
                "resume_same_saved_chains_only_with_a_new_time_allocation"))
    write_json(out/"next-phase-plan.json", report["next_phase_plan"])
    samples_path = out/"all-postwarmup-samples.tfbin"
    tf.io.write_file(str(samples_path), tf.io.serialize_tensor(tf.concat(history.samples, 0)))
    report["sample_archive"] = dict(path=str(samples_path), sha256=file_hash(samples_path), layout="draw,chain,parameter")
    after = resource.getrusage(resource.RUSAGE_SELF)
    report.update(wall_seconds=time.monotonic()-started,
                  cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    write_json(out/"report.json", report)
    sources = ["bayesfilter/testing/surrogate_hmc_m13_cumulative.py",
        "bayesfilter/inference/surrogate_hmc_deadline_extension.py",
        "bayesfilter/inference/surrogate_hmc_resume_program.py",
        "bayesfilter/inference/surrogate_hmc_recovery_program.py",
        "bayesfilter/testing/surrogate_hmc_m13_retained.py",
        "bayesfilter/inference/hmc_posterior_diagnostics.py"]
    manifest = dict(command=[sys.executable, *sys.argv], cwd=str(ROOT),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=sys.executable, tensorflow=tf.__version__, device="CPU; GPU intentionally hidden",
        source_hashes={name:file_hash(ROOT/name) for name in sources},
        source_run_manifest=str(run_root/"run-manifest.json"),
        source_run_manifest_sha256=file_hash(run_root/"run-manifest.json"),
        data=dict(seed=report["data_seed"], observation_checksum=report["observation_checksum"]),
        seeds=dict(fresh_sequence=history.payload["config"]["fresh_seed"], continuation=list(SEED),
                   complete_used_seed_history=history.used_seeds),
        wall_seconds=report["wall_seconds"], cpu_seconds=report["cpu_seconds"],
        new_hmc_transitions=0, plan=PLAN, result=str(out/"result.md"), report=str(out/"report.json"),
        samples=str(samples_path), jit_compile=False, jit_exception=report["diagnostic_exception"])
    for name in sources:
        destination = out/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    write_json(out/"run-manifest.json", manifest)
    _write_result(out, report)
    return report


def _write_result(out, report):
    prefix, full = report["historical_prefix"], report["diagnostics"]
    def pair(values, precision=4):
        return " / ".join(f"{x:.{precision}f}" for x in values)
    rows = []
    for label, item in (("Earlier stopping point", prefix), ("Complete saved history", full)):
        rows.append(f"| {label} | {item['draws_per_chain']:,} | {pair(item['rhat']['maximum'])} | "
                    f"{pair(item['ess']['bulk'], 0)} | {pair(item['ess']['tail'], 0)} | "
                    f"{pair(item['means']['pooled_mean'])} | {pair(item['means']['mean_mcse'])} |")
    text = f"""# M13 surrogate-HMC: corrected cumulative posterior result

The {prefix['draws_per_chain']:,} fixed-kernel draws formerly labeled verification
are posterior samples. Prefix status: {report['historical_prefix_decision']['status']}.
There is no separate validation sample requirement. Warmup is excluded.

The old master generated another {report['continuation_draws_per_chain']:,} draws per
chain. All {full['draws_per_chain']:,} saved post-warmup draws are now included.
Complete-history status: {report['status']}.
An earlier passing prefix cannot override adverse evidence from later samples.

Parameter order below is log process scale / log observation scale; four chains.

| Samples | Draws/chain | R-hat | Bulk ESS | Tail ESS | Posterior mean | Mean MCSE |
|---|---:|---|---|---|---|---|
{chr(10).join(rows)}

Acceptance by chain: earlier {pair(prefix['acceptance_per_chain'])}; complete
{pair(full['acceptance_per_chain'])}. Acceptance is descriptive.

All archived checksums, fixed-step and binding identities, seed histories,
exact saved state handoffs and health checks passed. No force fallback or
divergence was found. CPU reconstruction allows rounding only in the affine
map; archived handoffs remain exact. The earlier R-hat was reproduced.
No new HMC transitions were generated.

| Decision | Primary criterion | Veto checks | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Earlier prefix: {report['historical_prefix_decision']['status']} | R-hat {pair(prefix['rhat']['maximum'])} | Health passed | Later history may reveal slow exploration | Preserve the earlier stopping point | Permanent convergence |
| Complete archive: {report['status']} | R-hat {pair(full['rhat']['maximum'])} | Health passed | Slow exploration and MCSE | Preserve every sample; follow the refreshed plan | Failure of surrogate HMC as a method |

| Inference status | Interpretation |
|---|---|
| Hard veto screen | No numerical, target-status, seed or state-continuity veto found |
| Statistically supported ranking | None; this reclassification compares no methods |
| Descriptive-only differences | ESS, MCSE, means and acceptance; low ESS is a precision warning |
| Default-readiness | Not evaluated; the all-model campaign remains incomplete |
| Next evidence needed | Stable cumulative mixing; reference agreement is a separate scientific assessment, not an independent validation chain |

The target remains the frozen importance-likelihood posterior for
M13/data1704/T20. The samples use that endpoint with a frozen analytical
proposal field. This accounting repair does not establish equality with the
original exact model likelihood. The old equivalence calibration cannot be
applied to arbitrary pooled sample counts and has not been relabeled passed.

Post-run review: a favorable early diagnostic can miss slow exploration. Later
deterioration is counterevidence to durable convergence. A longer healthy
cumulative history could resolve this; it must not discard the adverse suffix.

The old deadline has expired. The total cap is {report['total_postwarmup_maximum']:,}
per chain; {full['draws_per_chain']:,} have been used. The runtime repair and archive
reclassification are complete. Any later continuation belongs to this same
master and starts from the saved chains; it requires a new time allocation.
"""
    (out/"result.md").write_text(text)
