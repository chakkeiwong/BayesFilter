#!/usr/bin/env python3
"""Bounded predator–prey/SIR diagnostic campaign; shared analytical LEDH only.

The controller is standard-library only. Workers configure GPU memory growth
before importing numerical modules. No configuration in this diagnostic runner
constitutes scope-specific tuning or admission. See the companion plan/README.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/ledh-nonlinear-master-20261002.md"
SCHEMA = "bayesfilter.nonlinear_moment_comparison.v1"
MODELS = {"predator_prey": (2, 2, 6), "sir_d18": (18, 9, 3)}
ROUTES = ("iid_dual_cap", "previous_inverse_cdf", "repaired_permutation")
ARMS = ("covariance_only", "original", "richer_marginal", "richer_pairwise", "guarded_pairwise")
BASE = dict(flow_substeps=8, reset_epsilon=102.4, reset_sinkhorn_steps=24,
            reset_balance_steps=12, correction_steps=4, correction_strength=.12,
            pairwise_steps=4, pairwise_strength=.03, reset_ridge=1e-5,
            correction_lm_damping=.01, correction_lm_scale_floor=.0001,
            correction_trust_radius=.5, pairwise_rms_cap=2.,
            coordinate_cap_power=8, state_map_policy="adaptive_empirical", hilbert_bits=12)


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def arm_settings(arm, base):
    controls = dict(base, moment_safety=False, coordinate_cap_identity_radius=0.)
    design = "repeated_axes"
    if arm == "covariance_only":
        controls.update(correction_steps=0, pairwise_steps=0)
    elif arm != "original":
        design = "normal_quantiles"
        controls["coordinate_cap_identity_radius"] = 8.
        if arm == "richer_marginal":
            controls["pairwise_steps"] = 0
        elif arm == "guarded_pairwise":
            controls["moment_safety"] = True
        elif arm != "richer_pairwise":
            raise ValueError(f"unknown arm: {arm}")
    return controls, design


def reference_for(entries, scope):
    """A provided reference must match the observed-data target exactly."""
    keys = ("target_id", "observation_sha256", "horizon", "theta", "parameter_names")
    matching = [r for r in entries if all(r.get(k) == scope[k] for k in keys)]
    if not matching:
        return None
    if len(matching) != 1:
        raise ValueError("duplicate references for the same target/data/parameter point")
    ref = matching[0]
    if ref.get("kind") not in ("exact", "numerically_converged", "approximate"):
        raise ValueError("reference kind must be explicit")
    if not ref.get("verification") or not ref.get("source"):
        raise ValueError("reference needs source and verification description")
    values = [ref["log_likelihood"], *ref["score"]]
    if len(ref["score"]) != len(scope["parameter_names"]) or not all(math.isfinite(x) for x in values):
        raise ValueError("invalid reference value/score")
    return ref


def validate_job(job):
    d, _, p = MODELS[job["model"]]
    n, horizon = job["particles"], job["horizon"]
    if n < 2*d or n % (2*d) or horizon < 1:
        raise ValueError("N must be divisible by 2d, N >= 2d, and T >= 1")
    # Same exact-divisor policy as the numerical selector; the worker also calls it.
    if n > 3000 and not any(n % k == 0 for k in range(2, 3001)):
        raise ValueError("particle count has no admissible transport divisor")
    for theta in job["theta_points"] or []:
        if len(theta) != p or not all(math.isfinite(x) for x in theta):
            raise ValueError("invalid physical parameter point")
        if job["model"] == "predator_prey" and any(x <= 0 for x in theta):
            raise ValueError("predator–prey physical parameters must be positive")
    if not job["design_seeds"] or len(set(job["design_seeds"])) != len(job["design_seeds"]):
        raise ValueError("particle-design seeds must be nonempty and distinct")
    if any(s < 0 or s >= 2**31-10000 for s in [job["data_seed"], *job["design_seeds"]]):
        raise ValueError("seeds must fit the stateless generator's positive int32 range")


def worker(job_path):
    started = time.monotonic()
    job = read(job_path)
    validate_job(job)
    out = Path(job_path).parent
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if job["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=job["device"] == "gpu")
    tf.config.experimental.enable_tensor_float_32_execution(job["dtype"] == "float32")
    from bayesfilter.highdim import sqmc_campaign_tf as common
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec, trace_kernel
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
    spec = NonlinearSQMCSpec(job["model"])
    dtype = tf.as_dtype(job["dtype"])
    device = "/GPU:0" if job["device"] == "gpu" else "/CPU:0"
    manifest = dict(schema=SCHEMA, job=job, git_commit=git("rev-parse", "HEAD"),
                    command=sys.argv, python=sys.executable, tensorflow=tf.__version__,
                    memory_policy=memory, device=device, cpu_gpu_hidden=job["device"] == "cpu",
                    tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
                    jit_compile=job["jit_compile"], chunk_size=select_transport_chunk_size(job["particles"]),
                    evidence_role="diagnostic_only", tuning_status="untuned_scope_warm_start",
                    plan=PLAN, score_method="analytical_recursive_total_selected_branch",
                    observations_timing="x0_then_transition_then_observe_y1_to_yT")
    dump(out / "manifest.json", manifest)
    with tf.device(device):
        observations = spec.observations(job["horizon"], job["data_seed"], dtype=dtype,
                                         jit_compile=job["jit_compile"])
        if not bool(tf.reduce_all(tf.math.is_finite(observations))):
            raise ValueError("generated observations are nonfinite")
        obs_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(observations).numpy())).hexdigest()
        dump(out / "dataset.json", dict(observations=observations.numpy().tolist(),
              observation_sha256=obs_hash, data_seed=job["data_seed"], target_id=spec.target_id,
              generator="canonical_adapter_same_target_transition_first_v1", dtype=dtype.name))
        entries = job["references"]
        points = job["theta_points"] or [spec.default_theta(dtype).numpy().tolist()]
        rows = []
        for point_index, point in enumerate(points):
            theta = tf.constant(point, dtype)
            scope = dict(target_id=spec.target_id, observation_sha256=obs_hash,
                         horizon=job["horizon"], theta=theta.numpy().tolist(),
                         parameter_names=list(spec.parameter_names))
            reference = reference_for(entries, scope)
            if entries and reference is None:
                raise ValueError("provided reference file has no exact match for this scope")
            for seed in job["design_seeds"]:
                inputs = common.random_inputs(job["route"], seed, job["particles"],
                                              spec.dimension, job["horizon"], dtype,
                                              jit_compile=job["jit_compile"])
                for arm in job["arms"]:
                    controls, design = arm_settings(arm, job["controls"])
                    diagnostics = {}
                    tick = time.monotonic()
                    value, score, valid = common.value_and_score(
                        spec, job["route"], controls, theta, observations, seed,
                        job["particles"], jit_compile=job["jit_compile"], inputs=inputs,
                        diagnostics=diagnostics, reset_design_kind=design)
                    ell, scores = float(value), score.numpy().tolist()
                    finite = bool(valid) and math.isfinite(ell) and all(math.isfinite(x) for x in scores)
                    row = dict(**scope, model=spec.name, route=job["route"], arm=arm,
                               point_index=point_index, particles=job["particles"], data_seed=job["data_seed"],
                               design_seed=seed, log_likelihood=ell, score=scores, valid=finite,
                               controls=common.numerical_settings(controls), reset_design=design,
                               diagnostics=diagnostics, reference=reference,
                               oracle_available=reference is not None and reference["kind"] != "approximate",
                               log_likelihood_error=None, score_error=None, score_l2_error=None)
                    if finite and reference is not None:
                        errors = [a-b for a, b in zip(scores, reference["score"])]
                        row.update(log_likelihood_error=ell-reference["log_likelihood"],
                                   score_error=errors, score_l2_error=math.sqrt(sum(e*e for e in errors)))
                    if job["trace"] and seed == job["design_seeds"][0]:
                        kernel = trace_kernel(spec, job["route"], controls, job["particles"],
                                              job["horizon"], dtype, jit_compile=job["jit_compile"],
                                              reset_design_kind=design)
                        trace_value, trace_score, trace = kernel(theta, *inputs, observations)
                        # Explicitly check that diagnostic tracing preserves the value program.
                        tol = (1e-5 if dtype == tf.float32 else 1e-10) * (1+abs(ell))
                        row["trace_value_difference"] = float(trace_value)-ell
                        row["trace_score_coordinate_zero_difference"] = float(trace_score[0])-scores[0]
                        row["trace_matches_value"] = finite and abs(float(trace_value)-ell) <= tol
                        score_tol = (1e-5 if dtype == tf.float32 else 1e-10) * (1+abs(scores[0]))
                        row["trace_matches_score"] = finite and abs(float(trace_score[0])-scores[0]) <= score_tol
                        record = [{k: v.numpy().tolist() for k, v in r.items()} for r in trace]
                        trace_name = f"trace-p{point_index}-s{seed}-{arm}.json"
                        dump(out / trace_name, json_safe(record))
                        row["trace_file"] = trace_name
                        row["valid"] = finite and row["trace_matches_value"] and row["trace_matches_score"]
                    row["wall_seconds"] = time.monotonic()-tick
                    rows.append(json_safe(row))
                    dump(out / "rows.json", rows)
                    print(json.dumps(dict(arm=arm, design_seed=seed, log_likelihood=json_safe(ell),
                                          score=json_safe(scores), valid=row["valid"])), flush=True)
    manifest["allocator"] = (tf.config.experimental.get_memory_info("GPU:0")
                              if job["device"] == "gpu" else None)
    manifest["completed_rows"] = len(rows)
    manifest["wall_seconds"] = time.monotonic()-started
    manifest["output_artifacts"] = [str(p) for p in sorted(out.iterdir())]
    manifest["environment_prefix"] = sys.prefix
    dump(out / "manifest.json", manifest)
    return 0 if all(r["valid"] for r in rows) else 2


def mean_se(values):
    return dict(mean=statistics.mean(values),
                se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)


def heuristic_comparisons(rows, keys):
    """Conditional paired errors; descriptive losses veto promotion, never rank."""
    valid = [r for r in rows if r["valid"] and r["reference"] is not None
             and r["reference"]["kind"] in ("exact", "numerically_converged")]
    lookup = {tuple(r[k] for k in keys)+(r["design_seed"],): r for r in valid}
    grouped = {}
    for row in valid:
        if row["arm"] not in ("richer_pairwise", "guarded_pairwise"):
            continue
        for adversary in ("covariance_only", "richer_marginal", "original"):
            other = lookup.get(tuple(row[k] for k in keys[:-1])+(adversary,row["design_seed"]))
            if other is None:
                continue
            key = tuple(row[k] for k in keys)+(adversary,)
            grouped.setdefault(key, []).append((
                row["score_l2_error"]-other["score_l2_error"],
                abs(row["log_likelihood_error"])-abs(other["log_likelihood_error"])))
    comparisons = []
    for key, pairs in grouped.items():
        score = mean_se([p[0] for p in pairs])
        value = mean_se([p[1] for p in pairs])
        comparisons.append(dict(zip((*keys,"comparator"),key), n=len(pairs),
            score_error_difference=score, absolute_likelihood_error_difference=value,
            descriptive_underperformance=score["mean"]>0 or value["mean"]>0))
    status = ("observed_underperformance_promotion_veto" if any(
        c["descriptive_underperformance"] for c in comparisons) else
        "no_observed_underperformance_no_promotion" if comparisons else
        "not_established_missing_reference_or_comparators")
    return dict(status=status, comparisons=comparisons,
        interpretation="Conditional descriptive comparison only; no statistical ranking. "
                       "Incomplete cells and reference uncertainty require separate assessment.")


def summarize(out, attempts):
    rows = []
    for item in attempts:
        path = out / item["directory"] / "rows.json"
        if path.exists():
            rows.extend(read(path))
    dump(out / "rows.json", rows)
    keys = ("model", "horizon", "particles", "data_seed", "point_index", "route", "arm")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries, components = [], []
    for key, group in groups.items():
        good = [r for r in group if r["valid"]]
        item = dict(zip(keys, key), n=len(group), valid_n=len(good),
                    log_likelihood=None, score=None, reference=group[0]["reference"],
                    mean_abs_log_likelihood_error=None, mean_score_l2_error=None)
        if good:
            item["log_likelihood"] = mean_se([r["log_likelihood"] for r in good])
            item["score"] = [mean_se([r["score"][i] for r in good]) for i in range(len(good[0]["score"]))]
            if good[0]["reference"]:
                item["mean_abs_log_likelihood_error"] = mean_se([abs(r["log_likelihood_error"]) for r in good])
                item["mean_score_l2_error"] = mean_se([r["score_l2_error"] for r in good])
            for name, stats in zip(["log_likelihood", *good[0]["parameter_names"]],
                                   [item["log_likelihood"], *item["score"]]):
                ref = item["reference"]
                oracle_value = None if ref is None else (ref["log_likelihood"] if name == "log_likelihood"
                    else ref["score"][good[0]["parameter_names"].index(name)])
                components.append(dict(zip(keys,key), quantity=name, **stats, reference_value=oracle_value,
                                       reference_kind=None if ref is None else ref["kind"], n=len(good)))
        summaries.append(item)
    paired = []
    base = {tuple(r[k] for k in keys[:-1])+(r["design_seed"],): r for r in rows
            if r["arm"] == "original" and r["valid"]}
    for r in rows:
        old = base.get(tuple(r[k] for k in keys[:-1])+(r["design_seed"],))
        if old and r["valid"] and r["arm"] != "original":
            paired.append(dict(zip(keys,[r[k] for k in keys]), design_seed=r["design_seed"],
                               log_likelihood_change=r["log_likelihood"]-old["log_likelihood"],
                               score_change=[a-b for a,b in zip(r["score"],old["score"])],
                               score_error_change=None if r["reference"] is None else
                               r["score_l2_error"]-old["score_l2_error"]))
    heuristics = heuristic_comparisons(rows, keys)
    dump(out / "summary.json", dict(cells=summaries, paired_changes=paired,
         heuristic_dominance=heuristics,
         inference_status="no default or statistical ranking; incomplete/invalid replicates are explicitly counted"))
    with (out / "values-and-scores.csv").open("w", newline="") as stream:
        fields = [*keys,"quantity","mean","se","reference_value","reference_kind","n"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(components)
    lines = ["# Nonlinear likelihood and score comparison", "",
             "Heuristic comparison: "+heuristics["status"]+". See summary.json for conditional paired errors.", "",
             "Means and standard errors are conditional on each dataset and parameter point. "
             "Missing references are unavailable, not zero. Invalid/unfinished runs are counted; "
             "means over valid runs alone must not support a ranking.", "",
             "| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |",
             "|---|---|---:|---:|---|"]
    for c in summaries:
        ell = "unavailable" if c["log_likelihood"] is None else f'{c["log_likelihood"]["mean"]:.9g}'
        scores = "unavailable" if c["score"] is None else ", ".join(f'{s["mean"]:.9g}' for s in c["score"])
        lines.append(f'| {c["model"]} / {c["horizon"]} / {c["particles"]} / {c["data_seed"]} / {c["point_index"]} | {c["route"]} / {c["arm"]} | {c["valid_n"]} / {c["n"]} | {ell} | {scores} |')
    lines += ["", "Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. "
              "Paired changes are in `summary.json`; all realized values are in `rows.json`.", "",
              "| Worker | Exit | Wall seconds |", "|---|---:|---:|"]
    lines += [f'| {a["directory"]} | {a["status"]} | {a["wall_seconds"]:.2f} |' for a in attempts]
    (out / "results.md").write_text("\n".join(lines)+"\n")


def controller(args):
    controls = read(args.controls_json) if args.controls_json else {}
    theta = read(args.theta_json) if args.theta_json else {}
    references = read(args.reference_file) if args.reference_file else []
    if not isinstance(references, list):
        raise ValueError("reference file must be a JSON list")
    for config in (controls, theta):
        if set(config)-set(MODELS):
            raise ValueError("configuration contains an unknown model")
    jobs = []
    for model, horizon, n, data_seed, route in itertools.product(
            args.models, args.horizons, args.particles, args.data_seeds, args.routes):
        job = dict(model=model, horizon=horizon, particles=n, data_seed=data_seed, route=route,
                   controls=dict(BASE, **controls.get(model, {})), theta_points=theta.get(model),
                   design_seeds=args.design_seeds, arms=args.arms, device=args.device,
                   dtype=args.dtype, jit_compile=not args.no_jit, trace=args.trace,
                   references=references)
        validate_job(job)
        jobs.append(job)
    description = dict(schema=SCHEMA, plan=PLAN, jobs=jobs, worker_count=len(jobs),
                       evidence_role="diagnostic_only", controls_status="untuned_scope_warm_start",
                       budget_seconds=args.budget_seconds, per_worker_seconds=args.worker_seconds,
                       reference_mode="supplied_same_target" if references else "unavailable",
                       heuristic_adversaries=["covariance_only", "richer_marginal", "original"])
    if args.mode == "plan":
        print(json.dumps(description, indent=2))
        return 0
    if args.budget_seconds is None or args.budget_seconds <= 0 or args.worker_seconds <= 0:
        raise ValueError("run requires positive --budget-seconds and --worker-seconds")
    if not all(math.isfinite(x) for x in (args.budget_seconds, args.worker_seconds)):
        raise ValueError("budgets must be finite")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "campaign.json", description)
    sources = [Path(__file__), ROOT/PLAN, *sorted((ROOT/"bayesfilter/highdim").glob("*.py"))]
    dump(out / "manifest.json", dict(git_commit=git("rev-parse","HEAD"),
          git_status=git("status","--short"), command=sys.argv, python=sys.executable,
          started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), plan=PLAN,
          sources={str(p.relative_to(ROOT)):sha(p) for p in sources}, output=str(out)))
    started, attempts = time.monotonic(), []
    interrupted = False
    for index, job in enumerate(jobs):
        remaining = args.budget_seconds-(time.monotonic()-started)
        if remaining <= 0:
            break
        directory = out/f"worker-{index+1:04d}-{job['model']}"
        directory.mkdir()
        dump(directory/"job.json",job)
        command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--job",str(directory/"job.json")]
        tick = time.monotonic()
        with (directory/"worker.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=min(remaining,args.worker_seconds))
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                interrupted = isinstance(exc,KeyboardInterrupt)
                status = "interrupted" if interrupted else "timeout"
        attempts.append(dict(directory=directory.name, status=status, command=command,
                             wall_seconds=time.monotonic()-tick))
        dump(out/"attempts.json", attempts)
        summarize(out, attempts)
        print(json.dumps(attempts[-1]), flush=True)
        if interrupted:
            break
    complete = len(attempts)==len(jobs) and all(a["status"]==0 for a in attempts)
    dump(out/"completion.json",dict(complete=complete, planned_workers=len(jobs),
         attempted_workers=len(attempts), elapsed_seconds=time.monotonic()-started,
         budget_seconds=args.budget_seconds, interrupted=interrupted))
    summarize(out, attempts)
    return 0 if complete else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run","_worker"))
    p.add_argument("--job",type=Path)
    p.add_argument("--output",type=Path,default=ROOT/"docs/plans/artifacts/ledh-nonlinear-master-20261002/run-01")
    p.add_argument("--models",nargs="+",choices=tuple(MODELS),default=list(MODELS))
    p.add_argument("--horizons",nargs="+",type=int,default=[20])
    p.add_argument("--particles",nargs="+",type=int,default=[1008])
    p.add_argument("--data-seeds",nargs="+",type=int,default=[260201,260202])
    p.add_argument("--design-seeds",nargs="+",type=int,default=list(range(260301,260309)))
    p.add_argument("--routes",nargs="+",choices=ROUTES,default=list(ROUTES))
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--dtype",choices=("float32","float64"),default="float32")
    p.add_argument("--device",choices=("gpu","cpu"),default="gpu")
    p.add_argument("--no-jit",action="store_true",help="explicit reference/debug exception")
    p.add_argument("--trace",action="store_true",help="save shared reset diagnostics for the first design seed")
    p.add_argument("--controls-json",type=Path)
    p.add_argument("--theta-json",type=Path)
    p.add_argument("--reference-file",type=Path)
    p.add_argument("--budget-seconds",type=float)
    p.add_argument("--worker-seconds",type=float,default=1800.)
    return p


if __name__ == "__main__":
    options = parser().parse_args()
    try:
        raise SystemExit(worker(options.job) if options.mode=="_worker" else controller(options))
    except Exception as exc:
        if options.mode=="_worker" and options.job:
            dump(options.job.parent/"failure.json", dict(error=type(exc).__name__, message=str(exc)))
        traceback.print_exc()
        raise SystemExit(1)
