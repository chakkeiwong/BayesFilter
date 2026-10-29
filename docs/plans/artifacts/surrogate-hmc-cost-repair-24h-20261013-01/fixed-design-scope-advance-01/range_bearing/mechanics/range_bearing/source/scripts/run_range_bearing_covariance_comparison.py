#!/usr/bin/env python3
"""Bounded M13 diagnostic, not an HMC admission or tuning authority.

Shared TensorFlow numerical kernels; this file only binds the existing model,
orchestrates disjoint tuning/evaluation, and reports independent references.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "docs/plans/ledh-range-bearing-covariance-comparison-2026-10-09.md"
DEFAULT_OUT = ROOT / "docs/plans/artifacts/ledh-range-bearing-covariance-20261009-01"
POINTS = [[0., 0.], [-.73272085, -.01465508]]
CALIBRATION = [18010, 18011]
VALIDATION = [18012, 18013]
HELDOUT = [1704, 18020, 18021]


def clean(x):
    if hasattr(x, "numpy"):
        return clean(x.numpy().tolist())
    if dataclasses.is_dataclass(x):
        return clean(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def checked_reference(path):
    row = load(path)
    scores = row.get("replicate_scores", [])
    values = row.get("replicate_values", [])
    valid = row.get("valid") and row.get("replicates") == 16
    valid = valid and len(row.get("score", [])) == 2 and len(row.get("mcse", [])) == 3
    valid = valid and len(values) == 16 and len(scores) == 16 and all(len(s) == 2 for s in scores)
    numbers = [row.get("value"), *row.get("score", []), *row.get("mcse", []), *values]
    numbers.extend(v for s in scores for v in s)
    valid = valid and all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers)
    valid = valid and all(v >= 0 for v in row.get("mcse", []))
    if not valid:
        raise ValueError("Invalid independent reference: " + str(path))
    return row


def emit(args, row):
    row = clean(row)
    with (args.output / (args.stage + ".jsonl")).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({k: row[k] for k in
          ("kind", "arm", "data_seed", "point", "seed", "valid", "wall_seconds") if k in row}), flush=True)


def prepare(args):
    # Must precede imports that create TensorFlow constants or devices.
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    from docs.benchmarks.run_ledh_covariance_proposal import prepare as setup
    tf, memory = setup(SimpleNamespace(device="gpu", dtype=args.dtype,
                                      no_tf32=args.no_tf32 or args.dtype == "float64"))
    return tf, memory


def scope(args):
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    return dict(model="M13_wrapped_range_bearing", horizon=20, particles=args.particles,
                state_dimension=4, observation_dimension=2, parameter_dimension=2,
                dtype=args.dtype, tf32=args.dtype == "float32" and not args.no_tf32, jit_compile=True,
                device="GPU", chunk=select_transport_chunks(args.particles).row_chunk_size, points=POINTS,
                evidence_role="scope_specific_diagnostic_not_canonical_admission")


def manifest(args, tf, memory):
    paths = [Path(__file__), PLAN, ROOT / "scripts/range_bearing_particle_ladder.py"]
    # Ordinary source checksums preserve dirty-tree provenance as well as HEAD.
    for directory in ("bayesfilter/highdim", "bayesfilter/testing", "bayesfilter/runtime",
                      "experiments/dpf_implementation/tf_tfp/fixtures"):
        paths.extend((ROOT / directory).glob("*.py"))
    return dict(scope=scope(args), command=[sys.executable, *sys.argv], python=sys.executable,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                tensorflow=tf.__version__, memory_policy=memory, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                devices=[d.name for d in tf.config.list_physical_devices("GPU")],
                trust_basis="escalated_local_GPU_execution", plan=PLAN,
                data_seeds=dict(calibration=CALIBRATION, validation=VALIDATION, heldout=HELDOUT),
                output=args.output, cpu_only=False, default_promoted=False)


class RangeBinding:
    """Bind the existing M13 callbacks to the shared filter, with identical x0.

    The generic initial input is a frozen particle cloud here. Its parameter
    tangent is zero because M13's initial distribution has no theta dependence.
    """
    dimension = 4
    observation_dimension = 2
    parameter_count = 2

    def __init__(self, spec, tf, dtype):
        self.spec, self.tf, self.dtype = spec, tf, dtype

    def model(self, theta, direction):
        from bayesfilter.highdim.ledh_canonical_batch_fused_tf import _single_cloud_model
        return _single_cloud_model(self.spec["model"], theta, direction), None

    def initial_cloud(self, theta, initial, direction):
        tf = self.tf
        covariance = tf.broadcast_to(tf.cast(self.spec["fixture"].P0, self.dtype), [initial.shape[0], 4, 4])
        return initial, covariance, tf.zeros_like(initial), tf.zeros_like(covariance)


def specs(tf, dtype, horizon=20):
    from bayesfilter.testing.marginal_weight_campaign import specification
    spec = specification("m13-t" + str(horizon), dtype)
    return spec, RangeBinding(spec, tf, dtype)


def observations(args, tf, seed, dtype, horizon=20):
    from bayesfilter.testing.marginal_weight_campaign import range_fixture
    path = args.output / "data" / (str(seed) + "-t" + str(horizon) + ".json")
    if not path.exists():
        with tf.device("/CPU:0"):
            values = tf.cast(range_fixture(horizon, seed).observations, tf.float32)
        save(path, dict(seed=seed, horizon=horizon, observations=values,
                       precision="FP32-rounded once; all arms/references consume these same observations"))
    return tf.constant(load(path)["observations"], dtype)


def inputs(tf, spec, n, seed, dtype):
    with tf.device("/CPU:0"):
        z = tf.random.stateless_normal([spec["t"] + 2, n, 4], [seed, 19], dtype=tf.float64)
        fixture = spec["fixture"]
        base = tf.cast(fixture.m0, tf.float64) + z[0] @ tf.transpose(tf.linalg.cholesky(tf.cast(fixture.P0, tf.float64)))
        reset = z[-1] - tf.reduce_mean(z[-1], axis=0)
        reset = tf.transpose(tf.linalg.triangular_solve(tf.linalg.cholesky(tf.transpose(reset) @ reset / n), tf.transpose(reset)))
        uniform = tf.random.stateless_uniform([spec["t"], n, 2], [seed, 29], dtype=tf.float64)
        return tuple(tf.cast(tf.cast(x, tf.float32), dtype) for x in (base, z[1:-1], uniform, reset))


def new_result(tf, program, theta, obs, fixed, beta):
    from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS
    start = time.monotonic()
    base, noise, uniform, _ = fixed
    calls = [program(theta, tf.eye(2, dtype=theta.dtype)[j], obs, base, noise, uniform, beta) for j in range(2)]
    values = [float(r["value"]) for r in calls]
    scores = [float(r["score"]) for r in calls]
    finite = all(math.isfinite(v) for v in values + scores)
    tolerance = 1e-4 if theta.dtype == tf.float32 else 1e-9
    valid = finite and all(bool(r["valid"]) and int(r["steps_completed"]) == obs.shape[0] for r in calls)
    valid = valid and abs(values[0] - values[1]) <= tolerance
    return dict(value=values[0], score=scores, valid=valid,
                primal_direction_difference=abs(values[0] - values[1]),
                steps_completed=[int(r["steps_completed"]) for r in calls],
                trace_fields=TRACE_FIELDS, trace=calls[0]["trace"], wall_seconds=time.monotonic() - start)


def old_result(program, theta, obs, fixed):
    start = time.monotonic()
    base, noise, _, reset = fixed
    result = clean(program(theta, base, noise, reset, obs))
    result["valid"] = bool(result["valid"] and result["value_valid"] and result["force_valid"]
                           and len(result["score"]) == 2
                           and all(math.isfinite(v) for v in [result["value"], *result["score"]]))
    result["wall_seconds"] = time.monotonic() - start
    return result


def check(args, tf, dtype):
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter
    from bayesfilter.testing.marginal_weight_campaign import compile_filter, controls
    horizon = 2 if dtype == tf.float64 else 20
    spec, binding = specs(tf, dtype, horizon)
    obs = observations(args, tf, CALIBRATION[0], dtype, horizon)
    fixed = inputs(tf, spec, args.particles, 19001, dtype)
    theta, beta = tf.constant(POINTS[0], dtype), tf.constant([1/3]*3, dtype)
    program = make_filter(binding, args.particles, horizon, ProposalControls(), dtype)
    row = new_result(tf, program, theta, obs, fixed, beta)
    row.update(kind="check", arm="new", horizon=horizon)
    # Executable call-chain check: the new consumer resolves the shared wrapped
    # observation callback, including direct parameter and state tangents.
    bound, _ = binding.model(theta, tf.constant([.3, -.2], dtype))
    x = fixed[0][:5]
    dx = tf.ones_like(x) * tf.constant(.013, dtype)
    theta_rows = tf.broadcast_to(theta[None], [5, 2])
    direction_rows = tf.broadcast_to(tf.constant([[.3, -.2]], dtype), [5, 2])
    original = spec["model"]
    row["callback_value_max_difference"] = tf.reduce_max(tf.abs(bound.observation_log_density_fn(theta, x, obs[0]) - original.observation_log_density_fn(theta_rows, x, obs[0])))
    row["callback_score_max_difference"] = tf.reduce_max(tf.abs(bound.observation_log_density_tangent_fn(theta, x, obs[0], dx) - original.observation_log_density_tangent_fn(theta_rows, x, obs[0], dx, direction_rows)))
    row["valid"] &= float(row["callback_value_max_difference"]) == 0 and float(row["callback_score_max_difference"]) == 0
    if dtype == tf.float64:
        differences = []
        for step in (1e-4, 5e-5):
            fd = []
            for j in range(2):
                delta = tf.eye(2, dtype=dtype)[j] * step
                plus = new_result(tf, program, theta + delta, obs, fixed, beta)
                minus = new_result(tf, program, theta - delta, obs, fixed, beta)
                row["valid"] &= plus["valid"] and minus["valid"]
                fd.append((plus["value"] - minus["value"]) / (2*step))
            error = max(abs(a-b)/(1+abs(b)) for a, b in zip(row["score"], fd))
            differences.append(dict(step=step, score=fd, max_scaled_error=error))
            row["valid"] &= error <= 2e-4
        row["finite_differences"] = differences
    else:
        save(args.output / (args.stage + "-new.json"), row)
        previous = old_result(compile_filter(spec, args.particles, controls(0), "marginal_mixture", dtype), theta, obs, fixed)
        emit(args, dict(kind="check", arm="old", **previous))
    emit(args, row)
    save(args.output / (args.stage + ".json"), row)


def references(args, tf):
    from bayesfilter.testing.marginal_weight_campaign import m13_fisher_reference
    spec, _ = specs(tf, tf.float64)
    for n in (16384, 65536, args.particles):
        seeds = VALIDATION if n == 16384 else HELDOUT
        if n == 16384:
            seeds = VALIDATION + HELDOUT
        kernel = m13_fisher_reference(spec, tf.zeros([20, 2], tf.float64), n, 16, dynamic_observations=True)
        for seed in seeds:
            obs = observations(args, tf, seed, tf.float64)
            for point, theta in enumerate(POINTS):
                reference_path = args.output / "references" / f"{seed}-{point}-{n}.json"
                if reference_path.exists():
                    checked_reference(reference_path)
                    continue
                start = time.monotonic()
                result = kernel(tf.constant(theta, tf.float64), obs, tf.constant([seed + 40000, n + point], tf.int32))
                row = dict(kind="reference", data_seed=seed, point=point, particles=n, replicates=16,
                           random_seed=[seed+40000, n+point], **clean(result), wall_seconds=time.monotonic()-start)
                row["valid"] = all(math.isfinite(v) for v in [row["value"], *row["score"], *row["mcse"]])
                emit(args, row)
                save(args.output / "references" / f"{seed}-{point}-{n}.json", row)


def calibrate_beta(args, tf, dtype, spec, binding, ctl, program):
    from bayesfilter.highdim.covariance_proposal_tf import build_maps, draw_from_maps, component_densities, observation_density
    from bayesfilter.highdim.covariance_proposal_beta_tf import make_beta_solver
    n = args.particles
    equal = tf.constant([1/3]*3, dtype)
    @tf.function(input_signature=[tf.TensorSpec([2], dtype), tf.TensorSpec([n,4], dtype),
        tf.TensorSpec([2], dtype), tf.TensorSpec([n,4], dtype), tf.TensorSpec([n,2], dtype)], jit_compile=True, autograph=False)
    def pilot(theta, ancestors, y, noise, u):
        model, _ = binding.model(theta, tf.zeros_like(theta))
        maps = build_maps(model, theta, ancestors, tf.zeros_like(ancestors), y, ctl)
        x, dx = draw_from_maps(maps, equal, u, noise)
        ld, _, valid = component_densities(x, dx, *maps[:4])
        return ld, observation_density(model, theta, x, dx, y)[0]+ld[:,0], valid & maps[12]
    components, targets, checks = [], [], []
    for k, data_seed in enumerate(CALIBRATION):
        obs = observations(args, tf, data_seed, dtype)
        fixed = inputs(tf, spec, n, 20001+k, dtype)
        for point in POINTS:
            theta = tf.constant(point, dtype)
            context = program(theta, tf.zeros([2], dtype), obs, *fixed[:3], equal)
            checks.append(bool(context["valid"]) and int(context["steps_completed"]) == 20)
            if not checks[-1]:
                return dict(valid=False, reason="invalid calibration context", steps_completed=int(context["steps_completed"]), trace=clean(context["trace"]))
            for t in (0, 10, 19):
                noise = tf.random.stateless_normal([n, 4], [22001+k, t], dtype=dtype)
                u = tf.random.stateless_uniform([n, 2], [23001+k, t], dtype=dtype)
                ld, la, valid = pilot(theta, context["ancestor_history"][t], obs[t], noise, u)
                checks.append(bool(valid)); components.append(ld); targets.append(la)
    fit = make_beta_solver(len(components), n, dtype)(tf.stack(components), tf.stack(targets),
                tf.constant(.1, dtype), tf.constant(1e-4 if dtype == tf.float32 else 1e-7, dtype))
    return dict(valid=all(checks) and bool(fit["converged"]),
                **{k: clean(fit[k]) for k in ("beta", "gap", "objective", "iterations", "converged")},
                context_checks=checks, calibration_seeds=CALIBRATION, floor=.1)


def validation(args, tf, dtype, spec, program, arm, beta=None):
    rows = []
    for ds, seed in enumerate(VALIDATION):
        obs = observations(args, tf, seed, dtype)
        for point, values in enumerate(POINTS):
            reference = checked_reference(args.output / "references" / f"{seed}-{point}-16384.json")
            for replicate in range(2):
                design_seed = 21001 + 100*ds + replicate
                fixed = inputs(tf, spec, args.particles, design_seed, dtype)
                theta = tf.constant(values, dtype)
                result = new_result(tf, program, theta, obs, fixed, beta) if arm == "new" else old_result(program, theta, obs, fixed)
                row = clean(dict(kind="validation", arm=arm, data_seed=seed, point=point, seed=design_seed, **result))
                if row["valid"]:
                    row["score_error"] = math.dist(row["score"], reference["score"])
                    row["value_error"] = abs(row["value"]-reference["value"])
                rows.append(row); emit(args, row)
    valid = all(r["valid"] for r in rows)
    return dict(valid=valid, mean_score_error=statistics.mean(r["score_error"] for r in rows) if valid else None,
                mean_value_error=statistics.mean(r["value_error"] for r in rows) if valid else None, rows=rows)


def tune(args, tf, dtype):
    from bayesfilter.highdim.covariance_proposal_tf import make_filter, ProposalControls
    from bayesfilter.testing.marginal_weight_campaign import compile_filter, controls
    spec, binding = specs(tf, dtype)
    ledger = dict(scope=scope(args), selection="minimum mean validation score-L2 error; mean value error breaks exact ties",
                  calibration=CALIBRATION, validation=VALIDATION, candidates=[], selected={})
    for arm, count in (("new", 5), ("old", 2)):
        for index in range(count):
            if arm == "new":
                ctl = [
                    ProposalControls(),
                    ProposalControls(flow_steps=8),
                    ProposalControls(reset_epsilon=.5),
                    ProposalControls(reset_steps=80),
                    ProposalControls(reset_steps=120),
                ][index]
                program = make_filter(binding, args.particles, 20, ctl, dtype)
                fit = calibrate_beta(args, tf, dtype, spec, binding, ctl, program)
                setting = clean(ctl)
                beta = tf.constant(fit.get("beta", [1/3]*3), dtype)
            else:
                setting = controls(index)
                program = compile_filter(spec, args.particles, setting, "marginal_mixture", dtype)
                fit, beta = dict(valid=True, role="no mixture beta in comparator"), None
            val = validation(args, tf, dtype, spec, program, arm, beta) if fit["valid"] else dict(valid=False, reason="calibration failed")
            candidate = dict(arm=arm, index=index, controls=setting, calibration=fit, validation=val, valid=fit["valid"] and val["valid"])
            ledger["candidates"].append(candidate)
            save(args.output / "tuning.json", ledger)
        viable = [c for c in ledger["candidates"] if c["arm"] == arm and c["valid"]]
        if viable:
            winner = min(viable, key=lambda c: (c["validation"]["mean_score_error"], c["validation"]["mean_value_error"]))
            ledger["selected"][arm] = dict(index=winner["index"], controls=winner["controls"], beta=winner["calibration"].get("beta"))
        save(args.output / "tuning.json", ledger)
    ledger["valid"] = set(ledger["selected"]) == {"new", "old"}
    save(args.output / "tuning.json", ledger)


def compare(args, tf, dtype):
    from bayesfilter.highdim.covariance_proposal_tf import make_filter, ProposalControls
    from bayesfilter.testing.marginal_weight_campaign import compile_filter
    ledger = load(args.output / "tuning.json")
    if ledger["scope"] != scope(args) or not ledger.get("valid"):
        raise ValueError("No two valid frozen scope-matched candidates; held-out comparison cannot run")
    if set(HELDOUT) & set(ledger["calibration"] + ledger["validation"]):
        raise ValueError("Held-out and tuning observations overlap")
    spec, binding = specs(tf, dtype)
    selected = ledger["selected"]
    new = make_filter(binding, args.particles, 20, ProposalControls(**selected["new"]["controls"]), dtype)
    old = compile_filter(spec, args.particles, selected["old"]["controls"], "marginal_mixture", dtype)
    betas = dict(new=selected["new"]["beta"], prior=[1.,0.,0.], global_guide=[.1,.9,0.])
    for ds, seed in enumerate(HELDOUT):
        obs = observations(args, tf, seed, dtype)
        for point, values in enumerate(POINTS):
            for rep in range(args.replicates):
                design_seed = 31001 + 100*ds + rep
                fixed = inputs(tf, spec, args.particles, design_seed, dtype)
                theta = tf.constant(values, dtype)
                for arm in ("new", "old", "prior", "global_guide"):
                    result = old_result(old, theta, obs, fixed) if arm == "old" else new_result(tf, new, theta, obs, fixed, tf.constant(betas[arm], dtype))
                    row = dict(kind="heldout", arm=arm, data_seed=seed, point=point, seed=design_seed, **result)
                    emit(args, row)


def report(args):
    path = args.output / "compare.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    cells = []
    for seed in HELDOUT:
        for point in range(2):
            fine_path = args.output / "references" / f"{seed}-{point}-65536.json"
            coarse_path = args.output / "references" / f"{seed}-{point}-16384.json"
            if not fine_path.exists() or not coarse_path.exists():
                continue
            fine, coarse = checked_reference(fine_path), checked_reference(coarse_path)
            arms = {arm: [r for r in rows if r["data_seed"] == seed and r["point"] == point and r["arm"] == arm] for arm in ("new", "old", "prior", "global_guide")}
            summary = {}
            for arm, values in arms.items():
                valid = [r for r in values if r["valid"]]
                summary[arm] = dict(count=len(values), valid_count=len(valid),
                    mean_score_error=statistics.mean(math.dist(r["score"], fine["score"]) for r in valid) if valid else None,
                    mean_absolute_value_error=statistics.mean(abs(r["value"]-fine["value"]) for r in valid) if valid else None,
                    mean_seconds=statistics.mean(r["wall_seconds"] for r in valid) if valid else None)
            expected_seeds = {31001 + 100*HELDOUT.index(seed) + rep for rep in range(args.replicates)}
            # Invalid heuristic arms do not erase valid primary pairs.
            complete_by_arm = {
                arm: len(values) == args.replicates and all(r["valid"] for r in values)
                and {r["seed"] for r in values} == expected_seeds
                for arm, values in arms.items()}
            primary_complete = all(complete_by_arm[a] for a in ("new", "old"))
            heuristic_complete = all(complete_by_arm[a] for a in ("prior", "global_guide"))
            matched = [(r, s) for r in arms["new"] for s in arms["old"] if r["seed"] == s["seed"] and r["valid"] and s["valid"]]
            differences = [math.dist(r["score"], fine["score"])-math.dist(s["score"], fine["score"]) for r,s in matched]
            interval = None
            # Predeclared small-sample Student interval, conditional on dataset.
            if primary_complete and len(differences) == args.replicates:
                critical = {4: 3.182446305, 8: 2.364624252}[len(differences)]
                mean = statistics.mean(differences)
                half = critical*statistics.stdev(differences)/math.sqrt(len(differences))
                interval = [mean-half, mean+half]
            # L2 error is 1-Lipschitz in reference score, so the paired contrast
            # changes at most twice its displacement. This conservative envelope
            # is a sensitivity check, not a confidence bound on particle bias.
            reference_shift = math.dist(fine["score"], coarse["score"])
            reference_sensitivity = 2*max(reference_shift, 3*math.hypot(*fine["mcse"][1:]))
            envelope = [interval[0]-reference_sensitivity, interval[1]+reference_sensitivity] if interval else None
            verdict = "incomplete_or_invalid" if interval is None else "no_supported_ranking"
            if envelope and envelope[1] < 0:
                verdict = "conditional_score_improvement_with_reference_sensitivity"
            elif envelope and envelope[0] > 0:
                verdict = "conditional_score_deterioration_with_reference_sensitivity"
            bootstrap_path = args.output / "references" / f"{seed}-{point}-{args.particles}.json"
            if bootstrap_path.exists():
                bootstrap = checked_reference(bootstrap_path)
                summary["bootstrap_fisher"] = dict(count=16, particles=args.particles,
                    mean_score_error=statistics.mean(math.dist(v, fine["score"]) for v in bootstrap["replicate_scores"]),
                    mean_absolute_value_error=statistics.mean(abs(v-fine["value"]) for v in bootstrap["replicate_values"]))
            heuristics = {k: v["mean_score_error"] for k, v in summary.items()
                          if k not in ("new", "old")
                          and (k == "bootstrap_fisher" or complete_by_arm[k])
                          and v.get("mean_score_error") is not None}
            new_error = summary["new"]["mean_score_error"]
            heuristic_losses = [k for k, error in heuristics.items()
                                if primary_complete and new_error > error]
            if not primary_complete:
                heuristic_status = "not_checked"
            elif heuristic_losses:
                heuristic_status = "descriptive_underperformance_promotion_veto"
            elif not heuristic_complete or not bootstrap_path.exists():
                heuristic_status = "incomplete_heuristic_screen"
            else:
                heuristic_status = "screen_passed_no_superiority_claim"
            cells.append(dict(data_seed=seed, point=point, theta=POINTS[point], reference=fine,
                reference_score_refinement=reference_shift, reference_sensitivity=reference_sensitivity,
                arms=summary, paired_differences=differences, paired_95_student_interval=interval,
                reference_sensitivity_envelope=envelope, verdict=verdict, heuristic_dominance=heuristic_status,
                heuristic_losses=heuristic_losses, complete_by_arm=complete_by_arm,
                primary_complete=primary_complete,
                heuristic_complete=heuristic_complete and bootstrap_path.exists(),
                complete=primary_complete and bootstrap_path.exists()))
    result = dict(cells=cells, heldout_rows=len(rows), scope=scope(args),
                  complete=len(cells) == 6 and len(rows) == 24*args.replicates and all(c["complete"] for c in cells),
                  heuristic_suite_complete=len(cells) == 6 and all(c["heuristic_complete"] for c in cells),
                  no_HMC_or_default_claim=True,
                  uncertainty=f"Conditional on each dataset; {args.replicates} particle designs; reference particle bias not bounded; invalid-subset means are descriptive only")
    save(args.output / "report.json", result)
    print(json.dumps(clean(dict(heldout_rows=len(rows), cells=[{k:c[k] for k in ("data_seed", "point", "verdict", "paired_95_student_interval", "heuristic_dominance")} for c in cells])), indent=2))


def run(args):
    if args.particle_ladder:
        from range_bearing_particle_ladder import run_ladder
        return run_ladder(args)
    start = time.monotonic()
    execution = dict(budget_seconds=args.budget_seconds, attempts=[])
    jobs = [("check-fp32", "float32"), ("check-fp64", "float64"), ("references", "float64"), ("tune", args.dtype), ("compare", args.dtype)]
    if args.extend_from:
        source = args.extend_from.resolve()
        source_manifest = load(source / "compare-manifest.json")
        ledger = load(source / "tuning.json")
        if args.replicates != 8 or not ledger.get("valid") or ledger["scope"] != scope(args):
            raise ValueError("Extension requires eight designs and unchanged valid frozen tuning scope")
        if source_manifest["scope"] != scope(args):
            raise ValueError("Extension scope differs from original comparison")
        # Reporting and orchestration may be repaired; numerical dependencies
        # must be identical so the original frozen tuning remains applicable.
        for relative, digest in source_manifest["source_sha256"].items():
            if relative.startswith(("bayesfilter/", "experiments/")):
                if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                    raise ValueError("Numerical dependency changed since tuning: " + relative)
        args.output.mkdir(parents=True, exist_ok=False)
        inherited = {}
        for name in ("tuning.json", "tune-manifest.json"):
            shutil.copy2(source / name, args.output / name)
            inherited[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("references", "data"):
            shutil.copytree(source / name, args.output / name)
            for path in sorted((source / name).glob("*.json")):
                inherited[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        execution["extension"] = dict(source=str(source), inherited_sha256=inherited,
            policy="unchanged frozen tuning; eight paired designs including original four; no holdout retuning")
        jobs = [("compare", args.dtype)]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "execution.json", execution)
    for stage, dtype in jobs:
        remaining = args.budget_seconds - (time.monotonic()-start)
        if remaining <= 0:
            break
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--stage", stage,
                   "--dtype", dtype, "--output", str(args.output), "--particles", str(args.particles),
                   "--gpu", str(args.gpu), "--replicates", str(args.replicates)]
        if args.no_tf32:
            command.append("--no-tf32")
        began = time.monotonic()
        try:
            with (args.output / (stage + ".log")).open("x") as stream:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = "budget_timeout"
        execution["attempts"].append(dict(stage=stage, command=command, exit_code=code, wall_seconds=time.monotonic()-began))
        execution["elapsed_seconds"] = time.monotonic()-start
        save(args.output / "execution.json", execution)
        print(json.dumps(execution["attempts"][-1]), flush=True)
        if code != 0:
            break
        if stage == "check-fp64" and not load(args.output / (stage+".json"))["valid"]:
            break
        if stage == "tune" and not load(args.output / "tuning.json").get("valid"):
            break
    report(args)


def main():
    global PLAN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "worker", "run", "report"])
    parser.add_argument("--stage", default="check-fp32", choices=["check-fp32", "check-fp64", "references", "tune", "compare", "ladder-check"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--particles", type=int, default=1008)
    parser.add_argument("--extend-from", type=Path, help="Reuse unchanged frozen tuning and references for the predeclared eight-design extension")
    parser.add_argument("--replicates", type=int, choices=[4,8], default=4)
    parser.add_argument("--budget-seconds", type=int, default=14400)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--no-tf32", action="store_true", help="Explicit precision diagnostic; requires fresh scope tuning")
    parser.add_argument("--gpu", default="0", help="CUDA ordinal or explicit GPU UUID")
    parser.add_argument("--particle-ladder", action="store_true", help="Bounded separately tuned N2016/N4032 extension")
    parser.add_argument("--retry-from", type=Path, help="Preserve failed ladder attempts and remaining aggregate budget")
    parser.add_argument("--plan-file", type=Path, default=PLAN)
    args = parser.parse_args()
    PLAN = args.plan_file.resolve()
    if args.particles < 1008 or args.particles > 4032:
        parser.error("This bounded scope requires 1008 <= N <= 4032; repository exact chunk policy applies")
    if args.command == "run":
        return run(args)
    if args.command == "report":
        if args.particle_ladder:
            from range_bearing_particle_ladder import progress
            return progress(args.output)
        return report(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / (args.stage + ".jsonl")).exists():
        raise FileExistsError("Use a fresh versioned output directory; do not overwrite prior evidence")
    tf, memory = prepare(args)
    dtype = tf.as_dtype(args.dtype)
    record = manifest(args, tf, memory)
    save(args.output / (args.stage+"-manifest.json"), record)
    began = time.monotonic()
    try:
        if args.stage == "ladder-check":
            from range_bearing_particle_ladder import repair_checks
            repair_checks(args, tf, dtype)
        elif args.command == "check" or args.stage.startswith("check"):
            check(args, tf, dtype)
        elif args.stage == "references":
            references(args, tf)
        elif args.stage == "tune":
            tune(args, tf, dtype)
        else:
            compare(args, tf, dtype)
    finally:
        record["wall_seconds"] = time.monotonic()-began
        record["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        save(args.output / (args.stage+"-manifest.json"), record)


if __name__ == "__main__":
    main()
