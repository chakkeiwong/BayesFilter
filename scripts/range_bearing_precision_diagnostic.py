#!/usr/bin/env python3
"""Matched arithmetic/timing diagnostic; no tuning, HMC or admission authority.

All filter calculations use the existing shared TensorFlow/XLA implementation.
Host-side orchestration, hashes and descriptive reporting use the standard library.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_range_bearing_covariance_comparison as base

PLAN = ROOT / "docs/plans/ledh-range-bearing-precision-ladder-2026-10-10.md"
ARTIFACTS = ROOT / "docs/plans/artifacts"
SOURCES = {
    1008: ARTIFACTS / "ledh-range-bearing-covariance-20261009-07",
    2016: ARTIFACTS / "ledh-range-bearing-particle-ladder-20261010-02/N2016",
    4032: ARTIFACTS / "ledh-range-bearing-particle-ladder-20261010-02/N4032",
}
MODES = ("fp64", "fp32", "fp32_tf32")
RESULT = ROOT / "docs/plans/ledh-range-bearing-precision-results-2026-10-10.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(base.clean(row), allow_nan=False) + "\n")


def signature(row):
    return row["data_seed"], row["point"], row["seed"]


def provenance():
    paths = [Path(__file__), Path(base.__file__), PLAN]
    for directory in ("bayesfilter/highdim", "bayesfilter/testing", "bayesfilter/runtime",
                      "experiments/dpf_implementation/tf_tfp/fixtures"):
        paths.extend((ROOT / directory).glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def worker(args):
    from_mode = dict(fp64=("float64", True), fp32=("float32", True),
                     fp32_tf32=("float32", False))
    args.dtype, args.no_tf32 = from_mode[args.mode]
    tf, memory = base.prepare(args)
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    dtype = tf.as_dtype(args.dtype)
    start = time.monotonic()
    out = args.output / f"N{args.particles}-{args.mode}"
    out.mkdir(exist_ok=False)
    source = SOURCES[args.particles]
    tuning = base.load(source / "tuning.json")
    selected = tuning["selected"]["new"]
    if not tuning["valid"] or tuning["scope"]["particles"] != args.particles:
        raise ValueError("Missing or wrong saved comparator tuning")
    spec64, _ = base.specs(tf, tf.float64)
    _, binding = base.specs(tf, dtype)
    program = make_filter(binding, args.particles, 20, ProposalControls(**selected["controls"]), dtype)

    def common(value):
        with tf.device("/CPU:0"):
            # The exact same FP32-representable inputs are used in every arm.
            return tf.cast(tf.constant(value, tf.float32), dtype)

    beta = common(selected["beta"])
    def tensors(ds, point, rep):
        data_seed = base.HELDOUT[ds]
        design_seed = 31001 + 100 * ds + rep
        obs = common(base.load(source / "data" / f"{data_seed}-t20.json")["observations"])
        fixed = base.inputs(tf, spec64, args.particles, design_seed, dtype)
        theta = common(base.POINTS[point])
        h = hashlib.sha256()
        with tf.device("/CPU:0"):
            for tensor in (theta, obs, *fixed, beta):
                h.update(tf.io.serialize_tensor(tf.cast(tensor, tf.float32)).numpy())
        return theta, obs, fixed, dict(data_seed=data_seed, point=point, seed=design_seed,
                                     input_sha256=h.hexdigest())

    manifest = dict(created_utc=utc(), command=[sys.executable, *sys.argv],
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256=provenance(), tuning_source=str(source / "tuning.json"),
                    tuning_sha256=sha(source / "tuning.json"), selected_controls=selected,
                    tuning_role="frozen_FP64_comparator_not_FP32_tuning_admission",
                    plan=str(PLAN), result_file=str(RESULT), output=str(out),
                    environment=sys.executable, tensorflow=tf.__version__,
                    scope=dict(model="M13_wrapped_range_bearing", horizon=20, state_dimension=4,
                               observation_dimension=2, parameter_dimension=2, particles=args.particles,
                               chunk=select_transport_chunks(args.particles).row_chunk_size,
                               dtype=args.dtype, tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
                               jit_compile=True, cpu_only=False, analytical_score=True),
                    cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], memory_policy=memory,
                    physical_devices=[tf.config.experimental.get_device_details(gpu)
                                      for gpu in tf.config.list_physical_devices("GPU")],
                    data_sha256={str(s): sha(source / "data" / f"{s}-t20.json") for s in base.HELDOUT},
                    design_seeds=[[31001 + 100*ds + rep for rep in range(8)] for ds in range(3)],
                    input_protocol="FP64_spec_generation_then_FP32_rounding_identical_across_arithmetic_modes",
                    trust_basis="escalated_local_GPU_execution", evidence_role="arithmetic_diagnostic_only")
    base.save(out / "manifest.json", manifest)
    preflight = []
    try:
        # Cold call plus two warm calls, including the difficult parameter point.
        for index, point in enumerate((0, 1, 0)):
            theta, obs, fixed, metadata = tensors(0, point, 0)
            row = base.new_result(tf, program, theta, obs, fixed, beta)
            row.update(metadata, phase="cold_compile" if index == 0 else "warmup", warmup_index=index)
            append(out / "preflight.jsonl", row)
            preflight.append(row)
        if not all(row["valid"] for row in preflight):
            manifest["status"] = "failed_preflight"
        else:
            for ds in range(3):
                for point in range(2):
                    for rep in range(8):
                        theta, obs, fixed, metadata = tensors(ds, point, rep)
                        row = base.new_result(tf, program, theta, obs, fixed, beta)
                        row.update(metadata, phase="measurement")
                        append(out / "measurements.jsonl", row)
                        print(json.dumps(dict(particles=args.particles, mode=args.mode, **metadata,
                                              valid=row["valid"], seconds=row["wall_seconds"])), flush=True)
            manifest["status"] = "completed"
    finally:
        manifest.update(wall_seconds=time.monotonic()-start,
                        graph_trace_count=program.experimental_get_tracing_count(),
                        allocator=tf.config.experimental.get_memory_info("GPU:0"), finished_utc=utc())
        base.save(out / "manifest.json", manifest)


def rms(values):
    return math.sqrt(statistics.mean(x*x for x in values)) if values else None


def report(root):
    data, manifests, timing, invalid = {}, {}, [], []
    for n in SOURCES:
        for mode in MODES:
            directory = root / f"N{n}-{mode}"
            if not (directory / "manifest.json").exists():
                continue
            m = base.load(directory / "manifest.json")
            r = rows(directory / "measurements.jsonl")
            data[n, mode] = r
            manifests[n, mode] = m
            if len({signature(x) for x in r}) != len(r):
                raise ValueError("Duplicate measurement identity")
            valid = [x for x in r if x["valid"]]
            complete = len(r) == 48 and len(valid) == 48 and m.get("status") == "completed"
            pre = rows(directory / "preflight.jsonl")
            invalid.extend(dict(particles=n, mode=mode, **{k:x[k] for k in
                           ("data_seed", "point", "seed", "steps_completed", "phase")})
                           for x in pre+r if not x["valid"])
            timing.append(dict(particles=n, mode=mode, status=m.get("status", "running"),
                               count=len(r), valid_count=len(valid), complete=complete,
                               median_seconds=statistics.median(x["wall_seconds"] for x in valid) if complete else None,
                               min_seconds=min(x["wall_seconds"] for x in valid) if complete else None,
                               max_seconds=max(x["wall_seconds"] for x in valid) if complete else None,
                               cold_seconds=pre[0]["wall_seconds"] if pre else None,
                               warmup_seconds=[x["wall_seconds"] for x in pre[1:]],
                               wall_seconds=m.get("wall_seconds"), graph_trace_count=m.get("graph_trace_count")))
    cells = []
    for ds, seed in enumerate(base.HELDOUT):
        for point in range(2):
            def select(n, mode):
                return {x["seed"]: x for x in data.get((n, mode), [])
                        if x["data_seed"] == seed and x["point"] == point}
            lo, hi = select(1008, "fp64"), select(2016, "fp64")
            gap_valid = len(lo) == len(hi) == 8 and all(x["valid"] for x in [*lo.values(), *hi.values()])
            score_gap = rms([math.dist(lo[k]["score"], hi[k]["score"]) for k in lo]) if gap_valid else None
            value_gap = rms([lo[k]["value"]-hi[k]["value"] for k in lo]) if gap_valid else None
            for n in SOURCES:
                ref = select(n, "fp64")
                old = {signature(x): x for x in rows(SOURCES[n] / "compare.jsonl") if x["arm"] == "new"}
                for mode in ("fp32", "fp32_tf32"):
                    arm = select(n, mode)
                    valid = len(ref) == len(arm) == 8 and all(x["valid"] for x in [*ref.values(), *arm.values()])
                    cell = dict(particles=n, mode=mode, data_seed=seed, point=point, complete=valid,
                                fp64_n1008_n2016_score_change_rms=score_gap,
                                fp64_n1008_n2016_value_change_rms=value_gap)
                    if valid:
                        if set(ref) != set(arm) or any(ref[k]["input_sha256"] != arm[k]["input_sha256"] for k in ref):
                            raise ValueError("Arithmetic pairs do not share exact input hashes")
                        score_delta = [math.dist(ref[k]["score"], arm[k]["score"]) for k in ref]
                        value_delta = [abs(ref[k]["value"]-arm[k]["value"]) for k in ref]
                        sr, vr = rms(score_delta), rms(value_delta)
                        spread = math.sqrt(sum(statistics.variance(x["score"][j] for x in ref.values()) for j in range(2)))
                        sd = statistics.stdev(x["value"] for x in ref.values())
                        cell.update(score_difference_rms=sr, score_difference_mean=statistics.mean(score_delta),
                                    score_difference_max=max(score_delta), value_difference_rms=vr,
                                    value_difference_mean=statistics.mean(value_delta), value_difference_max=max(value_delta),
                                    score_change_ratio=sr/score_gap if score_gap else None,
                                    value_change_ratio=vr/value_gap if value_gap else None,
                                    fp64_score_spread=spread, fp64_value_sd=sd,
                                    score_spread_ratio=sr/spread if spread else None,
                                    value_sd_ratio=vr/sd if sd else None,
                                    original_fp64_replay_score_max=max(math.dist(ref[k]["score"], old[signature(ref[k])]["score"]) for k in ref),
                                    original_fp64_replay_value_max=max(abs(ref[k]["value"]-old[signature(ref[k])]["value"]) for k in ref))
                        cell["materiality_screen_pass"] = bool(score_gap and value_gap and sr < .1*score_gap and vr < .1*value_gap)
                    cells.append(cell)
    result = dict(created_utc=utc(), plan=str(PLAN), timings=timing, cells=cells, invalid=invalid,
                  candidate_decisions={mode:dict(
                      all_cases_complete=sum(c["complete"] for c in cells if c["mode"] == mode) == 18,
                      all_materiality_screens_pass=sum(c.get("materiality_screen_pass", False) for c in cells if c["mode"] == mode) == 18,
                      default_promoted=False, hmc_tested=False) for mode in ("fp32", "fp32_tf32")},
                  inference_status=dict(hard_veto_screen="See invalid rows and completeness; existing guards unchanged",
                                        statistically_supported_ranking="None; arithmetic sensitivity and descriptive timing only",
                                        descriptive_only="All finite-design differences, ratios and runtime comparisons",
                                        default_readiness="Not assessed; FP32 controls frozen from FP64 tuning",
                                        next_evidence="Separate scope tuning and downstream HMC before posterior/default claims"))
    base.save(root / "report.json", result)
    return result


def status(root):
    summary = []
    for n in SOURCES:
        for mode in MODES:
            directory = root / f"N{n}-{mode}"
            manifest = directory / "manifest.json"
            if not manifest.exists():
                continue
            r = rows(directory / "measurements.jsonl")
            summary.append(dict(particles=n, mode=mode, status=base.load(manifest).get("status", "running"),
                                count=len(r), valid=sum(x["valid"] for x in r)))
    return summary


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    execution = dict(command=[sys.executable, *sys.argv], started_utc=utc(), plan=str(PLAN),
                     budget_seconds=5400, attempts=[], source_sha256=provenance())
    base.save(args.output / "execution.json", execution)
    # Rotate order across N to limit systematic timing-order confounding.
    jobs = [(n, mode) for i, n in enumerate(SOURCES) for mode in MODES[i:]+MODES[:i]]
    for n, mode in jobs:
        remaining = 5400-(time.monotonic()-start)
        if remaining <= 0:
            execution["status"] = "budget_exhausted"
            break
        if provenance() != execution["source_sha256"]:
            raise RuntimeError("Scientific sources changed during the precision campaign")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--output", str(args.output),
                   "--gpu", args.gpu, "--particles", str(n), "--mode", mode]
        job_start = time.monotonic()
        attempt = dict(particles=n, mode=mode, command=command, started_utc=utc())
        with (args.output / f"N{n}-{mode}.log").open("w") as stream:
            try:
                proc = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
                attempt["exit_code"] = proc.returncode
            except subprocess.TimeoutExpired:
                attempt.update(exit_code=None, failure="wall_budget_timeout")
        attempt["wall_seconds"] = time.monotonic()-job_start
        execution["attempts"].append(attempt)
        execution["wall_seconds"] = time.monotonic()-start
        base.save(args.output / "execution.json", execution)
        print(json.dumps(attempt), flush=True)
        if attempt["exit_code"] != 0:
            execution["status"] = "infrastructure_failure_or_budget_stop"
            break
    else:
        execution["status"] = "completed"
    execution.update(finished_utc=utc(), wall_seconds=time.monotonic()-start)
    base.save(args.output / "execution.json", execution)
    report(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "status", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="GPU-68251639-fe82-8f81-3ccc-2953c32e805b")
    parser.add_argument("--particles", type=int, choices=tuple(SOURCES), default=1008)
    parser.add_argument("--mode", choices=MODES, default="fp64")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "run":
        run(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "status":
        print(json.dumps(status(args.output)))
    else:
        r = report(args.output)
        print(json.dumps(dict(timings=r["timings"], decisions=r["candidate_decisions"])))


if __name__ == "__main__":
    main()
