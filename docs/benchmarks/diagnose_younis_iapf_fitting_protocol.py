"""Diagnostic only: bounded fitting protocol selection and untouched validation.

All scientific kernels and adaptive fitting use the shared repository routines.
Host-side statistics/reporting are diagnostic; this is not an admission route.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_younis_iapf_innovation_control as previous

prior, base = previous.prior, previous.prior.base
ROOT = REPO / "docs/plans/artifacts/younis-iapf-fitting-protocol-20260919-01"
PLAN = "docs/plans/younis-iapf-fitting-protocol-2026-09-19.md"
DATASETS = {"calibration": {2000: "weak", 2001: "weak", 2010: "curved", 2011: "curved"},
            "validation": {2200: "weak", 2201: "weak", 2210: "curved", 2211: "curved"},
            "precision": {2400: "weak", 2401: "weak", 2410: "curved", 2411: "curved"},
            "challenge": {1900: "weak", 1901: "weak", 2200: "weak"}}
ARMS = {"small_early": (16, 1, 100., 4), "large_early": (256, 1, 100., 4),
        "small_stable": (16, 3, .05, 12), "large_stable": (256, 3, .05, 12),
        "reserve_stable": (1024, 3, .05, 20)}
N, T, CAL, FINAL, COPIES = 4096, 2, 96, 64, 16
GPU_UUID = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"


def choose_protocol(datasets):
    """Only calibration records may nominate a protocol; invalid arms are out."""
    if set(datasets) != {str(x) for x in DATASETS["calibration"]}:
        raise ValueError("selection requires exactly the calibration observations")
    eligible = {}
    for name in ARMS:
        if name == "small_early":
            continue
        records = [d["arms"].get(name) for d in datasets.values()]
        if all(r and r["fit"]["status"] == "valid" and "summary" in r for r in records):
            eligible[name] = statistics.mean(r["summary"]["innovation"]["score_squared_error"] for r in records)
    chosen = min(eligible, key=lambda key: (eligible[key], key)) if eligible else None
    return dict(selected=chosen, calibration_objective=eligible,
                criterion="mean_corrected_score_squared_error_four_calibration_datasets",
                heuristic_metrics_used_for_selection=False, validation_seen=False)


def campaign_budget(output, stage):
    if output.parent != ROOT or output.exists():
        raise ValueError("fresh direct child output required")
    manifests = [json.loads(p.read_text()) for p in ROOT.glob("*/manifest.json")]
    if any(m["stage"] == stage and m["status"] == "complete" for m in manifests):
        raise RuntimeError("completed stage cannot be rerun on its assessment data")
    if any(m["status"] not in ("complete", "stopped") for m in manifests):
        raise RuntimeError("unfinished launch requires recovery before retry")
    counts = {key: sum(m["budget"]["counts"][key] - m["prior_counts"][key] for m in manifests)
              for key in ("adaptive_fits", "fixed_cloud_fits", "filter_calls")}
    seconds = sum(m["wall_seconds"] for m in manifests)
    if len(manifests) >= 4 or seconds >= 1800 or counts["adaptive_fits"] >= 32 or counts["filter_calls"] >= 8000:
        raise RuntimeError("campaign budget exhausted")
    budget = base.Budget(1800 - seconds)
    budget.counts.update(counts)
    budget.limits.update(adaptive_fits=32, fixed_cloud_fits=0, filter_calls=8000)
    return budget, seconds, len(manifests) + 1


def load_selection():
    paths = [p for p in ROOT.glob("*/manifest.json") if
             (m := json.loads(p.read_text()))["stage"] == "calibration" and m["status"] == "complete"]
    if len(paths) != 1:
        raise RuntimeError("one completed calibration stage required")
    manifest = json.loads(paths[0].read_text())
    path = paths[0].parent / "selection.json"
    if previous.digest(path) != manifest["selection_sha256"]:
        raise RuntimeError("frozen selection changed")
    selection = json.loads(path.read_text())
    if selection["selected"] is None:
        raise RuntimeError("no valid selected protocol")
    return selection, path


def fit_protocol(dataset, regime, observations, back, arm, budget, seed_for):
    import tensorflow as tf
    from bayesfilter.score_study.contracts import DiagnosticFailure
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    n, k, tau, iterations = ARMS[arm]
    config = dict(base.ARMS["floor_001"], k=k, tau=tau, max_iterations=iterations, max_particles=8*n)
    c, b = base.REGIMES[regime]
    settings = dict(dimension=1, observation_dimension=1, horizon=T, particles=n,
                    dtype="float32", jit_compile=True, transition_curve=c, observation_curve=b)
    budget.charge("adaptive_fits")
    budget.charge("filter_calls", 2*iterations)
    def draw(label, replicate=None, group=None):
        return seed_for(dataset, "offline_"+(group or "final"), replicate or 0, label)
    try:
        _, final, details, calls = execute_iapf(dict(iapf=config, role="mechanics", model="nonlinear_scalar"),
            settings, tf.constant(base.THETA, tf.float32), tf.cast(observations, tf.float32), draw)
        prior.finite(*final)
        steps = [step for record in details["fit_iterations"] for step in record.get("density_fit_diagnostics", [])]
        record = dict(status="valid", fit=details["fit"], details=details, actual_calls=calls,
            reserved_calls=2*iterations, boundary_active=any(step[3] != 0 for step in steps), config=config,
            max_projected_gradient=max(step[5] for step in steps), max_fit_steps=max(step[4] for step in steps))
        record.update(base.shape_for_fit(back, record["fit"]))
        return record
    except DiagnosticFailure as error:
        return dict(status="rejected", reason=str(error), diagnostics=error.diagnostics,
                    reserved_calls=2*iterations, config=config)


def reserve_needed(datasets):
    if choose_protocol(datasets)["selected"] is not None:
        return False
    reasons = [r["fit"]["reason"] for d in datasets.values() for name, r in d["arms"].items()
               if name != "small_early" and r["fit"]["status"] == "rejected"]
    return bool(reasons) and all("cap exhausted" in r or "unconverged" in r for r in reasons)


def run(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    budget, prior_seconds, launch = campaign_budget(output, args.stage)
    selection, selection_path = load_selection() if args.stage != "calibration" else (None, None)
    precision = args.stage in ("precision", "challenge")
    challenge = args.stage == "challenge"
    arms = (["N4096", "N16384"] if precision else
            ["small_early", selection["selected"]] if selection else list(ARMS)[:4])
    particles_by_arm = {arm: int(arm[1:]) if precision else N for arm in arms}
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("pin the exact RTX5080 UUID before TensorFlow import")
    reference_path = previous.CONTINUATION_ROOT / "wide01/manifest.json"
    old_manifest = json.loads(reference_path.read_text())
    runtime_sources = {p: h for p, h in old_manifest["source_sha256"].items()
                       if p.startswith("bayesfilter/") or p.startswith("docs/benchmarks/")}
    if any(previous.digest(REPO/p) != h for p, h in runtime_sources.items()):
        raise RuntimeError("shared implementation differs from the diagnosed reference")
    paths = set(runtime_sources) | {PLAN, str(Path(__file__).relative_to(REPO))}
    hashes = {p: previous.digest(REPO/p) for p in sorted(paths)}
    output.mkdir(parents=True)
    for path in paths:
        snapshot = output / "source_snapshot" / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((REPO/path).read_bytes())
    input_paths = [reference_path] + ([selection_path] if selection_path else [])
    frozen_failures = {}
    if challenge:
        old_path = previous.CONTINUATION_ROOT/"fresh01/results.json"
        new_path = ROOT/"validation01/results.json"
        input_paths.extend([old_path, new_path])
        old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
        for dataset in (1900, 1901):
            frozen_failures[dataset] = dict(fit=old["fits"][str(dataset)],
                observations=old["references"][str(dataset)]["executed_observations"],
                source=str(old_path.relative_to(REPO)), protocol="original_small_early_frozen")
        frozen_failures[2200] = dict(fit=new["datasets"]["2200"]["arms"]["large_stable"]["fit"],
            observations=new["datasets"]["2200"]["reference"]["executed_observations"],
            source=str(new_path.relative_to(REPO)), protocol="large_stable_frozen")
    manifest = dict(schema="iapf_fitting_protocol_diagnostic_v1", stage=args.stage, status="initializing",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        command=[sys.executable, *sys.argv], environment=sys.executable, plan_file=PLAN,
        result_file=str(output/"results.json"), source_sha256=hashes,
        input_sha256={str(p.relative_to(REPO)): previous.digest(p) for p in input_paths},
        prior_counts=dict(budget.counts), prior_driver_seconds=prior_seconds, launch=launch,
        budget=budget.record(), wall_seconds=0., cuda_visible_devices=GPU_UUID,
        gpu_trust_basis="escalated_local_gpu_execution", particles=sorted(set(particles_by_arm.values())), horizon=T,
        filter_dtype="float32", tf32=True, jit_compile=True, fitting_dtype="float64",
        reference_exception="CPU FP64 quadrature; FP64 diagnostic controls and regression",
        data_version="new_common_data_observations_20260919_fitting_protocol",
        datasets=DATASETS[args.stage], arms=arms, parameters=base.THETA,
        particles_by_arm=particles_by_arm, shared_fit_across_counts=precision,
        known_failure_replay=challenge, frozen_failure_inputs={str(k): v["source"] for k, v in frozen_failures.items()},
        control_calibration_replicates=CAL, assessment_replicates=FINAL,
        default_adoption=False, ledh_admission=False, hmc_force=False)
    results = dict(datasets={}, selection=selection)
    seeds, pairs = {}, {}
    old_pairs = set()
    for p in previous.CONTINUATION_ROOT.glob("*/seeds.json"):
        old_pairs.update(tuple(pair) for group in json.loads(p.read_text()).values() for pair in group)
    for p in ROOT.glob("*/seeds.json"):
        old_pairs.update(tuple(pair) for pair in json.loads(p.read_text()).values())
    def seed_for(dataset, partition, rep, label):
        identity = f"{dataset}/{partition}/{rep}/{label}"
        pair = base.seed(dataset, label, rep, f"fitting_protocol_20260919_{output.name}_{partition}")
        key = tuple(pair)
        if key in old_pairs or (key in pairs and pairs[key] != identity):
            raise RuntimeError("seed overlap outside declared pairing")
        pairs[key] = identity
        seeds[identity] = pair
        return pair
    def save(state):
        results.update(state=state, budget=budget.record())
        manifest.update(budget=budget.record(), wall_seconds=time.monotonic()-started)
        base.write(output/"results.json", results)
        base.write(output/"seeds.json", seeds)
        base.write(output/"manifest.json", manifest)
    save("initializing")
    try:
        from bayesfilter.score_study.runtime import configure_runtime, memory_usage
        manifest["runtime"] = configure_runtime(device="GPU", tf32=True, jit_compile=True)
        devices = manifest["runtime"]["memory_policy"]["physical_devices"]
        expected = old_manifest["runtime"]["memory_policy"]["physical_devices"]
        if len(devices) != 1 or devices[0]["device_details"] != expected[0]["device_details"]:
            raise RuntimeError("physical GPU differs from reference")
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
        from bayesfilter.score_study.innovation_controls_tf import make_innovation_control_kernel
        from bayesfilter.score_study.combinations_tf import make_combination_kernels
        from bayesfilter.score_study.nonlinear_tf import make_moment_filter, make_particle_filter
        prior.tf = base.tf = tf
        theta = tf.constant(base.THETA, tf.float32)
        controls_by_n = {n: make_innovation_control_kernel(n, T+1, 1, COPIES) for n in set(particles_by_arm.values())}
        combinations = {k: make_combination_kernels(6, k, "float64", control_input_dtype_name="float32") for k in (12, 18)}
        critical = float(tfp.distributions.StudentT(tf.constant(FINAL-1., tf.float64), 0., 1.).quantile(
            tf.constant(1.-.01/48., tf.float64)))
        manifest["bias_critical"] = critical

        def make_streams(n):
            @tf.function(input_signature=[tf.TensorSpec([6, 2], tf.int32)], jit_compile=True)
            def streams(seed):
                draws = (tf.random.stateless_normal([n, 1], seed[0]), tf.random.stateless_normal([T, n, 1], seed[1]),
                         tf.cast(tf.random.stateless_uniform([T+1, n], seed[2], 0, 2**23, tf.int32), tf.float32)*2.**-23,
                         tf.random.stateless_uniform([T, n], seed[3]))
                references = tf.concat([tf.random.stateless_normal([COPIES, n, 1], seed[4])[:, None],
                                        tf.random.stateless_normal([COPIES, T, n, 1], seed[5])], 1)
                return draws, references
            return streams
        streams_by_n = {n: make_streams(n) for n in controls_by_n}

        @tf.function(input_signature=[tf.TensorSpec([FINAL], tf.float64)], jit_compile=True)
        def interval(delta):
            indices = tf.random.stateless_uniform([40000, FINAL], [919, 7201], 0, FINAL, tf.int32)
            means = tf.sort(tf.reduce_mean(tf.gather(delta, indices), 1))
            return tf.stack([means[49], means[39949]])

        def get_streams(dataset, partition, rep, n=N):
            return streams_by_n[n](tf.constant([seed_for(dataset, partition, rep, label) for label in previous.LABELS], tf.int32))

        def metrics(scores, reference):
            means = [statistics.mean(column) for column in zip(*scores)]
            se = [statistics.stdev(column)/math.sqrt(FINAL) for column in zip(*scores)]
            errors = [a-b for a, b in zip(means, reference["score"])]
            intervals = [[e-critical*s-5e-6*(1+abs(t)), e+critical*s+5e-6*(1+abs(t))]
                         for e, s, t in zip(errors, se, reference["score"])]
            return dict(mean_score=means, score_standard_error=se, mean_score_error=errors,
                score_squared_error=statistics.mean(prior.squared_error(s, reference) for s in scores),
                total_score_variance=sum(statistics.variance(column) for column in zip(*scores)),
                mean_error_intervals=intervals, bias_screen_pass=all(lo <= 0 <= hi for lo, hi in intervals))

        prepared, frozen, proposal_cache = {}, {}, {}
        for dataset, regime in DATASETS[args.stage].items():
            c, b = base.REGIMES[regime]
            observations, back, reference = base.reference(dataset, regime)
            if challenge and reference["executed_observations"] != frozen_failures[dataset]["observations"]:
                raise RuntimeError("failure replay observations changed")
            obs = tf.cast(observations, tf.float32)
            record = dict(regime=regime, reference=reference, arms={}, deterministic={}, no_resampling=[])
            results["datasets"][str(dataset)] = record
            prepared[dataset] = (observations, back, obs)
            for name in ("ekf", "ukf"):
                moment = make_moment_filter(T, c, b, name, "float32")
                score = []
                for direction in tf.unstack(tf.eye(6)):
                    budget.charge("filter_calls")
                    out = moment(theta, direction, obs)
                    prior.finite(*out)
                    score.append(float(out[1]))
                record["deterministic"][name] = dict(score=score, score_squared_error=prior.squared_error(score, reference))

        def prepare_arm(dataset, arm):
            record = results["datasets"][str(dataset)]
            regime = record["regime"]
            observations, back, obs = prepared[dataset]
            protocol = frozen_failures[dataset]["protocol"] if challenge else selection["selected"] if precision else arm
            n = particles_by_arm.get(arm, N)
            fitting_key = (dataset, protocol)
            if fitting_key not in proposal_cache:
                proposal_cache[fitting_key] = (frozen_failures[dataset]["fit"] if challenge else
                    fit_protocol(dataset, regime, observations, back, protocol, budget, seed_for))
            fitted = proposal_cache[fitting_key]
            item = dict(fit=fitted, protocol=protocol, final_particles=n, calibration=[], assessment=[], regression={})
            record["arms"][arm] = item
            save(f"fit_{dataset}_{arm}")
            if fitted["status"] != "valid":
                print(f"fit rejected {dataset} {arm}: {fitted['reason']}", flush=True)
                return
            c, b = base.REGIMES[regime]
            kernel = make_fitted_twist_kernel(1, 1, n, T, "float32", transition_curve=c, observation_curve=b,
                include_fisher_score=True, include_resampling_controls=True, resampling_uniform_bits=23)
            coefficients = tuple(tf.constant(fitted["fit"][name], tf.float32) for name in ("centers", "covariances", "log_floors"))
            calibration_started = time.monotonic()
            for rep in range(CAL):
                draws, refs = get_streams(dataset, "control_calibration", rep, n)
                budget.charge("filter_calls")
                out, innovation = previous.observe(kernel, controls_by_n[n], theta, obs, draws, coefficients, refs)
                prior.finite(*out, innovation)
                if any(not value.device.endswith("GPU:0") for value in (out[0], out[3], out[4], innovation)):
                    raise RuntimeError("particle/control output not on GPU:0")
                item["actual_tensor_devices"] = [value.device for value in (out[0], out[3], out[4], innovation)]
                item["calibration"].append(dict(score=base.materialize(out[3]),
                    ancestor=base.materialize(tf.reshape(out[4], [12])), innovation=base.materialize(innovation)))
            item["calibration_seconds_including_compile"] = time.monotonic()-calibration_started
            raw = tf.constant([r["score"] for r in item["calibration"]], tf.float64)
            for name, size in (("ancestor", 12), ("innovation", 18)):
                matrix = tf.constant([r["ancestor"]+(r["innovation"] if size == 18 else []) for r in item["calibration"]], tf.float64)
                coefficient, rank, valid = combinations[size][0](raw, matrix)
                prior.finite(coefficient)
                if not bool(valid):
                    raise RuntimeError("control regression invalid")
                item["regression"][name] = dict(coefficient=base.materialize(coefficient), rank=int(rank))
            frozen[(dataset, arm)] = (kernel, coefficients)
            item["frozen_before_assessment"] = True
            save(f"frozen_{dataset}_{arm}")
            print(f"fit/calibration frozen {dataset} {arm}, N={fitted['fit']['particles']}", flush=True)

        def assess_arm(dataset, arm):
            record = results["datasets"][str(dataset)]
            item = record["arms"][arm]
            if item["fit"]["status"] != "valid":
                return
            kernel, coefficients = frozen[(dataset, arm)]
            obs = prepared[dataset][2]
            n = item["final_particles"]
            assessment_started = time.monotonic()
            for rep in range(FINAL):
                draws, refs = get_streams(dataset, "assessment", rep, n)
                budget.charge("filter_calls")
                out, innovation = previous.observe(kernel, controls_by_n[n], theta, obs, draws, coefficients, refs)
                prior.finite(*out, innovation)
                if any(not value.device.endswith("GPU:0") for value in (out[0], out[3], out[4], innovation)):
                    raise RuntimeError("assessment output not on GPU:0")
                item["assessment"].append(dict(value=float(out[0]), score=base.materialize(out[3]),
                    fixed_score=base.materialize(out[1]), ancestor=base.materialize(tf.reshape(out[4], [12])),
                    innovation=base.materialize(innovation)))
            item["assessment_seconds"] = time.monotonic()-assessment_started
            item["particle_time_work"] = dict(calibration=CAL*n*T, assessment=FINAL*n*T)
            raw = tf.constant([r["score"] for r in item["assessment"]], tf.float64)
            scores = {"raw_fisher": base.materialize(raw)}
            for name, size in (("ancestor", 12), ("innovation", 18)):
                matrix = tf.constant([r["ancestor"]+(r["innovation"] if size == 18 else []) for r in item["assessment"]], tf.float64)
                corrected = combinations[size][1](raw, matrix, tf.constant(item["regression"][name]["coefficient"], tf.float64))
                prior.finite(corrected)
                scores[name] = base.materialize(corrected)
            item["scores"] = scores
            item["summary"] = {name: metrics(values, record["reference"]) for name, values in scores.items()}
            item["kernel_trace_count"] = kernel.experimental_get_tracing_count()
            if item["kernel_trace_count"] != 1:
                raise RuntimeError("filter retraced")
            save(f"assessed_{dataset}_{arm}")
            print(f"assessed {dataset} {arm}: MSE={item['summary']['innovation']['score_squared_error']:.8g}", flush=True)

        for dataset in DATASETS[args.stage]:
            for arm in arms:
                prepare_arm(dataset, arm)
        save("all_initial_control_fits_frozen")
        (output/"frozen-calibration.json").write_bytes((output/"results.json").read_bytes())
        manifest["frozen_calibration_sha256"] = previous.digest(output/"frozen-calibration.json")
        for dataset in DATASETS[args.stage]:
            for arm in arms:
                assess_arm(dataset, arm)
        if args.stage == "calibration" and reserve_needed(results["datasets"]):
            results["reserve_trigger"] = "all_repaired_arms_invalid_due_to_fit_caps"
            for dataset in DATASETS[args.stage]:
                prepare_arm(dataset, "reserve_stable")
            save("all_reserve_control_fits_frozen")
            (output/"frozen-reserve.json").write_bytes((output/"results.json").read_bytes())
            manifest["frozen_reserve_sha256"] = previous.digest(output/"frozen-reserve.json")
            for dataset in DATASETS[args.stage]:
                assess_arm(dataset, "reserve_stable")
        if args.stage == "calibration":
            results["selection"] = choose_protocol(results["datasets"])
            base.write(output/"selection.json", results["selection"])
            manifest["selection_sha256"] = previous.digest(output/"selection.json")
        # Heuristics are computed for every situation but never used for selection.
        for dataset, regime in DATASETS[args.stage].items():
            record = results["datasets"][str(dataset)]
            c, b = base.REGIMES[regime]
            record["no_resampling_by_particles"] = {}
            for n in sorted(set(particles_by_arm.values())):
                plain = make_particle_filter(n, T, c, b, resampling=False, dtype_name="float32")
                scores = []
                for rep in range(FINAL):
                    draws, _ = get_streams(dataset, "assessment", rep, n)
                    budget.charge("filter_calls")
                    out = plain(theta, prepared[dataset][2], draws[0], draws[1], draws[2][1:])
                    prior.finite(*out)
                    scores.append(base.materialize(out[1]))
                record["no_resampling_by_particles"][str(n)] = dict(scores=scores, summary=metrics(scores, record["reference"]))
            record["no_resampling"] = record["no_resampling_by_particles"][str(N)]["scores"]
            record["no_resampling_summary"] = record["no_resampling_by_particles"][str(N)]["summary"]
            for item in record["arms"].values():
                if "summary" not in item:
                    continue
                heuristic_mse = {k: v["score_squared_error"] for k, v in record["deterministic"].items()}
                heuristic_mse["no_resampling"] = record["no_resampling_by_particles"][str(item["final_particles"])]["summary"]["score_squared_error"]
                mse = item["summary"]["innovation"]["score_squared_error"]
                losses = [name for name, value in heuristic_mse.items() if mse > value]
                item["heuristic_dominance"] = dict(pass_screen=not losses, observed_losses=losses)
            if args.stage in ("validation", "precision", "challenge"):
                candidate = record["arms"]["N16384" if precision else selection["selected"]]
                baseline = record["arms"]["N4096" if precision else "small_early"]
                if "scores" in candidate and "scores" in baseline:
                    delta = [prior.squared_error(a, record["reference"])-prior.squared_error(b, record["reference"])
                             for a, b in zip(candidate["scores"]["innovation"], baseline["scores"]["innovation"])]
                    ci = base.materialize(interval(tf.constant(delta, tf.float64)))
                    record["primary_comparison"] = dict(mean_difference=statistics.mean(delta),
                        paired_interval_99_75=ci, pass_screen=ci[1] < 0)
                else:
                    record["primary_comparison"] = dict(pass_screen=False, reason="invalid_fit")
            save(f"heuristics_{dataset}")
        if args.stage in ("validation", "precision", "challenge"):
            chosen = [d["arms"]["N16384" if precision else selection["selected"]] for d in results["datasets"].values()]
            results["decision"] = dict(primary_pass_count=sum(d["primary_comparison"]["pass_screen"] for d in results["datasets"].values()),
                primary_total=len(results["datasets"]), all_fits_valid=all(r["fit"]["status"] == "valid" for r in chosen),
                heuristic_dominance_pass=all(r.get("heuristic_dominance", {}).get("pass_screen", False) for r in chosen),
                bias_screens_pass=all(r.get("summary", {}).get("innovation", {}).get("bias_screen_pass", False) for r in chosen),
                boundary_veto=any(r["fit"].get("boundary_active", True) for r in chosen),
                default_adoption=False, ledh_admission=False, hmc_force=False)
        manifest["trace_counts"] = dict(fit12=combinations[12][0].experimental_get_tracing_count(), fit18=combinations[18][0].experimental_get_tracing_count(),
            apply12=combinations[12][1].experimental_get_tracing_count(), apply18=combinations[18][1].experimental_get_tracing_count())
        manifest["trace_counts"].update({f"streams_{n}": kernel.experimental_get_tracing_count() for n, kernel in streams_by_n.items()})
        manifest["trace_counts"].update({f"controls_{n}": kernel.experimental_get_tracing_count() for n, kernel in controls_by_n.items()})
        if any(n != 1 for n in manifest["trace_counts"].values()):
            raise RuntimeError("unexpected kernel retracing")
        if any(previous.digest(REPO/p) != h for p, h in hashes.items()):
            raise RuntimeError("source drift during run")
        manifest.update(status="complete", terminal_source_sha256_match=True, memory=memory_usage("GPU"),
            unique_seed_pairs=len(pairs), seed_pairing="same_dataset_across_arms_or_particle_counts_only", seeds_file="seeds.json")
        save("complete")
        print(json.dumps(dict(status="complete", selection=results.get("selection"), decision=results.get("decision"), budget=budget.record())), flush=True)
    except BaseException as error:
        manifest.update(status="stopped", error=repr(error))
        save("stopped")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibration", "validation", "precision", "challenge"), required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
