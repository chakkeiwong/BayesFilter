"""Diagnostic only: fresh same-law innovation controls under bounded plans.

This observer calls the existing iAPF endpoint once and consumes its actual
input innovations. All control fits are frozen before any final replication.
No runtime default, fitted proposal, or underlying particle law is changed.
"""
from __future__ import annotations

import argparse
import hashlib
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
import diagnose_younis_iapf_resampling_control as prior

ROOT = REPO / "docs/plans/artifacts/younis-iapf-resampling-control-20260918-01"
PLAN = "docs/plans/younis-iapf-innovation-control-2026-09-18.md"
CONTINUATION_ROOT = REPO / "docs/plans/artifacts/younis-iapf-pinned-continuation-20260919-01"
CONTINUATION_PLAN = "docs/plans/younis-iapf-pinned-continuation-2026-09-19.md"
FRESH_DATASETS = {1900: "weak", 1901: "weak", 1910: "curved", 1911: "curved"}
N, T, CALIBRATION, FINAL, COPIES = 4096, 2, 96, 64, 16
LABELS = ("initial", "process", "ancestors", "mixture", "reference_initial", "reference_process")


def observe(filter_kernel, control_kernel, theta, observations, draws, coefficients, references):
    import tensorflow as tf
    output = filter_kernel(theta, observations, *draws, *coefficients)
    # Input packing only: numerical moment/reduction work is inside the kernel.
    noise = tf.concat([draws[0][None], draws[1]], axis=0)
    return output, control_kernel(noise, references)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_gpu_uuid(manifest):
    """Reuse the actual prior physical device, never a reordered ordinal."""
    identifier = manifest.get("cuda_visible_devices")
    if not isinstance(identifier, str) or not identifier.startswith("GPU-") or "," in identifier:
        raise ValueError("reference must identify one physical GPU UUID")
    return identifier


def input_records():
    parent = ROOT / "conditioning-confirmation01"
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("prior confirmation incomplete")
    changed = [path for path, value in manifest["source_sha256"].items()
               if digest(REPO / path) != value]
    if changed:
        raise RuntimeError("source changed since checked prior comparison: " + str(changed))
    return (manifest, json.loads((ROOT / "attempt02/results.json").read_text()),
            json.loads((parent / "results.json").read_text()))


def seed_inventory(stage=None, attempt=None):
    used, fresh = set(), {}
    def reserve(dataset, rep, partition, group, labels, record=False):
        pairs = [prior.base.seed(dataset, label, rep, group) for label in labels]
        for pair in pairs:
            if tuple(pair) in used:
                raise RuntimeError("seed collision or calibration/final overlap")
            used.add(tuple(pair))
        if record:
            fresh[f"{dataset}/{partition}/{rep}"] = pairs
    for dataset, regime in prior.DATASETS.items():
        for n in ((4096,) if regime == "affine" else (1024, 4096)):
            for partition, count in (("calibration", 192), ("final", 128)):
                for rep in range(count):
                    reserve(dataset, rep, partition, f"resampling_control_20260918_{partition}_N{n}", LABELS[:4])
        for partition, count in (("calibration", 192), ("final", 128)):
            for rep in range(count):
                reserve(dataset, rep, partition, f"conditioning_confirmation_20260918_{partition}_N4096", LABELS[:4])
    old_count = len(used)
    for dataset in prior.DATASETS:
        for partition, count in (("calibration", CALIBRATION), ("final", FINAL)):
            for rep in range(count):
                reserve(dataset, rep, partition, f"innovation_confirmation_20260918_{partition}_N4096", LABELS, stage is None)
    if stage is not None:
        for path in sorted(CONTINUATION_ROOT.glob("*/seeds.json")):
            for pairs in json.loads(path.read_text()).values():
                for pair in pairs:
                    if tuple(pair) in used:
                        raise RuntimeError("previous continuation seed overlap")
                    used.add(tuple(pair))
        old_count = len(used)
        datasets = prior.DATASETS if stage == "pinned" else FRESH_DATASETS
        for dataset in datasets:
            for partition, count in (("calibration", CALIBRATION), ("final", FINAL)):
                for rep in range(count):
                    reserve(dataset, rep, partition,
                            f"innovation_continuation_20260919_{attempt}_{partition}_N4096", LABELS, True)
    return fresh, dict(old_seed_pairs=old_count, new_seed_pairs=len(used)-old_count,
                       all_particle_and_reference_seed_pairs_disjoint=True)


def continuation_budget(output, stage, started):
    if output.parent != CONTINUATION_ROOT or output.exists():
        raise RuntimeError("continuation requires a fresh directory in the declared root")
    manifests = [json.loads(path.read_text()) for path in sorted(CONTINUATION_ROOT.glob("*/manifest.json"))]
    if len(manifests) >= 4:
        raise RuntimeError("continuation launch budget exhausted")
    stages = {record.get("continuation_stage") for record in manifests if record["status"] == "complete"}
    if stage in stages:
        raise RuntimeError("stage already completed; no final-driven repeat")
    if stage in ("fresh", "wide") and "pinned" not in stages:
        raise RuntimeError("pinned stage must complete first")
    if stage == "wide":
        completed = [record for record in manifests
                     if record.get("continuation_stage") == "fresh" and record["status"] == "complete"]
        if not completed or not json.loads(Path(completed[-1]["result_file"]).read_text()).get("bound_repair_trigger"):
            raise RuntimeError("wide stage requires the predeclared offline bound trigger")
    counts = {"adaptive_fits": 0, "fixed_cloud_fits": 0, "filter_calls": 0}
    for record in manifests:
        charged = record["budget"]["counts"]
        initial = record["prior_budget"]
        for kind in counts:
            counts[kind] += charged[kind]-initial[kind]
    seconds = sum(record["wall_seconds"] for record in manifests)
    expected = 1116 if stage == "pinned" else 976
    if counts["filter_calls"]+expected > 8000 or counts["adaptive_fits"]+(0 if stage == "pinned" else 4) > 8:
        raise RuntimeError("insufficient remaining campaign calls or fits")
    if seconds >= 1800:
        raise RuntimeError("continuation wall budget exhausted")
    budget = prior.base.Budget(min(600., 1800.-seconds))
    budget.started = started
    budget.limits = {"adaptive_fits": 8, "fixed_cloud_fits": 0, "filter_calls": 8000}
    budget.counts.update(counts)
    return budget, seconds, len(manifests)+1, expected


def fresh_fit(dataset, regime, observations, back, stage, budget):
    import tensorflow as tf
    from bayesfilter.score_study.contracts import DiagnosticFailure
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    config = dict(prior.base.ARMS["floor_001"])
    if stage == "wide":
        config.update(mean_bound=8., sd_lower=.1, sd_upper=8.)
    c, b = prior.base.REGIMES[regime]
    settings = dict(dimension=1, observation_dimension=1, horizon=T, particles=16,
        dtype="float32", jit_compile=True, transition_curve=c, observation_curve=b)

    def draw(label, replicate=None, group=None):
        return prior.base.seed(dataset, label, replicate or 0,
                               "pinned_continuation_fit_20260919_"+(group or "independent_final"))

    budget.charge("adaptive_fits")
    budget.charge("filter_calls", 8)
    try:
        _, final, details, calls = execute_iapf(dict(iapf=config, role="mechanics", model="nonlinear_scalar"),
            settings, tf.constant(prior.base.THETA, tf.float32), tf.cast(observations, tf.float32), draw)
        prior.finite(*final)
        steps = [step for record in details["fit_iterations"]
                 for step in record.get("density_fit_diagnostics", [])]
        record = dict(status="valid", fit=details["fit"], details=details, actual_calls=calls,
            boundary_active=any(step[3] != 0 for step in steps),
            max_projected_gradient=max(step[5] for step in steps),
            max_fit_steps=max(step[4] for step in steps), config=config)
        record.update(prior.base.shape_for_fit(back, record["fit"]))
        return record
    except DiagnosticFailure as error:
        return dict(status="rejected", reason=str(error), diagnostics=error.diagnostics, config=config)


def run(args):
    started = time.monotonic()
    old_manifest, old, safety = input_records()
    if "tensorflow" in sys.modules:
        raise RuntimeError("GPU identity must be pinned before TensorFlow import")
    os.environ["CUDA_VISIBLE_DEVICES"] = reference_gpu_uuid(old_manifest)
    stage = args.continuation_stage
    output = Path(args.output).resolve()
    seeds, seed_check = seed_inventory(stage, output.name)
    datasets = prior.DATASETS if stage in (None, "pinned") else FRESH_DATASETS
    if stage is None:
        if output.parent != ROOT or len(list(ROOT.glob("*/manifest.json"))) != 3:
            raise RuntimeError("expected the final launch of the existing four-launch campaign")
        prior_seconds = old_manifest["prior_driver_seconds"] + old_manifest["wall_seconds"]
        budget = prior.base.Budget(min(600., 1800.-prior_seconds))
        budget.started = started
        budget.limits = old_manifest["budget"]["limits"]
        budget.counts.update(old_manifest["budget"]["counts"])
        if budget.counts["filter_calls"] != 6884:
            raise RuntimeError("unexpected starting call budget")
        plan, launch_number, expected_calls = PLAN, 4, 1116
    else:
        budget, prior_seconds, launch_number, expected_calls = continuation_budget(output, stage, started)
        plan = CONTINUATION_PLAN
    output.mkdir(parents=True, exist_ok=False)
    initial_calls = budget.counts["filter_calls"]
    paths = set(old_manifest["source_sha256"]) | {
        plan, str(Path(__file__).relative_to(REPO)),
        "bayesfilter/score_study/innovation_controls_tf.py"}
    hashes = {path: digest(REPO / path) for path in sorted(paths)}
    manifest = dict(schema="iapf_innovation_control_diagnostic_v2", plan=plan,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        command=sys.argv, cwd=str(REPO), python=sys.executable, python_version=sys.version,
        output=str(output), result_file=str(output / "results.json"), source_sha256=hashes,
        input_artifacts_sha256={str(path.relative_to(REPO)): digest(path) for path in (
            ROOT/"attempt02/results.json", ROOT/"conditioning-confirmation01/results.json",
            ROOT/"conditioning-confirmation01/manifest.json")},
        prior_driver_seconds=prior_seconds, prior_budget=dict(budget.counts), status="initializing",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_trust_basis="escalated_local_gpu_execution", datasets=datasets,
        continuation_stage=stage, campaign_launch_number=launch_number, expected_new_filter_calls=expected_calls,
        required_gpu_uuid=reference_gpu_uuid(old_manifest),
        theta=old_manifest["theta"], particles=N, horizon=T, calibration_replicates=CALIBRATION,
        final_replicates=FINAL, independent_reference_batches=COPIES, master_seed=9182026,
        data_version=("same_fixed_five_datasets_and_fits_as_resampling_control_20260918"
                      if stage in (None, "pinned") else "fresh_1900_1901_1910_1911_common_data"),
        seeds_file="seeds.json", seed_check=seed_check, filter_dtype="float32",
        control_dtype="float64", control_input_dtype="float32", jit_compile=True,
        reference_exception="CPU FP64 quadrature and FP64 diagnostic controls/regression",
        default_adoption=False, calibration_uses_oracle=False)
    results = dict(datasets={}, references={}, fits={})
    for path in hashes:
        snapshot = output / "source_snapshot" / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((REPO / path).read_bytes())
    prior.base.write(output / "manifest.json", manifest)
    prior.base.write(output / "seeds.json", seeds)
    def save(stage):
        results.update(stage=stage, budget=budget.record())
        prior.base.write(output / "results.json", results)

    try:
        from bayesfilter.score_study.runtime import configure_runtime, memory_usage
        manifest["runtime"] = configure_runtime(device="GPU", tf32=True, jit_compile=True)
        devices = manifest["runtime"]["memory_policy"]["physical_devices"]
        expected_devices = old_manifest["runtime"]["memory_policy"]["physical_devices"]
        if len(devices) != 1 or devices[0]["device_details"] != expected_devices[0]["device_details"]:
            raise RuntimeError("visible GPU differs from the pinned reference device")
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.score_study.combinations_tf import make_combination_kernels
        from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
        from bayesfilter.score_study.gaussian_tf import make_gaussian_kernel, parameterized_model
        from bayesfilter.score_study.innovation_controls_tf import make_innovation_control_kernel
        from bayesfilter.score_study.nonlinear_tf import make_particle_filter, make_moment_filter
        prior.tf = prior.base.tf = tf
        prior.base.THETA = old_manifest["theta"]
        prior.base.REGIMES["affine"] = (0., 0.)
        theta = tf.constant(prior.base.THETA, tf.float32)
        wide_baseline = None
        if stage == "wide":
            for path in sorted(CONTINUATION_ROOT.glob("*/manifest.json")):
                candidate = json.loads(path.read_text())
                if candidate.get("continuation_stage") == "fresh" and candidate["status"] == "complete":
                    source = Path(candidate["result_file"])
                    wide_baseline = json.loads(source.read_text())
                    manifest["input_artifacts_sha256"][str(source.relative_to(REPO))] = digest(source)

        @tf.function(input_signature=[tf.TensorSpec([6, 2], tf.int32)], jit_compile=True)
        def stream_kernel(seed):
            draws = (tf.random.stateless_normal([N, 1], seed[0]),
                tf.random.stateless_normal([T, N, 1], seed[1]),
                tf.cast(tf.random.stateless_uniform([T+1, N], seed[2], minval=0, maxval=2**23,
                    dtype=tf.int32), tf.float32) * tf.constant(2.**-23),
                tf.random.stateless_uniform([T, N], seed[3]))
            references = tf.concat([
                tf.random.stateless_normal([COPIES, N, 1], seed[4])[:, None],
                tf.random.stateless_normal([COPIES, T, N, 1], seed[5])], axis=1)
            return draws, references

        @tf.function(input_signature=[tf.TensorSpec([FINAL], tf.float64)], jit_compile=True)
        def intervals(difference):
            indices = tf.random.stateless_uniform([40000, FINAL], [918, 7141],
                minval=0, maxval=FINAL, dtype=tf.int32)
            means = tf.sort(tf.reduce_mean(tf.gather(difference, indices), 1))
            return tf.stack([means[49], means[39949], means[199], means[39799]])

        controls = make_innovation_control_kernel(N, T+1, 1, COPIES)
        combinations = {k: make_combination_kernels(6, k, "float64", control_input_dtype_name="float32")
                        for k in (12, 18)}
        critical = float(tfp.distributions.StudentT(tf.constant(FINAL-1., tf.float64), 0., 1.).quantile(
            tf.constant(1.-.01/60., tf.float64)))
        manifest["bias_interval_critical"] = critical
        prepared = {}
        for dataset, regime in datasets.items():
            key = str(dataset)
            c, b = prior.base.REGIMES[regime]
            observations, back, reference = prior.base.reference(dataset, regime)
            results["references"][key] = reference
            if stage in (None, "pinned") and reference["executed_observations"] != old["references"][key]["executed_observations"]:
                raise RuntimeError("fixed observations changed")
            fitted = (old["fits"][key] if stage in (None, "pinned") else
                      fresh_fit(dataset, regime, observations, back, stage, budget))
            results["fits"][key] = fitted
            save(f"proposal_fitted_{dataset}")
            if fitted["status"] != "valid":
                raise RuntimeError(f"proposal fit invalid for {dataset}: {fitted.get('reason')}")
            if stage in ("fresh", "wide"):
                fit_seeds = fitted["details"]["fit_seed_records"]
                reserved = {tuple(pair) for pairs in seeds.values() for pair in pairs}
                if any(tuple(pair) in reserved for pair in fit_seeds.values()):
                    raise RuntimeError("proposal fit seed overlaps score calibration/final")
                if len({tuple(pair) for pair in fit_seeds.values()}) != len(fit_seeds):
                    raise RuntimeError("proposal fitting stream collision")
                fitted["fit_seed_disjoint_from_score_streams"] = True
            if wide_baseline is not None:
                baseline = wide_baseline["fits"][key]
                fit_diff = max(float(tf.reduce_max(tf.abs(tf.constant(fitted["fit"][name], tf.float64)-
                    tf.constant(baseline["fit"][name], tf.float64))/(1+tf.abs(tf.constant(baseline["fit"][name], tf.float64)))))
                    for name in ("centers", "covariances", "log_floors"))
                old_fit_seeds = baseline["details"]["fit_seed_records"]
                if any(fit_seeds[name] != pair for name, pair in old_fit_seeds.items() if name in fit_seeds):
                    raise RuntimeError("paired fitting seed mismatch")
                fitted["bound_sensitivity"] = dict(baseline_boundary=baseline["boundary_active"],
                    relative_coefficient_discrepancy=fit_diff,
                    interior_nonharm_pass=baseline["boundary_active"] or fit_diff <= 2e-4,
                    shared_offline_streams_verified=True)
            obs = tf.cast(observations, tf.float32)
            fit = tuple(tf.constant(fitted["fit"][name], tf.float32)
                        for name in ("centers", "covariances", "log_floors"))
            kernel = make_fitted_twist_kernel(1, 1, N, T, "float32", transition_curve=c,
                observation_curve=b, include_fisher_score=True, include_resampling_controls=True,
                resampling_uniform_bits=23)
            record = dict(regime=regime, calibration=[], final=[], deterministic={}, regression={})
            results["datasets"][key] = record
            if regime == "affine":
                with tf.device("/CPU:0"):
                    exact = make_gaussian_kernel(1, 1, 6)(observations,
                        *parameterized_model(tf.constant(prior.base.THETA, tf.float64), 1, 1))
                prior.finite(*exact)
                if max(abs(a-z) for a, z in zip(prior.base.materialize(exact[1]), reference["score"])) > 1e-7:
                    raise RuntimeError("Kalman reference disagreement")
                record["deterministic"]["kalman"] = dict(score=prior.base.materialize(exact[1]),
                    score_squared_error=prior.squared_error(prior.base.materialize(exact[1]), reference))
            for name in ("ekf", "ukf"):
                moment = make_moment_filter(T, c, b, name, "float32")
                score = []
                for direction in tf.unstack(tf.eye(6)):
                    budget.charge("filter_calls")
                    out = moment(theta, direction, obs)
                    prior.finite(*out)
                    score.append(float(out[1]))
                record["deterministic"][name] = dict(score=score, score_squared_error=prior.squared_error(score, reference))
            for rep in range(CALIBRATION):
                draws, references = stream_kernel(tf.constant(seeds[f"{dataset}/calibration/{rep}"], tf.int32))
                budget.charge("filter_calls")
                out, innovation = observe(kernel, controls, theta, obs, draws, fit, references)
                prior.finite(*out, innovation)
                if any(not value.device.endswith("GPU:0") for value in (out[0], out[3], out[4], innovation)):
                    raise RuntimeError("particle or control kernel not on pinned GPU:0")
                record["actual_tensor_devices"] = dict(particle_value=out[0].device,
                    fisher_score=out[3].device, ancestor_controls=out[4].device,
                    innovation_controls=innovation.device)
                record["calibration"].append(dict(score=prior.base.materialize(out[3]),
                    ancestor=prior.base.materialize(tf.reshape(out[4], [12])),
                    innovation=prior.base.materialize(innovation)))
            scores = tf.constant([row["score"] for row in record["calibration"]], tf.float64)
            for name, k in (("ancestor_fresh", 12), ("innovation", 18)):
                columns = tf.constant([row["ancestor"]+(row["innovation"] if k == 18 else [])
                                       for row in record["calibration"]], tf.float64)
                coefficient, rank, valid = combinations[k][0](scores, columns)
                prior.finite(coefficient)
                if not bool(valid):
                    raise RuntimeError("control regression invalid")
                singular = tf.linalg.svd(columns-tf.reduce_mean(columns, 0), compute_uv=False)
                record["regression"][name] = dict(coefficient=prior.base.materialize(coefficient), rank=int(rank),
                    singular_values=prior.base.materialize(singular),
                    max_coefficient=float(tf.reduce_max(tf.abs(coefficient))), frozen_before_final=True)
            if stage in (None, "pinned"):
                record["regression"]["ancestor_previous"] = dict(
                    coefficient=safety["datasets"][key]["protected_coefficient"],
                    rank=safety["datasets"][key]["protected_rank"], calibration_replicates=192,
                    source="conditioning-confirmation01/results.json", frozen_before_final=True)
            prepared[key] = (kernel, obs, fit)
            save(f"calibration_frozen_{dataset}")
            print(f"calibration frozen: dataset {dataset}", flush=True)
        results["bound_repair_trigger"] = any(fit["boundary_active"] for fit in results["fits"].values())
        results["bound_trigger_frozen_before_final"] = True
        save("all_calibrations_and_bound_trigger_frozen")
        frozen_hash = digest(output / "results.json")
        manifest.update(status="final_running", all_coefficients_frozen_before_any_final=True,
                        frozen_calibration_sha256=frozen_hash)
        (output / "frozen-calibration.json").write_bytes((output / "results.json").read_bytes())
        prior.base.write(output / "manifest.json", manifest)

        for dataset, regime in datasets.items():
            key = str(dataset)
            c, b = prior.base.REGIMES[regime]
            kernel, obs, fit = prepared[key]
            record, reference = results["datasets"][key], results["references"][key]
            plain = make_particle_filter(N, T, c, b, resampling=False, dtype_name="float32")
            for rep in range(FINAL):
                draws, references = stream_kernel(tf.constant(seeds[f"{dataset}/final/{rep}"], tf.int32))
                budget.charge("filter_calls")
                out, innovation = observe(kernel, controls, theta, obs, draws, fit, references)
                prior.finite(*out, innovation)
                if any(not value.device.endswith("GPU:0") for value in (out[0], out[3], out[4], innovation)):
                    raise RuntimeError("final particle or control kernel not on pinned GPU:0")
                row = dict(value=float(out[0]), score=prior.base.materialize(out[3]),
                    fixed_score=prior.base.materialize(out[1]), ancestor=prior.base.materialize(tf.reshape(out[4], [12])),
                    innovation=prior.base.materialize(innovation))
                if regime != "affine":
                    budget.charge("filter_calls")
                    other = plain(theta, obs, draws[0], draws[1], draws[2][1:])
                    prior.finite(*other)
                    row["no_resampling"] = prior.base.materialize(other[1])
                    row["no_resampling_ess"] = float(other[2])
                record["final"].append(row)
            raw = tf.constant([row["score"] for row in record["final"]], tf.float64)
            score_rows = dict(raw_fisher=prior.base.materialize(raw),
                              fixed_label=[row["fixed_score"] for row in record["final"]])
            for name in record["regression"]:
                k = 18 if name == "innovation" else 12
                matrix = tf.constant([row["ancestor"]+(row["innovation"] if k == 18 else [])
                                      for row in record["final"]], tf.float64)
                coefficient = tf.constant(record["regression"][name]["coefficient"], tf.float64)
                corrected = combinations[k][1](raw, matrix, coefficient)
                prior.finite(corrected)
                score_rows[name] = prior.base.materialize(corrected)
            if regime != "affine":
                score_rows["no_resampling"] = [row["no_resampling"] for row in record["final"]]
            record["scores"] = score_rows
            record["summary"] = {}
            losses = {}
            for name, scores in score_rows.items():
                rows = [dict(score=score, value=row["value"]) for score, row in zip(scores, record["final"])]
                summary = prior.base.summarize(rows, reference)
                # Likelihood provenance applies only to the iAPF score rows.
                if name == "no_resampling":
                    summary = {k: v for k, v in summary.items() if "value" not in k and "likelihood" not in k}
                summary["total_score_variance"] = sum(statistics.variance(column) for column in zip(*scores))
                summary["mean_error_intervals"] = [[error-critical*se-5e-6*(1+abs(target)),
                    error+critical*se+5e-6*(1+abs(target))]
                    for error, se, target in zip(summary["mean_score_error"], summary["score_standard_error"], reference["score"])]
                summary["bias_screen_pass"] = all(lo <= 0 <= hi for lo, hi in summary["mean_error_intervals"])
                record["summary"][name] = summary
                losses[name] = [prior.squared_error(score, reference) for score in scores]
            for name, row in record["deterministic"].items():
                losses[name] = [row["score_squared_error"]]*FINAL
            record["comparisons"] = {}
            for name, loss in losses.items():
                if name == "innovation":
                    continue
                differences = [a-b for a, b in zip(losses["innovation"], loss)]
                interval = prior.base.materialize(intervals(tf.constant(differences, tf.float64)))
                record["comparisons"][name] = dict(mean_difference=statistics.mean(differences),
                    primary_interval_99_75=interval[:2], exploratory_interval_99=interval[2:],
                    primary_pass=interval[1] < 0)
            heuristic_names = ("kalman", "ekf", "ukf") if regime == "affine" else ("ekf", "ukf", "no_resampling")
            losses_to = [name for name in heuristic_names if record["comparisons"][name]["mean_difference"] > 0]
            record["heuristic_dominance"] = dict(pass_screen=not losses_to, observed_losses=losses_to,
                scope="conditional_on_this_fixed_dataset_and_frozen_coefficients")
            record["innovation_control_mean"] = [statistics.mean(column) for column in zip(*(r["innovation"] for r in record["final"]))]
            record["innovation_control_mcse"] = [statistics.stdev(column)/math.sqrt(FINAL) for column in zip(*(r["innovation"] for r in record["final"]))]
            record["kernel_trace_count"] = kernel.experimental_get_tracing_count()
            save(f"final_complete_{dataset}")
            print(f"final complete: dataset {dataset}, heuristic losses {losses_to}", flush=True)

        nonlinear = [record for record in results["datasets"].values() if record["regime"] != "affine"]
        decision = dict(primary_comparisons_pass=sum(r["comparisons"]["ancestor_fresh"]["primary_pass"] for r in nonlinear),
            primary_comparisons_total=4,
            nonlinear_bias_screens_pass=sum(r["summary"]["innovation"]["bias_screen_pass"] for r in nonlinear),
            heuristic_dominance_pass=all(r["heuristic_dominance"]["pass_screen"] for r in results["datasets"].values()),
            default_adoption=False, hmc_force=False, ledh_admission=False,
            interpretation="conditional diagnostic only; candidate losses do not invalidate the research direction")
        decision.update(bound_repair_trigger=results["bound_repair_trigger"],
            bound_contact_datasets=[key for key, fit in results["fits"].items() if fit["boundary_active"]],
            pinned_device_verified=True)
        manifest["trace_counts"] = dict(streams=stream_kernel.experimental_get_tracing_count(),
            controls=controls.experimental_get_tracing_count(), intervals=intervals.experimental_get_tracing_count(),
            fit12=combinations[12][0].experimental_get_tracing_count(), fit18=combinations[18][0].experimental_get_tracing_count(),
            apply12=combinations[12][1].experimental_get_tracing_count(), apply18=combinations[18][1].experimental_get_tracing_count())
        if any(value != 1 for value in manifest["trace_counts"].values()):
            raise RuntimeError("unexpected retracing")
        if any(r["kernel_trace_count"] != 1 for r in results["datasets"].values()):
            raise RuntimeError("particle kernel retraced")
        manifest["terminal_source_sha256_match"] = all(digest(REPO/path) == value for path, value in hashes.items())
        if not manifest["terminal_source_sha256_match"]:
            raise RuntimeError("source drift during execution")
        if budget.counts["filter_calls"]-initial_calls != expected_calls:
            raise RuntimeError("unexpected final call accounting")
        if time.monotonic()-started > budget.seconds:
            raise RuntimeError("wall budget exhausted")
        manifest.update(status="complete", memory_usage=memory_usage("GPU"))
        results["decision"] = decision
        prior.base.write(output / "decision.json", decision)
        save("complete")
    except Exception as error:
        manifest.update(status="stopped", error=repr(error))
        save("stopped")
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-started, budget=budget.record())
        prior.base.write(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--continuation-stage", choices=("pinned", "fresh", "wide"))
    arguments = parser.parse_args()
    if arguments.preflight:
        input_records()
        if arguments.continuation_stage:
            if not arguments.output:
                parser.error("continuation preflight requires --output")
            continuation_budget(Path(arguments.output).resolve(), arguments.continuation_stage, time.monotonic())
        _, check = seed_inventory(arguments.continuation_stage,
                                  Path(arguments.output).name if arguments.output else None)
        print(json.dumps(dict(source_inputs_checked=True,
                              planned_filter_calls=1116 if arguments.continuation_stage in (None, "pinned") else 976,
                              **check), sort_keys=True))
    elif arguments.output:
        run(arguments)
    else:
        parser.error("--output is required for execution")
