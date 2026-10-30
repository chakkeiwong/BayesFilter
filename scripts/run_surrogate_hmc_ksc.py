#!/usr/bin/env python3
"""Public-master KSC T2 exact-endpoint surrogate-HMC inference worker."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
from scripts.prepare_surrogate_hmc_ksc import (
    PLAN, ORCHESTRATION_ONLY, budgets, check_selection, run_bounded_worker,
    save, sha, sources, verify_qualification,
)

FAMILY_CPU = 4000.
FAMILY_GPU = 7200.


def check_reference(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    manifest = json.loads((root / "run-manifest.json").read_text())
    if (report.get("status") != "QUALIFIED_MODEL_POSTERIOR_REFERENCE" or report.get("qualified") is not True
            or report.get("model_id") != "M03-KSC-T2" or report.get("full_prior_integrated") is not True):
        raise ValueError("qualified same-law full-prior KSC posterior required")
    if sha(root / "data.json") != report["data_sha256"] or report["data_sha256"] != manifest["data_sha256"]:
        raise ValueError("posterior data drift")
    for name, digest in manifest["source_hashes"].items():
        if name.startswith("bayesfilter/") and name not in ORCHESTRATION_ONLY and sha(ROOT / name) != digest:
            raise ValueError("posterior numerical source drift: " + name)
    if (report.get("parameter_intervals") != [32, 64, 128, 256]
            or len(report.get("grids", [])) != 4
            or any(set(grid) != {"gamma", "beta", "density", "predictive"} for grid in report["grids"])):
        raise ValueError("incomplete posterior refinement grids")
    for grid in report["grids"]:
        for item in grid.values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("posterior grid checksum mismatch")
    if (any(len(report.get(key, [])) != 13 for key in ("allowance", "precision", "contraction", "margins", "function_names"))
            or any(value is not True for key in ("precision", "contraction") for value in report[key])
            or report["margins"] != [.1] * 5 + [.02] * 6 + [.03] * 2
            or any(not math.isfinite(v) or not 0 <= v <= margin / 10
                   for v, margin in zip(report["allowance"], report["margins"]))
            or len(report.get("expectations", [])) != 4
            or any(len(row) != 13 or any(not math.isfinite(v) for v in row) for row in report["expectations"])
            or len(report.get("means", [])) != 2 or any(not math.isfinite(v) for v in report["means"])
            or len(report.get("scales", [])) != 2 or any(not math.isfinite(v) or v <= 0 for v in report["scales"])
            or len(report.get("cuts", [])) != 2
            or any(len(row) != 3 or any(not math.isfinite(v) or not .1 < v < .9 for v in row)
                   or row != sorted(row) for row in report["cuts"])):
        raise ValueError("incomplete posterior functional precision")
    return report


def complete_forecast(cost_root):
    root = Path(cost_root)
    report = json.loads((root / "report.json").read_text())
    if report.get("status") != "COMPLETE_TRANSITION_COST_MEASURED":
        raise ValueError("measured complete KSC transition required")
    timings = report["complete_transition_costs"]
    if (len(timings) != 5 or timings[0]["compilation_included"] is not True
            or any(row["compilation_included"] is not False for row in timings[1:])):
        raise ValueError("missing compilation or steady transition costs")
    forecast = {}
    for key in ("cpu_worker_seconds", "wall_seconds"):
        all_values = [float(t[key]) for t in timings]
        if any(not math.isfinite(v) or v <= 0 for v in all_values):
            raise ValueError("invalid measured cost")
        values = all_values[1:]
        forecast[key] = 1.5 * (max(values) * 7172 + 240 + 90)
    return dict(required=forecast, safety_factor=1.5, equivalent_L1_transitions=7172,
                additional_compilation_seconds=240, comparator_and_analysis_seconds=90,
                cost_report_sha256=sha(root / "report.json"), heuristic_dominance="NOT_RUN")


def bind_arm(consumer, arm):
    from bayesfilter.inference import bind_neural_force_hmc_tuning_runner
    scope = "M03-KSC-T2-N1024-claim-attempt1:" + arm
    binding = bind_neural_force_hmc_tuning_runner(
        force=consumer.exact_force if arm == "A" else consumer.force,
        target=consumer.endpoint, target_scope=scope)

    class Adapter:
        target_scope = scope

        def adapter_signature(self):
            return scope + ":" + binding.binding_hash

    return Adapter(), binding


def tuning_config(scope, seed):
    from bayesfilter.inference import (
        TensorFlowHMCKernelTuningConfig, VerifiedTensorFlowHMCConfig, FourChainMeanBandAcceptancePolicy,
    )
    nomination = TensorFlowHMCKernelTuningConfig(
        parameter_dimension=2, evidence_role="diagnostic_candidate_screen",
        mass_window_results=(32, 64), step_adaptation_results=128, verification_results=64,
        max_leapfrog_steps=2, initial_step_size=.2,
        budget_provenance="KSC T2 first inference attempt; full two-look schedule reserved",
        initial_step_size_provenance=".2 unit-normal probit chart hypothesis, separately adapted",
        geometry_provenance="unit chart hypothesis, independent windowed covariance per arm",
        target_scope=scope, acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0., 1.), per_chain_band=(0., 1.)), target_accept_prob=.7,
        verification_repair_rounds=0, step_repair_factor=2., mass_shrinkage=.1,
        covariance_jitter=1e-9, eigenvalue_floor=1e-9, max_condition_number=1e8,
        seed=(20260921, seed), use_xla=True)
    return VerifiedTensorFlowHMCConfig(nomination, ((-2., -2.), (-.7, .7), (.7, -.7), (2., 2.)),
        "four probit prior-region starts, frozen before claim chains", 2000, 1000, (20260921, seed + 1), 1000.)


def worker(out):
    manifest = json.loads((out / "run-manifest.json").read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if len(tf.config.list_physical_devices("GPU")) != 1:
        raise ValueError("one designated GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.inference import (
        tune_hmc_kernel, load_verified_tensorflow_hmc_result,
        build_retained_bound_hmc_archive_runner_from_tuning_result, BoundRetainedHMCArchiveConfig,
    )
    from bayesfilter.inference.surrogate_hmc_equivalence import assess_equivalence, finite_report
    from bayesfilter.testing.surrogate_hmc_ksc_consumer import make_ksc_consumer
    from bayesfilter.testing.surrogate_hmc_ksc_reference import physical_parameters, make_ksc_reference, check_dataset
    selected = check_selection(manifest["controls_root"], validate_controls=True)
    ref_root = Path(manifest["posterior_root"])
    reference = check_reference(ref_root)
    data = json.loads((ref_root / "data.json").read_text())
    check_dataset(data)
    if data["partition"] != "claim":
        raise ValueError("untouched claim data required")
    consumer = make_ksc_consumer(data, controls=selected["controls"], batch_size=4)
    save(out / "consumer.json", consumer.manifest)
    save(out / "data.json", data)
    manifest.update(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        gpu_status=[d.name for d in tf.config.list_logical_devices("GPU")], data_sha256=sha(out / "data.json"),
        seeds=dict(data=data["seeds"], tuning={"A": [20260921, 26000], "B": [20260921, 27000]},
                   retained={"A": [[20260921, 28000 + i] for i in range(8)],
                             "B": [[20260921, 29000 + i] for i in range(8)]}),
        consumer=consumer.manifest, discarded_warmup_per_chain=250, verification_per_chain=1000,
        retained_looks=[1000, 2000], function_count=13, comparison_families=4,
        data_version="frozen KSC T2 nominal claim data; exact49-mixture likelihood; prior unchanged")
    save(out / "run-manifest.json", manifest)
    report = json.loads((out / "report.json").read_text())
    report.update(status="TUNING", arms={}, comparison_roles={"A/R": "exact-force comparator diagnostic",
        "B/R": "surrogate-HMC model posterior test", "C/D": "finite-target sampler test",
        "D/R": "finite-target discrepancy", "C/R": "combined finite-target inference"})
    save(out / "report.json", report)
    means, scales, cuts = [tf.constant(reference[k], tf.float64) for k in ("means", "scales", "cuts")]
    ref = make_ksc_reference(data, batch_size=1000)

    @tf.function(input_signature=[tf.TensorSpec([250, 4, 2], tf.float64)], jit_compile=True, autograph=False)
    def functionals(samples):
        theta = physical_parameters(tf.reshape(samples, [1000, 2]))
        standardized = (theta - means) / scales
        a, b = standardized[:, 0], standardized[:, 1]
        moments = tf.stack([a, b, a*a, b*b, a*b], axis=1)
        cdfs = tf.cast(theta[:, :, None] <= cuts[None, :, :], tf.float64)
        _, predictive = ref.statistics(theta)
        values = tf.concat([moments, tf.reshape(cdfs, [1000, 6]), predictive], axis=1)
        return tf.reshape(theta, [250, 4, 2]), tf.reshape(values, [250, 4, 13])

    for arm, tune_seed, sample_seed in (("A", 26000, 28000), ("B", 27000, 29000)):
        adapter, binding = bind_arm(consumer, arm)
        started, cpu = time.monotonic(), time.process_time()
        print("TUNING", arm, flush=True)
        try:
            result = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.),
                config=tuning_config(adapter.target_scope, tune_seed), parameter_scales=(1., 1.),
                runner_binding=binding, output_dir=out / arm / "tuning")
            row = dict(passed=result.passed, checks=result.checks, manifest=result.artifact_manifest_path,
                       selected_step_size=float(result.nomination.step_size),
                       selected_leapfrog_steps=int(result.nomination.num_leapfrog_steps),
                       metric_covariance=result.nomination.covariance.numpy().tolist())
        except ValueError as exc:
            if "nomination failed" not in str(exc):
                raise
            row = dict(passed=False, failure="nomination_failed", detail=str(exc))
        row.update(wall_seconds=time.monotonic() - started, cpu_worker_seconds=time.process_time() - cpu)
        report["arms"][arm] = dict(tuning=row, status="TUNING_PASSED" if row["passed"] else "TUNING_REJECTED",
                                   chunks=[], looks=[])
        save(out / "report.json", report)
        print(arm, report["arms"][arm]["status"], flush=True)
        if not row["passed"]:
            if arm == "B":
                report["posterior_comparisons"]["B/R"] = "TUNING_REJECTED"
            continue
        durable = load_verified_tensorflow_hmc_result(out / arm / "tuning/manifest.json",
            adapter=adapter, runner_binding=binding)
        report.update(status="RETAINED_SAMPLING", active_arm=arm)
        save(out / "report.json", finite_report(report))
        runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=durable, runner_binding=binding)
        previous, parameters, values = None, [], []
        for index in range(8):
            start, cpu = time.monotonic(), time.process_time()
            directory = out / arm / f"chunk-{index+1:02d}"
            archive = runner.run(BoundRetainedHMCArchiveConfig(250, (20260921, sample_seed + index), directory,
                "KSC T2 frozen-kernel posterior test; warmup and verification excluded", previous))
            previous = archive["archive_manifest_path"]
            report["arms"][arm]["chunks"].append(dict(manifest=previous, health_passed=archive["health_passed"],
                wall_seconds=time.monotonic()-start, cpu_worker_seconds=time.process_time()-cpu))
            if not archive["health_passed"]:
                report["arms"][arm]["status"] = "HEALTH_VETO"
                if arm == "B":
                    report["posterior_comparisons"]["B/R"] = "HEALTH_VETO"
                break
            archive_manifest = json.loads(Path(previous).read_text())
            sample_path = directory / archive_manifest["raw_samples"]["path"]
            if sha(sample_path) != archive_manifest["raw_samples"]["sha256"]:
                raise ValueError("retained sample corruption")
            samples = tf.io.parse_tensor(tf.io.read_file(str(sample_path)), tf.float64)
            theta, current = functionals(samples)
            parameters.append(theta)
            values.append(current)
            report["arms"][arm]["retained_draws"] = (index+1)*250
            if index+1 in (4, 8):
                look = assess_equivalence(tf.transpose(tf.concat(parameters, 0), [1, 0, 2]),
                    tf.transpose(tf.concat(values, 0), [1, 0, 2]),
                    tf.constant(reference["expectations"][-1], tf.float64),
                    tf.constant(reference["allowance"], tf.float64), tf.constant(reference["margins"], tf.float64),
                    comparison_families=4, parameter_dimension=2, function_count=13)
                report["arms"][arm]["looks"].append(look)
                report["arms"][arm]["status"] = look["status"]
                if arm == "B":
                    report["posterior_comparisons"]["B/R"] = look["status"]
                save(out / "report.json", finite_report(report))
                print(arm, "LOOK", (index+1)*250, look["status"], flush=True)
                if look["status"] in ("EQUIVALENT", "DISCREPANCY"):
                    break
            save(out / "report.json", finite_report(report))
            print(arm, "CHUNK", index+1, flush=True)
        save(out / "report.json", finite_report(report))
    report.update(status="COMPLETED_BOUNDED_INFERENCE", surrogate_hmc_tested=True,
                  retained_surrogate_hmc_tested=bool(report["arms"]["B"]["chunks"]), active_arm=None,
                  allocator=tf.config.experimental.get_memory_info("GPU:0"))
    save(out / "report.json", finite_report(report))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "qualification-root", "controls-root", "posterior-root", "cost-root", "worker"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cpu-worker-seconds", type=float, default=54000.)
    parser.add_argument("--max-gpu-seconds", type=float, default=226800.)
    parser.add_argument("--previous-accounting", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if any(getattr(args, name) is None for name in ("output", "qualification_root", "controls_root", "posterior_root", "cost_root")):
        parser.error("output, qualification, controls, posterior and cost roots required")
    out = args.output.resolve()
    if out.parent != ROOT / "docs/plans/artifacts" or not out.name.startswith("surrogate-hmc-all-models-20260921-ksc-t2-inference-"):
        raise ValueError("fresh versioned KSC inference root required")
    if out.exists():
        raise FileExistsError("preserve prior inference results")
    if (not math.isfinite(args.max_cpu_worker_seconds) or not math.isfinite(args.max_gpu_seconds)
            or not 0 < args.max_cpu_worker_seconds <= 54000 or not 0 < args.max_gpu_seconds <= 226800):
        raise ValueError("invalid or expanded campaign budget")
    total, family, _ = budgets(args.previous_accounting)
    prior = list((ROOT / "docs/plans/artifacts").glob("surrogate-hmc-all-models-*-ksc-t2-inference-*/report.json"))
    if prior:
        raise ValueError("prior KSC inference attempt requires explicit result-aware continuation")
    provenance = sources()
    for name in ("scripts/run_surrogate_hmc_ksc.py", "bayesfilter/testing/surrogate_hmc_ksc_posterior_reference.py",
                 "bayesfilter/inference/surrogate_hmc_equivalence.py", "bayesfilter/inference/hmc_tensorflow_verified.py"):
        provenance[name] = sha(ROOT / name)
    verify_qualification(args.qualification_root, provenance)
    selected = check_selection(args.controls_root)
    reference = check_reference(args.posterior_root)
    if reference["selection_sha256"] != sha(args.controls_root / "selection.json"):
        raise ValueError("posterior and controls selection mismatch")
    cost_report = json.loads((args.cost_root / "report.json").read_text())
    if cost_report["selection_sha256"] != sha(args.controls_root / "selection.json"):
        raise ValueError("cost and controls selection mismatch")
    forecast = complete_forecast(args.cost_root)
    cpu = min(3450., FAMILY_CPU-family["cpu_worker_seconds"], args.max_cpu_worker_seconds-total["cpu_worker_seconds"])
    wall = min(3600., FAMILY_GPU-family["gpu_seconds"], args.max_gpu_seconds-total["gpu_seconds"])
    if forecast["required"]["cpu_worker_seconds"] > cpu or forecast["required"]["wall_seconds"] > wall:
        raise ValueError("complete two-look KSC inference forecast exceeds allocation")
    launch = dict(schema="surrogate_hmc_ksc_inference_v1", plan=PLAN, stage="P3-KSC-inference", scope="M03-KSC-T2",
        question="Does endpoint-corrected HMC with a full canonical DCTR proposal field recover the KSC parameter posterior?",
        command=[sys.executable, *sys.argv], environment=sys.prefix,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=provenance, qualification_root=str(args.qualification_root.resolve()),
        controls_root=str(args.controls_root.resolve()), posterior_root=str(args.posterior_root.resolve()),
        cost_root=str(args.cost_root.resolve()), result_file=str(out / "report.json"),
        cpu_budget=cpu, wall_budget=wall, family_ceiling=dict(cpu_worker_seconds=FAMILY_CPU, gpu_seconds=FAMILY_GPU),
        previous_family_charge=family, previous_campaign_charge=total, complete_forecast=forecast,
        selected_multiplier=selected["selected_multiplier"], cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, cpu_budget=cpu, wall_budget=wall, forecast=forecast)), flush=True)
        return 0
    out.mkdir()
    started, parent_cpu = time.monotonic(), time.process_time()
    for name in provenance:
        target = out / "executed-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    save(out / "executed-source/sha256.json", provenance)
    save(out / "run-manifest.json", launch)
    report = dict(status="RUNNING", surrogate_hmc_tested=False, inference_attempts_consumed=1,
        model_id="M03-KSC-T2", heuristic_dominance="NOT_RUN", all_models_validated=False,
        posterior_comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")},
        plan=PLAN, exact_score_admission=False, default_readiness=False)
    save(out / "report.json", report)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code, watchdog = run_bounded_worker([sys.executable, str(Path(__file__).resolve()), "--worker", str(out)],
        log_path=out / "worker.log", cpu_budget=cpu-2, wall_budget=wall-2)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    charge = dict(cpu_worker_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu,
                  gpu_seconds=time.monotonic()-started)
    save(out / "accounting.json", dict(charged=charge, charged_this_run=charge, exit_code=code, watchdog=watchdog))
    report = json.loads((out / "report.json").read_text())
    if code:
        report.update(status="INFRASTRUCTURE_OR_RESOURCE_FAILURE", exit_code=code)
    if any(sha(ROOT / name) != digest for name, digest in provenance.items()):
        report.update(status="SOURCE_DRIFT_INVALIDATES_RUN")
        code = 2
    save(out / "report.json", report)
    manifest = json.loads((out / "run-manifest.json").read_text())
    manifest.update(wall_seconds=charge["gpu_seconds"], cpu_worker_seconds=charge["cpu_worker_seconds"], exit_code=code)
    save(out / "run-manifest.json", manifest)
    from bayesfilter.inference.surrogate_hmc_program import model_register
    register = model_register()
    for row in register["families"]:
        if row["id"] == "M03":
            row.update(status="PARTIAL_KSC_INFERENCE_EXECUTED" if report["status"] == "COMPLETED_BOUNDED_INFERENCE"
                       else "KSC_INFERENCE_ATTEMPT_INCOMPLETE", inference_status=report["posterior_comparisons"],
                       surrogate_hmc_tested=report["surrogate_hmc_tested"],
                       retained_surrogate_hmc_tested=report.get("retained_surrogate_hmc_tested", False),
                       evidence=str(out / "report.json"))
    save(out / "model-register.json", register)
    save(out / "decision.json", dict(status=report["status"], comparisons=report["posterior_comparisons"],
        all_models_validated=False, statistically_supported_ranking=False,
        heuristic_dominance="NOT_RUN", default_readiness=False, source=str(out / "report.json")))
    print(json.dumps(dict(status=report["status"], comparisons=report["posterior_comparisons"], accounting=charge, output=str(out))), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
