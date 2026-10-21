#!/usr/bin/env python3
"""Independent KSC C/D and C/R tests with the frozen full-DCTR endpoint."""
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
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
from scripts.prepare_surrogate_hmc_ksc import (
    ORCHESTRATION_ONLY, budgets, check_selection, run_bounded_worker, save, sha, sources, verify_qualification,
)
from scripts.prepare_surrogate_hmc_ksc_finite import PLAN, FAMILY_CPU, FAMILY_GPU, check_endpoint
from scripts.run_surrogate_hmc_ksc import check_reference, tuning_config


def check_finite_reference(root, *, exact_reference, endpoint_root, selection_sha, posterior_sha):
    root, endpoint_root = Path(root), Path(endpoint_root)
    report = json.loads((root / "report.json").read_text())
    manifest = json.loads((root / "run-manifest.json").read_text())
    endpoint, _ = check_endpoint(endpoint_root, selection_sha=selection_sha, posterior_sha=posterior_sha)
    if (report.get("status") != "QUALIFIED_FINITE_POSTERIOR_REFERENCE" or report.get("qualified") is not True
            or report.get("model_id") != "M03-KSC-T2" or report.get("full_prior_integrated") is not True
            or manifest.get("exit_code") != 0):
        raise ValueError("qualified independent finite reference required")
    expected = dict(endpoint_identity=endpoint["endpoint_identity"], selection_sha256=selection_sha,
                    endpoint_report_sha256=sha(endpoint_root / "report.json"),
                    data_sha256=exact_reference["data_sha256"])
    for name, value in expected.items():
        if report.get(name) != value or manifest.get(name) != value:
            raise ValueError("finite reference scope mismatch: " + name)
    if report.get("reference_report_sha256") != posterior_sha or manifest.get("posterior_report_sha256") != posterior_sha:
        raise ValueError("finite reference exact-model reference mismatch")
    if sha(root / "data.json") != expected["data_sha256"]:
        raise ValueError("finite reference data drift")
    for name, digest in manifest["source_hashes"].items():
        if name.startswith("bayesfilter/") and name not in ORCHESTRATION_ONLY and sha(ROOT / name) != digest:
            raise ValueError("finite reference numerical source drift: " + name)
    if (report.get("parameter_intervals") != [16, 32, 64, 128] or len(report.get("grids", [])) != 4
            or any(set(grid) != {"gamma", "beta", "density", "predictive"} for grid in report["grids"])):
        raise ValueError("incomplete finite reference grids")
    for item in [report["log_density"], *(item for grid in report["grids"] for item in grid.values())]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("finite reference tensor corruption")
    for key in ("means", "scales", "cuts", "margins", "function_names"):
        if report.get(key) != exact_reference[key]:
            raise ValueError("finite and exact reference functional mismatch: " + key)
    if (any(len(report.get(key, [])) != 13 for key in ("allowance", "precision", "contraction"))
            or any(value is not True for key in ("precision", "contraction") for value in report[key])
            or any(not math.isfinite(v) or not 0 <= v <= m/10 for v, m in zip(report["allowance"], report["margins"]))
            or len(report.get("expectations", [])) != 4
            or any(len(row) != 13 or any(not math.isfinite(v) for v in row) for row in report["expectations"])):
        raise ValueError("finite reference precision not established")
    return report


def bind_finite_arm(consumer, endpoint):
    from bayesfilter.inference import bind_neural_force_hmc_tuning_runner
    scope = "M03-KSC-T2-N1024-claim-attempt1:C"
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=endpoint, target_scope=scope)

    class Adapter:
        target_scope = scope

        def adapter_signature(self):
            return scope + ":" + binding.binding_hash

    return Adapter(), binding


def assess_comparisons(parameters, functions, references):
    import tensorflow as tf
    from bayesfilter.inference.surrogate_hmc_equivalence import assess_equivalence
    return {name: assess_equivalence(parameters, functions,
        tf.constant(reference["expectations"][-1], tf.float64),
        tf.constant(reference["allowance"], tf.float64), tf.constant(reference["margins"], tf.float64),
        comparison_families=4, parameter_dimension=2, function_count=13)
        for name, reference in references.items()}


def pending_comparisons(statuses, finite, reference):
    """A resolved comparison is closed even if the other needs its second look."""
    return {name: ref for name, ref in (("C/D", finite), ("C/R", reference))
            if statuses[name] not in ("EQUIVALENT", "DISCREPANCY")}


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
    from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
    from bayesfilter.testing.surrogate_hmc_ksc_consumer import make_ksc_consumer
    from bayesfilter.testing.surrogate_hmc_ksc_finite_endpoint import make_finite_endpoint
    from bayesfilter.testing.surrogate_hmc_ksc_reference import physical_parameters, make_ksc_reference, check_dataset
    selection = check_selection(manifest["controls_root"], validate_controls=True)
    reference = check_reference(manifest["posterior_root"])
    finite = check_finite_reference(manifest["finite_reference_root"], exact_reference=reference,
        endpoint_root=manifest["endpoint_root"], selection_sha=manifest["selection_sha256"],
        posterior_sha=manifest["posterior_report_sha256"])
    data = json.loads((Path(manifest["posterior_root"]) / "data.json").read_text())
    check_dataset(data)
    if data["partition"] != "claim":
        raise ValueError("frozen claim data required")
    consumer = make_ksc_consumer(data, controls=selection["controls"], batch_size=4)
    endpoint, _, _ = make_finite_endpoint(consumer)
    if endpoint.identity != finite["endpoint_identity"]:
        raise ValueError("finite endpoint changed after reference preparation")
    adapter, binding = bind_finite_arm(consumer, endpoint)
    save(out / "consumer.json", consumer.manifest)
    save(out / "data.json", data)
    manifest.update(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        gpu_status=[d.name for d in tf.config.list_logical_devices("GPU")], data_sha256=sha(out / "data.json"),
        endpoint_identity=endpoint.identity, force_identity=consumer.force.identity,
        seeds=dict(data=data["seeds"], tuning=[20260921, 30000], verification=[20260921, 30001],
                   retained=[[20260921, 31000+i] for i in range(8)]),
        comparison_families=4, function_count=13, planned_looks=[1000, 2000])
    save(out / "run-manifest.json", manifest)
    report = json.loads((out / "report.json").read_text())
    report.update(status="TUNING", D_vs_R=finite["D_vs_R"], endpoint_identity=endpoint.identity,
        arms={"C": dict(status="TUNING", chunks=[], looks=[])})
    report["posterior_comparisons"]["D/R"] = finite["D_vs_R"]["status"]
    save(out / "report.json", report)
    arm = report["arms"]["C"]
    start, cpu = time.monotonic(), time.process_time()
    try:
        result = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.),
            config=tuning_config(adapter.target_scope, 30000), parameter_scales=(1., 1.),
            runner_binding=binding, output_dir=out / "C/tuning")
        tuning = dict(passed=result.passed, checks=result.checks, manifest=result.artifact_manifest_path,
            selected_step_size=float(result.nomination.step_size), selected_leapfrog_steps=int(result.nomination.num_leapfrog_steps),
            metric_covariance=result.nomination.covariance.numpy().tolist())
    except ValueError as exc:
        if "nomination failed" not in str(exc):
            raise
        tuning = dict(passed=False, failure="nomination_failed", detail=str(exc))
    tuning.update(wall_seconds=time.monotonic()-start, cpu_worker_seconds=time.process_time()-cpu)
    arm.update(tuning=tuning, status="TUNING_PASSED" if tuning["passed"] else "TUNING_REJECTED")
    save(out / "report.json", finite_report(report))
    if tuning["passed"]:
        durable = load_verified_tensorflow_hmc_result(out / "C/tuning/manifest.json", adapter=adapter, runner_binding=binding)
        runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=durable, runner_binding=binding)
        report["status"] = "RETAINED_SAMPLING"
        means, scales, cuts = [tf.constant(reference[k], tf.float64) for k in ("means", "scales", "cuts")]
        exact = make_ksc_reference(data, batch_size=1000)

        @tf.function(input_signature=[tf.TensorSpec([250, 4, 2], tf.float64)], jit_compile=True, autograph=False)
        def functionals(samples):
            theta = physical_parameters(tf.reshape(samples, [1000, 2]))
            z = (theta-means)/scales
            a, b = z[:, 0], z[:, 1]
            moments = tf.stack([a, b, a*a, b*b, a*b], axis=1)
            cdfs = tf.cast(theta[:, :, None] <= cuts[None, :, :], tf.float64)
            _, predictive = exact.statistics(theta)
            values = tf.concat([moments, tf.reshape(cdfs, [1000, 6]), predictive], axis=1)
            return tf.reshape(theta, [250, 4, 2]), tf.reshape(values, [250, 4, 13])

        previous, parameters, values = None, [], []
        for index in range(8):
            start, cpu = time.monotonic(), time.process_time()
            directory = out / "C" / f"chunk-{index+1:02d}"
            archive = runner.run(BoundRetainedHMCArchiveConfig(250, (20260921, 31000+index), directory,
                "KSC independent finite-target test; warmup and verification excluded", previous))
            previous = archive["archive_manifest_path"]
            arm["chunks"].append(dict(manifest=previous, health_passed=archive["health_passed"],
                wall_seconds=time.monotonic()-start, cpu_worker_seconds=time.process_time()-cpu))
            if not archive["health_passed"]:
                arm["status"] = "HEALTH_VETO"
                report["posterior_comparisons"].update({key: "HEALTH_VETO" for key in ("C/D", "C/R")})
                break
            archive_manifest = json.loads(Path(previous).read_text())
            sample_path = directory / archive_manifest["raw_samples"]["path"]
            if sha(sample_path) != archive_manifest["raw_samples"]["sha256"]:
                raise ValueError("retained sample corruption")
            samples = tf.io.parse_tensor(tf.io.read_file(str(sample_path)), tf.float64)
            theta, current = functionals(samples)
            parameters.append(theta)
            values.append(current)
            arm["retained_draws"] = (index+1)*250
            if index+1 in (4, 8):
                comparisons = assess_comparisons(tf.transpose(tf.concat(parameters, 0), [1, 0, 2]),
                    tf.transpose(tf.concat(values, 0), [1, 0, 2]),
                    pending_comparisons(report["posterior_comparisons"], finite, reference))
                arm["looks"].append(dict(draws_per_chain=(index+1)*250, comparisons=comparisons))
                report["posterior_comparisons"].update({key: row["status"] for key, row in comparisons.items()})
                arm["status"] = report["posterior_comparisons"]["C/D"]
                print("C LOOK", (index+1)*250, report["posterior_comparisons"], flush=True)
                if not pending_comparisons(report["posterior_comparisons"], finite, reference):
                    break
            save(out / "report.json", finite_report(report))
            print("C CHUNK", index+1, flush=True)
    else:
        report["posterior_comparisons"].update({key: "TUNING_REJECTED" for key in ("C/D", "C/R")})
    report.update(status="COMPLETED_BOUNDED_FINITE_INFERENCE", surrogate_hmc_tested=True,
        retained_surrogate_hmc_tested=bool(arm["chunks"]), allocator=tf.config.experimental.get_memory_info("GPU:0"))
    save(out / "report.json", finite_report(report))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "qualification-root", "controls-root", "posterior-root", "endpoint-root", "finite-reference-root", "worker"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cpu-worker-seconds", type=float, default=54000.)
    parser.add_argument("--max-gpu-seconds", type=float, default=226800.)
    parser.add_argument("--previous-accounting", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    fields = ("output", "qualification_root", "controls_root", "posterior_root", "endpoint_root", "finite_reference_root")
    if any(getattr(args, key) is None for key in fields):
        parser.error("output, qualification, controls, posterior, endpoint and finite-reference roots required")
    out = args.output.resolve()
    prefix = "surrogate-hmc-all-models-20260921-ksc-t2-finite-inference-"
    if out.parent != ROOT / "docs/plans/artifacts" or not out.name.startswith(prefix) or out.exists():
        raise ValueError("fresh versioned KSC finite-inference root required")
    if (not math.isfinite(args.max_cpu_worker_seconds) or not 0 < args.max_cpu_worker_seconds <= 54000
            or not math.isfinite(args.max_gpu_seconds) or not 0 < args.max_gpu_seconds <= 226800):
        raise ValueError("invalid or expanded campaign budget")
    if list(out.parent.glob("surrogate-hmc-all-models-*-ksc-t2-finite-inference-*/report.json")):
        raise ValueError("prior C inference attempt requires result-aware continuation")
    total, family, _ = budgets(args.previous_accounting)
    provenance = sources()
    verify_qualification(args.qualification_root, provenance)
    check_selection(args.controls_root)
    reference = check_reference(args.posterior_root)
    endpoint, _ = check_endpoint(args.endpoint_root, selection_sha=sha(args.controls_root / "selection.json"),
                                 posterior_sha=sha(args.posterior_root / "report.json"))
    check_finite_reference(args.finite_reference_root, exact_reference=reference, endpoint_root=args.endpoint_root,
        selection_sha=sha(args.controls_root / "selection.json"), posterior_sha=sha(args.posterior_root / "report.json"))
    cpu = min(3100., FAMILY_CPU-family["cpu_worker_seconds"], args.max_cpu_worker_seconds-total["cpu_worker_seconds"])
    wall = min(3300., FAMILY_GPU-family["gpu_seconds"], args.max_gpu_seconds-total["gpu_seconds"])
    if endpoint["forecast"]["cpu_worker_seconds"]["C_inference"] > cpu or endpoint["forecast"]["gpu_seconds"]["C_inference"] > wall:
        raise ValueError("complete C inference forecast exceeds allocation")
    for name in json.loads((args.finite_reference_root / "run-manifest.json").read_text())["source_hashes"]:
        provenance[name] = sha(ROOT / name)
    provenance[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    provenance["scripts/run_surrogate_hmc_ksc.py"] = sha(ROOT / "scripts/run_surrogate_hmc_ksc.py")
    # Public lazy exports and relative imports are not all reached by the
    # preparation inventory's absolute-import walk. Preserve the actual tuner.
    for name in ("hmc_tensorflow_verified.py", "hmc_tensorflow_tuning.py", "hmc_tuning_dispatch.py",
                 "hmc_posterior_diagnostics.py", "tuning_contract.py", "neural_force_hmc.py",
                 "surrogate_hmc_equivalence.py"):
        path = "bayesfilter/inference/"+name
        provenance[path] = sha(ROOT / path)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, cpu_budget=cpu, wall_budget=wall, forecast=endpoint["forecast"])))
        return 0
    out.mkdir()
    started, parent_cpu = time.monotonic(), time.process_time()
    for name in provenance:
        destination = out / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    save(out / "executed-source/sha256.json", provenance)
    manifest = dict(schema="surrogate_hmc_ksc_finite_inference_v1", stage="P3-KSC-finite-inference", scope="M03-KSC-T2",
        question="Does endpoint-corrected full-DCTR-force HMC sample the independently integrated finite D posterior?",
        plan=PLAN, result_file=str(out / "report.json"), command=[sys.executable, *sys.argv], environment=sys.prefix,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=provenance, cpu_budget=cpu, wall_budget=wall, forecast=endpoint["forecast"],
        selection_sha256=sha(args.controls_root / "selection.json"), posterior_report_sha256=sha(args.posterior_root / "report.json"),
        finite_reference_report_sha256=sha(args.finite_reference_root / "report.json"),
        endpoint_report_sha256=sha(args.endpoint_root / "report.json"), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        data_version="frozen nominal KSC T2 claim data; independent finite endpoint and tuning")
    manifest.update({key: str(getattr(args, key).resolve()) for key in fields if key != "output"})
    save(out / "run-manifest.json", manifest)
    save(out / "report.json", dict(status="RUNNING", model_id="M03-KSC-T2", surrogate_hmc_tested=False,
        posterior_comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")},
        plan=PLAN, heuristic_dominance="NOT_RUN", exact_score_admission=False, default_readiness=False,
        all_models_validated=False, inference_attempts_consumed=1))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code, watchdog = run_bounded_worker([sys.executable, str(Path(__file__).resolve()), "--worker", str(out)],
        log_path=out / "worker.log", cpu_budget=cpu-2, wall_budget=wall-2)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    charge = dict(cpu_worker_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu,
                  gpu_seconds=time.monotonic()-started)
    report = json.loads((out / "report.json").read_text())
    if code:
        report.update(status="INFRASTRUCTURE_OR_RESOURCE_FAILURE", exit_code=code)
    if any(sha(ROOT / name) != digest for name, digest in provenance.items()):
        report.update(status="SOURCE_DRIFT_INVALIDATES_RUN")
        code = 2
    save(out / "report.json", report)
    save(out / "accounting.json", dict(charged_this_run=charge, exit_code=code, watchdog=watchdog))
    manifest = json.loads((out / "run-manifest.json").read_text())
    manifest.update(wall_seconds=charge["gpu_seconds"], cpu_worker_seconds=charge["cpu_worker_seconds"], exit_code=code)
    save(out / "run-manifest.json", manifest)
    print(json.dumps(dict(status=report["status"], accounting=charge, output=str(out))))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
