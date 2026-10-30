#!/usr/bin/env python3
"""First bounded SV20 B/R surrogate-HMC posterior test with public tuning."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/surrogate-hmc-sv20-inference-2026-09-22.md"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(ROOT))
from scripts.qualify_surrogate_hmc_sv_endpoint import sources
from scripts.qualify_surrogate_hmc_nonlinear_consumer import save, serial, sha, snapshot


def tuning_config(scope):
    from bayesfilter.inference import (
        TensorFlowHMCKernelTuningConfig, VerifiedTensorFlowHMCConfig, FourChainMeanBandAcceptancePolicy)
    PRIOR_SD = 1.
    nomination = TensorFlowHMCKernelTuningConfig(parameter_dimension=2,
        evidence_role="diagnostic_candidate_screen", mass_window_results=(32, 64),
        step_adaptation_results=128, verification_results=64, max_leapfrog_steps=1,
        initial_step_size=.1, budget_provenance="SV20 first B/R attempt; complete L1-only two-look schedule reserved",
        initial_step_size_provenance="0.1 relative to prior-scaled identity coordinates; independently adapted",
        geometry_provenance="SV20 standard-normal probit prior, independent windowed covariance",
        target_scope=scope, acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0., 1.), per_chain_band=(0., 1.)), target_accept_prob=.7,
        verification_repair_rounds=0, step_repair_factor=2., mass_shrinkage=.1,
        covariance_jitter=1e-9, eigenvalue_floor=1e-9, max_condition_number=1e8,
        seed=(20260922, 35000), use_xla=True)
    starts = tuple(tuple(x*PRIOR_SD for x in point) for point in ((-2., -2.), (-.7, .7), (.7, -.7), (2., 2.)))
    return VerifiedTensorFlowHMCConfig(nomination, starts,
        "four SV20 dispersed probit starts frozen before claim chains", 250, 1000,
        (20260922, 35001), 1000.)


def worker(out):
    manifest = json.loads((out/"run-manifest.json").read_text())
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT/name) != digest or sha(out/"source"/name) != digest:
            raise ValueError("source drift before inference: "+name)
    for record in manifest["inputs"].values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("frozen SV20 inference input changed")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if len(tf.config.list_physical_devices("GPU")) != 1:
        raise ValueError("one designated GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.inference import (
        tune_hmc_kernel, bind_neural_force_hmc_tuning_runner, load_verified_tensorflow_hmc_result,
        build_retained_bound_hmc_archive_runner_from_tuning_result, BoundRetainedHMCArchiveConfig)
    from bayesfilter.inference.surrogate_hmc_equivalence import assess_equivalence, finite_report
    from bayesfilter.inference.surrogate_hmc_force_health import original_force_archive_health
    from bayesfilter.testing.surrogate_hmc_sv_inference_consumer import make_sv_inference_consumer
    from bayesfilter.testing.surrogate_hmc_sv_inference_metrics import make_functionals
    PRIOR_SD = 1.
    read = lambda key: json.loads(Path(manifest["inputs"][key]["path"]).read_text())
    claim, selected, posterior = (read(k) for k in ("claim", "selection", "posterior_report"))
    reference = dict(posterior["numerical_refinement"], margins=posterior["margins"])
    consumer = make_sv_inference_consumer(claim, controls=selected["controls"], batch_size=4)
    save(out/"consumer.json", consumer.manifest)
    save(out/"data.json", claim)
    scope = "SV20-T20-N1024-claim-attempt1:B"
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint, target_scope=scope)
    class Adapter:
        target_scope = scope
        def adapter_signature(self):
            return scope+":"+binding.binding_hash
    adapter = Adapter()
    save(out/"binding.json", dict(binding.payload()))
    config = tuning_config(scope)
    manifest.update(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        gpu_status=[d.name for d in tf.config.list_logical_devices("GPU")], data_sha256=sha(out/"data.json"),
        consumer=consumer.manifest, discarded_warmup_per_chain=250, verification_per_chain=1000,
        retained_looks=[1000, 2000], function_count=13, comparison_families=4)
    save(out/"run-manifest.json", manifest)
    report = json.loads((out/"report.json").read_text())
    report.update(status="TUNING", arms={}, comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")})
    save(out/"report.json", report)
    started, cpu = time.monotonic(), time.process_time()
    try:
        result = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.),
            config=config, parameter_scales=(PRIOR_SD, PRIOR_SD), runner_binding=binding, output_dir=out/"B/tuning")
        tuning = dict(passed=result.passed, checks=result.checks, manifest=result.artifact_manifest_path,
            selected_step_size=float(result.nomination.step_size), selected_leapfrog_steps=int(result.nomination.num_leapfrog_steps),
            metric_covariance=result.nomination.covariance.numpy().tolist())
    except ValueError as exc:
        if "nomination failed" not in str(exc):
            raise
        tuning = dict(passed=False, failure="nomination_failed", detail=str(exc))
    tuning.update(wall_seconds=time.monotonic()-started, cpu_worker_seconds=time.process_time()-cpu)
    row = dict(tuning=tuning, status="TUNING_PASSED" if tuning["passed"] else "TUNING_REJECTED", chunks=[], looks=[])
    report["arms"]["B"] = row
    report["comparisons"]["B/R"] = row["status"]
    save(out/"report.json", finite_report(report))
    print("B", row["status"], flush=True)
    if tuning["passed"]:
        durable = load_verified_tensorflow_hmc_result(out/"B/tuning/manifest.json", adapter=adapter, runner_binding=binding)
        runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=durable, runner_binding=binding)
        report["status"] = "RETAINED_SAMPLING"
        save(out/"report.json", finite_report(report))
        functionals = make_functionals(claim, reference)
        previous, parameters, values = None, [], []
        for index in range(8):
            started, cpu = time.monotonic(), time.process_time()
            directory = out/"B"/f"chunk-{index+1:02d}"
            archive = runner.run(BoundRetainedHMCArchiveConfig(250, (20260922, 36000+index), directory,
                "SV20 first B/R posterior test; warmup and fresh verification excluded", previous))
            previous = archive["archive_manifest_path"]
            original_health = original_force_archive_health(previous, expected_transitions=250)
            row["chunks"].append(dict(manifest=previous, health_passed=archive["health_passed"],
                original_force_health=original_health,
                wall_seconds=time.monotonic()-started, cpu_worker_seconds=time.process_time()-cpu))
            if not archive["health_passed"] or not original_health["passed"]:
                row["status"] = "HEALTH_VETO"
                report["comparisons"]["B/R"] = "HEALTH_VETO"
                break
            archive_manifest = json.loads(Path(previous).read_text())
            sample_path = directory/archive_manifest["raw_samples"]["path"]
            if sha(sample_path) != archive_manifest["raw_samples"]["sha256"]:
                raise ValueError("retained SV20 sample corruption")
            samples = tf.io.parse_tensor(tf.io.read_file(str(sample_path)), tf.float64)
            theta, current, valid = functionals(samples)
            if not bool(tf.reduce_all(valid)) or not bool(tf.reduce_all(tf.math.is_finite(current))):
                raise ValueError("invalid required SV20 predictive functional")
            parameters.append(theta)
            values.append(current)
            row["retained_draws"] = (index+1)*250
            if index+1 in (4, 8):
                look = assess_equivalence(tf.transpose(tf.concat(parameters, 0), [1, 0, 2]),
                    tf.transpose(tf.concat(values, 0), [1, 0, 2]),
                    tf.constant(reference["expectations"][-1], tf.float64),
                    tf.constant(reference["allowance"], tf.float64), tf.constant(reference["margins"], tf.float64),
                    comparison_families=4, parameter_dimension=2, function_count=13)
                row["looks"].append(look)
                row["status"] = look["status"]
                report["comparisons"]["B/R"] = look["status"]
                save(out/"report.json", finite_report(report))
                print("B LOOK", row["retained_draws"], look["status"], flush=True)
                if look["status"] in ("EQUIVALENT", "DISCREPANCY"):
                    break
            save(out/"report.json", finite_report(report))
            print("B CHUNK", index+1, flush=True)
    report.update(status="COMPLETED_BOUNDED_INFERENCE", surrogate_hmc_tested=True,
        retained_surrogate_hmc_tested=bool(row["chunks"]), allocator=tf.config.experimental.get_memory_info("GPU:0"))
    save(out/"report.json", finite_report(report))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cpu-budget", type=float, default=12000.)
    parser.add_argument("--wall-budget", type=float, default=12600.)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    if not all(math.isfinite(x) and x > 0 for x in (args.cpu_budget, args.wall_budget)):
        raise ValueError("finite positive budgets required")
    from scripts.prepare_surrogate_hmc_ksc import budgets, run_bounded_worker
    from bayesfilter.inference.surrogate_hmc_sv_program import (
        sv20_endpoint_state, sv20_inference_history, family_remaining)
    started, parent_cpu = time.monotonic(), time.process_time()
    history = sv20_inference_history(ROOT)
    if history["status"] != "NOT_RUN":
        raise ValueError("existing SV20 inference attempt must be preserved; no automatic repeat")
    endpoint = sv20_endpoint_state(ROOT)["qualified"]
    if not endpoint:
        raise ValueError("qualified SV20 endpoint, controls and posterior required")
    endpoint_root = Path(endpoint["report"]).parent
    endpoint_manifest = json.loads((endpoint_root/"run-manifest.json").read_text())
    inputs = dict(endpoint_manifest["inputs"])
    inputs["endpoint_report"] = dict(path=endpoint["report"], sha256=endpoint["report_sha256"])
    inputs["endpoint_manifest"] = dict(path=str(endpoint_root/"run-manifest.json"), sha256=sha(endpoint_root/"run-manifest.json"))
    forecast = endpoint["complete_forecast"]
    family = family_remaining(ROOT)
    total = budgets()[0]
    cpu = min(args.cpu_budget, 12000., family["cpu_worker_seconds"], 54000-total["cpu_worker_seconds"])
    wall = min(args.wall_budget, 12600., family["gpu_seconds"], 226800-total["gpu_seconds"])
    if (not all(math.isfinite(v) and v > 0 for v in (cpu, wall)) or
            forecast["required"]["cpu_worker_seconds"] > cpu or forecast["required"]["wall_seconds"] > wall):
        raise ValueError("complete SV20 inference forecast exceeds allocation")
    if args.dry_run:
        print(json.dumps(dict(cpu_budget=cpu, wall_budget=wall, forecast=forecast, inputs=inputs)))
        return 0
    if not ("-sv20-inference-" in out.name or out.name == "P3-SV20-inference"):
        raise ValueError("versioned SV20 inference output required")
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = sources(["scripts/run_surrogate_hmc_sv.py"])
    source_hashes[PLAN] = sha(ROOT/PLAN)
    snapshot(out, source_hashes)
    manifest = dict(schema="surrogate_hmc_sv20_inference_v1", scope="SV20-T20-N1024", inference_attempt=1,
        command=[sys.executable, __file__, "--output", str(out), "--cpu-budget", str(cpu), "--wall-budget", str(wall)],
        launcher_command=[sys.executable, *sys.argv], source_hashes=source_hashes, inputs=inputs,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=sys.prefix, plan=PLAN, result_file=str(out/"report.json"), cpu_budget=cpu, wall_budget=wall,
        data_version="actual SV20 original claim data; Uniform(.1,.9) priors and probit chart",
        seeds=dict(tuning=[20260922, 35000], verification=[20260922, 35001],
                   retained=[[20260922, 36000+i] for i in range(8)]),
        complete_forecast=forecast, previous_campaign_charge=total, previous_family_remaining=family,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), memory_policy="not initialized",
        gpu_status="pending trusted initialization")
    save(out/"run-manifest.json", manifest)
    save(out/"report.json", dict(status="RUNNING", model_id="SV20-T20", inference_attempt=1,
        comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")},
        surrogate_hmc_tested=False, heuristic_dominance="NOT_RUN", default_readiness=False,
        exact_score_admission=False, all_models_validated=False))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code, watchdog = run_bounded_worker([sys.executable, __file__, "--worker", "--output", str(out)],
        log_path=out/"worker.log", cpu_budget=cpu-(time.process_time()-parent_cpu)-2,
        wall_budget=wall-(time.monotonic()-started)-1)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_used = time.monotonic()-started
    cpu_used = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu
    drift = [name for name, digest in source_hashes.items() if sha(ROOT/name) != digest]
    if drift:
        code = 2
    charge = dict(cpu_worker_seconds=cpu_used, gpu_seconds=wall_used)
    save(out/"accounting.json", dict(charged_this_run=charge, wall_seconds=wall_used, exit_code=code, watchdog=watchdog))
    manifest = json.loads((out/"run-manifest.json").read_text())
    manifest.update(exit_code=code, wall_seconds=wall_used, cpu_worker_seconds=cpu_used, watchdog=watchdog, source_drift=drift)
    save(out/"run-manifest.json", manifest)
    report = json.loads((out/"report.json").read_text())
    if code:
        report.update(status="SOURCE_DRIFT" if drift else "EXECUTION_FAILED", exit_code=code, watchdog=watchdog)
        save(out/"report.json", report)
    print(json.dumps(dict(status=report["status"], output=str(out), charged=charge, exit_code=code)))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
