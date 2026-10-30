"""AR2 retained surrogate-HMC test through the public typed-force tuner."""
import json
from pathlib import Path
import sys
import time

import tensorflow as tf

from bayesfilter.inference import (
    TensorFlowHMCKernelTuningConfig, VerifiedTensorFlowHMCConfig, FourChainMeanBandAcceptancePolicy,
    bind_neural_force_hmc_tuning_runner, tune_hmc_kernel, load_verified_tensorflow_hmc_result,
    build_retained_bound_hmc_archive_runner_from_tuning_result, BoundRetainedHMCArchiveConfig,
)
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.testing.surrogate_hmc_ar2_sampling import load_reference, selected_consumer
from bayesfilter.testing.surrogate_hmc_ar2_calibration import load_cost_contract, validate_cost_contract
from bayesfilter.testing.surrogate_hmc_ar2_reference import make_reference, PRIOR_SD
from bayesfilter.testing.surrogate_hmc_nonlinear_inference_metrics import scalar_functionals
from bayesfilter.testing.surrogate_hmc_ar2_assessment import METHOD, assess

CALIBRATION_ROOT = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-validation-06/ar2-calibration-repair-r2"


def bind_arm(consumer, arm, *, run_index=1):
    if arm not in ("A", "B"):
        raise ValueError("AR2 model-endpoint comparison supports only A and B")
    if type(run_index) is not int or run_index < 1:
        raise ValueError("positive run index required")
    scope = f"AR2-T10-N64-normal-priors-claim-run{run_index}:"+arm
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force if arm == "B" else consumer.exact_force,
                                               target=consumer.endpoint, target_scope=scope)

    class Adapter:
        target_scope = scope

        def adapter_signature(self):
            return scope+":"+binding.binding_hash

    return Adapter(), binding


def tuning_config(scope, seed):
    nomination = TensorFlowHMCKernelTuningConfig(parameter_dimension=2,
        evidence_role="diagnostic_candidate_screen", mass_window_results=(128, 256),
        step_adaptation_results=256, verification_results=128, max_leapfrog_steps=8,
        initial_step_size=.2, budget_provenance="AR2 measured complete-transition forecast, up to L8 and 7000 tuning/verification transitions",
        initial_step_size_provenance=".2 initial normalized-coordinate step hypothesis; public tuner adapts independently per arm",
        geometry_provenance="prior-scale warm start followed by own-target windowed covariance",
        target_scope=scope, acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(.5, 1.), per_chain_band=(.3, 1.)),
        target_accept_prob=.7, verification_repair_rounds=0, step_repair_factor=2.,
        mass_shrinkage=.1, covariance_jitter=0., eigenvalue_floor=64*sys.float_info.epsilon*PRIOR_SD**2,
        max_condition_number=sys.float_info.epsilon**-.5,
        seed=(20260923, seed), use_xla=True)
    starts = tuple(tuple(factor*PRIOR_SD for factor in row) for row in ((-1., -1.), (-.33, .33), (.33, -.33), (1., 1.)))
    return VerifiedTensorFlowHMCConfig(nomination, starts,
        "four dispersed prior-scale starts; target unchanged; fresh seeds; all warmup and verification archived and excluded", 2000, 2000, (20260923, seed+1), 1000.)


def check_inference_prerequisites(cost, calibration, cost_hash, available, arm):
    if arm not in ("A", "B"):
        raise ValueError("unknown AR2 inference arm")
    validate_cost_contract(cost)
    if (calibration.get("qualified") is not True or calibration.get("status") != "AR2_SCOPED_UNCERTAINTY_QUALIFIED"
            or calibration.get("cost_report_sha256") != cost_hash
            or calibration.get("protocol", {}).get("method") != METHOD
            or calibration.get("protocol", {}).get("looks") != cost["precision"]["retained_looks"]):
        raise ValueError("fresh scoped uncertainty prerequisite failed")
    forecast = cost["conservative_inference_forecast"][arm]
    if (forecast["cpu_worker_seconds"]+200. > available["cpu_worker_seconds"]
            or forecast["wall_seconds"] > available["gpu_seconds"]):
        raise ValueError("AR2 inference forecast exceeds its remaining reservation")


def metric_guard_diagnostic(config):
    """Dedicated non-harm check of this scope's proposal-metric protection.

    This uses the tuner's actual regularization, not a duplicate implementation.
    It is an independent diagnostic, not an alternative tuning authority.
    """
    from bayesfilter.inference.hmc_tensorflow_tuning import _regularize_covariance
    prior = tf.eye(2, dtype=tf.float64)*PRIOR_SD**2
    healthy, _, _ = _regularize_covariance(prior, target=prior, config=config.nomination)
    rank_one = tf.linalg.diag(tf.constant([0., PRIOR_SD**2], tf.float64))
    protected, _, eigenvalues = _regularize_covariance(rank_one, target=prior, config=config.nomination)
    expected = tf.linalg.diag(tf.constant([.1*PRIOR_SD**2, PRIOR_SD**2], tf.float64))
    tf.debugging.assert_near(healthy, prior, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_near(protected, expected, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_greater_equal(tf.reduce_min(eigenvalues), tf.constant(.1*PRIOR_SD**2*(1-1e-12), tf.float64))
    return dict(passed=True, healthy_max_change=float(tf.reduce_max(tf.abs(healthy-prior))),
                rank_one_protected_eigenvalues=eigenvalues.numpy().tolist(),
                reason="C=.9 E+.1 P0 >= .1 P0 for PSD E; no additive jitter needed; eigenfloor is numerical backup")


def _read_tensor(directory, record):
    path = directory/record["path"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("retained tensor checksum mismatch")
    return tf.io.parse_tensor(tf.io.read_file(str(path)), tf.as_dtype(record.get("dtype", "float64")))


def inference_stage(output, *, cost_root, calibration_root, arm):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    cost_path, calibration_path = Path(cost_root)/"report.json", Path(calibration_root)/"report.json"
    cost = load_cost_contract(cost_root)
    calibration = json.loads(calibration_path.read_text())
    master_manifest = json.loads((out.parent/"run-manifest.json").read_text())
    check_inference_prerequisites(cost, calibration, file_hash(cost_path), master_manifest["reservation"], arm)
    looks = cost["precision"]["retained_looks"]
    source_manifest = json.loads((Path(calibration_root).parent/"run-manifest.json").read_text())
    for path in ("bayesfilter/testing/surrogate_hmc_ar2_assessment.py",
                 "bayesfilter/testing/surrogate_hmc_scoped_assessment.py", "bayesfilter/testing/surrogate_hmc_mcse_randomization.py",
                 "bayesfilter/testing/surrogate_hmc_nonlinear_inference_metrics.py", "bayesfilter/inference/hmc_posterior_diagnostics.py"):
        if file_hash(ROOT/path) != source_manifest["source_hashes"].get(path):
            raise ValueError("calibrated diagnostic source drift: "+path)
    data, reference, arrays = load_reference()
    consumer = selected_consumer(data)
    write_json(out/"consumer.json", consumer.manifest)
    write_json(out/"data.json", data)
    run_index = int(out.parent.name.rsplit("-", 1)[-1])
    adapter, binding = bind_arm(consumer, arm, run_index=run_index)
    seed = (21000 if arm == "A" else 22000)+10000*(run_index-1)
    config = tuning_config(adapter.target_scope, seed)
    metric_check = metric_guard_diagnostic(config)
    comparison = arm+"/R"
    report = dict(status="TUNING", qualified=False, arm=arm, run_index=run_index, tuning={}, chunks=[], looks=[],
        data_sha256=file_hash(out/"data.json"), cost_report_sha256=file_hash(cost_path),
        calibration_report_sha256=file_hash(calibration_path), tuning_config=config.payload(),
        retained_looks=looks, retained_chunk_size=250, inference_launched=True, posterior_comparisons={comparison: "NOT_RUN"},
        metric_protection_check=metric_check,
        comparison_role="primary surrogate-HMC B/R" if arm == "B" else "descriptive exact-gradient A/R comparator",
        prior_scales=[PRIOR_SD, PRIOR_SD], retained_seeds=[[20260923, seed+100+i] for i in range(looks[-1]//250)])
    write_json(output, report)
    begin = time.monotonic()
    try:
        result = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.), config=config,
            parameter_scales=(PRIOR_SD, PRIOR_SD), runner_binding=binding, output_dir=out/"tuning")
    except ValueError as exc:
        if "nomination failed" not in str(exc):
            raise
        report.update(status="AR2_TUNING_REJECTED", tuning=dict(passed=False, detail=str(exc), wall_seconds=time.monotonic()-begin),
            next_action="Existing master: inspect tuning failure and use the bounded tuning repair; force-gradient disagreement is not a veto")
        report["posterior_comparisons"][comparison] = "TUNING_REJECTED"
        return report
    report["tuning"] = dict(passed=result.passed, checks=result.checks, manifest=result.artifact_manifest_path,
        step_size=float(result.nomination.step_size), leapfrog_steps=int(result.nomination.num_leapfrog_steps),
        metric_covariance=result.nomination.covariance.numpy().tolist(), wall_seconds=time.monotonic()-begin)
    report["tuning"]["adaptation_force_fallback_count"] = int(result.nomination.adaptation_force_fallback_count)
    report["tuning"]["nomination_force_fallback_count"] = int(result.nomination.verification_force_fallback_count)
    tuning_manifest = json.loads(Path(result.artifact_manifest_path).read_text())
    report["tuning"]["fresh_rhat"] = {
        name: _read_tensor(out/"tuning", record).numpy().tolist()
        for name, record in tuning_manifest["rhat"].items()}
    fresh_fallbacks = {part: int(tf.reduce_sum(tf.cast(_read_tensor(out/"tuning", tuning_manifest[part]["force_fallback"]), tf.int32)))
                      for part in ("warmup", "verification")}
    report["tuning"]["fresh_force_fallback_counts"] = fresh_fallbacks
    write_json(output, finite_report(report))
    if not result.passed:
        report.update(status="AR2_TUNING_REJECTED", next_action="Existing master: inspect fresh verification and run the bounded tuning repair")
        report["posterior_comparisons"][comparison] = "TUNING_REJECTED"
        return finite_report(report)
    if any(fresh_fallbacks.values()):
        report.update(status="AR2_TUNING_FORCE_VALIDITY_VETO",
                      next_action="Existing master: localize canonical force invalidity before retuning")
        report["posterior_comparisons"][comparison] = "FORCE_VALIDITY_VETO"
        return finite_report(report)
    durable = load_verified_tensorflow_hmc_result(out/"tuning/manifest.json", adapter=adapter, runner_binding=binding)
    runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=durable, runner_binding=binding)
    statistics, _, _ = make_reference(data["horizon"])
    observations = tf.constant(data["observations"], tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([250, 4, 2], tf.float64)], jit_compile=True, autograph=False)
    def functionals(samples):
        points = tf.reshape(samples, [1000, 2])
        _, predictive = statistics(points, observations)
        values = scalar_functionals(points, predictive, arrays["means"], arrays["scales"], arrays["cuts"])
        return tf.reshape(values, [250, 4, 13])

    previous, parameters, values, acceptances = None, [], [], []
    report["status"] = "RETAINED_SAMPLING"
    for index in range(looks[-1]//250):
        directory = out/f"chunk-{index+1:02d}"
        start = time.monotonic()
        archive = runner.run(BoundRetainedHMCArchiveConfig(250, (20260923, seed+100+index), directory,
            "prospective AR2 frozen-kernel model-posterior test; warmup and verification excluded", previous))
        previous = archive["archive_manifest_path"]
        manifest = json.loads(Path(previous).read_text())
        fallback_count = int(tf.reduce_sum(tf.cast(_read_tensor(directory, manifest["chunk"]["force_fallback"]), tf.int32)))
        chunk_health = archive["health_passed"] and fallback_count == 0
        report["chunks"].append(dict(manifest=previous, health_passed=chunk_health,
            force_fallback_count=fallback_count, graph_trace_count=manifest["chunk_graph_trace_count"],
            wall_seconds=time.monotonic()-start))
        if not chunk_health:
            report["status"] = "AR2_RETAINED_HEALTH_VETO"
            report["posterior_comparisons"][comparison] = "HEALTH_VETO"
            break
        samples = _read_tensor(directory, manifest["raw_samples"])
        parameters.append(samples)
        values.append(functionals(samples))
        acceptances.append(tf.cast(_read_tensor(directory, manifest["chunk"]["is_accepted"]), tf.float64))
        count = (index+1)*250
        report["retained_draws_per_chain"] = count
        if count in looks:
            look = assess(tf.transpose(tf.concat(parameters, 0), [1, 0, 2]), tf.transpose(tf.concat(values, 0), [1, 0, 2]),
                tf.constant(reference["reference_expectations"], tf.float64), tf.constant(reference["empirical_allowance"], tf.float64),
                tf.constant(reference["margins"], tf.float64), seed=[20260923, seed+1000+count])
            look.update(draws_per_chain=count, posterior_mean=tf.reduce_mean(tf.concat(parameters, 0), [0, 1]).numpy().tolist(),
                acceptance_by_chain=tf.reduce_mean(tf.concat(acceptances, 0), 0).numpy().tolist(),
                comparator_only=arm == "A")
            report["looks"].append(look)
            report["posterior_comparisons"][comparison] = look["status"]
            write_json(output, finite_report(report))
            print("AR2", arm, "LOOK", count, look["status"], flush=True)
            if look["status"] in ("EQUIVALENT", "DISCREPANCY"):
                break
        write_json(output, finite_report(report))
        print("AR2", arm, "CHUNK", index+1, flush=True)
    if report["status"] != "AR2_RETAINED_HEALTH_VETO":
        report["status"] = "AR2_POSTERIOR_TEST_COMPLETED"
    report.update(qualified=bool(report["looks"] and report["looks"][-1]["status"] == "EQUIVALENT"),
        ranking="NOT_ESTABLISHED", default_readiness=False, all_models_validated=False,
        next_action="Existing master: classify retained evidence, run independent comparator, then finite-target/reference and remaining-model tasks",
        allocator=tf.config.experimental.get_memory_info("GPU:0"))
    return finite_report(report)
