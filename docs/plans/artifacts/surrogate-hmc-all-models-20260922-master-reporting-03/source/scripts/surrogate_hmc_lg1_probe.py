#!/usr/bin/env python3
"""Bounded full-DCTR GPU/XLA consumer qualification, never retained inference."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, choices=(1024,), default=1024,
                        help="This diagnostic scope and its binding are fixed at N=1024")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    from bayesfilter.inference.surrogate_hmc_program import write_json
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_lg1_tf import make_diagnostic_dataset, make_lg1_consumer
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, neural_force_hmc_transition, neural_force_proposal,
        bind_neural_force_hmc_tuning_runner,
        _evaluate_force,
    )
    dataset = make_diagnostic_dataset()
    write_json(out / "data.json", dataset)
    consumer = make_lg1_consumer(dataset, batch_size=4, particle_count=args.particle_count)
    write_json(out / "consumer.json", consumer.manifest)
    report = dict(classification="P1_mechanics_and_cost_only", status="RUNNING", tensorflow=tf.__version__,
                  memory_policy=memory, device=tf.config.list_logical_devices("GPU")[0].name,
                  tf32=True, jit_compile=True, precision_classification="FP64_INTEGRATOR_FP32_DCTR",
                  checks={}, timings={}, candidate_attempt_consumed=False)
    def save():
        write_json(out / "diagnostics.json", report)
    save()
    def timed(label, function):
        started = time.monotonic()
        result = function()
        for item in tf.nest.flatten(result):
            if tf.is_tensor(item):
                item.numpy()
        report["timings"][label] = time.monotonic() - started
        save()
        return result
    def diagnostic_values(tensor):
        # Invalid original values remain visible as null plus the original status;
        # they must not crash JSON serialization or bypass the kernel's fallback.
        def finite(value):
            if isinstance(value, list):
                return [finite(item) for item in value]
            return None if isinstance(value, float) and not math.isfinite(value) else value
        return finite(tensor.numpy().tolist())
    def same(a, b):
        if a.dtype.is_floating:
            return bool(tf.reduce_all((a == b) | (tf.math.is_nan(a) & tf.math.is_nan(b))))
        return bool(tf.reduce_all(a == b))
    positions = tf.constant([[-2.], [-.5], [.7], [2.]], tf.float64)
    before = consumer.endpoint.function(positions)
    baseline = timed("force_compile_first", lambda: consumer.evaluate_force(positions))
    replay = timed("force_steady", lambda: consumer.evaluate_force(positions))
    value, gradient, status = baseline
    report["force_diagnostic"] = dict(log_posterior=diagnostic_values(value), gradient=diagnostic_values(gradient),
                                      status={k: v.numpy().tolist() for k, v in status.items()})
    report["original_force_all_finite"] = bool(tf.reduce_all(tf.math.is_finite(gradient)))
    report["original_force_all_valid"] = bool(tf.reduce_all(status["force_valid"]))
    effective, fallback = _evaluate_force(consumer.force, positions, dimension=1, dtype=tf.float64)
    report["effective_force"] = diagnostic_values(effective)
    report["effective_force_fallback"] = diagnostic_values(fallback)
    report["checks"]["finite_effective_force"] = bool(tf.reduce_all(tf.math.is_finite(effective)))
    report["checks"]["replay_identical"] = all(same(a,b) for a,b in zip(tf.nest.flatten(baseline), tf.nest.flatten(replay)))
    permutation = tf.constant([2,0,3,1])
    permuted = timed("force_permutation", lambda: consumer.evaluate_force(tf.gather(positions, permutation)))
    report["checks"]["permutation_identical"] = all(same(tf.gather(a,permutation),b)
                                                  for a,b in zip(tf.nest.flatten(baseline),tf.nest.flatten(permuted)))
    changed = tf.concat([positions[:1], tf.constant([[1.1], [-1.1], [0.]], tf.float64)],axis=0)
    replaced = timed("force_row_independence", lambda: consumer.evaluate_force(changed))
    report["checks"]["row_independence"] = all(same(a[0],b[0])
                                              for a,b in zip(tf.nest.flatten(baseline),tf.nest.flatten(replaced)))
    after = consumer.endpoint.function(positions)
    report["checks"]["endpoint_unchanged"] = bool(tf.reduce_all(before.potential == after.potential))
    report["checks"]["independent_identities"] = consumer.force.identity != consumer.endpoint.identity
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint,
                                                 target_scope="M01-LG1-T10-N1024-diagnostic-attempt4")
    write_json(out / "binding.json", dict(binding.payload()))
    report["checks"]["typed_binding"] = True
    scalar = make_lg1_consumer(dataset, batch_size=1, particle_count=args.particle_count)
    scalar_result = timed("scalar_force_compile_first", lambda: scalar.evaluate_force(positions[:1]))
    differences = [float(tf.reduce_max(tf.abs(a[:1]-b))) for a,b in zip(baseline[:2],scalar_result[:2])]
    report["scalar_batch_max_abs_difference"] = max(differences) if all(map(math.isfinite,differences)) else None
    report["scalar_batch_comparison_role"] = "explanatory FP32 comparison; four-chain row independence checked separately"
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=2,
                              inverse_mass_diagonal=(1.,), dtype="float64")
    sig = [tf.TensorSpec([4,1],tf.float64),tf.TensorSpec([4,1],tf.float64)]
    @tf.function(input_signature=sig, jit_compile=True, autograph=False)
    def proposal(q,p):
        return neural_force_proposal(q,p,consumer.force,cfg)
    momentum = tf.constant([[.2],[-.4],[.6],[-.8]],tf.float64)
    forward = timed("proposal_compile_first",lambda:proposal(positions,momentum))
    backward = timed("proposal_reverse",lambda:proposal(forward.position,forward.momentum))
    error = float(tf.maximum(tf.reduce_max(tf.abs(backward.position-positions)),tf.reduce_max(tf.abs(backward.momentum-momentum))))
    report["reversal_absolute_error"] = error
    # Diagnostic mixed-precision tolerance, 32 FP32 ulps at the observed scale.
    tolerance = 32*2**-23*max(1.,float(tf.reduce_max(tf.abs(positions))),float(tf.reduce_max(tf.abs(momentum))))
    report["reversal_tolerance"] = tolerance
    report["checks"]["proposal_involution"] = error <= tolerance
    @tf.function(input_signature=[tf.TensorSpec([4,1],tf.float64),tf.TensorSpec([4],tf.float64),tf.TensorSpec([2],tf.int32)],
                 jit_compile=True,autograph=False)
    def transition(q,u,seed):
        return neural_force_hmc_transition(q,u,consumer.force,consumer.endpoint,cfg,seed)
    first = timed("transition_compile_first",lambda:transition(positions,before.potential,tf.constant([20260918,101])))
    second = timed("transition_steady",lambda:transition(first.position,first.potential,tf.constant([20260918,102])))
    trace = second.trace
    direct_delta = trace.final_potential+trace.final_kinetic-trace.initial_potential-trace.initial_kinetic
    report["checks"]["per_chain_energy"] = bool(tf.reduce_all(tf.abs(trace.delta_h-direct_delta)<1e-11)) and trace.delta_h.shape == (4,)
    report["checks"]["finite_transition"] = bool(tf.reduce_all(tf.math.is_finite(second.position)))
    report["checks"]["cache_matches_endpoint"] = bool(tf.reduce_all(tf.abs(second.potential-consumer.endpoint.function(second.position).potential)<1e-11))
    report["transition"] = {"accepted":trace.accepted.numpy().tolist(),"delta_h":trace.delta_h.numpy().tolist(),
                            "force_fallback":trace.force_fallback.numpy().tolist(),
                            "endpoint_status":{k:v.numpy().tolist() for k,v in trace.endpoint_status.items()}}
    report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    # Exercise the actual public tuner with this full consumer. The outer tuner
    # is a documented non-XLA diagnostic exception; its DCTR kernel remains XLA.
    from bayesfilter.inference import (
        FourChainMeanBandAcceptancePolicy, TensorFlowHMCKernelTuningConfig,
        tune_hmc_kernel, build_retained_bound_hmc_archive_runner_from_tuning_result,
    )
    class Adapter:
        target_scope = binding.target_scope
        def adapter_signature(self):
            return consumer.endpoint.identity + ":" + consumer.force.identity
    tuning_config = TensorFlowHMCKernelTuningConfig(
        parameter_dimension=1, evidence_role="diagnostic_only", mass_window_results=(1,),
        step_adaptation_results=1, verification_results=1, max_leapfrog_steps=1,
        initial_step_size=.01, target_scope=Adapter.target_scope,
        budget_provenance="one-transition-per-stage interface diagnostic; not inference tuning",
        initial_step_size_provenance="convenience mechanics hypothesis, not selected",
        geometry_provenance="unit scale diagnostic, not a tuned mass",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
        target_accept_prob=.7, verification_repair_rounds=0, step_repair_factor=2.,
        mass_shrinkage=.1, covariance_jitter=1e-9, eigenvalue_floor=1e-9,
        max_condition_number=1e8, seed=(20260919,201))
    started = time.monotonic()
    tuning = tune_hmc_kernel(adapter=Adapter(), initial_position=tf.zeros([1],tf.float64),
                            parameter_scales=tf.ones([1],tf.float64), config=tuning_config,
                            runner_binding=binding, output_dir=out / "public-tuner")
    report["timings"]["public_tuner_diagnostic"] = time.monotonic()-started
    try:
        build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=tuning, runner_binding=binding)
    except ValueError as exc:
        if "not admission-capable" not in str(exc):
            raise
        rejection = str(exc)
    else:
        raise AssertionError("diagnostic tuner incorrectly issued a retained-inference handoff")
    report["tuner"] = dict(health_passed=bool(tuning.health_passed),
                           admission_supported=tuning.admission_supported,
                           handoff_eligible=bool(tuning.handoff_eligible),
                           artifact=str(tuning.artifact_manifest_path), rejection=rejection,
                           outer_jit_compile=False, force_jit_compile=True,
                           exception="this P1 configuration is diagnostic-only; use separately verified tuning for inference")
    report["checks"]["diagnostic_tuner_cannot_admit_inference"] = not bool(tuning.handoff_eligible)
    report["checks"]["tuner_accepted_state_finite"] = bool(tf.reduce_all(tf.math.is_finite(tuning.final_raw_state)))
    report["inference_blocker"] = "LG1_SCOPED_SELECTION_REFERENCE_AND_VERIFIED_TUNING_PENDING"
    report["status"] = "PASS_MECHANICS_ONLY" if all(report["checks"].values()) else "INVALID_IMPLEMENTATION_OR_NUMERICS"
    # L=2 is deliberately a timing probe, not a tuned choice. Forecast must be
    # redone at the selected L and include all tuning/reference work.
    report["forecast"] = dict(conditional_l2_2000_warmup_2000_retained_seconds=1.5*(4000*report["timings"]["transition_steady"]+sum(report["timings"].values())),
                               includes_tuning_and_reference=False, inference_launch_allowed=False)
    save()
    print(report["status"], "transition_seconds",report["timings"]["transition_steady"],flush=True)
    return 0 if report["status"] == "PASS_MECHANICS_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
