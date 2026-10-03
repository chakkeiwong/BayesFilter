"""Numerical candidate-set/reload smoke; engineering evidence, not a posterior.

GPU/XLA is the default. --cpu-reference is a deliberate CPU/non-XLA exception.
The small Gaussian grid, widened acceptance band, and draw counts are fixture
choices that bound test cost; consumers must specify their own tuning policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


class GaussianTarget:
    parameter_dim = 2

    def adapter_signature(self):
        return "docs-candidate-retained-gaussian-v1"

    def value_score_capability(self):
        from bayesfilter.inference import ValueScoreCapability
        return ValueScoreCapability(value_score_authority="graph_native",
            xla_hmc_ready=True, full_chain_xla_diagnostic_ready=True,
            target_scope="docs_candidate_retained", runtime_backend="tensorflow",
            evidence_path=__file__, nonclaims=("known Gaussian mechanics fixture",))

    def log_prob_and_grad(self, position):
        import tensorflow as tf
        position = tf.convert_to_tensor(position, tf.float64)
        return -0.5 * tf.reduce_sum(position * position, axis=-1), -position

    def target_status_telemetry(self, position):
        import tensorflow as tf
        shape = tf.shape(position)[:-1]
        return {"status_code": tf.zeros(shape, tf.int32),
            "valid_pre_regularized_score": tf.ones(shape, tf.bool),
            "floor_count_value": tf.zeros(shape, tf.int32)}


def run_example(output_dir: Path, *, cpu_reference: bool = False):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if cpu_reference:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not cpu_reference)
    from bayesfilter.inference import (
        HMCAcceptancePolicy, HMCCandidateExecutionConfig, HMCControllerConfig,
        PrecomputedMassArtifact, bind_hmc_candidate_set_execution, tune_hmc_kernel,
        build_retained_bound_hmc_archive_runner_from_candidate_set_result,
        build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
        load_hmc_candidate_retained_runner,
    )
    from bayesfilter.inference.hmc import FullChainHMCConfig, build_independent_chain_tfp_hmc_runner

    target = GaussianTarget()
    mass = PrecomputedMassArtifact(position=[0.,0.], covariance=tf.eye(2, dtype=tf.float64),
        factor=tf.eye(2, dtype=tf.float64), adapter_signature=target.adapter_signature(),
        position_role="reference_center", covariance_source="known Gaussian fixture")
    binding = bind_hmc_candidate_set_execution(adapter=target,
        initial_position=tf.constant([[-1.,-.5],[-.3,.2],[.4,-.2],[1.,.5]], tf.float64),
        mass_artifact=mass, target_scope="docs_candidate_retained",
        target_lineage={"model": "standard_normal_2d", "data": "none", "prior": "target_itself"},
        source_paths=[__file__], scope_id="docs-bridge", search_id="example-1",
        epsilon_domain=(.01,1.95), repair_factor=1.1, max_repairs_per_family=0,
        config=HMCCandidateExecutionConfig(measurement_num_results=128,
            verification_num_results=128, num_warmup_steps=8, seed=(20260914,11),
            acceptance_policy=HMCAcceptancePolicy(practical_region=(.55,.85), repair_region=(.5,.9)),
            target_status_trace_policy="per_chain_step", use_xla=not cpu_reference,
            non_xla_reason="CPU reference smoke" if cpu_reference else None))
    run = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        candidate_set_adapter=binding.typed_adapter, output_dir=output_dir / "tuning",
        config=HMCControllerConfig(primary_l_grid=(2,3),
            epsilon_by_l=((2,(1.1,1.3,1.5)),(3,(1.1,1.3,1.5))),
            total_budget_units=40, repair_reserve_units=3))
    if not run.result.verified_candidate_ids:
        raise RuntimeError("no verified member in the bounded engineering fixture")
    candidate_id = run.result.verified_candidate_ids[0]  # Representative, no ranking claim.
    builder = (build_retained_bound_hmc_archive_runner_from_candidate_set_result if cpu_reference
               else build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result)
    runner = builder(candidate_set_result=run.result, candidate_id=candidate_id, retained_binding=binding)
    member_path = runner.export(output_dir / "member.json")
    first = runner.run(num_results=12, seed=(71,1), output_dir=output_dir / "retained_1")
    restored = load_hmc_candidate_retained_runner(member_path, adapter=GaussianTarget(),
                                                 claim_eligible=not cpu_reference)
    second = restored.run(num_results=12, seed=(71,2), output_dir=output_dir / "retained_2",
                           previous_archive=first["archive_path"])
    # Match both compilation mode and device: stateless seeds do not promise
    # identical random draws between XLA and non-XLA implementations.
    with tf.device("/CPU:0" if cpu_reference else "/GPU:0"):
        reference = build_independent_chain_tfp_hmc_runner(binding._active_adapter,
            binding.initial_active_state, FullChainHMCConfig(num_results=12,
                num_burnin_steps=0, step_size=runner.candidate.epsilon,
                num_leapfrog_steps=runner.candidate.leapfrog_steps, seed=(71,2),
                use_xla=not cpu_reference, target_scope="docs_candidate_retained"))
        reference_result = reference.run(current_state=first["final_active_state"], mode="serial")
    max_error = float(tf.reduce_max(tf.abs(second["samples"] - reference_result.samples)))
    if max_error > 1e-10:  # Existing float64 reference-comparison tolerance.
        raise AssertionError(f"retained transition parity failed: {max_error}")
    from bayesfilter.inference.hmc_candidate_set_execution import _trace_from_payload, _tensor_from_payload
    archive = json.loads(Path(second["archive_path"]).read_text())
    trace = _trace_from_payload(archive["trace"])
    samples = _tensor_from_payload(archive["active_samples"])
    initial = _tensor_from_payload(archive["initial_active_state"])
    # Independent deterministic leapfrog reference, using the actual captured
    # momentum. This checks CPU versus GPU arithmetic without comparing RNGs.
    with tf.device("/CPU:0"):
        q0 = tf.concat([initial[None], samples[:-1]], axis=0)
        p0 = tf.identity(trace["initial_momentum"])
        q, p = q0, p0
        eps = tf.constant(runner.candidate.epsilon, tf.float64)
        for _ in range(runner.candidate.leapfrog_steps):
            p = p - .5 * eps * q
            q = q + eps * p
            p = p - .5 * eps * q
        expected_log_accept = .5 * tf.reduce_sum(q0*q0 + p0*p0 - q*q - p*p, axis=-1)
        position_error = float(tf.reduce_max(tf.abs(q - trace["proposed_state"])))
        momentum_error = float(tf.reduce_max(tf.abs(p - trace["final_momentum"])))
        acceptance_error = float(tf.reduce_max(tf.abs(expected_log_accept - trace["log_accept_ratio"])))
    if max(position_error, momentum_error, acceptance_error) > 1e-10:
        raise AssertionError("deterministic CPU leapfrog/energy reference failed")
    compiled_counts = [r._runner.experimental_get_tracing_count()
        for ensemble in binding._runners.values() for r in ensemble._runners]
    if not all(count == 1 for count in compiled_counts):
        raise AssertionError("unbounded/repeated runner tracing")
    return {"status": "passed", "verified_candidate_ids": run.result.verified_candidate_ids,
        "candidate_count": len(run.result.candidates), "selected_member": candidate_id,
        "member_hash": runner.member_hash, "member_path": str(member_path),
        "retained_archives": [first["archive_path"], second["archive_path"]],
        "maximum_same_backend_transition_error": max_error,
        "cpu_reference_errors": {"position": position_error, "momentum": momentum_error, "log_accept": acceptance_error},
        "compiled_runner_trace_counts": compiled_counts,
        "memory_policy": memory, "runtime_policy": binding._runtime,
        "source_closure": binding._spec["source_closure"],
        "execution_config": binding.config.payload(),
        "gpu_allocator": tf.config.experimental.get_memory_info("GPU:0") if not cpu_reference else None,
        "inference_status": "engineering parity only; no statistical ranking or posterior admission"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-reference", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"command": sys.argv, "git_commit": subprocess.check_output(
        ["git","rev-parse","HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "python": sys.executable, "cpu_reference": args.cpu_reference,
        "gpu_intentionally_hidden": args.cpu_reference, "data_version": "known_Gaussian_fixture_v1",
        "plan": "docs/plans/bayesfilter-typed-candidate-retained-bridge-plan-2026-09-14.md"}
    try:
        manifest["result"] = run_example(args.output_dir, cpu_reference=args.cpu_reference)
    except Exception as exc:
        manifest["result"] = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        (args.output_dir / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["result"]["status"], "wall_seconds": manifest["wall_seconds"]}))


if __name__ == "__main__":
    main()
