#!/usr/bin/env python3
"""Qualify an actual full-DCTR actual-SV consumer; no retained inference or tuning."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def json_values(value):
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if hasattr(value, "numpy"):
        return json_values(value.numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior-root", required=True, type=Path)
    parser.add_argument("--wall-budget", type=float, default=600.)
    parser.add_argument("--cpu-budget", type=float, default=450.)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json
    reference_manifest = json.loads((args.reference_root / "run-manifest.json").read_text())
    for name, digest in reference_manifest["source_hashes"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("reference source drift: " + name)
    posterior_manifest = json.loads((args.posterior_root / "run-manifest.json").read_text())
    posterior_report = json.loads((args.posterior_root / "report.json").read_text())
    if posterior_report.get("status") != "QUALIFIED_MODEL_POSTERIOR_REFERENCE" or posterior_report.get("qualified") is not True:
        raise ValueError("qualified SV20 posterior reference required")
    if posterior_manifest["parent_manifest_sha256"] != file_hash(args.reference_root / "run-manifest.json"):
        raise ValueError("posterior/likelihood reference parent mismatch")
    for name, digest in posterior_manifest["source_hashes"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("posterior source drift: " + name)
    data_path = args.reference_root / "calibration/data.json"
    data = json.loads(data_path.read_text())
    if (data["model_id"], data["horizon"], data["attempt"], data["partition"]) != ("SV20", 20, 1, "calibration"):
        raise ValueError("wrong SV20 qualification dataset")
    if file_hash(data_path) != reference_manifest["partitions"]["calibration"]["data_sha256"]:
        raise ValueError("calibration data drift")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_sv_consumer import make_sv_consumer
    from bayesfilter.inference.surrogate_hmc_dctr_selection import protection_controls
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, bind_neural_force_hmc_tuning_runner,
        neural_force_proposal, neural_force_hmc_transition,
    )
    consumer = make_sv_consumer(data, controls=protection_controls(1.), batch_size=4)
    write_json(out / "consumer.json", consumer.manifest)
    sources = dict(consumer.manifest["endpoint"]["sources"])
    for name in ("scripts/qualify_surrogate_hmc_sv_consumer.py",
                 "docs/plans/surrogate-hmc-sv20-consumer-2026-09-20.md",
                 "tests/test_surrogate_hmc_sv_consumer.py",
                 "bayesfilter/inference/neural_force_hmc.py",
                 "bayesfilter/inference/surrogate_hmc_dctr_selection.py"):
        sources[name] = file_hash(Path(name))
    for name in sources:
        destination = out / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    write_json(out / "executed-source/sha256.json", sources)
    started = time.monotonic()
    cpu_started = time.process_time()
    report = dict(status="RUNNING", classification="consumer_mechanics_and_cost_only",
        model="SV20", candidate_attempt_consumed=False, checks={}, timings={},
        tensorflow=tf.__version__, memory_policy=memory,
        device=tf.config.list_logical_devices("GPU")[0].name, jit_compile=True, tf32=True,
        control_status="unselected hypothesis; SV-specific calibration still required",
        heuristic_dominance="NOT_RUN", retained_inference=False)
    # Preserve provenance even if compilation or a numerical call fails.
    write_json(out / "run-manifest.json", dict(command=sys.argv, environment=sys.executable,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        status="RUNNING", data_path=str(data_path.resolve()), data_sha256=file_hash(data_path),
        plan="docs/plans/surrogate-hmc-sv20-consumer-2026-09-20.md",
        result=str(out/"report.json"), source_hashes=sources, memory_policy=memory,
        gpu_status=report["device"], cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        parent_manifest_sha256=file_hash(args.reference_root/"run-manifest.json"),
        posterior_manifest_sha256=file_hash(args.posterior_root/"run-manifest.json"),
        tf32=True, jit_compile=True, wall_budget=args.wall_budget, cpu_budget=args.cpu_budget))

    def save():
        write_json(out / "report.json", json_values(report))

    def timed(name, callback):
        start = time.monotonic()
        result = callback()
        for tensor in tf.nest.flatten(result):
            tensor.numpy()
        report["timings"][name] = time.monotonic() - start
        report["cpu_worker_seconds"] = time.process_time() - cpu_started
        allocator = tf.config.experimental.get_memory_info("GPU:0")
        report["allocator"] = allocator
        if (time.monotonic() - started > args.wall_budget or report["cpu_worker_seconds"] > args.cpu_budget
                or allocator["peak"] > 4 * 1024**3):
            report["status"] = "RESOURCE_LIMIT"
            save()
            raise RuntimeError("declared consumer resource limit exceeded")
        save()
        print(name, report["timings"][name], flush=True)
        return result

    def equal(left, right):
        tf.nest.assert_same_structure(left, right)
        return all(bool(tf.reduce_all(a == b)) for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)))

    physical = tf.constant([[.3, .3], [.5, .5], [.7, .7], [.8, .2]], tf.float64)
    z = tf.sqrt(tf.constant(2., tf.float64)) * tf.math.erfinv(2 * (physical - .1) / .8 - 1)
    baseline = timed("consumer_compile_first", lambda: consumer.evaluate_force(z))
    report["baseline"] = json_values(dict(value=baseline[0], score=baseline[1], status=baseline[2], physical=physical, z=z))
    report["original_force_healthy"] = bool(tf.reduce_all(baseline[2]["force_valid"]))
    save()
    replay = timed("consumer_replay", lambda: consumer.evaluate_force(z))
    report["checks"]["bitwise_replay"] = equal(baseline, replay)
    direct = consumer.fixture.compile(4, coordinates="unconstrained")
    direct_output = timed("direct_full_fixture", lambda: direct(tf.cast(z, tf.float32)))
    report["checks"]["actual_call_chain_parity"] = equal(baseline, direct_output)
    permutation = tf.constant([2, 0, 3, 1])
    permuted = timed("permuted_rows", lambda: consumer.evaluate_force(tf.gather(z, permutation)))
    report["checks"]["permutation"] = equal(tf.nest.map_structure(lambda x: tf.gather(x, permutation), baseline), permuted)
    changed = tf.tensor_scatter_nd_add(z, [[3, 0]], tf.constant([.1], tf.float64))
    changed_output = timed("one_row_changed", lambda: consumer.evaluate_force(changed))
    report["checks"]["row_independence"] = equal(tf.nest.map_structure(lambda x: x[:3], baseline),
                                                             tf.nest.map_structure(lambda x: x[:3], changed_output))
    before = timed("endpoint_compile_first", lambda: consumer.endpoint.function(z))
    consumer.evaluate_force(changed)
    after = consumer.endpoint.function(z)
    report["checks"]["endpoint_independent"] = equal(before, after)
    field = consumer.force.function(z)
    report["checks"]["field_sign_status"] = equal((field.force, field.force_valid),
        (-tf.cast(baseline[1], tf.float64), baseline[2]["force_valid"]))
    exact_logp, exact_score = consumer.evaluate_reference(z)
    report["checks"]["qualified_reference_endpoint"] = bool(tf.reduce_all(before.potential == -exact_logp))
    report["checks"]["reference_force_sign"] = bool(tf.reduce_all(consumer.exact_force.function(z).force == -exact_score))
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint,
                                                 target_scope="SV20-attempt1-T20-N1024-calibration-qualification")
    write_json(out / "binding.json", dict(binding.payload()))
    report["checks"]["independent_typed_bindings"] = consumer.force.identity != consumer.endpoint.identity
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=2,
                              inverse_mass_diagonal=(1., 1.), dtype="float64")

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)] * 2, jit_compile=True, autograph=False)
    def proposal(q, p):
        return neural_force_proposal(q, p, consumer.force, cfg)

    momentum = tf.constant([[.2, -.1], [-.4, .3], [.6, -.5], [-.8, .7]], tf.float64)
    forward = timed("proposal_compile_first", lambda: proposal(z, momentum))
    backward = timed("proposal_reverse", lambda: proposal(forward.position, forward.momentum))
    error = float(tf.maximum(tf.reduce_max(tf.abs(backward.position - z)),
                            tf.reduce_max(tf.abs(backward.momentum - momentum))))
    tolerance = 32 * 2**-23 * max(1., float(tf.reduce_max(tf.abs(z))), float(tf.reduce_max(tf.abs(momentum))))
    report.update(reversal_absolute_error=error, reversal_tolerance=tolerance)
    report["checks"]["proposal_involution"] = error <= tolerance

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                  tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def transition(q, potential, seed):
        return neural_force_hmc_transition(q, potential, consumer.force, consumer.endpoint, cfg, seed)

    first = timed("transition_compile_first", lambda: transition(z, before.potential, tf.constant([20260920, 201])))
    report["transitions"] = json_values([first.trace._asdict()])
    save()
    second = timed("transition_steady", lambda: transition(first.position, first.potential, tf.constant([20260920, 202])))
    trace = second.trace
    difference = trace.final_potential + trace.final_kinetic - trace.initial_potential - trace.initial_kinetic
    report["checks"]["per_chain_energy"] = trace.delta_h.shape == (4,) and bool(tf.reduce_all(tf.abs(trace.delta_h - difference) < 1e-10))
    report["checks"]["cache_matches_endpoint"] = bool(tf.reduce_all(tf.abs(second.potential - consumer.endpoint.function(second.position).potential) < 1e-10))
    report["checks"]["finite_states"] = bool(tf.reduce_all(tf.math.is_finite(second.position)))
    report["checks"]["endpoint_transition_health"] = all(bool(tf.reduce_all(
        tf.math.is_finite(step.trace.delta_h) & ~step.trace.endpoint_status["numerical_failure"]))
        for step in (first, second))
    report["exact_endpoint_logp"] = json_values(exact_logp)
    report["transitions"] = json_values([first.trace._asdict(), second.trace._asdict()])
    report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    report["qualified"] = all(report["checks"].values())
    report["status"] = "QUALIFIED_CONSUMER_MECHANICS" if report["qualified"] else "QUALIFICATION_FAILED"
    report["field_status"] = "HEALTHY_AT_PROBES" if report["original_force_healthy"] else "REPAIR_REQUIRED"
    report["surrogate_hmc_tested"] = False
    report["cpu_worker_seconds"] = time.process_time() - cpu_started
    for name, digest in sources.items():
        if file_hash(Path(name)) != digest:
            raise ValueError("executed source drift: " + name)
    report["wall_seconds"] = time.monotonic() - started
    save()
    write_json(out / "run-manifest.json", dict(command=sys.argv, environment=sys.executable,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        data_path=str(data_path.resolve()), data_sha256=file_hash(data_path), seeds=dict(data=data["data_seed"], fixed_inputs=consumer.manifest["fixture"]["seeds"], transitions=[[20260920,201],[20260920,202]]),
        plan="docs/plans/surrogate-hmc-sv20-consumer-2026-09-20.md",
        result=str(out / "report.json"), output=str(out.resolve()), source_hashes=sources,
        gpu_status=report["device"], memory_policy=memory, tf32=True, jit_compile=True,
        wall_seconds=report["wall_seconds"], cpu_worker_seconds=report["cpu_worker_seconds"],
        data_version="frozen actual SV20 raw returns, once rounded FP32",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        parent_reference=str(args.reference_root.resolve()), parent_manifest_sha256=file_hash(args.reference_root/"run-manifest.json"),
        posterior_reference=str(args.posterior_root.resolve()), posterior_manifest_sha256=file_hash(args.posterior_root/"run-manifest.json")))
    print(json.dumps(dict(qualified=report["qualified"], original_force_healthy=report["original_force_healthy"], checks=report["checks"])), flush=True)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
