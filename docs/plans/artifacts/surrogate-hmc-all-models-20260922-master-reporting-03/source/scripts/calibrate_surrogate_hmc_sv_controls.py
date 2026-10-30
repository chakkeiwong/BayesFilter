#!/usr/bin/env python3
"""Scoped full-DCTR actual-SV protection curves; no inference or tuning admission."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import resource
import shutil
import subprocess

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def finite_json(value):
    if isinstance(value, dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def no_new_failures(stronger_healthy, weaker_healthy):
    """Every individually healthy comparator point must remain healthy."""
    if len(stronger_healthy) != len(weaker_healthy) or not stronger_healthy:
        raise ValueError("aligned nonempty stress rows are required")
    return all(not weak or strong for strong, weak in zip(stronger_healthy, weaker_healthy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--wall-budget", type=float, default=600.)
    parser.add_argument("--cpu-budget", type=float, default=600.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from bayesfilter.inference.surrogate_hmc_program import write_json, file_hash
    reference_manifest = json.loads((args.reference_root / "run-manifest.json").read_text())
    for name, digest in reference_manifest["source_hashes"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("reference source drift: " + name)
    qualified = json.loads((args.qualification_root / "report.json").read_text())
    qualification_manifest = json.loads((args.qualification_root / "run-manifest.json").read_text())
    if qualified.get("qualified") is not True or qualified.get("original_force_healthy") is not True:
        raise ValueError("healthy actual-SV consumer qualification required")
    if qualification_manifest["parent_manifest_sha256"] != file_hash(args.reference_root / "run-manifest.json"):
        raise ValueError("qualification/reference mismatch")
    for name, digest in qualification_manifest["source_hashes"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("qualified consumer source drift: " + name)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
    from bayesfilter.testing.surrogate_hmc_sv_consumer import make_sv_fixture_target
    from bayesfilter.testing.surrogate_hmc_sv_reference import make_sv_reference
    from bayesfilter.inference.surrogate_hmc_dctr_selection import MULTIPLIERS, protection_controls, issue_selection
    scope = dict(model="SV20", horizon=20, attempt=1, state_dimension=1, parameter_dimension=2,
        particle_count=1024, chunk_extent=1024, dtype="float32", tf32=True, jit_compile=True,
        route="canonical_batch_fused_value_score", reset_contract_id="contract_e_chol_v1",
        law="stationary x0; transition gamma*x+N(0,1); raw Normal(0,beta^2*exp(x)); Uniform(.1,.9) gamma,beta; probit chart",
        data_regime="synthetic actual SV20; separate calibration/validation/claim partitions",
        control_family="full_canonical_geometric_protection_curve_v1")
    hashes = source_closure()
    for name in ("bayesfilter/inference/surrogate_hmc_dctr_selection.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_tf.py",
                 "bayesfilter/testing/surrogate_hmc_sv_reference.py",
                 "bayesfilter/testing/surrogate_hmc_sv_consumer.py",
                 "bayesfilter/inference/neural_force_hmc.py",
                 "scripts/calibrate_surrogate_hmc_sv_controls.py",
                 "scripts/calibrate_surrogate_hmc_linear_controls.py",
                 "docs/plans/surrogate-hmc-sv20-controls-2026-09-20.md"):
        hashes[name] = file_hash(Path(name))
    for name in hashes:
        destination = args.output / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    write_json(args.output / "scope.json", scope)
    manifest = dict(command=[sys.executable, *sys.argv], environment=sys.executable,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes=hashes, reference_manifest=str((args.reference_root / "run-manifest.json").resolve()),
        data_version="SV20 attempt1 replicate0 once-rounded FP32 observations",
        plan="docs/plans/surrogate-hmc-sv20-controls-2026-09-20.md", result=str(args.output / "report.json"),
        gpu_status=[device.name for device in tf.config.list_physical_devices("GPU")],
        memory_policy=memory, tf32=True, jit_compile=True, tensorflow=tf.__version__,
        qualification_manifest_sha256=file_hash(args.qualification_root/"run-manifest.json"),
        reference_manifest_sha256=file_hash(args.reference_root/"run-manifest.json"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    write_json(args.output / "run-manifest.json", manifest)
    fields = ("filtering_mean", "filtering_covariance", "post_reset_mean", "post_reset_covariance")
    nominal = tf.constant([[.3, .3], [.5, .5], [.7, .7], [.8, .2]], tf.float32)
    stress = tf.constant([[.1008, .1008], [.1008, .8992], [.8992, .1008], [.8992, .8992]], tf.float32)
    reference = make_sv_reference(batch_size=4)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([20], tf.float64)],
                 jit_compile=True, autograph=False)
    def exact(positions, observations):
        result = reference.physical_value_score(positions, observations)
        return result['log_likelihood'] - 2 * tf.math.log(tf.constant(.8, tf.float64)), result['score']

    data_paths, curves, datasets = {}, {}, {}
    started = time.monotonic()
    cpu_started = time.process_time()

    def resource_check():
        allocator = tf.config.experimental.get_memory_info("GPU:0")
        if (time.monotonic() - started > args.wall_budget or time.process_time() - cpu_started > args.cpu_budget
                or allocator["peak"] > 4 * 1024**3):
            raise RuntimeError("SV control calibration resource limit exceeded")

    for partition in ("calibration", "validation"):
        path = args.reference_root / partition / "data.json"
        data = json.loads(path.read_text())
        if (data.get("model_id"), data.get("horizon"), data.get("attempt"), data.get("partition")) != ("SV20", 20, 1, partition):
            raise ValueError("wrong SV20 control partition")
        if file_hash(path) != reference_manifest["partitions"][partition]["data_sha256"]:
            raise ValueError("reference data drift")
        datasets[partition] = data
        data_paths[partition] = path
        root = args.output / partition
        root.mkdir()
        outputs, diagnostics = [], []
        for index, multiplier in enumerate(MULTIPLIERS):
            fixture = make_sv_fixture_target(data, particle_count=1024, controls=protection_controls(multiplier))
            evaluate = fixture.compile(4, coordinates="physical", filter_moments=True)
            begin = time.monotonic()
            nominal_output = evaluate(nominal)
            stress_output = evaluate(stress)
            # Materialization synchronizes compiled execution before recording cost.
            stress_output[0].numpy()
            elapsed = time.monotonic() - begin
            arms = {}
            for name, positions, (value, force, status) in (
                ("nominal", nominal, nominal_output), ("stress", stress, stress_output)):
                valid = status["value_valid"] & status["force_valid"]
                traces = {key: status[key] for key in fields}
                finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(force), axis=1)
                for tensor in traces.values():
                    finite &= tf.reduce_all(tf.math.is_finite(tensor), axis=list(range(1,tensor.shape.rank)))
                # Broad fail-closed bound; never clip or change the accepted computation.
                bounded = tf.reduce_all(tf.abs(force) < 1e6, axis=1)
                for key in fields:
                    bounded &= tf.reduce_all(tf.abs(status[key]) < 1e6, axis=list(range(1,status[key].shape.rank)))
                healthy = valid & finite & bounded
                flagged = ~valid & ((status["value_failure_code"] != 0) | (status["force_failure_code"] != 0))
                reference_value, reference_force = exact(tf.cast(positions, tf.float64),
                                                         tf.cast(tf.constant(data["raw_observations"], tf.float32), tf.float64))
                arms[name] = dict(value=value.numpy().tolist(), force=force.numpy().tolist(),
                    status={k:v.numpy().tolist() for k,v in status.items()},
                    healthy_rows=healthy.numpy().tolist(),
                    all_healthy=bool(tf.reduce_all(healthy)),
                    all_failures_flagged=bool(tf.reduce_all(healthy | flagged)),
                    reference_value=reference_value.numpy().tolist(), reference_force=reference_force.numpy().tolist())
            arm = dict(multiplier=multiplier, controls=protection_controls(multiplier),
                       fixture=dict(fixture.manifest), elapsed_seconds=elapsed, situations=arms)
            write_json(root / f"arm-{index}.json", finite_json(arm))
            resource_check()
            outputs.append((nominal_output, stress_output))
            diagnostics.append(arm)
            print(partition, multiplier, {k:v["all_healthy"] for k,v in arms.items()}, elapsed, flush=True)
        pairs = []
        for index in range(3):
            maximum = 0.
            component_differences = {}
            for name, left, right in zip(("nominal", "stress"), outputs[index], outputs[index+1]):
                a_value, a_force, a_status = left
                b_value, b_force, b_status = right
                components = {"value": (a_value, b_value), "force": (a_force, b_force)}
                components.update({k:(a_status[k], b_status[k]) for k in fields})
                diffs = {key:float(tf.reduce_max(tf.abs(a-b)/(1+tf.abs(b)))) for key,(a,b) in components.items()}
                component_differences[name] = diffs
                if name == "nominal":
                    maximum = max(diffs.values())
            healthy_nominal = all(diagnostics[i]["situations"]["nominal"]["all_healthy"] for i in (index,index+1))
            safe_stress = all(diagnostics[i]["situations"]["stress"]["all_healthy"] or
                             diagnostics[i]["situations"]["stress"]["all_failures_flagged"] for i in (index,index+1))
            no_new_failure = no_new_failures(
                diagnostics[index]["situations"]["stress"]["healthy_rows"],
                diagnostics[index+1]["situations"]["stress"]["healthy_rows"])
            pair = dict(stronger_multiplier=MULTIPLIERS[index], weaker_multiplier=MULTIPLIERS[index+1],
                        healthy_nominal=healthy_nominal, stress_safe=safe_stress,
                        no_new_stress_failure=no_new_failure, normalized_maximum_change=maximum,
                        tolerance=1e-3, component_changes=component_differences,
                        non_harm_passed=healthy_nominal and safe_stress and no_new_failure and maximum <= 1e-3)
            pairs.append(finite_json(pair))
        curves[partition] = root / "curve.json"
        write_json(curves[partition], dict(scope=scope, source_hashes=hashes, data_sha256=file_hash(path),
            pairs=pairs, arms=[dict(path=str(root/f"arm-{i}.json"), sha256=file_hash(root/f"arm-{i}.json")) for i in range(4)]))
    try:
        selection = issue_selection(output=args.output / "selection.json", scope=scope, evidence_paths=curves,
                                    partition_data_paths=data_paths, source_hashes=hashes)
        verdict = "SELECTED_FOR_PROPOSAL_EXPERIMENT"
    except ValueError as exc:
        if str(exc) != "NO_STABLE_CONTROL_PAIR":
            raise
        selection = None
        verdict = "NO_STABLE_CONTROL_PAIR"
    forecast = (measure_cost(datasets["calibration"], selection["controls"], nominal, args.output, resource_check)
                if selection else {"status": "NOT_MEASURED_NO_SELECTED_CONTROLS"})
    manifest.update(wall_seconds=time.monotonic()-started,
        seeds={p: data["data_seed"] for p, data in datasets.items()},
        cpu_worker_seconds=time.process_time()-cpu_started,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        partition_data={p: dict(path=str(path.resolve()), sha256=file_hash(path)) for p,path in data_paths.items()})
    for name, digest in hashes.items():
        if file_hash(Path(name)) != digest:
            raise ValueError("executed source drift: " + name)
    write_json(args.output / "run-manifest.json", manifest)
    write_json(args.output / "report.json", dict(verdict=verdict, selected=selection, forecast=forecast,
        memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        scope=scope, source_hashes=hashes, elapsed_seconds=time.monotonic()-started,
        cpu_worker_seconds=time.process_time()-cpu_started,
        allocator=tf.config.experimental.get_memory_info("GPU:0"),
        plan=manifest["plan"], posterior_comparisons="NOT_RUN", heuristic_dominance="NOT_RUN",
        inference_attempts_consumed=0, scoped_control_attempts_consumed=1))
    print(verdict, forecast["status"], flush=True)
    return 0 if selection else 2


def measure_cost(dataset, controls, physical, output, resource_check):
    """Time a complete cheapest trajectory; never issue inference authority."""
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_sv_consumer import make_sv_consumer
    from bayesfilter.inference.neural_force_hmc import NeuralForceHMCConfig, neural_force_hmc_transition
    from bayesfilter.inference.surrogate_hmc_program import write_json
    consumer = make_sv_consumer(dataset, controls=controls, batch_size=4, particle_count=1024)
    theta = tf.cast(physical, tf.float64)
    position = tf.sqrt(tf.constant(2., tf.float64)) * tf.math.erfinv(2 * (theta - .1) / .8 - 1)
    potential = consumer.endpoint.function(position).potential
    config = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=1,
                                 inverse_mass_diagonal=(1., 1.), dtype="float64")

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                  tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def transition(q, u, seed):
        return neural_force_hmc_transition(q, u, consumer.force, consumer.endpoint, config, seed)

    records = []
    for i in range(5):
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.monotonic()
        step = transition(position, potential, tf.constant([20260920, 301+i], tf.int32))
        step.position.numpy()
        elapsed = time.monotonic() - start
        after = resource.getrusage(resource.RUSAGE_SELF)
        cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        trace = tf.nest.map_structure(lambda x: x.numpy().tolist(), step.trace._asdict())
        records.append(dict(compilation_included=i == 0, seed=[20260920, 301+i],
                            wall_seconds=elapsed, cpu_worker_seconds=cpu, trace=trace))
        write_json(output / "cost-transitions.json", finite_json(records))
        if (not bool(tf.reduce_all(tf.math.is_finite(step.position)))
                or not bool(tf.reduce_all(tf.math.is_finite(step.trace.delta_h)))
                or bool(tf.reduce_any(step.trace.endpoint_status["numerical_failure"]))
                or bool(tf.reduce_any(step.trace.force_fallback))
                or bool(tf.reduce_any(step.trace.divergence))):
            raise ValueError("invalid calibration transition; cost forecast unavailable")
        resource_check()
        position, potential = step.position, step.potential
    forecast = cost_forecast(records[1:])
    write_json(output / "cost-forecast.json", forecast)
    return forecast


# Shared mechanical budget screen has no model likelihood or consumer dependency.
from scripts.calibrate_surrogate_hmc_linear_controls import cost_forecast


if __name__ == "__main__":
    raise SystemExit(main())
