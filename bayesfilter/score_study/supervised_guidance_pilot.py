"""Diagnostic GPU/XLA supervised guide selection, without admission claims."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


def serializable(value):
    if tf.is_tensor(value):
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=1800)
    parser.add_argument("--seed-offset", type=int, default=0)
    arguments = parser.parse_args()
    started = time.monotonic()
    arguments.output.mkdir(parents=True, exist_ok=False)
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
    from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.score_study.sir_backward_guidance_diagnostic_tf import SIRGuideFitter
    from bayesfilter.score_study.supervised_guidance_tf import (
        make_sir_supervision_programs, moment_features, supervised_squared_loss,
        validate_seed_partitions,
    )
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("GPU required")
    dtype = tf.float64
    horizon, particles, components = arguments.horizon, 64, 32
    if select_transport_chunk_size(particles) != particles:
        raise ValueError("Dense shared executor requires the policy one-block scope")
    partitions = dict(calibration=[261019501, 261019502],
                      validation=[261019503, 261019504],
                      test=[261019505, 261019506, 261019507, 261019508])
    partitions = {name:[seed + arguments.seed_offset for seed in seeds]
                  for name,seeds in partitions.items()}
    validate_seed_partitions(partitions)
    particle_seeds = [261019801 + arguments.seed_offset, 261019802 + arguments.seed_offset]
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(dtype)
    design = tf.random.stateless_normal([particles, 18], [261019901, 0], dtype=dtype)
    controls = dict(flow_substeps=24, reset_policy="contract_e",
        reset_epsilon=1., reset_sinkhorn_steps=5, reset_balance_steps=32,
        reset_ridge=1e-5, reset_denominator_regularization=1e-7,
        correction_steps=1, correction_strength=.2, correction_lm_damping=.01,
        correction_lm_scale_floor=.0001, correction_trust_radius=.5, pairwise_steps=1,
        pairwise_strength=.02, pairwise_rms_cap=2., coordinate_cap=.94,
        coordinate_cap_power=8, moment_safety=False)
    manifest = dict(command=sys.argv, commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        plan="docs/plans/ledh-supervised-guidance-20261019.md",
        result=str(arguments.output / "summary.json"),
        environment=sys.executable, tensorflow=tf.__version__, memory_policy=memory,
        gpu_devices=[device.name for device in tf.config.list_logical_devices("GPU")],
        execution="GPU/XLA FP64 diagnostic exception; TF32 off",
        jit_compile=True, dtype="float64", tf32=False, horizon=horizon, particles=particles,
        chunk=particles, controls=controls, partitions=partitions, particle_seeds=particle_seeds,
        model="canonical_sir_d18_x0_then_transition_observe_v1; no clipping",
        guide_fit=dict(particles=512, validation_count=256, components=components),
        budget_seconds=arguments.budget_seconds, scientific_status="pilot_descriptive_only",
        data_version="fresh exact-consumer simulation; latent and particle seeds disjoint")
    sources = [Path(inspect.getsourcefile(item)) for item in
        (main, make_sir_supervision_programs, build_backward_guided_ledh_kernel, SIRGuideFitter, NonlinearSQMCSpec)]
    sources.extend(Path("bayesfilter/highdim") / filename for filename in
        ("ledh_canonical_score_tf.py", "ledh_canonical_models_tf.py",
         "ledh_canonical_reset_score_tf.py", "ledh_unified_correction_tf.py",
         "gaussian_mixture_conditioning_tf.py"))
    manifest["source_hashes"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    write_json(arguments.output / "manifest.json", manifest)
    def checkpoint(stage, **fields):
        write_json(arguments.output / "progress.json",
                   dict(stage=stage, wall_seconds=time.monotonic()-started, **fields))
        print(stage, fields, flush=True)
        if time.monotonic() - started > arguments.budget_seconds:
            raise TimeoutError("Pilot process budget exhausted")

    simulate, label = make_sir_supervision_programs(horizon, 1)
    datasets = {}
    for partition, seeds in partitions.items():
        for seed in seeds:
            paths, observations = simulate(theta, tf.constant(seed))
            labels = label(theta, paths, observations)
            if not bool(labels["valid"]):
                raise ValueError("Nonfinite simulation or label")
            datasets[seed] = (observations[0], labels["score"][0], paths[0, -1])
    calibration_labels = tf.stack([datasets[seed][1] for seed in partitions["calibration"]])
    scales = tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(calibration_labels), axis=0)), 1.)
    write_json(arguments.output / "labels.json", dict(scales=scales,
        datasets={str(seed):dict(observations=data[0], score=data[1], terminal_state=data[2])
                  for seed, data in datasets.items()}))
    checkpoint("labels_complete")
    common = dict(horizon=horizon, particle_count=particles, state_dimension=18,
        observation_dimension=9, parameter_dimension=3, dtype=dtype, jit_compile=True,
        return_trace=True, reset_design=design, **controls)
    factory = lambda parameter, direction: spec.model(parameter, direction)[0]
    kernels = dict(quadratic=build_backward_guided_ledh_kernel(factory, None,
        dynamic_quadratic=True, **common))
    kernels["raw_quadratic"] = build_backward_guided_ledh_kernel(factory, None,
        dynamic_quadratic=True, allow_indefinite_information=True, **common)
    for family, conditioning in (("kernel_information", "positive_information"),
                                 ("kernel_product", "mixture_product_moments")):
        kernels[family] = build_backward_guided_ledh_kernel(factory, None,
            mixture_components=components, mixture_conditioning=conditioning, **common)
    fitter = SIRGuideFitter(horizon, components=components)
    rows, selected = [], {}
    candidate_names = ("unguided", "csmc_half", "csmc_original", "csmc_raw",
                       "kernel_information", "kernel_product")
    for partition, seeds in partitions.items():
        for seed in seeds:
            observations, target_score, terminal = datasets[seed]
            checkpoint("fit_start", partition=partition, seed=seed)
            quadratic, mixture, fit = fitter.fit(observations, seed + 1000)
            write_json(arguments.output / f"fit-{seed}.json", fit)
            candidate_arguments = dict(
                unguided=("quadratic", (tf.zeros_like(quadratic.precision), tf.zeros_like(quadratic.linear))),
                csmc_half=("quadratic", (.5 * quadratic.precision, .5 * quadratic.linear)),
                csmc_original=("quadratic", (quadratic.precision, quadratic.linear)),
                csmc_raw=("raw_quadratic", (fit["raw_precision"], fit["raw_linear"])),
                kernel_information=("kernel_information", (mixture.logits, mixture.centers, mixture.covariances, mixture.active)),
                kernel_product=("kernel_product", (mixture.logits, mixture.centers, mixture.covariances, mixture.active)))
            for particle_seed in particle_seeds:
                states, covariances, d_states, d_covariances = spec.initial_cloud(theta,
                    tf.random.stateless_normal([particles, 18], [particle_seed, 0], dtype=dtype), tf.zeros([3], dtype))
                noises = tf.random.stateless_normal([horizon, particles, 18], [particle_seed, 1], dtype=dtype)
                for name in candidate_names:
                    family, coefficients = candidate_arguments[name]
                    scores, checks, values = [], [], []
                    for coordinate in range(3):
                        value, derivative, trace = kernels[family](theta, tf.one_hot(coordinate, 3, dtype=dtype),
                            states, covariances, noises, observations, d_states, d_covariances, *coefficients)
                        values.append(float(value))
                        scores.append(derivative[0])
                        checks.append({key:bool(tf.reduce_all(tf.stack([step[key] for step in trace])))
                            for key in ("program_valid", "ukf_predict_valid", "ukf_update_valid",
                                "reset_numerical_valid", "reset_transport_valid", "guided_proposal_valid")})
                    score = tf.stack(scores)
                    loss, finite = supervised_squared_loss(score, target_score, scales)
                    valid = bool(finite) and all(all(check.values()) for check in checks)
                    valid = valid and all(bool(tf.math.is_finite(value)) for value in values)
                    row = dict(partition=partition, observation_seed=seed, particle_seed=particle_seed,
                        method=name, valid=valid, checks=checks)
                    if valid and seed == partitions["calibration"][0] and particle_seed == particle_seeds[0]:
                        direction = tf.constant([1., .25, -.5], dtype)
                        increment = tf.constant(1e-5, dtype)
                        numerical_values = [kernels[family](theta + sign * increment * direction, direction,
                            states, covariances, noises, observations, d_states, d_covariances, *coefficients)[0]
                            for sign in (-1., 1.)]
                        numerical = (numerical_values[1] - numerical_values[0]) / (2 * increment)
                        analytical = tf.reduce_sum(score * direction)
                        derivative_valid = bool(tf.abs(numerical - analytical) <= 1e-4 * (1 + tf.abs(analytical)))
                        write_json(arguments.output / f"derivative-{name}.json",
                            dict(analytical=analytical, numerical=numerical, valid=derivative_valid,
                                 guide_frozen=True, finite_program_only=True))
                        if not derivative_valid:
                            raise ValueError(f"Analytical derivative mismatch: {name}")
                    if valid:
                        terminal_moments = tf.reduce_mean(moment_features(trace[-1]["states_after_reset"]), axis=0)
                        row.update(score=score, loss=loss, value=values[0],
                            terminal_moment_squared_error=tf.square(terminal_moments-moment_features(terminal)))
                    rows.append(serializable(row))
                    with (arguments.output / "rows.jsonl").open("a") as stream:
                        stream.write(json.dumps(serializable(row), allow_nan=False) + "\n")
                    checkpoint("evaluation", partition=partition, seed=seed, method=name, valid=valid, rows=len(rows))
        if partition == "validation":
            candidates = {}
            for name in candidate_names:
                observed = [row for row in rows if row["partition"] == "validation" and row["method"] == name]
                if all(row["valid"] for row in observed):
                    candidates[name] = sum(sum(row["loss"]) for row in observed) / len(observed)
            for group, names in dict(csmc=["csmc_half", "csmc_original", "csmc_raw"],
                                     kernel=["kernel_information", "kernel_product"]).items():
                eligible = [name for name in names if name in candidates]
                selected[group] = min(eligible, key=candidates.get) if eligible else None
            write_json(arguments.output / "selection.json", dict(selected=selected, validation_loss=candidates,
                frozen_before_test=True, scales=scales))
    test_summary = {}
    for name in candidate_names:
        observed = [row for row in rows if row["partition"] == "test" and row["method"] == name]
        test_summary[name] = dict(valid=sum(row["valid"] for row in observed), total=len(observed))
        if all(row["valid"] for row in observed):
            test_summary[name]["mean_component_loss"] = [sum(row["loss"][coordinate] for row in observed)/len(observed) for coordinate in range(3)]
    summary = dict(selected=selected, test=test_summary, rows=len(rows), wall_seconds=time.monotonic()-started,
        ranking="not statistically supported: four independent test datasets",
        promotion=False, next_action="Review validity and paired per-dataset differences before allocating a larger rung")
    write_json(arguments.output / "summary.json", summary)
    manifest.update(wall_seconds=summary["wall_seconds"], output_artifacts=[str(path) for path in arguments.output.iterdir()],
                    allocator=tf.config.experimental.get_memory_info("GPU:0"))
    write_json(arguments.output / "manifest.json", manifest)
    checkpoint("complete", rows=len(rows))


if __name__ == "__main__":
    main()
