"""Frozen GPU replay for merge diagnostics; no tuning or promotion authority."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


PARTICLE_ARMS = ("trained", "untrained", "plain", "conditional", "physical",
                 "csmc_unadapted", "csmc_adapted", "rbf_unadapted", "rbf_adapted")
HEURISTIC_ARMS = ("prediction", "observation_only", "recursive_gaussian")


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaves(value):
    if isinstance(value, list):
        return [number for item in value for number in leaves(item)]
    return [value]


def compare_numbers(reference, candidate, *, absolute=1e-8, relative=1e-8):
    left, right = leaves(reference), leaves(candidate)
    if len(left) != len(right):
        return dict(passed=False, reason="shape", reference_count=len(left),
                    candidate_count=len(right))
    finite = all(type(value) in (int, float) and math.isfinite(value)
                 for value in left + right)
    if not finite:
        return dict(passed=False, reason="nonfinite")
    differences = [abs(after - before) for before, after in zip(left, right)]
    ratios = [difference / (absolute + relative * abs(before))
              for before, difference in zip(left, differences)]
    return dict(passed=max(ratios, default=0.) <= 1.,
                max_absolute=max(differences, default=0.),
                max_tolerance_ratio=max(ratios, default=0.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("before", "after"), required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--plan", default="docs/plans/ledh-sir-postmerge-replay-20261021.md")
    parser.add_argument("--resume-from", type=Path)
    arguments = parser.parse_args()
    started = time.monotonic()
    arguments.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(arguments.source.resolve()))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    import tensorflow as tf
    import bayesfilter
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    if Path(bayesfilter.__file__).resolve().parent != arguments.source.resolve() / "bayesfilter":
        raise ValueError("Source import isolation failed")
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
    from bayesfilter.highdim.ledh_neural_kdm_tf import ObservationKDMNetwork
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
    from bayesfilter.score_study.moment_training_campaign import append, save, serializable
    from bayesfilter.score_study.moment_training_tf import (
        make_guide_adapters, make_moment_endpoint, make_moment_objective,
    )
    from bayesfilter.score_study.physical_proposal_diagnostic_campaign import PhysicalCampaign
    from bayesfilter.score_study.run_source_snapshot import verify_source_snapshot

    study = arguments.repository / "docs/plans/artifacts/ledh-sir-neural-kdm-20261020"
    holdout = study / "attempt-16-finalized/final-holdout"
    restricted = study / "attempt-19-restricted/study"
    original_manifest = read_json(holdout / "manifest.json")
    frozen = read_json(holdout / "frozen-selection.json")
    checkpoint_path = Path(frozen["checkpoint"])
    checkpoint = read_json(checkpoint_path)
    calibration_path = holdout / "calibration.json"
    calibration = read_json(calibration_path)
    selected = read_json(restricted / "frozen-selection.json")["selected"]
    source = read_json(arguments.source / "index.json")
    auxiliary_source = read_json(arguments.source / "auxiliary-index.json")
    source_check = verify_source_snapshot(arguments.source, source)
    if not source_check["unchanged"]:
        raise ValueError("Input source snapshot differs from its index")
    if not all(file_hash(arguments.source / name) == digest
               for name, digest in auxiliary_source.items()):
        raise ValueError("Auxiliary source snapshot differs from its index")
    if arguments.lane == "before" and source["source_hashes"] != checkpoint["source_hashes"]:
        raise ValueError("Before source differs from the frozen checkpoint")
    archived = {}
    for path in (holdout / "holdout.jsonl", restricted / "holdout.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["dataset"], row["arm"], row["replicate"])
            if key in archived:
                raise ValueError("Duplicate archived result")
            archived[key] = row
    datasets = list(range(263043001, 263043065))
    data = {dataset: read_json(holdout / f"dataset-{dataset}.json") for dataset in datasets}
    regime_first = {}
    for dataset, record in data.items():
        key = (record["observation_regime"], record["compartment_regime"])
        regime_first.setdefault(key, dataset)
    if len(regime_first) != 4:
        raise ValueError("Expected four joint regimes for analytical parity")
    score_datasets = sorted(regime_first.values())
    input_paths = [checkpoint_path, calibration_path, holdout / "frozen-selection.json",
                   restricted / "frozen-selection.json", holdout / "holdout.jsonl",
                   restricted / "holdout.jsonl"]
    input_paths += [holdout / f"dataset-{dataset}.json" for dataset in datasets]
    input_paths += [restricted / f"message-{284043000 + index}.json" for index in range(64)]
    input_hashes = {str(path): file_hash(path) for path in input_paths}
    resume_manifest = None
    if arguments.resume_from is not None:
        resume_manifest = read_json(arguments.resume_from / "manifest.json")
        for name, expected in (
            ("lane", arguments.lane), ("input_hashes", input_hashes),
            ("source_index_sha256", file_hash(arguments.source / "index.json")),
            ("auxiliary_source_hashes", auxiliary_source),
            ("custom_op_sha256", file_hash(arguments.repository / "bayesfilter/ops/_symmetric_sylvester_ops.so")),
        ):
            if resume_manifest[name] != expected:
                raise ValueError(f"Resume provenance mismatch: {name}")
    manifest = dict(lane=arguments.lane, command=sys.argv, environment=sys.executable,
        git_commit=source["commit"], source=str(arguments.source),
        source_index_sha256=file_hash(arguments.source / "index.json"),
        auxiliary_source_hashes=auxiliary_source,
        script_sha256=file_hash(Path(__file__)), input_hashes=input_hashes,
        tensorflow=tf.__version__, memory_policy=memory,
        gpu_devices=[device.name for device in tf.config.list_logical_devices("GPU")],
        gpu_details=[tf.config.experimental.get_device_details(device)
                     for device in tf.config.list_physical_devices("GPU")],
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cpu_only=False,
        dtype="float64", tf32=False, jit_compile=True,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        custom_op_sha256=file_hash(arguments.repository / "bayesfilter/ops/_symmetric_sylvester_ops.so"),
        budget_seconds=arguments.budget_seconds, horizon=5, particles=64,
        resume_from=None if arguments.resume_from is None else str(arguments.resume_from),
        resume_file_hashes={} if arguments.resume_from is None else {
            name: file_hash(arguments.resume_from / name)
            for name in ("manifest.json", "holdout.jsonl", "scores.jsonl")
            if (arguments.resume_from / name).exists()},
        chunk=select_transport_chunk_size(64), controls=original_manifest["controls"],
        configuration=frozen["configuration"], neural_updates=checkpoint["updates"],
        data_version=original_manifest["data_version"], dataset_seeds=datasets,
        particle_seed_base=283030000, replicates=4, reset_seed=263040002,
        score_datasets=score_datasets, score_directions="three coordinate unit vectors",
        plan=arguments.plan,
        result=str(arguments.output / "result.json"),
        exception="frozen FP64 regression on exposed data, no training or score-accuracy claim")
    save(arguments.output / "manifest.json", manifest)
    print(json.dumps(dict(stage="gpu_ready", lane=arguments.lane,
                          tensorflow=tf.__version__, score_datasets=score_datasets)), flush=True)

    def check_budget():
        if time.monotonic() - started > arguments.budget_seconds:
            raise TimeoutError("GPU process budget exhausted")

    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(tf.float64)
    moments = [tf.constant(value, tf.float64) for value in calibration["moment"]]
    objective = make_moment_objective(64, 18, *moments)
    configuration = frozen["configuration"]
    network = ObservationKDMNetwork(5, 18, configuration["components"], configuration["width"],
        calibration["observation_center"], calibration["observation_scale"], seed=configuration["seed"])
    initial_parameters = [tf.identity(parameter) for parameter in network.parameters]
    trained_parameters = [tf.constant(value, variable.dtype)
                          for value, variable in zip(checkpoint["parameters"], network.parameters)]
    if len(trained_parameters) != len(network.parameters) or any(
            value.shape != variable.shape for value, variable in zip(trained_parameters, network.parameters)):
        raise ValueError("Invalid frozen network tensor shapes")
    reset_design = tf.random.stateless_normal([64, 18], [263040002, 0], dtype=tf.float64)
    common = dict(horizon=5, particle_count=64, state_dimension=18, observation_dimension=9,
        parameter_dimension=3, dtype=tf.float64, jit_compile=True, return_trace=True,
        reset_design=reset_design, **original_manifest["controls"])
    factory = lambda parameter, direction: spec.model(parameter, direction)[0]
    upper_indices = [(row, column) for row in range(18) for column in range(row, 18)]

    def build_endpoint(**options):
        kernel = build_backward_guided_ledh_kernel(factory, None, **common, **options)
        endpoint = make_moment_endpoint(kernel, objective, 18)

        @tf.function(input_signature=list(endpoint.input_signature), jit_compile=True, autograph=False)
        def with_raw_moments(*inputs):
            result = endpoint.python_function(*inputs)
            center, scale, _ = moments
            normalized_mean = result["estimate"][:18]
            raw_mean = center + scale * normalized_mean
            row_indices = [row for row, _ in upper_indices]
            column_indices = [column for _, column in upper_indices]
            center_row, center_column = tf.gather(center, row_indices), tf.gather(center, column_indices)
            scale_row, scale_column = tf.gather(scale, row_indices), tf.gather(scale, column_indices)
            raw_second = (result["estimate"][18:] * scale_row * scale_column
                + center_row * scale_column * tf.gather(normalized_mean, column_indices)
                + center_column * scale_row * tf.gather(normalized_mean, row_indices)
                + center_row * center_column)
            return dict(result, raw_estimate=tf.concat([raw_mean, raw_second], axis=0))

        return with_raw_moments

    neural_endpoint = build_endpoint(neural_mixture_components=2, physical_mixture_weight=.9)
    row_signature = list(neural_endpoint.input_signature[:9]) + [tf.TensorSpec([18], tf.float64)]

    @tf.function(input_signature=row_signature, jit_compile=True, autograph=False)
    def neural_evaluate(*inputs):
        coefficients = network.predict.python_function(inputs[5])
        return neural_endpoint.python_function(*inputs[:-1], *coefficients, inputs[-1])

    endpoints = {name: build_endpoint(physical_mixture_weight=beta)
                 for name, beta in (("plain", 0.), ("conditional", 1.), ("physical", .9))}
    endpoints["neural"] = neural_evaluate
    endpoints["csmc"] = build_endpoint(dynamic_quadratic=True, physical_mixture_weight=.5)
    endpoints["rbf"] = build_endpoint(mixture_components=32,
        mixture_conditioning="mixture_product_moments", physical_mixture_weight=.5)
    adapters = {family: make_guide_adapters(5, 18, 32, candidate["capacity"] == "time_specific")[
        0 if family == "csmc" else 1] for family, candidate in selected.items()}
    heuristic = PhysicalCampaign.make_heuristics(
        SimpleNamespace(spec=spec, theta=theta, calibration=moments), return_blocks=True)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([5, 9], tf.float64),
                                  tf.TensorSpec([18], tf.float64)], jit_compile=True, autograph=False)
    def row_inputs(particle_seed, observations, label):
        normals = tf.random.stateless_normal([64, 18], tf.stack([particle_seed, 0]), dtype=tf.float64)
        states, covariances, d_states, d_covariances = spec.initial_cloud(theta, normals)
        noises = tf.random.stateless_normal([5, 64, 18], tf.stack([particle_seed, 1]), dtype=tf.float64)
        uniforms = tf.random.stateless_uniform([5, 64], tf.stack([particle_seed, 2]), dtype=tf.float64)
        return (theta, tf.zeros([3], tf.float64), states, covariances, noises,
                observations, d_states, d_covariances, uniforms, label)

    seen_routes = set()
    row_count, score_count = 0, 0
    tieout_maxima = {}
    source_valid = False

    def record_tieout(row):
        if arguments.lane != "before":
            return
        original = archived[(row["dataset"], row["arm"], row["replicate"])]
        fields = ("loss", "blocks", "estimate", "raw_blocks", "value", "score")
        differences = {field: compare_numbers(original[field], row[field])
                       for field in fields if field in original}
        for field, comparison in differences.items():
            tieout_maxima[field] = max(tieout_maxima.get(field, 0.), comparison.get("max_absolute", 0.))
        if original["valid"] != row["valid"] or not all(value["passed"] for value in differences.values()):
            save(arguments.output / "tieout-failure.json", dict(dataset=row["dataset"],
                 arm=row["arm"], replicate=row["replicate"], comparisons=differences,
                 old_valid=original["valid"], replay_valid=row["valid"]))
            raise ValueError("Pre-merge replay does not reproduce the archived result")

    cached_rows, cached_scores = {}, {}
    if arguments.resume_from is not None:
        for name, cache in (("holdout.jsonl", cached_rows), ("scores.jsonl", cached_scores)):
            path = arguments.resume_from / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = (row["dataset"], row["arm"], row["replicate"])
                if name == "scores.jsonl":
                    key += (row["direction"],)
                if key in cache:
                    raise ValueError("Duplicate resume row")
                if name == "holdout.jsonl":
                    record_tieout(row)
                elif not all(value["passed"] for value in row["direction_invariance"].values()):
                    raise ValueError("Invalid resumed directional check")
                row.setdefault("timing_origin", arguments.resume_from.name)
                cache[key] = row
                append(arguments.output / name, row)
        row_count, score_count = len(cached_rows), len(cached_scores)
    heuristic_seen = False

    try:
        for dataset_index, dataset in enumerate(datasets):
            check_budget()
            record = data[dataset]
            observations = tf.constant(record["observations"], tf.float64)
            label = tf.constant(record["label"], tf.float64)
            regimes = {name: record[name] for name in ("observation_regime", "compartment_regime")}
            need_heuristic = any((dataset, arm, None) not in cached_rows for arm in HEURISTIC_ARMS)
            if need_heuristic:
                begin = time.perf_counter()
                losses, validity, blocks = heuristic(observations, label)
                losses, validity, blocks = serializable((losses, validity, blocks))
                heuristic_seconds = time.perf_counter() - begin
            for arm_index, arm in enumerate(HEURISTIC_ARMS):
                if (dataset, arm, None) in cached_rows:
                    continue
                row = dict(dataset=dataset, arm=arm, replicate=None, particle_seed=None,
                    loss=losses[arm_index], valid=validity[arm_index], blocks=blocks[arm_index],
                    shared_evaluation_seconds=heuristic_seconds, cold=not heuristic_seen,
                    timing_origin=arguments.output.name, **regimes)
                append(arguments.output / "holdout.jsonl", row)
                record_tieout(row)
                row_count += 1
            heuristic_seen |= need_heuristic
            messages = read_json(restricted / f"message-{284043000 + dataset_index}.json")
            if not messages["valid"]:
                raise ValueError("Invalid frozen message")
            raw_coefficients = {
                "csmc": tuple(tf.constant(messages[name], tf.float64) for name in ("precision", "linear")),
                "rbf": tuple(tf.constant(messages[name], tf.bool if name == "active" else tf.float64)
                             for name in ("logits", "centers", "covariances", "active")),
            }
            coefficients = {}
            for family in ("csmc", "rbf"):
                eta = tf.constant(selected[family]["eta"], tf.float64)
                for variant in ("unadapted", "adapted"):
                    active_eta = tf.zeros_like(eta) if variant == "unadapted" else eta
                    coefficients[f"{family}_{variant}"] = adapters[family](active_eta, *raw_coefficients[family])
            for replicate in range(4):
                particle_seed = 283030000 + dataset_index * 4 + replicate
                inputs = row_inputs(tf.constant(particle_seed), observations, label)
                digest = hashlib.sha256()
                for tensor in (*inputs, reset_design):
                    digest.update(tf.io.serialize_tensor(tensor).numpy())
                input_hash = digest.hexdigest()
                for arm in PARTICLE_ARMS:
                    check_budget()
                    route = "neural" if arm in ("trained", "untrained") else arm.split("_")[0]
                    if route == "neural":
                        parameters = trained_parameters if arm == "trained" else initial_parameters
                        for variable, value in zip(network.parameters, parameters):
                            variable.assign(value)
                    extras = coefficients.get(arm, ())
                    active_inputs = (*inputs[:-1], *extras, inputs[-1])
                    key = (dataset, arm, replicate)
                    if key in cached_rows:
                        output = cached_rows[key]
                        if output["input_sha256"] != input_hash:
                            raise ValueError("Resumed particle inputs changed")
                    else:
                        cold = route not in seen_routes
                        begin = time.perf_counter()
                        output = serializable(endpoints[route](*active_inputs))
                        elapsed = time.perf_counter() - begin
                        seen_routes.add(route)
                        row = dict(dataset=dataset, arm=arm, replicate=replicate, particle_seed=particle_seed,
                            input_sha256=input_hash, evaluation_seconds=elapsed, cold=cold,
                            timing_origin=arguments.output.name, **regimes, **output)
                        append(arguments.output / "holdout.jsonl", row)
                        record_tieout(row)
                        row_count += 1
                    if dataset in score_datasets:
                        for direction_index in range(3):
                            if key + (direction_index,) in cached_scores:
                                if cached_scores[key + (direction_index,)]["input_sha256"] != input_hash:
                                    raise ValueError("Resumed directional inputs changed")
                                continue
                            check_budget()
                            directional_inputs = list(active_inputs)
                            directional_inputs[1] = tf.one_hot(direction_index, 3, dtype=tf.float64)
                            begin = time.perf_counter()
                            directional = serializable(endpoints[route](*directional_inputs))
                            elapsed = time.perf_counter() - begin
                            cold = route not in seen_routes
                            seen_routes.add(route)
                            invariant = {field: compare_numbers(output[field], directional[field])
                                         for field in ("value", "loss", "estimate")}
                            score_row = dict(dataset=dataset, arm=arm, replicate=replicate,
                                particle_seed=particle_seed, input_sha256=input_hash,
                                direction=direction_index, evaluation_seconds=elapsed,
                                cold=cold, timing_origin=arguments.output.name,
                                direction_invariance=invariant, **regimes, **directional)
                            append(arguments.output / "scores.jsonl", score_row)
                            score_count += 1
                            if not all(comparison["passed"] for comparison in invariant.values()):
                                raise ValueError("Value or moments depend on tangent direction")
            print(json.dumps(dict(stage="replay", lane=arguments.lane, datasets=dataset_index + 1,
                holdout_rows=row_count, score_rows=score_count,
                wall_seconds=time.monotonic() - started)), flush=True)
        final_source = verify_source_snapshot(arguments.source, source)
        source_valid = final_source["unchanged"]
        source_valid &= all(file_hash(arguments.source / name) == digest
                            for name, digest in auxiliary_source.items())
        input_valid = all(file_hash(Path(path)) == digest for path, digest in input_hashes.items())
        imported = {name: module.__file__ for name, module in tuple(sys.modules.items())
                    if name.split(".")[0] in ("bayesfilter", "experiments")
                    and getattr(module, "__file__", None)}
        imports_valid = all(Path(path).resolve().is_relative_to(arguments.source.resolve())
                            for path in imported.values())
        save(arguments.output / "imports.json", imported)
        if not source_valid or not input_valid or not imports_valid:
            raise ValueError("Source/input/import provenance changed during replay")
        result = dict(status="complete", lane=arguments.lane, holdout_rows=row_count,
            score_rows=score_count, wall_seconds=time.monotonic() - started,
            old_archive_tieout_max_absolute=tieout_maxima, source_check=final_source,
            input_hashes_unchanged=input_valid, isolated_imports=imports_valid,
            imported_modules=len(imported), gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"),
            graph_trace_counts={name: endpoint.experimental_get_tracing_count()
                                for name, endpoint in endpoints.items()})
        save(arguments.output / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        save(arguments.output / "failure.json", dict(error=repr(error),
            wall_seconds=time.monotonic() - started, holdout_rows=row_count, score_rows=score_count))
        raise


if __name__ == "__main__":
    main()
