"""Repair non-harm and analytical parity diagnostics; never guide selection."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from bayesfilter.score_study.moment_training_campaign import Campaign, tf, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.monotonic()
    campaign = Campaign(argparse.Namespace(output=arguments.output, budget_seconds=1200., smoke=False))
    from bayesfilter.highdim.higher_moment_contract_e import affine_restore_cloud_jvp

    baseline_path = arguments.diagnosis.parent / "original-higher_moment_contract_e.py"
    specification = importlib.util.spec_from_file_location("original_correction_diagnostic", baseline_path)
    baseline = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(baseline)
    signature = [tf.TensorSpec([64, 18], tf.float64), tf.TensorSpec([64], tf.float64),
        tf.TensorSpec([64, 18, 1], tf.float64), tf.TensorSpec([64, 1], tf.float64),
        tf.TensorSpec([64, 18], tf.float64), tf.TensorSpec([64, 18, 1], tf.float64),
        tf.TensorSpec([64, 18], tf.float64), tf.TensorSpec([64, 18, 1], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True)
    def local_check(source, weights, source_dot, weights_dot, points, points_dot, offsets, offsets_dot):
        ordinary = baseline.affine_restore_cloud_jvp(source, weights, source_dot, weights_dot, points, points_dot)
        stable = affine_restore_cloud_jvp(source, weights, source_dot, weights_dot, points, points_dot,
            point_offsets=offsets, point_offsets_tangent=offsets_dot)
        return ordinary, stable

    original_trace = json.loads((arguments.diagnosis / "trace.json").read_text())
    local = []
    for index in range(4):
        inputs = [tf.constant(value, tf.float64) for value in json.loads(
            (arguments.diagnosis / f"restore-input-{index + 1}.json").read_text())]
        step = original_trace[index]
        inputs += [tf.constant(step[name], tf.float64) for name in
            ("higher_moment_stage_post_cap_offsets", "higher_moment_stage_post_cap_offsets_tangent")]
        ordinary, stable = local_check(*inputs)
        differences = {name: tf.reduce_max(tf.abs(stable[name] - ordinary[name])) /
            tf.maximum(1., tf.reduce_max(tf.abs(ordinary[name])))
            for name in ("particles", "particles_tangent")}
        local.append(dict(time=index + 1, baseline_valid=ordinary["valid"], repaired_valid=stable["valid"],
            normalized_differences=differences, diagnostics={name: value for name, value in stable.items()
                if value.shape.rank == 0}))
    save(arguments.output / "local.json", local)

    endpoints = []
    for dataset_seed, particle_seed in ((262023017, 274000034), (262023001, 274000002)):
        data = campaign.dataset(dataset_seed)
        saved = json.loads((arguments.source / f"dataset-{dataset_seed}.json").read_text())
        input_matches = {name: bool(tf.reduce_all(data[name] == tf.constant(saved[name], tf.float64)))
            for name in ("observations", "label", "score_label")}
        if not all(input_matches.values()):
            raise ValueError("Replay inputs changed")
        initial = campaign.particles_for(particle_seed)
        if bool(tf.reduce_any(initial[3] != 0)) or bool(tf.reduce_any(initial[4] != 0)):
            raise ValueError("Endpoint finite difference requires parameter-independent initial cloud")
        outputs = []
        differences = []
        for coordinate in range(3):
            direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
            output = campaign.evaluate("csmc", "pooled", tf.zeros([1, 3], tf.float64),
                dataset_seed, particle_seed, direction=direction, multiplier=0.)
            outputs.append(output)
            for step_size in (1e-4, 1e-5):
                plus = campaign.evaluate("csmc", "pooled", tf.zeros([1, 3], tf.float64),
                    dataset_seed, particle_seed, multiplier=0., theta=campaign.theta + step_size * direction)
                minus = campaign.evaluate("csmc", "pooled", tf.zeros([1, 3], tf.float64),
                    dataset_seed, particle_seed, multiplier=0., theta=campaign.theta - step_size * direction)
                finite_difference = (plus["value"] - minus["value"]) / (2 * step_size)
                differences.append(dict(coordinate=coordinate, step_size=step_size,
                    valid=plus["valid"] & minus["valid"], analytical=output["score"],
                    finite_difference=finite_difference,
                    normalized_error=tf.abs(output["score"] - finite_difference) / tf.maximum(1., tf.abs(output["score"]))))
        states, covariances, noises, d_states, d_covariances = initial
        value, score, trace = campaign.kernels["csmc"](campaign.theta, tf.one_hot(0, 3, dtype=tf.float64),
            states, covariances, noises, data["observations"], d_states, d_covariances,
            *(tf.zeros_like(coefficient) for coefficient in data["csmc"]))
        save(arguments.output / f"trace-{dataset_seed}.json", trace)
        nonfinite = [{name: int(tf.reduce_sum(tf.cast(~tf.math.is_finite(tensor), tf.int32)))
                     for name, tensor in item.items() if tensor.dtype.is_floating
                     and not bool(tf.reduce_all(tf.math.is_finite(tensor)))} for item in trace]
        endpoints.append(dict(dataset_seed=dataset_seed, particle_seed=particle_seed,
            outputs=outputs, differences=differences, trace_nonfinite=nonfinite,
            input_matches=input_matches,
            materialized_shape_available=[bool(item["higher_moment_materialized_shape_available"]) for item in trace]))
        save(arguments.output / "endpoints.json", endpoints)
    paths = (Path("bayesfilter/highdim/higher_moment_contract_e.py"),
        Path("bayesfilter/highdim/ledh_unified_correction_tf.py"), Path(__file__))
    manifest = dict(campaign.manifest, plan="docs/plans/ledh-moment-restoration-repair-20261020.md",
        result=str(arguments.output / "result.json"), role="numerical repair validation only",
        repair_source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        baseline_hash=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        calls=campaign.calls + 2, wall_seconds=time.monotonic() - started,
        allocator=tf.config.experimental.get_memory_info("GPU:0"))
    verdict = dict(local_all_finite=all(bool(row["repaired_valid"]) for row in local),
        healthy_local_nonharm=all(float(error) <= 1e-10 for row in local[:3]
            for error in row["normalized_differences"].values()),
        endpoint_all_valid=all(bool(output["valid"]) for row in endpoints for output in row["outputs"]),
        endpoint_trace_all_finite=all(not errors for row in endpoints for errors in row["trace_nonfinite"]),
        endpoint_fd_pass=all(bool(item["valid"]) and float(item["normalized_error"]) <= 1e-3
            for row in endpoints for item in row["differences"]))
    shape_residuals = {"higher_moment_skew_residual", "higher_moment_kurtosis_residual",
        "higher_moment_pairwise_co_skew_residual", "higher_moment_pairwise_co_kurtosis_residual"}
    verdict["unavailable_shape_explicit"] = all(not errors or
        (not available and set(errors).issubset(shape_residuals))
        for row in endpoints for errors, available in zip(row["trace_nonfinite"], row["materialized_shape_available"]))
    baseline_row = json.loads((arguments.source / "test.jsonl").read_text().splitlines()[0])
    if (baseline_row["dataset_seed"], baseline_row["particle_seed"], baseline_row["method"]) != (262023001, 274000002, "unguided"):
        raise ValueError("Unexpected healthy comparator")
    healthy = endpoints[1]["outputs"]
    comparison = dict(baseline_loss=baseline_row["loss"], repaired_loss=healthy[0]["loss"],
        baseline_score=baseline_row["score"], repaired_score=[item["score"] for item in healthy],
        role="same-input numerical comparison only, not efficacy")
    save(arguments.output / "result.json", dict(manifest=manifest, verdict=verdict, healthy_comparison=comparison))
    save(arguments.output / "manifest.json", manifest)
    print(json.dumps(dict(verdict=verdict, wall_seconds=manifest["wall_seconds"])))


if __name__ == "__main__":
    main()
