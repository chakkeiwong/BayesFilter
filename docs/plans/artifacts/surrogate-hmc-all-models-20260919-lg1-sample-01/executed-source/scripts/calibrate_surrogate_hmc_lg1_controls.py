#!/usr/bin/env python3
"""Full-DCTR LG1 non-harm curves on separate calibration/validation data."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path(
        "docs/plans/artifacts/surrogate-hmc-all-models-20260919-lg1-reference-master-01/P2-reference"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target, source_closure
    from bayesfilter.testing.surrogate_hmc_lg1_tf import kalman_physical
    from bayesfilter.inference.surrogate_hmc_dctr_selection import MULTIPLIERS, protection_controls, issue_selection
    from bayesfilter.inference.surrogate_hmc_program import write_json, file_hash

    scope = dict(model="LG1", horizon=10, attempt=4, state_dimension=1, parameter_dimension=1,
        particle_count=1024, chunk_extent=1024, dtype="float32", tf32=True, jit_compile=True,
        route="canonical_batch_fused_value_score", reset_contract_id="contract_e_chol_v1",
        law="stationary x0, Q1 R.5, uniform phi[-.95,.95]",
        data_regime="synthetic nominal LG1; separate calibration/validation/claim partitions",
        control_family="full_canonical_geometric_protection_curve_v1")
    hashes = source_closure()
    hashes["bayesfilter/inference/surrogate_hmc_dctr_selection.py"] = file_hash(
        Path("bayesfilter/inference/surrogate_hmc_dctr_selection.py"))
    write_json(args.output / "scope.json", scope)
    fields = ("filtering_mean", "filtering_covariance", "post_reset_mean", "post_reset_covariance")
    nominal = tf.constant([[-.6], [0.], [.6], [.9]], tf.float32)
    stress = tf.constant([[-.949], [-.94], [.94], [.949]], tf.float32)
    data_paths, curves = {}, {}
    started = time.monotonic()
    for partition in ("calibration", "validation"):
        path = args.reference_root / partition / "data.json"
        data = json.loads(path.read_text())
        data_paths[partition] = path
        root = args.output / partition
        root.mkdir()
        outputs, diagnostics = [], []
        for index, multiplier in enumerate(MULTIPLIERS):
            fixture = make_linear_fixture_target(data, 1024, protection_controls(multiplier), dtype=tf.float32)
            evaluate = fixture.compile(4, coordinates="physical", filter_moments=True)
            begin = time.monotonic()
            nominal_output = evaluate(nominal)
            stress_output = evaluate(stress)
            # Materialization synchronizes compiled execution before recording cost.
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
                reference, reference_force, _ = kalman_physical(tf.cast(positions[:, 0], tf.float64),
                                                                tf.cast(fixture.observations, tf.float64))
                reference -= tf.math.log(tf.constant(1.9, tf.float64))
                arms[name] = dict(value=value.numpy().tolist(), force=force.numpy().tolist(),
                    status={k:v.numpy().tolist() for k,v in status.items()},
                    healthy_rows=healthy.numpy().tolist(),
                    all_healthy=bool(tf.reduce_all(healthy)),
                    all_failures_flagged=bool(tf.reduce_all(healthy | flagged)),
                    reference_value=reference.numpy().tolist(), reference_force=reference_force.numpy().tolist())
            arm = dict(multiplier=multiplier, controls=protection_controls(multiplier),
                       fixture=dict(fixture.manifest), elapsed_seconds=elapsed, situations=arms)
            write_json(root / f"arm-{index}.json", finite_json(arm))
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
    write_json(args.output / "report.json", dict(verdict=verdict, selected=selection,
        memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        scope=scope, source_hashes=hashes, elapsed_seconds=time.monotonic()-started,
        allocator=tf.config.experimental.get_memory_info("GPU:0"),
        plan="docs/plans/surrogate-hmc-lg1-preparation-2026-09-19.md", posterior_comparisons="NOT_RUN"))
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
