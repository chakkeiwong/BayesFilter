#!/usr/bin/env python3
"""Full-DCTR finite posterior quadrature, separate from HMC sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_lg1_scoped import prepare_consumer
    from bayesfilter.testing.surrogate_hmc_lg1_finite_endpoint import make_finite_endpoint
    from bayesfilter.testing.surrogate_hmc_lg1_finite_reference import (
        validate_endpoint_qualification, normalize_grid, compare_references, evaluate_kalman)
    from bayesfilter.inference.surrogate_hmc_program import write_json
    consumer, provenance = prepare_consumer(args.reference_root, args.selection)
    endpoint, physical, _ = make_finite_endpoint(consumer)
    qualification = validate_endpoint_qualification(args.endpoint_root, endpoint, provenance)
    reference_path = args.reference_root/"claim/reference.json"
    reference = json.loads(reference_path.read_text())
    record = reference["tensors"]["summary"]
    summary_path = reference_path.parent/record["path"]
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("reference summary checksum mismatch")
    summary = tf.io.parse_tensor(tf.io.read_file(str(summary_path)), tf.float64)
    observations = tf.cast(consumer.fixture.observations, tf.float64)
    report = dict(qualified=False, endpoint_identity=endpoint.identity, qualification=qualification,
        provenance=provenance, memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        reference_path=str(reference_path.resolve()), reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        plan="docs/plans/surrogate-hmc-lg1-finite-comparisons-2026-09-20.md", grids=[],
        function_names=reference["function_names"], margins=reference["margins"],
        rigorous_error_bound=False, sampler_accuracy_established=False)
    expectations = []
    for intervals in (128,256,512,1024):
        begun = time.monotonic()
        phi = tf.linspace(tf.constant(-.95,tf.float64),tf.constant(.95,tf.float64),intervals+1)
        evaluation_phi = tf.concat([tf.math.nextafter(phi[:1],tf.zeros([1],tf.float64)),phi[1:-1],
                                   tf.math.nextafter(phi[-1:],tf.zeros([1],tf.float64))],0)
        values, statuses = [], {}
        for offset in range(0,intervals+1,4):
            batch = evaluation_phi[offset:offset+4]
            count = int(tf.size(batch))
            padded = tf.concat([batch,tf.repeat(batch[-1:],4-count)],0)[:,None]
            value, status = physical(padded)
            values.append(value[:count])
            for key,tensor in status.items():
                statuses.setdefault(key,[]).append(tensor[:count])
        value = tf.concat(values,0)
        status = {key:tf.concat(rows,0) for key,rows in statuses.items()}
        valid = bool(tf.reduce_all(status["value_valid"] & tf.math.is_finite(value)))
        grid_root = args.output/f"grid-{intervals}"
        grid_root.mkdir()
        tensors = {"phi":phi,"log_density":value,**{f"status_{k}":v for k,v in status.items()}}
        row = dict(intervals=intervals, value_valid=valid, tensors={},
            force_invalid_count=int(tf.reduce_sum(tf.cast(~status["force_valid"],tf.int32))),
            coordinate_rounding_count=int(tf.reduce_sum(tf.cast(status["coordinate_boundary_rounding"],tf.int32))))
        if valid:
            predictive = evaluate_kalman(phi,observations)[2]
            density, functions, log_normalizer = normalize_grid(phi,value,predictive,summary)
            valid = bool(tf.reduce_all(tf.math.is_finite(functions)) & tf.math.is_finite(log_normalizer))
            row.update(value_valid=valid, expectations=functions.numpy().tolist(),log_normalizer=float(log_normalizer))
            tensors.update(density=density, functions=functions, summary=summary)
            expectations.append(functions)
        for key,tensor in tensors.items():
            path = grid_root/(key+".tensor")
            tf.io.write_file(str(path),tf.io.serialize_tensor(tensor))
            row["tensors"][key] = dict(path=str(path.relative_to(args.output)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        row["wall_seconds"] = time.monotonic()-begun
        report["grids"].append(row)
        write_json(args.output/"report.json",report)
        print("GRID",intervals,"valid",valid,"seconds",row["wall_seconds"],flush=True)
        if not valid:
            report["failure"] = "finite endpoint numerical value failure"
            break
    if len(expectations)==4 and all(row["value_valid"] for row in report["grids"]):
        functions = tf.stack(expectations)
        changes = tf.abs(functions[1:]-functions[:-1])
        allowance = 2*tf.reduce_max(changes[-2:],axis=0)
        margins = tf.constant(reference["margins"],tf.float64)
        qualified = bool(tf.reduce_all(allowance<=margins/10) & tf.reduce_all(changes[-1]<=changes[-2]*.5+1e-5))
        report.update(qualified=qualified,reference_expectations=functions[-1].numpy().tolist(),
            empirical_refinement_allowance=allowance.numpy().tolist(),refinement_changes=changes.numpy().tolist(),
            D_vs_R=compare_references(functions[-1],allowance,tf.constant(reference["reference_expectations"],tf.float64),
                                     tf.constant(reference["empirical_refinement_allowance"],tf.float64),margins,qualified))
    report.update(wall_seconds=time.monotonic()-start,allocator=tf.config.experimental.get_memory_info("GPU:0"),
        source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__),Path("bayesfilter/testing/surrogate_hmc_lg1_finite_reference.py"))})
    write_json(args.output/"report.json",report)
    print(json.dumps({"qualified":report["qualified"],"D/R":report.get("D_vs_R",{}).get("status"),
                      "wall_seconds":report["wall_seconds"]}),flush=True)
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
