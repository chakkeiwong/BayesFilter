#!/usr/bin/env python3
"""Bounded actual-route qualification of the named LG1 finite endpoint."""
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_lg1_scoped import prepare_consumer
    from bayesfilter.testing.surrogate_hmc_lg1_finite_endpoint import make_finite_endpoint
    from bayesfilter.inference.surrogate_hmc_program import write_json
    consumer, provenance = prepare_consumer(args.reference_root, args.selection)
    endpoint, physical, chart = make_finite_endpoint(consumer)
    @tf.function(input_signature=[tf.TensorSpec([4,1],tf.float32)], jit_compile=True, autograph=False)
    def original_physical(phi):
        return consumer.fixture.physical(phi)

    nominal = tf.constant([[-3.],[-1.],[1.],[3.]],tf.float64)
    extreme = tf.constant([[-100.],[-20.],[20.],[100.]],tf.float64)
    phi = tf.constant([[-.8],[-.2],[.2],[.8]],tf.float64)
    new, status = chart(nominal)
    old, _, old_status = consumer.evaluate_force(nominal)
    raw, _, raw_status = original_physical(tf.cast(phi,tf.float32))
    promoted, physical_status = physical(phi)
    tail, tail_status = chart(extreme)
    permutation = tf.constant([2,0,3,1])
    permuted, _ = chart(tf.gather(nominal,permutation))
    altered = tf.concat([nominal[:1],tf.constant([[-2.],[.5],[2.]],tf.float64)],axis=0)
    replaced, _ = chart(altered)
    replay, _ = chart(nominal)
    force_before = consumer.force.function(nominal)
    endpoint.function(extreme)
    force_after = consumer.force.function(nominal)
    discrepancy = tf.abs(new-tf.cast(old,tf.float64))
    checks = dict(
        physical_values_unchanged=bool(tf.reduce_all(promoted==tf.cast(raw,tf.float64))),
        physical_status_unchanged=bool(tf.reduce_all(physical_status["value_valid"]==raw_status["value_valid"])),
        nominal_value_valid=bool(tf.reduce_all(status["value_valid"] & old_status["value_valid"])),
        nominal_nonharm=bool(tf.reduce_all(discrepancy <= 1e-3*(1+tf.abs(tf.cast(old,tf.float64))))),
        extreme_value_valid=bool(tf.reduce_all(tail_status["value_valid"] & tf.math.is_finite(tail))),
        extreme_rounding_flagged=bool(tf.reduce_all(tail_status["coordinate_boundary_rounding"])),
        no_numerical_support_exclusion=not bool(tf.reduce_any(tail_status["outside_declared_support"])),
        replay=bool(tf.reduce_all(replay==new)),
        permutation=bool(tf.reduce_all(permuted==tf.gather(new,permutation))),
        row_independence=bool(replaced[0]==new[0]),
        proposal_force_unchanged=bool(tf.reduce_all(force_before.force==force_after.force)
                                    & tf.reduce_all(force_before.force_valid==force_after.force_valid)))
    report = dict(verdict="PASS" if all(checks.values()) else "REVISE", checks=checks,
        endpoint_identity=endpoint.identity, force_identity=consumer.force.identity,
        provenance=provenance, memory_policy=memory, tf32=True, jit_compile=True,
        endpoint_precision="FP32 full physical DCTR density; FP64 prior chart/Jacobian",
        named_endpoint_is_distinct_from_old_unconstrained_scalar=True,
        nominal_old=old.numpy().tolist(), nominal_new=new.numpy().tolist(),
        nominal_absolute_difference=discrepancy.numpy().tolist(),
        extreme_value=tail.numpy().tolist(),
        extreme_status={k:v.numpy().tolist() for k,v in tail_status.items()},
        sampler_accuracy_established=False, default_readiness=False,
        remaining="Independent D reference, C scoped tuning, C/D, D/R, C/R and endpoint-flag archive wiring",
        plan="docs/plans/surrogate-hmc-lg1-finite-endpoint-2026-09-19.md",
        wall_seconds=time.monotonic()-start,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path("bayesfilter/testing/surrogate_hmc_lg1_finite_endpoint.py"))})
    write_json(args.output/"report.json",report)
    print(json.dumps({"verdict":report["verdict"],"checks":checks,"wall_seconds":report["wall_seconds"]}),flush=True)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
