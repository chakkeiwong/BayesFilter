#!/usr/bin/env python3
"""CPU-only post-run diagnostics and checksums for retained HMC archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("inspection output must be new")
    import tensorflow as tf
    from bayesfilter.inference.surrogate_hmc_program import write_json
    report = json.loads((args.sampling_root/"report.json").read_text())
    result = dict(cpu_only=True, gpu_intentionally_hidden=True, inference_attempt=report["inference_attempt"],
                  numerical_backend="TensorFlow diagnostic eager exception", arms={})
    for arm, record in report["arms"].items():
        arrays = {}
        endpoint_fields = None
        for chunk in record.get("chunks", []):
            path = Path(chunk["manifest"])
            manifest = json.loads(path.read_text())
            fields = manifest.get("endpoint_status_fields")
            if endpoint_fields is not None and fields != endpoint_fields:
                raise ValueError("endpoint status schema changed between chunks")
            endpoint_fields = fields
            for name, tensor in manifest["chunk"].items():
                tensor_path = path.parent/tensor["path"]
                if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != tensor["sha256"]:
                    raise ValueError("archive checksum mismatch: " + str(tensor_path))
                value = tf.io.parse_tensor(tf.io.read_file(str(tensor_path)),tf.as_dtype(tensor["dtype"]))
                arrays.setdefault(name,[]).append(value)
            raw = manifest["raw_samples"]
            if hashlib.sha256((path.parent/raw["path"]).read_bytes()).hexdigest() != raw["sha256"]:
                raise ValueError("raw sample checksum mismatch")
        if not arrays:
            result["arms"][arm] = dict(status=record["status"], retained_draws=0)
            continue
        def joined(name):
            return tf.concat(arrays[name],axis=0)
        fallback, divergence = joined("force_fallback"), joined("divergence")
        statuses, energy = joined("endpoint_status"), joined("delta_h")
        result["arms"][arm] = dict(status=record["status"], retained_draws=int(tf.shape(energy)[0]),
            checksum_validation="PASS",
            acceptance_by_chain=tf.reduce_mean(tf.cast(joined("is_accepted"),tf.float64),axis=0).numpy().tolist(),
            force_fallback_by_chain=tf.reduce_sum(tf.cast(fallback,tf.int32),axis=0).numpy().tolist(),
            divergence_by_chain=tf.reduce_sum(tf.cast(divergence,tf.int32),axis=0).numpy().tolist(),
            numerical_endpoint_failures_by_chain=tf.reduce_sum(statuses[...,8],axis=0).numpy().tolist(),
            maximum_absolute_energy_error_by_chain=tf.reduce_max(tf.abs(energy),axis=0).numpy().tolist(),
            latent_minimum_by_chain=tf.reduce_min(joined("states"),axis=0).numpy().tolist(),
            latent_maximum_by_chain=tf.reduce_max(joined("states"),axis=0).numpy().tolist())
        if endpoint_fields and "coordinate_boundary_rounding" in endpoint_fields:
            index = endpoint_fields.index("coordinate_boundary_rounding")
            result["arms"][arm]["coordinate_boundary_rounding_by_chain"] = tf.reduce_sum(
                statuses[...,index],axis=0).numpy().tolist()
        else:
            result["arms"][arm]["coordinate_boundary_rounding_by_chain"] = "NOT_ARCHIVED_BY_HISTORICAL_SOURCE"
    write_json(args.output,result)
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    main()
