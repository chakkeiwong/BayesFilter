#!/usr/bin/env python3
"""Prepare independent LG1 model-posterior references; explicit CPU exception."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_lg1_tf import make_diagnostic_dataset
    from bayesfilter.testing.surrogate_hmc_lg1_reference import prepare_reference
    from bayesfilter.inference.surrogate_hmc_program import write_json, file_hash
    start = time.monotonic()
    reports = {}
    for partition in ("calibration", "validation", "claim"):
        root = args.output / partition
        root.mkdir()
        data = make_diagnostic_dataset(attempt=4, partition=partition)
        data["generation"] = "CPU TensorFlow tiny synthetic data; independent-reference exception"
        write_json(root / "data.json", data)
        report, tensors = prepare_reference(data["observations"])
        report["data_sha256"] = file_hash(root / "data.json")
        report["tensors"] = {}
        for key, tensor in tensors.items():
            path = root / (key + ".tfbin")
            tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
            report["tensors"][key] = {"path": path.name, "sha256": file_hash(path), "dtype": tensor.dtype.name}
        write_json(root / "reference.json", report)
        reports[partition] = dict(qualified=report["qualified"], mean=report["posterior_mean"],
                                 sd=report["posterior_sd"], reference=str(root / "reference.json"))
        print(partition, reports[partition], flush=True)
    files = [Path(__file__), Path("bayesfilter/testing/surrogate_hmc_lg1_reference.py"),
             Path("bayesfilter/testing/surrogate_hmc_lg1_tf.py")]
    write_json(args.output / "run-manifest.json", dict(
        command=sys.argv, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=sys.executable, tensorflow=tf.__version__, cpu_only=True, gpu_intentionally_hidden=True,
        plan="docs/plans/surrogate-hmc-lg1-preparation-2026-09-19.md", result="report.json",
        source_hashes={str(p): file_hash(p) for p in files}, wall_seconds=time.monotonic()-start,
        dataset_seeds="partition data.json", artifact_paths=reports))
    write_json(args.output / "report.json", dict(partitions=reports, qualified=all(x["qualified"] for x in reports.values()),
        posterior_comparisons="NOT_RUN", inference_attempts_consumed=0))
    if not all(x["qualified"] for x in reports.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
