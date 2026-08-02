#!/usr/bin/env python3
"""Prepare the fresh tail-certified one-shot T2 untouched cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_sampler_tf import (  # noqa: E402
    LaneBRetainedGridSampler,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tail_tf import (  # noqa: E402
    LaneBT2UntouchedTailCloud,
    SCHEMA,
    estimate_tail_log_normalizer,
    evaluate_t2_tail_chunk,
    tail_source_closure,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (  # noqa: E402
    generate_sealed_lane_b_dataset,
)


SAMPLE_COUNT = 16384
REFERENCE_SEED = 73804
TRANSITION_SEED = 73814
CHUNK_SIZE = 64
MEMORY_CAP_BYTES = 12 * 1024**3


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy().item() if value.shape.rank == 0 else value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_tensor(path: Path, value: tf.Tensor) -> Mapping[str, object]:
    tensor = tf.convert_to_tensor(value)
    serialized = tf.io.serialize_tensor(tensor)
    tf.io.write_file(path.as_posix(), serialized)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
    }


def build_result(parent_dir: Path, output_dir: Path) -> Mapping[str, object]:
    started = time.monotonic()
    if tf.config.list_logical_devices("GPU"):
        raise RuntimeError("T2 tail preparation must hide GPU devices")
    parent = load_lane_b_t1_artifact_v1_compat(parent_dir)
    sampler = LaneBRetainedGridSampler(parent)
    reference = tf.random.stateless_uniform(
        [18, SAMPLE_COUNT],
        seed=tf.constant([REFERENCE_SEED, 1], tf.int32),
        minval=tf.constant(1e-6, tf.float64),
        maxval=tf.constant(1.0 - 1e-6, tf.float64),
        dtype=tf.float64,
    )
    _states, observations, _all = generate_sealed_lane_b_dataset()
    z1_rows = []
    noise_rows = []
    correction_rows = []
    likelihood_rows = []
    transition_rows = []
    mask_rows = []
    margin_rows = []
    parity_rows = []
    for first in range(0, SAMPLE_COUNT, CHUNK_SIZE):
        last = first + CHUNK_SIZE
        retained = sampler.inverse(reference[:, first:last])
        noise = tf.random.stateless_normal(
            [CHUNK_SIZE, 18],
            seed=tf.constant([TRANSITION_SEED + first, 2], tf.int32),
            dtype=tf.float64,
        )
        tail = evaluate_t2_tail_chunk(
            z1=retained.physical_points,
            transition_noise=noise,
            observation=observations[1],
        )
        z1_rows.append(retained.physical_points)
        noise_rows.append(noise)
        correction_rows.append(retained.correction_log_weights)
        likelihood_rows.append(tail.log_likelihood)
        transition_rows.append(tail.transition_log_density)
        mask_rows.append(tail.nonrepresentable_mask)
        margin_rows.append(tail.overflow_log_margin)
        parity_rows.append(tail.ordinary_relative_residual)
    cloud = LaneBT2UntouchedTailCloud(
        reference_uniforms=reference,
        z1=tf.concat(z1_rows, axis=0),
        transition_noise=tf.concat(noise_rows, axis=0),
        previous_correction=tf.concat(correction_rows, axis=0),
        log_likelihood=tf.concat(likelihood_rows, axis=0),
        transition_log_density=tf.concat(transition_rows, axis=0),
        nonrepresentable_mask=tf.concat(mask_rows, axis=0),
        overflow_log_margin=tf.concat(margin_rows, axis=0),
        role="untouched",
        reference_seed=REFERENCE_SEED,
        transition_seed=TRANSITION_SEED,
    )
    parity = tf.concat(parity_rows, axis=0)
    estimate = estimate_tail_log_normalizer(cloud, tf.constant(0.0, tf.float64))
    shifted = cloud.log_importance_weight - tf.reduce_max(cloud.log_importance_weight)
    weights = tf.exp(shifted) / tf.reduce_sum(tf.exp(shifted))
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    tensors = {
        name: _write_tensor(output_dir / f"{name}.tensor", value)
        for name, value in {
            "reference_uniforms": cloud.reference_uniforms,
            "z1": cloud.z1,
            "transition_noise": cloud.transition_noise,
            "previous_correction": cloud.previous_correction,
            "log_likelihood": cloud.log_likelihood,
            "transition_log_density": cloud.transition_log_density,
            "nonrepresentable_mask": cloud.nonrepresentable_mask,
            "overflow_log_margin": cloud.overflow_log_margin,
        }.items()
    }
    nonrepresentable_count = int(
        tf.reduce_sum(tf.cast(cloud.nonrepresentable_mask, tf.int32)).numpy()
    )
    zero_target_count = int(
        tf.reduce_sum(tf.cast(cloud.zero_target_mask, tf.int32)).numpy()
    )
    minimum_tail_margin = tf.reduce_min(
        tf.boolean_mask(cloud.overflow_log_margin, cloud.nonrepresentable_mask)
    )
    maximum_ordinary_residual = tf.reduce_max(
        tf.boolean_mask(parity, ~cloud.nonrepresentable_mask)
    )
    gates = {
        "cpu_hidden": True,
        "sample_count": cloud.sample_count == SAMPLE_COUNT,
        "all_rows_retained": cloud.sample_count == SAMPLE_COUNT,
        "nonrepresentable_rows_certified": nonrepresentable_count >= 1
        and bool((minimum_tail_margin > 0.0).numpy()),
        "nonrepresentable_rows_are_zero_target": zero_target_count >= nonrepresentable_count,
        "ordinary_signed_log_parity": bool((maximum_ordinary_residual <= 2e-11).numpy()),
        "memory": peak_rss <= MEMORY_CAP_BYTES,
    }
    return {
        "schema_version": SCHEMA,
        "status": "PREPARED_T2_UNTOUCHED_TAIL_CLOUD",
        "cloud_manifest": cloud.manifest_payload(),
        "tensors": tensors,
        "zero_shift_estimate": estimate.manifest_payload(),
        "diagnostics": {
            "effective_sample_size": ess,
            "nonrepresentable_count": nonrepresentable_count,
            "zero_target_count": zero_target_count,
            "minimum_tail_overflow_log_margin": minimum_tail_margin,
            "maximum_ordinary_signed_log_relative_residual": maximum_ordinary_residual,
            "cpu_process_peak_rss_bytes": peak_rss,
        },
        "gates": gates,
        "source_sha256": dict(tail_source_closure()),
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": tuple(sys.argv),
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "device": "explicit_cpu_hidden_tail_preparation",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "worker_count": 1,
            "chunk_size": CHUNK_SIZE,
            "random_seeds": {
                "reference": REFERENCE_SEED,
                "transition_base": TRANSITION_SEED,
            },
            "wall_time_seconds": time.monotonic() - started,
        },
        "nonclaims": (
            "FP64 extended-real zero certificate not exact-real zero density",
            "no value admission before selected-artifact claim",
            "no score, T5/T10/T20, HMC, production KR, or scientific claim",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-t1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"tail output already exists: {output}")
    output.mkdir(parents=True)
    result = build_result(args.parent_t1_dir.resolve(), output)
    (output / "result.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    if not all(bool(value) for value in result["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
