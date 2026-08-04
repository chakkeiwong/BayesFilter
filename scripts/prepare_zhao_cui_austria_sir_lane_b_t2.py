#!/usr/bin/env python3
"""Prepare one sealed CPU T2 proposal cloud for later GPU/XLA training."""

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
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_prepared_tf import (  # noqa: E402
    PLAN_PATH,
    prepared_source_closure,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (  # noqa: E402
    LaneBT2ProposalCloud,
    estimate_t2_shifted_log_normalizer,
    generate_t2_proposal_cloud,
    select_t2_shift_constant,
    verify_b2_admission,
)


PLAN = Path(PLAN_PATH)
ROLE_SPECS = {
    "training": (4096, 73801, 73811),
    "validation": (8192, 73802, 73812),
    "calibration": (12288, 73803, 73813),
    "untouched": (16384, 73804, 73814),
}
CHUNK_SIZE = 64


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return value.numpy().item() if value.shape.rank == 0 else value.numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return str(value)
    return value


def _write_tensor(path: Path, value: tf.Tensor) -> Mapping[str, object]:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    tf.io.write_file(path.as_posix(), serialized)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
        "dtype": value.dtype.name,
        "shape": value.shape.as_list(),
    }


def _make_reference(sample_count: int, reference_seed: int) -> tf.Tensor:
    return tf.random.stateless_uniform(
        [18, sample_count],
        seed=tf.constant([reference_seed, 1], tf.int32),
        minval=tf.constant(1e-6, tf.float64),
        maxval=tf.constant(1.0 - 1e-6, tf.float64),
        dtype=tf.float64,
    )


def _chunked_cloud(
    *, artifact_dir: Path, role: str, sample_count: int, reference_seed: int, transition_seed: int
) -> LaneBT2ProposalCloud:
    artifact = load_lane_b_t1_artifact_v1_compat(artifact_dir)
    reference = _make_reference(sample_count, reference_seed)
    clouds = []
    for first in range(0, sample_count, CHUNK_SIZE):
        last = min(first + CHUNK_SIZE, sample_count)
        # Each transition substream is bound to its absolute first column.
        clouds.append(
            generate_t2_proposal_cloud(
                t1_artifact=artifact,
                reference_uniforms=reference[:, first:last],
                reference_seed=reference_seed,
                transition_seed=transition_seed + first,
                role=f"{role}_chunk_{first:05d}_{last:05d}",
            )
        )
    return LaneBT2ProposalCloud(
        joint_points=tf.concat([cloud.joint_points for cloud in clouds], axis=0),
        previous_log_target=tf.concat(
            [cloud.previous_log_target for cloud in clouds], axis=0
        ),
        previous_log_proposal=tf.concat(
            [cloud.previous_log_proposal for cloud in clouds], axis=0
        ),
        previous_correction=tf.concat(
            [cloud.previous_correction for cloud in clouds], axis=0
        ),
        transition_log_density=tf.concat(
            [cloud.transition_log_density for cloud in clouds], axis=0
        ),
        log_likelihood=tf.concat([cloud.log_likelihood for cloud in clouds], axis=0),
        reference_uniforms=reference,
        reference_seed=reference_seed,
        transition_seed=transition_seed,
        role=role,
    )


def build_result(artifact_dir: Path, role: str, output_dir: Path) -> Mapping[str, object]:
    started = time.monotonic()
    if tf.config.list_logical_devices("GPU"):
        raise RuntimeError("T2 preparation must hide GPU devices")
    sample_count, reference_seed, transition_seed = ROLE_SPECS[role]
    b2_hash = verify_b2_admission(ROOT)
    cloud = _chunked_cloud(
        artifact_dir=artifact_dir,
        role=role,
        sample_count=sample_count,
        reference_seed=reference_seed,
        transition_seed=transition_seed,
    )
    tensor_values = {
        "joint_points": cloud.joint_points,
        "previous_log_target": cloud.previous_log_target,
        "previous_log_proposal": cloud.previous_log_proposal,
        "previous_correction": cloud.previous_correction,
        "transition_log_density": cloud.transition_log_density,
        "log_likelihood": cloud.log_likelihood,
        "reference_uniforms": cloud.reference_uniforms,
    }
    tensors = {
        name: _write_tensor(output_dir / f"{name}.tensor", value)
        for name, value in tensor_values.items()
    }
    estimate = None
    shift = None
    if role == "calibration":
        shift = select_t2_shift_constant(cloud)
        estimate = estimate_t2_shifted_log_normalizer(cloud, shift)
    zero_estimate = estimate_t2_shifted_log_normalizer(
        cloud, tf.constant(0.0, tf.float64)
    )
    correction = cloud.log_importance_weight
    invalid_correction = (
        tf.math.is_nan(correction)
        | (tf.math.is_inf(correction) & (correction > 0.0))
    )
    finite_correction = tf.boolean_mask(correction, tf.math.is_finite(correction))
    shifted = correction - tf.reduce_max(correction)
    weights = tf.exp(shifted) / tf.reduce_sum(tf.exp(shifted))
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))
    minimum_ess = 200.0 if role == "calibration" else 100.0
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_prepared_cloud.v1",
        "status": "PREPARED_T2_CLOUD",
        "role": role,
        "cloud_manifest": cloud.manifest_payload(),
        "tensors": tensors,
        "zero_shift_estimate": zero_estimate.manifest_payload(),
        "calibrated_shift": shift,
        "calibrated_estimate": None if estimate is None else estimate.manifest_payload(),
        "diagnostics": {
            "finite_correction_log_weight_min": tf.reduce_min(finite_correction),
            "correction_log_weight_max": tf.reduce_max(correction),
            "zero_target_count": int(
                tf.reduce_sum(tf.cast(cloud.zero_target_mask, tf.int32)).numpy()
            ),
            "correction_effective_sample_size": ess,
            "minimum_effective_sample_size": minimum_ess,
            "cpu_process_peak_rss_bytes": peak_rss,
            "cpu_process_cap_bytes": 12 * 1024**3,
        },
        "gates": {
            "b2_admission_hash": b2_hash,
            "finite": bool(
                tf.reduce_all(
                    tf.math.is_finite(cloud.joint_points)
                ).numpy()
            )
            and not bool(tf.reduce_any(invalid_correction).numpy())
            and bool(tf.reduce_any(tf.math.is_finite(correction)).numpy()),
            "sample_count": cloud.sample_count == sample_count,
            "memory": peak_rss <= 12 * 1024**3,
            "effective_sample_size": bool((ess >= minimum_ess).numpy())
            if role != "untouched"
            else True,
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": tuple(sys.argv),
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "device": "explicit_cpu_hidden_sample_generation",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "worker_count": 1,
            "chunk_size": CHUNK_SIZE,
            "random_seeds": {
                "reference": reference_seed,
                "transition_base": transition_seed,
            },
            "plan": PLAN.as_posix(),
            "source_sha256": dict(prepared_source_closure()),
            "wall_time_seconds": time.monotonic() - started,
        },
        "nonclaims": (
            "prepared data only",
            "no T2 candidate selection or value admission",
            "no score, T20, HMC, production KR, or scientific claim",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--role", choices=tuple(ROLE_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    result = build_result(args.artifact_dir.resolve(), args.role, output)
    (output / "result.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    gates = result["gates"]
    if not all(
        bool(gates[name])
        for name in ("finite", "sample_count", "memory", "effective_sample_size")
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
