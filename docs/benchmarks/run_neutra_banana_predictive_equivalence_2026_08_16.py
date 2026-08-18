#!/usr/bin/env python3
"""Run a dependence-aware output-law diagnostic for the frozen banana HMC candidate.

This is a target-specific diagnostic.  It does not emit a formal equality
test or promote the NeuTra candidate.  The MMD is the biased V statistic; its
uncertainty is obtained by resampling complete within-chain draw blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import functools
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-neutra-banana-predictive-equivalence-plan-2026-08-16.md"
RUNNER = Path(__file__).resolve()
CONFIRMATION = ROOT / "docs/plans/artifacts/neutra-banana-hmc-l10-confirmation-2026-08-16-r1"
CANDIDATE_RESULT = CONFIRMATION / "result.json"
CANDIDATE_ARCHIVE = CONFIRMATION / "original_bank/archive/retained/original_bank-cumulative-model.tftensor"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-banana-predictive-equivalence-2026-08-16-r1"
DIMENSION = 16
CHAIN_COUNT = 4
DEFAULT_DRAW_COUNT = 1024
DEFAULT_BOOTSTRAP_COUNT = 256
DEFAULT_CALIBRATION_COUNT = 32
DEFAULT_OFFSETS = (0, 1024)
CURVATURE = 0.35
BANDWIDTHS = (2.0, 4.0, 8.0)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--time-cap", type=float, default=1800.0)
    parser.add_argument("--draw-count", type=int, default=DEFAULT_DRAW_COUNT)
    parser.add_argument("--bootstrap-count", type=int, default=DEFAULT_BOOTSTRAP_COUNT)
    parser.add_argument("--calibration-count", type=int, default=DEFAULT_CALIBRATION_COUNT)
    parser.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_OFFSETS))
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--plan-path", type=Path, default=PLAN)
    return parser.parse_args()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy().tolist())
    if isinstance(value, tuple) and all(isinstance(item, (int, type(None))) for item in value):
        return list(value)
    if hasattr(value, "as_list"):
        return _safe(value.as_list())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _validate_args(args: argparse.Namespace) -> None:
    if args.draw_count < 2 or args.draw_count % 128:
        raise ValueError("draw-count must be a positive multiple of 128")
    if args.bootstrap_count < 2:
        raise ValueError("bootstrap-count must be at least two")
    if args.calibration_count < 2:
        raise ValueError("calibration-count must be at least two")
    if not args.offsets or any(offset < 0 for offset in args.offsets):
        raise ValueError("offsets must be nonnegative")
    if len(set(args.offsets)) != len(args.offsets):
        raise ValueError("offsets must be unique")
    for block_length in (32, 64, 128):
        if args.draw_count % block_length:
            raise ValueError("draw-count must be divisible by block lengths 32, 64, and 128")


def _banana(tf: Any, latent: Any) -> Any:
    if latent.shape.rank != 3 or latent.shape[-1] != DIMENSION:
        raise ValueError("banana latent samples must have shape [chain, draw, 16]")
    correction = tf.constant(CURVATURE, tf.float64) * (
        tf.square(latent[:, :, 0]) - tf.constant(1.0, tf.float64)
    )
    return tf.concat(
        (
            latent[:, :, :1],
            latent[:, :, 1:2] + correction[:, :, tf.newaxis],
            latent[:, :, 2:],
        ),
        axis=-1,
    )


def _exact_bank(tf: Any, *, draw_count: int, seed: tuple[int, int]) -> Any:
    latent = tf.random.stateless_normal(
        [CHAIN_COUNT, draw_count, DIMENSION],
        seed=tf.constant(seed, tf.int32),
        dtype=tf.float64,
    )
    return _banana(tf, latent)


def _load_candidate(tf: Any, *, draw_count: int, offset: int) -> tuple[Any, Mapping[str, Any]]:
    if not CANDIDATE_RESULT.is_file() or not CANDIDATE_ARCHIVE.is_file():
        raise FileNotFoundError("frozen banana confirmation artifacts are missing")
    result = json.loads(CANDIDATE_RESULT.read_text(encoding="utf-8"))
    manifest = result.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("confirmation result has no manifest")
    if manifest.get("training_seed") != 15 or manifest.get("training_updates") != 6000:
        raise ValueError("confirmation artifact is not the frozen seed-15 6000-update candidate")
    kernel = manifest.get("kernel")
    if not isinstance(kernel, Mapping) or kernel.get("num_leapfrog_steps") != 10:
        raise ValueError("confirmation artifact is not the frozen L=10 candidate")
    archive_hash = _sha256(CANDIDATE_ARCHIVE)
    sequential = json.loads(
        (CONFIRMATION / "original_bank/sequential_result.json").read_text(encoding="utf-8")
    )
    archive_entries = sequential.get("cumulative_archives", {}).get("retained", {})
    if archive_entries.get("model_sha256") != archive_hash:
        raise ValueError("candidate archive SHA-256 does not match sequential result")
    samples = tf.io.parse_tensor(tf.io.read_file(str(CANDIDATE_ARCHIVE)), out_type=tf.float64)
    if samples.shape != (5000, CHAIN_COUNT, DIMENSION):
        raise ValueError(f"unexpected candidate archive shape: {samples.shape}")
    if offset + draw_count > samples.shape[0]:
        raise ValueError("candidate offset exceeds retained archive")
    # Archives are [draw, chain, coordinate]; the diagnostic contract is
    # [chain, draw, coordinate] so that chain dependence is retained.
    selected = tf.transpose(samples[offset : offset + draw_count], [1, 0, 2])
    tf.debugging.assert_all_finite(selected, "candidate samples must be finite")
    return selected, {
        "archive": CANDIDATE_ARCHIVE.as_posix(),
        "archive_sha256": archive_hash,
        "confirmation_result_sha256": _sha256(CANDIDATE_RESULT),
        "transport_state_hash": manifest.get("transport_state_hash"),
        "kernel": kernel,
        "source_result": CANDIDATE_RESULT.as_posix(),
    }


def _block_kernel_matrices(tf: Any, samples: Any, block_length: int) -> Any:
    """Return [bandwidth, block, block] averages of RBF kernels."""
    chain_count, draw_count, dimension = samples.shape
    if chain_count < 1 or dimension != DIMENSION or draw_count % block_length:
        raise ValueError("invalid sample shape or block length")
    block_count = chain_count * (draw_count // block_length)
    flat = tf.reshape(samples, [block_count * block_length, DIMENSION])
    squared_norm = tf.reduce_sum(tf.square(flat), axis=1, keepdims=True)
    squared_distance = tf.maximum(
        squared_norm + tf.transpose(squared_norm)
        - 2.0 * tf.matmul(flat, flat, transpose_b=True),
        tf.constant(0.0, tf.float64),
    )
    rows = []
    for bandwidth in BANDWIDTHS:
        kernel = tf.exp(-squared_distance / tf.constant(2.0 * bandwidth * bandwidth, tf.float64))
        kernel = tf.reshape(kernel, [block_count, block_length, block_count, block_length])
        rows.append(tf.reduce_mean(kernel, axis=(1, 3)))
    return tf.stack(rows, axis=0)


def _block_kernel_cross(tf: Any, left: Any, right: Any, block_length: int) -> Any:
    """Return [bandwidth, left_block, right_block] cross-kernel averages."""
    left_chain_count, draw_count, left_dimension = left.shape
    right_chain_count, right_draw_count, right_dimension = right.shape
    if draw_count != right_draw_count or left_dimension != right_dimension or draw_count % block_length:
        raise ValueError("cross-arm shapes must agree and divide by block length")
    left_block_count = left_chain_count * (draw_count // block_length)
    right_block_count = right_chain_count * (draw_count // block_length)
    left_flat = tf.reshape(left, [left_block_count * block_length, left_dimension])
    right_flat = tf.reshape(right, [right_block_count * block_length, right_dimension])
    left_norm = tf.reduce_sum(tf.square(left_flat), axis=1, keepdims=True)
    right_norm = tf.reduce_sum(tf.square(right_flat), axis=1, keepdims=True)
    squared_distance = tf.maximum(
        left_norm + tf.transpose(right_norm)
        - 2.0 * tf.matmul(left_flat, right_flat, transpose_b=True),
        tf.constant(0.0, tf.float64),
    )
    rows = []
    for bandwidth in BANDWIDTHS:
        kernel = tf.exp(-squared_distance / tf.constant(2.0 * bandwidth * bandwidth, tf.float64))
        kernel = tf.reshape(kernel, [left_block_count, block_length, right_block_count, block_length])
        rows.append(tf.reduce_mean(kernel, axis=(1, 3)))
    return tf.stack(rows, axis=0)


def _bootstrap_mmd(
    tf: Any,
    left: Any,
    right: Any,
    *,
    block_length: int,
    bootstrap_count: int,
    seed: tuple[int, int],
    jit_compile: bool = True,
) -> Mapping[str, Any]:
    if left.shape != right.shape:
        raise ValueError("MMD arms must have equal shape")
    draw_count = int(left.shape[1])
    blocks_per_chain = draw_count // block_length
    total_blocks = CHAIN_COUNT * blocks_per_chain
    if type(jit_compile) is not bool:
        raise ValueError("jit_compile must be a bool")
    matrix_kernel, cross_kernel_fn = _compiled_kernels(
        tf, block_length, int(left.shape[0]), draw_count, jit_compile
    )
    left_kernel = matrix_kernel(left)
    right_kernel = matrix_kernel(right)
    cross_kernel = cross_kernel_fn(left, right)
    identity = tf.ones([total_blocks], tf.float64)
    denominator = tf.constant(float(total_blocks * total_blocks), tf.float64)
    point_per_bandwidth = (
        tf.einsum("i,kij,j->k", identity, left_kernel, identity) / denominator
        + tf.einsum("i,kij,j->k", identity, right_kernel, identity) / denominator
        - 2.0 * tf.einsum("i,kij,j->k", identity, cross_kernel, identity) / denominator
    )
    seeds = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 2)
    selected_left = tf.random.stateless_uniform(
        [bootstrap_count, CHAIN_COUNT, blocks_per_chain],
        seed=seeds[0], minval=0, maxval=blocks_per_chain, dtype=tf.int32,
    )
    selected_right = tf.random.stateless_uniform(
        [bootstrap_count, CHAIN_COUNT, blocks_per_chain],
        seed=seeds[1], minval=0, maxval=blocks_per_chain, dtype=tf.int32,
    )
    chain_offsets = tf.reshape(
        tf.range(CHAIN_COUNT, dtype=tf.int32) * blocks_per_chain,
        [1, CHAIN_COUNT, 1],
    )
    left_indices = selected_left + chain_offsets
    right_indices = selected_right + chain_offsets
    left_counts = tf.reduce_sum(
        tf.one_hot(left_indices, total_blocks, dtype=tf.float64), axis=2
    )
    right_counts = tf.reduce_sum(
        tf.one_hot(right_indices, total_blocks, dtype=tf.float64), axis=2
    )
    left_counts = tf.reduce_sum(left_counts, axis=1)
    right_counts = tf.reduce_sum(right_counts, axis=1)
    left_per_bandwidth = tf.einsum("bi,kij,bj->bk", left_counts, left_kernel, left_counts) / denominator
    right_per_bandwidth = tf.einsum("bi,kij,bj->bk", right_counts, right_kernel, right_counts) / denominator
    cross_per_bandwidth = tf.einsum("bi,kij,bj->bk", left_counts, cross_kernel, right_counts) / denominator
    bootstrap_per_bandwidth = left_per_bandwidth + right_per_bandwidth - 2.0 * cross_per_bandwidth
    point = tf.reduce_mean(point_per_bandwidth)
    bootstrap = tf.reduce_mean(bootstrap_per_bandwidth, axis=1)
    sorted_bootstrap = tf.sort(bootstrap)
    upper_index = tf.cast(tf.math.ceil(tf.constant(0.99, tf.float64) * tf.cast(bootstrap_count - 1, tf.float64)), tf.int32)
    upper = tf.gather(sorted_bootstrap, upper_index)
    return {
        "point": point,
        "upper_99": upper,
        "bootstrap_mean": tf.reduce_mean(bootstrap),
        "bootstrap_sd": tf.math.reduce_std(bootstrap),
        "per_bandwidth_point": point_per_bandwidth,
        "per_bandwidth_bootstrap_upper_99": tf.gather(tf.sort(bootstrap_per_bandwidth, axis=0), upper_index, axis=0),
        "block_length": block_length,
        "blocks_per_chain": blocks_per_chain,
        "bootstrap_count": bootstrap_count,
    }


@functools.lru_cache(maxsize=16)
def _compiled_kernels(
    tf_module: Any,
    block_length: int,
    chain_count: int,
    draw_count: int,
    jit_compile: bool,
) -> tuple[Any, Any]:
    """Cache fixed-shape XLA traces across calibration and candidate arms."""
    matrix_kernel = tf_module.function(
        lambda value: _block_kernel_matrices(tf_module, value, block_length),
        autograph=False,
        jit_compile=jit_compile,
    )
    cross_kernel = tf_module.function(
        lambda first, second: _block_kernel_cross(tf_module, first, second, block_length),
        autograph=False,
        jit_compile=jit_compile,
    )
    return matrix_kernel, cross_kernel


def _moments(tf: Any, samples: Any) -> Mapping[str, Any]:
    latent_one = samples[:, :, 1] - tf.constant(CURVATURE, tf.float64) * (
        tf.square(samples[:, :, 0]) - tf.constant(1.0, tf.float64)
    )
    latent = tf.concat((samples[:, :, :1], latent_one[:, :, tf.newaxis], samples[:, :, 2:]), axis=-1)
    return {
        "coordinate_mean": tf.reduce_mean(samples, axis=(0, 1)),
        "coordinate_variance": tf.math.reduce_variance(samples, axis=(0, 1)),
        "latent_coordinate_mean": tf.reduce_mean(latent, axis=(0, 1)),
        "latent_coordinate_variance": tf.math.reduce_variance(latent, axis=(0, 1)),
        "banana_residual_cross_moment": tf.reduce_mean(
            tf.square(samples[:, :, 0]) * latent_one, axis=(0, 1)
        ),
    }


def _run_pair(tf: Any, left: Any, right: Any, *, bootstrap_count: int, seed: tuple[int, int], jit_compile: bool) -> Mapping[str, Any]:
    by_block = {
        str(block): _bootstrap_mmd(
            tf, left, right, block_length=block, bootstrap_count=bootstrap_count, seed=(seed[0], seed[1] + block)
            , jit_compile=jit_compile
        )
        for block in (32, 64, 128)
    }
    return by_block


def _empirical_quantile(tf: Any, values: Any, probability: float) -> Any:
    count = int(values.shape[0])
    index = int(tf.math.ceil(tf.constant(probability * (count - 1), tf.float64)))
    return tf.gather(tf.sort(values), index)


def _calibration_envelope(tf: Any, *, count: int, draw_count: int, bootstrap_count: int, offset: int, jit_compile: bool) -> Mapping[str, Any]:
    rows = []
    for index in range(count):
        base = 170000 + 1000 * index + offset
        left = _exact_bank(tf, draw_count=offset + draw_count, seed=(20260816, base))[:, offset:offset + draw_count]
        right = _exact_bank(tf, draw_count=offset + draw_count, seed=(20260816, base + 1))[:, offset:offset + draw_count]
        rows.append(_run_pair(tf, left, right, bootstrap_count=bootstrap_count, seed=(20260816, 270000 + index), jit_compile=jit_compile))
    envelope = {}
    for block in (32, 64, 128):
        values = tf.stack([row[str(block)]["upper_99"] for row in rows])
        envelope[str(block)] = {
            "upper_95_quantile_of_control_upper_intervals": _empirical_quantile(tf, values, 0.95),
            "upper_99_quantile_of_control_upper_intervals": _empirical_quantile(tf, values, 0.99),
            "control_upper_intervals": values,
            "control_count": count,
        }
    return envelope


def main() -> int:
    args = _args()
    _validate_args(args)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    output.mkdir(parents=True)
    if args.cpu_smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    started = time.perf_counter()
    import tensorflow as tf

    memory_policy: Mapping[str, Any]
    if args.cpu_smoke:
        memory_policy = {"mode": "cpu_smoke", "cuda_visible_devices": "-1"}
    else:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.set_soft_device_placement(False)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        if len(tf.config.list_logical_devices("GPU")) != 1:
            raise RuntimeError("the serious diagnostic requires exactly one visible GPU")
    plan_path = args.plan_path.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan_copy = output / plan_path.name
    plan_copy.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write(output / "progress.json", {"schema": "bayesfilter.neutra.banana_predictive_equivalence_progress.v1", "phase": "started"})
    manifest = {
        "schema": "bayesfilter.neutra.banana_predictive_equivalence_manifest.v1",
        "plan": plan_path.as_posix(),
        "command": [sys.executable, *sys.argv],
        "git_commit": _git_commit(),
        "runner_sha256": _sha256(RUNNER),
        "plan_sha256": _sha256(plan_path),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_policy": memory_policy,
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False if not args.cpu_smoke else None,
        "dimension": DIMENSION,
        "chain_count": CHAIN_COUNT,
        "draw_count": args.draw_count,
        "bootstrap_count": args.bootstrap_count,
        "calibration_count": args.calibration_count,
        "offsets": args.offsets,
        "bandwidths": BANDWIDTHS,
        "curvature": CURVATURE,
        "candidate_confirmation_root": CONFIRMATION.as_posix(),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted" if not args.cpu_smoke else "cpu_smoke_non_evidence",
        "nonclaims": ["no formal equality p-value", "no posterior correctness proof", "no default or production readiness", "no SSL-LSTM transfer"],
    }
    _write(output / "run_manifest.json", manifest)
    all_rows = {}
    for offset in args.offsets:
        if time.perf_counter() - started > args.time_cap:
            raise TimeoutError("time cap exhausted before offset")
        candidate, candidate_meta = _load_candidate(tf, draw_count=args.draw_count, offset=offset)
        exact = _exact_bank(tf, draw_count=offset + args.draw_count, seed=(20260816, 880000 + offset))[:, offset:offset + args.draw_count]
        calibration = _calibration_envelope(tf, count=args.calibration_count, draw_count=args.draw_count, bootstrap_count=args.bootstrap_count, offset=offset, jit_compile=True)
        candidate_rows = _run_pair(tf, candidate, exact, bootstrap_count=args.bootstrap_count, seed=(20260816, 990000 + offset), jit_compile=True)
        all_rows[str(offset)] = {
            "candidate": candidate_rows,
            "calibration": calibration,
            "candidate_meta": candidate_meta,
            "candidate_moments": _moments(tf, candidate),
            "exact_moments": _moments(tf, exact),
            "candidate_shape": candidate.shape,
            "exact_shape": exact.shape,
        }
        _write(output / f"offset_{offset}.json", all_rows[str(offset)])
        _write(output / "progress.json", {"schema": "bayesfilter.neutra.banana_predictive_equivalence_progress.v1", "phase": f"offset_{offset}", "completed_offsets": list(all_rows)})
    result = {
        "schema": "bayesfilter.neutra.banana_predictive_equivalence_result.v1",
        "manifest": manifest,
        "offsets": all_rows,
        "decision": {
            "status": "diagnostic_complete",
            "promotion": False,
            "interpretation": "The candidate is compared with an exact banana law using a calibrated finite-sample block-bootstrap MMD screen; this is not a formal equality test.",
            "nonclaims": manifest["nonclaims"],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "result.json", result)
    _write(output / "progress.json", {"schema": "bayesfilter.neutra.banana_predictive_equivalence_progress.v1", "phase": "complete", "completed_offsets": list(all_rows)})
    hashes = {
        p.relative_to(output).as_posix(): _sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.banana_predictive_equivalence_hashes.v1", "artifacts": hashes})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"], "offsets": list(all_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
