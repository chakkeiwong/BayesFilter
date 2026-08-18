#!/usr/bin/env python3
"""Decompose the frozen banana predictive MMD residual by feature family."""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNNER = ROOT / "docs/benchmarks/run_neutra_banana_predictive_equivalence_2026_08_16.py"
PLAN = ROOT / "docs/plans/bayesfilter-neutra-unfinished-lanes-closeout-plan-2026-08-17.md"
CONFIRMATION = ROOT / "docs/plans/artifacts/neutra-banana-hmc-l10-confirmation-2026-08-16-r1"
ARCHIVE = CONFIRMATION / "original_bank/archive/retained/original_bank-cumulative-model.tftensor"
RESULT = CONFIRMATION / "result.json"
DIMENSION = 16
CHAINS = 4
CURVATURE = 0.35
BLOCKS = (32, 64, 128)
WINDOWS = (0, 904)
DRAW_COUNT = 4096
BOOTSTRAPS = 256
CALIBRATION = 64


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("banana_mmd_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen banana runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy().tolist())
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
    path.write_text(json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _features(tf: Any, samples: Any) -> Mapping[str, Any]:
    c = tf.constant(CURVATURE, tf.float64)
    z1 = samples[:, :, 1] - c * (tf.square(samples[:, :, 0]) - 1.0)
    latent = tf.concat((samples[:, :, :1], z1[:, :, None], samples[:, :, 2:]), axis=-1)
    residual = samples[:, :, 1] - c * (tf.square(samples[:, :, 0]) - 1.0)
    return {
        "raw": samples,
        "latent": latent,
        "banana_pair": samples[:, :, :2],
        "banana_residual": tf.stack((samples[:, :, 0], residual), axis=-1),
        "nonlinear_tail": tf.stack((samples[:, :, 0], samples[:, :, 1], tf.square(samples[:, :, 0]), residual), axis=-1),
        "coordinate_0": samples[:, :, 0:1],
        "coordinate_1": samples[:, :, 1:2],
        "coordinate_2": samples[:, :, 2:3],
    }


def _bandwidths(dimension: int) -> tuple[float, ...]:
    scale = (2.0 * float(dimension)) ** 0.5
    return (0.5 * scale, scale, 2.0 * scale)


def _kernel_matrix(tf: Any, values: Any, block: int, bandwidth: float) -> Any:
    chains, draws, dim = values.shape
    count = chains * (draws // block)
    flat = tf.reshape(values, [count * block, dim])
    norms = tf.reduce_sum(tf.square(flat), axis=1, keepdims=True)
    dist = tf.maximum(norms + tf.transpose(norms) - 2.0 * tf.matmul(flat, flat, transpose_b=True), 0.0)
    kernel = tf.exp(-dist / tf.constant(2.0 * bandwidth * bandwidth, tf.float64))
    kernel = tf.reshape(kernel, [count, block, count, block])
    return tf.reduce_mean(kernel, axis=(1, 3))


def _cross_kernel(tf: Any, left: Any, right: Any, block: int, bandwidth: float) -> Any:
    lc, draws, dim = left.shape
    rc, rdraws, rdim = right.shape
    if draws != rdraws or dim != rdim:
        raise ValueError("feature arms must have equal draw/dimension shapes")
    ln = lc * (draws // block)
    rn = rc * (draws // block)
    lf = tf.reshape(left, [ln * block, dim])
    rf = tf.reshape(right, [rn * block, dim])
    lnorm = tf.reduce_sum(tf.square(lf), axis=1, keepdims=True)
    rnorm = tf.reduce_sum(tf.square(rf), axis=1, keepdims=True)
    dist = tf.maximum(lnorm + tf.transpose(rnorm) - 2.0 * tf.matmul(lf, rf, transpose_b=True), 0.0)
    kernel = tf.exp(-dist / tf.constant(2.0 * bandwidth * bandwidth, tf.float64))
    kernel = tf.reshape(kernel, [ln, block, rn, block])
    return tf.reduce_mean(kernel, axis=(1, 3))


@functools.lru_cache(maxsize=64)
def _compiled_mmd(tf_module: Any, chains: int, draws: int, dim: int, block: int, bootstraps: int) -> Any:
    """Cache one fixed-shape XLA MMD kernel for each feature family/block."""

    def core(left: Any, right: Any, seed: Any) -> tuple[Any, ...]:
        blocks = draws // block
        total = chains * blocks
        bandwidths = _bandwidths(dim)
        left_k = []
        right_k = []
        cross_k = []
        for bandwidth in bandwidths:
            left_k.append(_kernel_matrix(tf_module, left, block, bandwidth))
            right_k.append(_kernel_matrix(tf_module, right, block, bandwidth))
            cross_k.append(_cross_kernel(tf_module, left, right, block, bandwidth))
        ones = tf_module.ones([total], tf_module.float64)
        denom = tf_module.constant(float(total * total), tf_module.float64)
        point_bw = tf_module.stack([
            (tf_module.einsum("i,ij,j->", ones, a, ones)
             + tf_module.einsum("i,ij,j->", ones, b, ones)
             - 2.0 * tf_module.einsum("i,ij,j->", ones, c, ones)) / denom
            for a, b, c in zip(left_k, right_k, cross_k, strict=True)
        ])
        split = tf_module.random.experimental.stateless_split(seed, 2)
        li = tf_module.random.stateless_uniform(
            [bootstraps, chains, blocks], split[0], minval=0, maxval=blocks,
            dtype=tf_module.int32,
        )
        ri = tf_module.random.stateless_uniform(
            [bootstraps, chains, blocks], split[1], minval=0, maxval=blocks,
            dtype=tf_module.int32,
        )
        offsets = tf_module.reshape(
            tf_module.range(chains, dtype=tf_module.int32) * blocks,
            [1, chains, 1],
        )
        lc = tf_module.reduce_sum(
            tf_module.one_hot(li + offsets, total, dtype=tf_module.float64), axis=2
        )
        rc = tf_module.reduce_sum(
            tf_module.one_hot(ri + offsets, total, dtype=tf_module.float64), axis=2
        )
        lc = tf_module.reduce_sum(lc, axis=1)
        rc = tf_module.reduce_sum(rc, axis=1)
        boot_bw = tf_module.stack([
            (tf_module.einsum("bi,ij,bj->b", lc, a, lc)
             + tf_module.einsum("bi,ij,bj->b", rc, b, rc)
             - 2.0 * tf_module.einsum("bi,ij,bj->b", lc, c, rc)) / denom
            for a, b, c in zip(left_k, right_k, cross_k, strict=True)
        ], axis=1)
        boot = tf_module.reduce_mean(boot_bw, axis=1)
        sorted_boot = tf_module.sort(boot)
        index = tf_module.cast(
            tf_module.math.ceil(
                0.99 * tf_module.cast(bootstraps - 1, tf_module.float64)
            ),
            tf_module.int32,
        )
        return (
            tf_module.reduce_mean(point_bw),
            tf_module.gather(sorted_boot, index),
            point_bw,
            tf_module.gather(tf_module.sort(boot_bw, axis=0), index, axis=0),
        )

    return tf_module.function(core, autograph=False, jit_compile=True)


def _mmd(tf: Any, left: Any, right: Any, block: int, bootstraps: int, seed: tuple[int, int]) -> Mapping[str, Any]:
    chains, draws, dim = left.shape
    if left.shape != right.shape or draws % block:
        raise ValueError("MMD arms must have equal shapes divisible by block")
    point, upper, per_point, per_upper = _compiled_mmd(
        tf, int(chains), int(draws), int(dim), int(block), int(bootstraps)
    )(left, right, tf.constant(seed, tf.int32))
    return {
        "dimension": int(dim), "bandwidths": _bandwidths(int(dim)), "point": point,
        "per_bandwidth_point": per_point, "upper_99": upper,
        "per_bandwidth_upper_99": per_upper,
        "bootstrap_count": bootstraps, "block_length": block,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--draw-count", type=int, default=DRAW_COUNT)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--calibration-count", type=int, default=CALIBRATION)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.cpu_smoke else str(args.device)
    started = time.perf_counter()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = {"mode": "cpu_smoke"} if args.cpu_smoke else configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not args.cpu_smoke and len(logical) != 1:
        raise RuntimeError("expected one visible GPU")
    if not ARCHIVE.is_file() or not RESULT.is_file():
        raise FileNotFoundError("frozen banana confirmation artifacts missing")
    if _sha256(ARCHIVE) != json.loads((CONFIRMATION / "original_bank/sequential_result.json").read_text())["cumulative_archives"]["retained"]["model_sha256"]:
        raise RuntimeError("candidate archive hash mismatch")
    module = _load_runner()
    rows = {}
    if args.draw_count < 128 or args.draw_count % 128:
        raise ValueError("draw-count must be a multiple of 128")
    for offset in args.windows:
        candidate, meta = module._load_candidate(tf, draw_count=args.draw_count, offset=offset)
        exact = module._exact_bank(tf, draw_count=offset + args.draw_count, seed=(20260817, 880000 + offset))[:, offset:offset + args.draw_count]
        feature_rows = {}
        candidate_features = _features(tf, candidate); exact_features = _features(tf, exact)
        for name in candidate_features:
            calibration = {str(b): [] for b in BLOCKS}
            for rep in range(args.calibration_count):
                left = module._exact_bank(tf, draw_count=offset + args.draw_count, seed=(20260817, 100000 + offset + 1000 * rep))[:, offset:offset + args.draw_count]
                right = module._exact_bank(tf, draw_count=offset + args.draw_count, seed=(20260817, 200000 + offset + 1000 * rep))[:, offset:offset + args.draw_count]
                lf = _features(tf, left)[name]; rf = _features(tf, right)[name]
                for b in BLOCKS:
                    calibration[str(b)].append(_mmd(tf, lf, rf, b, args.bootstraps, (20260817, 300000 + rep + b))["upper_99"])
            feature_result = {}
            for b in BLOCKS:
                cand = _mmd(tf, candidate_features[name], exact_features[name], b, args.bootstraps, (20260817, 400000 + offset + b))
                controls = tf.stack(calibration[str(b)])
                q99 = tf.gather(tf.sort(controls), tf.cast(tf.math.ceil(0.99 * tf.cast(args.calibration_count - 1, tf.float64)), tf.int32))
                feature_result[str(b)] = {**cand, "control_q99": q99, "control_q95": tf.gather(tf.sort(controls), tf.cast(tf.math.ceil(0.95 * tf.cast(args.calibration_count - 1, tf.float64)), tf.int32)), "control_count": args.calibration_count}
            feature_rows[name] = feature_result
        rows[str(offset)] = {"candidate_meta": meta, "features": feature_rows}
        _write(output / f"offset_{offset}.json", rows[str(offset)])
    manifest = {"schema": "bayesfilter.neutra.banana_feature_decomposition_manifest.v1", "plan": PLAN.as_posix(), "command": sys.argv, "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "runner_sha256": _sha256(Path(__file__).resolve()), "plan_sha256": _sha256(PLAN), "memory_policy": memory, "gpu": None if args.cpu_smoke else str(logical[0]), "dtype": "float64", "jit_compile": True, "tf32_enabled": False if not args.cpu_smoke else None, "draw_count": args.draw_count, "bootstraps": args.bootstraps, "calibration_count": args.calibration_count, "windows": args.windows, "blocks": BLOCKS, "nonclaims": ["no formal equality p-value", "no retraining decision", "no posterior correctness proof"]}
    _write(output / "run_manifest.json", manifest)
    result = {"schema": "bayesfilter.neutra.banana_feature_decomposition_result.v1", "manifest": manifest, "windows": rows, "decision": {"status": "diagnostic_complete", "promotion": False, "next": "use localized feature family only for a reviewed transport repair"}, "wall_seconds": time.perf_counter() - started}
    _write(output / "result.json", result)
    _write(output / "progress.json", {"schema": "bayesfilter.neutra.banana_feature_decomposition_progress.v1", "phase": "complete", "windows": list(rows)})
    hashes = {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.banana_feature_decomposition_hashes.v1", "artifacts": hashes})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
