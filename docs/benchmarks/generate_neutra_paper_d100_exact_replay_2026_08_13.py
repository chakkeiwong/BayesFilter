#!/usr/bin/env python3
"""Generate CPU-only exact posterior replay for one paper d100 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-paper-d100-fresh-baseline-plan-2026-08-13.md"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("paper_funnel", "paper_ill_cond_gaussian"), required=True
    )
    parser.add_argument("--gaussian-constants", type=Path, required=True)
    parser.add_argument("--training-size", type=int, default=1_048_576)
    parser.add_argument("--selection-size", type=int, default=65_536)
    parser.add_argument("--audit-size", type=int, default=65_536)
    parser.add_argument("--calibration-size", type=int, default=65_536)
    parser.add_argument("--initial-size", type=int, default=4)
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = _parse_args()
    root = args.output_root.resolve()
    constants_path = args.gaussian_constants.resolve()
    if root.exists():
        raise FileExistsError(f"output root must be fresh: {root}")
    if not PLAN.is_file() or not constants_path.is_file():
        raise FileNotFoundError("plan or Gaussian constants are missing")
    sizes = {
        "training_rows": int(args.training_size),
        "selection_rows": int(args.selection_size),
        "audit_rows": int(args.audit_size),
        "calibration_rows": int(args.calibration_size),
        "initial_rows": int(args.initial_size),
    }
    if any(value <= 1 for value in sizes.values()):
        raise ValueError("all replay partition sizes must exceed one")
    root.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.perf_counter()
    import tensorflow as tf

    from bayesfilter.inference.neutra_paper_d100_target import (
        load_paper_gaussian_spec,
        make_paper_funnel_spec,
        paper_funnel_standardized_residuals,
        sample_paper_d100_exact,
    )

    spec = (
        make_paper_funnel_spec()
        if args.target == "paper_funnel"
        else load_paper_gaussian_spec(constants_path)
    )
    seed_values = {
        "training_rows": (20260813, 52001),
        "selection_rows": (20260813, 52002),
        "audit_rows": (20260813, 52003),
        "calibration_rows": (20260813, 52004),
        "initial_rows": (20260813, 52005),
    }
    tensors = {
        name: sample_paper_d100_exact(spec, count, seed=seed_values[name])
        for name, count in sizes.items()
    }
    receipts = {}
    for name, tensor in tensors.items():
        path = root / f"{name}.tftensor"
        path.write_bytes(tf.io.serialize_tensor(tensor).numpy())
        receipts[name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "shape": [int(value) for value in tensor.shape],
            "dtype": tensor.dtype.name,
        }
    training = tensors["training_rows"]
    smoke = {
        "row_mean": float(tf.reduce_mean(training).numpy()),
        "row_second_moment": float(tf.reduce_mean(tf.square(training)).numpy()),
        "all_finite": bool(tf.reduce_all(tf.math.is_finite(training)).numpy()),
    }
    if spec.name == "paper_funnel":
        residual = paper_funnel_standardized_residuals(spec, training)
        smoke.update(
            {
                "y_mean": float(tf.reduce_mean(training[:, 0]).numpy()),
                "y_second_moment": float(
                    tf.reduce_mean(tf.square(training[:, 0])).numpy()
                ),
                "standardized_residual_mean": float(tf.reduce_mean(residual).numpy()),
                "standardized_residual_second_moment": float(
                    tf.reduce_mean(tf.square(residual)).numpy()
                ),
            }
        )
    else:
        centered = training - tf.constant(spec.mean, tf.float64)[tf.newaxis, :]
        whitened = tf.transpose(
            tf.linalg.triangular_solve(
                tf.constant(spec.cholesky, tf.float64),
                tf.transpose(centered),
                lower=True,
            )
        )
        smoke.update(
            {
                "whitened_mean": float(tf.reduce_mean(whitened).numpy()),
                "whitened_second_moment": float(
                    tf.reduce_mean(tf.square(whitened)).numpy()
                ),
            }
        )
    manifest = {
        "schema": "bayesfilter.neutra.paper_d100_exact_replay.v1",
        "plan": PLAN.as_posix(),
        "target": spec.manifest_payload(),
        "gaussian_constants_path": constants_path.as_posix(),
        "gaussian_constants_sha256": _sha256(constants_path),
        "cpu_only": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "external_exact_sample_generation": True,
        "optimizer_update_performed": False,
        "sample_wise_loop_or_scalar_fallback": False,
        "partitions_disjoint_by_stateless_seed": True,
        "seeds": seed_values,
        "receipts": receipts,
        "smoke_diagnostics": smoke,
        "wall_seconds": time.perf_counter() - started,
        "command": " ".join(sys.argv),
    }
    _write(root / "replay_manifest.json", manifest)
    _write(
        root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.paper_d100_exact_replay_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(root.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "target": spec.name,
                "output_root": root.as_posix(),
                "wall_seconds": manifest["wall_seconds"],
                "smoke": smoke,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
