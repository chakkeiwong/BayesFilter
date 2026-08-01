#!/usr/bin/env python3
"""Exercise compiled fail-closed behavior for the clean-XLA LGSSM route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as model
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


DTYPE = tf.float64
THETA = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _parse()
    preparation_path = _path(args.preparation)
    output = _path(args.output)
    if output.exists():
        raise FileExistsError(output)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    time_steps = int(preparation["target"]["time_steps"])
    if time_steps != 2:
        raise ValueError("invalid-chart XLA control requires the T=2 preparation")
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:time_steps], DTYPE
    )
    nodes = tf.constant(preparation["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["quadrature"]["weights"], DTYPE)
    active_indices = tf.constant(preparation["active_indices"], tf.int32)
    valid_row_scales = tf.constant(preparation["row_scales"], DTYPE)
    invalid_row_scales = tf.tensor_scatter_nd_update(
        valid_row_scales, [[0, 0]], [-valid_row_scales[0, 0]]
    )

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("compiled invalid-chart control requires a trusted GPU")
    evaluate = model.make_contract_e_tp_lgssm_score_informed_recursive_tf(
        observations,
        nodes,
        weights,
        active_indices,
        invalid_row_scales,
        feature_mode="finite_lookahead",
        lookahead_steps=8,
        jit_compile=True,
    )
    started = time.perf_counter()
    result = evaluate(THETA)
    _ = result["score"].numpy()
    elapsed = time.perf_counter() - started
    validity = result["valid_history"].numpy().tolist()
    final_particles_finite = bool(
        tf.reduce_all(tf.math.is_finite(result["final_particles"])).numpy()
    )
    objective_finite = bool(tf.math.is_finite(result["objective"]).numpy())
    score_finite = bool(tf.reduce_all(tf.math.is_finite(result["score"])).numpy())
    pass_control = (
        not all(validity)
        and not final_particles_finite
        and not objective_finite
        and not score_finite
    )
    if not pass_control:
        raise RuntimeError("compiled invalid chart did not fail closed")
    payload = {
        "schema": "bayesfilter.contract_e_tp.clean_xla_invalid_chart.v1",
        "status": "PASS_COMPILED_FAIL_CLOSED",
        "preparation": {
            "path": str(preparation_path.relative_to(ROOT)),
            "sha256": _sha256(preparation_path),
        },
        "mutation": "row_scales[0,0] sign flipped; equality system unchanged but row-scale validity false",
        "result": {
            "valid_history": validity,
            "final_particles_finite": final_particles_finite,
            "objective_finite": objective_finite,
            "score_finite": score_finite,
        },
        "execution": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "tensorflow_version": tf.__version__,
            "visible_gpu": gpus[0].name,
            "output_device": result["objective"].backing_device,
            "jit_compile": True,
            "dtype": DTYPE.name,
            "compile_plus_execution_seconds": elapsed,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "command": " ".join(sys.argv),
        },
        "nonclaims": [
            "compiled negative control only",
            "not scientific accuracy or HMC readiness evidence",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result": payload["result"]}, indent=2))


if __name__ == "__main__":
    main()
