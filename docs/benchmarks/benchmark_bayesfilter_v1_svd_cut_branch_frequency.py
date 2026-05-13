"""CPU-only SVD-CUT branch-frequency artifact for BayesFilter v1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter import StatePartition  # noqa: E402
from bayesfilter.structural_tf import make_affine_structural_tf  # noqa: E402
from bayesfilter.testing import svd_cut_branch_frequency_summary  # noqa: E402


def _model_from_params(params: tf.Tensor, *, repeated_spectrum: bool = False):
    phi1 = params[0]
    sigma = params[1]
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    initial_covariance = (
        tf.eye(2, dtype=tf.float64)
        if repeated_spectrum
        else tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64))
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.constant([0.1, -0.2], dtype=tf.float64),
        initial_covariance=initial_covariance,
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.stack(
            [
                tf.stack([phi1, tf.constant(-0.12, dtype=tf.float64)]),
                tf.constant([1.0, 0.0], dtype=tf.float64),
            ]
        ),
        innovation_matrix=tf.reshape(
            tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]),
            [2, 1],
        ),
        innovation_covariance=tf.constant([[0.43]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.25]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.19]], dtype=tf.float64),
    )


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_logical_devices()
        ],
    }


def _markdown(payload: dict[str, Any], json_path: Path | None) -> str:
    result = payload["result"]
    json_name = str(json_path) if json_path is not None else "stdout"
    return "\n".join(
        [
            "# BayesFilter v1 SVD-CUT Branch Frequency",
            "",
            "Purpose: quantify SVD-CUT derivative branch labels over a small "
            "parameter box without promoting SVD-CUT to HMC readiness.",
            "",
            "## Claim Scope",
            "",
            "Diagnostic artifact only.  A smooth fraction of one on this tiny "
            "box is not a general SVD-CUT HMC claim.",
            "",
            "## Result",
            "",
            f"The JSON file is authoritative: `{json_name}`.",
            "",
            "| Status | Smooth | Active Floor | Weak Gap | Smooth Fraction |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| {status} | {smooth} / {total} | {active} | {weak} | {fraction:.4f} |".format(
                status=payload["status"],
                smooth=int(result["smooth_count"]),
                total=int(result["total_count"]),
                active=int(result["active_floor_count"]),
                weak=int(result["weak_spectral_gap_count"]),
                fraction=float(result["smooth_fraction"]),
            ),
            "",
            "## Interpretation",
            "",
            "The result quantifies this parameter box only.  SVD-CUT HMC remains "
            "blocked pending target-specific sampler evidence and broader branch "
            "coverage.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    parameter_grid = tf.constant(
        [[0.27, 0.23], [0.31, 0.27], [0.35, 0.31]],
        dtype=tf.float64,
    )
    summary = svd_cut_branch_frequency_summary(
        observations,
        parameter_grid,
        _model_from_params,
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )
    result = asdict(summary)
    result["smooth_fraction"] = summary.smooth_fraction
    status = (
        "diagnostic_smooth_box"
        if summary.smooth_count == summary.total_count
        else "diagnostic_blocked_box"
    )
    payload = {
        "benchmark": "bayesfilter_v1_svd_cut_branch_frequency",
        "claim_scope": "diagnostic_only",
        "parameter_grid": parameter_grid.numpy().tolist(),
        "environment": _environment(),
        "status": status,
        "result": result,
        "blocked_claims": ["hmc_ready_svd_cut", "global_smooth_branch"],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
