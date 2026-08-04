#!/usr/bin/env python3
"""Bounded KSC repair diagnostic before any KSC NeuTra training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/bayesfilter-neutra-four-blocked-target-repair-and-admission-plan-2026-07-31.md"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def run(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"fresh output required: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()

    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.highdim.sv_mixture_cut4 import (
        StochasticVolatilitySSM,
        independent_panel_sv_mixture_cut4_filter,
        independent_panel_sv_mixture_ukf_filter,
        scalar_sv_mixture_dense_reference,
    )
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        generate_frozen_exact_sv_dataset_tf,
        source_chart_physical_parameters,
    )
    from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
        KSC_UKF_RAW_OBSERVATION_SHA256,
        KSC_UKF_STATE_SHA256,
        transformed_ksc_observations,
    )

    states, observations = generate_frozen_exact_sv_dataset_tf()
    state_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(states).numpy())).hexdigest()
    observation_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(observations).numpy())
    ).hexdigest()
    if state_hash != KSC_UKF_STATE_SHA256 or observation_hash != KSC_UKF_RAW_OBSERVATION_SHA256:
        raise RuntimeError("frozen KSC dataset hash mismatch")
    fixed = tf.constant(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        tf.float64,
    )
    normal = tfp.distributions.Normal(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64))
    truth = normal.quantile(tf.constant([(0.6 - 0.1) / 0.8, (0.4 - 0.1) / 0.8], tf.float64))
    points = tf.concat((fixed, truth[None, :]), axis=0)
    transformed = transformed_ksc_observations(observations[:20])
    model = StochasticVolatilitySSM(sigma=1.0)
    rows = []
    for row in tf.unstack(points):
        gamma, beta = source_chart_physical_parameters(row[None, :])
        theta_model = tf.stack((normal.quantile(gamma[0]), tf.math.log(beta[0])))
        with tf.device("/CPU:0"):
            dense = scalar_sv_mixture_dense_reference(
                model, theta_model, observations[:20], order=601, radius=8.0
            )
            ukf = independent_panel_sv_mixture_ukf_filter(
                observations[:20], gamma=gamma, beta=beta, sigma=tf.constant([1.0], tf.float64)
            )
            cut4 = independent_panel_sv_mixture_cut4_filter(
                observations[:20], gamma=gamma, beta=beta, sigma=tf.constant([1.0], tf.float64)
            )
        rows.append(
            {
                "theta": row,
                "dense_value": dense.log_likelihood,
                "ukf_value": ukf.log_likelihood,
                "cut4_value": cut4.log_likelihood,
                "ukf_abs_gap": tf.abs(ukf.log_likelihood - dense.log_likelihood),
                "cut4_abs_gap": tf.abs(cut4.log_likelihood - dense.log_likelihood),
                "ukf_diagnostics": ukf.diagnostics,
                "cut4_diagnostics": cut4.diagnostics,
            }
        )
    value_rows = _safe(rows)
    max_ukf = max(float(r["ukf_abs_gap"]) for r in value_rows)
    max_cut4 = max(float(r["cut4_abs_gap"]) for r in value_rows)
    result = {
        "schema": "bayesfilter.neutra_ksc_ukf_repair_diagnostic.v1",
        "status": "TERMINAL_KSC_UKF_REPAIR_DIAGNOSTIC",
        "decision": "KEEP_KSC_UKF_TARGET_BLOCKED_REPAIR_NOT_ADMITTED",
        "question": "Does an existing component-enumerated higher-order filter justify a new KSC UKF repair?",
        "target": {
            "horizon": 20,
            "raw_observation_sha256": observation_hash,
            "state_sha256": state_hash,
            "transform": "log(y^2 + 1e-8)",
            "audit_point_count": int(points.shape[0]),
        },
        "dense_reference": {"order": 601, "radius": 8.0, "device": "/CPU:0"},
        "rows": value_rows,
        "max_absolute_gap": {"component_ukf": max_ukf, "component_cut4": max_cut4},
        "hard_vetoes": [
            "existing component-enumerated routes are diagnostics, not an admitted KSC-UKF target",
            "no bounded mixture-retention UKF implementation was executed",
        ],
        "training_launched": False,
        "nonclaims": [
            "not exact-SV evidence",
            "not KSC target admission",
            "not NeuTra or HMC evidence",
            "not a ranking of approximate filters",
        ],
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "plan": PLAN,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensorflow": tf.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_root)
    print(json.dumps({"decision": result["decision"], "max_absolute_gap": result["max_absolute_gap"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
