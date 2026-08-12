#!/usr/bin/env python3
"""Run the bounded pairwise-direction diagnostic on Austria SIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base


PLAN = Path(
    "docs/plans/"
    "bayesfilter-pairwise-radial-cap-zhao-cui-austria-diagnostic-plan-2026-08-06.md"
)
SCHEMA = "bayesfilter.pairwise_radial_cap_diagnostic.v1"
N = 1008
SEEDS = (98201, 98202, 98203)
CAPS = (0.0, 8.0, 4.0, 2.0)
BASE_CONTROLS = {
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1.0e-5,
    "higher_moment_correction_steps": 4,
    "higher_moment_strength": 0.2,
    "higher_moment_floor": 1.0e-5,
    "pairwise_moment_correction_steps": 4,
    "pairwise_moment_strength": 0.02,
    "pairwise_moment_floor": 1.0e-5,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def _target() -> dict[str, Any]:
    return base._build_targets()["austria_sir_T20"]


def _controls(cap: float) -> dict[str, Any]:
    return {**BASE_CONTROLS, "pairwise_particle_rms_cap": cap}


def _evaluator(target: dict[str, Any], cap: float):
    return base._make_evaluator(
        adapter=target["adapter"],
        horizon=20,
        observation_dim=9,
        state_dim=18,
        parameter_dim=3,
        transition_before_first_observation=True,
        controls=_controls(cap),
    )


def _row(
    evaluator: Any,
    target: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    initial, process = base._noise(seed, 20, 18)
    started = time.perf_counter()
    row = base._evaluate(
        evaluator,
        target["theta"],
        tf.cast(target["observations"], tf.float32),
        seed,
        target["design"],
    )
    # _evaluate creates the same stateless noise internally. Preserve the
    # shapes here as a manifest check without serializing the arrays.
    row["noise_shapes"] = [initial.shape.as_list(), process.shape.as_list()]
    row["wall_time_seconds"] = time.perf_counter() - started
    return row


def _arm(cap: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = all(bool(row["finite"] and row["program_valid"]) for row in rows)
    return {
        "arm_id": "uncapped" if cap == 0.0 else f"cap_{cap:g}",
        "cap": None if cap == 0.0 else cap,
        "controls": _controls(cap),
        "rows": rows,
        "all_finite": finite,
        "maximum_pre_cap_particle_rms": max(
            row["maximum_pairwise_pre_cap_particle_rms"] for row in rows
        ),
        "maximum_post_cap_particle_rms": max(
            row["maximum_pairwise_post_cap_particle_rms"] for row in rows
        ),
        "minimum_cap_scale": min(
            row["minimum_pairwise_particle_cap_scale"] for row in rows
        ),
    }


def run(output_root: Path, *, smoke_only: bool) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("radial-cap diagnostic requires a logical GPU")

    target = _target()
    seeds = SEEDS[:1] if smoke_only else SEEDS
    arms = []
    for cap in CAPS:
        evaluator = _evaluator(target, cap)
        rows = [_row(evaluator, target, seed) for seed in seeds]
        arms.append(_arm(cap, rows))
    hard_valid = all(arm["all_finite"] for arm in arms)
    status = (
        "PASS_GPU_XLA_SMOKE"
        if smoke_only and hard_valid
        else "PAIRWISE_RADIAL_CAP_DIAGNOSTIC_COMPLETE"
        if hard_valid
        else "PAIRWISE_RADIAL_CAP_DIAGNOSTIC_INVALID"
    )
    payload = {
        "schema_version": SCHEMA,
        "status": status,
        "hard_valid": hard_valid,
        "smoke_only": smoke_only,
        "plan": PLAN.as_posix(),
        "question": (
            "Does a smooth per-particle RMS cap reduce pairwise correction "
            "tail dominance while retaining finite Austria value and score?"
        ),
        "target": {
            "model_id": "austria_sir_T20",
            "target_owner": "empirical_weighted_particle_shape_targets",
            "zhao_cui_recursive_teacher_used": False,
            "zhao_cui_nonuse_reason": (
                "existing recursive Austria TT teacher fails before reset execution"
            ),
            "source_observation_sha256": target["source_observation_sha256"],
            "theta": [float(item) for item in target["theta"].numpy()],
            "horizon": 20,
            "particle_count": N,
            "state_dimension": 18,
            "parameter_dimension": 3,
        },
        "configuration": {
            "caps": [None if cap == 0.0 else cap for cap in CAPS],
            "seeds": list(seeds),
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "gpu_memory_growth": True,
        },
        "arms": arms,
        "device": {
            "logical_devices": [device.name for device in logical],
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "source_sha256": {
                path.as_posix(): _sha256(ROOT / path)
                for path in (
                    PLAN,
                    Path("bayesfilter/highdim/higher_moment_contract_e.py"),
                    Path("bayesfilter/highdim/cubature_genut_filter.py"),
                    Path("docs/benchmarks/run_moment_retuned_genut_whole_leaderboard.py"),
                    Path(__file__).relative_to(ROOT),
                )
            },
        },
        "inference_status": {
            "hard_veto_screen": "pass" if hard_valid else "fail",
            "statistically_supported_ranking": False,
            "descriptive_only": [
                "value and score differences",
                "tail-RMS reduction",
                "pairwise residual differences",
                "runtime differences",
            ],
            "default_readiness": False,
            "next_evidence_needed": (
                "heldout tuning plus the existing 16 common seeds if a finite cap "
                "is both active and descriptively viable"
            ),
        },
        "nonclaims": [
            "three seeds cannot rank stochastic arms",
            "the recursive Austria Zhao-Cui TT teacher remains blocked",
            "no exact-score, posterior, HMC, or default-readiness claim",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    payload = run(args.output_root.resolve(), smoke_only=args.smoke_only)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "hard_valid": payload["hard_valid"],
                "output": str(args.output_root.resolve()),
                "wall_time_seconds": payload["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
