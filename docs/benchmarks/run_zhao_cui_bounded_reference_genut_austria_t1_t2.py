#!/usr/bin/env python3
"""Run the bounded Zhao-Cui teacher/GenUT Austria T1/T2 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.set_soft_device_placement(False)
tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.highdim.cubature_genut_adapters import (  # noqa: E402
    parameterized_austria_sir_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design  # noqa: E402
from bayesfilter.highdim.cubature_genut_filter import finite_value_score  # noqa: E402
from bayesfilter.highdim.zhao_cui_austria_sir_bounded_teacher_tf import (  # noqa: E402
    load_austria_t1_t2_bounded_teacher,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (  # noqa: E402
    SIR_OBSERVATION_SHA256,
    generate_sealed_lane_b_dataset,
)
PLAN = Path(
    "docs/plans/"
    "bayesfilter-zhao-cui-bounded-reference-genut-austria-t1-t2-plan-2026-08-06.md"
)
SCHEMA = "bayesfilter.zhao_cui_bounded_reference_genut_austria_t1_t2.v1"
N = 1008
SEEDS = (98601, 98602, 98603)
CONTROLS = {
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
ARMS = (
    ("no_shape", 0, 0, 0.0),
    ("teacher_diagonal", 4, 0, 0.0),
    ("teacher_pairwise_uncapped", 4, 4, 0.0),
    ("teacher_pairwise_cap8", 4, 4, 8.0),
    ("teacher_pairwise_cap2", 4, 4, 2.0),
)


def _safe(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _noise(seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal([N, 18], [seed, 101], dtype=tf.float32),
        tf.random.stateless_normal([2, N, 18], [seed, 102], dtype=tf.float32),
    )


def _make_evaluator(teacher, diagonal_steps: int, pairwise_steps: int, cap: float):
    adapter = parameterized_austria_sir_candidate_adapter(latent_preclip=True)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                tf.ensure_shape(theta, [3]),
                tf.ensure_shape(observations, [2, 9]),
                tf.ensure_shape(initial_noise, [N, 18]),
                tf.ensure_shape(process_noise, [2, N, 18]),
                tf.ensure_shape(design, [N, 18]),
                epsilon=CONTROLS["epsilon"],
                sinkhorn_steps=CONTROLS["sinkhorn_steps"],
                balance_steps=CONTROLS["balance_steps"],
                ridge=CONTROLS["ridge"],
                transition_before_first_observation=True,
                higher_moment_correction_steps=diagonal_steps,
                higher_moment_strength=CONTROLS["higher_moment_strength"],
                higher_moment_floor=CONTROLS["higher_moment_floor"],
                pairwise_moment_correction_steps=pairwise_steps,
                pairwise_moment_strength=CONTROLS["pairwise_moment_strength"],
                pairwise_moment_floor=CONTROLS["pairwise_moment_floor"],
                pairwise_particle_rms_cap=cap,
                bounded_feature_teacher=teacher,
            )

    return evaluate


def _row(evaluate, theta, observations, design, seed: int) -> dict[str, Any]:
    initial, process = _noise(seed)
    started = time.perf_counter()
    value, score, diagnostics = evaluate(
        theta, observations, initial, process, design
    )
    finite = bool(diagnostics["program_valid"].numpy()) and bool(
        tf.math.is_finite(value).numpy()
    ) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    return {
        "seed": seed,
        "value": float(value.numpy()) if finite else None,
        "score": [float(item) for item in score.numpy()] if finite else None,
        "finite": finite,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "device": value.device,
        "wall_time_seconds": time.perf_counter() - started,
        "value_increments": _safe(diagnostics["value_increments"]),
        "score_increments": _safe(diagnostics["score_increments"]),
        "maximum_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "maximum_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "maximum_column_residual": float(diagnostics["max_col_residual"].numpy()),
        "maximum_shape_displacement": float(
            diagnostics["maximum_normalized_shape_displacement"].numpy()
        ),
        "diagonal_shape_objective": float(
            diagnostics["mean_normalized_shape_residual_objective"].numpy()
        ),
        "pairwise_shape_objective": float(
            diagnostics["mean_normalized_pairwise_shape_residual_objective"].numpy()
        ),
        "maximum_pairwise_pre_cap_particle_rms": float(
            diagnostics["maximum_pairwise_pre_cap_particle_rms"].numpy()
        ),
        "maximum_pairwise_post_cap_particle_rms": float(
            diagnostics["maximum_pairwise_post_cap_particle_rms"].numpy()
        ),
        "minimum_pairwise_particle_cap_scale": float(
            diagnostics["minimum_pairwise_particle_cap_scale"].numpy()
        ),
    }


def _finite_difference(evaluate, theta, observations, design, seed: int):
    initial, process = _noise(seed)
    step = tf.constant(1.0e-3, tf.float32)
    origin = evaluate(theta, observations, initial, process, design)
    rows = []
    for parameter in range(3):
        direction = tf.one_hot(parameter, 3, dtype=tf.float32)
        plus = evaluate(
            theta + step * direction, observations, initial, process, design
        )[0]
        minus = evaluate(
            theta - step * direction, observations, initial, process, design
        )[0]
        observed = (plus - minus) / (2.0 * step)
        expected = origin[1][parameter]
        absolute = tf.abs(observed - expected)
        normalized = absolute / tf.maximum(tf.abs(expected), 1.0)
        rows.append(
            {
                "parameter": parameter,
                "manual_score": float(expected.numpy()),
                "finite_difference": float(observed.numpy()),
                "absolute_residual": float(absolute.numpy()),
                "normalized_residual": float(normalized.numpy()),
            }
        )
    return rows


def run(teacher_dir: Path, output_root: Path, *, smoke_only: bool) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("bounded teacher diagnostic requires a logical GPU")
    teacher, teacher_manifest = load_austria_t1_t2_bounded_teacher(teacher_dir)
    _states, observations64, _all = generate_sealed_lane_b_dataset()
    observations = tf.cast(observations64[:2], tf.float32)
    theta = tf.zeros([3], tf.float32)
    design = cubature_design(dim=18, num_particles=N)
    arms = []
    selected_seeds = SEEDS[:1] if smoke_only else SEEDS
    for arm_id, diagonal_steps, pairwise_steps, cap in ARMS:
        evaluate = _make_evaluator(teacher, diagonal_steps, pairwise_steps, cap)
        rows = [
            _row(evaluate, theta, observations, design, seed)
            for seed in selected_seeds
        ]
        fd = None
        if arm_id in ("teacher_pairwise_uncapped", "teacher_pairwise_cap2"):
            fd = _finite_difference(
                evaluate, theta, observations, design, selected_seeds[0]
            )
        arms.append(
            {
                "arm_id": arm_id,
                "diagonal_steps": diagonal_steps,
                "pairwise_steps": pairwise_steps,
                "cap": None if cap == 0.0 else cap,
                "rows": rows,
                "finite_difference": fd,
            }
        )
    fd_rows = [row for arm in arms for row in (arm["finite_difference"] or [])]
    hard_valid = all(
        row["finite"] and "GPU" in row["device"].upper()
        for arm in arms
        for row in arm["rows"]
    ) and all(
        row["absolute_residual"] <= 0.08
        and row["normalized_residual"] <= 0.03
        for row in fd_rows
    )
    payload = {
        "schema_version": SCHEMA,
        "status": (
            "PASS_GPU_XLA_SMOKE"
            if smoke_only and hard_valid
            else "COMPLETE_BOUNDED_ZHAO_CUI_T1_T2_DIAGNOSTIC"
            if hard_valid
            else "INVALID_BOUNDED_ZHAO_CUI_T1_T2_DIAGNOSTIC"
        ),
        "hard_valid": hard_valid,
        "smoke_only": smoke_only,
        "plan": PLAN.as_posix(),
        "target": {
            "model_id": "austria_sir_lane_b_latent_preclip_T2",
            "horizon": 2,
            "particle_count": N,
            "source_observation_sha256": SIR_OBSERVATION_SHA256,
            "teacher_owner": "independent_sampled_zhao_cui_tt_marginal",
            "empirical_genut_particle_target_used": False,
        },
        "teacher_manifest": teacher_manifest,
        "controls": CONTROLS,
        "seeds": list(selected_seeds),
        "arms": arms,
        "configuration": {
            "dtype": "float32",
            "tf32": False,
            "jit_compile": True,
            "gpu_memory_growth": True,
        },
        "device": {
            "logical_devices": [device.name for device in logical],
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(MEMORY_POLICY),
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
                    Path("bayesfilter/highdim/cubature_genut_filter.py"),
                    Path("bayesfilter/highdim/higher_moment_contract_e.py"),
                    Path("bayesfilter/highdim/zhao_cui_austria_sir_bounded_teacher_tf.py"),
                    Path(__file__).relative_to(ROOT),
                )
            },
        },
        "inference_status": {
            "hard_veto_screen": "pass" if hard_valid else "fail",
            "statistically_supported_ranking": False,
            "descriptive_only": [
                "value and score differences",
                "moment residuals",
                "cap activity",
                "three-seed ranges",
            ],
            "default_readiness": False,
            "next_evidence_needed": "validated T3+ teacher sequence and larger independent teacher sample",
        },
        "nonclaims": [
            "sampled bounded-feature moments are not exact physical moments",
            "T2 does not establish T20, HMC, NeuTra, posterior, or default readiness",
            "three seeds do not support a statistical arm ranking",
            "the assembled route is extension_or_invention",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "hard_valid": payload["hard_valid"],
                "output": output_root.as_posix(),
                "wall_time_seconds": payload["wall_time_seconds"],
            }
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run(
        args.teacher_dir.resolve(),
        args.output_root.resolve(),
        smoke_only=args.smoke_only,
    )


if __name__ == "__main__":
    main()
