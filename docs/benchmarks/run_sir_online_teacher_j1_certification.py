"""Run the bounded J=1 dense-oracle mismatch screen for the SIR score teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.sir_latent_preclip_reference_tf import (  # noqa: E402
    dense_latent_sir_value_and_manual_score,
    prepare_reduced_dense_grids,
    reduced_latent_preclip_sir_model,
)
from bayesfilter.highdim.sir_online_score_teacher_tf import (  # noqa: E402
    PARAMETER_COUNT,
    TEACHER_ID,
    make_online_sir_teacher,
)


PLAN_PATH = Path(
    "docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-sir-remaining-gap-closure-phase2-j1-teacher-result-2026-07-16.md"
)
DTYPE = tf.float64
THETA = (0.03, -0.02, 0.04)
OBSERVATIONS = ((0.15,), (0.1,))
DEFAULT_PARTICLE_COUNTS = (64, 128, 256)
DEFAULT_REPLICATES = 16
DEFAULT_SEED_START = 86200
DENSE_CONFIGURATIONS = ((29, 6.0), (33, 6.0), (33, 7.0))
FAMILY_SIZE = 1 + PARAMETER_COUNT
FROZEN_ORACLE = {
    1: {
        "value": -0.37337136725883546,
        "score": (0.0, 0.0, -0.516089350201728),
        "u_value": 2.9808919776996845e-9,
        "u_score": 1.6652841994257983e-8,
    },
    2: {
        "value": -0.8570589548006784,
        "score": (
            -0.00013728907649588102,
            0.0043448641619948086,
            -1.0261056979707353,
        ),
        "u_value": 8.866663376849715e-7,
        "u_score": 1.323006956610584e-5,
    },
}


class SIRTeacherCertificationError(RuntimeError):
    """Raised when the Phase 2 evidence contract cannot be evaluated."""


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dense_transition_count(observation_count: int) -> int:
    if int(observation_count) < 1:
        raise ValueError("observation_count must be positive")
    return int(observation_count) - 1


def _summary(
    samples: Sequence[float],
    *,
    reference: float,
    reference_diagnostic_uncertainty: float,
    confidence: float = 0.95,
    family_size: int = FAMILY_SIZE,
) -> Mapping[str, Any]:
    values = tuple(float(value) for value in samples)
    if len(values) < 2:
        raise ValueError("at least two independent replicates are required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all interval samples must be finite")
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError("confidence must be in (0,1)")
    if int(family_size) < 1:
        raise ValueError("family_size must be positive")
    uncertainty = float(reference_diagnostic_uncertainty)
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError("reference diagnostic uncertainty must be finite and nonnegative")

    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    alpha = 1.0 - float(confidence)
    critical = float(
        student_t.ppf(1.0 - alpha / (2.0 * int(family_size)), len(values) - 1)
    )
    half_width = critical * standard_error
    mean_difference = mean - float(reference)
    lower = mean_difference - half_width - uncertainty
    upper = mean_difference + half_width + uncertainty
    return {
        "count": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "bonferroni_family_size": int(family_size),
        "student_critical": critical,
        "interval_half_width": half_width,
        "reference": float(reference),
        "reference_diagnostic_uncertainty": uncertainty,
        "mean_minus_reference": mean_difference,
        "expanded_difference_interval_lower": lower,
        "expanded_difference_interval_upper": upper,
        "contains_zero": bool(lower <= 0.0 <= upper),
    }


def _classify_largest_rung(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("at least one particle rung is required")
    largest = max(rows, key=lambda row: int(row["particle_count"]))
    if not bool(largest["all_finite"]):
        return "BLOCK_TEACHER_NONFINITE"
    if float(largest["maximum_backward_row_sum_error"]) > 1.0e-10:
        return "BLOCK_TEACHER_BACKWARD_NORMALIZATION"
    summaries = (largest["value_summary"], *largest["score_summaries"])
    if not all(bool(summary["contains_zero"]) for summary in summaries):
        return "BLOCK_TEACHER_J1_DISAGREEMENT"
    return "NO_TEACHER_J1_DISAGREEMENT_DETECTED_AT_CURRENT_PRECISION"


def _dense_oracle(observation_count: int) -> Mapping[str, Any]:
    transition_count = _dense_transition_count(observation_count)
    model = reduced_latent_preclip_sir_model()
    theta = tf.constant(THETA, DTYPE)
    observations = tf.constant(OBSERVATIONS[:observation_count], DTYPE)
    rows = []
    for order, radius in DENSE_CONFIGURATIONS:
        grids = prepare_reduced_dense_grids(
            model,
            theta,
            time_steps=transition_count,
            order=order,
            radius=radius,
        )
        result = dense_latent_sir_value_and_manual_score(
            model, theta, observations, grids
        )
        rows.append(
            {
                "order": order,
                "radius": radius,
                "value": float(result["objective"].numpy()),
                "score": result["score"].numpy().tolist(),
                "boundary_mass_history": result["boundary_mass_history"].numpy().tolist(),
            }
        )
    order_value_gap = abs(rows[1]["value"] - rows[0]["value"])
    range_value_gap = abs(rows[2]["value"] - rows[1]["value"])
    order_score_gap = float(
        np.max(np.abs(np.asarray(rows[1]["score"]) - np.asarray(rows[0]["score"])))
    )
    range_score_gap = float(
        np.max(np.abs(np.asarray(rows[2]["score"]) - np.asarray(rows[1]["score"])))
    )
    calculated = {
        "value": rows[2]["value"],
        "score": rows[2]["score"],
        "u_value": max(order_value_gap, range_value_gap),
        "u_score": max(order_score_gap, range_score_gap),
    }
    frozen = FROZEN_ORACLE[observation_count]
    if abs(calculated["value"] - frozen["value"]) > 5.0e-12:
        raise SIRTeacherCertificationError("frozen dense value no longer reproduces")
    if np.max(np.abs(np.asarray(calculated["score"]) - np.asarray(frozen["score"]))) > 5.0e-12:
        raise SIRTeacherCertificationError("frozen dense score no longer reproduces")
    if abs(calculated["u_value"] - frozen["u_value"]) > 5.0e-12:
        raise SIRTeacherCertificationError("frozen dense value uncertainty no longer reproduces")
    if abs(calculated["u_score"] - frozen["u_score"]) > 5.0e-12:
        raise SIRTeacherCertificationError("frozen dense score uncertainty no longer reproduces")
    return {
        "observation_count_T": observation_count,
        "dense_helper_time_steps": transition_count,
        "rows": rows,
        "order_value_gap": order_value_gap,
        "range_value_gap": range_value_gap,
        "order_score_gap": order_score_gap,
        "range_score_gap": range_score_gap,
        **calculated,
        "frozen_reproduction_status": "pass",
        "uncertainty_semantics": "max_observed_refinement_difference_not_rigorous_bound",
    }


def _teacher_rung(
    *,
    observation_count: int,
    particle_count: int,
    seeds: tf.Tensor,
    jit_compile: bool,
    oracle: Mapping[str, Any],
) -> Mapping[str, Any]:
    model = reduced_latent_preclip_sir_model()
    observations = tf.constant(OBSERVATIONS[:observation_count], DTYPE)
    teacher = make_online_sir_teacher(
        model,
        observations,
        seeds,
        num_particles=particle_count,
        jit_compile=jit_compile,
    )
    started = time.perf_counter()
    result = teacher(tf.constant(THETA, DTYPE))
    first_call_seconds = time.perf_counter() - started
    started = time.perf_counter()
    replay = teacher(tf.constant(THETA, DTYPE))
    warm_replay_seconds = time.perf_counter() - started

    values = result["log_likelihood"].numpy().tolist()
    scores = result["score"].numpy()
    value_summary = _summary(
        values,
        reference=float(oracle["value"]),
        reference_diagnostic_uncertainty=float(oracle["u_value"]),
    )
    score_summaries = [
        _summary(
            scores[:, index].tolist(),
            reference=float(oracle["score"][index]),
            reference_diagnostic_uncertainty=float(oracle["u_score"]),
        )
        for index in range(PARAMETER_COUNT)
    ]
    tensors = tuple(tf.nest.flatten(result))
    return {
        "observation_count_T": observation_count,
        "dense_helper_time_steps": _dense_transition_count(observation_count),
        "particle_count": particle_count,
        "replicate_count": int(seeds.shape[0]),
        "seeds": seeds.numpy().tolist(),
        "jit_compile": bool(jit_compile),
        "dtype": "float64",
        "first_call_seconds": first_call_seconds,
        "warm_replay_seconds": warm_replay_seconds,
        "warm_replay_exact": bool(
            tf.reduce_all(tf.equal(result["log_likelihood"], replay["log_likelihood"])).numpy()
            and tf.reduce_all(tf.equal(result["score"], replay["score"])).numpy()
        ),
        "output_devices": sorted({tensor.device for tensor in tensors if tf.is_tensor(tensor)}),
        "all_finite": bool(tf.reduce_all(result["finite"]).numpy()),
        "minimum_ess": result["minimum_ess"].numpy().tolist(),
        "maximum_backward_row_sum_error": float(
            tf.reduce_max(result["maximum_backward_row_sum_error"]).numpy()
        ),
        "log_likelihood_samples": values,
        "score_samples": scores.tolist(),
        "value_summary": value_summary,
        "score_summaries": score_summaries,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SIRTeacherCertificationError(
            "Phase 2 is a deliberate CPU-only reference run; set CUDA_VISIBLE_DEVICES=-1"
        )
    particle_counts = tuple(sorted(set(int(value) for value in args.particle_counts)))
    if not particle_counts or particle_counts[0] < 2:
        raise ValueError("particle counts must be at least two")
    if int(args.replicates) < 2:
        raise ValueError("replicates must be at least two")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    seeds = tf.range(
        int(args.seed_start),
        int(args.seed_start) + int(args.replicates),
        dtype=tf.int32,
    )
    horizon_rows = []
    for observation_count in (1, 2):
        oracle = _dense_oracle(observation_count)
        particle_rows = [
            _teacher_rung(
                observation_count=observation_count,
                particle_count=particle_count,
                seeds=seeds,
                jit_compile=bool(args.jit_compile),
                oracle=oracle,
            )
            for particle_count in particle_counts
        ]
        status = _classify_largest_rung(particle_rows)
        horizon_rows.append(
            {
                "observation_count_T": observation_count,
                "dense_helper_time_steps": _dense_transition_count(observation_count),
                "oracle": oracle,
                "particle_rows": particle_rows,
                "status": status,
            }
        )

    hard_vetoes = [
        row["status"] for row in horizon_rows if str(row["status"]).startswith("BLOCK_")
    ]
    terminal_status = (
        "BLOCK_TEACHER_J1_SCREEN"
        if hard_vetoes
        else "NO_TEACHER_J1_DISAGREEMENT_DETECTED_AT_CURRENT_PRECISION"
    )
    wall_time = time.perf_counter() - started
    source_paths = (
        Path("bayesfilter/highdim/sir_online_score_teacher_tf.py"),
        Path("bayesfilter/highdim/sir_latent_preclip_reference_tf.py"),
        Path("bayesfilter/highdim/sir_latent_preclip_tf.py"),
        Path("bayesfilter/highdim/models.py"),
        Path(__file__).resolve().relative_to(ROOT),
    )
    result_payload = {
        "schema": "bayesfilter.sir_online_teacher_j1_mismatch_screen.v1",
        "status": terminal_status,
        "teacher_id": TEACHER_ID,
        "target": {
            "model": "reduced_latent_preclip_sir_J1",
            "theta": list(THETA),
            "observations": [list(row) for row in OBSERVATIONS],
            "horizon_convention": "T_is_observation_count_dense_time_steps_equals_T_minus_1",
        },
        "interval_contract": {
            "confidence": 0.95,
            "method": "Bonferroni_adjusted_Student_interval_on_teacher_minus_dense_reference",
            "family_size_per_horizon_rung": FAMILY_SIZE,
            "reference_expansion": "max_observed_dense_refinement_difference",
            "positive_equivalence_claim_available": False,
        },
        "configuration": {
            "particle_counts": list(particle_counts),
            "replicates": int(args.replicates),
            "seed_start": int(args.seed_start),
            "jit_compile": bool(args.jit_compile),
            "device_policy": "deliberate_cpu_only_reference",
        },
        "horizons": horizon_rows,
        "hard_vetoes": hard_vetoes,
        "decision_table": {
            "decision": terminal_status,
            "primary_criterion_status": (
                "failed_mismatch_screen" if hard_vetoes else "no_disagreement_detected"
            ),
            "veto_diagnostic_status": hard_vetoes or "none",
            "main_uncertainty": (
                "finite-particle Monte Carlo intervals may be too wide to detect material bias"
            ),
            "next_justified_action": (
                "stop and repair teacher before later comparisons"
                if hard_vetoes
                else "proceed to exact-source GPU/XLA engineering certificate"
            ),
            "not_concluded": (
                "teacher unbiasedness, convergence, practical equivalence, LEDH accuracy, "
                "HMC readiness, leaderboard readiness"
            ),
        },
        "inference_status": {
            "hard_veto_screen": hard_vetoes or "none",
            "statistically_supported_ranking": "not_applicable_no_ranking_claim",
            "descriptive_only_differences": "all particle-count trends and interval widths",
            "default_readiness": "not_established",
            "next_evidence_needed": (
                "GPU/XLA and canonical identity before later LEDH-teacher disagreement screens"
            ),
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "interval containment may reflect low power rather than an accurate teacher"
            ),
            "result_that_would_overturn_decision": (
                "a reproduced largest-rung interval excluding the dense reference or a local-score defect"
            ),
            "weakest_evidence": (
                "only J=1 is externally checked and finite particle counts cannot prove convergence"
            ),
        },
        "nonclaims": [
            "not a practical-equivalence certificate",
            "not proof that the teacher is unbiased or converged",
            "not LEDH accuracy evidence",
            "not GPU execution evidence",
            "not HMC or leaderboard readiness",
        ],
    }
    _write_new_json(args.output_root / "result.json", result_payload)
    manifest = {
        "schema": "bayesfilter.sir_online_teacher_j1_run_manifest.v1",
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--short")),
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "scipy_reporting_backend": "scipy.stats.t",
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "cpu_gpu_status": "CPU_ONLY_CUDA_VISIBLE_DEVICES_MINUS_ONE",
        "physical_gpus_visible": [device.name for device in tf.config.list_physical_devices("GPU")],
        "jit_compile": bool(args.jit_compile),
        "dtype": "float64",
        "data_version": "fixed_reduced_J1_observations_0.15_0.1",
        "random_seeds": seeds.numpy().tolist(),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": wall_time,
        "output_root": str(args.output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(args.output_root / "result.json"),
        "phase_result_file": str(RESULT_PATH),
        "source_sha256": {str(path): _sha256(ROOT / path) for path in source_paths},
        "trust_basis": "deliberate_cpu_only_reference_exception",
    }
    _write_new_json(args.output_root / "run_manifest.json", manifest)
    hashes = {
        path.name: _sha256(path)
        for path in sorted(args.output_root.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new_json(
        args.output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.sir_online_teacher_j1_artifact_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--particle-counts", type=int, nargs="+", default=DEFAULT_PARTICLE_COUNTS
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--jit-compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"status": payload["status"], "output_root": str(args.output_root)}))


if __name__ == "__main__":
    main()

