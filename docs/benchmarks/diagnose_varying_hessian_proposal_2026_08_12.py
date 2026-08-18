#!/usr/bin/env python3
"""CPU-only importance-proposal diagnostic for the smooth ridge target."""

from __future__ import annotations

import argparse
import hashlib
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

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md"
)
SOURCE = Path("/home/ubuntu/python/dsge_hmc/src/dsge_hmc/benchmarks/nk_like_mild.py")
DEFAULT_CONSTANTS = Path(
    "/home/ubuntu/python/dsge_hmc/results/neutra/gate3/"
    "nk_strong_smooth_bridge_20260604/frozen_constants/"
    "strong_smooth_from_seed42_affine_lift.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=65_536)
    parser.add_argument(
        "--target-name",
        choices=("nk_like_mild_smooth", "nk_like_strong_smooth"),
        default="nk_like_strong_smooth",
    )
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _draw_weighted_rows(
    spec: Any, proposal: Mapping[str, Any], count: int, seed: tuple[int, int]
) -> tuple[Any, Any]:
    from bayesfilter.inference.neutra_varying_hessian_target import (
        varying_hessian_log_prob_and_score_batch,
    )
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob,
        sample_gaussian_mixture,
    )

    rows, _labels = sample_gaussian_mixture(
        count,
        proposal["probabilities"],
        proposal["means"],
        proposal["covariances"],
        seed=seed,
    )
    target, score = varying_hessian_log_prob_and_score_batch(spec, rows)
    proposal_log_prob = gaussian_mixture_log_prob(
        rows, proposal["probabilities"], proposal["means"], proposal["covariances"]
    )
    return rows, target - proposal_log_prob


def _candidate(tf: Any, spec: Any, proposal: Mapping[str, Any], count: int, seed: tuple[int, int]) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_varying_hessian_target import (
        varying_hessian_log_prob_and_score_batch,
    )

    rows, log_weights = _draw_weighted_rows(spec, proposal, count, seed)
    target, score = varying_hessian_log_prob_and_score_batch(spec, rows)
    weights = tf.nn.softmax(log_weights)
    local = tf.transpose(
        tf.linalg.triangular_solve(
            tf.constant(spec.lchol, tf.float64),
            tf.transpose(rows - tf.constant(spec.mu, tf.float64)),
            lower=True,
        )
    )
    norm = tf.sqrt(tf.reduce_sum(tf.square(local), axis=1))
    q99_index = int(0.99 * (int(count) - 1))
    return {
        "proposal_identity": proposal["identity"],
        "effective_sample_size": tf.math.reciprocal(tf.reduce_sum(tf.square(weights))),
        "effective_sample_size_fraction": tf.math.reciprocal(
            tf.reduce_sum(tf.square(weights))
        ) / tf.cast(count, tf.float64),
        "maximum_normalized_weight": tf.reduce_max(weights),
        "target_score_all_finite": tf.reduce_all(tf.math.is_finite(score)),
        "target_value_all_finite": tf.reduce_all(tf.math.is_finite(target)),
        "local_norm_mean": tf.reduce_mean(norm),
        "local_norm_q99": tf.sort(norm)[q99_index],
    }


def main() -> int:
    args = _parse_args()
    if int(args.sample_count) < 1024:
        raise ValueError("sample_count must be at least 1024")
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    constants = args.constants.resolve()
    if not SOURCE.is_file() or not constants.is_file():
        raise FileNotFoundError("source target or frozen constants are unavailable")
    args.output_root.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.monotonic()
    import tensorflow as tf

    from bayesfilter.inference.neutra_varying_hessian_target import (
        affine_ridge_tangent_mixture_proposal,
        affine_scale_mixture_proposal,
        fit_defensive_branch_mixture_proposal,
        fit_reflected_positive_branch_mixture_proposal,
        load_varying_hessian_target_spec,
        reflect_first_local_coordinate,
        varying_hessian_log_prob_and_score_batch,
        VaryingHessianTargetError,
    )

    spec = load_varying_hessian_target_spec(
        constants, expected_name=args.target_name
    )
    candidates = {
        "axis_scale": affine_scale_mixture_proposal(spec),
        "ridge_tangent": affine_ridge_tangent_mixture_proposal(spec),
        "ridge_tangent_broad": affine_ridge_tangent_mixture_proposal(
            spec,
            radii=(0.0, 5.0, 5.0, 18.0, 18.0, 48.0, 48.0, 96.0, 96.0),
            signs=(0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
            weak_scales=(3.0, 7.0, 7.0, 22.0, 22.0, 48.0, 48.0, 80.0, 80.0),
            stiff_scales=(1.5, 1.5, 1.5, 1.3, 1.3, 1.0, 1.0, 0.8, 0.8),
            probabilities=(0.04, 0.08, 0.08, 0.16, 0.16, 0.18, 0.18, 0.06, 0.06),
        ),
    }
    pilot_rows, pilot_log_weights = _draw_weighted_rows(
        spec,
        candidates["ridge_tangent"],
        max(262_144, 4 * int(args.sample_count)),
        (20260812, 15190),
    )
    candidate_failures = {}
    try:
        candidates["pilot_branch_defensive"] = fit_defensive_branch_mixture_proposal(
            spec,
            pilot_rows,
            pilot_log_weights,
            defensive_proposal=candidates["ridge_tangent"],
            defensive_weight=0.05,
        )
    except VaryingHessianTargetError as error:
        candidate_failures["pilot_branch_defensive"] = str(error)
    reflected_rows = reflect_first_local_coordinate(spec, pilot_rows)
    pilot_value, pilot_score = varying_hessian_log_prob_and_score_batch(spec, pilot_rows)
    reflected_value, reflected_score = varying_hessian_log_prob_and_score_batch(spec, reflected_rows)
    lchol = tf.constant(spec.lchol, tf.float64)
    pilot_local_score = tf.linalg.matvec(lchol, pilot_score, transpose_a=True)
    reflected_local_score = tf.linalg.matvec(lchol, reflected_score, transpose_a=True)
    expected_reflected_local_score = tf.concat(
        (-pilot_local_score[:, :1], pilot_local_score[:, 1:]), axis=1
    )
    symmetry_value_difference = tf.abs(pilot_value - reflected_value)
    symmetry_score_difference = tf.abs(
        reflected_local_score - expected_reflected_local_score
    )
    symmetry_value_scale = tf.maximum(tf.abs(pilot_value), tf.abs(reflected_value))
    symmetry_score_scale = tf.maximum(
        tf.abs(reflected_local_score), tf.abs(expected_reflected_local_score)
    )
    symmetry_value_max_abs_error = tf.reduce_max(symmetry_value_difference)
    symmetry_score_max_abs_error = tf.reduce_max(symmetry_score_difference)
    symmetry_value_max_relative_error = tf.reduce_max(
        symmetry_value_difference / (tf.constant(1.0, tf.float64) + symmetry_value_scale)
    )
    symmetry_score_max_relative_error = tf.reduce_max(
        symmetry_score_difference / (tf.constant(1.0, tf.float64) + symmetry_score_scale)
    )
    symmetry_checked = bool(
        bool(tf.reduce_all(
            symmetry_value_difference <= tf.constant(1.0e-9, tf.float64)
            + tf.constant(1.0e-10, tf.float64) * symmetry_value_scale
        ).numpy())
        and bool(tf.reduce_all(
            symmetry_score_difference <= tf.constant(1.0e-10, tf.float64)
            + tf.constant(1.0e-10, tf.float64) * symmetry_score_scale
        ).numpy())
    )
    if not symmetry_checked:
        raise RuntimeError("source-bound local reflection symmetry check failed")
    candidates["pilot_positive_reflected_defensive"] = (
        fit_reflected_positive_branch_mixture_proposal(
            spec,
            pilot_rows,
            pilot_log_weights,
            defensive_proposal=candidates["ridge_tangent"],
            defensive_weight=0.05,
        )
    )
    rows = {
        name: _candidate(
            tf, spec, proposal, int(args.sample_count), (20260812, 15100 + index)
        )
        for index, (name, proposal) in enumerate(candidates.items())
    }
    selection = max(
        rows,
        key=lambda name: float(rows[name]["effective_sample_size_fraction"].numpy()),
    )
    selected = rows[selection]
    selected_proposal = _ready(candidates[selection])
    selected_proposal_payload = {
        "schema": "bayesfilter.varying_hessian_frozen_replay_proposal.v1",
        "target_constants_sha256": spec.constants_sha256,
        "target_name": spec.name,
        "proposal": selected_proposal,
    }
    selected_proposal_payload["proposal_hash"] = _stable_hash(selected_proposal_payload)
    passed = bool(
        selected["target_score_all_finite"].numpy()
        and selected["target_value_all_finite"].numpy()
        and float(selected["effective_sample_size_fraction"].numpy()) >= 0.05
        and float(selected["maximum_normalized_weight"].numpy()) <= 0.01
    )
    manifest = {
        "schema": "bayesfilter.varying_hessian_proposal_diagnostic_manifest.v1",
        "plan": PLAN.as_posix(),
        "command": " ".join(sys.argv),
        "cpu_only": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "source_target_path": SOURCE.as_posix(),
        "source_target_sha256": _sha256(SOURCE),
        "constants_path": constants.as_posix(),
        "constants_sha256": _sha256(constants),
        "target": spec.manifest_payload(),
        "sample_count": int(args.sample_count),
        "pilot_sample_count": max(262_144, 4 * int(args.sample_count)),
        "pilot_proposal_identity": candidates["ridge_tangent"]["identity"],
        "pilot_validation_disjoint": True,
        "selected_proposal_path": "selected_proposal.json",
        "selected_proposal_hash": selected_proposal_payload["proposal_hash"],
        "local_reflection_symmetry": {
            "transform": "x0 -> -x0; x[1:] unchanged",
            "value_max_abs_error": symmetry_value_max_abs_error,
            "value_max_relative_error": symmetry_value_max_relative_error,
            "local_score_max_abs_error": symmetry_score_max_abs_error,
            "local_score_max_relative_error": symmetry_score_max_relative_error,
            "comparison": "abs <= 1e-9 + 1e-10*abs(value); score abs <= 1e-10 + 1e-10*abs(score)",
            "checked": symmetry_checked,
        },
        "wall_seconds": time.monotonic() - started,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
    }
    result = {
        "schema": "bayesfilter.varying_hessian_proposal_diagnostic_result.v1",
        "research_question": (
            "Does a full-support replay proposal have enough corrected-weight "
            "support to nominate target-specific weighted NeuTra training?"
        ),
        "candidates": rows,
        "candidate_failures": candidate_failures,
        "selected_candidate": selection,
        "selection_rule": "maximum measured importance ESS fraction",
        "candidate_passed": passed,
        "gates": {
            "finite_target_value_score": bool(
                selected["target_score_all_finite"].numpy()
                and selected["target_value_all_finite"].numpy()
            ),
            "effective_sample_size_fraction_at_least_0p05": float(
                selected["effective_sample_size_fraction"].numpy()
            ) >= 0.05,
            "maximum_weight_at_most_0p01": float(
                selected["maximum_normalized_weight"].numpy()
            ) <= 0.01,
        },
        "diagnostic_role": "proposal_nomination_only_not_posterior_evidence",
        "nonclaims": (
            "importance diagnostics do not establish posterior correctness",
            "component proposal selection does not rank training objectives",
            "no HMC or varying-Hessian transport claim",
        ),
        "manifest": manifest,
    }
    _write(args.output_root / "result.json", result)
    _write(args.output_root / "run_manifest.json", manifest)
    _write(args.output_root / "selected_proposal.json", selected_proposal_payload)
    _write(
        args.output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.varying_hessian_proposal_diagnostic_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in args.output_root.iterdir()
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"passed": passed, "selected": selection, "output_root": args.output_root.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
