"""16-seed replication of actual-SV cross-family raw-y gaps (dense routes).

Follow-up to the three-route simulation benchmark: replicates the dense
Gaussian-mixture / Kalman vs exact-dense comparison over 16 independent
simulated actual-SV paths per dimension to decide whether the dim-2 gap seen
in attempt01 is path-level Monte Carlo scatter or a systematic effect.

TT routes are excluded on purpose: they passed their own same-target checks
and are not the quantity under question.

Plan: docs/plans/bayesfilter-actual-sv-cross-family-gap-16-seed-replication-plan-2026-08-14.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

import bayesfilter.highdim as highdim
from docs.benchmarks.benchmark_actual_sv_three_route_simulation import (
    _exact_log_square_jacobian_log_abs_det,
    _fit_log_chi_square_mixture,
    _observations,
    _offset_log_square_jacobian_log_abs_det,
    _panel_mixture_kalman_log_likelihood,
    _physical_parameters,
)

PLAN = (
    "docs/plans/"
    "bayesfilter-actual-sv-cross-family-gap-16-seed-replication-plan-2026-08-14.md"
)

NONCLAIMS = (
    "fixture-distribution intervals only; no approximation-family ranking",
    "no HMC, posterior-correctness, or default-change claims",
    "KSC/Kalman rows target the offset log-square mixture density, corrected to raw-y",
    "not a production timing benchmark",
)

# Two-sided 97.5% Student-t critical values by degrees of freedom.
_T_CRITICAL = {15: 2.131449545559323}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dims", default="1,2,3")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=83120)
    parser.add_argument("--seed-stride", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--dense-order", type=int, default=401)
    parser.add_argument("--dense-radius", type=float, default=8.0)
    parser.add_argument("--ksc-transform-offset", type=float, default=1e-8)
    parser.add_argument("--mixture-components", default="7,14,28")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def _dense_ksc_reference(
    observations: tf.Tensor,
    *,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    mixture: highdim.SVLogChiSquareGaussianMixture,
    order: int,
    radius: float,
    transform_offset: float,
) -> tf.Tensor:
    total = tf.constant(0.0, tf.float64)
    for axis in range(int(observations.shape[1])):
        model = highdim.StochasticVolatilitySSM(sigma=sigma[axis])
        theta_axis = model.unconstrained_from_physical(gamma=gamma[axis], beta=beta[axis])
        total = total + highdim.scalar_sv_mixture_dense_reference(
            model,
            theta_axis,
            observations[:, axis : axis + 1],
            mixture=mixture,
            order=order,
            radius=radius,
            transform_offset=transform_offset,
        ).log_likelihood
    return total


def _summary(values: list[float]) -> dict[str, Any]:
    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((v - mean) ** 2 for v in values) / (count - 1)
        sd = math.sqrt(variance)
        critical = _T_CRITICAL.get(count - 1)
        half_width = critical * sd / math.sqrt(count) if critical is not None else None
    else:
        sd = 0.0
        half_width = None
    return {
        "count": count,
        "mean": mean,
        "sd": sd,
        "t95_half_width": half_width,
        "t95_interval": (
            [mean - half_width, mean + half_width] if half_width is not None else None
        ),
        "min": min(values),
        "max": max(values),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = _parse_args()
    started = time.time()
    dims = [int(part) for part in args.dims.split(",") if part.strip()]
    component_counts = [
        int(part) for part in args.mixture_components.split(",") if part.strip()
    ]
    seed_bases = [args.seed_base + args.seed_stride * k for k in range(args.seeds)]

    ksc7 = highdim.ksc_1998_log_chi_square_mixture()
    fitted: list[tuple[int, highdim.SVLogChiSquareGaussianMixture]] = []
    fit_rows: list[dict[str, Any]] = []
    for count in component_counts:
        mixture, diagnostics = _fit_log_chi_square_mixture(count)
        fitted.append((count, mixture))
        fit_rows.append(diagnostics)

    per_path_rows: list[dict[str, Any]] = []
    for dim in dims:
        gamma, beta, sigma = _physical_parameters(dim)
        for seed_index, seed_base in enumerate(seed_bases):
            observations = _observations(dim, seed_base=seed_base, horizon=args.horizon)
            exact_dense = highdim.exact_transformed_sv_independent_panel_dense_reference(
                observations,
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                order=args.dense_order,
                radius=args.dense_radius,
            ).log_likelihood
            exact_raw = float(
                (exact_dense + _exact_log_square_jacobian_log_abs_det(observations)).numpy()
            )
            offset_correction = _offset_log_square_jacobian_log_abs_det(
                observations, args.ksc_transform_offset
            )
            ksc_dense = _dense_ksc_reference(
                observations,
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                mixture=ksc7,
                order=args.dense_order,
                radius=args.dense_radius,
                transform_offset=args.ksc_transform_offset,
            )
            ksc_raw = float((ksc_dense + offset_correction).numpy())
            kalman_raw: dict[str, float] = {}
            for count, mixture in fitted:
                value = _panel_mixture_kalman_log_likelihood(
                    observations,
                    gamma=gamma,
                    beta=beta,
                    sigma=sigma,
                    mixture=mixture,
                    transform_offset=args.ksc_transform_offset,
                )
                kalman_raw[f"fitted_{count}"] = float((value + offset_correction).numpy())
            values = [exact_raw, ksc_raw, *kalman_raw.values()]
            if not all(math.isfinite(v) for v in values):
                raise AssertionError(
                    f"non-finite value at dim={dim} seed_base={seed_base}: {values}"
                )
            per_path_rows.append(
                {
                    "dim": dim,
                    "seed_index": seed_index,
                    "seed_base": seed_base,
                    "raw_y_exact_dense": exact_raw,
                    "raw_y_ksc7_dense": ksc_raw,
                    "raw_y_kalman": kalman_raw,
                    "diff_ksc7_minus_exact": ksc_raw - exact_raw,
                    "diffs_kalman_minus_exact": {
                        key: value - exact_raw for key, value in kalman_raw.items()
                    },
                }
            )

    summaries: list[dict[str, Any]] = []
    for dim in dims:
        rows = [row for row in per_path_rows if row["dim"] == dim]
        entry: dict[str, Any] = {"dim": dim}
        entry["ksc7_minus_exact"] = _summary(
            [row["diff_ksc7_minus_exact"] for row in rows]
        )
        for count, _mixture in fitted:
            key = f"fitted_{count}"
            entry[f"kalman_{key}_minus_exact"] = _summary(
                [row["diffs_kalman_minus_exact"][key] for row in rows]
            )
        original = next(row for row in rows if row["seed_index"] == 0)
        finest = f"fitted_{fitted[-1][0]}"
        finest_summary = entry[f"kalman_{finest}_minus_exact"]
        original_value = original["diffs_kalman_minus_exact"][finest]
        entry["original_seed_finest_diff"] = original_value
        entry["original_within_seed_range"] = bool(
            finest_summary["min"] <= original_value <= finest_summary["max"]
        )
        summaries.append(entry)

    result: dict[str, Any] = {
        "schema_version": "actual_sv_cross_family_gap_seed_sweep.v1",
        "plan": PLAN,
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_commit(),
        "dims": dims,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "seed_bases": seed_bases,
        "dense_order": args.dense_order,
        "dense_radius": args.dense_radius,
        "ksc_transform_offset": args.ksc_transform_offset,
        "mixture_component_counts": component_counts,
        "mixture_fit_rows": fit_rows,
        "per_path_rows": per_path_rows,
        "summaries": summaries,
        "wall_time_seconds": time.time() - started,
        "nonclaims": list(NONCLAIMS),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.markdown_output is not None:
        lines = [
            "# Actual-SV cross-family gap 16-seed sweep",
            "",
            f"- JSON artifact: `{output_path}`",
            f"- Plan: `{PLAN}`",
            "",
        ]
        for entry in summaries:
            lines.append(f"## dim {entry['dim']}")
            lines.append("")
            for key, summary in entry.items():
                if not isinstance(summary, dict):
                    continue
                interval = summary["t95_interval"]
                interval_text = (
                    f"[{interval[0]:.5g}, {interval[1]:.5g}]" if interval else "n/a"
                )
                lines.append(
                    f"- {key}: mean={summary['mean']:.5g}, sd={summary['sd']:.5g}, "
                    f"t95={interval_text}, range=[{summary['min']:.5g}, {summary['max']:.5g}]"
                )
            lines.append(
                f"- original seed diff (finest): {entry['original_seed_finest_diff']:.5g}, "
                f"within seed range: {entry['original_within_seed_range']}"
            )
            lines.append("")
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "wall_time_seconds": result["wall_time_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
