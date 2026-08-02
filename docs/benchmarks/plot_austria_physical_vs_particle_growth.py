#!/usr/bin/env python3
"""Plot deterministic Austria transition growth against full-filter growth.

This is a diagnostic/reporting lane.  The physical curve is the mean of eight
directional JVP growth probes through the deterministic nominal Austria RK4
transition.  The full-filter curves are pooled from the immutable attempt-08
validation artifact.  They live in different tangent spaces and are compared
for timing/mechanism, not as a common operator-norm estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = ROOT / (
    "docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260731/"
    "attempt08/result.json"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260731/"
    "derived_physical_vs_full_20260801"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _nominal_physical_growth(*, probe_count: int = 8) -> dict[str, list[float]]:
    """Compute directional log growth through the deterministic RK4 transition."""

    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    model = zhao_cui_sir_austria_model()
    state = tf.identity(model.initial_mean)
    probes = tf.random.stateless_normal(
        [probe_count, model.state_dim()], seed=[98201, 1701], dtype=tf.float64
    )
    state_rows: list[dict[str, float]] = []
    per_probe: list[list[float]] = [[] for _ in range(probe_count)]
    for time_index in range(20):
        old_norm = tf.sqrt(tf.reduce_sum(tf.square(probes), axis=1))
        next_state = model.transition_mean(state[None, :])[0]
        next_probes = []
        for probe in tf.unstack(probes, axis=0):
            with tf.autodiff.ForwardAccumulator(state, probe) as accumulator:
                next_state_for_jvp = model.transition_mean(state[None, :])[0]
            next_probes.append(accumulator.jvp(next_state_for_jvp))
        next_probes = tf.stack(next_probes, axis=0)
        next_norm = tf.sqrt(tf.reduce_sum(tf.square(next_probes), axis=1))
        log_growth = tf.math.log(next_norm / old_norm)
        values = [float(value) for value in log_growth.numpy()]
        for probe_index, value in enumerate(values):
            per_probe[probe_index].append(value)
        state_rows.append(
            {
                "time_index": time_index + 1,
                "mean_log_growth": sum(values) / len(values),
                "min_log_growth": min(values),
                "max_log_growth": max(values),
            }
        )
        probes = next_probes / next_norm[:, None]
        state = next_state
    return {
        "rows": state_rows,
        "probe_log_growth": per_probe,
    }


def _full_filter_rows(payload: dict[str, Any], arm_id: str) -> list[dict[str, float]]:
    arm = next(item for item in payload["arms"] if item["arm_id"] == arm_id)
    rows = arm["rows"]
    horizon = len(rows[0]["finite_time_directional_growth"]["per_step_log_growth"])
    result = []
    for time_index in range(horizon):
        values = [
            float(value)
            for row in rows
            for value in row["finite_time_directional_growth"]["per_step_log_growth"][time_index]
        ]
        result.append(
            {
                "time_index": time_index + 1,
                "mean_log_growth": sum(values) / len(values),
                "min_log_growth": min(values),
                "max_log_growth": max(values),
            }
        )
    return result


def _write_plot(
    output: Path,
    physical: list[dict[str, float]],
    diagonal: list[dict[str, float]],
    pairwise: list[dict[str, float]],
    *,
    particle_count: int,
) -> None:
    time = [row["time_index"] for row in physical]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True, constrained_layout=True)
    colors = {"physical": "#1d4ed8", "diagonal": "#b91c1c", "pairwise": "#047857"}
    series = {
        "physical": physical,
        "diagonal": diagonal,
        "pairwise": pairwise,
    }
    labels = {
        "physical": "Austria deterministic RK4 transition (18-D state)",
        "diagonal": f"Full particle filter, diagonal correction (N={particle_count})",
        "pairwise": f"Full particle filter, pairwise correction (N={particle_count})",
    }
    for key, rows in series.items():
        mean = [row["mean_log_growth"] for row in rows]
        low = [row["min_log_growth"] for row in rows]
        high = [row["max_log_growth"] for row in rows]
        axes[0].plot(time, mean, marker="o", linewidth=2, label=labels[key], color=colors[key])
        axes[0].fill_between(time, low, high, color=colors[key], alpha=0.10)
        cumulative = []
        total = 0.0
        for value in mean:
            total += value
            cumulative.append(total)
        axes[1].plot(time, cumulative, marker="o", linewidth=2, label=labels[key], color=colors[key])
    axes[0].axhline(0.0, color="#374151", linewidth=1, linestyle="--")
    axes[1].axhline(0.0, color="#374151", linewidth=1, linestyle="--")
    axes[0].set_ylabel("per-step log growth")
    axes[1].set_ylabel("cumulative log growth")
    axes[1].set_xlabel("transition/filter step")
    axes[0].set_title(
        f"Austria: physical transition versus full particle-filter tangent growth (N={particle_count})"
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    fig.text(
        0.01,
        0.005,
        "Shading is the probe range. The physical and full-filter curves use different tangent spaces; "
        "compare timing and sign, not absolute operator norms.",
        fontsize=8,
    )
    fig.savefig(output / "austria_physical_vs_full_particle_growth.png", dpi=180)
    plt.close(fig)


def run(input_path: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    physical = _nominal_physical_growth()
    diagonal = _full_filter_rows(payload, "austria_diagonal")
    pairwise = _full_filter_rows(payload, "austria_pairwise")
    if len(physical["rows"]) != len(diagonal) or len(diagonal) != len(pairwise):
        raise ValueError("physical and full-filter horizons do not match")
    output.mkdir(parents=True, exist_ok=False)
    particle_count = int(payload["configuration"]["particles"])
    _write_plot(
        output,
        physical["rows"],
        diagonal,
        pairwise,
        particle_count=particle_count,
    )
    with (output / "growth_by_step.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_index",
                "physical_mean",
                "physical_min",
                "physical_max",
                "diagonal_mean",
                "diagonal_min",
                "diagonal_max",
                "pairwise_mean",
                "pairwise_min",
                "pairwise_max",
            ],
        )
        writer.writeheader()
        for p, d, q in zip(physical["rows"], diagonal, pairwise):
            writer.writerow(
                {
                    "time_index": p["time_index"],
                    "physical_mean": p["mean_log_growth"],
                    "physical_min": p["min_log_growth"],
                    "physical_max": p["max_log_growth"],
                    "diagonal_mean": d["mean_log_growth"],
                    "diagonal_min": d["min_log_growth"],
                    "diagonal_max": d["max_log_growth"],
                    "pairwise_mean": q["mean_log_growth"],
                    "pairwise_min": q["min_log_growth"],
                    "pairwise_max": q["max_log_growth"],
                }
            )
    physical_mean = [row["mean_log_growth"] for row in physical["rows"]]
    diagonal_mean = [row["mean_log_growth"] for row in diagonal]
    pairwise_mean = [row["mean_log_growth"] for row in pairwise]
    markdown = [
        "# Austria Physical Versus Full-Filter Growth",
        "",
        "This derived diagnostic compares the deterministic nominal RK4 transition with the full particle-filter tangent.",
        f"The physical curve is 18-D; the full-filter curves are N={particle_count} x 18 and are pooled over three seeds and eight probes.",
        "",
        "| Step | Physical transition | Full diagonal | Full pairwise |",
        "|---:|---:|---:|---:|",
    ]
    for index, (p, d, q) in enumerate(zip(physical_mean, diagonal_mean, pairwise_mean), start=1):
        markdown.append(f"| {index} | {p:+.6f} | {d:+.6f} | {q:+.6f} |")
    markdown.extend(
        [
            "",
            f"- Physical cumulative log growth: `{sum(physical_mean):+.6f}` (factor `{math.exp(sum(physical_mean)):.3g}`).",
            f"- Full diagonal cumulative log growth: `{sum(diagonal_mean):+.6f}` (factor `{math.exp(sum(diagonal_mean)):.3g}`).",
            f"- Full pairwise cumulative log growth: `{sum(pairwise_mean):+.6f}` (factor `{math.exp(sum(pairwise_mean)):.3g}`).",
            "- Interpretation: the physical transition is strongly expanding early and becomes contracting later; the full particle map remains near-neutral or expansive at several later steps, so filtering operations add amplification beyond the physical transition.",
            "- Nonclaim: this is not a common-dimensional operator-norm comparison and does not identify a causal percentage for OT/reset stages.",
        ]
    )
    (output / "result.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    manifest = {
        "schema": "bayesfilter.austria_physical_vs_full_growth_plot.v1",
        "input_result": str(input_path.relative_to(ROOT)),
        "input_result_sha256": _sha256(input_path),
        "output": str(output.relative_to(ROOT)),
        "git_commit": _git_commit(),
        "host": platform.node(),
        "execution": {
            "cpu_only": True,
            "cuda_visible_devices": "-1",
            "physical_probe_count": 8,
            "physical_horizon": 20,
            "full_filter_source": input_path.name,
        },
        "artifacts": {
            "plot": "austria_physical_vs_full_particle_growth.png",
            "csv": "growth_by_step.csv",
            "markdown": "result.md",
        },
        "nonclaims": [
            "not a common-dimensional operator-norm comparison",
            "not a causal attribution percentage for filter stages",
            "not an asymptotic Lyapunov estimate",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
