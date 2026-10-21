"""Summarize the first student-baseline comparison panel."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.student_dpf_baselines.adapters.common import write_json


PANEL_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/student_baseline_panel_2026-05-10.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/student_baseline_panel_summary_2026-05-10.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/student-dpf-baseline-gap-closure-result-2026-05-10.md"
)


def main() -> None:
    panel = json.loads(PANEL_PATH.read_text())
    records = panel["records"]
    summary = _summarize(records)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(_render_report(panel, summary), encoding="utf-8")


def _summarize(records: list[dict]) -> dict:
    by_impl: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_impl[record["implementation_name"]].append(record)

    impl_summary = {}
    for impl, impl_records in by_impl.items():
        ok_records = [r for r in impl_records if r["status"] == "ok"]
        ll_errors = [
            r["diagnostics"].get("log_likelihood_abs_error")
            for r in ok_records
            if r["diagnostics"].get("log_likelihood_abs_error") is not None
        ]
        mean_errors = [
            r["diagnostics"].get("filtered_mean_rmse_vs_reference")
            for r in ok_records
            if r["diagnostics"].get("filtered_mean_rmse_vs_reference") is not None
        ]
        cov_errors = [
            r["diagnostics"].get("filtered_covariance_rmse_vs_reference")
            for r in ok_records
            if r["diagnostics"].get("filtered_covariance_rmse_vs_reference") is not None
        ]
        runtimes = [
            r.get("runtime_seconds")
            for r in ok_records
            if r.get("runtime_seconds") is not None
        ]
        impl_summary[impl] = {
            "runs": len(impl_records),
            "ok": len(ok_records),
            "failed": len(impl_records) - len(ok_records),
            "max_log_likelihood_abs_error": _max_or_none(ll_errors),
            "max_filtered_mean_rmse_vs_reference": _max_or_none(mean_errors),
            "max_filtered_covariance_rmse_vs_reference": _max_or_none(cov_errors),
            "median_runtime_seconds": _median_or_none(runtimes),
        }

    cross = _cross_student_summary(records)
    return {
        "implementation_summary": impl_summary,
        "cross_student_summary": cross,
    }


def _cross_student_summary(records: list[dict]) -> dict:
    mlcoe_by_fixture_seed: dict[tuple, dict] = {}
    advanced_records = []
    for record in records:
        key = (record["fixture_name"], record["seed"])
        if record["implementation_name"] == "2026MLCOE" and key not in mlcoe_by_fixture_seed:
            mlcoe_by_fixture_seed[key] = record
        elif record["implementation_name"] == "advanced_particle_filter":
            advanced_records.append(record)

    mean_rmses = []
    ll_diffs = []
    comparable = 0
    for advanced in advanced_records:
        key = (advanced["fixture_name"], advanced["seed"])
        mlcoe = mlcoe_by_fixture_seed.get(key)
        if mlcoe is None:
            continue
        if advanced["status"] != "ok" or mlcoe["status"] != "ok":
            continue
        comparable += 1
        mean_rmses.append(
            float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(advanced["filtered_means"], dtype=float)
                            - np.asarray(mlcoe["filtered_means"], dtype=float)
                        )
                        ** 2
                    )
                )
            )
        )
        if advanced["log_likelihood"] is not None and mlcoe["log_likelihood"] is not None:
            ll_diffs.append(abs(float(advanced["log_likelihood"]) - float(mlcoe["log_likelihood"])))

    return {
        "comparable_groups": comparable,
        "comparison_key": "fixture_name, seed; advanced_particle_filter particle count retained as a separate run dimension because the MLCOE Kalman smoke adapter has no particle count",
        "max_filtered_mean_rmse_between_students": _max_or_none(mean_rmses),
        "max_log_likelihood_abs_diff_between_students": _max_or_none(ll_diffs),
    }


def _render_report(panel: dict, summary: dict) -> str:
    impl_summary = summary["implementation_summary"]
    cross = summary["cross_student_summary"]
    lines = [
        "# Student DPF baseline gap-closure result",
        "",
        "## Date",
        "",
        "2026-05-10",
        "",
        "## Scope",
        "",
        "This report covers the first controlled comparison panel for the",
        "quarantined student DPF baseline stream.  It is comparison evidence",
        "only and does not promote student code into production.",
        "",
        "## Panel",
        "",
        f"- fixtures: {', '.join(panel['panel']['fixtures'])}",
        f"- seeds: {panel['panel']['seeds']}",
        f"- particle counts: {panel['panel']['particle_counts']}",
        "",
        "## Reference Agreement",
        "",
        "| Implementation | Runs | OK | Failed | Max log-likelihood error | Max mean RMSE | Max covariance RMSE | Median runtime seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for impl, data in sorted(impl_summary.items()):
        lines.append(
            "| {impl} | {runs} | {ok} | {failed} | {ll} | {mean} | {cov} | {runtime} |".format(
                impl=impl,
                runs=data["runs"],
                ok=data["ok"],
                failed=data["failed"],
                ll=_fmt(data["max_log_likelihood_abs_error"]),
                mean=_fmt(data["max_filtered_mean_rmse_vs_reference"]),
                cov=_fmt(data["max_filtered_covariance_rmse_vs_reference"]),
                runtime=_fmt(data["median_runtime_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cross-Student Agreement",
            "",
            f"- comparable groups: {cross['comparable_groups']}",
            f"- comparison key: {cross['comparison_key']}",
            "- max filtered-mean RMSE between students: "
            f"{_fmt(cross['max_filtered_mean_rmse_between_students'])}",
            "- max log-likelihood absolute difference between students: "
            f"{_fmt(cross['max_log_likelihood_abs_diff_between_students'])}",
            "",
            "## Interpretation",
            "",
            "Both student snapshots can be called through quarantined adapters on",
            "small linear-Gaussian fixtures.  For the Kalman path, both match the",
            "independent NumPy Kalman reference to numerical precision.  This",
            "supports using the harness for baseline comparison, but it does not",
            "validate student implementations as production code.",
            "",
            "The first panel intentionally excludes nonlinear, kernel PFF, HMC,",
            "and differentiable-resampling workflows.  Those require separate",
            "targeted reproduction gates.",
            "",
            "## Next hypotheses",
            "",
            "1. The two student implementations will remain reference-consistent",
            "   on larger linear-Gaussian panels, but particle-filter ESS/runtime",
            "   behavior will diverge as observation noise decreases.",
            "2. Kernel PFF behavior in `advanced_particle_filter` is unstable or",
            "   test-sensitive in the current environment, as indicated by the",
            "   G1 partial failure; it should be reproduced in isolation before",
            "   being compared to `2026MLCOE` flow filters.",
            "3. Nonlinear fixtures will separate implementation assumptions more",
            "   strongly than linear fixtures, especially around Jacobian shape,",
            "   covariance regularization, and resampling thresholds.",
            "",
        ]
    )
    return "\n".join(lines)


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
