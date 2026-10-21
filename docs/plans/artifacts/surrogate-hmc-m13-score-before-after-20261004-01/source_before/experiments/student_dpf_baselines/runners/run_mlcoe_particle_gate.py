"""Run the MP1 MLCOE particle adapter gate."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.student_dpf_baselines.adapters.advanced_particle_filter_adapter import (
    run_smoke_fixture as run_advanced_particle_filter,
)
from experiments.student_dpf_baselines.adapters.common import write_json
from experiments.student_dpf_baselines.adapters.mlcoe_adapter import run_bpf_fixture
from experiments.student_dpf_baselines.fixtures.common_fixtures import make_fixture
from experiments.student_dpf_baselines.fixtures.stress_fixtures import make_stress_fixture
from experiments.student_dpf_baselines.runners.run_reference_fixtures import (
    run_kalman_reference,
)


OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/mlcoe_particle_gate_2026-05-10.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/mlcoe_particle_gate_summary_2026-05-10.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/student-dpf-baseline-mlcoe-particle-gate-result-2026-05-10.md"
)

COMMON_FIXTURES = ["lgssm_1d_short", "lgssm_cv_2d_short"]
STRESS_FIXTURES = ["lgssm_1d_long", "lgssm_cv_2d_long", "lgssm_cv_2d_low_noise"]
SEEDS = [0, 1, 2]
PARTICLE_COUNTS = [64, 128, 512]


def main() -> None:
    records = []
    references = {}
    for fixture_name in [*COMMON_FIXTURES, *STRESS_FIXTURES]:
        fixture = _make_fixture(fixture_name)
        reference = run_kalman_reference(fixture)
        references[fixture_name] = reference
        for seed in SEEDS:
            for n_particles in PARTICLE_COUNTS:
                records.append(
                    _with_reference_metrics(
                        run_bpf_fixture(
                            fixture,
                            seed=seed,
                            num_particles=n_particles,
                        ),
                        reference,
                    )
                )
                records.append(
                    _with_reference_metrics(
                        run_advanced_particle_filter(
                            fixture,
                            seed=seed,
                            num_particles=n_particles,
                        ),
                        reference,
                    )
                )

    summary = summarize(records)
    panel = {
        "date": "2026-05-10",
        "panel": {
            "fixtures": [*COMMON_FIXTURES, *STRESS_FIXTURES],
            "seeds": SEEDS,
            "particle_counts": PARTICLE_COUNTS,
        },
        "references": references,
        "records": records,
        "summary": summary,
    }
    write_json(OUTPUT_PATH, panel)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [r for r in records if r["status"] == "ok"]
    by_impl: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_impl_fixture_particles: dict[tuple[str, str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for record in records:
        by_impl[record["implementation_name"]].append(record)
        if record["num_particles"] is not None:
            by_impl_fixture_particles[
                (
                    record["implementation_name"],
                    record["fixture_name"],
                    int(record["num_particles"]),
                )
            ].append(record)

    implementation_summary = {}
    for impl, impl_records in sorted(by_impl.items()):
        ok = [r for r in impl_records if r["status"] == "ok"]
        implementation_summary[impl] = {
            "runs": len(impl_records),
            "ok": len(ok),
            "failed": len(impl_records) - len(ok),
            "median_runtime_seconds": _median(r.get("runtime_seconds") for r in ok),
            "max_particle_mean_rmse_vs_kalman": _max(
                r["diagnostics"].get("particle_mean_rmse_vs_kalman") for r in ok
            ),
            "max_particle_covariance_rmse_vs_kalman": _max(
                r["diagnostics"].get("particle_covariance_rmse_vs_kalman")
                for r in ok
            ),
            "min_average_ess": _min(
                r["diagnostics"].get("particle_filter_average_ess") for r in ok
            ),
            "median_threshold_or_direct_resampling_count": _median(
                r.get("resampling_count") for r in ok
            ),
            "log_likelihood_available_runs": sum(
                1 for r in ok if r.get("log_likelihood") is not None
            ),
        }

    particle_summary = {}
    for key, group in sorted(by_impl_fixture_particles.items()):
        impl, fixture, particles = key
        ok = [r for r in group if r["status"] == "ok"]
        particle_summary[f"{impl}/{fixture}/N={particles}"] = {
            "runs": len(group),
            "ok": len(ok),
            "median_particle_mean_rmse_vs_kalman": _median(
                r["diagnostics"].get("particle_mean_rmse_vs_kalman") for r in ok
            ),
            "median_particle_covariance_rmse_vs_kalman": _median(
                r["diagnostics"].get("particle_covariance_rmse_vs_kalman")
                for r in ok
            ),
            "median_average_ess": _median(
                r["diagnostics"].get("particle_filter_average_ess") for r in ok
            ),
            "min_average_ess": _min(
                r["diagnostics"].get("particle_filter_average_ess") for r in ok
            ),
            "median_runtime_seconds": _median(r.get("runtime_seconds") for r in ok),
            "median_resampling_count": _median(r.get("resampling_count") for r in ok),
        }

    cross_student = _summarize_cross_student(ok_records)

    return {
        "implementation_summary": implementation_summary,
        "particle_summary": particle_summary,
        "cross_student_particle_summary": cross_student,
        "hypothesis_results": {
            "h1_mlcoe_bpf_adapter_feasibility": _interpret_h1(by_impl),
            "h2_diagnostic_extraction": _interpret_h2(by_impl),
            "h3_linear_stress_particle_behavior": _interpret_h3(particle_summary),
            "h4_cross_student_particle_comparison": _interpret_h4(cross_student),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Student DPF baseline MP1 MLCOE particle-gate result",
        "",
        "## Date",
        "",
        "2026-05-10",
        "",
        "## Scope",
        "",
        "This report covers the MP1 MLCOE BPF adapter gate in the quarantined",
        "student DPF experimental-baseline stream.  It is comparison-only",
        "evidence and does not promote student code into production.",
        "",
        "## Implementation Summary",
        "",
        "| Implementation | Runs | OK | Failed | Max mean RMSE vs Kalman | Max covariance RMSE vs Kalman | Min avg ESS | Median runtime seconds | Likelihood runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for impl, data in sorted(summary["implementation_summary"].items()):
        lines.append(
            f"| {impl} | {data['runs']} | {data['ok']} | {data['failed']} | "
            f"{_fmt(data['max_particle_mean_rmse_vs_kalman'])} | "
            f"{_fmt(data['max_particle_covariance_rmse_vs_kalman'])} | "
            f"{_fmt(data['min_average_ess'])} | "
            f"{_fmt(data['median_runtime_seconds'])} | "
            f"{data['log_likelihood_available_runs']} |"
        )

    lines.extend(
        [
            "",
            "## Particle Summary",
            "",
            "| Implementation / fixture / particles | Runs | OK | Median mean RMSE vs Kalman | Median covariance RMSE vs Kalman | Median avg ESS | Min avg ESS | Median resampling count | Median runtime seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in sorted(summary["particle_summary"].items()):
        lines.append(
            f"| {key} | {data['runs']} | {data['ok']} | "
            f"{_fmt(data['median_particle_mean_rmse_vs_kalman'])} | "
            f"{_fmt(data['median_particle_covariance_rmse_vs_kalman'])} | "
            f"{_fmt(data['median_average_ess'])} | "
            f"{_fmt(data['min_average_ess'])} | "
            f"{_fmt(data['median_resampling_count'])} | "
            f"{_fmt(data['median_runtime_seconds'])} |"
        )

    lines.extend(
        [
            "",
            "## Cross-Student Particle Comparison",
            "",
            "| Fixture / particles | Groups | Median mean RMSE | Median covariance RMSE | Median ESS difference | Median runtime ratio MLCOE/advanced |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in sorted(summary["cross_student_particle_summary"].items()):
        lines.append(
            f"| {key} | {data['groups']} | "
            f"{_fmt(data['median_particle_mean_rmse_between_students'])} | "
            f"{_fmt(data['median_particle_covariance_rmse_between_students'])} | "
            f"{_fmt(data['median_average_ess_difference_mlcoe_minus_advanced'])} | "
            f"{_fmt(data['median_runtime_ratio_mlcoe_over_advanced'])} |"
        )

    lines.extend(["", "## Hypothesis Results", ""])
    for name, text in summary["hypothesis_results"].items():
        lines.extend([f"### {name}", "", text, ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            "MLCOE BPF likelihood fields remain null because the vendored BPF path",
            "does not expose a defensible likelihood estimator.  MLCOE resampling",
            "counts are threshold-inferred from the documented `ess < 0.1 * N`",
            "condition and should not be treated as direct event logs.",
            "",
        ]
    )
    return "\n".join(lines)


def _make_fixture(name: str) -> Any:
    if name in COMMON_FIXTURES:
        return make_fixture(name)
    return make_stress_fixture(name)


def _with_reference_metrics(result: Any, reference: dict[str, Any]) -> dict[str, Any]:
    record = result.to_dict()
    diagnostics = record.setdefault("diagnostics", {})
    if record["status"] != "ok":
        return record

    reference_means = np.asarray(reference["filtered_means"], dtype=float)
    reference_covariances = np.asarray(reference["filtered_covariances"], dtype=float)

    particle_means = record.get("particle_means")
    if particle_means is not None:
        diagnostics["particle_mean_rmse_vs_kalman"] = _rmse(
            np.asarray(particle_means, dtype=float),
            reference_means,
        )
    particle_covariances = record.get("particle_covariances")
    if particle_covariances is not None:
        diagnostics["particle_covariance_rmse_vs_kalman"] = _rmse(
            np.asarray(particle_covariances, dtype=float),
            reference_covariances,
        )

    ess = record.get("ess_by_time")
    if ess is not None:
        ess_array = np.asarray(ess, dtype=float)
        diagnostics["particle_filter_average_ess"] = float(np.mean(ess_array))
        diagnostics["particle_filter_min_ess"] = float(np.min(ess_array))

    reference_ll = float(reference["log_likelihood"])
    pf_ll = diagnostics.get("particle_filter_log_likelihood")
    if pf_ll is not None:
        diagnostics["particle_filter_log_likelihood_abs_error"] = abs(
            float(pf_ll) - reference_ll
        )
    return record


def _summarize_cross_student(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["num_particles"] is None:
            continue
        key = (record["fixture_name"], int(record["seed"]), int(record["num_particles"]))
        grouped[key][record["implementation_name"]] = record

    by_fixture_particles: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (fixture, _seed, particles), group in grouped.items():
        mlcoe = group.get("2026MLCOE")
        advanced = group.get("advanced_particle_filter")
        if mlcoe is None or advanced is None:
            continue
        comparison = {
            "particle_mean_rmse_between_students": _rmse(
                np.asarray(mlcoe["particle_means"], dtype=float),
                np.asarray(advanced["particle_means"], dtype=float),
            ),
            "particle_covariance_rmse_between_students": _rmse(
                np.asarray(mlcoe["particle_covariances"], dtype=float),
                np.asarray(advanced["particle_covariances"], dtype=float),
            ),
            "average_ess_difference_mlcoe_minus_advanced": (
                float(np.mean(np.asarray(mlcoe["ess_by_time"], dtype=float)))
                - float(np.mean(np.asarray(advanced["ess_by_time"], dtype=float)))
            ),
            "runtime_ratio_mlcoe_over_advanced": (
                float(mlcoe["runtime_seconds"])
                / max(float(advanced["runtime_seconds"]), 1e-12)
            ),
        }
        by_fixture_particles[(fixture, particles)].append(comparison)

    summary = {}
    for (fixture, particles), group in sorted(by_fixture_particles.items()):
        summary[f"{fixture}/N={particles}"] = {
            "groups": len(group),
            "median_particle_mean_rmse_between_students": _median(
                g["particle_mean_rmse_between_students"] for g in group
            ),
            "median_particle_covariance_rmse_between_students": _median(
                g["particle_covariance_rmse_between_students"] for g in group
            ),
            "median_average_ess_difference_mlcoe_minus_advanced": _median(
                g["average_ess_difference_mlcoe_minus_advanced"] for g in group
            ),
            "median_runtime_ratio_mlcoe_over_advanced": _median(
                g["runtime_ratio_mlcoe_over_advanced"] for g in group
            ),
        }
    return summary


def _interpret_h1(by_impl: dict[str, list[dict[str, Any]]]) -> str:
    mlcoe_records = by_impl.get("2026MLCOE", [])
    ok = [r for r in mlcoe_records if r["status"] == "ok"]
    if ok:
        return (
            "Supported.  MLCOE BPF ran through the quarantined adapter without "
            "vendored-code edits and returned particle trajectories plus ESS "
            "diagnostics."
        )
    return (
        "Not supported.  No MLCOE BPF run completed successfully; inspect "
        "structured blocker records before continuing."
    )


def _interpret_h2(by_impl: dict[str, list[dict[str, Any]]]) -> str:
    ok = [r for r in by_impl.get("2026MLCOE", []) if r["status"] == "ok"]
    if not ok:
        return "Inconclusive because no MLCOE BPF run completed."
    diagnostics_ok = all(
        r.get("particle_means") is not None
        and r.get("particle_covariances") is not None
        and r.get("ess_by_time") is not None
        and r.get("resampling_count") is not None
        for r in ok
    )
    if diagnostics_ok:
        return (
            "Supported.  MLCOE BPF exposes enough state, weight, and ESS data "
            "for particle mean, weighted covariance, ESS, runtime, and "
            "threshold-inferred resampling diagnostics.  Likelihood remains "
            "unavailable."
        )
    return (
        "Partially supported.  Some MLCOE BPF runs completed but one or more "
        "diagnostic fields were unavailable."
    )


def _interpret_h3(particle_summary: dict[str, Any]) -> str:
    low_noise = {
        key: value
        for key, value in particle_summary.items()
        if key.startswith("2026MLCOE/lgssm_cv_2d_low_noise")
    }
    if not low_noise:
        return "Inconclusive: no MLCOE low-noise stress records were produced."
    errors = [
        data["median_particle_mean_rmse_vs_kalman"]
        for data in low_noise.values()
        if data["median_particle_mean_rmse_vs_kalman"] is not None
    ]
    ess_values = [
        data["min_average_ess"]
        for data in low_noise.values()
        if data["min_average_ess"] is not None
    ]
    if errors and ess_values and max(errors) > 0.5 and min(ess_values) < 25.0:
        return (
            "Supported for this bounded panel.  MLCOE BPF shows materially "
            "larger Kalman-reference state error and low ESS under the "
            "low-noise stress fixture."
        )
    if errors and ess_values:
        return (
            "Partially supported.  MLCOE BPF produced interpretable low-noise "
            "stress diagnostics, but the degradation pattern was weaker than "
            "the pre-specified qualitative threshold."
        )
    return "Inconclusive: low-noise records lacked interpretable error or ESS metrics."


def _interpret_h4(cross_student: dict[str, Any]) -> str:
    if cross_student:
        return (
            "Supported as comparison-only evidence.  Matched fixture, seed, and "
            "particle-count groups were summarized without treating student "
            "agreement as correctness and without fabricating MLCOE likelihoods."
        )
    return (
        "Not supported.  No matched cross-student particle groups were available "
        "after filtering to successful runs."
    )


def _rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch for RMSE: {lhs.shape} != {rhs.shape}")
    return float(np.sqrt(np.mean((lhs - rhs) ** 2)))


def _clean(values: Any) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(float(v))]


def _max(values: Any) -> float | None:
    vals = _clean(values)
    return max(vals) if vals else None


def _min(values: Any) -> float | None:
    vals = _clean(values)
    return min(vals) if vals else None


def _median(values: Any) -> float | None:
    vals = _clean(values)
    return float(np.median(vals)) if vals else None


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
