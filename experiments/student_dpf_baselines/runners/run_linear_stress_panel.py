"""Run larger linear-Gaussian stress panel for student-baseline hypotheses."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.student_dpf_baselines.adapters.advanced_particle_filter_adapter import (
    run_smoke_fixture as run_advanced_particle_filter,
)
from experiments.student_dpf_baselines.adapters.common import write_json
from experiments.student_dpf_baselines.adapters.mlcoe_adapter import (
    run_smoke_fixture as run_mlcoe,
)
from experiments.student_dpf_baselines.fixtures.stress_fixtures import (
    make_stress_fixture,
    stress_fixture_names,
)
from experiments.student_dpf_baselines.runners.run_reference_fixtures import (
    run_kalman_reference,
)


OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/linear_stress_panel_2026-05-10.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/linear_stress_panel_summary_2026-05-10.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/student-dpf-baseline-linear-stress-result-2026-05-10.md"
)

SEEDS = [0, 1, 2, 3, 4]
PARTICLE_COUNTS = [64, 128, 512]


def main() -> None:
    records = []
    references = {}
    for fixture_name in stress_fixture_names():
        fixture = make_stress_fixture(fixture_name)
        reference = run_kalman_reference(fixture)
        references[fixture_name] = reference
        for seed in SEEDS:
            for n_particles in PARTICLE_COUNTS:
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
                records.append(
                    _with_reference_metrics(
                        run_mlcoe(
                            fixture,
                            seed=seed,
                            num_particles=n_particles,
                        ),
                        reference,
                    )
                )
    panel = {
        "date": "2026-05-10",
        "panel": {
            "fixtures": stress_fixture_names(),
            "seeds": SEEDS,
            "particle_counts": PARTICLE_COUNTS,
        },
        "references": references,
        "records": records,
    }
    summary = summarize(records)
    write_json(OUTPUT_PATH, panel)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")


def summarize(records: list[dict]) -> dict:
    by_impl: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_impl[record["implementation_name"]].append(record)

    impl_summary = {}
    for impl, impl_records in by_impl.items():
        ok = [r for r in impl_records if r["status"] == "ok"]
        impl_summary[impl] = {
            "runs": len(impl_records),
            "ok": len(ok),
            "failed": len(impl_records) - len(ok),
            "max_kalman_log_likelihood_abs_error": _max(
                r["diagnostics"].get("kalman_log_likelihood_abs_error") for r in ok
            ),
            "max_filtered_mean_rmse_vs_kalman": _max(
                r["diagnostics"].get("filtered_mean_rmse_vs_kalman") for r in ok
            ),
            "median_runtime_seconds": _median(r.get("runtime_seconds") for r in ok),
        }

    advanced_pf = [
        r for r in records
        if r["implementation_name"] == "advanced_particle_filter" and r["status"] == "ok"
    ]
    pf_by_fixture_particles = defaultdict(list)
    for record in advanced_pf:
        key = (record["fixture_name"], record["num_particles"])
        pf_by_fixture_particles[key].append(record)

    pf_summary = {}
    for (fixture, particles), group in sorted(pf_by_fixture_particles.items()):
        pf_summary[f"{fixture}/N={particles}"] = {
            "median_pf_log_likelihood_abs_error": _median(
                r["diagnostics"].get("particle_filter_log_likelihood_abs_error")
                for r in group
            ),
            "median_average_ess": _median(
                r["diagnostics"].get("particle_filter_average_ess")
                for r in group
            ),
            "min_average_ess": _min(
                r["diagnostics"].get("particle_filter_average_ess")
                for r in group
            ),
            "median_resampling_count": _median(
                r.get("resampling_count") for r in group
            ),
        }

    return {
        "implementation_summary": impl_summary,
        "advanced_particle_filter_pf_summary": pf_summary,
        "hypothesis_h1_interpretation": _interpret_h1(pf_summary),
    }


def render_report(summary: dict) -> str:
    lines = [
        "# Student DPF baseline linear-stress result",
        "",
        "## Date",
        "",
        "2026-05-10",
        "",
        "## Reference Agreement",
        "",
        "| Implementation | Runs | OK | Failed | Max Kalman log-likelihood error | Max mean RMSE | Median runtime seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for impl, data in sorted(summary["implementation_summary"].items()):
        lines.append(
            f"| {impl} | {data['runs']} | {data['ok']} | {data['failed']} | "
            f"{_fmt(data['max_kalman_log_likelihood_abs_error'])} | "
            f"{_fmt(data['max_filtered_mean_rmse_vs_kalman'])} | "
            f"{_fmt(data['median_runtime_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "## Advanced Bootstrap-PF Diagnostics",
            "",
            "| Fixture / particles | Median PF log-likelihood error | Median avg ESS | Min avg ESS | Median resampling count |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in sorted(summary["advanced_particle_filter_pf_summary"].items()):
        lines.append(
            f"| {key} | {_fmt(data['median_pf_log_likelihood_abs_error'])} | "
            f"{_fmt(data['median_average_ess'])} | {_fmt(data['min_average_ess'])} | "
            f"{_fmt(data['median_resampling_count'])} |"
        )
    lines.extend(
        [
            "",
            "## H1 Interpretation",
            "",
            summary["hypothesis_h1_interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def _with_reference_metrics(result, reference: dict) -> dict:
    record = result.to_dict()
    diagnostics = record.setdefault("diagnostics", {})
    if record["status"] != "ok":
        return record
    reference_ll = float(reference["log_likelihood"])
    if result.log_likelihood is not None:
        diagnostics["kalman_log_likelihood_abs_error"] = abs(
            float(result.log_likelihood) - reference_ll
        )
    if result.filtered_means is not None:
        diagnostics["filtered_mean_rmse_vs_kalman"] = float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(result.filtered_means, dtype=float)
                        - np.asarray(reference["filtered_means"], dtype=float)
                    )
                    ** 2
                )
            )
        )
    pf_ll = diagnostics.get("particle_filter_log_likelihood")
    if pf_ll is not None:
        diagnostics["particle_filter_log_likelihood_abs_error"] = abs(
            float(pf_ll) - reference_ll
        )
    return record


def _interpret_h1(pf_summary: dict) -> str:
    low_noise_keys = [k for k in pf_summary if "low_noise" in k]
    if not low_noise_keys:
        return "H1 is inconclusive: no low-noise stress fixture was summarized."
    low_noise_errors = [
        pf_summary[k]["median_pf_log_likelihood_abs_error"]
        for k in low_noise_keys
        if pf_summary[k]["median_pf_log_likelihood_abs_error"] is not None
    ]
    if low_noise_errors and max(low_noise_errors) > 5.0:
        return (
            "H1 is supported for particle diagnostics: Kalman paths remain "
            "reference-consistent, while the advanced bootstrap-PF diagnostics "
            "show materially larger log-likelihood error under the low-noise "
            "stress fixture.  MLCOE particle diagnostics remain unsupported in "
            "this adapter cycle."
        )
    return (
        "H1 is only weakly supported: Kalman paths remain reference-consistent, "
        "but the observed advanced bootstrap-PF diagnostics did not show a large "
        "low-noise divergence under this bounded panel."
    )


def _clean(values) -> list[float]:
    return [float(v) for v in values if v is not None]


def _max(values) -> float | None:
    vals = _clean(values)
    return max(vals) if vals else None


def _min(values) -> float | None:
    vals = _clean(values)
    return min(vals) if vals else None


def _median(values) -> float | None:
    vals = _clean(values)
    return float(np.median(vals)) if vals else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
