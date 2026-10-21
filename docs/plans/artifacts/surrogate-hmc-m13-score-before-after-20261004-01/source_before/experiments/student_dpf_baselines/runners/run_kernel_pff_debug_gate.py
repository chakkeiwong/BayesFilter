"""Run the MP3 bounded kernel PFF debug gate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.student_dpf_baselines.adapters.advanced_particle_filter_adapter import (
    SOURCE_COMMIT as ADVANCED_COMMIT,
    VENDOR_ROOT as ADVANCED_VENDOR_ROOT,
)
from experiments.student_dpf_baselines.adapters.common import (
    BaselineStatus,
    exception_result,
    prepend_sys_path,
    write_json,
)


OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/kernel_pff_debug_gate_2026-05-11.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/kernel_pff_debug_gate_summary_2026-05-11.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/advanced-particle-filter-kernel-pff-debug-gate-result-2026-05-11.md"
)


@dataclass(frozen=True, slots=True)
class KernelPffCase:
    fixture_name: str
    kernel_type: str
    tolerance_label: str
    tolerance: float
    num_particles: int
    max_iterations: int
    horizon: int
    seed: int = 123
    initial_step_size: float = 0.05


def main() -> None:
    cases = _build_cases()
    records = [_run_case(case) for case in cases]
    summary = summarize(records)
    write_json(
        OUTPUT_PATH,
        {
            "date": "2026-05-11",
            "source_commit": ADVANCED_COMMIT,
            "records": records,
            "summary": summary,
        },
    )
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")


def _build_cases() -> list[KernelPffCase]:
    cases = []
    for fixture_name, horizon in [
        ("lgssm_1d_reduced", 5),
        ("lgssm_cv_2d_reduced", 4),
    ]:
        for kernel_type in ["scalar", "matrix"]:
            for tolerance_label, tolerance in [
                ("loose", 1e-3),
                ("strict", 1e-5),
            ]:
                for num_particles in [64, 128]:
                    cases.append(
                        KernelPffCase(
                            fixture_name=fixture_name,
                            kernel_type=kernel_type,
                            tolerance_label=tolerance_label,
                            tolerance=tolerance,
                            num_particles=num_particles,
                            max_iterations=40,
                            horizon=horizon,
                        )
                    )
    return cases


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == "ok"]
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tolerance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ok:
        by_kernel[record["kernel_type"]].append(record)
        by_tolerance[record["tolerance_label"]].append(record)
        by_fixture[record["fixture_name"]].append(record)

    kernel_summary = {
        key: _group_summary(group)
        for key, group in sorted(by_kernel.items())
    }
    tolerance_summary = {
        key: _group_summary(group)
        for key, group in sorted(by_tolerance.items())
    }
    fixture_summary = {
        key: _group_summary(group)
        for key, group in sorted(by_fixture.items())
    }
    decision = _readiness_decision(ok)
    return {
        "runs": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "kernel_summary": kernel_summary,
        "tolerance_summary": tolerance_summary,
        "fixture_summary": fixture_summary,
        "readiness_decision": decision,
        "hypothesis_results": {
            "k1_reduced_fixtures_runnable": _interpret_k1(ok),
            "k2_tolerance_sensitivity": _interpret_k2(tolerance_summary),
            "k3_max_iteration_failure_mode": _interpret_k3(ok),
            "k4_routine_panel_readiness": _interpret_k4(decision),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Advanced particle filter kernel PFF debug-gate result",
        "",
        "## Date",
        "",
        "2026-05-11",
        "",
        "## Scope",
        "",
        "This report covers MP3 of the quarantined student DPF experimental-baseline",
        "stream.  It narrows the prior kernel PFF timeout/failure classification",
        "using reduced bounded diagnostics.  It does not promote kernel PFF into",
        "routine comparison panels.",
        "",
        "## Overall",
        "",
        f"- runs: {summary['runs']}",
        f"- ok: {summary['ok']}",
        f"- failed: {summary['failed']}",
        f"- readiness decision: `{summary['readiness_decision']}`",
        "",
        "## Kernel Summary",
        "",
        "| Kernel | Runs | Median runtime seconds | Median avg iterations | Max hit-max fraction | Median RMSE vs KF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, data in sorted(summary["kernel_summary"].items()):
        lines.append(
            f"| {key} | {data['runs']} | {_fmt(data['median_runtime_seconds'])} | "
            f"{_fmt(data['median_average_iterations'])} | "
            f"{_fmt(data['max_hit_max_fraction'])} | "
            f"{_fmt(data['median_rmse_vs_kalman'])} |"
        )
    lines.extend(
        [
            "",
            "## Tolerance Summary",
            "",
            "| Tolerance | Runs | Median runtime seconds | Median avg iterations | Max hit-max fraction | Median final flow magnitude |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in sorted(summary["tolerance_summary"].items()):
        lines.append(
            f"| {key} | {data['runs']} | {_fmt(data['median_runtime_seconds'])} | "
            f"{_fmt(data['median_average_iterations'])} | "
            f"{_fmt(data['max_hit_max_fraction'])} | "
            f"{_fmt(data['median_final_flow_magnitude'])} |"
        )
    lines.extend(["", "## Hypothesis Results", ""])
    for name, text in summary["hypothesis_results"].items():
        lines.extend([f"### {name}", "", text, ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "A completed filter run is not the same as converged flow iterations.",
            "Runs that complete while every step hits `max_iterations` remain",
            "debug evidence only.  Kernel PFF should not enter routine panels",
            "unless bounded convergence is consistent across reduced cases.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_case(case: KernelPffCase) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.filters import (  # type: ignore
                KalmanFilter,
                MatrixKernelPFF,
                ScalarKernelPFF,
            )
            from advanced_particle_filter.simulation import simulate  # type: ignore

            model = _make_model(case.fixture_name)
            trajectory = simulate(
                model,
                T=case.horizon,
                rng=np.random.default_rng(42),
            )
            kalman = KalmanFilter().filter(model, trajectory.observations)
            pff_cls = ScalarKernelPFF if case.kernel_type == "scalar" else MatrixKernelPFF
            pff = pff_cls(
                n_particles=case.num_particles,
                max_iterations=case.max_iterations,
                tolerance=case.tolerance,
                initial_step_size=case.initial_step_size,
                seed=case.seed,
                store_diagnostics=True,
            )
            result = pff.filter(
                model,
                trajectory.observations,
                return_diagnostics=True,
                rng=np.random.default_rng(case.seed),
            )
            iterations = np.asarray(
                [diag.n_iterations for diag in result.diagnostics],
                dtype=float,
            )
            final_flow = np.asarray(
                [diag.final_flow_magnitude for diag in result.diagnostics],
                dtype=float,
            )
            hit_max = iterations >= case.max_iterations
            rmse_truth = float(result.mean_rmse(trajectory.states))
            rmse_kalman = float(np.sqrt(np.mean((result.means - kalman.means) ** 2)))
            return {
                "implementation_name": "advanced_particle_filter",
                "source_commit": ADVANCED_COMMIT,
                "fixture_name": case.fixture_name,
                "kernel_type": case.kernel_type,
                "tolerance_label": case.tolerance_label,
                "tolerance": case.tolerance,
                "num_particles": case.num_particles,
                "max_iterations": case.max_iterations,
                "horizon": case.horizon,
                "status": str(BaselineStatus.OK),
                "runtime_seconds": time.perf_counter() - start,
                "diagnostics": {
                    "average_iterations": float(np.mean(iterations)),
                    "max_iterations_observed": int(np.max(iterations)),
                    "hit_max_count": int(np.sum(hit_max)),
                    "hit_max_fraction": float(np.mean(hit_max)),
                    "median_final_flow_magnitude": float(np.median(final_flow)),
                    "max_final_flow_magnitude": float(np.max(final_flow)),
                    "rmse_vs_truth": rmse_truth,
                    "rmse_vs_kalman": rmse_kalman,
                    "completed_filter_run": True,
                    "all_steps_converged": bool(not np.any(hit_max)),
                },
            }
    except Exception as exc:
        record = exception_result(
            implementation_name="advanced_particle_filter",
            source_commit=ADVANCED_COMMIT,
            fixture_name=case.fixture_name,
            exc=exc,
            num_particles=case.num_particles,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update(
            {
                "kernel_type": case.kernel_type,
                "tolerance_label": case.tolerance_label,
                "tolerance": case.tolerance,
                "max_iterations": case.max_iterations,
                "horizon": case.horizon,
            }
        )
        return record


def _make_model(fixture_name: str) -> Any:
    from advanced_particle_filter.models import make_lgssm  # type: ignore

    if fixture_name == "lgssm_1d_reduced":
        A = np.array([[0.9]], dtype=float)
        C = np.array([[1.0]], dtype=float)
        Q = np.array([[0.1]], dtype=float)
        R = np.array([[0.5]], dtype=float)
        m0 = np.array([0.0], dtype=float)
        P0 = np.array([[1.0]], dtype=float)
    elif fixture_name == "lgssm_cv_2d_reduced":
        dt = 1.0
        A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        q = 0.1
        Q = q * np.array(
            [
                [dt**3 / 3, 0.0, dt**2 / 2, 0.0],
                [0.0, dt**3 / 3, 0.0, dt**2 / 2],
                [dt**2 / 2, 0.0, dt, 0.0],
                [0.0, dt**2 / 2, 0.0, dt],
            ],
            dtype=float,
        )
        R = 0.5 * np.eye(2)
        m0 = np.array([0.0, 0.0, 1.0, 0.5], dtype=float)
        P0 = 0.1 * np.eye(4)
    else:
        raise ValueError(f"unknown fixture {fixture_name!r}")
    return make_lgssm(A, C, Q, R, m0, P0)


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(records),
        "median_runtime_seconds": _median(r.get("runtime_seconds") for r in records),
        "median_average_iterations": _median(
            r["diagnostics"].get("average_iterations") for r in records
        ),
        "max_hit_max_fraction": _max(
            r["diagnostics"].get("hit_max_fraction") for r in records
        ),
        "median_final_flow_magnitude": _median(
            r["diagnostics"].get("median_final_flow_magnitude") for r in records
        ),
        "median_rmse_vs_kalman": _median(
            r["diagnostics"].get("rmse_vs_kalman") for r in records
        ),
        "all_runs_converged": all(
            bool(r["diagnostics"].get("all_steps_converged")) for r in records
        ),
    }


def _readiness_decision(records: list[dict[str, Any]]) -> str:
    if not records:
        return "excluded_pending_debug"
    all_converged = all(r["diagnostics"].get("all_steps_converged") for r in records)
    max_runtime = max(float(r["runtime_seconds"]) for r in records)
    max_hit = max(float(r["diagnostics"].get("hit_max_fraction", 1.0)) for r in records)
    if all_converged and max_runtime < 5.0:
        return "routine_ready_reduced_cases_only"
    if max_runtime < 20.0 and max_hit < 1.0:
        return "slow_experimental_only"
    return "excluded_pending_debug"


def _interpret_k1(records: list[dict[str, Any]]) -> str:
    kernels = {r["kernel_type"] for r in records}
    fixtures = {r["fixture_name"] for r in records}
    if {"scalar", "matrix"}.issubset(kernels) and len(fixtures) >= 2:
        return (
            "Supported.  Reduced scalar and matrix kernel PFF runs completed on "
            "both reduced fixture families."
        )
    return "Blocked or partially supported: reduced kernel coverage is incomplete."


def _interpret_k2(tolerance_summary: dict[str, Any]) -> str:
    loose = tolerance_summary.get("loose")
    strict = tolerance_summary.get("strict")
    if not loose or not strict:
        return "Inconclusive: loose and strict tolerance groups are not both present."
    loose_iter = loose.get("median_average_iterations")
    strict_iter = strict.get("median_average_iterations")
    loose_runtime = loose.get("median_runtime_seconds")
    strict_runtime = strict.get("median_runtime_seconds")
    if (
        loose_iter is not None
        and strict_iter is not None
        and loose_runtime is not None
        and strict_runtime is not None
        and (loose_iter < strict_iter or loose_runtime < strict_runtime)
    ):
        return (
            "Supported.  Loose tolerance reduced median iterations or runtime "
            "relative to strict tolerance."
        )
    return (
        "Not supported for this bounded panel.  Loose tolerance did not reduce "
        "median iterations or runtime."
    )


def _interpret_k3(records: list[dict[str, Any]]) -> str:
    if not records:
        return "Inconclusive: no completed records."
    hit_max_records = [
        r for r in records
        if r["diagnostics"].get("hit_max_fraction", 0.0) > 0.0
    ]
    if hit_max_records:
        return (
            "Supported.  Non-converged behavior appears as finite completed "
            "runs with `hit_max_iter` diagnostics, not missing dependencies."
        )
    return "Not observed: no reduced run hit the maximum iteration cap."


def _interpret_k4(decision: str) -> str:
    if decision == "routine_ready_reduced_cases_only":
        return (
            "Partially supported.  Reduced cases are bounded, but routine-panel "
            "readiness should remain limited to reduced cases until larger "
            "fixtures are tested."
        )
    if decision == "slow_experimental_only":
        return (
            "Supported.  Kernel PFF should remain slow experimental evidence "
            "rather than routine-panel evidence."
        )
    return (
        "Supported.  Kernel PFF should remain excluded from routine panels "
        "pending further debug."
    )


def _clean(values: Any) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(float(v))]


def _max(values: Any) -> float | None:
    vals = _clean(values)
    return max(vals) if vals else None


def _median(values: Any) -> float | None:
    vals = _clean(values)
    return float(np.median(vals)) if vals else None


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
