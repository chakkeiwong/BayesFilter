"""Run bounded nonlinear smoke checks for student DPF baselines."""

from __future__ import annotations

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
from experiments.student_dpf_baselines.adapters.mlcoe_adapter import (
    SNAPSHOT_ROOT as MLCOE_SNAPSHOT_ROOT,
    SOURCE_COMMIT as MLCOE_COMMIT,
)


OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/nonlinear_smoke_2026-05-10.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/student-dpf-baseline-nonlinear-smoke-result-2026-05-10.md"
)


def main() -> None:
    records = [
        _run_advanced_range_bearing(),
        _run_mlcoe_range_bearing(),
    ]
    summary = _summarize(records)
    write_json(
        OUTPUT_PATH,
        {
            "date": "2026-05-10",
            "records": records,
            "summary": summary,
        },
    )
    REPORT_PATH.write_text(_render_report(records, summary), encoding="utf-8")


def _run_advanced_range_bearing() -> dict[str, Any]:
    start = time.perf_counter()
    fixture_name = "advanced_range_bearing_student_t_short"
    try:
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.filters import (  # type: ignore
                BootstrapParticleFilter,
                ExtendedKalmanFilter,
                UnscentedKalmanFilter,
            )
            from advanced_particle_filter.models.range_bearing import (  # type: ignore
                make_range_bearing_ssm,
            )
            from advanced_particle_filter.simulation import simulate  # type: ignore

            model = make_range_bearing_ssm(
                dt=0.05,
                q_diag=0.02,
                s_r=0.05,
                s_th=0.02,
                m0=(1.0, 0.5, 0.03, -0.01),
                P0_diag=(0.05, 0.05, 0.02, 0.02),
            )
            trajectory = simulate(model, T=10, seed=501)
            ekf = ExtendedKalmanFilter().filter(model, trajectory.observations)
            ukf = UnscentedKalmanFilter().filter(model, trajectory.observations)
            pf = BootstrapParticleFilter(n_particles=128, seed=0).filter(
                model,
                trajectory.observations,
                rng=np.random.default_rng(0),
            )
            return {
                "implementation_name": "advanced_particle_filter",
                "source_commit": ADVANCED_COMMIT,
                "fixture_name": fixture_name,
                "status": str(BaselineStatus.OK),
                "runtime_seconds": time.perf_counter() - start,
                "diagnostics": {
                    "ekf_log_likelihood": float(ekf.log_likelihood),
                    "ukf_log_likelihood": float(ukf.log_likelihood),
                    "pf_log_likelihood": float(pf.log_likelihood),
                    "pf_average_ess": float(pf.average_ess()),
                    "state_shape": list(trajectory.states.shape),
                    "observation_shape": list(trajectory.observations.shape),
                    "classification": "advanced_nonlinear_smoke_runnable",
                },
            }
    except Exception as exc:
        return exception_result(
            implementation_name="advanced_particle_filter",
            source_commit=ADVANCED_COMMIT,
            fixture_name=fixture_name,
            exc=exc,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()


def _run_mlcoe_range_bearing() -> dict[str, Any]:
    start = time.perf_counter()
    fixture_name = "mlcoe_range_bearing_short"
    try:
        with prepend_sys_path(MLCOE_SNAPSHOT_ROOT):
            import tensorflow as tf  # type: ignore
            from src.filters.classical import DTYPE, EKF, UKF  # type: ignore
            from src.models.classic_ssm import RangeBearingModel  # type: ignore

            tf.random.set_seed(0)
            model = RangeBearingModel(
                dt=0.05,
                sigma_q=0.2,
                sigma_r=0.5,
                sigma_theta=0.1,
                omega=-0.02,
            )
            states, observations = model.generate(T=10)
            ekf = EKF(model)
            ukf = UKF(model)
            x0 = tf.zeros(4, dtype=DTYPE)
            p0 = tf.eye(4, dtype=DTYPE)
            ekf.init(x0, p0)
            ukf.init(x0, p0)
            ekf_estimates = []
            ukf_estimates = []
            for obs in observations:
                ekf_estimates.append(ekf.step(obs).numpy().tolist())
                ukf_estimates.append(ukf.step(obs).numpy().tolist())
            return {
                "implementation_name": "2026MLCOE",
                "source_commit": MLCOE_COMMIT,
                "fixture_name": fixture_name,
                "status": str(BaselineStatus.OK),
                "runtime_seconds": time.perf_counter() - start,
                "diagnostics": {
                    "ekf_final_estimate": ekf_estimates[-1],
                    "ukf_final_estimate": ukf_estimates[-1],
                    "state_shape": list(states.shape),
                    "observation_shape": list(observations.shape),
                    "classification": "mlcoe_nonlinear_smoke_runnable",
                },
            }
    except Exception as exc:
        return exception_result(
            implementation_name="2026MLCOE",
            source_commit=MLCOE_COMMIT,
            fixture_name=fixture_name,
            exc=exc,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    if len(ok) == len(records):
        interpretation = (
            "H3 is supported as a smoke result: both student snapshots expose "
            "nonlinear range-bearing paths that can run through quarantined "
            "wrappers.  This is not a reference-consistency result."
        )
    elif ok:
        interpretation = (
            "H3 is partially supported: at least one nonlinear smoke path runs, "
            "but another is blocked and should not be compared yet."
        )
    else:
        interpretation = (
            "H3 is blocked: nonlinear smoke paths did not run through current "
            "wrappers without additional assumptions."
        )
    return {
        "ok": len(ok),
        "failed": len(failed),
        "interpretation": interpretation,
    }


def _render_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Student DPF baseline nonlinear-smoke result",
        "",
        "## Date",
        "",
        "2026-05-10",
        "",
        "## Outcomes",
        "",
        "| Implementation | Fixture | Status | Runtime seconds | Classification / reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for record in records:
        diagnostics = record.get("diagnostics", {})
        reason = diagnostics.get("classification") or record.get("failure_reason")
        lines.append(
            f"| {record['implementation_name']} | {record['fixture_name']} | "
            f"{record['status']} | {_fmt(record.get('runtime_seconds'))} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "The smoke paths are useful for deciding whether nonlinear adapter work",
            "is feasible.  They do not provide correctness, convergence, or HMC",
            "target evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
