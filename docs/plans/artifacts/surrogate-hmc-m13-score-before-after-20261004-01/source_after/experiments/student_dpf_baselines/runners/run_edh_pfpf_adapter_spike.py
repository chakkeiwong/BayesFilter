"""Run the bounded EDH/PFPF adapter spike selected by MP4."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import platform
import sys
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
from experiments.student_dpf_baselines.fixtures.nonlinear_fixtures import (
    make_nonlinear_fixture,
)
from experiments.student_dpf_baselines.runners.run_nonlinear_reference_panel import (
    _MlcoeRangeBearingModel,
    _make_advanced_model,
    _trajectory_metrics,
)


DATE = "2026-05-11"
OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "edh_pfpf_adapter_spike_2026-05-11.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "edh_pfpf_adapter_spike_summary_2026-05-11.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/"
    "student-dpf-baseline-edh-pfpf-adapter-spike-result-2026-05-11.md"
)

BASE_FIXTURE_NAME = "range_bearing_gaussian_moderate"
REDUCED_HORIZON = 8
NUM_PARTICLES = 64
FLOW_STEPS = 10
SEED = 17
RUNTIME_WARNING_SECONDS = 30.0


def main() -> None:
    fixture = _make_reduced_fixture()
    records = [
        _run_advanced_edh_pfpf(fixture),
        _run_mlcoe_pfpf_edh(fixture),
    ]
    summary = _summarize(records)
    payload = {
        "date": DATE,
        "command": (
            "python -m "
            "experiments.student_dpf_baselines.runners.run_edh_pfpf_adapter_spike"
        ),
        "working_directory": str(Path.cwd()),
        "environment": _environment_record(),
        "panel": {
            "base_fixture": BASE_FIXTURE_NAME,
            "fixture": fixture.name,
            "target": fixture.target,
            "horizon": fixture.horizon,
            "num_particles": NUM_PARTICLES,
            "flow_steps": FLOW_STEPS,
            "seed": SEED,
            "runtime_warning_seconds": RUNTIME_WARNING_SECONDS,
        },
        "records": records,
        "summary": summary,
    }
    write_json(OUTPUT_PATH, payload)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(_render_report(summary), encoding="utf-8")


def _make_reduced_fixture() -> Any:
    fixture = make_nonlinear_fixture(BASE_FIXTURE_NAME)
    if fixture.horizon < REDUCED_HORIZON:
        raise ValueError(
            f"fixture {fixture.name!r} has horizon {fixture.horizon}, "
            f"expected at least {REDUCED_HORIZON}"
        )
    return replace(
        fixture,
        name=f"{fixture.name}_edh_pfpf_h{REDUCED_HORIZON}",
        states=fixture.states[: REDUCED_HORIZON + 1].copy(),
        observations=fixture.observations[:REDUCED_HORIZON].copy(),
    )


def _run_advanced_edh_pfpf(fixture: Any) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        model = _make_advanced_model(fixture)
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.filters import EDHParticleFilter  # type: ignore

            filt = EDHParticleFilter(
                n_particles=NUM_PARTICLES,
                n_flow_steps=FLOW_STEPS,
                resample_criterion="ess",
                ess_threshold=0.5,
                seed=SEED,
            )
            result = filt.filter(
                model,
                fixture.observations,
                return_particles=False,
                return_diagnostics=True,
                rng=np.random.default_rng(SEED),
            )
        runtime = time.perf_counter() - start
        metrics = _trajectory_metrics(result.means, fixture)
        ess = None if result.ess is None else np.asarray(result.ess, dtype=float)
        if ess is not None:
            metrics["average_ess"] = float(np.mean(ess))
            metrics["min_ess"] = float(np.min(ess))
        resampled = (
            None
            if result.resampled is None
            else np.asarray(result.resampled, dtype=bool)
        )
        return {
            "implementation_name": "advanced_particle_filter",
            "source_commit": ADVANCED_COMMIT,
            "fixture_name": fixture.name,
            "method": "EDHParticleFilter",
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": SEED,
            "num_particles": NUM_PARTICLES,
            "flow_steps": FLOW_STEPS,
            "runtime_seconds": runtime,
            "metrics": metrics,
            "diagnostics": {
                "target_label": "advanced/EDHParticleFilter/gaussian_range_bearing",
                "selected_by": "MP4 flow and DPF readiness review",
                "finite_means": bool(np.all(np.isfinite(result.means))),
                "finite_covariances": bool(np.all(np.isfinite(result.covariances))),
                "average_ess": None if ess is None else float(np.mean(ess)),
                "min_ess": None if ess is None else float(np.min(ess)),
                "resampling_count": (
                    None if resampled is None else int(np.sum(resampled))
                ),
                "resampling_count_semantics": "implementation_resampled_flags",
                "likelihood_status": "implementation_specific_not_cross_compared",
                "log_likelihood": (
                    None
                    if result.log_likelihood is None
                    else float(result.log_likelihood)
                ),
                "runtime_warning": runtime > RUNTIME_WARNING_SECONDS,
            },
        }
    except Exception as exc:
        record = exception_result(
            implementation_name="advanced_particle_filter",
            source_commit=ADVANCED_COMMIT,
            fixture_name=fixture.name,
            exc=exc,
            seed=SEED,
            num_particles=NUM_PARTICLES,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update(
            {
                "method": "EDHParticleFilter",
                "target": fixture.target,
                "flow_steps": FLOW_STEPS,
                "metrics": {},
                "diagnostics": {
                    "target_label": "advanced/EDHParticleFilter/gaussian_range_bearing",
                    "selected_by": "MP4 flow and DPF readiness review",
                },
            }
        )
        return record


def _run_mlcoe_pfpf_edh(fixture: Any) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with prepend_sys_path(MLCOE_SNAPSHOT_ROOT):
            import tensorflow as tf  # type: ignore
            from src.filters.classical import DTYPE  # type: ignore
            from src.filters.flow_filters import PFPF_EDH  # type: ignore

            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            model = _MlcoeRangeBearingModel(fixture, tf=tf, dtype=DTYPE)
            filt = PFPF_EDH(model, N=NUM_PARTICLES, steps=FLOW_STEPS)
            filt.init(
                tf.constant(fixture.m0, dtype=DTYPE),
                tf.constant(fixture.P0, dtype=DTYPE),
            )
            means = [np.asarray(fixture.m0, dtype=float)]
            ess_by_time = []
            inferred_resampling = []
            for obs in fixture.observations:
                estimate = filt.step(tf.constant(obs, dtype=DTYPE))
                means.append(np.asarray(estimate.numpy(), dtype=float).reshape(-1))
                ess = float(filt.ess.numpy())
                ess_by_time.append(ess)
                inferred_resampling.append(ess < NUM_PARTICLES / 2.0)
        runtime = time.perf_counter() - start
        means_array = np.asarray(means, dtype=float)
        metrics = _trajectory_metrics(means_array, fixture)
        ess_array = np.asarray(ess_by_time, dtype=float)
        metrics["average_ess"] = float(np.mean(ess_array))
        metrics["min_ess"] = float(np.min(ess_array))
        return {
            "implementation_name": "2026MLCOE",
            "source_commit": MLCOE_COMMIT,
            "fixture_name": fixture.name,
            "method": "PFPF_EDH",
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": SEED,
            "num_particles": NUM_PARTICLES,
            "flow_steps": FLOW_STEPS,
            "runtime_seconds": runtime,
            "metrics": metrics,
            "diagnostics": {
                "target_label": "mlcoe/PFPF_EDH/gaussian_range_bearing",
                "selected_by": "MP4 flow and DPF readiness review",
                "finite_means": bool(np.all(np.isfinite(means_array))),
                "finite_covariances": None,
                "average_ess": float(np.mean(ess_array)),
                "min_ess": float(np.min(ess_array)),
                "resampling_count": int(np.sum(inferred_resampling)),
                "resampling_count_semantics": "inferred_ess_lt_0.5N_after_step",
                "likelihood_status": "not_exposed_by_mlcoe_pfpf_edh",
                "log_likelihood": None,
                "runtime_warning": runtime > RUNTIME_WARNING_SECONDS,
            },
        }
    except Exception as exc:
        record = exception_result(
            implementation_name="2026MLCOE",
            source_commit=MLCOE_COMMIT,
            fixture_name=fixture.name,
            exc=exc,
            seed=SEED,
            num_particles=NUM_PARTICLES,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update(
            {
                "method": "PFPF_EDH",
                "target": fixture.target,
                "flow_steps": FLOW_STEPS,
                "metrics": {},
                "diagnostics": {
                    "target_label": "mlcoe/PFPF_EDH/gaussian_range_bearing",
                    "selected_by": "MP4 flow and DPF readiness review",
                },
            }
        )
        return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    by_impl = {record["implementation_name"]: record for record in records}
    decision = _decision(records)
    return {
        "records": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "implementation_summary": {
            impl: _record_summary(record)
            for impl, record in sorted(by_impl.items())
        },
        "cross_summary": _cross_summary(ok),
        "hypothesis_results": {
            "E1_advanced_edh_pfpf_runs": _interpret_run(
                by_impl.get("advanced_particle_filter")
            ),
            "E2_mlcoe_pfpf_edh_runs": _interpret_run(by_impl.get("2026MLCOE")),
            "E3_proxy_comparison_interpretable": _interpret_e3(records),
            "E4_next_phase_decision": decision,
        },
        "decision": decision,
        "next_phase_recommendation": _next_phase_recommendation(decision),
    }


def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    diagnostics = record.get("diagnostics", {})
    return {
        "status": record["status"],
        "failure_reason": record.get("failure_reason"),
        "method": record.get("method"),
        "runtime_seconds": record.get("runtime_seconds"),
        "state_rmse": metrics.get("state_rmse"),
        "position_rmse": metrics.get("position_rmse"),
        "final_position_error": metrics.get("final_position_error"),
        "observation_proxy_rmse": metrics.get("observation_proxy_rmse"),
        "average_ess": metrics.get("average_ess"),
        "min_ess": metrics.get("min_ess"),
        "finite_means": diagnostics.get("finite_means"),
        "resampling_count": diagnostics.get("resampling_count"),
        "runtime_warning": diagnostics.get("runtime_warning"),
    }


def _cross_summary(ok: list[dict[str, Any]]) -> dict[str, Any]:
    if len(ok) != 2:
        return {"status": "not_available"}
    first, second = ok
    first_metrics = first["metrics"]
    second_metrics = second["metrics"]
    return {
        "status": "available_proxy_only",
        "position_rmse_difference": (
            first_metrics.get("position_rmse") - second_metrics.get("position_rmse")
        ),
        "state_rmse_difference": (
            first_metrics.get("state_rmse") - second_metrics.get("state_rmse")
        ),
        "runtime_ratio_first_over_second": _safe_ratio(
            first.get("runtime_seconds"),
            second.get("runtime_seconds"),
        ),
        "interpretation": (
            "Differences are proxy diagnostics only and are not correctness claims."
        ),
    }


def _decision(records: list[dict[str, Any]]) -> str:
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    if len(ok) != len(records):
        return "blocked_missing_assumption"
    if any(record["diagnostics"].get("runtime_warning") for record in ok):
        return "excluded_due_to_runtime_or_numerics"
    if not all(record["diagnostics"].get("finite_means") for record in ok):
        return "excluded_due_to_runtime_or_numerics"
    return "adapter_spike_success_needs_replication"


def _interpret_run(record: dict[str, Any] | None) -> str:
    if record is None:
        return "blocked_no_record"
    if record["status"] != str(BaselineStatus.OK):
        return f"blocked: {record.get('failure_reason')}"
    if not record["diagnostics"].get("finite_means"):
        return "unsupported_nonfinite_output"
    return "supported"


def _interpret_e3(records: list[dict[str, Any]]) -> str:
    required = {
        "state_rmse",
        "position_rmse",
        "final_position_error",
        "observation_proxy_rmse",
    }
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    if len(ok) != len(records):
        return "blocked_until_both_runs_ok"
    if not all(required.issubset(record["metrics"]) for record in ok):
        return "blocked_missing_proxy_metrics"
    return "supported_proxy_only"


def _next_phase_recommendation(decision: str) -> str:
    if decision == "adapter_spike_success_needs_replication":
        return (
            "Create a replicated EDH/PFPF panel with both nonlinear fixtures, "
            "multiple seeds, and the same adapter-owned bridges."
        )
    if decision == "edh_pfpf_panel_ready":
        return "Promote directly to replicated EDH/PFPF panel."
    if decision == "blocked_missing_assumption":
        return (
            "Do not expand the panel.  Inspect the blocked model-contract "
            "assumption and decide whether an adapter-owned bridge is justified."
        )
    return "Keep EDH/PFPF excluded until runtime or numerical blockers are resolved."


def _environment_record() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf  # type: ignore

        versions["tensorflow"] = tf.__version__
        versions["tensorflow_devices"] = [
            {"name": device.name, "type": device.device_type}
            for device in tf.config.list_physical_devices()
        ]
    except Exception as exc:  # pragma: no cover - environment dependent.
        versions["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import tensorflow_probability as tfp  # type: ignore

        versions["tensorflow_probability"] = tfp.__version__
    except Exception as exc:  # pragma: no cover - environment dependent.
        versions["tensorflow_probability_error"] = f"{type(exc).__name__}: {exc}"
    return versions


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Student DPF baseline EDH/PFPF adapter-spike result",
        "",
        "## Date",
        "",
        DATE,
        "",
        "## Scope",
        "",
        "This report covers the bounded EDH/PFPF adapter spike selected by MP4.",
        "It is comparison-only evidence.  It does not promote student code,",
        "modify vendored snapshots, or make production BayesFilter claims.",
        "",
        "## Command",
        "",
        "`python -m experiments.student_dpf_baselines.runners.run_edh_pfpf_adapter_spike`",
        "",
        "Working directory: `/home/ubuntu/python/BayesFilter`",
        "",
        "## Provenance",
        "",
        f"- `advanced_particle_filter`: `{ADVANCED_COMMIT}`",
        f"- `2026MLCOE`: `{MLCOE_COMMIT}`",
        "",
        "## Panel",
        "",
        f"- base fixture: `{BASE_FIXTURE_NAME}`",
        f"- reduced horizon: `{REDUCED_HORIZON}`",
        f"- particles: `{NUM_PARTICLES}`",
        f"- flow steps: `{FLOW_STEPS}`",
        f"- seed: `{SEED}`",
        "",
        "## Implementation Summary",
        "",
        "| Implementation | Method | Status | Runtime seconds | State RMSE | Position RMSE | Final-position error | Avg ESS | Min ESS | Resampling count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for impl, data in summary["implementation_summary"].items():
        lines.append(
            f"| {impl} | {data['method']} | `{data['status']}` | "
            f"{_fmt(data['runtime_seconds'])} | {_fmt(data['state_rmse'])} | "
            f"{_fmt(data['position_rmse'])} | {_fmt(data['final_position_error'])} | "
            f"{_fmt(data['average_ess'])} | {_fmt(data['min_ess'])} | "
            f"{_fmt(data['resampling_count'])} |"
        )
    lines.extend(["", "## Cross Summary", ""])
    for key, value in summary["cross_summary"].items():
        lines.append(f"- `{key}`: {_fmt_or_text(value)}")
    lines.extend(["", "## Hypothesis Results", ""])
    for key, value in summary["hypothesis_results"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{summary['decision']}`",
            "",
            "## Interpretation",
            "",
            "The spike result is proxy evidence only.  Latent-state and position",
            "RMSE are evaluated against the shared simulated fixture.  ESS and",
            "resampling semantics are implementation-specific diagnostics.  The",
            "report does not use student agreement or likelihood values as",
            "correctness evidence.",
            "",
            "## Next Phase Recommendation",
            "",
            summary["next_phase_recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _fmt_or_text(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) or value is None:
        return _fmt(value)
    return str(value)


if __name__ == "__main__":
    main()
