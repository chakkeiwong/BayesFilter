"""Run the replicated EDH/PFPF panel selected for the student DPF baseline."""

from __future__ import annotations

from collections import defaultdict
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


DATE = "2026-05-12"
COMMAND = (
    "python -m "
    "experiments.student_dpf_baselines.runners.run_replicated_edh_pfpf_panel"
)
OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "replicated_edh_pfpf_panel_2026-05-12.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "replicated_edh_pfpf_panel_summary_2026-05-12.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/"
    "student-dpf-baseline-replicated-edh-pfpf-panel-result-2026-05-12.md"
)

FIXTURE_NAMES = [
    "range_bearing_gaussian_moderate",
    "range_bearing_gaussian_low_noise",
]
SEEDS = [17, 23, 31]
REDUCED_HORIZON = 8
NUM_PARTICLES = 64
FLOW_STEPS = 10
RUNTIME_WARNING_SECONDS = 30.0
PLANNED_RECORDS = len(FIXTURE_NAMES) * len(SEEDS) * 2


def main() -> None:
    fixtures = [_make_reduced_fixture(name) for name in FIXTURE_NAMES]
    records = []
    for fixture in fixtures:
        for seed in SEEDS:
            records.append(_run_advanced_edh_pfpf(fixture, seed))
            records.append(_run_mlcoe_pfpf_edh(fixture, seed))
    summary = _summarize(records)
    payload = {
        "date": DATE,
        "command": COMMAND,
        "working_directory": str(Path.cwd()),
        "environment": _environment_record(),
        "panel": {
            "base_fixtures": FIXTURE_NAMES,
            "reduced_horizon": REDUCED_HORIZON,
            "seeds": SEEDS,
            "num_particles": NUM_PARTICLES,
            "flow_steps": FLOW_STEPS,
            "runtime_warning_seconds": RUNTIME_WARNING_SECONDS,
            "planned_records": PLANNED_RECORDS,
            "interpretation": "comparison_only_proxy_metrics",
        },
        "records": records,
        "summary": summary,
    }
    write_json(OUTPUT_PATH, payload)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(_render_report(summary), encoding="utf-8")


def _make_reduced_fixture(name: str) -> Any:
    fixture = make_nonlinear_fixture(name)
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


def _run_advanced_edh_pfpf(fixture: Any, seed: int) -> dict[str, Any]:
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
                seed=seed,
            )
            result = filt.filter(
                model,
                fixture.observations,
                return_particles=False,
                return_diagnostics=True,
                rng=np.random.default_rng(seed),
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
            "base_fixture_name": _base_fixture_name(fixture.name),
            "method": "EDHParticleFilter",
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": seed,
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
                "ess_semantics": "post_filter_result_ess",
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
            seed=seed,
            num_particles=NUM_PARTICLES,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update(
            {
                "base_fixture_name": _base_fixture_name(fixture.name),
                "method": "EDHParticleFilter",
                "target": fixture.target,
                "flow_steps": FLOW_STEPS,
                "metrics": {},
                "diagnostics": {
                    "target_label": "advanced/EDHParticleFilter/gaussian_range_bearing",
                    "selected_by": "MP4 flow and DPF readiness review",
                    "runtime_warning": False,
                },
            }
        )
        return record


def _run_mlcoe_pfpf_edh(fixture: Any, seed: int) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with prepend_sys_path(MLCOE_SNAPSHOT_ROOT):
            import tensorflow as tf  # type: ignore
            from src.filters.classical import DTYPE  # type: ignore
            from src.filters.flow_filters import PFPF_EDH  # type: ignore

            np.random.seed(seed)
            tf.random.set_seed(seed)
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
            "base_fixture_name": _base_fixture_name(fixture.name),
            "method": "PFPF_EDH",
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": seed,
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
                "ess_semantics": "post_step_filter_ess",
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
            seed=seed,
            num_particles=NUM_PARTICLES,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update(
            {
                "base_fixture_name": _base_fixture_name(fixture.name),
                "method": "PFPF_EDH",
                "target": fixture.target,
                "flow_steps": FLOW_STEPS,
                "metrics": {},
                "diagnostics": {
                    "target_label": "mlcoe/PFPF_EDH/gaussian_range_bearing",
                    "selected_by": "MP4 flow and DPF readiness review",
                    "runtime_warning": False,
                },
            }
        )
        return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    by_impl = _group_by(records, "implementation_name")
    by_impl_fixture = _group_by(records, "implementation_name", "base_fixture_name")
    impl_summary = {
        impl: _summarize_group(group) for impl, group in sorted(by_impl.items())
    }
    fixture_summary = {
        f"{impl}/{fixture}": _summarize_group(group)
        for (impl, fixture), group in sorted(by_impl_fixture.items())
    }
    pressure = _low_noise_pressure(fixture_summary)
    decision = _decision(records, pressure)
    return {
        "records": len(records),
        "planned_records": PLANNED_RECORDS,
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "implementation_summary": impl_summary,
        "fixture_summary": fixture_summary,
        "low_noise_pressure": pressure,
        "hypothesis_results": {
            "R1_both_paths_remain_runnable": _interpret_r1(records),
            "R2_low_observation_noise_increases_pressure": _interpret_r2(pressure),
            "R3_proxy_comparison_remains_interpretable": _interpret_r3(records),
            "R4_runtime_remains_bounded": _interpret_r4(records),
            "R5_next_baseline_decision": decision,
        },
        "decision": decision,
        "next_phase_recommendation": _next_phase_recommendation(decision),
    }


def _group_by(
    records: list[dict[str, Any]], *keys: str
) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(k) for k in keys)
        if len(key) == 1:
            key = key[0]
        grouped[key].append(record)
    return grouped


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    return {
        "runs": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "failure_reasons": sorted(
            {
                str(record.get("failure_reason"))
                for record in records
                if record.get("failure_reason")
            }
        ),
        "median_runtime_seconds": _median(record.get("runtime_seconds") for record in ok),
        "max_runtime_seconds": _max(record.get("runtime_seconds") for record in ok),
        "runtime_warning_count": sum(
            1 for record in ok if record.get("diagnostics", {}).get("runtime_warning")
        ),
        "median_state_rmse": _median(
            record.get("metrics", {}).get("state_rmse") for record in ok
        ),
        "median_position_rmse": _median(
            record.get("metrics", {}).get("position_rmse") for record in ok
        ),
        "median_final_position_error": _median(
            record.get("metrics", {}).get("final_position_error") for record in ok
        ),
        "median_observation_proxy_rmse": _median(
            record.get("metrics", {}).get("observation_proxy_rmse") for record in ok
        ),
        "median_average_ess": _median(
            record.get("metrics", {}).get("average_ess") for record in ok
        ),
        "min_average_ess": _min(
            record.get("metrics", {}).get("average_ess") for record in ok
        ),
        "median_min_ess": _median(
            record.get("metrics", {}).get("min_ess") for record in ok
        ),
        "minimum_min_ess": _min(
            record.get("metrics", {}).get("min_ess") for record in ok
        ),
        "median_resampling_count": _median(
            record.get("diagnostics", {}).get("resampling_count") for record in ok
        ),
        "max_resampling_count": _max(
            record.get("diagnostics", {}).get("resampling_count") for record in ok
        ),
        "finite_means_count": sum(
            1 for record in ok if record.get("diagnostics", {}).get("finite_means")
        ),
        "resampling_count_semantics": sorted(
            {
                str(record.get("diagnostics", {}).get("resampling_count_semantics"))
                for record in ok
                if record.get("diagnostics", {}).get("resampling_count_semantics")
            }
        ),
        "ess_semantics": sorted(
            {
                str(record.get("diagnostics", {}).get("ess_semantics"))
                for record in ok
                if record.get("diagnostics", {}).get("ess_semantics")
            }
        ),
    }


def _low_noise_pressure(fixture_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        moderate = fixture_summary.get(f"{impl}/range_bearing_gaussian_moderate")
        low = fixture_summary.get(f"{impl}/range_bearing_gaussian_low_noise")
        if not moderate or not low or moderate["ok"] == 0 or low["ok"] == 0:
            results[impl] = {"status": "not_available"}
            continue
        signals = {
            "median_average_ess_decreased": _lt(
                low.get("median_average_ess"), moderate.get("median_average_ess")
            ),
            "minimum_min_ess_decreased": _lt(
                low.get("minimum_min_ess"), moderate.get("minimum_min_ess")
            ),
            "median_resampling_count_increased": _gt(
                low.get("median_resampling_count"),
                moderate.get("median_resampling_count"),
            ),
            "median_runtime_seconds_increased": _gt(
                low.get("median_runtime_seconds"),
                moderate.get("median_runtime_seconds"),
            ),
            "median_position_rmse_increased": _gt(
                low.get("median_position_rmse"),
                moderate.get("median_position_rmse"),
            ),
            "median_observation_proxy_rmse_increased": _gt(
                low.get("median_observation_proxy_rmse"),
                moderate.get("median_observation_proxy_rmse"),
            ),
        }
        results[impl] = {
            "status": "available",
            "moderate": _pressure_metrics(moderate),
            "low_noise": _pressure_metrics(low),
            "signals": signals,
            "any_pressure_signal": any(value is True for value in signals.values()),
        }
    return results


def _pressure_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "median_average_ess": summary.get("median_average_ess"),
        "minimum_min_ess": summary.get("minimum_min_ess"),
        "median_resampling_count": summary.get("median_resampling_count"),
        "median_runtime_seconds": summary.get("median_runtime_seconds"),
        "median_position_rmse": summary.get("median_position_rmse"),
        "median_observation_proxy_rmse": summary.get("median_observation_proxy_rmse"),
    }


def _decision(records: list[dict[str, Any]], pressure: dict[str, Any]) -> str:
    if len(records) != PLANNED_RECORDS:
        return "blocked_or_excluded"
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    if not ok:
        return "blocked_or_excluded"
    if _fixture_all_failures(records):
        return "needs_targeted_debug"
    if _nonfinite_ok_records(ok):
        return "needs_targeted_debug"
    if _runtime_warning_count(ok) > 0:
        return "replicated_panel_ready_with_caveats"
    if any(item.get("status") == "not_available" for item in pressure.values()):
        return "replicated_panel_ready_with_caveats"
    if not any(item.get("any_pressure_signal") for item in pressure.values()):
        return "replicated_panel_ready_with_caveats"
    if len(ok) == len(records):
        return "replicated_panel_ready"
    return "replicated_panel_ready_with_caveats"


def _interpret_r1(records: list[dict[str, Any]]) -> str:
    if len(records) != PLANNED_RECORDS:
        return f"blocked_expected_{PLANNED_RECORDS}_records_observed_{len(records)}"
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        for fixture in FIXTURE_NAMES:
            group = [
                record
                for record in records
                if record.get("implementation_name") == impl
                and record.get("base_fixture_name") == fixture
            ]
            ok = [
                record
                for record in group
                if record["status"] == str(BaselineStatus.OK)
            ]
            if not ok:
                return f"blocked_no_ok_runs_for_{impl}_{fixture}"
    failures = [record for record in records if record["status"] != str(BaselineStatus.OK)]
    if failures:
        return "partially_supported_structured_blockers_recorded"
    return "supported_all_planned_runs_ok"


def _interpret_r2(pressure: dict[str, Any]) -> str:
    available = [
        item for item in pressure.values() if item.get("status") == "available"
    ]
    if not available:
        return "blocked_no_fixture_pressure_summary"
    if any(item.get("any_pressure_signal") for item in available):
        return "supported_directional_pressure_signal_observed"
    return "not_supported_no_directional_pressure_signal_observed"


def _interpret_r3(records: list[dict[str, Any]]) -> str:
    required_metrics = {
        "state_rmse",
        "position_rmse",
        "final_position_error",
        "observation_proxy_rmse",
    }
    required_record = {
        "fixture_name",
        "base_fixture_name",
        "seed",
        "num_particles",
        "flow_steps",
        "target",
        "runtime_seconds",
    }
    for record in records:
        if not required_record.issubset(record):
            return "blocked_missing_record_fields"
        diagnostics = record.get("diagnostics", {})
        if "target_label" not in diagnostics:
            return "blocked_missing_target_label"
        if record["status"] == str(BaselineStatus.OK):
            if not required_metrics.issubset(record.get("metrics", {})):
                return "blocked_missing_proxy_metrics"
            if "resampling_count_semantics" not in diagnostics:
                return "blocked_missing_resampling_semantics"
            if "ess_semantics" not in diagnostics:
                return "blocked_missing_ess_semantics"
    return "supported_proxy_only"


def _interpret_r4(records: list[dict[str, Any]]) -> str:
    warnings = _runtime_warning_count(records)
    if warnings == 0:
        return "supported_no_runtime_warnings"
    return f"supported_with_{warnings}_runtime_warnings"


def _next_phase_recommendation(decision: str) -> str:
    if decision == "replicated_panel_ready":
        return (
            "Use the replicated EDH/PFPF panel as a quarantined experimental "
            "baseline artifact, then test whether the same adapter discipline "
            "can support a controlled full-horizon sensitivity panel."
        )
    if decision == "replicated_panel_ready_with_caveats":
        return (
            "Keep the panel as a caveated experimental baseline.  Run targeted "
            "debug on the caveat before expanding horizon, particles, or flow "
            "families."
        )
    if decision == "needs_targeted_debug":
        return (
            "Do not expand the panel.  Run targeted debug on failed fixtures, "
            "nonfinite outputs, or runtime warnings."
        )
    return "Stop EDH/PFPF panel expansion until the blocker is classified."


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
        "# Student DPF baseline replicated EDH/PFPF panel result",
        "",
        "## Date",
        "",
        DATE,
        "",
        "## Scope",
        "",
        "This report covers the replicated EDH/PFPF panel in the quarantined",
        "student DPF experimental-baseline stream.  It is comparison-only",
        "evidence and does not promote student code, modify vendored snapshots,",
        "or make production BayesFilter claims.",
        "",
        "## Command",
        "",
        f"`{COMMAND}`",
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
        f"- base fixtures: `{', '.join(FIXTURE_NAMES)}`",
        f"- reduced horizon: `{REDUCED_HORIZON}`",
        f"- seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        f"- particles: `{NUM_PARTICLES}`",
        f"- flow steps: `{FLOW_STEPS}`",
        f"- runtime warning threshold seconds: `{RUNTIME_WARNING_SECONDS:g}`",
        f"- planned records: `{PLANNED_RECORDS}`",
        "",
        "## Implementation Summary",
        "",
        "| Implementation | Runs | OK | Failed | Median runtime seconds | Max runtime seconds | Median position RMSE | Median avg ESS | Runtime warnings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for impl, data in summary["implementation_summary"].items():
        lines.append(
            f"| {impl} | {data['runs']} | {data['ok']} | {data['failed']} | "
            f"{_fmt(data['median_runtime_seconds'])} | "
            f"{_fmt(data['max_runtime_seconds'])} | "
            f"{_fmt(data['median_position_rmse'])} | "
            f"{_fmt(data['median_average_ess'])} | "
            f"{data['runtime_warning_count']} |"
        )

    lines.extend(
        [
            "",
            "## Fixture Summary",
            "",
            "| Implementation / fixture | Runs | OK | Median state RMSE | Median position RMSE | Median final-position error | Median obs proxy RMSE | Median avg ESS | Minimum min ESS | Median resampling count | Median runtime seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in summary["fixture_summary"].items():
        lines.append(
            f"| {key} | {data['runs']} | {data['ok']} | "
            f"{_fmt(data['median_state_rmse'])} | "
            f"{_fmt(data['median_position_rmse'])} | "
            f"{_fmt(data['median_final_position_error'])} | "
            f"{_fmt(data['median_observation_proxy_rmse'])} | "
            f"{_fmt(data['median_average_ess'])} | "
            f"{_fmt(data['minimum_min_ess'])} | "
            f"{_fmt(data['median_resampling_count'])} | "
            f"{_fmt(data['median_runtime_seconds'])} |"
        )

    lines.extend(["", "## Low-Noise Pressure", ""])
    for impl, data in summary["low_noise_pressure"].items():
        lines.append(f"### {impl}")
        lines.append("")
        if data.get("status") != "available":
            lines.append("Pressure summary not available.")
            lines.append("")
            continue
        lines.append(f"- any pressure signal: `{data['any_pressure_signal']}`")
        for signal, value in data["signals"].items():
            lines.append(f"- `{signal}`: `{value}`")
        lines.append("")

    lines.extend(["## Hypothesis Results", ""])
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
            "The panel result is proxy evidence only.  Latent-state and position",
            "RMSE are evaluated against the shared simulated fixtures.  ESS and",
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


def _base_fixture_name(fixture_name: str) -> str:
    suffix = f"_edh_pfpf_h{REDUCED_HORIZON}"
    if fixture_name.endswith(suffix):
        return fixture_name[: -len(suffix)]
    return fixture_name


def _fixture_all_failures(records: list[dict[str, Any]]) -> bool:
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        for fixture in FIXTURE_NAMES:
            group = [
                record
                for record in records
                if record.get("implementation_name") == impl
                and record.get("base_fixture_name") == fixture
            ]
            if group and all(
                record["status"] != str(BaselineStatus.OK) for record in group
            ):
                return True
    return False


def _nonfinite_ok_records(records: list[dict[str, Any]]) -> bool:
    return any(
        not record.get("diagnostics", {}).get("finite_means") for record in records
    )


def _runtime_warning_count(records: list[dict[str, Any]]) -> int:
    return sum(
        1
        for record in records
        if record.get("diagnostics", {}).get("runtime_warning")
    )


def _clean(values: Any) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(float(v))]


def _min(values: Any) -> float | None:
    vals = _clean(values)
    return min(vals) if vals else None


def _max(values: Any) -> float | None:
    vals = _clean(values)
    return max(vals) if vals else None


def _median(values: Any) -> float | None:
    vals = _clean(values)
    return float(np.median(vals)) if vals else None


def _lt(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return bool(float(left) < float(right))


def _gt(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return bool(float(left) > float(right))


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


if __name__ == "__main__":
    main()
