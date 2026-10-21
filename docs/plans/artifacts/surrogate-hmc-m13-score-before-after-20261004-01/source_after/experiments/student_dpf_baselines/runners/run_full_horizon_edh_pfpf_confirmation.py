"""Run the full-horizon EDH/PFPF confirmation panel."""

from __future__ import annotations

from collections import defaultdict
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
    "experiments.student_dpf_baselines.runners.run_full_horizon_edh_pfpf_confirmation"
)
OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "full_horizon_edh_pfpf_confirmation_2026-05-12.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "full_horizon_edh_pfpf_confirmation_summary_2026-05-12.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/"
    "student-dpf-baseline-full-horizon-edh-pfpf-confirmation-result-2026-05-12.md"
)

FIXTURE_NAMES = [
    "range_bearing_gaussian_moderate",
    "range_bearing_gaussian_low_noise",
]
SEEDS = [31, 43, 59, 71, 83]
NUM_PARTICLES = 128
FLOW_STEPS_BY_FIXTURE = {
    "range_bearing_gaussian_moderate": [10, 20],
    "range_bearing_gaussian_low_noise": [20],
}
RUNTIME_WARNING_SECONDS = 45.0
PLANNED_RECORDS = (
    sum(len(FLOW_STEPS_BY_FIXTURE[name]) for name in FIXTURE_NAMES)
    * len(SEEDS)
    * 2
)
LOW_NOISE_REFERENCE = {
    "advanced_particle_filter": {
        "median_average_ess": 29.6,
        "median_position_rmse": 0.0466,
        "median_observation_proxy_rmse": 0.0165,
        "median_resampling_count": 18.0,
    },
    "2026MLCOE": {
        "median_average_ess": 43.8,
        "median_position_rmse": 0.0480,
        "median_observation_proxy_rmse": 0.0166,
        "median_resampling_count": 17.0,
    },
}


def main() -> None:
    fixtures = [make_nonlinear_fixture(name) for name in FIXTURE_NAMES]
    records = []
    for fixture in fixtures:
        for seed in SEEDS:
            for flow_steps in FLOW_STEPS_BY_FIXTURE[fixture.name]:
                records.append(
                    _run_advanced_edh_pfpf(
                        fixture,
                        seed=seed,
                        num_particles=NUM_PARTICLES,
                        flow_steps=flow_steps,
                    )
                )
                records.append(
                    _run_mlcoe_pfpf_edh(
                        fixture,
                        seed=seed,
                        num_particles=NUM_PARTICLES,
                        flow_steps=flow_steps,
                    )
                )
    summary = _summarize(records)
    payload = {
        "date": DATE,
        "command": COMMAND,
        "working_directory": str(Path.cwd()),
        "environment": _environment_record(),
        "panel": {
            "fixtures": FIXTURE_NAMES,
            "horizon": "full_fixture_horizon",
            "seeds": SEEDS,
            "num_particles": NUM_PARTICLES,
            "flow_steps_by_fixture": FLOW_STEPS_BY_FIXTURE,
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


def _run_advanced_edh_pfpf(
    fixture: Any, *, seed: int, num_particles: int, flow_steps: int
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        model = _make_advanced_model(fixture)
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.filters import EDHParticleFilter  # type: ignore

            filt = EDHParticleFilter(
                n_particles=num_particles,
                n_flow_steps=flow_steps,
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
            "method": "EDHParticleFilter",
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": seed,
            "num_particles": num_particles,
            "flow_steps": flow_steps,
            "horizon": fixture.horizon,
            "runtime_seconds": runtime,
            "metrics": metrics,
            "diagnostics": {
                "target_label": "advanced/EDHParticleFilter/gaussian_range_bearing",
                "selected_by": "full-horizon EDH/PFPF confirmation",
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
            num_particles=num_particles,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update(
            {
                "method": "EDHParticleFilter",
                "target": fixture.target,
                "flow_steps": flow_steps,
                "horizon": fixture.horizon,
                "metrics": {},
                "diagnostics": {
                    "target_label": "advanced/EDHParticleFilter/gaussian_range_bearing",
                    "selected_by": "full-horizon EDH/PFPF confirmation",
                    "runtime_warning": False,
                },
            }
        )
        return record


def _run_mlcoe_pfpf_edh(
    fixture: Any, *, seed: int, num_particles: int, flow_steps: int
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with prepend_sys_path(MLCOE_SNAPSHOT_ROOT):
            import tensorflow as tf  # type: ignore
            from src.filters.classical import DTYPE  # type: ignore
            from src.filters.flow_filters import PFPF_EDH  # type: ignore

            np.random.seed(seed)
            tf.random.set_seed(seed)
            model = _MlcoeRangeBearingModel(fixture, tf=tf, dtype=DTYPE)
            filt = PFPF_EDH(model, N=num_particles, steps=flow_steps)
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
                inferred_resampling.append(ess < num_particles / 2.0)
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
            "seed": seed,
            "num_particles": num_particles,
            "flow_steps": flow_steps,
            "horizon": fixture.horizon,
            "runtime_seconds": runtime,
            "metrics": metrics,
            "diagnostics": {
                "target_label": "mlcoe/PFPF_EDH/gaussian_range_bearing",
                "selected_by": "full-horizon EDH/PFPF confirmation",
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
            num_particles=num_particles,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update(
            {
                "method": "PFPF_EDH",
                "target": fixture.target,
                "flow_steps": flow_steps,
                "horizon": fixture.horizon,
                "metrics": {},
                "diagnostics": {
                    "target_label": "mlcoe/PFPF_EDH/gaussian_range_bearing",
                    "selected_by": "full-horizon EDH/PFPF confirmation",
                    "runtime_warning": False,
                },
            }
        )
        return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    by_impl = _group_by(records, "implementation_name")
    by_fixture = _group_by(records, "implementation_name", "fixture_name")
    by_grid = _group_by(records, "implementation_name", "fixture_name", "flow_steps")
    impl_summary = {
        impl: _summarize_group(group) for impl, group in sorted(by_impl.items())
    }
    fixture_summary = {
        f"{impl}/{fixture}": _summarize_group(group)
        for (impl, fixture), group in sorted(by_fixture.items())
    }
    grid_summary = {
        _grid_key(impl, fixture, steps): _summarize_group(group)
        for (impl, fixture, steps), group in sorted(by_grid.items())
    }
    low_noise_confirmation = _low_noise_confirmation(grid_summary)
    moderate_policy = _moderate_policy(grid_summary)
    clean_room_spec = _clean_room_spec(moderate_policy)
    decision = _decision(records, low_noise_confirmation, moderate_policy)
    return {
        "records": len(records),
        "planned_records": PLANNED_RECORDS,
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "implementation_summary": impl_summary,
        "fixture_summary": fixture_summary,
        "grid_summary": grid_summary,
        "low_noise_confirmation": low_noise_confirmation,
        "moderate_flow_step_policy": moderate_policy,
        "clean_room_specification_inputs": clean_room_spec,
        "hypothesis_results": {
            "C1_selected_full_horizon_setting_seed_stable": _interpret_c1(records),
            "C2_low_noise_128_particle_pressure_reduction_persists": _interpret_c2(
                low_noise_confirmation
            ),
            "C3_moderate_noise_flow_step_policy_resolved": _interpret_c3(
                moderate_policy
            ),
            "C4_clean_room_baseline_specification_ready": _interpret_c4(
                clean_room_spec
            ),
            "C5_next_baseline_decision": decision,
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
        "ess_semantics": sorted(
            {
                str(record.get("diagnostics", {}).get("ess_semantics"))
                for record in ok
                if record.get("diagnostics", {}).get("ess_semantics")
            }
        ),
        "resampling_count_semantics": sorted(
            {
                str(record.get("diagnostics", {}).get("resampling_count_semantics"))
                for record in ok
                if record.get("diagnostics", {}).get("resampling_count_semantics")
            }
        ),
    }


def _low_noise_confirmation(grid_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    fixture = "range_bearing_gaussian_low_noise"
    steps = 20
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        summary = grid_summary.get(_grid_key(impl, fixture, steps))
        reference = LOW_NOISE_REFERENCE[impl]
        if not summary or summary["ok"] == 0:
            result[impl] = {"status": "not_available", "reference": reference}
            continue
        thresholds = {
            "median_average_ess_floor": 0.75 * reference["median_average_ess"],
            "median_position_rmse_ceiling": 1.5 * reference["median_position_rmse"],
            "median_observation_proxy_rmse_ceiling": (
                1.5 * reference["median_observation_proxy_rmse"]
            ),
            "median_resampling_count_ceiling": 20.0,
        }
        checks = {
            "median_average_ess_not_materially_worse": _ge(
                summary.get("median_average_ess"),
                thresholds["median_average_ess_floor"],
            ),
            "median_position_rmse_not_materially_worse": _le(
                summary.get("median_position_rmse"),
                thresholds["median_position_rmse_ceiling"],
            ),
            "median_observation_proxy_rmse_not_materially_worse": _le(
                summary.get("median_observation_proxy_rmse"),
                thresholds["median_observation_proxy_rmse_ceiling"],
            ),
            "median_resampling_count_within_horizon": _le(
                summary.get("median_resampling_count"),
                thresholds["median_resampling_count_ceiling"],
            ),
            "finite_outputs_dominant": summary.get("finite_means_count", 0)
            >= max(1, int(0.9 * summary.get("runs", 0))),
        }
        result[impl] = {
            "status": "available",
            "summary": _key_metrics(summary),
            "reference": reference,
            "thresholds": thresholds,
            "checks": checks,
            "confirmed": all(value is True for value in checks.values()),
        }
    return result


def _moderate_policy(grid_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fixture = "range_bearing_gaussian_moderate"
    details = {}
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        steps_10 = grid_summary.get(_grid_key(impl, fixture, 10))
        steps_20 = grid_summary.get(_grid_key(impl, fixture, 20))
        if not steps_10 or not steps_20 or steps_10["ok"] == 0 or steps_20["ok"] == 0:
            details[impl] = {"status": "not_available"}
            continue
        runtime_ratio = _safe_ratio(
            steps_20.get("median_runtime_seconds"),
            steps_10.get("median_runtime_seconds"),
        )
        position_improved = _lt(
            steps_20.get("median_position_rmse"),
            steps_10.get("median_position_rmse"),
        )
        observation_improved = _lt(
            steps_20.get("median_observation_proxy_rmse"),
            steps_10.get("median_observation_proxy_rmse"),
        )
        bounded_runtime = None if runtime_ratio is None else runtime_ratio <= 2.5
        bounded_benefit = (
            (position_improved is True or observation_improved is True)
            and bounded_runtime is True
        )
        details[impl] = {
            "status": "available",
            "steps_10": _key_metrics(steps_10),
            "steps_20": _key_metrics(steps_20),
            "runtime_ratio_20_over_10": runtime_ratio,
            "position_rmse_improved": position_improved,
            "observation_proxy_rmse_improved": observation_improved,
            "bounded_runtime": bounded_runtime,
            "bounded_benefit": bounded_benefit,
        }
    available = [item for item in details.values() if item.get("status") == "available"]
    if len(available) != 2:
        recommendation = "moderate_keep_both_as_diagnostic"
        rationale = "missing implementation comparison"
    elif all(item["bounded_benefit"] for item in available):
        recommendation = "moderate_use_20_steps"
        rationale = "both implementations show bounded benefit from 20 steps"
    elif not any(item["bounded_benefit"] for item in available):
        recommendation = "moderate_use_10_steps"
        rationale = "neither implementation shows bounded benefit from 20 steps"
    else:
        recommendation = "moderate_keep_both_as_diagnostic"
        rationale = "implementation-specific benefit differs"
    return {
        "fixture": fixture,
        "num_particles": NUM_PARTICLES,
        "details": details,
        "recommendation": recommendation,
        "rationale": rationale,
    }


def _clean_room_spec(moderate_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready_for_specification",
        "fixtures": FIXTURE_NAMES,
        "horizon": "full_fixture_horizon_20_observations",
        "seed_policy": "use independent fixed seeds and report seed list",
        "particle_count": NUM_PARTICLES,
        "flow_step_policy": {
            "low_noise": 20,
            "moderate": moderate_policy.get("recommendation"),
        },
        "metrics": [
            "state_rmse",
            "position_rmse",
            "final_position_error",
            "observation_proxy_rmse",
            "average_ess",
            "min_ess",
            "resampling_count",
            "runtime_seconds",
            "finite_means",
        ],
        "caveats": [
            "comparison_only",
            "student_code_not_production",
            "student_agreement_not_correctness",
            "ess_and_resampling_semantics_implementation_specific",
        ],
        "exclusions": [
            "copying_student_code",
            "production_bayesfilter_changes",
            "hmc_claims",
            "monograph_lane_changes",
            "kernel_pff",
            "neural_ot",
            "differentiable_resampling",
        ],
    }


def _decision(
    records: list[dict[str, Any]],
    low_noise_confirmation: dict[str, Any],
    moderate_policy: dict[str, Any],
) -> str:
    if len(records) != PLANNED_RECORDS:
        return "blocked_or_excluded"
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    if len(ok) < int(0.9 * PLANNED_RECORDS):
        return "needs_targeted_debug"
    if _fixture_all_failures(records):
        return "needs_targeted_debug"
    if _nonfinite_ok_records(ok):
        return "needs_targeted_debug"
    if _runtime_warning_count(ok) > 0:
        return "confirmation_ready_with_caveats"
    if not all(item.get("confirmed") for item in low_noise_confirmation.values()):
        return "confirmation_ready_with_caveats"
    if moderate_policy.get("recommendation") == "moderate_keep_both_as_diagnostic":
        return "confirmation_ready_with_caveats"
    return "confirmation_ready_for_clean_room_spec"


def _interpret_c1(records: list[dict[str, Any]]) -> str:
    if len(records) != PLANNED_RECORDS:
        return f"blocked_expected_{PLANNED_RECORDS}_records_observed_{len(records)}"
    ok = [record for record in records if record["status"] == str(BaselineStatus.OK)]
    if len(ok) < int(0.9 * PLANNED_RECORDS):
        return "not_supported_less_than_90_percent_ok"
    if _fixture_all_failures(records):
        return "blocked_fixture_all_failures"
    if _nonfinite_ok_records(ok):
        return "not_supported_nonfinite_ok_records"
    return "supported_seed_stable"


def _interpret_c2(low_noise_confirmation: dict[str, Any]) -> str:
    if not low_noise_confirmation:
        return "blocked_no_low_noise_summary"
    if all(item.get("confirmed") for item in low_noise_confirmation.values()):
        return "supported_low_noise_pressure_reduction_persists"
    return "supported_with_caveats_low_noise_material_degradation_detected"


def _interpret_c3(moderate_policy: dict[str, Any]) -> str:
    recommendation = moderate_policy.get("recommendation")
    if recommendation in {
        "moderate_use_10_steps",
        "moderate_use_20_steps",
        "moderate_keep_both_as_diagnostic",
    }:
        return f"resolved_{recommendation}"
    return "blocked_no_moderate_policy"


def _interpret_c4(clean_room_spec: dict[str, Any]) -> str:
    required = {
        "fixtures",
        "horizon",
        "seed_policy",
        "particle_count",
        "flow_step_policy",
        "metrics",
        "caveats",
        "exclusions",
    }
    if required.issubset(clean_room_spec):
        return "supported_clean_room_inputs_ready"
    return "blocked_missing_clean_room_inputs"


def _next_phase_recommendation(decision: str) -> str:
    if decision == "confirmation_ready_for_clean_room_spec":
        return (
            "Write a clean-room controlled-baseline specification that uses the "
            "confirmed fixtures, metrics, seed policy, and flow-step policy "
            "without copying student implementation code."
        )
    if decision == "confirmation_ready_with_caveats":
        return (
            "Write a caveated clean-room specification only after explicitly "
            "carrying forward the low-noise or moderate-flow-step caveats."
        )
    if decision == "needs_targeted_debug":
        return (
            "Do not proceed to clean-room specification.  Debug failed, "
            "nonfinite, or runtime-warning groups first."
        )
    return "Stop confirmation expansion until blockers are classified."


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
        "# Student DPF baseline full-horizon EDH/PFPF confirmation result",
        "",
        "## Date",
        "",
        DATE,
        "",
        "## Scope",
        "",
        "This report covers the full-horizon EDH/PFPF confirmation panel in the",
        "quarantined student DPF experimental-baseline stream.  It is",
        "comparison-only evidence and does not promote student code, modify",
        "vendored snapshots, or make production BayesFilter claims.",
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
        f"- fixtures: `{', '.join(FIXTURE_NAMES)}`",
        "- horizon: full fixture horizon, 20 observations",
        f"- seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        f"- particles: `{NUM_PARTICLES}`",
        "- low-noise flow steps: `20`",
        "- moderate-noise flow steps: `10, 20`",
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
            "## Grid Summary",
            "",
            "| Implementation / fixture / steps | Runs | OK | Median position RMSE | Median obs proxy RMSE | Median avg ESS | Minimum min ESS | Median resampling count | Median runtime seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in summary["grid_summary"].items():
        lines.append(
            f"| {key} | {data['runs']} | {data['ok']} | "
            f"{_fmt(data['median_position_rmse'])} | "
            f"{_fmt(data['median_observation_proxy_rmse'])} | "
            f"{_fmt(data['median_average_ess'])} | "
            f"{_fmt(data['minimum_min_ess'])} | "
            f"{_fmt(data['median_resampling_count'])} | "
            f"{_fmt(data['median_runtime_seconds'])} |"
        )

    lines.extend(["", "## Low-Noise Confirmation", ""])
    for impl, data in summary["low_noise_confirmation"].items():
        lines.append(f"### {impl}")
        lines.append("")
        if data.get("status") != "available":
            lines.append("Low-noise confirmation is not available.")
            lines.append("")
            continue
        lines.append(f"- confirmed: `{data['confirmed']}`")
        for key, value in data["checks"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    policy = summary["moderate_flow_step_policy"]
    lines.extend(
        [
            "## Moderate-Noise Flow-Step Policy",
            "",
            f"- recommendation: `{policy['recommendation']}`",
            f"- rationale: {policy['rationale']}",
            "",
        ]
    )
    for impl, data in policy["details"].items():
        if data.get("status") != "available":
            lines.append(f"- `{impl}`: not available")
            continue
        lines.append(
            f"- `{impl}` bounded benefit: `{data['bounded_benefit']}`, "
            f"runtime ratio 20/10: `{_fmt(data['runtime_ratio_20_over_10'])}`, "
            f"position improved: `{data['position_rmse_improved']}`, "
            f"obs proxy improved: `{data['observation_proxy_rmse_improved']}`"
        )

    lines.extend(["", "## Clean-Room Specification Inputs", ""])
    spec = summary["clean_room_specification_inputs"]
    lines.append(f"- status: `{spec['status']}`")
    lines.append(f"- fixtures: `{', '.join(spec['fixtures'])}`")
    lines.append(f"- horizon: `{spec['horizon']}`")
    lines.append(f"- particle count: `{spec['particle_count']}`")
    lines.append(f"- flow-step policy: `{spec['flow_step_policy']}`")

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
            "The confirmation result is proxy evidence only.  Latent-state and",
            "position RMSE are evaluated against the shared simulated fixtures.",
            "ESS and resampling semantics are implementation-specific",
            "diagnostics.  The report does not use student agreement or",
            "likelihood values as correctness evidence.",
            "",
            "## Next Phase Recommendation",
            "",
            summary["next_phase_recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def _key_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "median_runtime_seconds": summary.get("median_runtime_seconds"),
        "median_position_rmse": summary.get("median_position_rmse"),
        "median_observation_proxy_rmse": summary.get("median_observation_proxy_rmse"),
        "median_average_ess": summary.get("median_average_ess"),
        "minimum_min_ess": summary.get("minimum_min_ess"),
        "median_resampling_count": summary.get("median_resampling_count"),
        "runtime_warning_count": summary.get("runtime_warning_count"),
        "finite_means_count": summary.get("finite_means_count"),
        "runs": summary.get("runs"),
        "ok": summary.get("ok"),
    }


def _grid_key(impl: Any, fixture: Any, steps: Any) -> str:
    return f"{impl}/{fixture}/N{NUM_PARTICLES}/steps{steps}"


def _fixture_all_failures(records: list[dict[str, Any]]) -> bool:
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        for fixture in FIXTURE_NAMES:
            group = [
                record
                for record in records
                if record.get("implementation_name") == impl
                and record.get("fixture_name") == fixture
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


def _le(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return bool(float(left) <= float(right))


def _ge(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return bool(float(left) >= float(right))


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


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
