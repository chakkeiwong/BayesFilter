"""Run the MP2 nonlinear reference/proxy-metric panel."""

from __future__ import annotations

from collections import defaultdict
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
from experiments.student_dpf_baselines.fixtures.nonlinear_fixtures import (
    RangeBearingFixture,
    make_nonlinear_fixture,
    nonlinear_fixture_names,
    observation_residual,
    range_bearing_jacobian,
    range_bearing_observation,
)


OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/nonlinear_reference_panel_2026-05-10.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/nonlinear_reference_panel_summary_2026-05-10.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/student-dpf-baseline-nonlinear-reference-panel-result-2026-05-10.md"
)

PF_SEEDS = [0, 1, 2, 3, 4]
NUM_PARTICLES = 256


def main() -> None:
    records = []
    for fixture_name in nonlinear_fixture_names():
        fixture = make_nonlinear_fixture(fixture_name)
        records.extend(_run_advanced_methods(fixture))
        records.extend(_run_mlcoe_methods(fixture))
    summary = summarize(records)
    panel = {
        "date": "2026-05-10",
        "panel": {
            "fixtures": nonlinear_fixture_names(),
            "pf_seeds": PF_SEEDS,
            "num_particles": NUM_PARTICLES,
            "target": "gaussian_range_bearing",
        },
        "records": records,
        "summary": summary,
    }
    write_json(OUTPUT_PATH, panel)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_impl: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_method: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_impl[record["implementation_name"]].append(record)
        by_method[
            (
                record["implementation_name"],
                record["fixture_name"],
                record["method"],
            )
        ].append(record)

    implementation_summary = {}
    for impl, group in sorted(by_impl.items()):
        ok = [r for r in group if r["status"] == "ok"]
        implementation_summary[impl] = {
            "runs": len(group),
            "ok": len(ok),
            "failed": len(group) - len(ok),
            "median_runtime_seconds": _median(r.get("runtime_seconds") for r in ok),
            "min_position_rmse": _min(r["metrics"].get("position_rmse") for r in ok),
            "max_position_rmse": _max(r["metrics"].get("position_rmse") for r in ok),
        }

    method_summary = {}
    for (impl, fixture, method), group in sorted(by_method.items()):
        ok = [r for r in group if r["status"] == "ok"]
        method_summary[f"{impl}/{fixture}/{method}"] = {
            "runs": len(group),
            "ok": len(ok),
            "failed": len(group) - len(ok),
            "median_state_rmse": _median(r["metrics"].get("state_rmse") for r in ok),
            "median_position_rmse": _median(
                r["metrics"].get("position_rmse") for r in ok
            ),
            "median_final_position_error": _median(
                r["metrics"].get("final_position_error") for r in ok
            ),
            "median_average_ess": _median(
                r["metrics"].get("average_ess") for r in ok
            ),
            "min_average_ess": _min(r["metrics"].get("average_ess") for r in ok),
            "median_runtime_seconds": _median(r.get("runtime_seconds") for r in ok),
        }

    comparison_summary = _summarize_comparisons(records)
    return {
        "implementation_summary": implementation_summary,
        "method_summary": method_summary,
        "comparison_summary": comparison_summary,
        "hypothesis_results": {
            "n1_shared_nonlinear_fixture": _interpret_n1(by_impl),
            "n2_mlcoe_ekf_zero_behavior": _interpret_n2(records),
            "n3_nonlinear_pf_degeneracy": _interpret_n3(method_summary),
            "n4_comparison_only_reporting": _interpret_n4(records),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Student DPF baseline MP2 nonlinear reference-panel result",
        "",
        "## Date",
        "",
        "2026-05-10",
        "",
        "## Scope",
        "",
        "This report covers the MP2 nonlinear reference/proxy-metric spine for",
        "the quarantined student DPF experimental-baseline stream.  It is",
        "comparison-only evidence and does not promote student code into",
        "production.",
        "",
        "## Implementation Summary",
        "",
        "| Implementation | Runs | OK | Failed | Min position RMSE | Max position RMSE | Median runtime seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for impl, data in sorted(summary["implementation_summary"].items()):
        lines.append(
            f"| {impl} | {data['runs']} | {data['ok']} | {data['failed']} | "
            f"{_fmt(data['min_position_rmse'])} | {_fmt(data['max_position_rmse'])} | "
            f"{_fmt(data['median_runtime_seconds'])} |"
        )

    lines.extend(
        [
            "",
            "## Method Summary",
            "",
            "| Implementation / fixture / method | Runs | OK | Median state RMSE | Median position RMSE | Median final-position error | Median avg ESS | Min avg ESS | Median runtime seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in sorted(summary["method_summary"].items()):
        lines.append(
            f"| {key} | {data['runs']} | {data['ok']} | "
            f"{_fmt(data['median_state_rmse'])} | "
            f"{_fmt(data['median_position_rmse'])} | "
            f"{_fmt(data['median_final_position_error'])} | "
            f"{_fmt(data['median_average_ess'])} | "
            f"{_fmt(data['min_average_ess'])} | "
            f"{_fmt(data['median_runtime_seconds'])} |"
        )

    lines.extend(["", "## Comparison Summary", ""])
    for key, value in sorted(summary["comparison_summary"].items()):
        lines.append(f"- `{key}`: {_fmt_or_text(value)}")

    lines.extend(["", "## Hypothesis Results", ""])
    for name, text in summary["hypothesis_results"].items():
        lines.extend([f"### {name}", "", text, ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            "All MP2 metrics are proxy/reference metrics against a shared Gaussian",
            "range-bearing fixture.  They do not certify either student",
            "implementation as production quality.  Likelihood values are not",
            "used for cross-student conclusions.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_advanced_methods(fixture: RangeBearingFixture) -> list[dict[str, Any]]:
    records = []
    try:
        model = _make_advanced_model(fixture)
    except Exception as exc:
        return [
            exception_result(
                implementation_name="advanced_particle_filter",
                source_commit=ADVANCED_COMMIT,
                fixture_name=fixture.name,
                exc=exc,
            ).to_dict()
        ]

    with prepend_sys_path(ADVANCED_VENDOR_ROOT):
        from advanced_particle_filter.filters import (  # type: ignore
            BootstrapParticleFilter,
            ExtendedKalmanFilter,
            UnscentedKalmanFilter,
        )

        records.append(
            _advanced_filter_record(
                fixture,
                method="EKF",
                runner=lambda: ExtendedKalmanFilter().filter(
                    model, fixture.observations
                ),
            )
        )
        records.append(
            _advanced_filter_record(
                fixture,
                method="UKF",
                runner=lambda: UnscentedKalmanFilter().filter(
                    model, fixture.observations
                ),
            )
        )
        for seed in PF_SEEDS:
            records.append(
                _advanced_filter_record(
                    fixture,
                    method="BPF",
                    seed=seed,
                    num_particles=NUM_PARTICLES,
                    runner=lambda seed=seed: BootstrapParticleFilter(
                        n_particles=NUM_PARTICLES,
                        seed=seed,
                    ).filter(
                        model,
                        fixture.observations,
                        rng=np.random.default_rng(seed),
                    ),
                )
            )
    return records


def _run_mlcoe_methods(fixture: RangeBearingFixture) -> list[dict[str, Any]]:
    records = []
    with prepend_sys_path(MLCOE_SNAPSHOT_ROOT):
        import tensorflow as tf  # type: ignore
        from src.filters.classical import DTYPE, EKF, UKF  # type: ignore
        from src.filters.particle import BPF  # type: ignore

        model = _MlcoeRangeBearingModel(fixture, tf=tf, dtype=DTYPE)
        records.append(
            _mlcoe_recursive_record(
                fixture,
                method="EKF",
                filter_factory=lambda: EKF(model),
                tf=tf,
                dtype=DTYPE,
                initial_mean=fixture.m0,
                diagnostic="non_origin_initialization",
            )
        )
        records.append(
            _mlcoe_recursive_record(
                fixture,
                method="UKF",
                filter_factory=lambda: UKF(model),
                tf=tf,
                dtype=DTYPE,
                initial_mean=fixture.m0,
                diagnostic="non_origin_initialization",
            )
        )
        records.append(
            _mlcoe_recursive_record(
                fixture,
                method="EKF_origin_diagnostic",
                filter_factory=lambda: EKF(model),
                tf=tf,
                dtype=DTYPE,
                initial_mean=np.zeros(fixture.state_dim, dtype=float),
                diagnostic="origin_initialization",
            )
        )
        for seed in PF_SEEDS:
            records.append(
                _mlcoe_bpf_record(
                    fixture,
                    seed=seed,
                    num_particles=NUM_PARTICLES,
                    tf=tf,
                    dtype=DTYPE,
                    model=model,
                    filter_factory=lambda: BPF(model, N=NUM_PARTICLES),
                )
            )
    return records


def _advanced_filter_record(
    fixture: RangeBearingFixture,
    *,
    method: str,
    runner: Any,
    seed: int | None = None,
    num_particles: int | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = runner()
        metrics = _trajectory_metrics(result.means, fixture)
        if result.ess is not None:
            ess = np.asarray(result.ess, dtype=float)
            metrics["average_ess"] = float(np.mean(ess))
            metrics["min_ess"] = float(np.min(ess))
        return {
            "implementation_name": "advanced_particle_filter",
            "source_commit": ADVANCED_COMMIT,
            "fixture_name": fixture.name,
            "method": method,
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": seed,
            "num_particles": num_particles,
            "runtime_seconds": time.perf_counter() - start,
            "metrics": metrics,
            "diagnostics": {
                "target_label": f"advanced/{method}/gaussian_range_bearing",
                "likelihood_status": "implementation_specific_not_cross_compared",
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
        record.update({"method": method, "target": fixture.target, "metrics": {}})
        return record


def _mlcoe_recursive_record(
    fixture: RangeBearingFixture,
    *,
    method: str,
    filter_factory: Any,
    tf: Any,
    dtype: Any,
    initial_mean: np.ndarray,
    diagnostic: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        filt = filter_factory()
        filt.init(
            tf.constant(initial_mean, dtype=dtype),
            tf.constant(fixture.P0, dtype=dtype),
        )
        means = [np.asarray(initial_mean, dtype=float)]
        for obs in fixture.observations:
            estimate = filt.step(tf.constant(obs, dtype=dtype))
            means.append(np.asarray(estimate.numpy(), dtype=float).reshape(-1))
        means_array = np.asarray(means, dtype=float)
        return {
            "implementation_name": "2026MLCOE",
            "source_commit": MLCOE_COMMIT,
            "fixture_name": fixture.name,
            "method": method,
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": None,
            "num_particles": None,
            "runtime_seconds": time.perf_counter() - start,
            "metrics": _trajectory_metrics(means_array, fixture),
            "diagnostics": {
                "target_label": f"mlcoe/{method}/gaussian_range_bearing",
                "initialization": diagnostic,
                "likelihood_status": "not_exposed_by_mlcoe_recursive_filter",
            },
        }
    except Exception as exc:
        record = exception_result(
            implementation_name="2026MLCOE",
            source_commit=MLCOE_COMMIT,
            fixture_name=fixture.name,
            exc=exc,
            runtime_seconds=time.perf_counter() - start,
        ).to_dict()
        record.update({"method": method, "target": fixture.target, "metrics": {}})
        return record


def _mlcoe_bpf_record(
    fixture: RangeBearingFixture,
    *,
    seed: int,
    num_particles: int,
    tf: Any,
    dtype: Any,
    model: Any,
    filter_factory: Any,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        filt = filter_factory()
        filt.init(
            tf.constant(fixture.m0, dtype=dtype),
            tf.constant(fixture.P0, dtype=dtype),
        )
        means = [_weighted_mean(filt)]
        ess_by_time = []
        threshold_resampled = []
        for obs in fixture.observations:
            filt.step(tf.constant(obs, dtype=dtype))
            means.append(_weighted_mean(filt))
            ess = float(filt.ess.numpy())
            ess_by_time.append(ess)
            threshold_resampled.append(ess < num_particles * 0.1)
        means_array = np.asarray(means, dtype=float)
        metrics = _trajectory_metrics(means_array, fixture)
        ess_array = np.asarray(ess_by_time, dtype=float)
        metrics["average_ess"] = float(np.mean(ess_array))
        metrics["min_ess"] = float(np.min(ess_array))
        return {
            "implementation_name": "2026MLCOE",
            "source_commit": MLCOE_COMMIT,
            "fixture_name": fixture.name,
            "method": "BPF",
            "target": fixture.target,
            "status": str(BaselineStatus.OK),
            "seed": seed,
            "num_particles": num_particles,
            "runtime_seconds": time.perf_counter() - start,
            "metrics": metrics,
            "diagnostics": {
                "target_label": "mlcoe/BPF/gaussian_range_bearing",
                "likelihood_status": "not_exposed_by_mlcoe_bpf",
                "ess_semantics": "pre_resampling_step_ess",
                "resampling_count": int(np.sum(threshold_resampled)),
                "resampling_count_semantics": "threshold_inferred_ess_lt_0.1N",
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
        record.update({"method": "BPF", "target": fixture.target, "metrics": {}})
        return record


def _make_advanced_model(fixture: RangeBearingFixture) -> Any:
    with prepend_sys_path(ADVANCED_VENDOR_ROOT):
        from advanced_particle_filter.models.base import StateSpaceModel  # type: ignore

        def obs_log_prob(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            predicted = range_bearing_observation(x)
            residual = observation_residual(predicted, y)
            inv_r = np.linalg.inv(fixture.R)
            sign, logdet = np.linalg.slogdet(fixture.R)
            if sign <= 0:
                raise np.linalg.LinAlgError("observation covariance is not positive definite")
            quad = np.einsum("ni,ij,nj->n", residual, inv_r, residual)
            return -0.5 * (fixture.obs_dim * np.log(2 * np.pi) + logdet + quad)

        def obs_sample(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            observation = rng.multivariate_normal(range_bearing_observation(x), fixture.R)
            observation[1] = ((observation[1] + np.pi) % (2.0 * np.pi)) - np.pi
            return observation

        return StateSpaceModel(
            state_dim=fixture.state_dim,
            obs_dim=fixture.obs_dim,
            initial_mean=fixture.m0,
            initial_cov=fixture.P0,
            dynamics_mean=lambda x: x @ fixture.A.T,
            dynamics_cov=fixture.Q,
            dynamics_jacobian=lambda _x: fixture.A,
            obs_mean=range_bearing_observation,
            obs_cov=fixture.R,
            obs_jacobian=range_bearing_jacobian,
            obs_log_prob=obs_log_prob,
            obs_sample=obs_sample,
        )


class _MlcoeRangeBearingModel:
    def __init__(self, fixture: RangeBearingFixture, *, tf: Any, dtype: Any) -> None:
        self.state_dim = fixture.state_dim
        self.obs_dim = fixture.obs_dim
        self.F = tf.constant(fixture.A, dtype=dtype)
        self.Q_filter = tf.constant(fixture.Q, dtype=dtype)
        self.R_filter = tf.constant(fixture.R, dtype=dtype)
        self.R_inv_filter = tf.linalg.inv(self.R_filter)

    def h_func(self, x: Any) -> Any:
        import tensorflow as tf  # type: ignore

        x_in = x
        squeeze_output = False
        if len(x.shape) == 1:
            x_in = x[None, :]
            squeeze_output = True
        px = x_in[:, 0]
        py = x_in[:, 1]
        y = tf.stack([tf.sqrt(px**2 + py**2 + 1e-12), tf.atan2(py, px)], axis=1)
        return y[0] if squeeze_output else y

    def jacobian_h(self, x: Any) -> Any:
        import tensorflow as tf  # type: ignore

        x_vec = tf.reshape(x, [-1])
        px, py = x_vec[0], x_vec[1]
        r2 = tf.maximum(px**2 + py**2, 1e-12)
        r = tf.sqrt(r2)
        return tf.stack(
            [
                tf.stack([px / r, py / r, 0.0, 0.0]),
                tf.stack([-py / r2, px / r2, 0.0, 0.0]),
            ]
        )


def _trajectory_metrics(means: np.ndarray, fixture: RangeBearingFixture) -> dict[str, float]:
    means_arr = np.asarray(means, dtype=float)
    states = fixture.states
    if means_arr.shape != states.shape:
        raise ValueError(f"mean/state shape mismatch: {means_arr.shape} != {states.shape}")
    state_diff = means_arr - states
    position_diff = state_diff[:, :2]
    final_position_diff = position_diff[-1]
    observation_pred = range_bearing_observation(means_arr[1:])
    observation_resid = observation_residual(observation_pred, fixture.observations)
    return {
        "state_rmse": float(np.sqrt(np.mean(state_diff**2))),
        "position_rmse": float(np.sqrt(np.mean(position_diff**2))),
        "final_position_error": float(np.sqrt(np.sum(final_position_diff**2))),
        "observation_proxy_rmse": float(np.sqrt(np.mean(observation_resid**2))),
    }


def _weighted_mean(filt: Any) -> np.ndarray:
    particles = np.asarray(filt.X.numpy(), dtype=float)
    weights = np.asarray(filt.W.numpy(), dtype=float)
    weights = weights / np.sum(weights)
    return np.sum(particles * weights[:, None], axis=0)


def _summarize_comparisons(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for fixture in nonlinear_fixture_names():
        for impl in ["advanced_particle_filter", "2026MLCOE"]:
            ekf = _single_ok(records, impl, fixture, "EKF")
            ukf = _single_ok(records, impl, fixture, "UKF")
            if ekf and ukf:
                summary[f"{impl}/{fixture}/ekf_position_minus_ukf_position_rmse"] = (
                    ekf["metrics"]["position_rmse"] - ukf["metrics"]["position_rmse"]
                )
        for impl in ["advanced_particle_filter", "2026MLCOE"]:
            moderate_or_low = [
                r for r in records
                if r["implementation_name"] == impl
                and r["fixture_name"] == fixture
                and r["method"] == "BPF"
                and r["status"] == "ok"
            ]
            if moderate_or_low:
                summary[f"{impl}/{fixture}/bpf_position_rmse_seed_std"] = float(
                    np.std([r["metrics"]["position_rmse"] for r in moderate_or_low])
                )
    return summary


def _single_ok(
    records: list[dict[str, Any]],
    impl: str,
    fixture: str,
    method: str,
) -> dict[str, Any] | None:
    matches = [
        r for r in records
        if r["implementation_name"] == impl
        and r["fixture_name"] == fixture
        and r["method"] == method
        and r["status"] == "ok"
    ]
    return matches[0] if matches else None


def _interpret_n1(by_impl: dict[str, list[dict[str, Any]]]) -> str:
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        ok_recursive = [
            r for r in by_impl.get(impl, [])
            if r["status"] == "ok" and r["method"] in {"EKF", "UKF"}
        ]
        if not ok_recursive:
            return f"Blocked: {impl} did not produce an EKF/UKF result."
    return (
        "Supported.  Both snapshots produced at least one EKF/UKF result on "
        "the shared Gaussian range-bearing fixtures without vendored-code edits."
    )


def _interpret_n2(records: list[dict[str, Any]]) -> str:
    non_origin = [
        r for r in records
        if r["implementation_name"] == "2026MLCOE"
        and r["method"] == "EKF"
        and r["status"] == "ok"
    ]
    origin = [
        r for r in records
        if r["implementation_name"] == "2026MLCOE"
        and r["method"] == "EKF_origin_diagnostic"
        and r["status"] == "ok"
    ]
    if not non_origin or not origin:
        return "Blocked: MLCOE EKF origin/non-origin diagnostics did not both run."
    non_origin_max = max(r["metrics"]["position_rmse"] for r in non_origin)
    origin_max = max(r["metrics"]["position_rmse"] for r in origin)
    if np.isfinite(non_origin_max) and origin_max > non_origin_max * 1.5:
        return (
            "Supported.  MLCOE EKF is usable away from the origin, while the "
            "origin diagnostic is materially worse, consistent with a "
            "range-bearing Jacobian initialization artifact."
        )
    return (
        "Partially supported.  MLCOE EKF runs under both initializations, but "
        "the origin diagnostic was not materially worse under this fixture."
    )


def _interpret_n3(method_summary: dict[str, Any]) -> str:
    supported = []
    for impl in ["advanced_particle_filter", "2026MLCOE"]:
        moderate = method_summary.get(
            f"{impl}/range_bearing_gaussian_moderate/BPF", {}
        )
        low = method_summary.get(
            f"{impl}/range_bearing_gaussian_low_noise/BPF", {}
        )
        if not moderate or not low:
            return f"Blocked: missing BPF summaries for {impl}."
        low_ess = low.get("min_average_ess")
        moderate_ess = moderate.get("min_average_ess")
        low_rmse = low.get("median_position_rmse")
        moderate_rmse = moderate.get("median_position_rmse")
        supported.append(
            low_ess is not None
            and moderate_ess is not None
            and low_rmse is not None
            and moderate_rmse is not None
            and (low_ess <= moderate_ess or low_rmse >= moderate_rmse)
        )
    if all(supported):
        return (
            "Supported for this proxy panel.  BPF summaries show lower ESS or "
            "larger RMSE pressure on the low-noise fixture for both snapshots."
        )
    return (
        "Partially supported.  Repeated-seed PF summaries were produced, but "
        "the low-noise pressure pattern was not consistent for both snapshots."
    )


def _interpret_n4(records: list[dict[str, Any]]) -> str:
    ok = [r for r in records if r["status"] == "ok"]
    if ok and all("target_label" in r.get("diagnostics", {}) for r in ok):
        return (
            "Supported.  Records include target labels and rely on latent-state "
            "RMSE, EKF/UKF diagnostics, PF dispersion, ESS, and runtime rather "
            "than student agreement or direct likelihood comparison."
        )
    return "Partially supported: some successful records lack target labels."


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


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6g}"


def _fmt_or_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _fmt(value)


if __name__ == "__main__":
    main()
