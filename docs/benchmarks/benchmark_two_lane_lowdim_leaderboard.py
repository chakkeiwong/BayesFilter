#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument(
    "--requested-device",
    choices=("cpu",),
    default="cpu",
    help="Lowdim leaderboard worker is CPU-only; GPU is intentionally hidden before TensorFlow import.",
)
_PRE_ARGS, _ = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.requested_device == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

import bayesfilter.highdim as highdim  # noqa: E402
from bayesfilter.nonlinear.fixed_sgqf_structural_adapter_tf import (  # noqa: E402
    tf_structural_to_fixed_sgqf_model,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import (  # noqa: E402
    TFFixedSGQFAffineModel,
    tf_fixed_sgqf_cloud,
    tf_fixed_sgqf_filter,
)
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter  # noqa: E402
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_filter  # noqa: E402
from bayesfilter.structural_tf import affine_structural_to_linear_gaussian_tf  # noqa: E402
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood  # noqa: E402

DTYPE = tf.float64
VALUE_ROWS = {
    "lgssm_exact_kalman_dim_1_2_3",
    "sv_ksc_gaussian_mixture_surrogate_dim_1_2_3",
}
DIAGNOSTIC_ROWS = {
    "p44_cubic_additive_gaussian_dim_1_2_3",
    "p44_quadratic_observation_dim_1_2_3",
    "p44_nonlinear_transition_h2_dim_1_2_3",
    "p44_nonlinear_transition_h4_cut4_extension_dim_1_2_3",
}
DATE = "2026-06-30"
DEFAULT_JSON = ROOT / f"docs/plans/bayesfilter-two-lane-lowdim-leaderboard-results-{DATE}.json"
DEFAULT_MD = ROOT / f"docs/plans/bayesfilter-two-lane-lowdim-leaderboard-results-{DATE}.md"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _time_call(fn, repeats: int) -> tuple[Any, float, float]:
    start = time.perf_counter()
    result = fn()
    first = time.perf_counter() - start
    steady_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        steady_times.append(time.perf_counter() - start)
    mean_steady = sum(steady_times) / len(steady_times) if steady_times else first
    return result, first, mean_steady


def _safe_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P44_LGSSM = _load_module("two_lane_p44_lgssm", "tests/highdim/test_p44_lgssm_exact_baseline.py")
P43_SV = _load_module("two_lane_p43_sv", "tests/highdim/test_p43_sv_value_gradient_cut4_zhaocui.py")


def _theta_lgssm() -> tf.Tensor:
    return tf.constant([0.25, math.log(0.18), math.log(0.12), 0.04], dtype=DTYPE)


def _lgssm_observations(dim: int) -> tf.Tensor:
    values = tf.constant(
        [[0.10, -0.04, 0.06], [-0.03, 0.08, -0.02], [0.05, -0.01, 0.07]], dtype=DTYPE
    )
    return values[:, : int(dim)]


def _lgssm_structural_model(dim: int):
    return P44_LGSSM._structural_model(_theta_lgssm(), dim)


def _lgssm_kalman_value(dim: int) -> tf.Tensor:
    structural = _lgssm_structural_model(dim)
    linear = affine_structural_to_linear_gaussian_tf(structural)
    return tf_linear_gaussian_log_likelihood(
        _lgssm_observations(dim),
        linear,
        backend="tf_cholesky",
        jitter=tf.constant(0.0, dtype=DTYPE),
        return_filtered=True,
    ).log_likelihood


def _lgssm_ukf_value(dim: int) -> tf.Tensor:
    return tf_svd_sigma_point_filter(
        _lgssm_observations(dim),
        _lgssm_structural_model(dim),
        backend="tf_svd_ukf",
        innovation_floor=tf.constant(1e-12, dtype=DTYPE),
        return_filtered=True,
    ).log_likelihood


def _lgssm_cut4_value(dim: int) -> tf.Tensor:
    return tf_svd_cut4_filter(
        _lgssm_observations(dim),
        _lgssm_structural_model(dim),
        innovation_floor=tf.constant(1e-12, dtype=DTYPE),
        return_filtered=True,
    ).log_likelihood


def _lgssm_zhaocui_value(dim: int) -> tf.Tensor:
    return P44_LGSSM._zhaocui_artifact_value(_theta_lgssm(), dim)[0]


def _lgssm_fixed_sgqf_value(dim: int) -> tf.Tensor:
    structural = _lgssm_structural_model(dim)
    linear = affine_structural_to_linear_gaussian_tf(structural)
    affine = TFFixedSGQFAffineModel(
        initial_mean=linear.initial_mean,
        initial_covariance=linear.initial_covariance,
        transition_matrix=linear.transition_matrix,
        process_covariance=linear.transition_covariance,
        observation_matrix=linear.observation_matrix,
        observation_covariance=linear.observation_covariance,
    )
    return tf_fixed_sgqf_filter(
        _lgssm_observations(dim),
        affine,
        cloud=tf_fixed_sgqf_cloud(dim=dim, sparse_level=2),
        return_filtered=True,
    ).log_likelihood


def _ksc_observations(dim: int) -> tf.Tensor:
    return P43_SV._observations(dim)


def _ksc_physical_parameters(dim: int):
    return P43_SV._physical_parameters(dim)


def _ksc_theta(dim: int) -> tf.Tensor:
    gamma, beta, _sigma = _ksc_physical_parameters(dim)
    return P43_SV._theta_from_physical(gamma, beta)


def _ksc_kalman_value(dim: int) -> tf.Tensor:
    observations = _ksc_observations(dim)
    _gamma, _beta, sigma = _ksc_physical_parameters(dim)
    return P43_SV._ksc_kalman_value(_ksc_theta(dim), observations, sigma)


def _ksc_ukf_value(dim: int) -> tf.Tensor:
    observations = _ksc_observations(dim)
    gamma, beta, sigma = _ksc_physical_parameters(dim)
    return highdim.independent_panel_sv_mixture_ukf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _ksc_cut4_value(dim: int) -> tf.Tensor:
    observations = _ksc_observations(dim)
    _gamma, _beta, sigma = _ksc_physical_parameters(dim)
    return P43_SV._ksc_cut4_value(_ksc_theta(dim), observations, sigma)


def _ksc_zhaocui_config(seed: str) -> Any:
    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 48)],
        convention,
    )
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1),
            ridge=1e-12,
            max_sweeps=2,
            sweep_order=(0,),
            row_budget=512,
            column_budget=128,
            dense_matrix_byte_budget=200_000,
            normal_matrix_byte_budget=100_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=5e-4,
        ),
        density_tau=0.0,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=tf.float64),
                matrix=tf.constant([[8.0]], dtype=tf.float64),
            ),
        ),
        measure_convention=convention,
        deterministic_seed=seed,
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(tf.ones([1, product_basis.bases[0].basis_dim, 1], dtype=tf.float64)),
        ),
        fit_quadrature_order=141,
    )


def _ksc_zhaocui_value(dim: int) -> tf.Tensor:
    observations = _ksc_observations(dim)
    gamma, beta, sigma = _ksc_physical_parameters(dim)
    return highdim.independent_panel_sv_mixture_zhaocui_tt_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        config=_ksc_zhaocui_config(f"two-lane-ksc-zhaocui-dim-{dim}"),
        fixture_id=f"two-lane.ksc.zhaocui.dim-{dim}",
        branch_seed_prefix=f"two-lane-ksc-zhaocui-dim-{dim}",
    ).log_likelihood


def _ksc_fixed_sgqf_value(dim: int) -> tf.Tensor:
    observations = _ksc_observations(dim)
    _gamma, _beta, sigma = _ksc_physical_parameters(dim)
    return P43_SV._ksc_fixed_sgqf_value(_ksc_theta(dim), observations, sigma)


def _model_c_case() -> tuple[Any, tf.Tensor]:
    structural = _load_module(
        "two_lane_nonlinear_models", "tests/test_nonlinear_benchmark_models_tf.py"
    ).make_univariate_nonlinear_growth_model_tf()
    observations = tf.constant([[1.2], [0.8]], dtype=DTYPE)
    return structural, observations


def _model_c_reference_value(observations: tf.Tensor) -> tf.Tensor:
    model = _load_module(
        "two_lane_nonlinear_models_ref", "tests/test_nonlinear_benchmark_models_tf.py"
    ).make_univariate_nonlinear_growth_model_tf()
    return highdim.FixedBranchSquaredTTFilter(
        highdim.FixedBranchFilterConfig(
            fit_config=None,
            density_tau=0.0,
            normalizer_floor=1e-12,
            denominator_floor=1e-12,
            retained_storage_byte_budget=20_000_000,
            coordinate_maps=(
                highdim.AffineCoordinateMap(
                    offset=tf.constant([0.0], dtype=DTYPE),
                    matrix=tf.constant([[8.0], [0.0]], dtype=DTYPE)[:,:1],
                ),
            ),
            measure_convention=highdim.MeasureConvention(
                density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
                mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
                reference_weight_name="omega",
            ),
            deterministic_seed="two-lane-lowdim-model-c-dense",
            fit_quadrature_order=81,
        )
    ).log_likelihood(model, tf.zeros([0], dtype=DTYPE), observations).log_likelihood


def _model_c_ukf_value(observations: tf.Tensor) -> tf.Tensor:
    model = _load_module(
        "two_lane_nonlinear_models_ukf", "tests/test_nonlinear_benchmark_models_tf.py"
    ).make_univariate_nonlinear_growth_model_tf()
    return tf_svd_sigma_point_filter(
        observations,
        model,
        backend="tf_svd_ukf",
        innovation_floor=tf.constant(1e-12, dtype=DTYPE),
        return_filtered=True,
    ).log_likelihood


def _model_c_cut4_value(observations: tf.Tensor) -> tf.Tensor:
    model = _load_module(
        "two_lane_nonlinear_models_cut4", "tests/test_nonlinear_benchmark_models_tf.py"
    ).make_univariate_nonlinear_growth_model_tf()
    return tf_svd_cut4_filter(
        observations,
        model,
        innovation_floor=tf.constant(1e-12, dtype=DTYPE),
        return_filtered=True,
    ).log_likelihood


def _model_c_fixed_sgqf_value(observations: tf.Tensor) -> tf.Tensor:
    adapted = tf_structural_to_fixed_sgqf_model(
        _load_module(
            "two_lane_nonlinear_models_sgqf", "tests/test_nonlinear_benchmark_models_tf.py"
        ).make_univariate_nonlinear_growth_model_tf()
    )
    if not adapted.eligible or adapted.model is None:
        raise ValueError(adapted.reason or "fixed_sgqf adapter ineligible")
    return tf_fixed_sgqf_filter(
        observations,
        adapted.model,
        cloud=tf_fixed_sgqf_cloud(dim=2, sparse_level=2),
        return_filtered=True,
    ).log_likelihood


def _row_payload(*, lane: str, row_id: str, algorithm_id: str, reference_type: str, comparison_status: str, reason: str | None, log_likelihood: float | None, reference_log_likelihood: float | None, first_seconds: float | None, steady_seconds: float | None, repeats: int | None, notes: list[str]) -> dict[str, Any]:
    abs_error = None
    if log_likelihood is not None and reference_log_likelihood is not None:
        abs_error = abs(log_likelihood - reference_log_likelihood)
    return {
        "lane": lane,
        "row_id": row_id,
        "algorithm_id": algorithm_id,
        "reference_type": reference_type,
        "comparison_status": comparison_status,
        "reason": reason,
        "log_likelihood": log_likelihood,
        "reference_log_likelihood": reference_log_likelihood,
        "abs_log_likelihood_error": abs_error,
        "runtime": {
            "compile_warmup_policy": "first_call_then_mean_of_repeats_no_tf_function_compile",
            "first_call_seconds": first_seconds,
            "mean_steady_seconds": steady_seconds,
            "repeats": repeats,
        },
        "notes": notes,
    }


def _mark_fixed_variant_zhao_cui_value_row(row: dict[str, Any], *, route_id: str) -> dict[str, Any]:
    """Identify a fixed-branch Zhao-Cui value comparator without a score claim."""

    row = dict(row)
    row.update(
        {
            "route_id": route_id,
            "route_role": "fixed_variant_value_only_comparator",
            "hmc_route_policy_id": highdim.ZHAO_CUI_HMC_ROUTE_POLICY_ID,
            "hmc_target_scope_admitted": False,
            "score_status": "value_only_no_analytical_score_in_lowdim_harness",
            "score_status_reason": (
                "This lowdim harness evaluates the fixed-branch value only; "
                "it does not emit an analytical HMC score"
            ),
        }
    )
    row["notes"] = list(row.get("notes", [])) + [
        "fixed-branch value comparator; no analytical-score admission",
    ]
    return row


def build_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Lowdim rankable: LGSSM dims 1-3, using existing exact/approximate routes.
    for dim in (1, 2, 3):
        row_id = f"lgssm_exact_kalman_dim_{dim}"
        ref, first, steady = _time_call(lambda: _lgssm_kalman_value(dim), repeats)
        ref_value = _safe_float(ref.numpy())
        rows.append(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id="kalman_exact_or_mixture_enumeration", reference_type="exact", comparison_status="reference_only", reason=None, log_likelihood=ref_value, reference_log_likelihood=ref_value, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=["exact LGSSM baseline"]))
        for algorithm_id, fn in [
            ("ukf", _lgssm_ukf_value),
            ("cut4", _lgssm_cut4_value),
            ("fixed_sgqf", _lgssm_fixed_sgqf_value),
        ]:
            value, first, steady = _time_call(lambda fn=fn: fn(dim), repeats)
            rows.append(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id=algorithm_id, reference_type="exact", comparison_status="executed_rankable", reason=None, log_likelihood=_safe_float(value.numpy()), reference_log_likelihood=ref_value, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=["same-target lowdim LGSSM"]))
        value, first, steady = _time_call(lambda: _lgssm_zhaocui_value(dim), repeats)
        rows.append(_mark_fixed_variant_zhao_cui_value_row(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id="zhao_cui_scalar_or_multistate", reference_type="exact", comparison_status="diagnostic_only", reason="value-only Zhao-Cui comparator; analytical score not emitted by this lowdim harness", log_likelihood=_safe_float(value.numpy()), reference_log_likelihood=ref_value, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=["value-only comparator"]), route_id="zhao_cui_lgssm_exact_oracle_affine_adapter"))

    # Lowdim rankable surrogate: KSC dims 1-3.
    for dim in (1, 2, 3):
        row_id = f"sv_ksc_gaussian_mixture_surrogate_dim_{dim}"
        ref, first, steady = _time_call(lambda: _ksc_kalman_value(dim), repeats)
        ref_value = _safe_float(ref.numpy())
        rows.append(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id="kalman_exact_or_mixture_enumeration", reference_type="gaussian_mixture_surrogate", comparison_status="reference_only", reason=None, log_likelihood=ref_value, reference_log_likelihood=ref_value, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=["declared surrogate reference"] ))
        for algorithm_id, fn, notes in [
            ("ukf", _ksc_ukf_value, ["same-target surrogate lane"]),
            ("cut4", _ksc_cut4_value, ["same-target surrogate lane"]),
            ("fixed_sgqf", _ksc_fixed_sgqf_value, ["tiny_same_target_surrogate_fixture_only"]),
        ]:
            value, first, steady = _time_call(lambda fn=fn: fn(dim), repeats)
            rows.append(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id=algorithm_id, reference_type="gaussian_mixture_surrogate", comparison_status="executed_rankable", reason=None, log_likelihood=_safe_float(value.numpy()), reference_log_likelihood=ref_value, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=notes))
        value, first, steady = _time_call(lambda: _ksc_zhaocui_value(dim), repeats)
        rows.append(_mark_fixed_variant_zhao_cui_value_row(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id="zhao_cui_scalar_or_multistate", reference_type="gaussian_mixture_surrogate", comparison_status="diagnostic_only", reason="value-only Zhao-Cui comparator; analytical score not emitted by this lowdim harness", log_likelihood=_safe_float(value.numpy()), reference_log_likelihood=ref_value, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=["value-only comparator"]), route_id="zhao_cui_ksc_mixture_fixed_branch_tt"))

    # Dedicated SGQF-only lowdim exact-eligible fixture so SGQF timing/accuracy is real, but status-only for cross-algorithm leaderboard.
    model_c, observations = _model_c_case()
    row_id = "fixed_sgqf_model_c_autonomous_nonlinear_growth_fixture"
    value, first, steady = _time_call(lambda: _model_c_fixed_sgqf_value(observations), repeats)
    rows.append(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id="fixed_sgqf", reference_type="dense_reference_local_fixture", comparison_status="executed_status_only", reason="SGQF-exact-eligible local fixture; not one of the frozen four-way lowdim rankable rows", log_likelihood=_safe_float(value.numpy()), reference_log_likelihood=None, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=["status-only SGQF timing anchor", "not used for cross-algorithm rankable leaderboard"]))
    for algorithm_id, fn in [
        ("ukf", _model_c_ukf_value),
        ("cut4", _model_c_cut4_value),
    ]:
        value, first, steady = _time_call(lambda fn=fn: fn(observations), repeats)
        rows.append(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id=algorithm_id, reference_type="dense_reference_local_fixture", comparison_status="executed_status_only", reason="local SGQF-exact-eligible fixture; auxiliary status-only timing row", log_likelihood=_safe_float(value.numpy()), reference_log_likelihood=None, first_seconds=_safe_float(first), steady_seconds=_safe_float(steady), repeats=repeats, notes=["auxiliary status-only timing row"]))

    # Diagnostic-only rows from P44 where SGQF remains blocked; keep visible rather than omitted.
    for row_id in sorted(DIAGNOSTIC_ROWS):
        for algorithm_id in ("fixed_sgqf", "ukf", "cut4", "zhao_cui_scalar_or_multistate"):
            reason = "diagnostic-only row; not rankable in lowdim leaderboard"
            status = "diagnostic_only"
            if algorithm_id == "fixed_sgqf":
                reason = "fixed SGQF not same-target or adapter not admitted on this diagnostic-only P44 row"
                status = "blocked"
            rows.append(_row_payload(lane="lowdim_same_target", row_id=row_id, algorithm_id=algorithm_id, reference_type="diagnostic", comparison_status=status, reason=reason, log_likelihood=None, reference_log_likelihood=None, first_seconds=None, steady_seconds=None, repeats=None, notes=["visible non-rankable row"]))

    return rows


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# Two-Lane Lowdim Leaderboard Result",
        "",
        f"Authoritative JSON artifact: `{json_path.relative_to(ROOT)}`.",
        "",
        "## Rankable lowdim rows",
        "",
        "| Row | Algorithm | Status | Abs error | First s | Steady s | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        if row["comparison_status"] not in {"executed_rankable", "reference_only"}:
            continue
        err = "n/a" if row["abs_log_likelihood_error"] is None else f"{row['abs_log_likelihood_error']:.6e}"
        first = "n/a" if row["runtime"]["first_call_seconds"] is None else f"{row['runtime']['first_call_seconds']:.6f}"
        steady = "n/a" if row["runtime"]["mean_steady_seconds"] is None else f"{row['runtime']['mean_steady_seconds']:.6f}"
        notes = "; ".join(row["notes"])
        lines.append(f"| {row['row_id']} | {row['algorithm_id']} | {row['comparison_status']} | {err} | {first} | {steady} | {notes} |")
    lines.extend([
        "",
        "## Status-only and blocked rows",
        "",
        "| Row | Algorithm | Status | Reason |",
        "| --- | --- | --- | --- |",
    ])
    for row in payload["rows"]:
        if row["comparison_status"] in {"executed_rankable", "reference_only"}:
            continue
        lines.append(f"| {row['row_id']} | {row['algorithm_id']} | {row['comparison_status']} | {row['reason'] or ''} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requested-device", choices=("cpu",), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--plan-path",
        type=str,
        default="docs/plans/bayesfilter-two-lane-filter-comparison-p5-lowdim-execution-subplan-2026-06-24.md",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default="docs/plans/bayesfilter-two-lane-lowdim-leaderboard-results-2026-06-24.json",
    )
    args = parser.parse_args()

    rows = build_rows(args.repeats)
    payload = _json_safe(
        {
            "benchmark": "bayesfilter_two_lane_lowdim_leaderboard_cpu_reference",
            "metadata_date": DATE,
            "manifest": {
                "command": " ".join(sys.argv),
                "git_commit": _git_commit(),
                "requested_device": args.requested_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "cpu_gpu_visibility_policy": "cpu_only_hidden_gpu_before_tensorflow_import",
                "tensorflow_version": tf.__version__,
                "artifact_paths": [str(args.output.relative_to(ROOT)), str(args.markdown_output.relative_to(ROOT))],
                "governing_plan_path": args.plan_path,
                "excluded_algorithms": [
                    "bootstrap_dpf_current",
                    "ledh_pfpf_alg1_ukf_current",
                    "ledh_pfpf_ot",
                ],
            },
            "rows": rows,
            "nonclaims": [
                "CPU-only reference leaderboard; not a production-GPU timing result.",
                "LEDH/PFPF-OT and DPF transport rows are omitted from this June 30 rebuild.",
                "Status-only rows are visible and not ranked.",
                "Actual transformed SV is not merged with KSC surrogate SV.",
                "SGQF timing/accuracy outside same-target admitted rows and the dedicated local fixture remain blocked or status-only.",
            ],
        }
    )
    args.output.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
