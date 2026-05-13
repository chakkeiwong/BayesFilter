"""CPU-only nonlinear sigma-point benchmark harness for BayesFilter v1.

The harness compares SVD cubature, SVD-UKF, and SVD-CUT4 on the first-rung
nonlinear validation models.  Model A has an exact linear-Gaussian reference.
Models B-C report dense one-step Gaussian projection errors only; they do not
have an exact full nonlinear likelihood reference in this artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter import (  # noqa: E402
    StatePartition,
    affine_structural_to_linear_gaussian_tf,
    make_affine_structural_tf,
)
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood  # noqa: E402
from bayesfilter.nonlinear.cut_tf import tf_cut4g_sigma_point_rule  # noqa: E402
from bayesfilter.nonlinear.sigma_points_tf import tf_unit_sigma_point_rule  # noqa: E402
from bayesfilter.testing import (  # noqa: E402
    dense_projection_first_step,
    make_affine_gaussian_structural_oracle_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_a_observations_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    nonlinear_sigma_point_diagnostic_snapshot,
    nonlinear_sigma_point_value_branch_summary,
    sigma_point_projection_first_step,
    tf_nonlinear_sigma_point_value_filter,
)


BACKENDS = ("tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4")


@dataclass(frozen=True)
class NonlinearBenchmarkRow:
    model: str
    backend: str
    reference_kind: str
    timesteps: int
    state_dim: int
    innovation_dim: int
    observation_dim: int
    point_count: int
    polynomial_degree: int
    max_integration_rank: int
    log_likelihood: float
    reference_log_likelihood: float | None
    abs_log_likelihood_error: float | None
    first_step_reference_kind: str | None
    first_step_abs_log_likelihood_error: float | None
    first_step_filtered_mean_l2_error: float | None
    first_step_filtered_covariance_fro_error: float | None
    exact_filtered_mean_max_l2_error: float | None
    exact_filtered_covariance_max_fro_error: float | None
    deterministic_residual: float
    support_residual: float
    min_placement_eigen_gap: float
    min_innovation_eigen_gap: float
    branch_ok_count: int
    branch_total_count: int
    branch_ok_fraction: float
    branch_active_floor_count: int
    branch_weak_spectral_gap_count: int
    branch_nonfinite_count: int
    first_call_seconds: float
    mean_steady_seconds: float
    max_rss_before_mb: float
    max_rss_after_mb: float
    status: str
    error: str | None


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _model_a_builder(params: tf.Tensor):
    phi = params[0]
    sigma = params[1]
    obs_scale = params[2]
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.stack(
            [
                tf.stack([phi, tf.constant(-0.10, dtype=tf.float64)]),
                tf.constant([1.0, 0.0], dtype=tf.float64),
            ]
        ),
        innovation_matrix=tf.reshape(
            tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]),
            [2, 1],
        ),
        innovation_covariance=tf.constant([[1.0]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.reshape(tf.stack([obs_scale, 0.0]), [1, 2]),
        observation_covariance=tf.constant([[0.15**2]], dtype=tf.float64),
    )


def _model_b_builder(params: tf.Tensor):
    return make_nonlinear_accumulation_model_tf(
        rho=float(params[0].numpy()),
        sigma=float(params[1].numpy()),
        beta=float(params[2].numpy()),
    )


def _model_c_builder(params: tf.Tensor):
    return make_univariate_nonlinear_growth_model_tf(
        process_sigma=float(params[0].numpy()),
        observation_sigma=float(params[1].numpy()),
        initial_variance=float(params[2].numpy()),
    )


def _model_cases() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "model_a_affine_gaussian_structural_oracle",
            "model": make_affine_gaussian_structural_oracle_tf(),
            "observations": model_a_observations_tf(),
            "reference_kind": "exact_linear_gaussian_kalman",
            "branch_grid": tf.constant(
                [[0.32, 0.22, 1.0], [0.35, 0.25, 1.0], [0.38, 0.28, 1.0]],
                dtype=tf.float64,
            ),
            "builder": _model_a_builder,
        },
        {
            "name": "model_b_nonlinear_accumulation",
            "model": make_nonlinear_accumulation_model_tf(),
            "observations": model_b_observations_tf(),
            "reference_kind": "dense_one_step_projection_only",
            "branch_grid": tf.constant(
                [[0.66, 0.23, 0.75], [0.70, 0.25, 0.80], [0.74, 0.27, 0.85]],
                dtype=tf.float64,
            ),
            "builder": _model_b_builder,
        },
        {
            "name": "model_c_autonomous_nonlinear_growth",
            "model": make_univariate_nonlinear_growth_model_tf(),
            "observations": model_c_observations_tf(),
            "reference_kind": "dense_one_step_projection_only",
            "branch_grid": tf.constant(
                [[0.90, 1.00, 0.20], [1.00, 1.00, 0.20], [1.10, 1.10, 0.25]],
                dtype=tf.float64,
            ),
            "builder": _model_c_builder,
        },
    )


def _rule_for_backend(backend: str, dim: int):
    if backend == "tf_svd_cubature":
        return tf_unit_sigma_point_rule(dim, rule="cubature")
    if backend == "tf_svd_ukf":
        return tf_unit_sigma_point_rule(dim, rule="unscented")
    if backend == "tf_svd_cut4":
        return tf_cut4g_sigma_point_rule(dim)
    raise ValueError(f"unknown backend: {backend}")


def _time_call(fn, repeats: int) -> tuple[Any, float, float]:
    start = time.perf_counter()
    result = fn()
    first = time.perf_counter() - start
    steady_times = []
    steady_result = result
    for _ in range(repeats):
        start = time.perf_counter()
        steady_result = fn()
        steady_times.append(time.perf_counter() - start)
    mean_steady = sum(steady_times) / len(steady_times) if steady_times else first
    return steady_result, first, mean_steady


def _exact_reference(case: dict[str, Any]):
    if case["reference_kind"] != "exact_linear_gaussian_kalman":
        return None
    linear = affine_structural_to_linear_gaussian_tf(case["model"])
    return tf_linear_gaussian_log_likelihood(
        case["observations"],
        linear,
        backend="tf_cholesky",
        jitter=tf.constant(0.0, dtype=tf.float64),
        return_filtered=True,
    )


def _first_step_projection_errors(case: dict[str, Any], backend: str) -> dict[str, float]:
    model = case["model"]
    observations = case["observations"]
    dim = model.partition.state_dim + model.partition.innovation_dim
    dense = dense_projection_first_step(model, observations[0], nodes_per_dim=9)
    sigma = sigma_point_projection_first_step(
        model,
        observations[0],
        sigma_rule=_rule_for_backend(backend, dim),
    )
    return {
        "first_step_abs_log_likelihood_error": abs(
            float(sigma.log_likelihood.numpy()) - float(dense.log_likelihood.numpy())
        ),
        "first_step_filtered_mean_l2_error": float(
            tf.linalg.norm(sigma.filtered_mean - dense.filtered_mean).numpy()
        ),
        "first_step_filtered_covariance_fro_error": float(
            tf.linalg.norm(sigma.filtered_covariance - dense.filtered_covariance).numpy()
        ),
    }


def _exact_filtered_errors(result, reference) -> tuple[float, float]:
    if reference is None or result.filtered_means is None or result.filtered_covariances is None:
        return None, None
    mean_errors = tf.linalg.norm(result.filtered_means - reference.filtered_means, axis=1)
    cov_errors = tf.linalg.norm(
        result.filtered_covariances - reference.filtered_covariances,
        axis=[1, 2],
    )
    return float(tf.reduce_max(mean_errors).numpy()), float(tf.reduce_max(cov_errors).numpy())


def _run_row(case: dict[str, Any], backend: str, repeats: int) -> NonlinearBenchmarkRow:
    rss_before = _max_rss_mb()
    model = case["model"]
    observations = case["observations"]
    reference = _exact_reference(case)
    try:
        result, first_seconds, steady_seconds = _time_call(
            lambda: tf_nonlinear_sigma_point_value_filter(
                observations,
                model,
                backend=backend,
                return_filtered=True,
            ),
            repeats,
        )
        snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="value")
        branch = nonlinear_sigma_point_value_branch_summary(
            observations,
            case["branch_grid"],
            case["builder"],
            backend=backend,
        )
        if reference is None:
            reference_log_likelihood = None
            abs_error = None
        else:
            reference_log_likelihood = float(reference.log_likelihood.numpy())
            abs_error = abs(float(result.log_likelihood.numpy()) - reference_log_likelihood)
        first_step_errors = _first_step_projection_errors(case, backend)
        mean_error, cov_error = _exact_filtered_errors(result, reference)
        status = "ok"
        error = None
    except Exception as exc:  # pragma: no cover - benchmark artifact path.
        snapshot = None
        branch = None
        reference_log_likelihood = None
        abs_error = None
        first_step_errors = {
            "first_step_abs_log_likelihood_error": None,
            "first_step_filtered_mean_l2_error": None,
            "first_step_filtered_covariance_fro_error": None,
        }
        mean_error = None
        cov_error = None
        first_seconds = 0.0
        steady_seconds = 0.0
        result = None
        status = "error"
        error = repr(exc)
    rss_after = _max_rss_mb()

    if snapshot is None or branch is None or result is None:
        return NonlinearBenchmarkRow(
            model=case["name"],
            backend=backend,
            reference_kind=case["reference_kind"],
            timesteps=int(observations.shape[0]),
            state_dim=int(model.partition.state_dim),
            innovation_dim=int(model.partition.innovation_dim),
            observation_dim=int(model.observation_dim),
            point_count=0,
            polynomial_degree=0,
            max_integration_rank=0,
            log_likelihood=float("nan"),
            reference_log_likelihood=reference_log_likelihood,
            abs_log_likelihood_error=abs_error,
            first_step_reference_kind=None,
            first_step_abs_log_likelihood_error=None,
            first_step_filtered_mean_l2_error=None,
            first_step_filtered_covariance_fro_error=None,
            exact_filtered_mean_max_l2_error=mean_error,
            exact_filtered_covariance_max_fro_error=cov_error,
            deterministic_residual=float("nan"),
            support_residual=float("nan"),
            min_placement_eigen_gap=float("nan"),
            min_innovation_eigen_gap=float("nan"),
            branch_ok_count=0,
            branch_total_count=0,
            branch_ok_fraction=0.0,
            branch_active_floor_count=0,
            branch_weak_spectral_gap_count=0,
            branch_nonfinite_count=0,
            first_call_seconds=first_seconds,
            mean_steady_seconds=steady_seconds,
            max_rss_before_mb=rss_before,
            max_rss_after_mb=rss_after,
            status=status,
            error=error,
        )

    return NonlinearBenchmarkRow(
        model=case["name"],
        backend=backend,
        reference_kind=case["reference_kind"],
        timesteps=int(observations.shape[0]),
        state_dim=int(model.partition.state_dim),
        innovation_dim=int(model.partition.innovation_dim),
        observation_dim=int(model.observation_dim),
        point_count=snapshot.point_count,
        polynomial_degree=snapshot.polynomial_degree,
        max_integration_rank=snapshot.max_integration_rank,
        log_likelihood=float(result.log_likelihood.numpy()),
        reference_log_likelihood=reference_log_likelihood,
        abs_log_likelihood_error=abs_error,
        first_step_reference_kind="dense_one_step_gaussian_projection",
        first_step_abs_log_likelihood_error=first_step_errors[
            "first_step_abs_log_likelihood_error"
        ],
        first_step_filtered_mean_l2_error=first_step_errors[
            "first_step_filtered_mean_l2_error"
        ],
        first_step_filtered_covariance_fro_error=first_step_errors[
            "first_step_filtered_covariance_fro_error"
        ],
        exact_filtered_mean_max_l2_error=mean_error,
        exact_filtered_covariance_max_fro_error=cov_error,
        deterministic_residual=snapshot.deterministic_residual,
        support_residual=snapshot.support_residual,
        min_placement_eigen_gap=snapshot.min_placement_eigen_gap,
        min_innovation_eigen_gap=snapshot.min_innovation_eigen_gap,
        branch_ok_count=branch.ok_count,
        branch_total_count=branch.total_count,
        branch_ok_fraction=branch.ok_fraction,
        branch_active_floor_count=branch.active_floor_count,
        branch_weak_spectral_gap_count=branch.weak_spectral_gap_count,
        branch_nonfinite_count=branch.nonfinite_count,
        first_call_seconds=first_seconds,
        mean_steady_seconds=steady_seconds,
        max_rss_before_mb=rss_before,
        max_rss_after_mb=rss_after,
        status=status,
        error=error,
    )


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_logical_devices()
        ],
    }


def _markdown(payload: dict[str, Any], json_path: Path | None) -> str:
    json_name = str(json_path) if json_path is not None else "stdout"
    lines = [
        "# BayesFilter v1 Nonlinear Filter Benchmark",
        "",
        f"The JSON file is authoritative: `{json_name}`.",
        "",
        "## Claim Scope",
        "",
        "CPU-only benchmark.  Model A uses an exact linear-Gaussian Kalman "
        "reference.  Models B-C use dense one-step Gaussian projection "
        "references only, so their full log-likelihoods are recorded as "
        "filter outputs, not exact-error evidence.",
        "",
        "## Rows",
        "",
        "| Model | Backend | Reference | Loglik Error | First-Step Error | Points | Branch OK | Steady Seconds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        log_error = row["abs_log_likelihood_error"]
        first_error = row["first_step_abs_log_likelihood_error"]
        lines.append(
            "| {model} | {backend} | {reference} | {log_error} | {first_error} | "
            "{points} | {ok}/{total} | {seconds:.6f} |".format(
                model=row["model"],
                backend=row["backend"],
                reference=row["reference_kind"],
                log_error="n/a" if log_error is None else f"{log_error:.3e}",
                first_error="n/a" if first_error is None else f"{first_error:.3e}",
                points=row["point_count"],
                ok=row["branch_ok_count"],
                total=row["branch_total_count"],
                seconds=row["mean_steady_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "CUT4 point counts are larger than cubature and UKF.  This artifact "
            "is designed to test whether that larger rule improves the small "
            "nonlinear moment-projection rows enough to justify further GPU/XLA "
            "or HMC-specific work.",
        ]
    )
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-12.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-12.md",
    )
    args = parser.parse_args()

    rows = [
        _run_row(case, backend, repeats=args.repeats)
        for case in _model_cases()
        for backend in BACKENDS
    ]
    payload = _json_safe({
        "benchmark": "bayesfilter_v1_nonlinear_filters",
        "claim_scope": "cpu_value_and_one_step_projection_only",
        "environment": _environment(),
        "repeats": args.repeats,
        "rows": [asdict(row) for row in rows],
        "blocked_claims": [
            "full_exact_nonlinear_likelihood_for_models_b_c",
            "nonlinear_models_b_c_analytic_score",
            "nonlinear_hmc_readiness",
            "gpu_xla_speedup",
        ],
    })
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
