"""Reproducible two-lane actual-SV comparison harness.

This benchmark emits a JSON artifact and optional markdown summary for:

- Lane A exact-transformed actual-SV dense and SGQF value/score comparisons;
- Lane B augmented-noise Gaussian-closure dense, SGQF, and UKF comparisons;
- optional KSC surrogate rows kept explicitly separate;
- optional low-dimensional control-model score rows for Models B and C.

The harness is intentionally tiny-fixture focused.  It is a research-grade
comparison artifact, not a production performance benchmark.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import bayesfilter.highdim as highdim
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_log_likelihood
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_log_likelihood
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
    tf_svd_cubature_score,
    tf_svd_cut4_score,
    tf_svd_ukf_score,
)
from bayesfilter.testing import (
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_b_observations_tf,
    model_c_observations_tf,
)

_STD_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, dtype=tf.float64),
    scale=tf.constant(1.0, dtype=tf.float64),
)

NONCLAIMS = (
    "simulation-driven comparison harness only",
    "not a production timing benchmark",
    "not HMC convergence evidence",
    "Routes are compared only after exact transformation correction and within-target validation",
    "KSC surrogate rows are reported separately from actual-SV rows",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dims", default="1,2,3")
    parser.add_argument("--include-ksc", action="store_true")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--lane-a-order", type=int, default=401)
    parser.add_argument("--lane-a-radius", type=float, default=8.0)
    parser.add_argument("--lane-a-sparse-level", type=int, default=2)
    parser.add_argument("--lane-b-order", type=int, default=81)
    parser.add_argument("--lane-b-radius", type=float, default=7.0)
    parser.add_argument("--lane-b-sparse-level", type=int, default=4)
    parser.add_argument("--fd-step", type=float, default=1e-5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def _parse_dims(value: str) -> list[int]:
    dims = [int(part) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("dims must not be empty")
    if any(dim <= 0 or dim > 3 for dim in dims):
        raise ValueError("dims must be between 1 and 3 for this harness")
    return dims


def _observations(dim: int, seed: int = 83120, horizon: int = 20) -> tf.Tensor:
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import generate_frozen_exact_sv_dataset_tf

    columns = []
    for axis in range(int(dim)):
        _states, observations = generate_frozen_exact_sv_dataset_tf(
            seed=seed + 1000 * axis, horizon=horizon
        )
        columns.append(observations[:, 0])
    return tf.stack(columns, axis=1)


def _physical_parameters(dim: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gamma = tf.constant([0.60, 0.52, 0.47], dtype=tf.float64)[: int(dim)]
    beta = tf.constant([0.40, 0.35, 0.45], dtype=tf.float64)[: int(dim)]
    sigma = tf.constant([1.00, 0.85, 0.75], dtype=tf.float64)[: int(dim)]
    return gamma, beta, sigma


def _theta_from_physical(gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.stack([_STD_NORMAL.quantile(gamma), tf.math.log(beta)], axis=1), [-1])


def _physical_from_theta(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    theta_matrix = tf.reshape(tf.convert_to_tensor(theta, dtype=tf.float64), [-1, 2])
    return _STD_NORMAL.cdf(theta_matrix[:, 0]), tf.exp(theta_matrix[:, 1])


def _centered_finite_difference_score(value_fn, theta: tf.Tensor, step: float) -> tf.Tensor:
    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    score = []
    for axis in range(int(theta_tensor.shape[0])):
        direction = tf.one_hot(axis, int(theta_tensor.shape[0]), dtype=tf.float64) * tf.constant(step, dtype=tf.float64)
        plus = value_fn(theta_tensor + direction)
        minus = value_fn(theta_tensor - direction)
        score.append((plus - minus) / (2.0 * tf.constant(step, dtype=tf.float64)))
    return tf.stack(score)


def _relative_error(candidate: tf.Tensor, reference: tf.Tensor) -> float:
    value = tf.linalg.norm(candidate - reference) / tf.maximum(
        tf.constant(1.0, dtype=tf.float64),
        tf.linalg.norm(reference),
    )
    return float(value.numpy())


def _lane_a_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor, *, sparse_level: int) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.exact_transformed_sv_independent_panel_fixed_sgqf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=sparse_level,
    ).log_likelihood


def _lane_b_sgqf_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor, *, sparse_level: int) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=sparse_level,
    ).log_likelihood


def _lane_b_ukf_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.actual_transformed_sv_independent_panel_augmented_noise_ukf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _ksc_kalman_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_kalman_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _collect_actual_sv_rows(args: argparse.Namespace, dims: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim in dims:
        observations = _observations(dim, seed=83120 + dim, horizon=20)
        gamma, beta, sigma = _physical_parameters(dim)
        theta = _theta_from_physical(gamma, beta)

        lane_a_dense = highdim.exact_transformed_sv_independent_panel_dense_reference(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            order=args.lane_a_order,
            radius=args.lane_a_radius,
        )
        lane_a_sgqf = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_filter(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            sparse_level=args.lane_a_sparse_level,
        )
        lane_a_score = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_score(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            sparse_level=args.lane_a_sparse_level,
        )
        lane_a_fd = _centered_finite_difference_score(
            lambda current_theta: _lane_a_value(
                current_theta,
                observations,
                sigma,
                sparse_level=args.lane_a_sparse_level,
            ),
            theta,
            args.fd_step,
        )

        lane_b_dense = highdim.actual_transformed_sv_independent_panel_augmented_noise_dense_gaussian_closure_reference(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            order=args.lane_b_order,
            radius=args.lane_b_radius,
        )
        lane_b_dense_score = _centered_finite_difference_score(
            lambda current_theta: highdim.actual_transformed_sv_independent_panel_augmented_noise_dense_gaussian_closure_reference(
                observations,
                gamma=_physical_from_theta(current_theta)[0],
                beta=_physical_from_theta(current_theta)[1],
                sigma=sigma,
                order=args.lane_b_order,
                radius=args.lane_b_radius,
            ).log_likelihood,
            theta,
            args.fd_step,
        )
        lane_b_sgqf = highdim.actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            sparse_level=args.lane_b_sparse_level,
        )
        lane_b_sgqf_score = highdim.actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_score(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            sparse_level=args.lane_b_sparse_level,
        )
        lane_b_sgqf_fd = _centered_finite_difference_score(
            lambda current_theta: _lane_b_sgqf_value(
                current_theta,
                observations,
                sigma,
                sparse_level=args.lane_b_sparse_level,
            ),
            theta,
            args.fd_step,
        )

        lane_b_ukf = highdim.actual_transformed_sv_independent_panel_augmented_noise_ukf_filter(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
        )
        lane_b_ukf_score = highdim.actual_transformed_sv_independent_panel_augmented_noise_ukf_score(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
        )
        lane_b_ukf_fd = _centered_finite_difference_score(
            lambda current_theta: _lane_b_ukf_value(current_theta, observations, sigma),
            theta,
            args.fd_step,
        )

        rows.append(
            {
                "dim": dim,
                "lane_a": {
                    "dense_log_likelihood": float(lane_a_dense.log_likelihood.numpy()),
                    "sgqf_log_likelihood": float(lane_a_sgqf.log_likelihood.numpy()),
                    "value_gap": float(abs((lane_a_sgqf.log_likelihood - lane_a_dense.log_likelihood).numpy())),
                    "score_relative_error": _relative_error(lane_a_score.score, lane_a_fd),
                    "sparse_level": args.lane_a_sparse_level,
                },
                "lane_b": {
                    "dense_log_likelihood": float(lane_b_dense.log_likelihood.numpy()),
                    "sgqf_log_likelihood": float(lane_b_sgqf.log_likelihood.numpy()),
                    "ukf_log_likelihood": float(lane_b_ukf.log_likelihood.numpy()),
                    "sgqf_value_gap": float(abs((lane_b_sgqf.log_likelihood - lane_b_dense.log_likelihood).numpy())),
                    "ukf_value_gap": float(abs((lane_b_ukf.log_likelihood - lane_b_dense.log_likelihood).numpy())),
                    "sgqf_score_relative_error": _relative_error(lane_b_sgqf_score.score, lane_b_sgqf_fd),
                    "ukf_score_relative_error": _relative_error(lane_b_ukf_score.score, lane_b_ukf_fd),
                    "sgqf_gradient_relative_error_to_dense": _relative_error(lane_b_sgqf_score.score, lane_b_dense_score),
                    "ukf_gradient_relative_error_to_dense": _relative_error(lane_b_ukf_score.score, lane_b_dense_score),
                    "sgqf_gradient_abs_gap_norm_to_dense": float(tf.linalg.norm(lane_b_sgqf_score.score - lane_b_dense_score).numpy()),
                    "ukf_gradient_abs_gap_norm_to_dense": float(tf.linalg.norm(lane_b_ukf_score.score - lane_b_dense_score).numpy()),
                    "sgqf_sparse_level": args.lane_b_sparse_level,
                },
                "cross_lane": {
                    "dense_value_gap": float(abs((lane_b_dense.log_likelihood - lane_a_dense.log_likelihood).numpy())),
                },
                "simulation": {
                    "seed": 83120 + dim,
                    "horizon": 20,
                    "fixture_kind": "simulated_exact_actual_sv_path",
                    "sample_count": int(observations.shape[0]),
                },
            }
        )
    return rows


def _collect_ksc_rows(dims: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim in dims:
        observations = _observations(dim)
        gamma, beta, sigma = _physical_parameters(dim)
        theta = _theta_from_physical(gamma, beta)
        value = _ksc_kalman_value(theta, observations, sigma)
        rows.append(
            {
                "dim": dim,
                "kalman_mixture_log_likelihood": float(value.numpy()),
                "nonclaim": "surrogate row only, not actual-SV evidence",
            }
        )
    return rows


def _model_b_and_derivatives(params: tf.Tensor):
    model = make_nonlinear_accumulation_model_tf(rho=params[0], sigma=params[1], beta=params[2])
    derivatives = make_nonlinear_accumulation_first_derivatives_tf(rho=params[0], sigma=params[1], beta=params[2])
    return model, derivatives


def _model_b_value(params: tf.Tensor, backend: str) -> tf.Tensor:
    model, _ = _model_b_and_derivatives(params)
    if backend == "tf_svd_cut4":
        value, _means, _covariances, _diagnostics = tf_svd_cut4_log_likelihood(
            model_b_observations_tf(), model, innovation_floor=tf.constant(1e-12, dtype=tf.float64)
        )
        return value
    rule = "cubature" if backend == "tf_svd_cubature" else "unscented"
    value, _means, _covariances, _diagnostics = tf_svd_sigma_point_log_likelihood(
        model_b_observations_tf(), model, rule=rule, innovation_floor=tf.constant(1e-12, dtype=tf.float64)
    )
    return value


def _model_c_and_derivatives(params: tf.Tensor):
    model = make_univariate_nonlinear_growth_model_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
        initial_variance=params[2],
        initial_phase_variance=tf.constant(0.05, dtype=tf.float64),
    )
    derivatives = make_univariate_nonlinear_growth_first_derivatives_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
    )
    return model, derivatives


def _model_c_value(params: tf.Tensor, backend: str) -> tf.Tensor:
    model, _ = _model_c_and_derivatives(params)
    if backend == "tf_svd_cut4":
        value, _means, _covariances, _diagnostics = tf_svd_cut4_log_likelihood(
            model_c_observations_tf(), model, innovation_floor=tf.constant(1e-12, dtype=tf.float64)
        )
        return value
    rule = "cubature" if backend == "tf_svd_cubature" else "unscented"
    value, _means, _covariances, _diagnostics = tf_svd_sigma_point_log_likelihood(
        model_c_observations_tf(), model, rule=rule, innovation_floor=tf.constant(1e-12, dtype=tf.float64)
    )
    return value


def _collect_control_rows(fd_step: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params_b = tf.constant([0.70, 0.25, 0.80], dtype=tf.float64)
    model_b, derivatives_b = _model_b_and_derivatives(params_b)
    for score_fn, backend in (
        (tf_svd_cubature_score, "tf_svd_cubature"),
        (tf_svd_ukf_score, "tf_svd_ukf"),
        (tf_svd_cut4_score, "tf_svd_cut4"),
    ):
        analytic = score_fn(
            model_b_observations_tf(),
            model_b,
            derivatives_b,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        )
        finite = _centered_finite_difference_score(lambda values: _model_b_value(values, backend), params_b, fd_step)
        rows.append(
            {
                "model": "model_b_nonlinear_accumulation",
                "backend": backend,
                "log_likelihood": float(analytic.log_likelihood.numpy()),
                "score_relative_error": _relative_error(analytic.score, finite),
            }
        )

    params_c = tf.constant([1.0, 1.0, 0.20], dtype=tf.float64)
    model_c, derivatives_c = _model_c_and_derivatives(params_c)
    for score_fn, backend in (
        (tf_svd_cubature_score, "tf_svd_cubature"),
        (tf_svd_ukf_score, "tf_svd_ukf"),
        (tf_svd_cut4_score, "tf_svd_cut4"),
    ):
        analytic = score_fn(
            model_c_observations_tf(),
            model_c,
            derivatives_c,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        )
        finite = _centered_finite_difference_score(lambda values: _model_c_value(values, backend), params_c, fd_step)
        rows.append(
            {
                "model": "model_c_smooth_phase_growth",
                "backend": backend,
                "log_likelihood": float(analytic.log_likelihood.numpy()),
                "score_relative_error": _relative_error(analytic.score, finite),
            }
        )
    return rows


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Actual-SV Two-Lane Comparison",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Dims: `{result['dims']}`",
        f"- Include KSC surrogate rows: `{result['include_ksc']}`",
        f"- Include control rows: `{result['include_controls']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    lines.extend(["", "## Actual-SV rows", ""])
    for row in result["actual_sv_rows"]:
        lines.append(
            f"- dim {row['dim']}: Lane-A gap={row['lane_a']['value_gap']:.6g}, "
            f"Lane-B SGQF gap={row['lane_b']['sgqf_value_gap']:.6g}, "
            f"Lane-B UKF gap={row['lane_b']['ukf_value_gap']:.6g}, "
            f"cross-lane gap={row['cross_lane']['dense_value_gap']:.6g}"
        )
    if result["ksc_rows"]:
        lines.extend(["", "## KSC surrogate rows", ""])
        for row in result["ksc_rows"]:
            lines.append(
                f"- dim {row['dim']}: log-likelihood={row['kalman_mixture_log_likelihood']:.6g} ({row['nonclaim']})"
            )
    if result["control_rows"]:
        lines.extend(["", "## Control rows", ""])
        for row in result["control_rows"]:
            lines.append(
                f"- {row['model']} / {row['backend']}: log-likelihood={row['log_likelihood']:.6g}, "
                f"score_relerr={row['score_relative_error']:.6g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    dims = _parse_dims(args.dims)
    result: dict[str, Any] = {
        "schema_version": "actual_sv_two_lane.v1",
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dims": dims,
        "include_ksc": bool(args.include_ksc),
        "include_controls": bool(args.include_controls),
        "lane_a_order": args.lane_a_order,
        "lane_a_radius": args.lane_a_radius,
        "lane_a_sparse_level": args.lane_a_sparse_level,
        "lane_b_order": args.lane_b_order,
        "lane_b_radius": args.lane_b_radius,
        "lane_b_sparse_level": args.lane_b_sparse_level,
        "fd_step": args.fd_step,
        "actual_sv_rows": _collect_actual_sv_rows(args, dims),
        "ksc_rows": _collect_ksc_rows(dims) if args.include_ksc else [],
        "control_rows": _collect_control_rows(args.fd_step) if args.include_controls else [],
        "nonclaims": list(NONCLAIMS),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_path, result, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
