#!/usr/bin/env python3
"""Bounded mechanics and GPU/XLA feasibility for the TT moment teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


PLAN = Path("docs/plans/bayesfilter-zhao-cui-moment-teacher-plan-2026-07-30.md")
RESULT_NOTE = Path(
    "docs/plans/bayesfilter-zhao-cui-moment-teacher-result-2026-07-30.md"
)
SCHEMA = "bayesfilter.zhao_cui_moment_teacher_mechanics.v1"
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
SOURCE_PATHS = (
    Path("bayesfilter/highdim/zhao_cui_moment_teacher.py"),
    Path("bayesfilter/highdim/higher_moment_contract_e.py"),
    Path("docs/benchmarks/run_zhao_cui_moment_teacher_mechanics.py"),
    Path("tests/highdim/test_zhao_cui_moment_teacher.py"),
)


def _safe(value):
    if isinstance(value, tf.Tensor):
        raw = value.numpy()
        return raw.item() if value.shape.rank == 0 else raw.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def _source_hashes() -> dict[str, str]:
    return {
        str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _convention(highdim):
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _gpu_xla_probe(moment_teacher) -> dict[str, object]:
    axis_count = 40
    rank = 8
    basis_size = 6
    raw = tf.random.stateless_normal(
        [axis_count, rank, basis_size, rank], [711, 712], dtype=tf.float32
    ) * tf.constant(0.015, tf.float32)
    identity_channel = tf.eye(rank, dtype=tf.float32)[None, :, None, :]
    basis_mask = tf.one_hot(0, basis_size, dtype=tf.float32)[None, None, :, None]
    cores = raw + identity_channel * basis_mask
    left_mask = tf.one_hot(0, rank, dtype=tf.float32)[:, None, None]
    right_mask = tf.one_hot(0, rank, dtype=tf.float32)[None, None, :]
    cores = tf.concat(
        [
            cores[:1] * left_mask[None, :, :, :],
            cores[1:-1],
            cores[-1:] * right_mask[None, :, :, :],
        ],
        axis=0,
    )
    dot_cores = tf.random.stateless_normal(
        tf.shape(cores), [713, 714], dtype=tf.float32
    ) * tf.constant(1e-3, tf.float32)
    dot_cores = tf.concat(
        [
            dot_cores[:1] * left_mask[None, :, :, :],
            dot_cores[1:-1],
            dot_cores[-1:] * right_mask[None, :, :, :],
        ],
        axis=0,
    )
    mass = tf.broadcast_to(
        tf.eye(basis_size, dtype=tf.float32)[None, :, :],
        [axis_count, basis_size, basis_size],
    )
    observable_axis = tf.linalg.diag(
        tf.linspace(tf.constant(0.85, tf.float32), tf.constant(1.15, tf.float32), basis_size)
    )
    observable = tf.broadcast_to(
        observable_axis[None, :, :], [axis_count, basis_size, basis_size]
    )
    zeros = tf.zeros_like(observable)
    arguments = (
        cores,
        dot_cores,
        observable,
        zeros,
        mass,
        zeros,
        tf.constant(0.01, tf.float32),
        tf.constant(0.0, tf.float32),
        tf.constant(0.5, tf.float32),
        tf.constant(0.0, tf.float32),
        tf.constant(1.0, tf.float32),
        tf.constant(0.0, tf.float32),
    )
    moment_teacher.padded_squared_tt_observable_jvp_xla(*arguments)
    tf.config.experimental.reset_memory_stats("GPU:0")
    started = time.perf_counter()
    result = None
    repetitions = 25
    for _ in range(repetitions):
        result = moment_teacher.padded_squared_tt_observable_jvp_xla(*arguments)
    # Host materialization is the TensorFlow 2.19 synchronization boundary.
    tuple(item.numpy() for item in result)
    elapsed = time.perf_counter() - started
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    concrete = moment_teacher.padded_squared_tt_observable_jvp_xla.get_concrete_function(
        *arguments
    )
    operations = sorted({node.op for node in concrete.graph.as_graph_def().node})
    return {
        "axis_count": axis_count,
        "padded_rank": rank,
        "basis_size": basis_size,
        "dtype": "float32",
        "tf32": True,
        "jit_compile": True,
        "repetitions": repetitions,
        "wall_time_seconds": elapsed,
        "mean_seconds_per_call": elapsed / repetitions,
        "value": result[0],
        "tangent": result[1],
        "normalizer": result[2],
        "normalizer_tangent": result[3],
        "finite": tf.reduce_all(tf.math.is_finite(tf.stack(result))),
        "allocator_current_bytes": int(allocator["current"]),
        "allocator_peak_bytes": int(allocator["peak"]),
        "graph_has_while": "StatelessWhile" in operations or "While" in operations,
        "graph_has_pyfunc": "PyFunc" in operations or "EagerPyFunc" in operations,
        "graph_operation_types": operations,
        "scope": "contraction_primitive_only_no_tt_fit_no_filter",
    }


def _lgssm_probe(highdim, moment_teacher) -> dict[str, object]:
    convention = _convention(highdim)
    product = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 28)],
        convention,
    )
    config = highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1), ridge=1e-12, max_sweeps=1, sweep_order=(0,),
            row_budget=128, column_budget=64, dense_matrix_byte_budget=300_000,
            normal_matrix_byte_budget=10_000, condition_number_warning=1e10,
            condition_number_veto=1e14, holdout_tolerance=1e6,
        ),
        density_tau=0.0, normalizer_floor=1e-14, denominator_floor=1e-14,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(highdim.AffineCoordinateMap(
            offset=tf.constant([0.0], tf.float64),
            matrix=tf.constant([[2.0]], tf.float64),
        ),),
        measure_convention=convention,
        deterministic_seed="tt-moment-teacher-lgssm-t1",
        product_basis=product,
        fit_quadrature_order=64,
    )
    model = highdim.LinearGaussianSSM(
        initial_mean=tf.constant([0.0], tf.float64),
        initial_covariance=tf.constant([[1.0]], tf.float64),
        transition_matrix=tf.constant([[0.7]], tf.float64),
        transition_covariance=tf.constant([[0.25]], tf.float64),
        observation_matrix=tf.constant([[1.0]], tf.float64),
        observation_covariance=tf.constant([[0.09]], tf.float64),
    )
    started = time.perf_counter()
    result = highdim.FixedBranchSquaredTTFilter(config).log_likelihood(
        model, tf.zeros([0], tf.float64), tf.constant([0.2], tf.float64)
    )
    elapsed = time.perf_counter() - started
    density = result.steps[0].density
    if density is None:
        raise RuntimeError("LGSSM TT density was not produced")
    moments = moment_teacher.squared_tt_reference_moments(density, (0,))
    tt_mean = 2.0 * moments.mean[0]
    tt_variance = 4.0 * moments.covariance[0, 0]
    kalman_mean = result.retained_filter.diagnostics["mean"][0]
    kalman_variance = result.retained_filter.diagnostics["covariance"][0, 0]
    coefficient = tf.math.rsqrt(moments.covariance[0, 0])
    offset = -moments.mean[0] * coefficient
    skew = moment_teacher.squared_tt_affine_form_moment(
        density, tf.reshape(coefficient, [1]), offset, 3
    )
    kurtosis = moment_teacher.squared_tt_affine_form_moment(
        density, tf.reshape(coefficient, [1]), offset, 4
    )
    return {
        "scope": "scalar_lgssm_t1_fitted_density_reference_diagnostic",
        "dtype": "float64",
        "state_dimension": 1,
        "basis_degree": 28,
        "fit_quadrature_order": 64,
        "tt_rank_tuple": density.sqrt_tt.rank_tuple(),
        "fit_wall_time_seconds": elapsed,
        "fit_residual": result.steps[0].fit_result.fit_residual,
        "kalman_log_likelihood": result.log_likelihood,
        "kalman_mean": kalman_mean,
        "tt_mean": tt_mean,
        "mean_abs_error": tf.abs(tt_mean - kalman_mean),
        "kalman_variance": kalman_variance,
        "tt_variance": tt_variance,
        "variance_abs_error": tf.abs(tt_variance - kalman_variance),
        "tt_standardized_skew": skew,
        "skew_abs_error_from_wick": tf.abs(skew),
        "tt_standardized_kurtosis": kurtosis,
        "kurtosis_abs_error_from_wick": tf.abs(kurtosis - 3.0),
        "classification": "finite_fit_diagnostic_not_exact_gaussian_representation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter import highdim
    from bayesfilter.highdim import zhao_cui_moment_teacher as moment_teacher

    started = time.perf_counter()
    payload = {
        "schema": SCHEMA,
        "plan": str(PLAN),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": _source_hashes(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
        },
        "device": {
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "memory_policy": memory_policy,
            "trust_basis": TRUST_BASIS,
        },
        "gpu_xla_contraction": _gpu_xla_probe(moment_teacher),
        "lgssm_fitted_teacher": _lgssm_probe(highdim, moment_teacher),
        "hard_vetoes": {
            "nonfinite": False,
            "graph_pyfunc": False,
            "missing_graph_while": False,
        },
        "nonclaims": [
            "no nonlinear filtering result",
            "no recursive TT fit score",
            "no HMC readiness",
            "no default promotion",
            "no high-dimensional rank-boundedness claim",
        ],
    }
    payload["wall_time_seconds"] = time.perf_counter() - started
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "bayesfilter.run_manifest.v1",
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": digest,
        "plan": str(PLAN),
        "git_commit": payload["git_commit"],
        "source_sha256": payload["source_sha256"],
        "command": payload["command"],
        "environment": payload["environment"],
        "device": _safe(payload["device"]),
        "random_seeds": [711, 712, 713, 714],
        "data_version": "deterministic synthetic mechanics fixtures",
        "wall_time_seconds": payload["wall_time_seconds"],
        "artifact_paths": [str(result_path.relative_to(ROOT))],
        "result_file": str(RESULT_NOTE),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
