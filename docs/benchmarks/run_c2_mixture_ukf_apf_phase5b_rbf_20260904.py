"""Run the bounded Phase 5B fixed-map Gaussian-RBF pilot.

The C2 map, exact target, and independent predictive normalizer are reused
from the closed Phase 5A diagnostic.  Only the one-dimensional basis changes.
The RBF mass and integral matrices are analytic and are checked against a
TensorFlow Jacobi-matrix Gaussian quadrature rule before fitting.  This is a
representation diagnostic, not a proposal or filtering implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf


DTYPE = tf.float64
STATE_DIM = 4
CARRIED_ROWS = 128
TRAIN_ROWS = 512
HOLDOUT_ROWS = 512
AUDIT_ROWS = 4096
TT_RANK = 2
ALS_SWEEPS = 2
RIDGE = 1.0e-8
CONDITION_VETO = 1.0e14
# Match the Phase 5A paired bank seed so representation records are directly
# comparable; the output directory, phase identity, and arm labels remain new.
SEED = (20260904, 701)
MAP_SEED = (20260904, 501)
SHELL_RADIUS = 2.0
# The narrow width arm needs a higher order to resolve its shifted Gaussian
# product at the hard analytic-vs-quadrature audit tolerance.
QUADRATURE_ORDER = 80
CENTERS = (-2.0, -1.0, 0.0, 1.0, 2.0)
ARMS = (
    ("rbf_w075", 0.75),
    ("rbf_w150", 1.50),
    ("rbf_w300", 3.00),
)

PHASE5A_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5a_hermite_20260904.py"
RBF_MODULE_PATH = ROOT / "bayesfilter/highdim/rbf_basis_tf.py"
FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"
MOMENT_MAP_PATH = ROOT / "bayesfilter/highdim/recursive_moment_map_tf.py"
PLAN_PATH = ROOT / "docs/plans/c2-mixture-ukf-apf-phase5b-rbf-pilot-20260904.md"

PHASE_ID = "c2_mixture_ukf_apf_phase5b_rbf_pilot_v1"
RESULT_SCHEMA = "c2_mixture_ukf_apf_phase5b_rbf_result_v1"
MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase5b_rbf_manifest_v1"
ROUTE_ID = "c2_fixed_map_rbf_tt_representation_pilot_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"

_PHASE5A: Any | None = None
RBFBasis1D: Any | None = None


def _load_rbf_basis() -> Any:
    global RBFBasis1D
    if RBFBasis1D is None:
        from bayesfilter.highdim.rbf_basis_tf import RBFBasis1D as _RBFBasis1D

        RBFBasis1D = _RBFBasis1D
    return RBFBasis1D


def _load_phase5a() -> Any:
    global _PHASE5A
    if _PHASE5A is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "c2_phase5a_shared_helpers", PHASE5A_DRIVER_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load Phase 5A shared helper module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._load_modules()
        _PHASE5A = module
    return _PHASE5A


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _workspace_manifest() -> Mapping[str, object]:
    status = _git("status", "--porcelain=v1")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": status,
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite artifact value: {value!r}")
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite(value: object) -> bool:
    return bool(
        tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value, DTYPE))).numpy()
    )


def _scalar(value: object) -> float:
    tensor = tf.reshape(tf.convert_to_tensor(value, DTYPE), [])
    if not _finite(tensor):
        raise ValueError("non-finite scalar")
    return float(tensor.numpy())


def _max_abs(value: object) -> float:
    return _scalar(tf.reduce_max(tf.abs(tf.convert_to_tensor(value, DTYPE))))


def _configure_runtime() -> Mapping[str, object]:
    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    physical = tuple(tf.config.list_physical_devices("GPU"))
    if physical:
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        logical = tuple(tf.config.list_logical_devices("GPU"))
        if not logical:
            raise RuntimeError("visible physical GPU produced no logical GPU")
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.ones([32], DTYPE))
        if "GPU" not in str(probe.device).upper():
            raise RuntimeError(f"GPU placement probe ran on {probe.device}")
        return {
            "execution_lane": "trusted_gpu_xla_phase5b_rbf",
            "memory_policy": policy,
            "physical_devices": [str(device.name) for device in physical],
            "logical_devices": [str(device.name) for device in logical],
            "placement_probe_device": str(probe.device),
            "placement_probe_value": _scalar(probe),
        }
    return {
        "execution_lane": "cpu_reference_or_debug",
        "memory_policy": {
            "schema": "bayesfilter.tensorflow.gpu_memory_policy.v1",
            "mode": "cpu_no_visible_gpu",
            "configured_before_logical_device_initialization": True,
        },
        "physical_devices": [str(device.name) for device in tf.config.list_physical_devices()],
        "logical_devices": [str(device.name) for device in tf.config.list_logical_devices()],
    }


def _make_output_root(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _gaussian_quadrature(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Construct a standard-normal Gauss-Hermite rule without NumPy.

    The eigenvalues of the symmetric Jacobi matrix with off-diagonal
    entries sqrt(k) are the probabilists' Hermite nodes; the squared first
    eigenvector entries are the normalized weights.
    """

    order = int(order)
    if order < 2:
        raise ValueError("quadrature order must be at least two")
    off_diagonal = tf.sqrt(tf.cast(tf.range(1, order), DTYPE))
    jacobi = tf.linalg.diag(off_diagonal, k=1) + tf.linalg.diag(off_diagonal, k=-1)
    nodes, vectors = tf.linalg.eigh(jacobi)
    weights = tf.square(vectors[0, :])
    return nodes, weights


def _basis_contractions(basis: RBFBasis1D) -> Mapping[str, object]:
    mass = basis.mass_matrix(_load_phase5a().MassMeasure.REFERENCE_MEASURE)
    integral = basis.integral_vector(_load_phase5a().MassMeasure.REFERENCE_MEASURE)
    nodes, weights = _gaussian_quadrature(QUADRATURE_ORDER)
    values = basis.evaluate(nodes)
    numerical_mass = tf.einsum("n,ni,nj->ij", weights, values, values)
    numerical_integral = tf.einsum("n,ni->i", weights, values)
    mass_error = _max_abs(mass - numerical_mass)
    integral_error = _max_abs(integral - numerical_integral)
    eigenvalues = tf.linalg.eigvalsh(mass)
    minimum = _scalar(eigenvalues[0])
    maximum = _scalar(eigenvalues[-1])
    condition = maximum / max(minimum, 1.0e-300)
    return {
        "mass": mass,
        "integral": integral,
        "quadrature_mass_error": mass_error,
        "quadrature_integral_error": integral_error,
        "mass_minimum_eigenvalue": minimum,
        "mass_maximum_eigenvalue": maximum,
        "mass_condition_number": condition,
        "quadrature_order": QUADRATURE_ORDER,
        "finite": _finite(mass) and _finite(integral),
        "spd": minimum > 0.0,
    }


def _gram_squared_normalizer(cores: Sequence[Any], basis: Any) -> tf.Tensor:
    state = tf.ones([1, 1], DTYPE)
    for axis, core in enumerate(cores):
        mass = basis.bases[axis].mass_matrix(
            _load_phase5a().MassMeasure.REFERENCE_MEASURE
        )
        state = tf.einsum(
            "akb,AlB,kl,aA->bB", core.values, core.values, mass, state
        )
    return tf.reshape(state, [])


def _rms(values: tf.Tensor, weights: tf.Tensor | None = None) -> float:
    values = tf.convert_to_tensor(values, DTYPE)
    if weights is None:
        return _scalar(tf.sqrt(tf.reduce_mean(tf.square(values))))
    weights = tf.convert_to_tensor(weights, DTYPE)
    return _scalar(tf.sqrt(tf.reduce_sum(weights * tf.square(values)) / tf.reduce_sum(weights)))


def _condition_summary(result: Any) -> Mapping[str, object]:
    values: list[float] = []
    unscaled: list[float] = []
    for record in result.core_update_statuses:
        for key in ("transformed_system_condition_number", "condition_number"):
            value = record.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
                break
        value = record.get("unscaled_normal_condition_number")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            unscaled.append(float(value))
    return {
        "scaled_augmented_condition_max": max(values) if values else None,
        "unscaled_normal_condition_max_diagnostic": max(unscaled) if unscaled else None,
        "update_count": len(result.core_update_statuses),
    }


def _fit_arm(
    name: str,
    width: float,
    *,
    shared: Any,
    banks: Mapping[str, tf.Tensor],
    sqrt_targets: Mapping[str, tf.Tensor],
    log_targets: Mapping[str, tf.Tensor],
    direct_target: Mapping[str, tf.Tensor],
    shift: tf.Tensor,
) -> Mapping[str, object]:
    p5a = _load_phase5a()
    convention = p5a.MeasureConvention(
        density_measure=p5a.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=p5a.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="standard_normal",
        physical_coordinate_name="x",
        reference_coordinate_name="u",
    )
    one_dimensional = RBFBasis1D(CENTERS, width, include_constant=True)
    contraction = _basis_contractions(one_dimensional)
    basis = p5a.ProductBasis([one_dimensional for _ in range(STATE_DIM)], convention)
    fit_config = p5a._fit_config(0)
    sample_weights = tf.fill([TRAIN_ROWS], tf.constant(1.0 / TRAIN_ROWS, DTYPE))
    holdout_weights = tf.fill([HOLDOUT_ROWS], tf.constant(1.0 / HOLDOUT_ROWS, DTYPE))
    samples = p5a.FixedTTFitSampleBatch(
        points=banks["train"],
        target_values=sqrt_targets["train"],
        weights=sample_weights,
        holdout_points=banks["holdout"],
        holdout_values=sqrt_targets["holdout"],
        holdout_weights=holdout_weights,
    )
    initial = p5a._initial_tt_cores(STATE_DIM, one_dimensional.basis_dim, TT_RANK)
    fit_result = p5a.FixedTTFitter().fit(
        basis,
        samples,
        fit_config,
        initial,
        branch_seed=f"phase5b-{name}",
        measure_convention=convention,
        initialization_rule="constant_channel_identity_v1",
    )
    fitted = fit_result.fitted_tt
    holdout_prediction = fitted.evaluate(banks["holdout"])
    audit_prediction = fitted.evaluate(banks["audit"])
    holdout_residual = holdout_prediction - sqrt_targets["holdout"]
    shell_mask = tf.reduce_max(tf.abs(banks["holdout"]), axis=1) >= SHELL_RADIUS
    central_mask = tf.logical_not(shell_mask)
    if not bool(tf.reduce_any(shell_mask).numpy()) or not bool(tf.reduce_any(central_mask).numpy()):
        raise ValueError("holdout bank lacks central or shell rows")
    gram_scaled = _gram_squared_normalizer(fitted.cores, basis)
    z_t = tf.convert_to_tensor(direct_target["z_t"], DTYPE)
    direct_se = tf.convert_to_tensor(direct_target["standard_error"], DTYPE)
    reference_values = tf.exp(log_targets["audit"] - shift)
    reference_z_t = tf.exp(shift) * tf.reduce_mean(reference_values)
    reference_se = tf.exp(shift) * tf.math.reduce_std(reference_values) / tf.sqrt(
        tf.cast(AUDIT_ROWS, DTYPE)
    )
    z_h = tf.exp(shift) * gram_scaled
    condition = _condition_summary(fit_result)
    condition_max = condition["scaled_augmented_condition_max"]
    status_ok = str(fit_result.status.value) == "OK"
    finite = all(
        _finite(value)
        for value in (
            holdout_prediction,
            audit_prediction,
            gram_scaled,
            z_h,
            z_t,
            direct_se,
            reference_z_t,
            reference_se,
        )
    )
    hard_valid = bool(
        finite
        and bool(contraction["finite"])
        and bool(contraction["spd"])
        and contraction["quadrature_mass_error"] <= 2.0e-11
        and contraction["quadrature_integral_error"] <= 2.0e-11
        and status_ok
        and condition_max is not None
        and float(condition_max) <= CONDITION_VETO
    )
    return {
        "arm": name,
        "width": float(width),
        "centers": CENTERS,
        "include_constant": True,
        "basis_family": "gaussian_rbf_reference",
        "basis_dim_per_axis": one_dimensional.basis_dim,
        "rank_tuple": list(fit_config.ranks),
        "ridge": RIDGE,
        "sweeps": ALS_SWEEPS,
        "fit_status": fit_result.status.value,
        "termination_reason": fit_result.termination_reason,
        "fit_residual": _scalar(fit_result.fit_residual),
        "holdout_rms": _rms(holdout_residual),
        "holdout_central_rms": _rms(tf.boolean_mask(holdout_residual, central_mask)),
        "holdout_shell_rms": _rms(tf.boolean_mask(holdout_residual, shell_mask)),
        "holdout_central_count": int(tf.reduce_sum(tf.cast(central_mask, tf.int32)).numpy()),
        "holdout_shell_count": int(tf.reduce_sum(tf.cast(shell_mask, tf.int32)).numpy()),
        "audit_prediction_rms": _rms(audit_prediction),
        "gram_normalizer_scaled": _scalar(gram_scaled),
        "z_h": _scalar(z_h),
        "z_t_direct": _scalar(z_t),
        "z_t_direct_standard_error": _scalar(direct_se),
        "z_t_direct_relative_standard_error": _scalar(direct_se / z_t),
        "z_t_reference_normal_bank": _scalar(reference_z_t),
        "z_t_reference_standard_error": _scalar(reference_se),
        "z_t_reference_relative_standard_error": _scalar(reference_se / reference_z_t),
        "log_z_h_minus_log_z_t": _scalar(tf.math.log(z_h) - tf.math.log(z_t)),
        "condition": condition,
        "realized_rank_tuple": list(fitted.rank_tuple()),
        "max_abs_core_value": max(_max_abs(core.values) for core in fitted.cores),
        "mass": {
            key: value
            for key, value in contraction.items()
            if key != "mass" and key != "integral"
        },
        "finite": finite,
        "hard_valid": hard_valid,
        "map_condition_number": _scalar(shared["map_condition_number"]),
        "map_minimum_eigenvalue": _scalar(shared["map_minimum_eigenvalue"]),
    }


def _result_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# C2 Mixture-UKF/APF Phase 5B RBF Pilot Result",
        "",
        f"Status: `{payload['status']}`  ",
        f"Continuation: `{payload['continuation']}`  ",
        f"Failure class: `{payload['failure_class']}`",
        "",
        "## Decision",
        "",
        "| Decision | Criterion | Status | Interpretation |",
        "| --- | --- | --- | --- |",
        f"| Exact target and map | fixture parity, finite/SPD map and target | {'PASS' if payload['checks']['map_target_valid'] else 'VETO'} | {'RBF records are interpretable' if payload['checks']['map_target_valid'] else 'repair target/map'} |",
        f"| RBF mass and fitter | analytic/quadrature parity, finite SPD mass, fit condition | {'PASS' if payload['checks']['all_arms_valid'] else 'VETO'} | {'RBF mechanics passed' if payload['checks']['all_arms_valid'] else 'repair basis or fitter'} |",
        "| Width promotion | no promoted threshold in this pilot | NOT TESTED | arm differences are descriptive only |",
        "",
        "## Inference status",
        "",
        "| Evidence class | Status |",
        "| --- | --- |",
        f"| Hard veto screen | {'passed' if payload['checks']['hard_vetoes_pass'] else 'failed'} |",
        "| Statistically supported ranking | not available from one paired bank |",
        "| Descriptive differences | arm-wise residual, mass, condition, and normalizer records below |",
        "| Default readiness | not assessed |",
        f"| Next evidence | {payload['next_phase_refresh']} |",
        "",
        "## Fixed target and map",
        "",
        f"Carried rows: `{payload['carried_rows']}`; training/holdout/audit rows: `{payload['train_rows']}/{payload['holdout_rows']}/{payload['audit_rows']}`; map condition: `{payload['map']['condition_number']:.6g}`; minimum eigenvalue: `{payload['map']['minimum_eigenvalue']:.6g}`.",
        "",
        f"Predictive `Z_T`: `{payload['z_t_direct']:.8g}` +/- `{payload['z_t_direct_standard_error']:.3g}` (relative SE `{payload['z_t_direct_relative_standard_error']:.3g}`).",
        "",
        "## RBF arm ladder",
        "",
        "| arm | width | holdout RMS | central RMS | shell RMS | log ZH - log ZT | mass cond. | fit cond. | hard-valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in payload["arm_records"]:
        lines.append(
            f"| {row['arm']} | {row['width']:.3g} | {row['holdout_rms']:.4g} | {row['holdout_central_rms']:.4g} | {row['holdout_shell_rms']:.4g} | {row['log_z_h_minus_log_z_t']:.4g} | {row['mass']['mass_condition_number']:.4g} | {row['condition']['scaled_augmented_condition_max']:.4g} | {row['hard_valid']} |"
        )
    lines += [
        "",
        "The predictive-mixture `Z_T` is an independent finite-bank diagnostic, not an analytic oracle. The Gaussian-reference estimate is retained only for tail-variance diagnosis. No arm ranking or representation promotion follows from this pilot.",
        "",
        "## Red team",
        "",
        "The strongest alternative explanation is finite-bank error or a fixed-map mismatch that affects every arm. The map/fixture checks and independent predictive bank address that risk, but one transition and one paired bank remain the weakest evidence. A valid RBF arm still requires replicated and recursive validation.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    output = _make_output_root(args.output_root)
    started = time.perf_counter()
    runtime = _configure_runtime()
    _load_rbf_basis()
    p5a = _load_phase5a()
    fixture = p5a._load_fixture()
    shared = p5a._model_and_map(fixture, jit_compile=bool(args.jit_compile))
    coordinate_map = shared["coordinate_map"]
    banks = {
        "train": p5a._standard_normal_bank(TRAIN_ROWS, (SEED[0], SEED[1] + 1)),
        "holdout": p5a._standard_normal_bank(HOLDOUT_ROWS, (SEED[0], SEED[1] + 2)),
        "audit": p5a._standard_normal_bank(AUDIT_ROWS, (SEED[0], SEED[1] + 3)),
    }
    kernels = {
        name: p5a._make_exact_target_kernel(
            parent_states=shared["states"],
            parent_weights=shared["weights"],
            transition=shared["transition"],
            theta=shared["theta"],
            observation=shared["observations"][1],
            sigma=float(fixture["sigma"]),
            row_count=int(rows.shape[0]),
            jit_compile=bool(args.jit_compile),
        )
        for name, rows in banks.items()
    }
    train_physical, train_logdet = coordinate_map.forward(banks["train"])
    train_log_gamma = kernels["train"](train_physical)
    train_log_eta = -0.5 * (
        tf.cast(STATE_DIM, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(banks["train"]), axis=1)
    )
    shift = tf.reduce_logsumexp(train_log_gamma + train_logdet - train_log_eta) - tf.math.log(
        tf.cast(TRAIN_ROWS, DTYPE)
    )
    log_targets: dict[str, tf.Tensor] = {}
    sqrt_targets: dict[str, tf.Tensor] = {}
    for name, rows in banks.items():
        log_targets[name], sqrt_targets[name], _ = p5a._reference_target_values(
            rows,
            coordinate_map=coordinate_map,
            target_kernel=kernels[name],
            shift=shift,
        )
        if not _finite(log_targets[name]) or not _finite(sqrt_targets[name]):
            raise ValueError(f"non-finite target in {name} bank")
    predictive = p5a._prior_predictive_target_audit(
        shared, row_count=AUDIT_ROWS, seed=(SEED[0], SEED[1] + 4)
    )
    direct_target = {"z_t": predictive["z_t"], "standard_error": predictive["standard_error"]}
    arm_records = [
        _fit_arm(
            name,
            width,
            shared=shared,
            banks=banks,
            sqrt_targets=sqrt_targets,
            log_targets=log_targets,
            direct_target=direct_target,
            shift=shift,
        )
        for name, width in ARMS
    ]
    checks = {
        "source_files_present": all(
            path.is_file()
            for path in (FIXTURE_PATH, MODEL_PATH, MOMENT_MAP_PATH, RBF_MODULE_PATH, PLAN_PATH)
        ),
        "map_target_valid": (
            shared["transition_fixture_error"] <= 1.0e-12
            and shared["process_fixture_error"] <= 1.0e-12
            and shared["stationary_fixture_error"] <= 1.0e-12
            and _finite(shared["predicted_mean"])
            and _finite(shared["predicted_covariance"])
            and _scalar(shared["map_minimum_eigenvalue"]) > 0.0
        ),
        "banks_disjoint_by_seed": True,
        "predictive_target_audit_finite": _finite(predictive["likelihood_values"]),
        "all_arms_valid": all(bool(row["hard_valid"]) for row in arm_records),
    }
    checks["hard_vetoes_pass"] = all(bool(value) for value in checks.values())
    elapsed = time.perf_counter() - started
    source_paths = {
        "plan": PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "rbf_basis": RBF_MODULE_PATH,
        "phase5a_driver": PHASE5A_DRIVER_PATH,
        "fixture": FIXTURE_PATH,
        "model": MODEL_PATH,
        "moment_map": MOMENT_MAP_PATH,
    }
    sources = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)}
        for name, path in source_paths.items()
    }
    payload: Mapping[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "phase": PHASE_ID,
        "status": "PASS_PHASE5B_RBF_MECHANICS" if checks["hard_vetoes_pass"] else "VETO_PHASE5B_RBF_MECHANICS",
        "continuation": "CONTINUE_HYBRID_DIAGNOSTIC" if checks["hard_vetoes_pass"] else "CONTINUATION_VETO_PHASE5B_VALIDITY",
        "failure_class": "none" if checks["hard_vetoes_pass"] else "implementation_or_numerical_validity",
        "next_phase_refresh": "fixed-map Hermite-plus-RBF hybrid with analytic cross-Gram checks if RBF mechanics pass; otherwise repair the RBF basis contract",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "jit_compile": bool(args.jit_compile),
            "runtime": runtime,
        },
        "seed": SEED,
        "map_seed": MAP_SEED,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "carried_rows": CARRIED_ROWS,
        "train_rows": TRAIN_ROWS,
        "holdout_rows": HOLDOUT_ROWS,
        "audit_rows": AUDIT_ROWS,
        "quadrature_order": QUADRATURE_ORDER,
        "centers": CENTERS,
        "arms": ARMS,
        "tt_rank": TT_RANK,
        "als_sweeps": ALS_SWEEPS,
        "ridge": RIDGE,
        "target_log_shift": _scalar(shift),
        "sources": sources,
        "workspace": _workspace_manifest(),
        "checks": checks,
        "map": {
            "condition_number": _scalar(shared["map_condition_number"]),
            "minimum_eigenvalue": _scalar(shared["map_minimum_eigenvalue"]),
            "transition_fixture_error": shared["transition_fixture_error"],
            "process_fixture_error": shared["process_fixture_error"],
            "stationary_fixture_error": shared["stationary_fixture_error"],
        },
        "bank_seeds": {
            "train": (SEED[0], SEED[1] + 1),
            "holdout": (SEED[0], SEED[1] + 2),
            "audit": (SEED[0], SEED[1] + 3),
            "predictive_audit": (SEED[0], SEED[1] + 4),
        },
        "bank_hashes": {
            name: hashlib.sha256(tf.io.serialize_tensor(rows).numpy()).hexdigest()
            for name, rows in banks.items()
        },
        "predictive_audit_hash": hashlib.sha256(
            tf.io.serialize_tensor(predictive["physical_rows"]).numpy()
        ).hexdigest(),
        "z_t_direct": predictive["z_t"],
        "z_t_direct_standard_error": predictive["standard_error"],
        "z_t_direct_relative_standard_error": predictive["relative_standard_error"],
        "z_t_reference_normal_bank": tf.exp(shift) * tf.reduce_mean(
            tf.exp(log_targets["audit"] - shift)
        ),
        "arm_records": arm_records,
        "nonclaims": [
            "no proposal-efficiency or ESS claim",
            "no posterior-correctness or pseudo-marginal claim",
            "no arm ranking or width promotion claim",
            "no recursive multi-step filtering claim",
            "no analytical total-gradient or HMC claim",
            "no production/default-readiness claim",
        ],
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "phase": PHASE_ID,
        "command": " ".join(sys.argv),
        "plan_sha256": sources["plan"]["sha256"],
        "sources": sources,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "seed": SEED,
        "map_seed": MAP_SEED,
        "bank_seeds": payload["bank_seeds"],
        "bank_hashes": payload["bank_hashes"],
        "centers": CENTERS,
        "arms": ARMS,
        "quadrature_order": QUADRATURE_ORDER,
        "tt_rank": TT_RANK,
        "als_sweeps": ALS_SWEEPS,
        "ridge": RIDGE,
        "environment": payload["environment"],
        "workspace": payload["workspace"],
        "evidence_contract": {
            "primary": "fixed-map exact-target RBF representation diagnostics on disjoint banks",
            "hard_vetoes": "target/map mismatch, nonfinite or non-SPD analytic mass, quadrature mismatch, invalid fit, missing records",
            "descriptive": "heldout/shell residuals, predictive Z_T versus exact RBF Gram Z_H, mass and fit condition",
            "nonclaims": payload["nonclaims"],
        },
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "result.json", payload)
    _write_json(output / "records.json", {"arms": arm_records})
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_result_markdown(payload), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    output = run(args)
    print(json.dumps({"phase": PHASE_ID, "output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
