"""Initial HMC geometry construction and public mass-matrix helpers.

The initializer owns hint precedence, validation, mass construction and the
starting epsilon/L formula. It performs one-time TensorFlow preparation only;
bootstrap, mass adaptation and candidate verification are separate stages.
Historical imports from hmc_kernel_tuning remain compatibility aliases.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from bayesfilter.inference.hmc import PrecomputedMassArtifact, stable_adapter_signature
from bayesfilter.inference.hmc_artifact_identity import mass_artifact_signature as _mass_artifact_signature
from bayesfilter.inference.hmc_verification import _float64_tensor
from bayesfilter.runtime import stable_config_hash

from bayesfilter.inference.mass_matrix import (
    MassMatrixResult,
    covariance_from_negative_hessian,
    covariance_from_precision,
    regularize_covariance,
    regularize_precision,
    whitening_from_covariance,
)

__all__ = [
    "GEOMETRY_INITIALIZATION_NONCLAIMS",
    "HMCGeometryInitializationConfig",
    "HMCGeometryInitializationResult",
    "MassMatrixResult",
    "PrecomputedMassArtifact",
    "covariance_from_negative_hessian",
    "covariance_from_precision",
    "initialize_hmc_kernel_geometry",
    "regularize_covariance",
    "regularize_precision",
    "whitening_from_covariance",
]


GEOMETRY_INITIALIZATION_NONCLAIMS = (
    "geometry initialization only",
    "no HMC runtime claim",
    "no tuning success claim",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
)

_GEOMETRY_MIN_LEAPFROG = 3

_GEOMETRY_MAX_LEAPFROG = 25

def _validate_max_leapfrog_steps(value: Any, *, name: str = "max_leapfrog_steps") -> int:
    max_l = int(value)
    if max_l < _GEOMETRY_MIN_LEAPFROG:
        raise ValueError(f"{name} must be at least {_GEOMETRY_MIN_LEAPFROG}")
    return max_l

@dataclass(frozen=True)
class HMCGeometryInitializationConfig:
    """Configuration for geometry-derived initial HMC kernel parameters.

    The config deliberately does not ask the caller for a step size, leapfrog
    count, trajectory grid, mass-window schedule, or tuning budget schedule.
    Those belong to later internal BayesFilter tuning phases.
    """

    geometry_scaling_c: float = 0.5
    stability_guard: float = 0.8
    covariance_jitter: float = 1.0e-9
    eigenvalue_floor: float | None = 1.0e-9
    max_condition_number: float | None = None
    max_leapfrog_steps: int = _GEOMETRY_MAX_LEAPFROG
    allow_geometry_fallback: bool = False
    position_role: str = "initial_position"
    negative_hessian_source: str = "negative_hessian"
    mass_policy: str = "windowed_adaptive"
    seed: tuple[int, int] = (20260621, 2)
    source: str = "bayesfilter.inference.hmc_kernel_tuning"

    def __post_init__(self) -> None:
        scaling = float(self.geometry_scaling_c)
        guard = float(self.stability_guard)
        if not math.isfinite(scaling) or scaling <= 0.0:
            raise ValueError("geometry_scaling_c must be positive and finite")
        if not math.isfinite(guard) or guard <= 0.0:
            raise ValueError("stability_guard must be positive and finite")
        jitter = float(self.covariance_jitter)
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("covariance_jitter must be finite and non-negative")
        floor = (
            None
            if self.eigenvalue_floor is None
            else float(self.eigenvalue_floor)
        )
        if floor is not None and (not math.isfinite(floor) or floor < 0.0):
            raise ValueError("eigenvalue_floor must be finite and non-negative")
        condition = (
            None
            if self.max_condition_number is None
            else float(self.max_condition_number)
        )
        if condition is not None and (
            not math.isfinite(condition) or condition <= 1.0
        ):
            raise ValueError("max_condition_number must be finite and greater than 1")
        max_l = _validate_max_leapfrog_steps(self.max_leapfrog_steps)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        position_role = str(self.position_role)
        if not position_role:
            raise ValueError("position_role must be non-empty")
        negative_hessian_source = str(self.negative_hessian_source)
        if not negative_hessian_source:
            raise ValueError("negative_hessian_source must be non-empty")
        mass_policy = str(self.mass_policy)
        if mass_policy not in {"windowed_adaptive", "fixed_identity"}:
            raise ValueError(
                "mass_policy must be 'windowed_adaptive' or 'fixed_identity'"
            )
        object.__setattr__(self, "geometry_scaling_c", scaling)
        object.__setattr__(self, "stability_guard", guard)
        object.__setattr__(self, "covariance_jitter", jitter)
        object.__setattr__(self, "eigenvalue_floor", floor)
        object.__setattr__(self, "max_condition_number", condition)
        object.__setattr__(self, "max_leapfrog_steps", max_l)
        object.__setattr__(self, "allow_geometry_fallback", bool(self.allow_geometry_fallback))
        object.__setattr__(self, "position_role", position_role)
        object.__setattr__(self, "negative_hessian_source", negative_hessian_source)
        object.__setattr__(self, "mass_policy", mass_policy)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "source", source)

    def payload(self) -> Mapping[str, Any]:
        return {
            "geometry_scaling_c": self.geometry_scaling_c,
            "stability_guard": self.stability_guard,
            "covariance_jitter": self.covariance_jitter,
            "eigenvalue_floor": self.eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "max_leapfrog_steps": self.max_leapfrog_steps,
            "allow_geometry_fallback": self.allow_geometry_fallback,
            "position_role": self.position_role,
            "negative_hessian_source": self.negative_hessian_source,
            "mass_policy": self.mass_policy,
            "seed": self.seed,
            "source": self.source,
        }

@dataclass(frozen=True)
class HMCGeometryInitializationResult:
    """Geometry-derived starting kernel, not HMC tuning evidence."""

    config: HMCGeometryInitializationConfig
    adapter_signature: str
    target_dimension: int
    mass_artifact: PrecomputedMassArtifact
    mass_artifact_signature: str
    initial_step_size: float
    initial_num_leapfrog_steps: int
    unclamped_num_leapfrog_steps: int
    target_trajectory_length: float
    hint_report: Mapping[str, Any]
    curvature_report: Mapping[str, Any]
    formula_report: Mapping[str, Any]
    seed_report: Mapping[str, Any]
    nonclaims: tuple[str, ...] = GEOMETRY_INITIALIZATION_NONCLAIMS

    def __post_init__(self) -> None:
        signature = str(self.adapter_signature)
        if not signature:
            raise ValueError("adapter_signature must be non-empty")
        dimension = int(self.target_dimension)
        if dimension <= 0:
            raise ValueError("target_dimension must be positive")
        if self.mass_artifact.adapter_signature != signature:
            raise ValueError("mass artifact adapter_signature mismatch")
        if self.mass_artifact.dimension != dimension:
            raise ValueError("mass artifact dimension mismatch")
        step = float(self.initial_step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("initial_step_size must be positive and finite")
        leapfrogs = int(self.initial_num_leapfrog_steps)
        raw_leapfrogs = int(self.unclamped_num_leapfrog_steps)
        if leapfrogs <= 0 or raw_leapfrogs <= 0:
            raise ValueError("leapfrog counts must be positive")
        trajectory = float(self.target_trajectory_length)
        if not math.isfinite(trajectory) or trajectory <= 0.0:
            raise ValueError("target_trajectory_length must be positive and finite")
        mass_signature = str(self.mass_artifact_signature)
        if not mass_signature:
            raise ValueError("mass_artifact_signature must be non-empty")
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "adapter_signature", signature)
        object.__setattr__(self, "target_dimension", dimension)
        object.__setattr__(self, "mass_artifact_signature", mass_signature)
        object.__setattr__(self, "initial_step_size", step)
        object.__setattr__(self, "initial_num_leapfrog_steps", leapfrogs)
        object.__setattr__(self, "unclamped_num_leapfrog_steps", raw_leapfrogs)
        object.__setattr__(self, "target_trajectory_length", trajectory)
        object.__setattr__(self, "hint_report", dict(self.hint_report))
        object.__setattr__(self, "curvature_report", dict(self.curvature_report))
        object.__setattr__(self, "formula_report", dict(self.formula_report))
        object.__setattr__(self, "seed_report", dict(self.seed_report))
        object.__setattr__(self, "nonclaims", nonclaims)

    def payload(self, *, include_mass_arrays: bool = False) -> Mapping[str, Any]:
        return {
            "artifact_type": "bayesfilter_hmc_geometry_initialization_result",
            "schema_version": 1,
            "config": self.config.payload(),
            "adapter_signature": self.adapter_signature,
            "target_dimension": self.target_dimension,
            "mass_artifact_payload": self.mass_artifact.to_payload(
                include_arrays=include_mass_arrays
            ),
            "mass_artifact_signature": self.mass_artifact_signature,
            "initial_step_size": self.initial_step_size,
            "initial_num_leapfrog_steps": self.initial_num_leapfrog_steps,
            "unclamped_num_leapfrog_steps": self.unclamped_num_leapfrog_steps,
            "target_trajectory_length": self.target_trajectory_length,
            "hint_report": self.hint_report,
            "curvature_report": self.curvature_report,
            "formula_report": self.formula_report,
            "seed_report": self.seed_report,
            "reports_hmc_runtime_readiness": False,
            "reports_tuning_success": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "nonclaims": self.nonclaims,
        }

    @property
    def artifact_hash(self) -> str:
        return stable_config_hash(self.payload(include_mass_arrays=True))

    @property
    def numerical_hash(self) -> str:
        """Numerical provenance for streams; the full audit hash retains clocks."""
        payload = dict(self.payload(include_mass_arrays=True))
        formula = dict(payload["formula_report"])
        startup = formula.get("bootstrap_initialization")
        if startup is not None:
            startup = dict(startup)
            startup["rounds"] = [
                {key: value for key, value in row.items() if key != "wall_seconds"}
                for row in startup["rounds"]]
            if startup.get("first_invalid_proposal") is not None:
                startup["first_invalid_proposal"] = {
                    key: value for key, value in startup["first_invalid_proposal"].items()
                    if key != "wall_seconds"}
            formula["bootstrap_initialization"] = startup
        payload["formula_report"] = formula
        return stable_config_hash(payload)

def initialize_hmc_kernel_geometry(
    *,
    adapter: Any,
    initial_position: Any,
    config: HMCGeometryInitializationConfig | None = None,
    negative_hessian: Any | None = None,
    initial_covariance: Any | None = None,
    parameter_scales: Any | None = None,
) -> HMCGeometryInitializationResult:
    """Build an initial mass artifact and formula-derived epsilon/L.

    ``negative_hessian`` is interpreted as a local precision approximation
    ``-d2 log p(theta)`` in the same unconstrained coordinates as
    ``initial_position``.  This function is a geometry initializer only; it does
    not run HMC or adapt the kernel.
    """

    cfg = HMCGeometryInitializationConfig() if config is None else config
    if not isinstance(cfg, HMCGeometryInitializationConfig):
        raise TypeError("config must be HMCGeometryInitializationConfig")
    adapter_signature = stable_adapter_signature(adapter)
    position = _validate_position(initial_position)
    dimension = int(position.shape[0])
    hint = _select_geometry_hint(
        position=position,
        negative_hessian=negative_hessian,
        initial_covariance=initial_covariance,
        parameter_scales=parameter_scales,
        config=cfg,
    )
    mass_artifact = _build_mass_artifact(
        position=position,
        adapter_signature=adapter_signature,
        hint=hint,
        config=cfg,
    )
    omega = _curvature_frequencies(
        covariance=mass_artifact.covariance,
        precision=hint.precision_for_formula,
    )
    target_trajectory = _target_trajectory_length(omega)
    epsilon = _initial_step_size(omega, dimension=dimension, config=cfg)
    unclamped_l = int(math.ceil(target_trajectory / epsilon))
    leapfrogs = int(
        min(max(unclamped_l, _GEOMETRY_MIN_LEAPFROG), cfg.max_leapfrog_steps)
    )
    mass_signature = _mass_artifact_signature(mass_artifact)
    return HMCGeometryInitializationResult(
        config=cfg,
        adapter_signature=adapter_signature,
        target_dimension=dimension,
        mass_artifact=mass_artifact,
        mass_artifact_signature=mass_signature,
        initial_step_size=epsilon,
        initial_num_leapfrog_steps=leapfrogs,
        unclamped_num_leapfrog_steps=unclamped_l,
        target_trajectory_length=target_trajectory,
        hint_report=hint.report,
        curvature_report=_curvature_report(omega),
        formula_report={
            "formula": "epsilon=min(c*d^(-1/4)/rms(omega), rho*2/max(omega)); L=ceil(tau/epsilon)",
            "geometry_scaling_c": cfg.geometry_scaling_c,
            "stability_guard": cfg.stability_guard,
            "target_trajectory_length": target_trajectory,
            "leapfrog_clamped": leapfrogs != unclamped_l,
            "internal_min_leapfrog": _GEOMETRY_MIN_LEAPFROG,
            "internal_max_leapfrog": cfg.max_leapfrog_steps,
            "default_max_leapfrog_steps": _GEOMETRY_MAX_LEAPFROG,
        },
        seed_report={
            "root_seed": cfg.seed,
            "seed_owner": "BayesFilter",
            "geometry_seed": _derive_seed(cfg.seed, stage_index=0),
            "fresh_verification_seed_reserved": _derive_seed(cfg.seed, stage_index=5),
            "nonclaim": "geometry seed provenance only; no HMC run executed",
        },
    )

@dataclass(frozen=True)
class _GeometryHint:
    kind: str
    covariance: Any
    precision_for_formula: Any
    report: Mapping[str, Any]

def _validate_position(position: Any) -> Any:
    import tensorflow as tf

    try:
        array = _float64_tensor(position)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_position must be a numeric vector") from exc
    if array.shape.rank != 1:
        raise ValueError("initial_position must be a one-dimensional vector")
    if array.shape[0] <= 0:
        raise ValueError("initial_position must be non-empty")
    try:
        tf.debugging.assert_all_finite(array, "initial_position must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise ValueError("initial_position must be finite") from exc
    return array

def _select_geometry_hint(
    *,
    position: Any,
    negative_hessian: Any | None,
    initial_covariance: Any | None,
    parameter_scales: Any | None,
    config: HMCGeometryInitializationConfig,
) -> _GeometryHint:
    supplied = {
        "negative_hessian": negative_hessian is not None,
        "initial_covariance": initial_covariance is not None,
        "parameter_scales": parameter_scales is not None,
    }
    failures: list[Mapping[str, Any]] = []
    if config.mass_policy == "fixed_identity":
        return _identity_hint(
            position=position,
            supplied=supplied,
            failures=(
                {
                    "kind": "fixed_identity",
                    "reason": "caller geometry hints ignored by explicit mass policy",
                },
            ),
        )
    for kind, value in (
        ("negative_hessian", negative_hessian),
        ("initial_covariance", initial_covariance),
        ("parameter_scales", parameter_scales),
    ):
        if value is None:
            continue
        try:
            return _hint_from_value(
                kind=kind,
                value=value,
                position=position,
                config=config,
                supplied=supplied,
                failures=tuple(failures),
            )
        except Exception as exc:
            failures.append({"kind": kind, "error": str(exc)})
            if not config.allow_geometry_fallback:
                raise
    return _identity_hint(position=position, supplied=supplied, failures=tuple(failures))

def _hint_from_value(
    *,
    kind: str,
    value: Any,
    position: Any,
    config: HMCGeometryInitializationConfig,
    supplied: Mapping[str, bool],
    failures: tuple[Mapping[str, Any], ...],
) -> _GeometryHint:
    import tensorflow as tf

    dimension = int(position.shape[0])
    if kind == "negative_hessian":
        precision = _validate_matrix(value, dimension=dimension, name=kind)
        artifact = PrecomputedMassArtifact.from_negative_hessian(
            position=position,
            negative_hessian=precision,
            adapter_signature="geometry_hint_validation_adapter",
            position_role=config.position_role,
            covariance_source=config.negative_hessian_source,
            source="geometry_initialization_probe",
            jitter=config.covariance_jitter,
            eigenvalue_floor=config.eigenvalue_floor,
            max_condition_number=config.max_condition_number,
        )
        report = {
            "selected_hint": kind,
            "covariance_source": config.negative_hessian_source,
            "hint_precedence": (
                "negative_hessian",
                "initial_covariance",
                "parameter_scales",
                "identity",
            ),
            "supplied_hints": dict(supplied),
            "fallback_used": bool(failures),
            "fallback_failures": failures,
            "sign_convention": "-d2 log posterior in unconstrained coordinates",
            "parameterization": "same as initial_position",
            "regularization_report": artifact.regularization_report,
        }
        return _GeometryHint(
            kind=kind,
            covariance=tf.convert_to_tensor(artifact.covariance, dtype=tf.float64),
            precision_for_formula=tf.linalg.pinv(
                tf.convert_to_tensor(artifact.covariance, dtype=tf.float64),
                rcond=tf.constant(1.0e-15, dtype=tf.float64),
            ),
            report=report,
        )
    if kind == "initial_covariance":
        covariance = _validate_matrix(value, dimension=dimension, name=kind)
        covariance = covariance + config.covariance_jitter * tf.eye(
            dimension, dtype=tf.float64
        )
        _validate_spd(covariance, name=kind)
        return _GeometryHint(
            kind=kind,
            covariance=covariance,
            precision_for_formula=tf.linalg.pinv(
                covariance,
                rcond=tf.constant(1.0e-15, dtype=tf.float64),
            ),
            report={
                "selected_hint": kind,
                "hint_precedence": (
                    "negative_hessian",
                    "initial_covariance",
                    "parameter_scales",
                    "identity",
                ),
                "supplied_hints": dict(supplied),
                "fallback_used": bool(failures),
                "fallback_failures": failures,
                "parameterization": "same as initial_position",
                "regularization_report": {
                    "method": "covariance_jitter",
                    "covariance_jitter": config.covariance_jitter,
                },
            },
        )
    if kind == "parameter_scales":
        try:
            scales = _float64_tensor(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameter_scales must be numeric") from exc
        if scales.shape != (dimension,):
            raise ValueError("parameter_scales shape must match initial_position")
        try:
            tf.debugging.assert_all_finite(scales, "parameter_scales must be finite")
        except tf.errors.InvalidArgumentError as exc:
            raise ValueError("parameter_scales must be finite") from exc
        if bool(tf.reduce_any(scales <= 0.0).numpy()):
            raise ValueError("parameter_scales must be positive")
        covariance = tf.linalg.diag(tf.square(scales)) + config.covariance_jitter * tf.eye(
            dimension, dtype=tf.float64
        )
        return _GeometryHint(
            kind=kind,
            covariance=covariance,
            precision_for_formula=tf.linalg.diag(
                1.0 / tf.linalg.diag_part(covariance)
            ),
            report={
                "selected_hint": kind,
                "hint_precedence": (
                    "negative_hessian",
                    "initial_covariance",
                    "parameter_scales",
                    "identity",
                ),
                "supplied_hints": dict(supplied),
                "fallback_used": bool(failures),
                "fallback_failures": failures,
                "parameterization": "same as initial_position",
                "regularization_report": {
                    "method": "diagonal_scales",
                    "covariance_jitter": config.covariance_jitter,
                },
            },
        )
    raise ValueError(f"unknown geometry hint kind: {kind}")

def _identity_hint(
    *,
    position: Any,
    supplied: Mapping[str, bool],
    failures: tuple[Mapping[str, Any], ...],
) -> _GeometryHint:
    import tensorflow as tf

    dimension = int(position.shape[0])
    return _GeometryHint(
        kind="identity",
        covariance=tf.eye(dimension, dtype=tf.float64),
        precision_for_formula=tf.eye(dimension, dtype=tf.float64),
        report={
            "selected_hint": "identity",
            "hint_precedence": (
                "negative_hessian",
                "initial_covariance",
                "parameter_scales",
                "identity",
            ),
            "supplied_hints": dict(supplied),
            "fallback_used": bool(failures) or any(supplied.values()),
            "fallback_failures": failures,
            "parameterization": "same as initial_position",
            "regularization_report": {"method": "identity_fallback"},
        },
    )

def _validate_matrix(value: Any, *, dimension: int, name: str) -> Any:
    import tensorflow as tf

    try:
        matrix = _float64_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} shape must match initial_position dimension")
    try:
        tf.debugging.assert_all_finite(matrix, f"{name} must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise ValueError(f"{name} must be finite") from exc
    return 0.5 * (matrix + tf.transpose(matrix))

def _validate_spd(matrix: Any, *, name: str) -> None:
    import tensorflow as tf

    eigenvalues = tf.linalg.eigvalsh(_float64_tensor(matrix))
    try:
        tf.debugging.assert_all_finite(eigenvalues, f"{name} eigenvalues must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise ValueError(f"{name} eigenvalues must be finite") from exc
    if bool(tf.reduce_any(eigenvalues <= 0.0).numpy()):
        raise ValueError(f"{name} must be positive definite")

def _build_mass_artifact(
    *,
    position: Any,
    adapter_signature: str,
    hint: _GeometryHint,
    config: HMCGeometryInitializationConfig,
) -> PrecomputedMassArtifact:
    source = f"geometry_initialization_{hint.kind}"
    covariance_source = str(hint.report.get("covariance_source", hint.kind))
    return PrecomputedMassArtifact.from_covariance(
        position=position,
        covariance=hint.covariance,
        adapter_signature=adapter_signature,
        position_role=config.position_role,
        covariance_source=covariance_source,
        matrix_used_for_square_root="geometry_initialization_covariance",
        source=source,
        jitter=0.0,
        regularization_report={
            **dict(hint.report.get("regularization_report", {})),
            "geometry_initializer_source": config.source,
        },
        nonclaims=GEOMETRY_INITIALIZATION_NONCLAIMS,
    )

def _curvature_frequencies(
    *,
    covariance: Any,
    precision: Any,
) -> Any:
    import tensorflow as tf

    covariance_tensor = _float64_tensor(covariance)
    precision_tensor = _float64_tensor(precision)
    eigenvalues_c, eigenvectors_c = tf.linalg.eigh(
        0.5 * (covariance_tensor + tf.transpose(covariance_tensor))
    )
    if bool(tf.reduce_any(~tf.math.is_finite(eigenvalues_c)).numpy()) or bool(
        tf.reduce_any(eigenvalues_c <= 0.0).numpy()
    ):
        raise ValueError("covariance square-root eigenvalues must be finite and positive")
    factor = tf.matmul(
        tf.matmul(eigenvectors_c, tf.linalg.diag(tf.sqrt(eigenvalues_c))),
        tf.transpose(eigenvectors_c),
    )
    scaled = tf.matmul(tf.matmul(factor, precision_tensor), factor)
    scaled = 0.5 * (scaled + tf.transpose(scaled))
    eigenvalues = tf.linalg.eigvalsh(scaled)
    if bool(tf.reduce_any(~tf.math.is_finite(eigenvalues)).numpy()):
        raise ValueError("mass-scaled curvature eigenvalues must be finite")
    return tf.sqrt(tf.maximum(eigenvalues, tf.constant(1.0e-16, tf.float64)))

def _curvature_median(omega: Any) -> float:
    import tensorflow as tf

    omega_tensor = tf.reshape(_float64_tensor(omega), [-1])
    if int(omega_tensor.shape[0]) <= 0:
        raise ValueError("curvature frequencies must be non-empty")
    sorted_omega = tf.sort(omega_tensor)
    count = int(sorted_omega.shape[0])
    median_tensor = (
        sorted_omega[count // 2]
        if count % 2
        else 0.5 * (sorted_omega[count // 2 - 1] + sorted_omega[count // 2])
    )
    return float(median_tensor.numpy())

def _target_trajectory_length(omega: Any) -> float:
    median = _curvature_median(omega)
    if not math.isfinite(median) or median <= 0.0:
        raise ValueError("median curvature frequency must be positive and finite")
    return float(math.pi / (2.0 * median))

def _initial_step_size(
    omega: Any,
    *,
    dimension: int,
    config: HMCGeometryInitializationConfig,
) -> float:
    import tensorflow as tf

    omega_tensor = tf.reshape(_float64_tensor(omega), [-1])
    rms = float(tf.sqrt(tf.reduce_mean(tf.square(omega_tensor))).numpy())
    max_omega = float(tf.reduce_max(omega_tensor).numpy())
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("rms curvature frequency must be positive and finite")
    if not math.isfinite(max_omega) or max_omega <= 0.0:
        raise ValueError("max curvature frequency must be positive and finite")
    dimension_scale = float(dimension) ** (-0.25)
    scaled = config.geometry_scaling_c * dimension_scale / rms
    stable = config.stability_guard * 2.0 / max_omega
    step = float(min(scaled, stable))
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("formula-derived initial step size must be positive and finite")
    return step

def _curvature_report(omega: Any) -> Mapping[str, Any]:
    import tensorflow as tf

    omega_tensor = tf.reshape(_float64_tensor(omega), [-1])
    return {
        "omega_min": float(tf.reduce_min(omega_tensor).numpy()),
        "omega_median": _curvature_median(omega_tensor),
        "omega_rms": float(tf.sqrt(tf.reduce_mean(tf.square(omega_tensor))).numpy()),
        "omega_max": float(tf.reduce_max(omega_tensor).numpy()),
        "omega_count": int(omega_tensor.shape[0]),
        "finite": bool(tf.reduce_all(tf.math.is_finite(omega_tensor)).numpy()),
        "positive": bool(tf.reduce_all(omega_tensor > 0.0).numpy()),
    }

def _derive_seed(root_seed: tuple[int, int], *, stage_index: int) -> tuple[int, int]:
    return (int(root_seed[0]) + 1009 * int(stage_index), int(root_seed[1]) + 9176)
