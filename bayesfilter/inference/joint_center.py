"""Bounded full-vector TensorFlow Probability center localization.

For a positive diagonal scale ``S`` and baseline ``theta0``, this module uses
the standardized chart ``theta(z) = theta0 + S z``. If ``score_theta`` is the
gradient of the log target, the minimized objective and gradient are

``F(z) = -log p(theta(z))`` and ``grad_z F = -S score_theta(theta(z))``.

The L-BFGS endpoint is a finite-neighborhood locator only. Quasi-Newton
correction pairs are never returned or interpreted as curvature, covariance,
mass-matrix, or HMC evidence. The exact endpoint is replayed before acceptance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


JOINT_CENTER_NONCLAIMS = (
    "single-start finite-neighborhood locator only",
    "not a certified local or global MAP",
    "not multimodal coverage evidence",
    "not optimizer-sequence convergence evidence",
    "not Hessian, covariance, or mass-matrix evidence",
    "not HMC readiness evidence",
    "not posterior correctness evidence",
    "not default-readiness evidence",
)


@dataclass(frozen=True)
class JointCenterLocatorConfig:
    """Frozen L-BFGS and execution limits for one full-vector locator."""

    num_correction_pairs: int = 10
    max_iterations: int = 30
    max_line_search_iterations: int = 20
    gradient_tolerance: float = 0.02
    f_relative_tolerance: float = 0.0
    f_absolute_tolerance: float = 0.0
    x_tolerance: float = 0.0
    parallel_iterations: int = 1
    max_objective_evaluations: int = 601
    jit_compile: bool = True
    max_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "num_correction_pairs",
            "max_iterations",
            "max_line_search_iterations",
            "parallel_iterations",
            "max_objective_evaluations",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "gradient_tolerance",
            "f_relative_tolerance",
            "f_absolute_tolerance",
            "x_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "jit_compile", bool(self.jit_compile))
        if self.max_wall_seconds is not None:
            wall = float(self.max_wall_seconds)
            if not np.isfinite(wall) or wall <= 0.0:
                raise ValueError("max_wall_seconds must be positive finite")
            if self.jit_compile:
                raise ValueError("max_wall_seconds requires jit_compile=False")
            object.__setattr__(self, "max_wall_seconds", wall)

    def payload(self) -> Mapping[str, Any]:
        return {
            "num_correction_pairs": self.num_correction_pairs,
            "max_iterations": self.max_iterations,
            "max_line_search_iterations": self.max_line_search_iterations,
            "gradient_tolerance": self.gradient_tolerance,
            "f_relative_tolerance": self.f_relative_tolerance,
            "f_absolute_tolerance": self.f_absolute_tolerance,
            "x_tolerance": self.x_tolerance,
            "parallel_iterations": self.parallel_iterations,
            "max_objective_evaluations": self.max_objective_evaluations,
            "jit_compile": self.jit_compile,
            "max_wall_seconds": self.max_wall_seconds,
        }


@dataclass(frozen=True)
class JointCenterStagedConfig:
    """Frozen limits for one checkpointed same-state L-BFGS path."""

    num_correction_pairs: int = 10
    checkpoint_iterations: int = 30
    total_iterations: int = 60
    max_line_search_iterations: int = 20
    gradient_tolerance: float = 0.02
    f_relative_tolerance: float = 0.0
    f_absolute_tolerance: float = 0.0
    x_tolerance: float = 0.0
    parallel_iterations: int = 1
    max_objective_evaluations: int = 600
    jit_compile: bool = True
    max_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "num_correction_pairs",
            "checkpoint_iterations",
            "total_iterations",
            "max_line_search_iterations",
            "parallel_iterations",
            "max_objective_evaluations",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.total_iterations <= self.checkpoint_iterations:
            raise ValueError(
                "total_iterations must be greater than checkpoint_iterations"
            )
        for name in (
            "gradient_tolerance",
            "f_relative_tolerance",
            "f_absolute_tolerance",
            "x_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "jit_compile", bool(self.jit_compile))
        if self.max_wall_seconds is not None:
            wall = float(self.max_wall_seconds)
            if not np.isfinite(wall) or wall <= 0.0:
                raise ValueError("max_wall_seconds must be positive finite")
            if self.jit_compile:
                raise ValueError("max_wall_seconds requires jit_compile=False")
            object.__setattr__(self, "max_wall_seconds", wall)

    def payload(self) -> Mapping[str, Any]:
        return {
            "num_correction_pairs": self.num_correction_pairs,
            "checkpoint_iterations": self.checkpoint_iterations,
            "total_iterations": self.total_iterations,
            "max_line_search_iterations": self.max_line_search_iterations,
            "gradient_tolerance": self.gradient_tolerance,
            "f_relative_tolerance": self.f_relative_tolerance,
            "f_absolute_tolerance": self.f_absolute_tolerance,
            "x_tolerance": self.x_tolerance,
            "parallel_iterations": self.parallel_iterations,
            "max_objective_evaluations": self.max_objective_evaluations,
            "jit_compile": self.jit_compile,
            "max_wall_seconds": self.max_wall_seconds,
        }


@dataclass(frozen=True)
class JointCenterResult:
    """One bounded locator result with private arrays and public-safe payload."""

    status: str
    endpoint_accepted: bool
    initial_position: np.ndarray
    endpoint_position: np.ndarray
    initial_score: np.ndarray
    endpoint_score: np.ndarray
    initial_objective: float
    endpoint_objective: float
    initial_score_l2: float
    endpoint_score_l2: float
    initial_score_max_abs: float
    endpoint_score_max_abs: float
    optimizer_converged: bool
    optimizer_failed: bool
    optimizer_iterations: int
    reported_objective_evaluations: int
    callback_attempts: int
    optimizer_target_rows: int
    physical_target_rows: int
    cap_exhausted: bool
    wall_time_exhausted: bool
    jit_compile: bool
    exception_type: str | None
    nonclaims: tuple[str, ...] = JOINT_CENTER_NONCLAIMS

    def __post_init__(self) -> None:
        for name in (
            "initial_position",
            "endpoint_position",
            "initial_score",
            "endpoint_score",
        ):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "endpoint_accepted", bool(self.endpoint_accepted))
        object.__setattr__(self, "optimizer_converged", bool(self.optimizer_converged))
        object.__setattr__(self, "optimizer_failed", bool(self.optimizer_failed))
        object.__setattr__(self, "cap_exhausted", bool(self.cap_exhausted))
        object.__setattr__(
            self, "wall_time_exhausted", bool(self.wall_time_exhausted)
        )
        object.__setattr__(self, "jit_compile", bool(self.jit_compile))
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))

    def payload(self) -> Mapping[str, Any]:
        """Return an array-free status and accounting payload."""

        return {
            "schema": "bayesfilter.joint_center.public.v1",
            "status": self.status,
            "endpoint_accepted": self.endpoint_accepted,
            "optimizer_converged": self.optimizer_converged,
            "optimizer_failed": self.optimizer_failed,
            "optimizer_iterations": self.optimizer_iterations,
            "reported_objective_evaluations": (
                self.reported_objective_evaluations
            ),
            "callback_attempts": self.callback_attempts,
            "optimizer_target_rows": self.optimizer_target_rows,
            "physical_target_rows": self.physical_target_rows,
            "cap_exhausted": self.cap_exhausted,
            "wall_time_exhausted": self.wall_time_exhausted,
            "jit_compile": self.jit_compile,
            "objective_nondecreasing": (
                self.endpoint_objective >= self.initial_objective
            ),
            "exception_type": self.exception_type,
            "nonclaims": list(self.nonclaims),
        }


@dataclass(frozen=True)
class JointCenterCheckpoint:
    """Exact private stage-one replay with aggregate public accounting."""

    status: str
    endpoint_accepted: bool
    position: np.ndarray
    score: np.ndarray
    objective: float
    score_l2: float
    score_max_abs: float
    optimizer_converged: bool
    optimizer_failed: bool
    optimizer_iterations: int
    reported_objective_evaluations: int
    callback_attempts: int
    optimizer_target_rows: int
    physical_target_rows: int

    def __post_init__(self) -> None:
        for name in ("position", "score"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.joint_center.checkpoint.public.v1",
            "status": self.status,
            "endpoint_accepted": self.endpoint_accepted,
            "optimizer_converged": self.optimizer_converged,
            "optimizer_failed": self.optimizer_failed,
            "optimizer_iterations": self.optimizer_iterations,
            "reported_objective_evaluations": (
                self.reported_objective_evaluations
            ),
            "callback_attempts": self.callback_attempts,
            "optimizer_target_rows": self.optimizer_target_rows,
            "physical_target_rows": self.physical_target_rows,
        }


@dataclass(frozen=True)
class JointCenterStagedResult:
    """One checkpointed locator path without exported quasi-Newton state."""

    status: str
    endpoint_accepted: bool
    initial_position: np.ndarray
    initial_score: np.ndarray
    initial_objective: float
    initial_score_l2: float
    initial_score_max_abs: float
    checkpoint: JointCenterCheckpoint
    checkpoint_validated: bool
    checkpoint_validator_calls: int
    continuation_started: bool
    endpoint_position: np.ndarray
    endpoint_score: np.ndarray
    endpoint_objective: float
    endpoint_score_l2: float
    endpoint_score_max_abs: float
    optimizer_converged: bool
    optimizer_failed: bool
    optimizer_iterations: int
    reported_objective_evaluations: int
    callback_attempts: int
    optimizer_target_rows: int
    physical_target_rows: int
    cap_exhausted: bool
    wall_time_exhausted: bool
    jit_compile: bool
    exception_type: str | None
    nonclaims: tuple[str, ...] = JOINT_CENTER_NONCLAIMS

    def __post_init__(self) -> None:
        for name in (
            "initial_position",
            "initial_score",
            "endpoint_position",
            "endpoint_score",
        ):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.joint_center.staged.public.v1",
            "status": self.status,
            "endpoint_accepted": self.endpoint_accepted,
            "checkpoint": dict(self.checkpoint.payload()),
            "checkpoint_validated": self.checkpoint_validated,
            "checkpoint_validator_calls": self.checkpoint_validator_calls,
            "continuation_started": self.continuation_started,
            "optimizer_converged": self.optimizer_converged,
            "optimizer_failed": self.optimizer_failed,
            "optimizer_iterations": self.optimizer_iterations,
            "reported_objective_evaluations": (
                self.reported_objective_evaluations
            ),
            "callback_attempts": self.callback_attempts,
            "optimizer_target_rows": self.optimizer_target_rows,
            "physical_target_rows": self.physical_target_rows,
            "cap_exhausted": self.cap_exhausted,
            "wall_time_exhausted": self.wall_time_exhausted,
            "jit_compile": self.jit_compile,
            "objective_nondecreasing_from_checkpoint": (
                self.endpoint_objective >= self.checkpoint.objective
            ),
            "exception_type": self.exception_type,
            "nonclaims": list(self.nonclaims),
        }


def _standardized_objective_and_gradient(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: tf.Tensor,
    scale: tf.Tensor,
    z: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map standardized ``z`` to ``theta`` and apply the exact chain rule."""

    z_tensor = tf.reshape(tf.convert_to_tensor(z, tf.float64), [-1])
    theta = initial_position + scale * z_tensor
    value, score = value_and_score_fn(theta)
    value_tensor = tf.reshape(tf.convert_to_tensor(value, tf.float64), [])
    score_tensor = tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1])
    return -value_tensor, -scale * score_tensor


def locate_joint_center(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: Any,
    *,
    scale: Any | None = None,
    config: JointCenterLocatorConfig | None = None,
) -> JointCenterResult:
    """Run one bounded full-vector L-BFGS locator with exact endpoint replay."""

    cfg = JointCenterLocatorConfig() if config is None else config
    initial = _vector(initial_position, "initial_position")
    dimension = int(initial.shape[0])
    if dimension <= 0:
        raise ValueError("initial_position must be nonempty")
    scale_tensor = (
        tf.ones([dimension], tf.float64)
        if scale is None
        else _vector(scale, "scale")
    )
    if scale_tensor.shape != (dimension,) or not bool(
        tf.reduce_all(tf.math.is_finite(scale_tensor) & (scale_tensor > 0.0)).numpy()
    ):
        raise ValueError("scale must be positive finite with one entry per coordinate")

    initial_value, initial_score, initial_valid = _evaluate_value_score(
        value_and_score_fn, initial, dimension
    )
    initial_np = np.asarray(initial.numpy(), dtype=float)
    if not initial_valid:
        return _result(
            status="initial_target_invalid",
            accepted=False,
            initial_position=initial_np,
            endpoint_position=initial_np,
            initial_value=initial_value,
            endpoint_value=initial_value,
            initial_score=initial_score,
            endpoint_score=initial_score,
            optimizer_converged=False,
            optimizer_failed=False,
            optimizer_iterations=0,
            reported_evaluations=0,
            callback_attempts=0,
            optimizer_target_rows=0,
            physical_target_rows=1,
            cap_exhausted=False,
            wall_time_exhausted=False,
            jit_compile=cfg.jit_compile,
            exception_type=None,
            scale=np.asarray(scale_tensor.numpy(), dtype=float),
        )

    attempts = tf.Variable(0, trainable=False, dtype=tf.int32)
    target_rows = tf.Variable(0, trainable=False, dtype=tf.int32)
    cap_exhausted = tf.Variable(False, trainable=False, dtype=tf.bool)
    wall_time_exhausted = tf.Variable(False, trainable=False, dtype=tf.bool)
    cap = tf.constant(cfg.max_objective_evaluations, tf.int32)
    started = time.monotonic()

    def wall_guard_passed() -> tf.Tensor:
        if cfg.max_wall_seconds is None:
            return tf.constant(True)

        def remaining() -> np.ndarray:
            return np.asarray(
                time.monotonic() - started <= cfg.max_wall_seconds,
                dtype=np.bool_,
            )

        return tf.reshape(tf.py_function(remaining, [], tf.bool), [])

    def objective_and_gradient(z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        updated_attempts = attempts.assign_add(1)

        def evaluate_target() -> tuple[tf.Tensor, tf.Tensor]:
            updated_rows = target_rows.assign_add(1)
            with tf.control_dependencies([updated_rows]):
                return _standardized_objective_and_gradient(
                    value_and_score_fn, initial, scale_tensor, z
                )

        def guarded_nonfinite(flag: tf.Variable) -> tuple[tf.Tensor, tf.Tensor]:
            updated_flag = flag.assign(True)
            with tf.control_dependencies([updated_flag]):
                z_tensor = tf.reshape(tf.convert_to_tensor(z, tf.float64), [-1])
                return (
                    tf.constant(np.inf, tf.float64),
                    tf.fill(tf.shape(z_tensor), tf.constant(np.nan, tf.float64)),
                )

        with tf.control_dependencies([updated_attempts]):
            return tf.cond(
                wall_guard_passed(),
                lambda: tf.cond(
                    target_rows.read_value() < cap,
                    evaluate_target,
                    lambda: guarded_nonfinite(cap_exhausted),
                ),
                lambda: guarded_nonfinite(wall_time_exhausted),
            )

    def run_optimizer() -> Any:
        return tfp.optimizer.lbfgs_minimize(
            objective_and_gradient,
            initial_position=tf.zeros([dimension], tf.float64),
            num_correction_pairs=cfg.num_correction_pairs,
            tolerance=tf.constant(cfg.gradient_tolerance, tf.float64),
            x_tolerance=tf.constant(cfg.x_tolerance, tf.float64),
            f_relative_tolerance=tf.constant(
                cfg.f_relative_tolerance, tf.float64
            ),
            max_iterations=cfg.max_iterations,
            parallel_iterations=cfg.parallel_iterations,
            max_line_search_iterations=cfg.max_line_search_iterations,
            f_absolute_tolerance=tf.constant(
                cfg.f_absolute_tolerance, tf.float64
            ),
        )

    compiled_optimizer = tf.function(
        run_optimizer,
        jit_compile=cfg.jit_compile,
        reduce_retracing=False,
    )
    try:
        optimizer = compiled_optimizer()
    except Exception as exc:  # noqa: BLE001 - typed fail-closed locator result.
        return _result(
            status="optimizer_exception",
            accepted=False,
            initial_position=initial_np,
            endpoint_position=initial_np,
            initial_value=initial_value,
            endpoint_value=initial_value,
            initial_score=initial_score,
            endpoint_score=initial_score,
            optimizer_converged=False,
            optimizer_failed=False,
            optimizer_iterations=0,
            reported_evaluations=int(attempts.numpy()),
            callback_attempts=int(attempts.numpy()),
            optimizer_target_rows=int(target_rows.numpy()),
            physical_target_rows=1 + int(target_rows.numpy()),
            cap_exhausted=bool(cap_exhausted.numpy()),
            wall_time_exhausted=bool(wall_time_exhausted.numpy()),
            jit_compile=cfg.jit_compile,
            exception_type=type(exc).__name__,
            scale=np.asarray(scale_tensor.numpy(), dtype=float),
        )

    callback_attempts = int(attempts.numpy())
    optimizer_target_rows = int(target_rows.numpy())
    exhausted = bool(cap_exhausted.numpy())
    timed_out = bool(wall_time_exhausted.numpy())
    reported_evaluations = int(optimizer.num_objective_evaluations.numpy())
    optimizer_iterations = int(optimizer.num_iterations.numpy())
    optimizer_converged = bool(optimizer.converged.numpy())
    optimizer_failed = bool(optimizer.failed.numpy())
    candidate_z = tf.reshape(tf.convert_to_tensor(optimizer.position, tf.float64), [-1])
    candidate = initial + scale_tensor * candidate_z
    endpoint_value, endpoint_score, endpoint_valid = _evaluate_value_score(
        value_and_score_fn, candidate, dimension
    )
    candidate_np = np.asarray(candidate.numpy(), dtype=float)
    accounting_valid = bool(
        reported_evaluations == callback_attempts
        and (
            (
                not exhausted
                and not timed_out
                and callback_attempts == optimizer_target_rows
                and optimizer_target_rows <= cfg.max_objective_evaluations
            )
            or (
                (exhausted or timed_out)
                and callback_attempts > optimizer_target_rows
                and (
                    timed_out
                    or optimizer_target_rows == cfg.max_objective_evaluations
                )
            )
        )
    )
    accepted = bool(
        accounting_valid
        and not exhausted
        and not timed_out
        and not optimizer_failed
        and endpoint_valid
        and endpoint_value >= initial_value
    )
    if not accounting_valid:
        status = "evaluation_accounting_invalid"
    elif exhausted:
        status = "evaluation_cap_exhausted"
    elif timed_out:
        status = "wall_time_exhausted"
    elif optimizer_failed:
        status = "optimizer_failed"
    elif not endpoint_valid:
        status = "endpoint_target_invalid"
    elif endpoint_value < initial_value:
        status = "endpoint_objective_decrease"
    elif optimizer_converged:
        status = "converged"
    else:
        status = "iteration_limit"
    return _result(
        status=status,
        accepted=accepted,
        initial_position=initial_np,
        endpoint_position=candidate_np,
        initial_value=initial_value,
        endpoint_value=endpoint_value,
        initial_score=initial_score,
        endpoint_score=endpoint_score,
        optimizer_converged=optimizer_converged,
        optimizer_failed=optimizer_failed,
        optimizer_iterations=optimizer_iterations,
        reported_evaluations=reported_evaluations,
        callback_attempts=callback_attempts,
        optimizer_target_rows=optimizer_target_rows,
        physical_target_rows=optimizer_target_rows + 2,
        cap_exhausted=exhausted,
        wall_time_exhausted=timed_out,
        jit_compile=cfg.jit_compile,
        exception_type=None,
        scale=np.asarray(scale_tensor.numpy(), dtype=float),
    )


def locate_joint_center_staged(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: Any,
    *,
    checkpoint_validator: Callable[[JointCenterCheckpoint], bool],
    scale: Any | None = None,
    config: JointCenterStagedConfig | None = None,
) -> JointCenterStagedResult:
    """Replay a checkpoint, validate it once, and resume the same L-BFGS state."""

    cfg = JointCenterStagedConfig() if config is None else config
    initial = _vector(initial_position, "initial_position")
    dimension = int(initial.shape[0])
    if dimension <= 0:
        raise ValueError("initial_position must be nonempty")
    scale_tensor = (
        tf.ones([dimension], tf.float64)
        if scale is None
        else _vector(scale, "scale")
    )
    if scale_tensor.shape != (dimension,) or not bool(
        tf.reduce_all(tf.math.is_finite(scale_tensor) & (scale_tensor > 0.0)).numpy()
    ):
        raise ValueError("scale must be positive finite with one entry per coordinate")
    scale_np = np.asarray(scale_tensor.numpy(), dtype=float)
    initial_value, initial_score, initial_valid = _evaluate_value_score(
        value_and_score_fn, initial, dimension
    )
    initial_np = np.asarray(initial.numpy(), dtype=float)
    if not initial_valid:
        checkpoint = _checkpoint(
            status="initial_target_invalid",
            accepted=False,
            position=initial_np,
            value=initial_value,
            score=initial_score,
            scale=scale_np,
            optimizer_converged=False,
            optimizer_failed=False,
            optimizer_iterations=0,
            reported_evaluations=0,
            callback_attempts=0,
            optimizer_target_rows=0,
            physical_target_rows=1,
        )
        return _staged_result(
            status="initial_target_invalid",
            accepted=False,
            initial_position=initial_np,
            initial_value=initial_value,
            initial_score=initial_score,
            checkpoint=checkpoint,
            checkpoint_validated=False,
            checkpoint_validator_calls=0,
            continuation_started=False,
            endpoint_position=initial_np,
            endpoint_value=initial_value,
            endpoint_score=initial_score,
            optimizer_converged=False,
            optimizer_failed=False,
            optimizer_iterations=0,
            reported_evaluations=0,
            callback_attempts=0,
            optimizer_target_rows=0,
            physical_target_rows=1,
            cap_exhausted=False,
            wall_time_exhausted=False,
            jit_compile=cfg.jit_compile,
            exception_type=None,
            scale=scale_np,
        )

    attempts = tf.Variable(0, trainable=False, dtype=tf.int32)
    target_rows = tf.Variable(0, trainable=False, dtype=tf.int32)
    cap_exhausted = tf.Variable(False, trainable=False, dtype=tf.bool)
    wall_time_exhausted = tf.Variable(False, trainable=False, dtype=tf.bool)
    cap = tf.constant(cfg.max_objective_evaluations, tf.int32)
    started = time.monotonic()

    def wall_guard_passed() -> tf.Tensor:
        if cfg.max_wall_seconds is None:
            return tf.constant(True)

        def remaining() -> np.ndarray:
            return np.asarray(
                time.monotonic() - started <= cfg.max_wall_seconds,
                dtype=np.bool_,
            )

        return tf.reshape(tf.py_function(remaining, [], tf.bool), [])

    def objective_and_gradient(z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        updated_attempts = attempts.assign_add(1)

        def evaluate_target() -> tuple[tf.Tensor, tf.Tensor]:
            updated_rows = target_rows.assign_add(1)
            with tf.control_dependencies([updated_rows]):
                return _standardized_objective_and_gradient(
                    value_and_score_fn, initial, scale_tensor, z
                )

        def guarded_nonfinite(flag: tf.Variable) -> tuple[tf.Tensor, tf.Tensor]:
            updated_flag = flag.assign(True)
            with tf.control_dependencies([updated_flag]):
                z_tensor = tf.reshape(tf.convert_to_tensor(z, tf.float64), [-1])
                return (
                    tf.constant(np.inf, tf.float64),
                    tf.fill(tf.shape(z_tensor), tf.constant(np.nan, tf.float64)),
                )

        with tf.control_dependencies([updated_attempts]):
            return tf.cond(
                wall_guard_passed(),
                lambda: tf.cond(
                    target_rows.read_value() < cap,
                    evaluate_target,
                    lambda: guarded_nonfinite(cap_exhausted),
                ),
                lambda: guarded_nonfinite(wall_time_exhausted),
            )

    def optimizer_kwargs(max_iterations: int) -> dict[str, Any]:
        return {
            "num_correction_pairs": cfg.num_correction_pairs,
            "tolerance": tf.constant(cfg.gradient_tolerance, tf.float64),
            "x_tolerance": tf.constant(cfg.x_tolerance, tf.float64),
            "f_relative_tolerance": tf.constant(
                cfg.f_relative_tolerance, tf.float64
            ),
            "max_iterations": max_iterations,
            "parallel_iterations": cfg.parallel_iterations,
            "max_line_search_iterations": cfg.max_line_search_iterations,
            "f_absolute_tolerance": tf.constant(
                cfg.f_absolute_tolerance, tf.float64
            ),
        }

    def run_checkpoint() -> Any:
        return tfp.optimizer.lbfgs_minimize(
            objective_and_gradient,
            initial_position=tf.zeros([dimension], tf.float64),
            **optimizer_kwargs(cfg.checkpoint_iterations),
        )

    compiled_checkpoint = tf.function(
        run_checkpoint,
        jit_compile=cfg.jit_compile,
        reduce_retracing=False,
    )
    try:
        optimizer = compiled_checkpoint()
    except Exception as exc:  # noqa: BLE001 - fail-closed staged result.
        checkpoint = _checkpoint(
            status="optimizer_exception",
            accepted=False,
            position=initial_np,
            value=initial_value,
            score=initial_score,
            scale=scale_np,
            optimizer_converged=False,
            optimizer_failed=False,
            optimizer_iterations=0,
            reported_evaluations=int(attempts.numpy()),
            callback_attempts=int(attempts.numpy()),
            optimizer_target_rows=int(target_rows.numpy()),
            physical_target_rows=1 + int(target_rows.numpy()),
        )
        return _staged_result(
            status="optimizer_exception",
            accepted=False,
            initial_position=initial_np,
            initial_value=initial_value,
            initial_score=initial_score,
            checkpoint=checkpoint,
            checkpoint_validated=False,
            checkpoint_validator_calls=0,
            continuation_started=False,
            endpoint_position=initial_np,
            endpoint_value=initial_value,
            endpoint_score=initial_score,
            optimizer_converged=False,
            optimizer_failed=False,
            optimizer_iterations=0,
            reported_evaluations=int(attempts.numpy()),
            callback_attempts=int(attempts.numpy()),
            optimizer_target_rows=int(target_rows.numpy()),
            physical_target_rows=1 + int(target_rows.numpy()),
            cap_exhausted=bool(cap_exhausted.numpy()),
            wall_time_exhausted=bool(wall_time_exhausted.numpy()),
            jit_compile=cfg.jit_compile,
            exception_type=type(exc).__name__,
            scale=scale_np,
        )

    checkpoint_candidate = initial + scale_tensor * tf.reshape(
        tf.convert_to_tensor(optimizer.position, tf.float64), [-1]
    )
    checkpoint_value, checkpoint_score, checkpoint_valid = _evaluate_value_score(
        value_and_score_fn, checkpoint_candidate, dimension
    )
    checkpoint_np = np.asarray(checkpoint_candidate.numpy(), dtype=float)
    checkpoint_attempts = int(attempts.numpy())
    checkpoint_rows = int(target_rows.numpy())
    checkpoint_reported = int(optimizer.num_objective_evaluations.numpy())
    checkpoint_converged = bool(optimizer.converged.numpy())
    checkpoint_failed = bool(optimizer.failed.numpy())
    checkpoint_iterations = int(optimizer.num_iterations.numpy())
    exhausted = bool(cap_exhausted.numpy())
    timed_out = bool(wall_time_exhausted.numpy())
    checkpoint_accounting = _accounting_valid(
        reported_evaluations=checkpoint_reported,
        callback_attempts=checkpoint_attempts,
        optimizer_target_rows=checkpoint_rows,
        max_objective_evaluations=cfg.max_objective_evaluations,
        cap_exhausted=exhausted,
        wall_time_exhausted=timed_out,
    )
    checkpoint_accepted = bool(
        checkpoint_accounting
        and not exhausted
        and not timed_out
        and not checkpoint_failed
        and checkpoint_valid
        and checkpoint_value >= initial_value
    )
    checkpoint_status = _optimizer_status(
        accounting_valid=checkpoint_accounting,
        cap_exhausted=exhausted,
        wall_time_exhausted=timed_out,
        optimizer_failed=checkpoint_failed,
        endpoint_valid=checkpoint_valid,
        objective_nondecreasing=checkpoint_value >= initial_value,
        optimizer_converged=checkpoint_converged,
    )
    checkpoint = _checkpoint(
        status=checkpoint_status,
        accepted=checkpoint_accepted,
        position=checkpoint_np,
        value=checkpoint_value,
        score=checkpoint_score,
        scale=scale_np,
        optimizer_converged=checkpoint_converged,
        optimizer_failed=checkpoint_failed,
        optimizer_iterations=checkpoint_iterations,
        reported_evaluations=checkpoint_reported,
        callback_attempts=checkpoint_attempts,
        optimizer_target_rows=checkpoint_rows,
        physical_target_rows=checkpoint_rows + 2,
    )
    if not checkpoint_accepted:
        return _staged_result(
            status=checkpoint_status,
            accepted=False,
            initial_position=initial_np,
            initial_value=initial_value,
            initial_score=initial_score,
            checkpoint=checkpoint,
            checkpoint_validated=False,
            checkpoint_validator_calls=0,
            continuation_started=False,
            endpoint_position=checkpoint_np,
            endpoint_value=checkpoint_value,
            endpoint_score=checkpoint_score,
            optimizer_converged=checkpoint_converged,
            optimizer_failed=checkpoint_failed,
            optimizer_iterations=checkpoint_iterations,
            reported_evaluations=checkpoint_reported,
            callback_attempts=checkpoint_attempts,
            optimizer_target_rows=checkpoint_rows,
            physical_target_rows=checkpoint_rows + 2,
            cap_exhausted=exhausted,
            wall_time_exhausted=timed_out,
            jit_compile=cfg.jit_compile,
            exception_type=None,
            scale=scale_np,
        )

    try:
        checkpoint_validated = bool(checkpoint_validator(checkpoint))
    except Exception as exc:  # noqa: BLE001 - validator failure is typed.
        return _staged_result(
            status="checkpoint_validator_exception",
            accepted=False,
            initial_position=initial_np,
            initial_value=initial_value,
            initial_score=initial_score,
            checkpoint=checkpoint,
            checkpoint_validated=False,
            checkpoint_validator_calls=1,
            continuation_started=False,
            endpoint_position=checkpoint_np,
            endpoint_value=checkpoint_value,
            endpoint_score=checkpoint_score,
            optimizer_converged=checkpoint_converged,
            optimizer_failed=checkpoint_failed,
            optimizer_iterations=checkpoint_iterations,
            reported_evaluations=checkpoint_reported,
            callback_attempts=checkpoint_attempts,
            optimizer_target_rows=checkpoint_rows,
            physical_target_rows=checkpoint_rows + 2,
            cap_exhausted=exhausted,
            wall_time_exhausted=timed_out,
            jit_compile=cfg.jit_compile,
            exception_type=type(exc).__name__,
            scale=scale_np,
        )
    if not checkpoint_validated:
        return _staged_result(
            status="checkpoint_rejected",
            accepted=False,
            initial_position=initial_np,
            initial_value=initial_value,
            initial_score=initial_score,
            checkpoint=checkpoint,
            checkpoint_validated=False,
            checkpoint_validator_calls=1,
            continuation_started=False,
            endpoint_position=checkpoint_np,
            endpoint_value=checkpoint_value,
            endpoint_score=checkpoint_score,
            optimizer_converged=checkpoint_converged,
            optimizer_failed=checkpoint_failed,
            optimizer_iterations=checkpoint_iterations,
            reported_evaluations=checkpoint_reported,
            callback_attempts=checkpoint_attempts,
            optimizer_target_rows=checkpoint_rows,
            physical_target_rows=checkpoint_rows + 2,
            cap_exhausted=exhausted,
            wall_time_exhausted=timed_out,
            jit_compile=cfg.jit_compile,
            exception_type=None,
            scale=scale_np,
        )

    def run_continuation() -> Any:
        return tfp.optimizer.lbfgs_minimize(
            objective_and_gradient,
            initial_position=None,
            previous_optimizer_results=optimizer,
            **optimizer_kwargs(cfg.total_iterations),
        )

    compiled_continuation = tf.function(
        run_continuation,
        jit_compile=cfg.jit_compile,
        reduce_retracing=False,
    )
    try:
        final_optimizer = compiled_continuation()
    except Exception as exc:  # noqa: BLE001 - fail-closed continuation.
        return _staged_result(
            status="optimizer_exception",
            accepted=False,
            initial_position=initial_np,
            initial_value=initial_value,
            initial_score=initial_score,
            checkpoint=checkpoint,
            checkpoint_validated=True,
            checkpoint_validator_calls=1,
            continuation_started=True,
            endpoint_position=checkpoint_np,
            endpoint_value=checkpoint_value,
            endpoint_score=checkpoint_score,
            optimizer_converged=checkpoint_converged,
            optimizer_failed=checkpoint_failed,
            optimizer_iterations=checkpoint_iterations,
            reported_evaluations=int(attempts.numpy()),
            callback_attempts=int(attempts.numpy()),
            optimizer_target_rows=int(target_rows.numpy()),
            physical_target_rows=2 + int(target_rows.numpy()),
            cap_exhausted=bool(cap_exhausted.numpy()),
            wall_time_exhausted=bool(wall_time_exhausted.numpy()),
            jit_compile=cfg.jit_compile,
            exception_type=type(exc).__name__,
            scale=scale_np,
        )

    final_candidate = initial + scale_tensor * tf.reshape(
        tf.convert_to_tensor(final_optimizer.position, tf.float64), [-1]
    )
    endpoint_value, endpoint_score, endpoint_valid = _evaluate_value_score(
        value_and_score_fn, final_candidate, dimension
    )
    endpoint_np = np.asarray(final_candidate.numpy(), dtype=float)
    callback_attempts = int(attempts.numpy())
    optimizer_target_rows = int(target_rows.numpy())
    exhausted = bool(cap_exhausted.numpy())
    timed_out = bool(wall_time_exhausted.numpy())
    reported_evaluations = int(final_optimizer.num_objective_evaluations.numpy())
    optimizer_iterations = int(final_optimizer.num_iterations.numpy())
    optimizer_converged = bool(final_optimizer.converged.numpy())
    optimizer_failed = bool(final_optimizer.failed.numpy())
    accounting_valid = _accounting_valid(
        reported_evaluations=reported_evaluations,
        callback_attempts=callback_attempts,
        optimizer_target_rows=optimizer_target_rows,
        max_objective_evaluations=cfg.max_objective_evaluations,
        cap_exhausted=exhausted,
        wall_time_exhausted=timed_out,
    )
    accepted = bool(
        accounting_valid
        and not exhausted
        and not timed_out
        and not optimizer_failed
        and endpoint_valid
        and endpoint_value >= checkpoint_value
    )
    status = _optimizer_status(
        accounting_valid=accounting_valid,
        cap_exhausted=exhausted,
        wall_time_exhausted=timed_out,
        optimizer_failed=optimizer_failed,
        endpoint_valid=endpoint_valid,
        objective_nondecreasing=endpoint_value >= checkpoint_value,
        optimizer_converged=optimizer_converged,
    )
    return _staged_result(
        status=status,
        accepted=accepted,
        initial_position=initial_np,
        initial_value=initial_value,
        initial_score=initial_score,
        checkpoint=checkpoint,
        checkpoint_validated=True,
        checkpoint_validator_calls=1,
        continuation_started=True,
        endpoint_position=endpoint_np,
        endpoint_value=endpoint_value,
        endpoint_score=endpoint_score,
        optimizer_converged=optimizer_converged,
        optimizer_failed=optimizer_failed,
        optimizer_iterations=optimizer_iterations,
        reported_evaluations=reported_evaluations,
        callback_attempts=callback_attempts,
        optimizer_target_rows=optimizer_target_rows,
        physical_target_rows=optimizer_target_rows + 3,
        cap_exhausted=exhausted,
        wall_time_exhausted=timed_out,
        jit_compile=cfg.jit_compile,
        exception_type=None,
        scale=scale_np,
    )


def _vector(value: Any, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 1:
        raise ValueError(f"{name} must be a rank-1 vector")
    if tensor.shape[0] is None or not bool(
        tf.reduce_all(tf.math.is_finite(tensor)).numpy()
    ):
        raise ValueError(f"{name} must be a finite vector with static size")
    return tensor


def _evaluate_value_score(
    function: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    position: tf.Tensor,
    dimension: int,
) -> tuple[float, np.ndarray, bool]:
    try:
        value, score = function(position)
        value_tensor = tf.reshape(tf.convert_to_tensor(value, tf.float64), [])
        score_tensor = tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1])
        if score_tensor.shape != (dimension,):
            return float("nan"), np.full(dimension, np.nan), False
        value_np = float(value_tensor.numpy())
        score_np = np.asarray(score_tensor.numpy(), dtype=float)
        valid = bool(np.isfinite(value_np) and np.all(np.isfinite(score_np)))
        return value_np, score_np, valid
    except Exception:  # noqa: BLE001 - malformed exact target is invalid evidence.
        return float("nan"), np.full(dimension, np.nan), False


def _accounting_valid(
    *,
    reported_evaluations: int,
    callback_attempts: int,
    optimizer_target_rows: int,
    max_objective_evaluations: int,
    cap_exhausted: bool,
    wall_time_exhausted: bool,
) -> bool:
    return bool(
        reported_evaluations == callback_attempts
        and (
            (
                not cap_exhausted
                and not wall_time_exhausted
                and callback_attempts == optimizer_target_rows
                and optimizer_target_rows <= max_objective_evaluations
            )
            or (
                (cap_exhausted or wall_time_exhausted)
                and callback_attempts > optimizer_target_rows
                and (
                    wall_time_exhausted
                    or optimizer_target_rows == max_objective_evaluations
                )
            )
        )
    )


def _optimizer_status(
    *,
    accounting_valid: bool,
    cap_exhausted: bool,
    wall_time_exhausted: bool,
    optimizer_failed: bool,
    endpoint_valid: bool,
    objective_nondecreasing: bool,
    optimizer_converged: bool,
) -> str:
    if not accounting_valid:
        return "evaluation_accounting_invalid"
    if cap_exhausted:
        return "evaluation_cap_exhausted"
    if wall_time_exhausted:
        return "wall_time_exhausted"
    if optimizer_failed:
        return "optimizer_failed"
    if not endpoint_valid:
        return "endpoint_target_invalid"
    if not objective_nondecreasing:
        return "endpoint_objective_decrease"
    return "converged" if optimizer_converged else "iteration_limit"


def _checkpoint(
    *,
    status: str,
    accepted: bool,
    position: np.ndarray,
    value: float,
    score: np.ndarray,
    scale: np.ndarray,
    optimizer_converged: bool,
    optimizer_failed: bool,
    optimizer_iterations: int,
    reported_evaluations: int,
    callback_attempts: int,
    optimizer_target_rows: int,
    physical_target_rows: int,
) -> JointCenterCheckpoint:
    scaled_score = np.asarray(score, dtype=float) * scale
    return JointCenterCheckpoint(
        status=str(status),
        endpoint_accepted=bool(accepted),
        position=position,
        score=score,
        objective=float(value),
        score_l2=float(np.linalg.norm(scaled_score)),
        score_max_abs=float(np.max(np.abs(scaled_score))),
        optimizer_converged=bool(optimizer_converged),
        optimizer_failed=bool(optimizer_failed),
        optimizer_iterations=int(optimizer_iterations),
        reported_objective_evaluations=int(reported_evaluations),
        callback_attempts=int(callback_attempts),
        optimizer_target_rows=int(optimizer_target_rows),
        physical_target_rows=int(physical_target_rows),
    )


def _staged_result(
    *,
    status: str,
    accepted: bool,
    initial_position: np.ndarray,
    initial_value: float,
    initial_score: np.ndarray,
    checkpoint: JointCenterCheckpoint,
    checkpoint_validated: bool,
    checkpoint_validator_calls: int,
    continuation_started: bool,
    endpoint_position: np.ndarray,
    endpoint_value: float,
    endpoint_score: np.ndarray,
    optimizer_converged: bool,
    optimizer_failed: bool,
    optimizer_iterations: int,
    reported_evaluations: int,
    callback_attempts: int,
    optimizer_target_rows: int,
    physical_target_rows: int,
    cap_exhausted: bool,
    wall_time_exhausted: bool,
    jit_compile: bool,
    exception_type: str | None,
    scale: np.ndarray,
) -> JointCenterStagedResult:
    initial_scaled = np.asarray(initial_score, dtype=float) * scale
    endpoint_scaled = np.asarray(endpoint_score, dtype=float) * scale
    return JointCenterStagedResult(
        status=str(status),
        endpoint_accepted=bool(accepted),
        initial_position=initial_position,
        initial_score=initial_score,
        initial_objective=float(initial_value),
        initial_score_l2=float(np.linalg.norm(initial_scaled)),
        initial_score_max_abs=float(np.max(np.abs(initial_scaled))),
        checkpoint=checkpoint,
        checkpoint_validated=bool(checkpoint_validated),
        checkpoint_validator_calls=int(checkpoint_validator_calls),
        continuation_started=bool(continuation_started),
        endpoint_position=endpoint_position,
        endpoint_score=endpoint_score,
        endpoint_objective=float(endpoint_value),
        endpoint_score_l2=float(np.linalg.norm(endpoint_scaled)),
        endpoint_score_max_abs=float(np.max(np.abs(endpoint_scaled))),
        optimizer_converged=bool(optimizer_converged),
        optimizer_failed=bool(optimizer_failed),
        optimizer_iterations=int(optimizer_iterations),
        reported_objective_evaluations=int(reported_evaluations),
        callback_attempts=int(callback_attempts),
        optimizer_target_rows=int(optimizer_target_rows),
        physical_target_rows=int(physical_target_rows),
        cap_exhausted=bool(cap_exhausted),
        wall_time_exhausted=bool(wall_time_exhausted),
        jit_compile=bool(jit_compile),
        exception_type=exception_type,
    )


def _result(
    *,
    status: str,
    accepted: bool,
    initial_position: np.ndarray,
    endpoint_position: np.ndarray,
    initial_value: float,
    endpoint_value: float,
    initial_score: np.ndarray,
    endpoint_score: np.ndarray,
    optimizer_converged: bool,
    optimizer_failed: bool,
    optimizer_iterations: int,
    reported_evaluations: int,
    callback_attempts: int,
    optimizer_target_rows: int,
    physical_target_rows: int,
    cap_exhausted: bool,
    wall_time_exhausted: bool,
    jit_compile: bool,
    exception_type: str | None,
    scale: np.ndarray,
) -> JointCenterResult:
    initial_scaled = np.asarray(initial_score, dtype=float) * scale
    endpoint_scaled = np.asarray(endpoint_score, dtype=float) * scale
    return JointCenterResult(
        status=status,
        endpoint_accepted=accepted,
        initial_position=initial_position,
        endpoint_position=endpoint_position,
        initial_score=initial_score,
        endpoint_score=endpoint_score,
        initial_objective=float(initial_value),
        endpoint_objective=float(endpoint_value),
        initial_score_l2=float(np.linalg.norm(initial_scaled)),
        endpoint_score_l2=float(np.linalg.norm(endpoint_scaled)),
        initial_score_max_abs=float(np.max(np.abs(initial_scaled))),
        endpoint_score_max_abs=float(np.max(np.abs(endpoint_scaled))),
        optimizer_converged=optimizer_converged,
        optimizer_failed=optimizer_failed,
        optimizer_iterations=int(optimizer_iterations),
        reported_objective_evaluations=int(reported_evaluations),
        callback_attempts=int(callback_attempts),
        optimizer_target_rows=int(optimizer_target_rows),
        physical_target_rows=int(physical_target_rows),
        cap_exhausted=cap_exhausted,
        wall_time_exhausted=wall_time_exhausted,
        jit_compile=jit_compile,
        exception_type=exception_type,
    )
