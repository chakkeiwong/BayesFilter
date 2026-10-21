"""Transactional block-coordinate composition of exact sequential locators.

For an ordered block ``b``, the module freezes all other coordinates, calls
the existing exact-target sequential locator on ``theta_b``, embeds the
candidate back into the full state, and replays the complete value and score.
Only a finite, nondecreasing full objective commits the Gauss-Seidel update.

This one-sweep diagnostic does not establish a MAP, convergence, geometry, or
HMC readiness. In particular, block-local precision and covariance estimates
are deliberately discarded at the transaction boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tensorflow as tf

from bayesfilter.inference.program_cache_scope import scoped_program_cache
from bayesfilter.inference.sequential_map_covariance import (
    SequentialMapCovarianceConfig,
)
from bayesfilter.ops.host_tensor_io import numeric_tensor

BLOCK_COORDINATE_CENTER_NONCLAIMS = (
    "one ordered block-coordinate sweep only",
    "conditional exact-target localization diagnostic only",
    "not a certified local or global MAP",
    "not convergence evidence",
    "not Hessian or mass-matrix evidence",
    "not HMC readiness evidence",
    "not posterior correctness evidence",
    "not default-readiness evidence",
)

_DISCARDED_INTERNAL_GEOMETRY_FIELDS = frozenset(
    {
        "precision",
        "covariance",
        "matrix",
        "eigenvalues",
        "raw_precision_z",
        "projected_precision_z",
        "raw_eigenvalues",
        "projected_eigenvalues",
    }
)
_EPS64_SQRT = math.sqrt(math.ulp(1.0))


@dataclass(frozen=True)
class BlockCoordinateCenterBlock:
    """One half-open coordinate block and its existing locator policy."""

    name: str
    start: int
    stop: int
    sequential_config: SequentialMapCovarianceConfig

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("block name must be nonempty")
        start = int(self.start)
        stop = int(self.stop)
        if start < 0 or stop <= start:
            raise ValueError("block bounds must satisfy 0 <= start < stop")
        if not isinstance(self.sequential_config, SequentialMapCovarianceConfig):
            raise TypeError("sequential_config must be SequentialMapCovarianceConfig")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)


@dataclass(frozen=True)
class BlockCoordinateCenterConfig:
    """Global transaction, reversal, cycle, and row-budget policy."""

    max_physical_target_rows: int = 4096
    reversal_ratio: float = 1.25
    repeat_threshold_factor: float = 10.0
    stop_on_material_reversal: bool = True
    require_scheduled_block_maxima_no_worse: bool = True

    def __post_init__(self) -> None:
        rows = int(self.max_physical_target_rows)
        ratio = float(self.reversal_ratio)
        repeat_factor = float(self.repeat_threshold_factor)
        if rows <= 0:
            raise ValueError("max_physical_target_rows must be positive")
        if not math.isfinite(ratio) or ratio <= 1.0:
            raise ValueError("reversal_ratio must be finite and greater than one")
        if not math.isfinite(repeat_factor) or repeat_factor <= 0.0:
            raise ValueError("repeat_threshold_factor must be positive finite")
        object.__setattr__(self, "max_physical_target_rows", rows)
        object.__setattr__(self, "reversal_ratio", ratio)
        object.__setattr__(self, "repeat_threshold_factor", repeat_factor)
        object.__setattr__(
            self,
            "stop_on_material_reversal",
            bool(self.stop_on_material_reversal),
        )
        object.__setattr__(
            self,
            "require_scheduled_block_maxima_no_worse",
            bool(self.require_scheduled_block_maxima_no_worse),
        )


@dataclass(frozen=True)
class BlockCoordinateCenterResult:
    """One-sweep result with an array-free public payload by default."""

    completed: bool
    status: str
    initial_center: tf.Tensor
    final_center: tf.Tensor
    initial_score: tf.Tensor
    final_score: tf.Tensor
    initial_objective: float
    final_objective: float
    initial_score_l2: float
    final_score_l2: float
    initial_score_max_abs: float
    final_score_max_abs: float
    completed_block_count: int
    accepted_block_count: int
    transaction_rejection_count: int
    sequential_exact_evaluations: int
    physical_target_rows: int
    maximum_physical_target_rows: int
    material_reversal_detected: bool
    repeat_cycle_detected: bool
    two_step_return_cycle_detected: bool
    scheduled_block_maxima_no_worse: bool
    stop_on_material_reversal: bool
    require_scheduled_block_maxima_no_worse: bool
    private_block_records: tuple[Mapping[str, Any], ...]
    nonclaims: tuple[str, ...] = BLOCK_COORDINATE_CENTER_NONCLAIMS
    numerical_summary: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        for name in ("initial_center", "final_center", "initial_score", "final_score"):
            array = tf.identity(numeric_tensor(getattr(self, name), tf.float64))
            object.__setattr__(self, name, array)
        object.__setattr__(self, "completed", bool(self.completed))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(
            self,
            "private_block_records",
            tuple(_json_ready(dict(row)) for row in self.private_block_records),
        )
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))

    def payload(self) -> Mapping[str, Any]:
        """Return the public summary without states, scores, or block traces."""

        payload = {
            "schema": "bayesfilter.block_coordinate_center.public.v1",
            "completed": self.completed,
            "status": self.status,
            "completed_block_count": self.completed_block_count,
            "accepted_block_count": self.accepted_block_count,
            "transaction_rejection_count": self.transaction_rejection_count,
            "sequential_exact_evaluations": self.sequential_exact_evaluations,
            "physical_target_rows": self.physical_target_rows,
            "maximum_physical_target_rows": self.maximum_physical_target_rows,
            **(dict(self.numerical_summary) if self.numerical_summary is not None else {
                # Directly constructed records retain compatibility summaries;
                # the public controller supplies its completed tensor decisions.
                "objective_nondecreasing": self.final_objective >= self.initial_objective,
                "objective_progress_resolvable": _resolvable_decrease(
                    -self.initial_objective, -self.final_objective),
                "score_l2_progress_resolvable": _resolvable_decrease(
                    self.initial_score_l2, self.final_score_l2),
                "score_max_progress_resolvable": _resolvable_decrease(
                    self.initial_score_max_abs, self.final_score_max_abs),
            }),
            "material_reversal_detected": self.material_reversal_detected,
            "repeat_cycle_detected": self.repeat_cycle_detected,
            "two_step_return_cycle_detected": (
                self.two_step_return_cycle_detected
            ),
            "scheduled_block_maxima_no_worse": (
                self.scheduled_block_maxima_no_worse
            ),
            "nonclaims": list(self.nonclaims),
        }
        if (
            not self.stop_on_material_reversal
            or not self.require_scheduled_block_maxima_no_worse
        ):
            payload["policy_overrides"] = {
                "stop_on_material_reversal": self.stop_on_material_reversal,
                "require_scheduled_block_maxima_no_worse": (
                    self.require_scheduled_block_maxima_no_worse
                ),
            }
        return payload

    def private_payload(self) -> Mapping[str, Any]:
        """Return the explicit private state, score, and block trace payload."""

        return _json_ready(
            {
                "schema": "bayesfilter.block_coordinate_center.private.v1",
                "public_summary": self.payload(),
                "initial_center": self.initial_center,
                "final_center": self.final_center,
                "initial_score": self.initial_score,
                "final_score": self.final_score,
                "initial_objective": self.initial_objective,
                "final_objective": self.final_objective,
                "initial_score_l2": self.initial_score_l2,
                "final_score_l2": self.final_score_l2,
                "initial_score_max_abs": self.initial_score_max_abs,
                "final_score_max_abs": self.final_score_max_abs,
                "block_records": self.private_block_records,
                "nonclaims": self.nonclaims,
            }
        )


def classify_center_trace_cycles(
    standardized_centers: Sequence[Any], repeat_threshold: float
) -> Mapping[str, bool]:
    """Classify defensive repeat/two-step cycles in a standardized trace.

    A valid one-sweep nonoverlapping partition cannot generate a resolvable
    cycle. This helper remains public for direct invariant tests and for future
    schedules that may have a separately reviewed overlap policy.
    """

    threshold = float(repeat_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("repeat_threshold must be positive finite")
    centers = tuple(tf.reshape(numeric_tensor(center, tf.float64), [-1]) for center in standardized_centers)
    if not centers:
        return {"repeat_cycle": False, "two_step_return_cycle": False}
    if any(center.shape != centers[0].shape for center in centers):
        raise ValueError("standardized centers must be finite and shape-compatible")
    repeat, two_step, finite = _numerical_call(
        _cycle_core, tf.stack(centers), tf.constant(threshold, tf.float64)
    )
    if not bool(finite):
        raise ValueError("standardized centers must be finite and shape-compatible")
    return {"repeat_cycle": bool(repeat), "two_step_return_cycle": bool(two_step)}


def locate_block_coordinate_center(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_center: Any,
    *,
    blocks: Sequence[BlockCoordinateCenterBlock],
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None = None,
    scale: Any | None = None,
    config: BlockCoordinateCenterConfig | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> BlockCoordinateCenterResult:
    """Execute one ordered transactional Gauss-Seidel sweep in a single XLA call.

    Progress callbacks receive completed observations after numerical execution;
    consumers requiring a live deadline must use independent process supervision.
    """

    cfg = BlockCoordinateCenterConfig() if config is None else config
    center = tf.reshape(numeric_tensor(initial_center, tf.float64), [-1])
    if center.shape[0] is None:
        raise ValueError("initial_center must have a static dimension")
    dimension = int(center.shape[0])
    if dimension <= 0:
        raise ValueError("initial_center must be nonempty")
    scale_tf = (
        tf.ones([dimension], tf.float64)
        if scale is None
        else tf.reshape(numeric_tensor(scale, tf.float64), [-1])
    )
    if scale_tf.shape != (dimension,) or not bool(_numerical_call(_valid_scale_core, scale_tf)):
        raise ValueError("scale must be positive finite with one entry per coordinate")
    ordered_blocks = _validate_blocks(blocks, dimension)
    prospective_rows = 1 + sum(
        block.sequential_config.max_exact_evaluations + 1
        for block in ordered_blocks
    )
    if cfg.max_physical_target_rows < prospective_rows:
        raise ValueError(
            "max_physical_target_rows is below the prospective one-sweep cap "
            f"{prospective_rows}"
        )

    from bayesfilter.inference.block_controller_report import block_result
    from bayesfilter.inference.block_controller_tf import block_controller

    owner = block_controller(value_and_score_fn, batched_value_and_score_fn,
        dimension, ordered_blocks, cfg, progress=progress_callback is not None)
    computed = owner(center, scale_tf)
    return block_result(computed, center, ordered_blocks, cfg, progress_callback)


def _validate_blocks(
    blocks: Sequence[BlockCoordinateCenterBlock], dimension: int
) -> tuple[BlockCoordinateCenterBlock, ...]:
    ordered = tuple(blocks)
    if not ordered:
        raise ValueError("at least one block is required")
    if any(not isinstance(block, BlockCoordinateCenterBlock) for block in ordered):
        raise TypeError("blocks must contain BlockCoordinateCenterBlock values")
    names = tuple(block.name for block in ordered)
    if len(set(names)) != len(names):
        raise ValueError("block names must be unique")
    occupied: set[int] = set()
    for block in ordered:
        if block.stop > dimension:
            raise ValueError("block bounds exceed the full center dimension")
        coordinates = set(range(block.start, block.stop))
        if occupied.intersection(coordinates):
            raise ValueError("block bounds overlap")
        occupied.update(coordinates)
    return ordered


def _full_value_score(
    function: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    position: tf.Tensor,
    dimension: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    value, score, finite = _target_program(function, dimension, None)(position)
    if tf.executing_eagerly() and not bool(finite):
        raise ValueError("full target replay must be finite")
    return value, score


def _full_batched_value_score(
    function: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    positions: tf.Tensor,
    dimension: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    if positions.shape[0] is None:
        raise ValueError("batched full target must have a static row count")
    values, scores, finite = _target_program(function, dimension, int(positions.shape[0]))(positions)
    if tf.executing_eagerly() and not bool(finite):
        raise ValueError("batched full target rows must be finite")
    return values, scores


def _embed_scalar(
    center: tf.Tensor, block_position: Any, start: int, stop: int
) -> tf.Tensor:
    block = tf.reshape(tf.convert_to_tensor(block_position, tf.float64), [-1])
    if block.shape != (stop - start,):
        raise ValueError("block candidate has the wrong dimension")
    return _embedding_program(int(center.shape[0]), start, stop, None)(center, block)


def _embed_batch(
    center: tf.Tensor, block_positions: Any, start: int, stop: int
) -> tf.Tensor:
    block = tf.convert_to_tensor(block_positions, tf.float64)
    if block.shape.rank != 2 or block.shape[0] is None or block.shape[1] != stop - start:
        raise ValueError("batched block candidates must have static [rows, block] shape")
    return _embedding_program(int(center.shape[0]), start, stop, int(block.shape[0]))(center, block)


@lru_cache(maxsize=64)
def _numerical_program(function, specifications):
    return tf.function(function, input_signature=specifications, jit_compile=True, autograph=False)


def _numerical_call(function, *arguments):
    specifications = tuple(tf.TensorSpec(value.shape, value.dtype) for value in arguments)
    return _numerical_program(function, specifications)(*arguments)


@scoped_program_cache(maxsize=64)
def _target_program(function, dimension, rows):
    shape = [dimension] if rows is None else [rows, dimension]

    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float64)], jit_compile=True, autograph=False)
    def evaluate(position):
        values, scores = function(position)
        values = tf.reshape(tf.convert_to_tensor(values, tf.float64), [] if rows is None else [-1])
        scores = tf.convert_to_tensor(scores, tf.float64)
        if rows is None:
            scores = tf.reshape(scores, [-1])
            if scores.shape != (dimension,):
                raise ValueError("full score must have one entry per coordinate")
        elif values.shape != (rows,) or scores.shape != (rows, dimension):
            raise ValueError("batched full target must preserve row and score shapes")
        finite = tf.reduce_all(tf.math.is_finite(values)) & tf.reduce_all(tf.math.is_finite(scores))
        # An XLA assertion alone can be discarded. Retain a numerical rejection
        # signal when this callback is enclosed by another compiled locator.
        nan = tf.constant(float("nan"), tf.float64)
        return tf.where(finite, values, nan), tf.where(finite, scores, nan), finite

    return evaluate


@scoped_program_cache(maxsize=64)
def _embedding_program(dimension, start, stop, rows):
    block_shape = [stop - start] if rows is None else [rows, stop - start]

    @tf.function(input_signature=[tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec(block_shape, tf.float64)], jit_compile=True, autograph=False)
    def embed(center, block):
        if rows is None:
            return tf.concat([center[:start], block, center[stop:]], axis=0)
        return tf.concat([
            tf.broadcast_to(center[None, :start], [rows, start]), block,
            tf.broadcast_to(center[None, stop:], [rows, dimension - stop]),
        ], axis=1)

    return embed


@scoped_program_cache(maxsize=64)
def _block_target_program(function, dimension, start, stop, rows):
    block_shape = [stop - start] if rows is None else [rows, stop - start]

    @tf.function(input_signature=[tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec(block_shape, tf.float64)], jit_compile=True, autograph=False)
    def evaluate(center, block):
        full = _embedding_program(dimension, start, stop, rows)(center, block)
        values, scores, finite = _target_program(function, dimension, rows)(full)
        return values, scores[start:stop] if rows is None else scores[:, start:stop], finite

    return evaluate


def _cycle_core(centers, threshold):
    finite = tf.reduce_all(tf.math.is_finite(centers))
    if centers.shape[0] < 3:
        return tf.constant(False), tf.constant(False), finite
    moved = tf.linalg.norm(centers[-1] - centers[-2]) > threshold
    repeat = moved & tf.reduce_any(tf.linalg.norm(centers[-1] - centers[:-2], axis=1) <= threshold)
    two_step = (moved & (tf.linalg.norm(centers[-2] - centers[-3]) > threshold)
        & (tf.linalg.norm(centers[-1] - centers[-3]) <= threshold))
    return repeat, two_step, finite


def _valid_scale_core(scale):
    return tf.reduce_all(tf.math.is_finite(scale) & (scale > 0.0))


def _score_summary_core(score, scale):
    scaled = score * scale
    return tf.linalg.norm(scaled), tf.reduce_max(tf.abs(scaled))


def _block_maxima_core(score, scale, bounds):
    coordinates = tf.range(tf.shape(score)[0])[None, :]
    selected = (coordinates >= bounds[:, :1]) & (coordinates < bounds[:, 1:])
    return tf.reduce_max(tf.where(selected, tf.abs(score * scale)[None, :],
        tf.constant(float("-inf"), tf.float64)), axis=1)


def _reversal_core(current, previous, ratio):
    floor = tf.constant(_EPS64_SQRT, tf.float64) * tf.maximum(previous, 1.0)
    material = ((current - previous) > floor) & (current > ratio * previous)
    return floor, material, tf.reduce_any(material)


def _terminal_core(initial, final, scale, bounds, initial_value, final_value, require_no_worse):
    initial_maxima = _block_maxima_core(initial, scale, bounds)
    final_maxima = _block_maxima_core(final, scale, bounds)
    epsilon = tf.constant(_EPS64_SQRT, tf.float64)
    no_worse = tf.reduce_all(final_maxima <= initial_maxima + epsilon * tf.maximum(initial_maxima, 1.0))
    initial_l2, initial_max = _score_summary_core(initial, scale)
    final_l2, final_max = _score_summary_core(final, scale)
    progress = ((final_value >= initial_value)
        & (initial_l2 - final_l2 > epsilon * tf.maximum(tf.abs(initial_l2), 1.0))
        & (initial_max - final_max > epsilon * tf.maximum(tf.abs(initial_max), 1.0))
        & (no_worse | ~require_no_worse))
    return no_worse, progress


def _displacement_core(before, after):
    return tf.linalg.norm(after - before)


def _resolvable_decrease(before: float, after: float) -> bool:
    floor = _EPS64_SQRT * max(1.0, abs(float(before)))
    return bool(float(before) - float(after) > floor)


def _without_internal_geometry(value: Any) -> Any:
    """Retain private locator progress while discarding fitted geometry arrays."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_internal_geometry(child)
            for key, child in value.items()
            if str(key) not in _DISCARDED_INTERNAL_GEOMETRY_FIELDS
        }
    if isinstance(value, (tuple, list)):
        return [_without_internal_geometry(child) for child in value]
    return value


def _emit_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    stage: str,
    **fields: Any,
) -> None:
    if callback is not None:
        callback(_json_ready({"stage": stage, **fields}))


def _json_ready(value: Any) -> Any:
    if tf.is_tensor(value) or hasattr(value, "dtype"):
        return numeric_tensor(value).numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(child) for child in value]
    return value
