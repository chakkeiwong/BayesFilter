"""GPU/XLA-oriented reverse-KL training for BayesFilter NeuTra transports.

The target supplies graph-native values and scores. GradientTape is restricted
to the trainable transport; it never differentiates through the target/filter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import (
    finalize_dense_iaf_neutra_artifact_payload,
)

# The plain dense-IAF campaign API is retained in a separate compatibility
# module for existing end-to-end and PP-UKF lanes.  The current trainer above
# remains the default named-family training implementation.
from bayesfilter.inference.neutra_training_legacy import (
    NeuTraTrainingError,
    PlainDenseIAFSegmentedTrainingResult,
    PlainDenseIAFTrainingConfig,
    PlainDenseIAFTrainingResult,
    PlainDenseIAFTransport,
    restore_plain_dense_iaf_flow,
    train_plain_dense_iaf,
    train_plain_dense_iaf_infrastructure_segments,
)


NEUTRA_TRAINING_NONCLAIMS = (
    "reverse-KL trainer engineering surface only",
    "training loss is not a transport promotion criterion",
    "no HMC or sampler-validity claim",
    "no posterior-correctness claim",
    "no predictive or scientific-validity claim",
    "no default or production-readiness claim",
)

DSGE_PAPER_NEUTRA_FAMILY = "dsge_paper_dense_iaf"
PURE_PAPER_NEUTRA_FAMILY = "pure_paper_dense_iaf"
PURE_BOUNDED_NEUTRA_FAMILY = "pure_bounded_dense_iaf"
QUADRATIC_ANCHOR_INITIALIZATION_MODE = "quadratic_triangular_anchor_v1"
LEGACY_INITIALIZATION_MODE = "legacy"
SSL_LSTM_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_capacity_dense_iaf"
SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_tuned_capacity_dense_iaf"
SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_deep_capacity_dense_iaf"
SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_wide_capacity_dense_iaf"
SSL_LSTM_PURE_NEUTRA_FAMILY = "ssl_lstm_pure_dense_iaf"
COMPOSED_NEUTRA_FAMILIES = frozenset(
    (
        DSGE_PAPER_NEUTRA_FAMILY,
        PURE_PAPER_NEUTRA_FAMILY,
        PURE_BOUNDED_NEUTRA_FAMILY,
        SSL_LSTM_CAPACITY_NEUTRA_FAMILY,
        SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
        SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
        SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
        SSL_LSTM_PURE_NEUTRA_FAMILY,
    )
)
PURE_NEUTRA_FAMILIES = frozenset(
    (
        PURE_PAPER_NEUTRA_FAMILY,
        PURE_BOUNDED_NEUTRA_FAMILY,
        SSL_LSTM_PURE_NEUTRA_FAMILY,
    )
)
DSGE_PAPER_TRAINING_STEPS = 5000
DSGE_PAPER_TRAINING_BATCH_SIZE = 480
DSGE_PAPER_LR_BOUNDARIES = (999, 3999)
# Adam serializes its scalar learning rate through float32 in some TensorFlow
# versions.  Permit only the resulting sub-ppm roundoff above the configured
# rate; a materially larger value remains an invalid checkpoint.
_LEARNING_RATE_RESTORE_REL_TOL = 1.0e-6
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _resolved_paper_piecewise_boundaries(
    boundaries: Sequence[int],
) -> tuple[int, ...]:
    """Return validated zero-based boundaries, preserving the paper default."""

    values = tuple(boundaries)
    if not values:
        return DSGE_PAPER_LR_BOUNDARIES
    if any(type(value) is not int for value in values):
        raise ValueError("paper_piecewise_boundaries must contain only integers")
    if values[0] < 0 or any(
        previous >= current for previous, current in zip(values, values[1:])
    ):
        raise ValueError(
            "paper_piecewise_boundaries must be nonnegative and strictly increasing"
        )
    return values


def _paper_schedule_extension_preserves_history(
    *,
    saved_config: Mapping[str, Any],
    active_config: Mapping[str, Any],
    state_step: int,
) -> bool:
    """Return whether an explicit paper-schedule append changes no past update.

    Keras schedule boundaries are zero-based and inclusive.  A checkpoint at
    ``state_step`` has applied iterations ``0`` through ``state_step - 1``, so
    every new boundary must be at or beyond that last applied iteration.
    """

    saved = dict(saved_config)
    active = dict(active_config)
    if (
        saved.get("learning_rate_schedule") != "paper_piecewise"
        or active.get("learning_rate_schedule") != "paper_piecewise"
    ):
        return False
    saved_boundaries = _resolved_paper_piecewise_boundaries(
        saved.pop("paper_piecewise_boundaries", ())
    )
    active_boundaries = _resolved_paper_piecewise_boundaries(
        active.pop("paper_piecewise_boundaries", ())
    )
    if saved != active:
        return False
    if (
        len(active_boundaries) <= len(saved_boundaries)
        or active_boundaries[: len(saved_boundaries)] != saved_boundaries
    ):
        return False
    last_applied_iteration = state_step - 1
    return all(
        boundary >= last_applied_iteration
        for boundary in active_boundaries[len(saved_boundaries) :]
    )


@dataclass(frozen=True)
class NeuTraTrainerConfig:
    """Configuration for a trainable diagonal-affine or dense-IAF transport."""

    dimension: int
    family: str = "dense_iaf"
    hidden_layers: tuple[int, ...] = (8, 8)
    activation: str = "tanh"
    s_max: float = 1.0
    initialization_scale: float = 0.02
    initialization_seed: tuple[int, int] = (20260714, 2101)
    learning_rate: float = 1.0e-3
    learning_rate_schedule: str = "constant"
    paper_piecewise_boundaries: tuple[int, ...] = ()
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    gradient_clip_norm: float = 10.0
    gradient_clip_mode: str = "global"
    kernel_initialization: str = "legacy_zero_output"
    scale_transform: str = "bounded_tanh"
    initialization_mode: str = LEGACY_INITIALIZATION_MODE
    initial_anchor_factor: tuple[tuple[float, ...], ...] = ()
    anchor_release_steps: int = 0
    anchor_estimator_signature: str | None = None
    anchor_factor_orientation: str = "row_lower_cholesky"
    stages: int = 1
    fixed_translation: tuple[float, ...] = ()
    fixed_output_scale: tuple[float, ...] = ()
    fixed_output_factor: tuple[tuple[float, ...], ...] = ()
    initial_output_shift: tuple[float, ...] = ()
    initial_output_scale_log: tuple[float, ...] = ()
    target_parameter_names: tuple[str, ...] = ()
    target_chart: str = "unspecified"
    chart_signature: str | None = None
    target_signature: str | None = None
    target_adapter_signature: str | None = None
    jit_compile: bool = True

    def __post_init__(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError("dimension must be positive")
        if self.family not in {"affine_diag", "dense_iaf", *COMPOSED_NEUTRA_FAMILIES}:
            raise ValueError(
                "unsupported NeuTra training family"
            )
        if self.family in {"dense_iaf", *COMPOSED_NEUTRA_FAMILIES} and (
            not self.hidden_layers or any(int(width) <= 0 for width in self.hidden_layers)
        ):
            raise ValueError("dense_iaf hidden layers must be positive")
        if self.activation not in {"elu", "tanh", "relu"}:
            raise ValueError("unsupported activation")
        if not math.isfinite(self.s_max) or self.s_max <= 0.0:
            raise ValueError("s_max must be finite and positive")
        if not math.isfinite(self.initialization_scale) or self.initialization_scale < 0.0:
            raise ValueError("initialization_scale must be finite and nonnegative")
        if len(self.initialization_seed) != 2:
            raise ValueError("initialization_seed must contain two integers")
        for name in ("learning_rate", "epsilon", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("beta1", "beta2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.learning_rate_schedule not in {
            "constant",
            "paper_piecewise",
            "adaptive_constant",
        }:
            raise ValueError("unsupported learning_rate_schedule")
        boundaries = tuple(self.paper_piecewise_boundaries)
        if boundaries and self.learning_rate_schedule != "paper_piecewise":
            raise ValueError(
                "paper_piecewise_boundaries requires paper_piecewise schedule"
            )
        _resolved_paper_piecewise_boundaries(boundaries)
        if self.gradient_clip_mode not in {"global", "per_variable", "none"}:
            raise ValueError("unsupported gradient_clip_mode")
        if self.kernel_initialization not in {
            "legacy_zero_output",
            "paper_variance_scaling",
        }:
            raise ValueError("unsupported kernel_initialization")
        if self.scale_transform not in {
            "bounded_tanh",
            "dsge_bounded_tanh",
            "identity",
        }:
            raise ValueError("unsupported scale_transform")
        if self.initialization_mode not in {
            LEGACY_INITIALIZATION_MODE,
            QUADRATIC_ANCHOR_INITIALIZATION_MODE,
        }:
            raise ValueError("unsupported initialization_mode")
        anchor_release_steps = int(self.anchor_release_steps)
        if anchor_release_steps < 0:
            raise ValueError("anchor_release_steps must be nonnegative")
        object.__setattr__(self, "anchor_release_steps", anchor_release_steps)
        if int(self.stages) <= 0:
            raise ValueError("stages must be positive")
        translation = tuple(float(value) for value in self.fixed_translation)
        output_scale = tuple(float(value) for value in self.fixed_output_scale)
        factor_rows = tuple(
            tuple(float(item) for item in row) for row in self.fixed_output_factor
        )
        anchor_rows = tuple(
            tuple(float(item) for item in row) for row in self.initial_anchor_factor
        )
        initial_shift = tuple(float(value) for value in self.initial_output_shift)
        initial_scale_log = tuple(float(value) for value in self.initial_output_scale_log)
        if any(not math.isfinite(value) for value in translation):
            raise ValueError("fixed_translation must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in output_scale):
            raise ValueError("fixed_output_scale must be finite and positive")
        if output_scale and factor_rows:
            raise ValueError("fixed_output_scale and fixed_output_factor are mutually exclusive")
        if factor_rows:
            if len(factor_rows) != int(self.dimension) or any(
                len(row) != int(self.dimension) for row in factor_rows
            ):
                raise ValueError("fixed_output_factor must be square with dimension rows")
            factor_tensor = tf.constant(factor_rows, dtype=tf.float64)
            sign, logdet = tf.linalg.slogdet(factor_tensor)
            if not bool(tf.reduce_all(tf.math.is_finite(factor_tensor)).numpy()):
                raise ValueError("fixed_output_factor must be finite")
            if not bool(tf.math.is_finite(logdet).numpy()) or bool(tf.equal(sign, 0.0).numpy()):
                raise ValueError("fixed_output_factor must be nonsingular")
        if self.chart_signature is not None and len(str(self.chart_signature)) != 64:
            raise ValueError("chart_signature must be a sha256 hex digest")
        if factor_rows and self.chart_signature is None:
            raise ValueError("fixed_output_factor requires chart_signature")
        if any(not math.isfinite(value) for value in initial_shift + initial_scale_log):
            raise ValueError("initial output values must be finite")
        if any(not math.isfinite(value) for row in anchor_rows for value in row):
            raise ValueError("initial_anchor_factor must be finite")
        anchor_mode = self.initialization_mode == QUADRATIC_ANCHOR_INITIALIZATION_MODE
        if anchor_mode:
            if self.family not in PURE_NEUTRA_FAMILIES:
                raise ValueError("quadratic anchor mode is restricted to pure NeuTra")
            if len(anchor_rows) != int(self.dimension) or any(
                len(row) != int(self.dimension) for row in anchor_rows
            ):
                raise ValueError("initial_anchor_factor must be square with dimension rows")
            anchor_tensor = tf.constant(anchor_rows, dtype=tf.float64)
            strict_upper = tf.linalg.band_part(anchor_tensor, 0, -1) - tf.linalg.diag(
                tf.linalg.diag_part(anchor_tensor)
            )
            if bool(tf.reduce_any(tf.not_equal(strict_upper, 0.0)).numpy()):
                raise ValueError("initial_anchor_factor must be lower triangular")
            diagonal = tf.linalg.diag_part(anchor_tensor)
            if bool(tf.reduce_any(diagonal <= 0.0).numpy()):
                raise ValueError("initial_anchor_factor diagonal must be positive")
            if self.anchor_factor_orientation != "row_lower_cholesky":
                raise ValueError("unsupported anchor_factor_orientation")
            signature = self.anchor_estimator_signature
            if signature is None or not _SHA256_HEX_RE.fullmatch(str(signature)):
                raise ValueError("quadratic anchor mode requires anchor_estimator_signature")
        elif anchor_rows or self.anchor_release_steps or self.anchor_estimator_signature is not None:
            raise ValueError("anchor fields require quadratic anchor initialization_mode")
        names = tuple(str(value) for value in self.target_parameter_names)
        if len(set(names)) != len(names):
            raise ValueError("target_parameter_names must be unique")
        if self.family in COMPOSED_NEUTRA_FAMILIES:
            if self.family in {
                DSGE_PAPER_NEUTRA_FAMILY,
                PURE_PAPER_NEUTRA_FAMILY,
                PURE_BOUNDED_NEUTRA_FAMILY,
            }:
                hidden_layers = (int(self.dimension), int(self.dimension))
            elif self.family == SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY:
                hidden_layers = (32, 32, 32)
            elif self.family == SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY:
                hidden_layers = (64, 64)
            elif self.family == SSL_LSTM_PURE_NEUTRA_FAMILY:
                hidden_layers = (32, 32)
            else:
                hidden_layers = (32, 32)
            required = {
                "hidden_layers": hidden_layers,
                "activation": "elu",
                "s_max": (
                    1.0
                    if self.family not in PURE_NEUTRA_FAMILIES
                    else float(self.s_max)
                ),
                "epsilon": 1.0e-7,
                "beta1": 0.9,
                "beta2": 0.999,
                "stages": 3,
            }
            if self.family in {
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
            }:
                required["gradient_clip_mode"] = "per_variable"
                required["learning_rate_schedule"] = "adaptive_constant"
            elif self.family == PURE_PAPER_NEUTRA_FAMILY:
                required.update(
                    {
                        "initialization_scale": 0.02,
                        "learning_rate": 0.01,
                        "learning_rate_schedule": "paper_piecewise",
                        "epsilon": 1.0e-8,
                        "gradient_clip_norm": 10.0,
                        "gradient_clip_mode": "none",
                        "kernel_initialization": "paper_variance_scaling",
                        "target_chart": "direct_physical",
                    }
                )
                if not anchor_mode:
                    required["scale_transform"] = "identity"
            elif self.family == PURE_BOUNDED_NEUTRA_FAMILY:
                required.update(
                    {
                        "initialization_scale": 0.02,
                        "learning_rate": 0.01,
                        "learning_rate_schedule": "paper_piecewise",
                        "epsilon": 1.0e-8,
                        "gradient_clip_norm": 10.0,
                        "gradient_clip_mode": "none",
                        "kernel_initialization": "paper_variance_scaling",
                        "target_chart": "direct_physical",
                    }
                )
                if not anchor_mode:
                    required["scale_transform"] = "bounded_tanh"
            elif self.family == SSL_LSTM_PURE_NEUTRA_FAMILY:
                if self.learning_rate_schedule == "paper_piecewise":
                    required.update(
                        {
                            "initialization_scale": 0.02,
                            "learning_rate": 0.01,
                            "epsilon": 1.0e-8,
                            "gradient_clip_mode": "none",
                            "kernel_initialization": "paper_variance_scaling",
                            "scale_transform": "identity",
                            "target_chart": "direct_physical",
                        }
                    )
                else:
                    required["learning_rate_schedule"] = "adaptive_constant"
                    required["gradient_clip_mode"] = "per_variable"
                    required["target_chart"] = "direct_physical"
            else:
                required.update(
                    {
                        "initialization_scale": 0.02,
                        "learning_rate": 0.01,
                        "learning_rate_schedule": "paper_piecewise",
                        "gradient_clip_norm": 10.0,
                    }
                )
            actual = {
                "hidden_layers": tuple(self.hidden_layers),
                "activation": self.activation,
                "s_max": float(self.s_max),
                "initialization_scale": float(self.initialization_scale),
                "learning_rate": float(self.learning_rate),
                "learning_rate_schedule": self.learning_rate_schedule,
                "epsilon": float(self.epsilon),
                "beta1": float(self.beta1),
                "beta2": float(self.beta2),
                "gradient_clip_norm": float(self.gradient_clip_norm),
                "gradient_clip_mode": self.gradient_clip_mode,
                "kernel_initialization": self.kernel_initialization,
                "scale_transform": self.scale_transform,
                "stages": int(self.stages),
                "target_chart": self.target_chart,
            }
            mismatches = [key for key, value in required.items() if actual[key] != value]
            if mismatches:
                raise ValueError(
                    f"{self.family} preset mismatch: " + ", ".join(mismatches)
                )
            if self.family in {
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
            } or (
                self.family in PURE_NEUTRA_FAMILIES
                and self.learning_rate_schedule == "adaptive_constant"
            ):
                if not 1.0e-4 <= float(self.learning_rate) <= 2.0e-3:
                    raise ValueError("tuned capacity learning_rate outside search contract")
                if float(self.initialization_scale) not in {0.005, 0.01, 0.02}:
                    raise ValueError(
                        "tuned capacity initialization_scale outside search contract"
                    )
                if float(self.gradient_clip_norm) not in {5.0, 10.0}:
                    raise ValueError(
                        "tuned capacity gradient_clip_norm outside search contract"
                    )
            if len(translation) != int(self.dimension):
                if self.family not in PURE_NEUTRA_FAMILIES:
                    raise ValueError(f"{self.family} requires fixed_translation")
            if output_scale and len(output_scale) != int(self.dimension):
                raise ValueError(f"{self.family} fixed_output_scale length mismatch")
            if len(names) != int(self.dimension):
                raise ValueError(f"{self.family} requires target_parameter_names")
            for field_name in ("target_signature", "target_adapter_signature"):
                value = getattr(self, field_name)
                if value is None or len(str(value)) != 64:
                    raise ValueError(f"{self.family} requires {field_name}")
            if self.family in PURE_NEUTRA_FAMILIES:
                if translation or output_scale or factor_rows:
                    raise ValueError("pure NeuTra forbids fixed affine chart fields")
                if len(initial_shift) != int(self.dimension) or len(initial_scale_log) != int(self.dimension):
                    raise ValueError("pure NeuTra requires initial output shift and log scale")
                if (
                    self.scale_transform in {"bounded_tanh", "dsge_bounded_tanh"}
                    and not float(self.s_max) > max(abs(value) for value in initial_scale_log)
                ):
                    raise ValueError("pure NeuTra s_max must exceed initial log-scale magnitude")
                if anchor_mode and self.scale_transform == "bounded_tanh":
                    raise ValueError(
                        "quadratic anchor mode requires paper_direct_log or dsge_bounded_tanh"
                    )
        else:
            if self.learning_rate_schedule != "constant":
                raise ValueError(
                    "paper_piecewise is reserved for named composed IAF presets"
                )
            if int(self.stages) != 1 or translation or output_scale or factor_rows:
                raise ValueError(
                    "stages, fixed_translation, fixed_output_scale, and "
                    "fixed_output_factor are reserved "
                    "for composed IAF"
                )

    def manifest_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["hidden_layers"] = list(self.hidden_layers)
        payload["initialization_seed"] = list(self.initialization_seed)
        payload["fixed_translation"] = list(self.fixed_translation)
        payload["fixed_output_scale"] = list(self.fixed_output_scale)
        payload["fixed_output_factor"] = [list(row) for row in self.fixed_output_factor]
        payload["initial_output_shift"] = list(self.initial_output_shift)
        payload["initial_output_scale_log"] = list(self.initial_output_scale_log)
        payload["initial_anchor_factor"] = [list(row) for row in self.initial_anchor_factor]
        payload["target_parameter_names"] = list(self.target_parameter_names)
        if self.paper_piecewise_boundaries:
            payload["paper_piecewise_boundaries"] = list(
                self.paper_piecewise_boundaries
            )
        else:
            payload.pop("paper_piecewise_boundaries")
        if self.initialization_mode == LEGACY_INITIALIZATION_MODE:
            # Keep the serialized contract of pre-anchor trainer states stable.
            for key in (
                "initialization_mode",
                "initial_anchor_factor",
                "anchor_release_steps",
                "anchor_estimator_signature",
                "anchor_factor_orientation",
            ):
                payload.pop(key, None)
        if self.kernel_initialization == "legacy_zero_output":
            payload.pop("kernel_initialization")
        # The bounded pure family carries the transform explicitly so a
        # serialized artifact cannot silently fall back to the loader default.
        if self.scale_transform == "bounded_tanh" and self.family != PURE_BOUNDED_NEUTRA_FAMILY:
            payload.pop("scale_transform")
        payload["schema"] = "bayesfilter.neutra.trainer_config.v1"
        return payload


def dsge_paper_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the frozen Rotemberg/SGU plain-NeuTra procedure preset."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=DSGE_PAPER_NEUTRA_FAMILY,
        hidden_layers=(int(dimension), int(dimension)),
        activation="elu",
        s_max=1.0,
        initialization_scale=0.02,
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=0.01,
        learning_rate_schedule="paper_piecewise",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=10.0,
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the three-stage 32x32 SSL-LSTM capacity-repair preset."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(32, 32),
        activation="elu",
        s_max=1.0,
        initialization_scale=0.02,
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=0.01,
        learning_rate_schedule="paper_piecewise",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=10.0,
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_tuned_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    learning_rate: float,
    initialization_scale: float,
    gradient_clip_norm: float,
    fixed_output_scale: Sequence[float] = (),
    fixed_output_factor: Sequence[Sequence[float]] = (),
    target_chart: str = "identity",
    chart_signature: str | None = None,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the bounded `(32,32)` SSL-LSTM tuning preset."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(32, 32),
        activation="elu",
        s_max=1.0,
        initialization_scale=float(initialization_scale),
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=float(learning_rate),
        learning_rate_schedule="adaptive_constant",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=float(gradient_clip_norm),
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        fixed_output_scale=tuple(float(value) for value in fixed_output_scale),
        fixed_output_factor=tuple(
            tuple(float(item) for item in row) for row in fixed_output_factor
        ),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart=str(target_chart),
        chart_signature=None if chart_signature is None else str(chart_signature),
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_deep_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    learning_rate: float,
    initialization_scale: float,
    gradient_clip_norm: float,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the explicitly labeled three-hidden-layer SSL-LSTM diagnostic."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(32, 32, 32),
        activation="elu",
        s_max=1.0,
        initialization_scale=float(initialization_scale),
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=float(learning_rate),
        learning_rate_schedule="adaptive_constant",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=float(gradient_clip_norm),
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_wide_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    learning_rate: float,
    initialization_scale: float,
    gradient_clip_norm: float,
    fixed_output_scale: Sequence[float] = (),
    fixed_output_factor: Sequence[Sequence[float]] = (),
    target_chart: str = "identity",
    chart_signature: str | None = None,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the explicitly labeled two-hidden-layer 64x64 diagnostic."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(64, 64),
        activation="elu",
        s_max=1.0,
        initialization_scale=float(initialization_scale),
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=float(learning_rate),
        learning_rate_schedule="adaptive_constant",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=float(gradient_clip_norm),
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        fixed_output_scale=tuple(float(value) for value in fixed_output_scale),
        fixed_output_factor=tuple(
            tuple(float(item) for item in row) for row in fixed_output_factor
        ),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart=str(target_chart),
        chart_signature=None if chart_signature is None else str(chart_signature),
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_pure_neutra_config(
    *,
    dimension: int,
    initial_output_shift: Sequence[float],
    initial_output_scale_log: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    s_max: float = 5.0,
    learning_rate: float = 1.0e-3,
    initialization_scale: float = 0.02,
    gradient_clip_norm: float = 10.0,
    learning_rate_schedule: str = "adaptive_constant",
    paper_piecewise_boundaries: Sequence[int] = (),
    gradient_clip_mode: str = "per_variable",
    kernel_initialization: str = "legacy_zero_output",
    scale_transform: str = "bounded_tanh",
    epsilon: float = 1.0e-7,
    initialization_seed: tuple[int, int] = (20260812, 9901),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the pure three-stage IAF preset with no frozen affine component."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_PURE_NEUTRA_FAMILY,
        hidden_layers=(32, 32),
        activation="elu",
        s_max=float(s_max),
        initialization_scale=float(initialization_scale),
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=float(learning_rate),
        learning_rate_schedule=str(learning_rate_schedule),
        paper_piecewise_boundaries=tuple(paper_piecewise_boundaries),
        beta1=0.9,
        beta2=0.999,
        epsilon=float(epsilon),
        gradient_clip_norm=float(gradient_clip_norm),
        gradient_clip_mode=str(gradient_clip_mode),
        kernel_initialization=str(kernel_initialization),
        scale_transform=str(scale_transform),
        stages=3,
        initial_output_shift=tuple(float(value) for value in initial_output_shift),
        initial_output_scale_log=tuple(float(value) for value in initial_output_scale_log),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="direct_physical",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def pure_paper_neutra_config(
    *,
    dimension: int,
    initial_output_shift: Sequence[float],
    initial_output_scale_log: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    initialization_seed: tuple[int, int] = (20260820, 1903),
    initialization_mode: str = LEGACY_INITIALIZATION_MODE,
    initial_anchor_factor: Sequence[Sequence[float]] = (),
    anchor_release_steps: int = 0,
    anchor_estimator_signature: str | None = None,
    anchor_factor_orientation: str = "row_lower_cholesky",
    scale_transform: str = "identity",
    s_max: float = 1.0,
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the no-affine, dimension-width NeuTra paper recipe.

    The standard-normal base maps directly into the target coordinates through
    three trainable dense IAF stages and two reverse permutations. Initial
    shift and log scale seed the final trainable MADE bias; they are not a
    fixed outer chart.
    """

    active_dimension = int(dimension)
    return NeuTraTrainerConfig(
        dimension=active_dimension,
        family=PURE_PAPER_NEUTRA_FAMILY,
        hidden_layers=(active_dimension, active_dimension),
        activation="elu",
        s_max=float(s_max),
        initialization_scale=0.02,
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=0.01,
        learning_rate_schedule="paper_piecewise",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        gradient_clip_norm=10.0,
        gradient_clip_mode="none",
        kernel_initialization="paper_variance_scaling",
        scale_transform=str(scale_transform),
        initialization_mode=str(initialization_mode),
        initial_anchor_factor=tuple(
            tuple(float(item) for item in row) for row in initial_anchor_factor
        ),
        anchor_release_steps=int(anchor_release_steps),
        anchor_estimator_signature=(
            None if anchor_estimator_signature is None else str(anchor_estimator_signature)
        ),
        anchor_factor_orientation=str(anchor_factor_orientation),
        stages=3,
        initial_output_shift=tuple(float(value) for value in initial_output_shift),
        initial_output_scale_log=tuple(
            float(value) for value in initial_output_scale_log
        ),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="direct_physical",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def quadratic_anchor_neutra_config(
    *,
    dimension: int,
    initial_output_shift: Sequence[float],
    initial_output_scale_log: Sequence[float],
    initial_anchor_factor: Sequence[Sequence[float]],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    anchor_estimator_signature: str,
    anchor_release_steps: int = 0,
    scale_transform: str = "identity",
    s_max: float = 1.0,
    initialization_seed: tuple[int, int] = (20260824, 2601),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Build the named pure IAF quadratic-anchor initialization arm.

    The factor and center seed trainable conditioner variables; they are not a
    fixed outer affine chart.  ``scale_transform`` is deliberately restricted
    by the underlying config to the direct-log or exact dsge bounded contract.
    """

    return pure_paper_neutra_config(
        dimension=dimension,
        initial_output_shift=initial_output_shift,
        initial_output_scale_log=initial_output_scale_log,
        target_parameter_names=target_parameter_names,
        target_signature=target_signature,
        target_adapter_signature=target_adapter_signature,
        initialization_seed=initialization_seed,
        initialization_mode=QUADRATIC_ANCHOR_INITIALIZATION_MODE,
        initial_anchor_factor=initial_anchor_factor,
        anchor_release_steps=anchor_release_steps,
        anchor_estimator_signature=anchor_estimator_signature,
        anchor_factor_orientation="row_lower_cholesky",
        scale_transform=scale_transform,
        s_max=s_max,
        jit_compile=jit_compile,
    )


def pure_bounded_neutra_config(
    *,
    dimension: int,
    initial_output_shift: Sequence[float],
    initial_output_scale_log: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    s_max: float,
    initialization_seed: tuple[int, int] = (20260824, 2401),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the reviewed pure `(d,d)` IAF with a bounded scale output.

    This is deliberately a distinct method from ``pure_paper_neutra_config``:
    it keeps the paper topology and schedule but uses
    ``s_max * tanh(raw / s_max)`` for every dense IAF scale.
    """

    active_dimension = int(dimension)
    return NeuTraTrainerConfig(
        dimension=active_dimension,
        family=PURE_BOUNDED_NEUTRA_FAMILY,
        hidden_layers=(active_dimension, active_dimension),
        activation="elu",
        s_max=float(s_max),
        initialization_scale=0.02,
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=0.01,
        learning_rate_schedule="paper_piecewise",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        gradient_clip_norm=10.0,
        gradient_clip_mode="none",
        kernel_initialization="paper_variance_scaling",
        scale_transform="bounded_tanh",
        stages=3,
        initial_output_shift=tuple(float(value) for value in initial_output_shift),
        initial_output_scale_log=tuple(
            float(value) for value in initial_output_scale_log
        ),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="direct_physical",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def pure_bounded_neutra_config(
    *,
    dimension: int,
    initial_output_shift: Sequence[float],
    initial_output_scale_log: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    s_max: float,
    initialization_seed: tuple[int, int] = (20260824, 2401),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the reviewed pure `(d,d)` IAF with a bounded scale output.

    This is deliberately a distinct method from ``pure_paper_neutra_config``:
    it keeps the paper topology and schedule but uses
    ``s_max * tanh(raw / s_max)`` for every dense IAF scale.
    """

    active_dimension = int(dimension)
    return NeuTraTrainerConfig(
        dimension=active_dimension,
        family=PURE_BOUNDED_NEUTRA_FAMILY,
        hidden_layers=(active_dimension, active_dimension),
        activation="elu",
        s_max=float(s_max),
        initialization_scale=0.02,
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=0.01,
        learning_rate_schedule="paper_piecewise",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        gradient_clip_norm=10.0,
        gradient_clip_mode="none",
        kernel_initialization="paper_variance_scaling",
        scale_transform="bounded_tanh",
        stages=3,
        initial_output_shift=tuple(float(value) for value in initial_output_shift),
        initial_output_scale_log=tuple(
            float(value) for value in initial_output_scale_log
        ),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="direct_physical",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def dsge_paper_learning_rate(
    learning_rate: float = 0.01,
    *,
    boundaries: Sequence[int] = (),
) -> tf.keras.optimizers.schedules.PiecewiseConstantDecay:
    """Build the zero-based DSGE schedule, optionally at explicit boundaries."""

    rate = float(learning_rate)
    active_boundaries = _resolved_paper_piecewise_boundaries(boundaries)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=list(active_boundaries),
        values=[rate * (0.1**index) for index in range(len(active_boundaries) + 1)],
    )


@dataclass(frozen=True)
class NeuTraTrainStep:
    """Tensor outputs from one reverse-KL gradient or update evaluation."""

    loss: tf.Tensor
    surrogate: tf.Tensor
    target_value_mean: tf.Tensor
    logdet_mean: tf.Tensor
    gradient_norm: tf.Tensor
    clipped_gradient_norm: tf.Tensor
    clipping_applied: tf.Tensor
    step: tf.Tensor


@dataclass(frozen=True)
class NeuTraChartPreflight:
    """Diagnostics for a caller-declared affine NeuTra training chart.

    The chart is model-owned: BayesFilter checks its algebra and numerical
    contract but does not infer scientifically meaningful parameter scales.
    """

    chart_name: str
    dimension: int
    roundtrip_max_abs: float
    value_max_abs_residual: float | None
    score_max_abs_residual: float | None
    logdet: float
    finite: bool
    passed: bool


def preflight_neutra_affine_chart(
    *,
    chart_name: str,
    center: Any,
    factor: Any,
    latent: Any,
    physical_target: Any | None = None,
    transformed_target: Any | None = None,
    strict: bool = True,
    value_score_tolerance: float = 1.0e-10,
) -> NeuTraChartPreflight:
    """Validate a caller-supplied ``theta=center+z@factor.T`` chart.

    This check validates finite/nonsingular chart algebra and, when the target
    exposes a value/score call, the exact affine score chain rule. It does not
    decide whether a model-specific scale is scientifically adequate.
    """

    center_t = tf.convert_to_tensor(center, dtype=tf.float64)
    factor_t = tf.convert_to_tensor(factor, dtype=tf.float64)
    latent_t = tf.convert_to_tensor(latent, dtype=tf.float64)
    if center_t.shape.rank != 1 or factor_t.shape.rank != 2:
        raise ValueError("center must be rank 1 and factor rank 2")
    dimension = center_t.shape[0]
    if dimension is None or factor_t.shape != (dimension, dimension):
        raise ValueError("affine chart dimensions do not match")
    if latent_t.shape.rank != 2 or latent_t.shape[-1] != dimension:
        raise ValueError("latent must have shape [batch, dimension]")
    inputs_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    (
                        tf.reshape(center_t, [-1]),
                        tf.reshape(factor_t, [-1]),
                        tf.reshape(latent_t, [-1]),
                    ),
                    axis=0,
                )
            )
        ).numpy()
    )
    sign, slogdet_t = tf.linalg.slogdet(factor_t)
    lower = tf.linalg.band_part(factor_t, -1, 0)
    upper = tf.linalg.band_part(factor_t, 0, -1)
    triangular = bool(tf.reduce_all(tf.equal(factor_t, lower)).numpy()) or bool(
        tf.reduce_all(tf.equal(factor_t, upper)).numpy()
    )
    logdet_t = (
        tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(factor_t))))
        if triangular
        else slogdet_t
    )
    logdet_finite = bool(tf.math.is_finite(logdet_t).numpy())
    nonsingular = (
        inputs_finite
        and logdet_finite
        and bool(tf.not_equal(sign, 0.0).numpy())
    )
    theta = center_t[tf.newaxis, :] + tf.matmul(latent_t, factor_t, transpose_b=True)
    if nonsingular:
        recovered = tf.transpose(
            tf.linalg.solve(factor_t, tf.transpose(theta - center_t[tf.newaxis, :]))
        )
        roundtrip = tf.reduce_max(tf.abs(recovered - latent_t))
    else:
        roundtrip = tf.constant(math.inf, tf.float64)
    finite = inputs_finite and logdet_finite
    finite = finite and bool(tf.reduce_all(tf.math.is_finite(theta)).numpy())
    value_residual = None
    score_residual = None
    if nonsingular and physical_target is not None and transformed_target is not None:
        physical_call = getattr(physical_target, "batch_value_and_score", None)
        transformed_call = getattr(transformed_target, "batch_value_and_score", None)
        if not callable(physical_call) or not callable(transformed_call):
            raise ValueError(
                "physical_target and transformed_target must expose "
                "batch_value_and_score"
            )
        physical_value, physical_score = physical_call(theta)
        transformed_value, transformed_score = transformed_call(latent_t)
        expected_value = physical_value + logdet_t
        expected_score = tf.matmul(physical_score, factor_t)
        value_residual = float(
            tf.reduce_max(tf.abs(transformed_value - expected_value)).numpy()
        )
        score_residual = float(
            tf.reduce_max(tf.abs(transformed_score - expected_score)).numpy()
        )
        finite = finite and bool(
            tf.reduce_all(
                tf.math.is_finite(
                    tf.concat(
                        (
                            tf.reshape(physical_value, [-1]),
                            tf.reshape(physical_score, [-1]),
                            tf.reshape(transformed_value, [-1]),
                            tf.reshape(transformed_score, [-1]),
                        ),
                        axis=0,
                    )
                )
            ).numpy()
        )
    value_score_tolerance = float(value_score_tolerance)
    if not math.isfinite(value_score_tolerance) or value_score_tolerance < 0.0:
        raise ValueError("value_score_tolerance must be finite and nonnegative")
    passed = (
        nonsingular
        and finite
        and bool(float(roundtrip.numpy()) <= value_score_tolerance)
    )
    if value_residual is not None:
        passed = passed and value_residual <= value_score_tolerance
    if score_residual is not None:
        passed = passed and score_residual <= value_score_tolerance
    report = NeuTraChartPreflight(
        chart_name=str(chart_name),
        dimension=int(dimension),
        roundtrip_max_abs=float(roundtrip.numpy()),
        value_max_abs_residual=value_residual,
        score_max_abs_residual=score_residual,
        logdet=float(logdet_t.numpy()),
        finite=finite,
        passed=passed,
    )
    if strict and not passed:
        raise ValueError("NeuTra affine chart preflight failed")
    return report


@dataclass(frozen=True)
class NeuTraValidation:
    """Non-updating reverse-KL diagnostics on an independent base batch."""

    per_sample_loss: tf.Tensor
    target_value: tf.Tensor
    theta: tf.Tensor
    logdet: tf.Tensor
    scale_log: tf.Tensor
    scale_logits: tf.Tensor
    hidden_preactivations: tf.Tensor


class _TrainableTransport:
    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        raise NotImplementedError

    @property
    def variable_keys(self) -> tuple[str, ...]:
        raise NotImplementedError

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Return raw scale logits and hidden preactivations for diagnostics."""

        scale_log = self.scale_log(z)
        batch = tf.shape(z)[0]
        return scale_log, tf.zeros((batch, 0, 0), dtype=z.dtype)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        raise NotImplementedError


class _TrainableAffineDiagonal(_TrainableTransport):
    def __init__(self, config: NeuTraTrainerConfig) -> None:
        self.dimension = int(config.dimension)
        self.shift = tf.Variable(
            tf.zeros((self.dimension,), dtype=tf.float64),
            name="neutra_affine_shift",
        )
        self.raw_scale = tf.Variable(
            tf.zeros((self.dimension,), dtype=tf.float64),
            name="neutra_affine_raw_scale",
        )

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return (self.shift, self.raw_scale)

    @property
    def variable_keys(self) -> tuple[str, ...]:
        return ("shift", "raw_scale")

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale = tf.exp(self.raw_scale)
        theta = self.shift + z * scale
        logdet = tf.zeros(tf.shape(z)[:-1], dtype=z.dtype) + tf.reduce_sum(
            self.raw_scale
        )
        return theta, logdet

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z) + self.raw_scale

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self.scale_log(z), tf.zeros((tf.shape(z)[0], 0, 0), dtype=z.dtype)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        return {
            "component_id": component_id,
            "kind": "affine",
            "dim": self.dimension,
            "dtype": "float64",
            "offset": _tensor_values(self.shift),
            "scale": _tensor_values(tf.exp(self.raw_scale)),
        }


class _TrainableDenseIAF(_TrainableTransport):
    def __init__(self, config: NeuTraTrainerConfig, *, stage_index: int = 0) -> None:
        self.dimension = int(config.dimension)
        self.stage_index = int(stage_index)
        self.family = str(config.family)
        self.hidden_layers = tuple(int(width) for width in config.hidden_layers)
        self.activation = str(config.activation)
        self.s_max = float(config.s_max)
        self.scale_transform = str(config.scale_transform)
        self.initialization_mode = str(config.initialization_mode)
        self.masks = _dense_iaf_masks(self.dimension, self.hidden_layers)
        layer_sizes = (self.dimension, *self.hidden_layers, 2 * self.dimension)
        weights = []
        biases = []
        seed = tf.random.experimental.stateless_fold_in(
            tf.constant(config.initialization_seed, dtype=tf.int32),
            self.stage_index,
        )
        for index, (input_width, output_width) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layer_seed = tf.random.experimental.stateless_fold_in(seed, index)
            if config.kernel_initialization == "paper_variance_scaling":
                # The NeuTra reference code uses TF's variance-scaling
                # initializer with scale 0.02 for every MADE kernel. TensorFlow
                # divides by the truncated-normal standard deviation so the
                # realized variance remains scale / fan_in.
                stddev = math.sqrt(float(config.initialization_scale) / input_width)
                stddev /= 0.87962566103423978
                initial_weight = tf.random.stateless_truncated_normal(
                    (input_width, output_width),
                    seed=layer_seed,
                    stddev=tf.cast(stddev, tf.float64),
                    dtype=tf.float64,
                )
            else:
                scale = 0.0 if index == len(layer_sizes) - 2 else float(
                    config.initialization_scale
                )
                initial_weight = tf.random.stateless_normal(
                    (input_width, output_width),
                    seed=layer_seed,
                    dtype=tf.float64,
                ) * tf.cast(scale, tf.float64)
            weights.append(
                tf.Variable(
                    initial_weight,
                    name=f"neutra_dense_iaf_{self.stage_index}_weight_{index}",
                )
            )
            biases.append(
                tf.Variable(
                    tf.zeros((output_width,), dtype=tf.float64),
                    name=f"neutra_dense_iaf_{self.stage_index}_bias_{index}",
                )
            )
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        self.anchor_shift_weight: tf.Variable | None = None
        self.anchor_shift_bias: tf.Variable | None = None
        self.anchor_scale_raw: tf.Variable | None = None
        if self.initialization_mode == QUADRATIC_ANCHOR_INITIALIZATION_MODE:
            # Embed the local lower-Cholesky chart in the first IAF
            # conditioner.  Zeroing the nonlinear output kernel makes the
            # declared initial map exact; all anchor variables remain
            # trainable and are released after the mechanics warm-up.
            self.weights[-1].assign(tf.zeros_like(self.weights[-1]))
            factor = tf.constant(config.initial_anchor_factor, dtype=tf.float64)
            strict_lower = tf.linalg.band_part(factor, -1, 0) - tf.linalg.diag(
                tf.linalg.diag_part(factor)
            )
            if self.stage_index == 0:
                anchor_weight = strict_lower
                anchor_bias = tf.constant(config.initial_output_shift, tf.float64)
                scale_log = tf.constant(config.initial_output_scale_log, tf.float64)
                if self.scale_transform == "dsge_bounded_tanh":
                    anchor_scale = tf.math.atanh(scale_log / self.s_max)
                else:
                    anchor_scale = scale_log
            else:
                anchor_weight = tf.zeros((self.dimension, self.dimension), tf.float64)
                anchor_bias = tf.zeros((self.dimension,), tf.float64)
                anchor_scale = tf.zeros((self.dimension,), tf.float64)
            self.anchor_shift_weight = tf.Variable(
                anchor_weight,
                name=f"neutra_dense_iaf_{self.stage_index}_anchor_shift_weight",
            )
            self.anchor_shift_bias = tf.Variable(
                anchor_bias,
                name=f"neutra_dense_iaf_{self.stage_index}_anchor_shift_bias",
            )
            self.anchor_scale_raw = tf.Variable(
                anchor_scale,
                name=f"neutra_dense_iaf_{self.stage_index}_anchor_scale_raw",
            )
        elif (
            self.stage_index == int(config.stages) - 1
            and config.family in PURE_NEUTRA_FAMILIES
        ):
            scale_log = tf.constant(config.initial_output_scale_log, tf.float64)
            shift = tf.constant(config.initial_output_shift, tf.float64)
            if config.scale_transform == "identity":
                raw_scale = scale_log
            else:
                scaled = tf.clip_by_value(scale_log / self.s_max, -0.999999, 0.999999)
                raw_scale = self.s_max * tf.atanh(scaled)
            self.biases[-1].assign(tf.concat((raw_scale, shift), axis=0))

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        output = []
        for weight, bias in zip(self.weights, self.biases):
            output.extend((weight, bias))
        if self.anchor_shift_weight is not None:
            output.extend(
                (
                    self.anchor_shift_weight,
                    self.anchor_shift_bias,
                    self.anchor_scale_raw,
                )
            )
        return tuple(output)

    @property
    def variable_keys(self) -> tuple[str, ...]:
        output = []
        for index in range(len(self.weights)):
            output.extend((f"weight[{index}]", f"bias[{index}]"))
        if self.anchor_shift_weight is not None:
            output.extend(
                (
                    "anchor_shift_weight",
                    "anchor_shift_bias",
                    "anchor_scale_raw",
                )
            )
        return tuple(output)

    @property
    def anchor_variable_indices(self) -> tuple[int, ...]:
        if self.anchor_shift_weight is None:
            return ()
        first = 2 * len(self.weights)
        return (first, first + 1, first + 2)

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale_log, shift = self._network(z)
        theta = z * tf.exp(scale_log) + shift
        return theta, tf.reduce_sum(scale_log, axis=-1)

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        scale_log, _ = self._network(z)
        return scale_log

    def _network(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale_log, shift, _, _ = self._network_with_diagnostics(z)
        return scale_log, shift

    def _network_with_diagnostics(
        self, z: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        h = z
        preactivations = []
        for weight, bias, mask in zip(
            self.weights[:-1], self.biases[:-1], self.masks[:-1]
        ):
            h = tf.matmul(h, weight * mask) + bias
            preactivations.append(h)
            h = _activation(h, self.activation)
        raw = tf.matmul(h, self.weights[-1] * self.masks[-1]) + self.biases[-1]
        scale_logits = raw[..., : self.dimension]
        shift = raw[..., self.dimension :]
        if self.anchor_shift_weight is not None:
            anchor_weight = tf.linalg.band_part(
                self.anchor_shift_weight, -1, 0
            ) - tf.linalg.diag(tf.linalg.diag_part(self.anchor_shift_weight))
            scale_logits = scale_logits + self.anchor_scale_raw
            shift = shift + tf.matmul(z, anchor_weight, transpose_b=True)
            shift = shift + self.anchor_shift_bias
        if self.scale_transform == "identity":
            scale_log = scale_logits
        elif self.scale_transform == "dsge_bounded_tanh":
            # Exact dsge_hmc convention; do not replace with raw/s_max.
            scale_log = self.s_max * tf.math.tanh(scale_logits)
        else:
            # Historical normalized bounded family, retained for old arms.
            scale_log = self.s_max * tf.math.tanh(scale_logits / self.s_max)
        max_width = max(self.hidden_layers, default=0)
        padded = [
            tf.pad(values, [[0, 0], [0, max_width - int(values.shape[-1])]])
            for values in preactivations
        ]
        hidden = (
            tf.stack(padded, axis=1)
            if padded
            else tf.zeros((tf.shape(z)[0], 0, max_width), dtype=z.dtype)
        )
        return scale_log, shift, scale_logits, hidden

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        _, _, scale_logits, hidden = self._network_with_diagnostics(z)
        return scale_logits, hidden

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        payload = {
            "component_id": component_id,
            "kind": "dense_autoregressive_iaf",
            "dim": self.dimension,
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
            "s_max": self.s_max,
            "masks_policy": "legacy_degree_masks_v1",
            "dtype": "float64",
            "weights": [_tensor_values(weight) for weight in self.weights],
            "biases": [_tensor_values(bias) for bias in self.biases],
        }
        if (
            self.scale_transform != "bounded_tanh"
            or self.family == PURE_BOUNDED_NEUTRA_FAMILY
        ):
            payload["scale_transform"] = self.scale_transform
        if self.initialization_mode == QUADRATIC_ANCHOR_INITIALIZATION_MODE:
            payload.update(
                {
                    "initialization_mode": self.initialization_mode,
                    "anchor_shift_weight": _tensor_values(self.anchor_shift_weight),
                    "anchor_shift_bias": _tensor_values(self.anchor_shift_bias),
                    "anchor_scale_raw": _tensor_values(self.anchor_scale_raw),
                    "anchor_factor_orientation": "row_lower_cholesky",
                }
            )
        return payload


class _FixedMixingReverse(_TrainableTransport):
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.matrix = tf.reverse(tf.eye(self.dimension, dtype=tf.float64), axis=(0,))

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return ()

    @property
    def variable_keys(self) -> tuple[str, ...]:
        return ()

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.matmul(z, self.matrix), tf.zeros(tf.shape(z)[:-1], z.dtype)

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        return {
            "component_id": component_id,
            "kind": "mixing_linear",
            "dim": self.dimension,
            "dtype": "float64",
            "matrix": _tensor_values(self.matrix),
        }


class _FixedTranslation(_TrainableTransport):
    def __init__(
        self,
        values: Sequence[float],
        scale: Sequence[float] = (),
        factor: Sequence[Sequence[float]] = (),
    ) -> None:
        self.offset = tf.constant(tuple(float(value) for value in values), tf.float64)
        self.dimension = int(self.offset.shape[0])
        scale_values = tuple(float(value) for value in scale)
        factor_rows = tuple(tuple(float(item) for item in row) for row in factor)
        if scale_values and factor_rows:
            raise ValueError("fixed output scale and factor are mutually exclusive")
        self.output_scale = tf.constant(
            scale_values if scale_values else (1.0,) * self.dimension, tf.float64
        )
        self.output_factor = (
            None if not factor_rows else tf.constant(factor_rows, tf.float64)
        )
        if self.output_factor is not None:
            lower = tf.linalg.band_part(self.output_factor, -1, 0)
            upper = tf.linalg.band_part(self.output_factor, 0, -1)
            self._factor_triangular = bool(
                tf.reduce_all(tf.equal(self.output_factor, lower)).numpy()
            ) or bool(tf.reduce_all(tf.equal(self.output_factor, upper)).numpy())

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return ()

    @property
    def variable_keys(self) -> tuple[str, ...]:
        return ()

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if self.output_factor is None:
            output = self.offset + z * self.output_scale
            logdet = tf.reduce_sum(tf.math.log(self.output_scale))
        else:
            output = self.offset + tf.matmul(z, self.output_factor, transpose_b=True)
            if self._factor_triangular:
                logdet = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(self.output_factor))))
            else:
                logdet = tf.linalg.slogdet(self.output_factor)[1]
        return output, tf.zeros(tf.shape(z)[:-1], z.dtype) + logdet

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        payload = {
            "component_id": component_id,
            "dim": self.dimension,
            "dtype": "float64",
            "offset": _tensor_values(self.offset),
        }
        if self.output_factor is None:
            payload.update({"kind": "affine", "scale": _tensor_values(self.output_scale)})
        else:
            payload.update({"kind": "affine_dense", "matrix": _tensor_values(self.output_factor)})
        return payload


class _TrainableComposedIAF(_TrainableTransport):
    def __init__(self, config: NeuTraTrainerConfig) -> None:
        components: list[_TrainableTransport] = []
        for stage in range(int(config.stages)):
            components.append(_TrainableDenseIAF(config, stage_index=stage))
            if stage + 1 < int(config.stages):
                components.append(_FixedMixingReverse(config.dimension))
        if config.family not in PURE_NEUTRA_FAMILIES:
            components.append(
                _FixedTranslation(
                    config.fixed_translation,
                    config.fixed_output_scale,
                    config.fixed_output_factor,
                )
            )
        self.components = tuple(components)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(
            variable
            for component in self.components
            for variable in component.trainable_variables
        )

    @property
    def variable_keys(self) -> tuple[str, ...]:
        keys = []
        for component_index, component in enumerate(self.components):
            keys.extend(
                f"component[{component_index}].{key}" for key in component.variable_keys
            )
        return tuple(keys)

    @property
    def anchor_variable_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        offset = 0
        for component in self.components:
            if isinstance(component, _TrainableDenseIAF):
                indices.extend(offset + index for index in component.anchor_variable_indices)
            offset += len(component.trainable_variables)
        return tuple(indices)

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = z
        logdet = tf.zeros(tf.shape(z)[:-1], z.dtype)
        for component in self.components:
            values, increment = component.forward_and_logdet(values)
            logdet = logdet + increment
        return values, logdet

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        values = z
        stage_scales = []
        for component in self.components:
            if isinstance(component, _TrainableDenseIAF):
                stage_scales.append(component.scale_log(values))
            values, _ = component.forward_and_logdet(values)
        return tf.concat(stage_scales, axis=-1)

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = z
        stage_logits = []
        stage_hidden = []
        for component in self.components:
            if isinstance(component, _TrainableDenseIAF):
                scale_log, shift, scale_logits, hidden = (
                    component._network_with_diagnostics(values)
                )
                stage_logits.append(scale_logits)
                stage_hidden.append(hidden)
                values = values * tf.exp(scale_log) + shift
            else:
                values, _ = component.forward_and_logdet(values)
        return tf.stack(stage_logits, axis=1), tf.stack(stage_hidden, axis=1)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        raise NeuTraTrainingError("composed transport serializes its ordered children")

    def frozen_components(self) -> tuple[Mapping[str, Any], ...]:
        output = []
        iaf_index = 0
        mix_index = 0
        for component in self.components:
            if isinstance(component, _TrainableDenseIAF):
                component_id = f"dense_iaf_{iaf_index:02d}"
                iaf_index += 1
            elif isinstance(component, _FixedMixingReverse):
                component_id = f"mixing_reverse_{mix_index:02d}"
                mix_index += 1
            else:
                component_id = "fixed_translation_00"
            output.append(component.frozen_component_payload(component_id=component_id))
        return tuple(output)


class NeuTraReverseKLTrainer:
    """Reverse-KL optimizer with an explicit target-score boundary."""

    def __init__(self, target: Any, config: NeuTraTrainerConfig) -> None:
        self.target = target
        self.config = config
        if config.family in COMPOSED_NEUTRA_FAMILIES and config.target_chart in {
            "identity",
            "unspecified",
        }:
            warnings.warn(
                "BayesFilter NeuTra training assumes the target chart is already "
                "appropriately scaled for standard-normal latent inputs; no "
                "automatic physical-parameter scaling is applied. Declare and "
                "validate a model-owned affine chart for heterogeneous units.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.transport: _TrainableTransport
        if config.family == "affine_diag":
            self.transport = _TrainableAffineDiagonal(config)
        elif config.family == "dense_iaf":
            self.transport = _TrainableDenseIAF(config)
        else:
            self.transport = _TrainableComposedIAF(config)
            _validate_named_composed_target(target, config)
        self.variables = self.transport.trainable_variables
        self._anchor_variable_indices = tuple(
            getattr(self.transport, "anchor_variable_indices", ())
        )
        if not self.variables:
            raise NeuTraTrainingError("trainer requires trainable variables")
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False, name="neutra_step")
        self.optimizer: tf.keras.optimizers.Adam | None = None
        if config.family in COMPOSED_NEUTRA_FAMILIES:
            self.first_moments = ()
            self.second_moments = ()
            optimizer_learning_rate: Any = (
                dsge_paper_learning_rate(
                    config.learning_rate,
                    boundaries=config.paper_piecewise_boundaries,
                )
                if config.learning_rate_schedule == "paper_piecewise"
                else float(config.learning_rate)
            )
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=optimizer_learning_rate,
                beta_1=config.beta1,
                beta_2=config.beta2,
                epsilon=config.epsilon,
            )
            self.optimizer.build(self.variables)
        else:
            self.first_moments = tuple(
                tf.Variable(
                    tf.zeros_like(variable),
                    trainable=False,
                    name=f"neutra_m_{index}",
                )
                for index, variable in enumerate(self.variables)
            )
            self._generic_learning_rate = tf.Variable(
                float(config.learning_rate),
                dtype=tf.float64,
                trainable=False,
                name="neutra_generic_learning_rate",
            )
            self.second_moments = tuple(
                tf.Variable(
                    tf.zeros_like(variable),
                    trainable=False,
                    name=f"neutra_v_{index}",
                )
                for index, variable in enumerate(self.variables)
            )
        batch_program = self._train_step_impl
        self._compiled_train_step = tf.function(
            batch_program,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )
        self._compiled_validation = tf.function(
            self._validation_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )
        # The process-parallel target route evaluates values/scores outside
        # this process.  This parent-only program keeps the transport update
        # on the selected GPU while accepting detached worker outputs.
        self._compiled_external_train_step = tf.function(
            self._external_train_step_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )
        # The chunked bridge keeps target evaluation and reverse-KL gradient
        # graphs at a bounded static shape. The Python caller aggregates raw
        # gradients before one optimizer update, preserving full-batch means.
        self._compiled_external_gradients = tf.function(
            self._external_gradients_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )

    def forward_and_logdet(self, z: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        return self.transport.forward_and_logdet(values)

    def loss_and_gradients(
        self,
        z: Any,
    ) -> tuple[NeuTraTrainStep, tuple[tf.Tensor, ...]]:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        outputs = self._loss_and_gradients_impl(values)
        return outputs[0], outputs[1]

    def train_step(self, z: Any) -> NeuTraTrainStep:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        rows = self._compiled_train_step(values)
        if not bool(rows[-1].numpy()):
            raise NeuTraTrainingError(
                "compiled NeuTra step rejected nonfinite loss or gradient before update"
            )
        return NeuTraTrainStep(*rows[:-1])

    def train_step_with_external_value_score(
        self,
        z: Any,
        target_value: Any,
        target_score: Any,
    ) -> NeuTraTrainStep:
        """Update the transport from values/scores returned by CPU workers.

        ``target_value`` and ``target_score`` must correspond, in order, to
        ``transport.forward_and_logdet(z)``.  The target is deliberately not
        called in this method; its analytic score is treated as a detached
        custom-gradient payload, matching the DSGE-HMC worker bridge.
        """

        values = _rank2(z, dimension=self.config.dimension, name="z")
        value_tensor = tf.convert_to_tensor(target_value, tf.float64)
        score_tensor = tf.convert_to_tensor(target_score, tf.float64)
        if value_tensor.shape != (values.shape[0],):
            raise ValueError("external target value shape mismatch")
        if score_tensor.shape != values.shape:
            raise ValueError("external target score shape mismatch")
        rows = self._compiled_external_train_step(values, value_tensor, score_tensor)
        if not bool(rows[-1].numpy()):
            raise NeuTraTrainingError(
                "external NeuTra step rejected nonfinite loss or gradient before update"
            )
        return NeuTraTrainStep(*rows[:-1])

    def train_step_with_external_value_score_chunks(
        self,
        z_chunks: Sequence[Any],
        target_value_chunks: Sequence[Any],
        target_score_chunks: Sequence[Any],
        row_counts: Sequence[int],
    ) -> NeuTraTrainStep:
        """Apply one full-batch update from fixed-shape detached-score chunks.

        Each chunk is evaluated with the same transport state and contributes
        ``row_count / total_rows`` of the un-clipped reverse-KL gradient. The
        aggregate is clipped once and applied once, exactly matching the
        full-batch external-score contract up to floating-point summation
        order. Chunks may contain deterministic padding; ``row_counts``
        excludes those rows from every aggregate statistic.
        """

        if not z_chunks or not (
            len(z_chunks) == len(target_value_chunks)
            == len(target_score_chunks)
            == len(row_counts)
        ):
            raise ValueError("chunk sequences must be nonempty and have equal length")
        counts = tuple(int(count) for count in row_counts)
        if any(count <= 0 for count in counts):
            raise ValueError("row_counts must be positive")
        total_rows = sum(counts)
        raw_outputs = []
        for z, target_value, target_score, count in zip(
            z_chunks,
            target_value_chunks,
            target_score_chunks,
            counts,
            strict=True,
        ):
            values = _rank2(z, dimension=self.config.dimension, name="z chunk")
            value_tensor = tf.convert_to_tensor(target_value, tf.float64)
            score_tensor = tf.convert_to_tensor(target_score, tf.float64)
            if value_tensor.shape != (values.shape[0],):
                raise ValueError("external chunk target value shape mismatch")
            if score_tensor.shape != values.shape:
                raise ValueError("external chunk target score shape mismatch")
            if count > int(values.shape[0]):
                raise ValueError("row_count exceeds chunk row count")
            raw_outputs.append(
                self._compiled_external_gradients(
                    values,
                    value_tensor,
                    score_tensor,
                    tf.concat(
                        (
                            tf.ones((count,), tf.float64),
                            tf.zeros((int(values.shape[0]) - count,), tf.float64),
                        ),
                        axis=0,
                    ),
                )
            )

        weight = tf.cast(1.0 / float(total_rows), tf.float64)
        loss = weight * tf.add_n([output[0] for output in raw_outputs])
        surrogate = tf.add_n(
            [weight * output[1] for output in raw_outputs]
        )
        target_value_mean = tf.add_n(
            [weight * output[2] for output in raw_outputs]
        )
        logdet_mean = tf.add_n(
            [weight * output[3] for output in raw_outputs]
        )
        gradients = tuple(
            weight * tf.add_n([output[4 + index] for output in raw_outputs])
            for index in range(len(self.variables))
        )
        gradients = self._mask_anchor_gradients(gradients)
        for index, gradient in enumerate(gradients):
            _assert_finite(gradient, f"chunked gradient[{index}]")
        gradient_norm = tf.linalg.global_norm(gradients)
        if self.config.gradient_clip_mode == "none":
            clipped = gradients
            clipping_applied = tf.constant(False)
        elif self.config.gradient_clip_mode == "per_variable":
            clipped = tuple(
                tf.clip_by_norm(gradient, self.config.gradient_clip_norm)
                for gradient in gradients
            )
            clipping_applied = tf.reduce_any(
                tf.stack(
                    [
                        tf.linalg.norm(gradient) > self.config.gradient_clip_norm
                        for gradient in gradients
                    ]
                )
            )
        else:
            clipped_rows, _ = tf.clip_by_global_norm(
                gradients,
                tf.cast(self.config.gradient_clip_norm, tf.float64),
                use_norm=gradient_norm,
            )
            clipped = tuple(clipped_rows)
            clipping_applied = gradient_norm > tf.cast(
                self.config.gradient_clip_norm, gradient_norm.dtype
            )
        clipped_norm = tf.linalg.global_norm(clipped)
        finite_step = bool(
            tf.reduce_all(
                tf.stack(
                    (
                        tf.reduce_all(tf.math.is_finite(loss)),
                        tf.reduce_all(tf.math.is_finite(surrogate)),
                        tf.reduce_all(tf.math.is_finite(target_value_mean)),
                        tf.reduce_all(tf.math.is_finite(logdet_mean)),
                        tf.reduce_all(tf.math.is_finite(gradient_norm)),
                        tf.reduce_all(tf.math.is_finite(clipped_norm)),
                        *(tf.reduce_all(tf.math.is_finite(gradient)) for gradient in clipped),
                    )
                )
            ).numpy()
        )
        if not finite_step:
            raise NeuTraTrainingError(
                "chunked external NeuTra step rejected nonfinite loss or gradient"
            )
        if self.optimizer is not None:
            self._apply_optimizer_gradients_with_anchor_policy(clipped)
            next_step = tf.cast(self.optimizer.iterations, tf.int64)
        else:
            next_step = self.step + tf.constant(1, dtype=tf.int64)
            beta1 = tf.cast(self.config.beta1, tf.float64)
            beta2 = tf.cast(self.config.beta2, tf.float64)
            learning_rate = tf.cast(self._generic_learning_rate, tf.float64)
            epsilon = tf.cast(self.config.epsilon, tf.float64)
            step_float = tf.cast(next_step, tf.float64)
            for variable, gradient, first, second in zip(
                self.variables, clipped, self.first_moments, self.second_moments, strict=True
            ):
                first.assign(beta1 * first + (1.0 - beta1) * gradient)
                second.assign(beta2 * second + (1.0 - beta2) * tf.square(gradient))
                variable.assign_sub(
                    learning_rate * (first / (1.0 - tf.pow(beta1, step_float))) /
                    (tf.sqrt(second / (1.0 - tf.pow(beta2, step_float))) + epsilon)
                )
                _assert_finite(variable, "updated transport variable")
        self.step.assign(next_step)
        return NeuTraTrainStep(
            loss=loss,
            surrogate=surrogate,
            target_value_mean=target_value_mean,
            logdet_mean=logdet_mean,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=clipped_norm,
            clipping_applied=clipping_applied,
            step=tf.identity(self.step),
        )

    def validation_batch(self, z: Any) -> NeuTraValidation:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        rows = self._compiled_validation(values)
        return NeuTraValidation(*rows)

    def validation_batch_with_external_value(
        self,
        z: Any,
        target_value: Any,
    ) -> NeuTraValidation:
        """Evaluate validation diagnostics from detached worker values."""

        values = _rank2(z, dimension=self.config.dimension, name="z")
        value_tensor = tf.convert_to_tensor(target_value, tf.float64)
        if value_tensor.shape != (values.shape[0],):
            raise ValueError("external validation value shape mismatch")
        theta, logdet = self.transport.forward_and_logdet(values)
        scale_log = self.transport.scale_log(values)
        scale_logits, hidden_preactivations = self.transport.diagnostics(values)
        per_sample_loss = -tf.stop_gradient(value_tensor) - logdet
        _assert_finite(per_sample_loss, "external validation loss")
        _assert_finite(theta, "external validation theta")
        _assert_finite(scale_log, "external validation scale_log")
        _assert_finite(scale_logits, "external validation scale_logits")
        _assert_finite(
            hidden_preactivations,
            "external validation hidden_preactivations",
        )
        return NeuTraValidation(
            per_sample_loss=per_sample_loss,
            target_value=tf.stop_gradient(value_tensor),
            theta=theta,
            logdet=logdet,
            scale_log=scale_log,
            scale_logits=scale_logits,
            hidden_preactivations=hidden_preactivations,
        )

    def sample_base(self, *, batch_size: int, seed: Sequence[int]) -> tf.Tensor:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if len(tuple(seed)) != 2:
            raise ValueError("seed must contain two integers")
        return tf.random.stateless_normal(
            (int(batch_size), int(self.config.dimension)),
            seed=tf.constant(tuple(int(item) for item in seed), dtype=tf.int32),
            dtype=tf.float64,
        )

    def learning_rate_at(self, iteration: int) -> tf.Tensor:
        if int(iteration) < 0:
            raise ValueError("iteration must be nonnegative")
        if self.config.learning_rate_schedule == "paper_piecewise":
            return tf.cast(
                dsge_paper_learning_rate(
                    self.config.learning_rate,
                    boundaries=self.config.paper_piecewise_boundaries,
                )(
                    tf.constant(int(iteration), tf.int64)
                ),
                tf.float64,
            )
        if self.config.family in {
            SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_PURE_NEUTRA_FAMILY,
        }:
            if self.optimizer is None:
                raise NeuTraTrainingError("tuned capacity optimizer is unavailable")
            return tf.cast(self.optimizer.learning_rate, tf.float64)
        if self.optimizer is None:
            return tf.identity(self._generic_learning_rate)
        return tf.constant(self.config.learning_rate, tf.float64)

    def set_learning_rate(self, learning_rate: float) -> None:
        """Assign the effective LR for the tuned family without resetting Adam."""

        if self.config.learning_rate_schedule == "paper_piecewise":
            raise NeuTraTrainingError(
                "manual learning-rate mutation is forbidden for paper_piecewise schedule"
            )
        value = float(learning_rate)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if value > float(self.config.learning_rate):
            raise ValueError("learning_rate cannot exceed configured initial rate")
        if self.optimizer is None:
            self._generic_learning_rate.assign(value)
        else:
            if self.config.family not in {
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_PURE_NEUTRA_FAMILY,
            }:
                raise NeuTraTrainingError(
                    "mutable learning rate is restricted to generic or tuned capacity families"
                )
            self.optimizer.learning_rate.assign(value)

    def state_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": "bayesfilter.neutra.reverse_kl_trainer_state.v1",
            "config": self.config.manifest_payload(),
            "step": int(self.step.numpy()),
            "variable_keys": list(self.transport.variable_keys),
            "variables": [_tensor_values(variable) for variable in self.variables],
            "first_moments": [_tensor_values(value) for value in self.first_moments],
            "second_moments": [_tensor_values(value) for value in self.second_moments],
            "effective_learning_rate": float(self.learning_rate_at(int(self.step.numpy())).numpy()),
            "nonclaims": list(NEUTRA_TRAINING_NONCLAIMS),
        }
        if self.optimizer is not None:
            payload["optimizer_variables"] = [
                _native_tensor_values(value) for value in self.optimizer.variables
            ]
            payload["optimizer_variable_specs"] = [
                {
                    "name": value.name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for value in self.optimizer.variables
            ]
        return {**payload, "state_hash": _stable_hash(payload)}

    def restore_state(
        self,
        payload: Mapping[str, Any],
        *,
        allow_paper_schedule_extension: bool = False,
    ) -> None:
        """Restore an exact trainer checkpoint.

        ``allow_paper_schedule_extension`` permits only a prospective append
        to a paper piecewise schedule.  It cannot alter a boundary governing an
        update already represented by the checkpoint, and it never relaxes
        state-hash, variable, shape, dtype, or Adam-state validation.
        """

        state = dict(payload)
        supplied_hash = str(state.pop("state_hash", ""))
        if supplied_hash != _stable_hash(state):
            raise NeuTraTrainingError("trainer state_hash mismatch")
        if state.get("schema") != "bayesfilter.neutra.reverse_kl_trainer_state.v1":
            raise NeuTraTrainingError("unsupported trainer state schema")
        step = int(state.get("step", -1))
        if step < 0:
            raise NeuTraTrainingError("trainer step must be nonnegative")
        saved_config = state.get("config")
        active_config = self.config.manifest_payload()
        if saved_config != active_config and not (
            allow_paper_schedule_extension
            and isinstance(saved_config, Mapping)
            and _paper_schedule_extension_preserves_history(
                saved_config=saved_config,
                active_config=active_config,
                state_step=step,
            )
        ):
            raise NeuTraTrainingError("trainer state config mismatch")
        keys = tuple(str(item) for item in state.get("variable_keys", ()))
        if keys != self.transport.variable_keys:
            raise NeuTraTrainingError("trainer variable keys mismatch")
        effective_learning_rate = float(
            state.get("effective_learning_rate", self.config.learning_rate)
        )
        configured_learning_rate = float(self.config.learning_rate)
        learning_rate_tolerance = max(
            1.0e-12, _LEARNING_RATE_RESTORE_REL_TOL * configured_learning_rate
        )
        if (
            not math.isfinite(effective_learning_rate)
            or effective_learning_rate <= 0.0
            or effective_learning_rate > configured_learning_rate + learning_rate_tolerance
        ):
            raise NeuTraTrainingError("trainer effective_learning_rate is invalid")
        if effective_learning_rate > configured_learning_rate:
            effective_learning_rate = configured_learning_rate
        validated_variables = _validated_rows(
            self.variables, state.get("variables"), "variables", dtype=tf.float64
        )
        validated_first = _validated_rows(
            self.first_moments, state.get("first_moments"), "first_moments", dtype=tf.float64
        )
        validated_second = _validated_rows(
            self.second_moments, state.get("second_moments"), "second_moments", dtype=tf.float64
        )
        validated_optimizer = None
        if self.optimizer is not None:
            expected_specs = [
                {
                    "name": value.name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for value in self.optimizer.variables
            ]
            if state.get("optimizer_variable_specs") != expected_specs:
                raise NeuTraTrainingError("trainer optimizer variable specs mismatch")
            validated_optimizer = _validated_rows(
                self.optimizer.variables,
                state.get("optimizer_variables"),
                "optimizer_variables",
                dtype=None,
            )
            if int(validated_optimizer[0].numpy()) != step:
                raise NeuTraTrainingError("trainer optimizer iteration mismatch")
        for variable, value in zip(self.variables, validated_variables):
            variable.assign(value)
        for variable, value in zip(self.first_moments, validated_first):
            variable.assign(value)
        for variable, value in zip(self.second_moments, validated_second):
            variable.assign(value)
        if self.optimizer is not None and validated_optimizer is not None:
            for variable, value in zip(self.optimizer.variables, validated_optimizer):
                variable.assign(value)
        if self.optimizer is None:
            self._generic_learning_rate.assign(effective_learning_rate)
        elif self.config.learning_rate_schedule == "adaptive_constant" and self.config.family in {
            SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_PURE_NEUTRA_FAMILY,
        }:
            self.optimizer.learning_rate.assign(effective_learning_rate)
        self.step.assign(step)

    def frozen_transport_payload(
        self,
        *,
        transport_id: str,
        target_signature: str,
    ) -> Mapping[str, Any]:
        if not transport_id:
            raise ValueError("transport_id must be nonempty")
        if len(target_signature) != 64:
            raise ValueError("target_signature must be a sha256 hex digest")
        if (
            self.config.family in COMPOSED_NEUTRA_FAMILIES
            and target_signature != self.config.target_signature
        ):
            raise NeuTraTrainingError("frozen target_signature does not match trainer target")
        state = self.state_payload()
        if isinstance(self.transport, _TrainableComposedIAF):
            components = self.transport.frozen_components()
        else:
            component_id = f"{self.config.family}_00"
            components = (
                self.transport.frozen_component_payload(component_id=component_id),
            )
        raw = {
            "schema": "bayesfilter.neutra.dense_iaf_frozen_transport.v1",
            "transport_id": transport_id,
            "dimension": int(self.config.dimension),
            "target_signature": target_signature,
            "log_jacobian_available": True,
            "component_order": [component["component_id"] for component in components],
            "components": list(components),
            "training_state_hash": state["state_hash"],
            "nonclaims": list(NEUTRA_TRAINING_NONCLAIMS),
        }
        if self.config.family in COMPOSED_NEUTRA_FAMILIES:
            procedure = {
                DSGE_PAPER_NEUTRA_FAMILY: "dsge_hmc_rotemberg_sgu_plain_neutra_v1",
                PURE_PAPER_NEUTRA_FAMILY: "bayesfilter_pure_paper_dense_iaf_v1",
                PURE_BOUNDED_NEUTRA_FAMILY: (
                    "bayesfilter_pure_bounded_dense_iaf_v1"
                ),
                SSL_LSTM_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_capacity_32x32_neutra_v1"
                ),
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_tuned_capacity_32x32_neutra_v1"
                ),
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_deep_capacity_32x32x32_neutra_v1"
                ),
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_wide_capacity_64x64_neutra_v1"
                ),
                SSL_LSTM_PURE_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_pure_32x32_neutra_v1"
                ),
            }[self.config.family]
            if (
                self.config.initialization_mode
                == QUADRATIC_ANCHOR_INITIALIZATION_MODE
            ):
                procedure = "bayesfilter_pure_paper_dense_iaf_quadratic_anchor_v1"
            raw.update(
                {
                    "target_adapter_signature": self.config.target_adapter_signature,
                    "target_chart": self.config.target_chart,
                    "target_parameter_names": list(self.config.target_parameter_names),
                    "fixed_translation": list(self.config.fixed_translation),
                    "fixed_output_scale": list(self.config.fixed_output_scale),
                    "fixed_output_factor": [
                        list(row) for row in self.config.fixed_output_factor
                    ],
                    "chart_signature": self.config.chart_signature,
                    "initial_output_shift": list(self.config.initial_output_shift),
                    "initial_output_scale_log": list(self.config.initial_output_scale_log),
                    "procedure": procedure,
                }
            )
            if (
                self.config.initialization_mode
                == QUADRATIC_ANCHOR_INITIALIZATION_MODE
            ):
                raw.update(
                    {
                        "initialization_mode": self.config.initialization_mode,
                        "anchor_release_steps": int(self.config.anchor_release_steps),
                        "anchor_estimator_signature": self.config.anchor_estimator_signature,
                        "anchor_factor_orientation": self.config.anchor_factor_orientation,
                        "initial_anchor_factor": [
                            list(row) for row in self.config.initial_anchor_factor
                        ],
                    }
                )
        return finalize_dense_iaf_neutra_artifact_payload(raw)

    def _loss_and_gradients_impl(
        self,
        z: tf.Tensor,
    ) -> tuple[NeuTraTrainStep, tuple[tf.Tensor, ...]]:
        theta_for_target, _ = self.transport.forward_and_logdet(z)
        target_value, target_score = _target_value_and_score(
            self.target,
            tf.stop_gradient(theta_for_target),
        )
        target_value = tf.stop_gradient(target_value)
        target_score = tf.stop_gradient(target_score)
        _assert_finite(target_value, "target value")
        _assert_finite(target_score, "target score")
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            theta, logdet = self.transport.forward_and_logdet(z)
            surrogate = tf.reduce_mean(
                -tf.reduce_sum(target_score * theta, axis=-1) - logdet
            )
        gradients = tuple(tape.gradient(surrogate, self.variables))
        if any(gradient is None for gradient in gradients):
            raise NeuTraTrainingError("reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        gradients = self._mask_anchor_gradients(gradients)
        if self.config.family in COMPOSED_NEUTRA_FAMILIES:
            for index, gradient in enumerate(gradients):
                _assert_finite(gradient, f"gradient[{index}]")
        loss = tf.reduce_mean(-target_value - logdet)
        if self.config.gradient_clip_mode == "none":
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped = gradients
            clipping_applied = tf.constant(False)
        elif self.config.gradient_clip_mode == "per_variable":
            clip_inputs = gradients
            gradient_norm = tf.linalg.global_norm(clip_inputs)
            clipped = tuple(
                tf.clip_by_norm(gradient, self.config.gradient_clip_norm)
                for gradient in clip_inputs
            )
            clipping_applied = tf.reduce_any(
                tf.stack(
                    [
                        tf.linalg.norm(gradient) > self.config.gradient_clip_norm
                        for gradient in clip_inputs
                    ]
                )
            )
        else:
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped_rows, _ = tf.clip_by_global_norm(
                gradients,
                tf.cast(self.config.gradient_clip_norm, tf.float64),
                use_norm=gradient_norm,
            )
            clipped = tuple(clipped_rows)
            clipping_applied = gradient_norm > tf.cast(
                self.config.gradient_clip_norm, gradient_norm.dtype
            )
        clipped_norm = tf.linalg.global_norm(clipped)
        _assert_finite(loss, "reverse-KL loss")
        _assert_finite(surrogate, "reverse-KL surrogate")
        for index, gradient in enumerate(clipped):
            _assert_finite(gradient, f"gradient[{index}]")
        result = NeuTraTrainStep(
            loss=loss,
            surrogate=surrogate,
            target_value_mean=tf.reduce_mean(target_value),
            logdet_mean=tf.reduce_mean(logdet),
            gradient_norm=gradient_norm,
            clipped_gradient_norm=clipped_norm,
            clipping_applied=clipping_applied,
            step=tf.identity(self.step),
        )
        return result, clipped

    def _apply_optimizer_gradients_with_anchor_policy(
        self, gradients: tuple[tf.Tensor, ...]
    ) -> None:
        """Apply Adam while preserving anchor variables/slots during warm-up."""

        if self.optimizer is None:
            raise NeuTraTrainingError("composed NeuTra optimizer is unavailable")
        if not self._anchor_variable_indices:
            self.optimizer.apply_gradients(zip(gradients, self.variables))
            return
        frozen = self.step < tf.cast(self.config.anchor_release_steps, tf.int64)
        anchor_state = tuple(
            (
                index,
                tf.identity(self.variables[index]),
                tf.identity(self.optimizer._momentums[index]),
                tf.identity(self.optimizer._velocities[index]),
            )
            for index in self._anchor_variable_indices
        )
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        def restore_anchor_state() -> tf.Tensor:
            for index, variable, momentum, velocity in anchor_state:
                self.variables[index].assign(variable)
                self.optimizer._momentums[index].assign(momentum)
                self.optimizer._velocities[index].assign(velocity)
            return tf.constant(0, tf.int32)

        tf.cond(frozen, restore_anchor_state, lambda: tf.constant(0, tf.int32))

    def _mask_anchor_gradients(
        self, gradients: tuple[tf.Tensor, ...]
    ) -> tuple[tf.Tensor, ...]:
        """Hold the embedded quadratic anchor through the zero-based warm-up.

        This mask is applied before clipping and optimizer updates in direct,
        detached-worker, and chunked reverse-KL routes.  It changes only the
        trainable conditioner gradients; the estimator's external ``mu`` and
        ``L`` inputs are immutable after construction.
        """

        if not self._anchor_variable_indices:
            return gradients
        released = self.step >= tf.cast(self.config.anchor_release_steps, tf.int64)
        anchor_indices = self._anchor_variable_indices
        return tuple(
            tf.where(released, gradient, tf.zeros_like(gradient))
            if index in anchor_indices
            else gradient
            for index, gradient in enumerate(gradients)
        )

    def _validation_impl(self, z: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta, logdet = self.transport.forward_and_logdet(z)
        target_value, _ = _target_value_and_score(self.target, theta)
        scale_log = self.transport.scale_log(z)
        scale_logits, hidden_preactivations = self.transport.diagnostics(z)
        per_sample_loss = -target_value - logdet
        _assert_finite(per_sample_loss, "validation loss")
        _assert_finite(theta, "validation theta")
        _assert_finite(scale_log, "validation scale_log")
        _assert_finite(scale_logits, "validation scale_logits")
        _assert_finite(hidden_preactivations, "validation hidden_preactivations")
        return (
            per_sample_loss,
            target_value,
            theta,
            logdet,
            scale_log,
            scale_logits,
            hidden_preactivations,
        )

    def _train_step_impl(self, z: tf.Tensor) -> tuple[tf.Tensor, ...]:
        result, gradients = self._loss_and_gradients_impl(z)
        if self.optimizer is not None:
            finite_step = tf.reduce_all(
                tf.stack(
                    (
                        tf.reduce_all(tf.math.is_finite(result.loss)),
                        tf.reduce_all(tf.math.is_finite(result.surrogate)),
                        tf.reduce_all(tf.math.is_finite(result.target_value_mean)),
                        tf.reduce_all(tf.math.is_finite(result.logdet_mean)),
                        tf.reduce_all(tf.math.is_finite(result.gradient_norm)),
                        tf.reduce_all(tf.math.is_finite(result.clipped_gradient_norm)),
                        *(
                            tf.reduce_all(tf.math.is_finite(gradient))
                            for gradient in gradients
                        ),
                    )
                )
            )

            def apply_update() -> tf.Tensor:
                self._apply_optimizer_gradients_with_anchor_policy(gradients)
                return tf.cast(self.optimizer.iterations, tf.int64)

            next_step = tf.cond(
                finite_step,
                apply_update,
                lambda: tf.identity(self.step),
            )
            self.step.assign(next_step)
            return (
                result.loss,
                result.surrogate,
                result.target_value_mean,
                result.logdet_mean,
                result.gradient_norm,
                result.clipped_gradient_norm,
                result.clipping_applied,
                tf.identity(self.step),
                finite_step,
            )
        next_step = self.step + tf.constant(1, dtype=tf.int64)
        beta1 = tf.cast(self.config.beta1, tf.float64)
        beta2 = tf.cast(self.config.beta2, tf.float64)
        learning_rate = tf.cast(self._generic_learning_rate, tf.float64)
        epsilon = tf.cast(self.config.epsilon, tf.float64)
        step_float = tf.cast(next_step, tf.float64)
        candidate_rows = []
        for variable, gradient, first, second in zip(
            self.variables,
            gradients,
            self.first_moments,
            self.second_moments,
        ):
            next_first = beta1 * first + (1.0 - beta1) * gradient
            next_second = beta2 * second + (1.0 - beta2) * tf.square(gradient)
            first_hat = next_first / (1.0 - tf.pow(beta1, step_float))
            second_hat = next_second / (1.0 - tf.pow(beta2, step_float))
            next_variable = variable - (
                learning_rate * first_hat / (tf.sqrt(second_hat) + epsilon)
            )
            candidate_rows.append((next_variable, next_first, next_second))
        finite_step = tf.reduce_all(
            tf.stack(
                (
                    tf.reduce_all(tf.math.is_finite(result.loss)),
                    tf.reduce_all(tf.math.is_finite(result.surrogate)),
                    tf.reduce_all(tf.math.is_finite(result.target_value_mean)),
                    tf.reduce_all(tf.math.is_finite(result.logdet_mean)),
                    tf.reduce_all(tf.math.is_finite(result.gradient_norm)),
                    tf.reduce_all(tf.math.is_finite(result.clipped_gradient_norm)),
                    *(
                        tf.reduce_all(tf.math.is_finite(value))
                        for row in candidate_rows
                        for value in row
                    ),
                )
            )
        )

        def apply_generic_update() -> tf.Tensor:
            for (variable, first, second), (
                next_variable,
                next_first,
                next_second,
            ) in zip(
                zip(self.variables, self.first_moments, self.second_moments),
                candidate_rows,
            ):
                variable.assign(next_variable)
                first.assign(next_first)
                second.assign(next_second)
            self.step.assign(next_step)
            return tf.identity(self.step)

        applied_step = tf.cond(
            finite_step,
            apply_generic_update,
            lambda: tf.identity(self.step),
        )
        return (
            result.loss,
            result.surrogate,
            result.target_value_mean,
            result.logdet_mean,
            result.gradient_norm,
            result.clipped_gradient_norm,
            result.clipping_applied,
            applied_step,
            finite_step,
        )

    def _external_train_step_impl(
        self,
        z: tf.Tensor,
        target_value: tf.Tensor,
        target_score: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Parent-side optimizer program for detached worker scores."""

        target_value = tf.stop_gradient(tf.convert_to_tensor(target_value, tf.float64))
        target_score = tf.stop_gradient(tf.convert_to_tensor(target_score, tf.float64))
        _assert_finite(target_value, "external target value")
        _assert_finite(target_score, "external target score")

        @tf.custom_gradient
        def target_values_with_worker_score(
            theta_live: tf.Tensor,
        ) -> tuple[tf.Tensor, Any]:
            del theta_live

            def grad(upstream: tf.Tensor) -> tf.Tensor:
                return tf.reshape(
                    tf.cast(upstream, tf.float64), (-1, 1)
                ) * target_score

            return target_value, grad

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            theta, logdet = self.transport.forward_and_logdet(z)
            bridged_target_value = target_values_with_worker_score(theta)
            loss = tf.reduce_mean(-bridged_target_value - logdet)
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(gradient is None for gradient in gradients):
            raise NeuTraTrainingError("external reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        gradients = self._mask_anchor_gradients(gradients)
        surrogate = tf.reduce_mean(
            -tf.reduce_sum(target_score * tf.stop_gradient(theta), axis=-1)
            - tf.stop_gradient(logdet)
        )
        if self.config.gradient_clip_mode == "none":
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped = gradients
            clipping_applied = tf.constant(False)
        elif self.config.gradient_clip_mode == "per_variable":
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped = tuple(
                tf.clip_by_norm(gradient, self.config.gradient_clip_norm)
                for gradient in gradients
            )
            clipping_applied = tf.reduce_any(tf.stack([
                tf.linalg.norm(gradient) > self.config.gradient_clip_norm
                for gradient in gradients
            ]))
        else:
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped_rows, _ = tf.clip_by_global_norm(
                gradients,
                tf.cast(self.config.gradient_clip_norm, tf.float64),
                use_norm=gradient_norm,
            )
            clipped = tuple(clipped_rows)
            clipping_applied = gradient_norm > tf.cast(
                self.config.gradient_clip_norm, gradient_norm.dtype
            )
        clipped_norm = tf.linalg.global_norm(clipped)
        finite_step = tf.reduce_all(tf.stack((
            tf.reduce_all(tf.math.is_finite(loss)),
            tf.reduce_all(tf.math.is_finite(surrogate)),
            tf.reduce_all(tf.math.is_finite(gradient_norm)),
            tf.reduce_all(tf.math.is_finite(clipped_norm)),
            *(tf.reduce_all(tf.math.is_finite(gradient)) for gradient in clipped),
        )))

        def apply_update() -> tf.Tensor:
            if self.optimizer is not None:
                self._apply_optimizer_gradients_with_anchor_policy(clipped)
                return tf.cast(self.optimizer.iterations, tf.int64)
            next_step = self.step + tf.constant(1, dtype=tf.int64)
            beta1 = tf.cast(self.config.beta1, tf.float64)
            beta2 = tf.cast(self.config.beta2, tf.float64)
            learning_rate = tf.cast(self._generic_learning_rate, tf.float64)
            epsilon = tf.cast(self.config.epsilon, tf.float64)
            step_float = tf.cast(next_step, tf.float64)
            for variable, gradient, first, second in zip(
                self.variables, clipped, self.first_moments, self.second_moments
            ):
                first.assign(beta1 * first + (1.0 - beta1) * gradient)
                second.assign(beta2 * second + (1.0 - beta2) * tf.square(gradient))
                variable.assign_sub(
                    learning_rate * (first / (1.0 - tf.pow(beta1, step_float))) /
                    (tf.sqrt(second / (1.0 - tf.pow(beta2, step_float))) + epsilon)
                )
            return next_step

        next_step = tf.cond(finite_step, apply_update, lambda: tf.identity(self.step))
        self.step.assign(next_step)
        return (
            loss,
            surrogate,
            tf.reduce_mean(target_value),
            tf.reduce_mean(logdet),
            gradient_norm,
            clipped_norm,
            clipping_applied,
            tf.identity(self.step),
            finite_step,
        )

    def _external_gradients_impl(
        self,
        z: tf.Tensor,
        target_value: tf.Tensor,
        target_score: tf.Tensor,
        valid_mask: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Return un-clipped detached-score gradients for one static chunk."""

        target_value = tf.stop_gradient(tf.convert_to_tensor(target_value, tf.float64))
        target_score = tf.stop_gradient(tf.convert_to_tensor(target_score, tf.float64))
        valid_mask = tf.stop_gradient(tf.convert_to_tensor(valid_mask, tf.float64))
        _assert_finite(target_value, "external chunk target value")
        _assert_finite(target_score, "external chunk target score")
        _assert_finite(valid_mask, "external chunk valid mask")

        @tf.custom_gradient
        def target_values_with_worker_score(theta_live: tf.Tensor) -> tuple[tf.Tensor, Any]:
            del theta_live

            def grad(upstream: tf.Tensor) -> tf.Tensor:
                return tf.reshape(tf.cast(upstream, tf.float64), (-1, 1)) * target_score

            return target_value, grad

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            theta, logdet = self.transport.forward_and_logdet(z)
            bridged_target_value = target_values_with_worker_score(theta)
            loss = tf.reduce_sum(valid_mask * (-bridged_target_value - logdet))
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(gradient is None for gradient in gradients):
            raise NeuTraTrainingError("external chunk reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        gradients = self._mask_anchor_gradients(gradients)
        surrogate = tf.reduce_sum(
            valid_mask
            * (
                -tf.reduce_sum(target_score * tf.stop_gradient(theta), axis=-1)
                - tf.stop_gradient(logdet)
            )
        )
        return (
            loss,
            surrogate,
            tf.reduce_sum(valid_mask * target_value),
            tf.reduce_sum(valid_mask * logdet),
            *gradients,
        )


def _target_value_and_score(target: Any, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_method = getattr(target, "batch_value_and_score", None)
    if callable(batch_method):
        value, score = batch_method(theta)
    else:
        method = getattr(target, "log_prob_and_grad", None)
        if not callable(method):
            raise TypeError("target must expose batch_value_and_score or log_prob_and_grad")
        value, score = method(theta)
    value = tf.convert_to_tensor(value, dtype=tf.float64)
    score = tf.convert_to_tensor(score, dtype=tf.float64)
    if value.shape != theta.shape[:-1]:
        raise NeuTraTrainingError("target value shape mismatch")
    if score.shape != theta.shape:
        raise NeuTraTrainingError("target score shape mismatch")
    return value, score


def _validate_named_composed_target(target: Any, config: NeuTraTrainerConfig) -> None:
    label = config.family
    dimension = getattr(target, "parameter_dim", None)
    names = getattr(target, "parameter_names", None)
    if int(dimension) != int(config.dimension):
        raise NeuTraTrainingError(f"{label} target dimension mismatch")
    if tuple(str(value) for value in names) != tuple(config.target_parameter_names):
        raise NeuTraTrainingError(f"{label} target parameter names/order mismatch")
    target_signature = getattr(target, "target_signature", None)
    adapter_signature = getattr(target, "adapter_signature", None)
    if not callable(target_signature) or target_signature() != config.target_signature:
        raise NeuTraTrainingError(f"{label} target signature mismatch")
    if not callable(adapter_signature) or adapter_signature() != config.target_adapter_signature:
        raise NeuTraTrainingError(f"{label} target adapter signature mismatch")
    target_config = getattr(target, "config", None)
    signature_payload = getattr(target_config, "signature_payload", None)
    if not callable(signature_payload):
        raise NeuTraTrainingError(f"{label} target chart manifest unavailable")
    manifest = signature_payload()
    transform = manifest.get("parameter_transform", {})
    if transform.get("orientation") != "identity" or transform.get(
        "inverse_orientation"
    ) != "identity":
        raise NeuTraTrainingError(f"{label} target chart is not identity-oriented")
    # The affine chart is model-owned and may be centered away from the
    # physical target's prior center.  Target identity is checked above;
    # chart identity/factor are carried separately in the trainer config.


def _rank2(value: Any, *, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank != 2:
        raise ValueError(f"{name} must have rank 2")
    if tensor.shape[-1] != int(dimension):
        raise ValueError(f"{name} trailing dimension mismatch")
    return tensor


def _activation(values: tf.Tensor, activation: str) -> tf.Tensor:
    if activation == "elu":
        return tf.nn.elu(values)
    if activation == "tanh":
        return tf.math.tanh(values)
    if activation == "relu":
        return tf.nn.relu(values)
    raise NeuTraTrainingError(f"unsupported activation: {activation}")


def _dense_iaf_masks(dim: int, hidden_layers: tuple[int, ...]) -> tuple[tf.Tensor, ...]:
    degrees: list[list[int]] = [list(range(1, dim + 1))]
    maximum = max(1, dim - 1)
    for width in hidden_layers:
        degrees.append([1 + (index % maximum) for index in range(width)])
    degrees.append(list(range(1, dim + 1)) + list(range(1, dim + 1)))
    masks = []
    for index, (source_degrees, target_degrees) in enumerate(
        zip(degrees[:-1], degrees[1:])
    ):
        output_layer = index == len(degrees) - 2
        masks.append(
            tf.constant(
                [
                    [
                        1.0
                        if ((source < target) if output_layer else (source <= target))
                        else 0.0
                        for target in target_degrees
                    ]
                    for source in source_degrees
                ],
                dtype=tf.float64,
            )
        )
    return tuple(masks)


def _assign_rows(
    variables: Sequence[tf.Variable],
    rows: Any,
    name: str,
) -> None:
    if not isinstance(rows, (tuple, list)) or len(rows) != len(variables):
        raise NeuTraTrainingError(f"trainer {name} length mismatch")
    for index, (variable, row) in enumerate(zip(variables, rows)):
        value = tf.convert_to_tensor(row, dtype=tf.float64)
        if value.shape != variable.shape:
            raise NeuTraTrainingError(f"trainer {name}[{index}] shape mismatch")
        _assert_finite(value, f"trainer {name}[{index}]")
        variable.assign(value)


def _validated_rows(
    variables: Sequence[tf.Variable],
    rows: Any,
    name: str,
    *,
    dtype: tf.dtypes.DType | None,
) -> tuple[tf.Tensor, ...]:
    if not isinstance(rows, (tuple, list)) or len(rows) != len(variables):
        raise NeuTraTrainingError(f"trainer {name} length mismatch")
    validated = []
    for index, (variable, row) in enumerate(zip(variables, rows)):
        value = tf.convert_to_tensor(
            row,
            dtype=variable.dtype if dtype is None else dtype,
        )
        if value.shape != variable.shape:
            raise NeuTraTrainingError(f"trainer {name}[{index}] shape mismatch")
        _assert_finite(tf.cast(value, tf.float64), f"trainer {name}[{index}]")
        validated.append(value)
    return tuple(validated)


def _tensor_values(value: tf.Tensor) -> Any:
    return tf.convert_to_tensor(value, dtype=tf.float64).numpy().tolist()


def _native_tensor_values(value: tf.Tensor) -> Any:
    return tf.convert_to_tensor(value).numpy().tolist()


def _assert_finite(value: tf.Tensor, name: str) -> None:
    tf.debugging.assert_all_finite(value, f"{name} must be finite")


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
